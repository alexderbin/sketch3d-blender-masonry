"""State layers of masonry and buildings are assets. Where and how much is decided by the
masonry logic: it writes a mask (a mesh attribute 0…1 — face, vertex or corner) onto its objects; a material receives
the layer once and reads the mask by name, so the same stone gets dirty differently on different walls without new
materials.

  import bl_states as SS
  SS.layer(material, "mud", mask="st_mud", strength=1.0)       # attribute name | node socket | number
  SS.layer_objects(objs, "moss", mask="st_moss")                  # all materials of the objects
  SS.KINDS — the kinds and their default parameters

Each layer: coverage = the mask, the edge is broken by the layer's own 3D noise (a threshold on the noise gives a
natural edge, the mask may be coarse); inside — the kind's color, roughness and relief. There is no repetition: 3D
noise, and scans (moss, mud) use Mikkelsen tiling. The cost is a few noise nodes per layer; scans are 1–3 images per
layer (the Metal shader limit is 16)."""
import json
import math
import zlib
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
from core.surface import bl_surface as SF  # noqa: E402  (nodes and Mikkelsen tiling of the surface engine)

LIB = os.path.join(ROOT, "library", "textures", "states")

# kind: color (linear), roughness, relief (bump strength), edge noise scale (1/m), edge width, source
KINDS = {
    "mud":     dict(color=(0.12, 0.1, 0.08), frac=0.85, rough=0.92, bump=0.3, edge=5.0, width=0.22, scan="brown_mud_02", speckle=1.0),
    "damp":   dict(color=None, darker=0.62, rough_k=0.55, bump=0.0, edge=2.5, width=0.10, line=0.12),
    "film":     dict(color=(0.07, 0.12, 0.035), frac=0.75, rough=0.8, bump=0.0, edge=4.0, width=0.2),
    "moss":       dict(color=None, rough=0.95, bump=0.8, edge=3.5, width=0.16, scan="Moss003", up=0.35, joints=0.4, darker=0.7, colonies=0.45),
    "soot_stain":    dict(color=(0.025, 0.022, 0.02), frac=0.8, rough=0.95, bump=0.0, edge=1.2, width=0.9, noise=1.4, plume=1.0),
    "efflorescence":    dict(color=(0.82, 0.81, 0.77), frac=0.8, rough=1.0, bump=0.15, edge=7.0, width=0.15),
    "cracks":   dict(color=(0.012, 0.011, 0.01), frac=0.93, rough=0.95, bump=0.9, atlas="cracks_stone", border=1.12, edge=2.0, width=0.3),
    "peeling": dict(lighter=1.4, rough=0.97, bump=1.4, spots=4.0, threshold=0.5, edge=2.5, width=0.2),
    "pitting":   dict(lighter=1.4, pit=0.2, rough=0.98, bump=1.4, grid=4.0, cell_frac=0.55, radius=0.26, n_rays=(4, 8), edge=2.0, width=0.25),
}


def meta(aid):
    p = os.path.join(LIB, aid, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"bl_states: no asset {aid} — .venv/bin/python -m components.constructions.textures.states.fetch")
    return json.load(open(p))


def _img(path, color):
    im = bpy.data.images.load(os.path.join(ROOT, path), check_existing=True)
    if not color:
        im.colorspace_settings.name = "Non-Color"
    return im


def _src(sock):
    return sock.links[0].from_socket if sock.is_linked else None


def _pct(im, qs):
    """Brightness percentiles of an image (every 7th pixel) — the streak threshold taken from the image itself: with some
    scans the streaks are pale (012C: 97 % of the tone is lighter than 0.1), with others dense; the cache is an image
    property."""
    key = "sk_pct_" + "_".join(str(q) for q in qs)
    if key in im:
        return list(im[key])
    import numpy as np
    w, h = im.size; px = np.empty(w * h * im.channels, np.float32); im.pixels.foreach_get(px)
    a = px.reshape(-1, im.channels)[::7, :3].mean(1)
    v = [float(np.percentile(a, q)) for q in qs]; im[key] = v
    return v


class _L:
    """Assembly of a single layer in the material tree."""

    def __init__(self, m, kind, seed):
        self.m, self.nt, self.N = m, m.node_tree, SF._N(m.node_tree)
        self.bsdf = next(n for n in self.nt.nodes if n.type == "BSDF_PRINCIPLED")
        self.kind = kind
        geo = self.N.new("ShaderNodeNewGeometry")
        h = zlib.crc32(kind.encode()) % 997 + seed * 131              # each kind has its own noise offset (hash(str) is random in every run)
        self.P = self.N.vmath("ADD", geo.outputs["Position"], (h * 1.37, h * 0.71, h * 0.29))
        self.Nrm = geo.outputs["Normal"]

    def mask(self, m):
        if isinstance(m, str):
            a = self.N.new("ShaderNodeAttribute"); a.attribute_type = "GEOMETRY"; a.attribute_name = m
            return a.outputs["Fac"]
        return m

    def offset(kind, seed=0):
        h = zlib.crc32(kind.encode()) % 997 + seed * 131
        return (h * 1.37, h * 0.71, h * 0.29)

    def noise(self, scale, detail=3.0, rough=0.55, vec=None):
        nz = self.N.new("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        self.N.put(nz.inputs["Vector"], vec if vec is not None else self.P)
        return nz.outputs["Fac"]

    def ramp(self, x, a, b, va=0.0, vb=1.0):
        mr = self.N.new("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"; mr.clamp = True
        self.N.put(mr.inputs["Value"], x); self.N.put(mr.inputs["From Min"], a); self.N.put(mr.inputs["From Max"], b)
        mr.inputs["To Min"].default_value = va; mr.inputs["To Max"].default_value = vb
        return mr.outputs["Result"]

    def cover(self, M, scale, width, strength, amp=0.9, vec=None):
        """Coverage: mask + edge noise (height blend): M≈0.5 — the noise decides, 1 — covered, 0 — clean; amp — how much the noise
        tears the edge (soot — as a cloud), vec — own noise coordinates (the plume is stretched upward)."""
        nz = self.noise(scale, 4.0, 0.6, vec)
        v = self.N.math("ADD", self.N.math("MULTIPLY", M, strength), self.N.math("MULTIPLY", self.N.math("SUBTRACT", nz, 0.5), amp))
        return self.ramp(v, 0.5 - width / 2, 0.5 + width / 2)

    def color(self, fn):
        """Base Color ← fn(previous color) (a color socket)."""
        s = self.bsdf.inputs["Base Color"]; src = _src(s)
        base = src if src is not None else tuple(s.default_value)
        self.N.put(s, fn(base))

    def rough(self, fn):
        s = self.bsdf.inputs["Roughness"]; src = _src(s)
        if src is not None:
            bw = self.N.new("ShaderNodeRGBToBW"); self.N.put(bw.inputs[0], src); r = bw.outputs[0]
        else:
            r = float(s.default_value)
        self.N.put(s, self.N.math("MINIMUM", self.N.math("MAXIMUM", fn(r), 0.05), 1.0))

    def bump(self, h, strength, dist=0.004):
        if strength <= 0:
            return
        s = self.bsdf.inputs["Normal"]; src = _src(s)
        bp = self.N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = strength; bp.inputs["Distance"].default_value = dist
        self.N.put(bp.inputs["Height"], h)
        if src is not None:
            self.N.put(bp.inputs["Normal"], src)
        self.N.put(s, bp.outputs["Normal"])

    def mix(self, f, a, b, kind="MIX"):
        return self.N.mix_rgb(kind, f, a, b)

    def rgb(self, c):
        return (float(c[0]), float(c[1]), float(c[2]), 1.0)

    def gray(self, x):
        cc = self.N.new("ShaderNodeCombineColor")
        for k in ("Red", "Green", "Blue"):
            self.N.put(cc.inputs[k], x)
        return cc.outputs[0]

    def scan(self, aid, keys, square=None, rot=0.5, uv="sk_uv"):
        """Asset maps with Mikkelsen tiling: by the uv unrolling (meters) or, if the mesh has none, by world x+y, z."""
        mt = meta(aid); sq = float(square or (mt.get("size_m") or [1.0])[0])
        uvn = self.N.new("ShaderNodeUVMap"); uvn.uv_map = uv
        st = self.N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / sq)
        maps = {k: (_img(mt["maps"][k], k == "color"), k == "color") for k in keys if k in mt["maps"]}
        return SF.hex_maps(self.N, st, maps, rot)


def _rgb_mul(L, base, k):
    return L.mix(1.0, base, L.gray(k), "MULTIPLY")


def layer(m, kind, mask=1.0, strength=1.0, seed=0, **p):
    """A state layer in front of the BSDF of material m. mask — an attribute name (0…1) | socket | number; strength — the mask
    multiplier; p — overrides of the kind's parameters (KINDS). → m."""
    P = {**KINDS[kind], **p}
    L = _L(m, kind, seed); N = L.N
    M = N.math("MULTIPLY", L.mask(mask), strength) if not isinstance(mask, (int, float)) else float(mask) * strength
    vec = None
    if P.get("plume"):                                                  # plume: the noise is stretched upward — soot and smoke stains rise
        sv = N.new("ShaderNodeVectorMath"); sv.operation = "MULTIPLY"; N.put(sv.inputs[0], L.P)
        sv.inputs[1].default_value = (1.0, 1.0, 1.0 / (1.0 + 2.0 * P["plume"])); vec = sv.outputs[0]
    c = L.cover(M, P.get("edge", 3.0), P.get("width", 0.2), 1.0, P.get("noise", 0.9), vec)

    if kind == "mud":
        o = L.scan(P["scan"], ("color", "height"))                      # scan dirt: pattern and clods
        sp = N.new("ShaderNodeTexVoronoi", Scale=60.0); N.put(sp.inputs["Vector"], L.P)
        spk = L.ramp(sp.outputs["Distance"], 0.3, 0.12)                   # splash speckle above the solid coverage
        spk = N.math("MULTIPLY", spk, N.math("MULTIPLY", L.ramp(c, 0.0, 0.6, 1.0, 0.0), P.get("speckle", 1.0)))
        patch = L.ramp(N.math("ADD", o.get("height", 0.5), N.math("MULTIPLY", L.noise(11.0, 3.0, 0.6), 0.5)), 0.25, 0.75)   # in patches, not as paint
        cc = N.math("MULTIPLY", c, N.math("ADD", 0.45, N.math("MULTIPLY", patch, 0.55)))
        cc = N.math("MAXIMUM", cc, N.math("MULTIPLY", spk, L.ramp(M, 0.05, 0.4)))
        mud = L.mix(0.5, L.rgb(P["color"]), o["color"], "MULTIPLY")
        mud = L.mix(1.0, mud, L.gray(2.0), "MULTIPLY")
        L.color(lambda b: L.mix(N.math("MULTIPLY", cc, P["frac"]), b, mud))
        L.rough(lambda r: N.math("ADD", r, N.math("MULTIPLY", cc, P["rough"] - 0.6)))
        L.bump(N.math("MULTIPLY", o["height"] if "height" in o else 0.5, cc), P["bump"])
    elif kind == "damp":
        line = N.math("MULTIPLY", L.ramp(c, 0.25, 0.5), L.ramp(c, 0.75, 0.5))   # tide line — salt at the edge
        L.color(lambda b: L.mix(N.math("MULTIPLY", line, P["line"]), _rgb_mul(L, b, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", c, 1 - P["darker"]))),
                                L.rgb((0.75, 0.74, 0.7))))
        L.rough(lambda r: N.math("MULTIPLY", r, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", c, 1 - P["rough_k"]))))
    elif kind in ("film", "soot_stain", "efflorescence"):
        if kind == "efflorescence":                                             # salt streaks downward: the noise is stretched vertically
            sv = N.new("ShaderNodeVectorMath"); sv.operation = "MULTIPLY"; N.put(sv.inputs[0], L.P); sv.inputs[1].default_value = (9.0, 9.0, 2.5)
            c = N.math("MULTIPLY", c, L.ramp(L.noise(1.0, 3.0, 0.6, sv.outputs[0]), 0.35, 0.6))
        var = L.noise(9.0, 3.0, 0.6)
        col = L.mix(N.math("MULTIPLY", var, 0.5), L.rgb(P["color"]), L.gray(1.25 if kind != "soot_stain" else 0.6), "MULTIPLY")
        k = N.math("MULTIPLY", c, P.get("frac", 0.8))
        L.color(lambda b: L.mix(k, b, col))
        L.rough(lambda r: N.math("ADD", r, N.math("MULTIPLY", c, P["rough"] - 0.6)))
        if P.get("bump", 0) > 0:
            L.bump(N.math("MULTIPLY", L.noise(40.0, 4.0, 0.65), c), P["bump"])
    elif kind == "moss":                                                  # a layer under the volumetric tufts (bl_moss): joints and tops first
        o = L.scan(P["scan"], ("color", "height"))
        h = o.get("height", 0.5)
        up = L.ramp(N.vmath("DOT_PRODUCT", L.Nrm, (0.0, 0.0, 1.0)), 0.2, 0.7)                       # faces looking up come first
        uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = "sk_uv"
        p_ = N.new("ShaderNodeSeparateXYZ"); N.put(p_.inputs[0], uvn.outputs["UV"])
        lo = N.new("ShaderNodeSeparateXYZ"); N.put(lo.inputs[0], N.attr("sk_lo")); hi = N.new("ShaderNodeSeparateXYZ"); N.put(hi.inputs[0], N.attr("sk_hi"))
        d = N.math("MINIMUM", N.math("MINIMUM", N.math("SUBTRACT", p_.outputs["X"], lo.outputs["X"]), N.math("SUBTRACT", hi.outputs["X"], p_.outputs["X"])),
                   N.math("MINIMUM", N.math("SUBTRACT", p_.outputs["Y"], lo.outputs["Y"]), N.math("SUBTRACT", hi.outputs["Y"], p_.outputs["Y"])))
        joint = L.ramp(d, 0.04, 0.0)                                                                  # at the stone edge is the joint: moss starts there
        colony = L.ramp(L.noise(3.5, 3.0, 0.55), 0.5 - P["colonies"] * 0.2, 0.5 + P["colonies"] * 0.25)  # colonies in patches, like the tufts
        where = N.math("ADD", 0.3, N.math("MULTIPLY", N.math("MINIMUM", N.math("ADD", up, joint), 1.0), 0.7))   # on vertical surfaces mostly the joints
        loc = N.math("MULTIPLY", N.math("MULTIPLY", c, colony), where)
        cm = L.ramp(N.math("ADD", loc, N.math("MULTIPLY", N.math("SUBTRACT", h, 0.5), 0.5)), 0.3, 0.55)
        moss = L.mix(N.math("MULTIPLY", N.math("SUBTRACT", 1.0, h), 1.0 - P["darker"]), o["color"], (0.0, 0.0, 0.0, 1.0))   # deep in the tufts it is darker
        L.color(lambda b: L.mix(cm, b, moss))
        L.rough(lambda r: N.math("ADD", r, N.math("MULTIPLY", cm, P["rough"] - 0.6)))
        L.bump(N.math("MULTIPLY", h, cm), P["bump"], 0.01)
    elif kind == "cracks":
        mt = meta(P["atlas"]); sq = float(mt["size_m"][0])                                            # crack growth atlas (crack_atlas.py)
        rr = N.new("ShaderNodeSeparateXYZ"); N.put(rr.inputs[0], N.attr("sk_rnd"))
        uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = "sk_uv"
        rot = N.new("ShaderNodeVectorRotate"); rot.rotation_type = "Z_AXIS"
        N.put(rot.inputs["Vector"], N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / sq))
        N.put(rot.inputs["Angle"], N.math("MULTIPLY", rr.outputs["X"], 2 * math.pi))                # its own piece of the atlas per stone
        sh = N.new("ShaderNodeCombineXYZ"); N.put(sh.inputs["X"], N.math("MULTIPLY", rr.outputs["Y"], 31.0)); N.put(sh.inputs["Y"], N.math("MULTIPLY", rr.outputs["Z"], 31.0))
        vec = N.vmath("ADD", rot.outputs[0], sh.outputs[0])
        tm = N.new("ShaderNodeTexImage"); tm.image = _img(mt["maps"]["mask"], False); tm.interpolation = "Linear"; N.put(tm.inputs["Vector"], vec)
        th = N.new("ShaderNodeTexImage"); th.image = _img(mt["maps"]["height"], False); N.put(th.inputs["Vector"], vec)
        lvl = tm.outputs["Color"]
        bw = N.new("ShaderNodeRGBToBW"); N.put(bw.inputs[0], lvl)
        thr = N.math("SUBTRACT", 1.0, N.math("MULTIPLY", c, 0.75))                                   # order: with little mask only the main cracks
        core = N.math("MULTIPLY", L.ramp(bw.outputs[0], N.math("SUBTRACT", thr, 0.08), thr), L.ramp(bw.outputs[0], 0.05, 0.2))
        hb = N.new("ShaderNodeRGBToBW"); N.put(hb.inputs[0], th.outputs["Color"])
        rim = N.math("MULTIPLY", N.math("SUBTRACT", N.math("SUBTRACT", 1.0, hb.outputs[0]), core), c)  # rim: a fresh chip at the edge
        k = N.math("MULTIPLY", core, P["frac"])
        L.color(lambda b: L.mix(k, _rgb_mul(L, b, N.math("ADD", 1.0, N.math("MULTIPLY", N.math("MAXIMUM", rim, 0.0), P["border"] - 1.0))),
                                L.rgb(P["color"])))
        L.rough(lambda r: N.math("ADD", r, N.math("MULTIPLY", core, P["rough"] - 0.6)))
        L.bump(N.math("SUBTRACT", 1.0, N.math("MULTIPLY", N.math("SUBTRACT", 1.0, hb.outputs[0]), N.math("MINIMUM", N.math("MULTIPLY", c, 2.0), 1.0))),
               P["bump"], 0.004)
    elif kind == "peeling":
        n = L.noise(P["spots"], 3.0, 0.55)
        patch = N.math("MULTIPLY", L.ramp(n, P["threshold"], P["threshold"] + 0.03), c)                         # a spalled piece
        L.color(lambda b: _rgb_mul(L, b, N.math("ADD", 1.0, N.math("MULTIPLY", patch, P["lighter"] - 1.0))))
        L.rough(lambda r: N.math("ADD", r, N.math("MULTIPLY", patch, P["rough"] - 0.6)))
        L.bump(N.math("SUBTRACT", 1.0, patch), P["bump"], 0.003)                                       # the edge step catches the light
    elif kind == "pitting":                                              # bullet crater: ragged edge, cone, hole, crack rays
        uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = "sk_uv"
        vo = N.new("ShaderNodeTexVoronoi", Scale=P["grid"]); vo.voronoi_dimensions = "2D"; vo.feature = "F1"
        N.put(vo.inputs["Vector"], uvn.outputs["UV"])
        rc = N.new("ShaderNodeSeparateColor"); N.put(rc.inputs[0], vo.outputs["Color"])
        v = N.vmath("SCALE", N.vmath("SUBTRACT", uvn.outputs["UV"], vo.outputs["Position"]), s=P["grid"])   # from the hit point: the Voronoi Position is in the original coordinates
        vs = N.new("ShaderNodeSeparateXYZ"); N.put(vs.inputs[0], v)
        rl = N.new("ShaderNodeVectorMath"); rl.operation = "LENGTH"; N.put(rl.inputs[0], v); r = rl.outputs["Value"]
        th = N.math("ARCTAN2", vs.outputs["Y"], vs.outputs["X"])
        alive = N.math("MULTIPLY", N.math("LESS_THAN", rc.outputs["Red"], P["cell_frac"]), L.ramp(c, 0.3, 0.6))
        R = N.math("MULTIPLY", P["radius"], N.math("ADD", 0.55, N.math("MULTIPLY", rc.outputs["Green"], 0.9)))   # its own radius
        ring = N.new("ShaderNodeCombineXYZ")
        N.put(ring.inputs["X"], N.math("MULTIPLY", N.math("COSINE", th), 2.2)); N.put(ring.inputs["Y"], N.math("MULTIPLY", N.math("SINE", th), 2.2))
        N.put(ring.inputs["Z"], N.math("MULTIPLY", rc.outputs["Blue"], 40.0))
        jn = N.new("ShaderNodeTexNoise", Scale=1.0, Detail=4.0, Roughness=0.7); N.put(jn.inputs["Vector"], ring.outputs[0])
        Rt = N.math("MULTIPLY", R, N.math("ADD", 0.72, N.math("MULTIPLY", jn.outputs["Fac"], 0.56)))            # ragged edge
        inside = N.math("MULTIPLY", L.ramp(r, Rt, N.math("MULTIPLY", Rt, 0.86), 0.0, 1.0), alive)
        depth = N.math("MULTIPLY", N.math("POWER", N.math("MAXIMUM", N.math("SUBTRACT", 1.0, N.math("DIVIDE", r, Rt)), 0.0), 0.65), alive)
        pit = N.math("MULTIPLY", L.ramp(r, N.math("MULTIPLY", R, 0.2), N.math("MULTIPLY", R, 0.1), 0.0, 1.0), alive)
        n_c = N.math("ADD", P["n_rays"][0], N.math("FLOOR", N.math("MULTIPLY", rc.outputs["Green"], P["n_rays"][1] - P["n_rays"][0])))
        fr = N.math("FRACT", N.math("ADD", N.math("DIVIDE", N.math("MULTIPLY", th, n_c), 2 * math.pi), N.math("MULTIPLY", rc.outputs["Blue"], 7.0)))
        arc = N.math("MULTIPLY", N.math("ABSOLUTE", N.math("SUBTRACT", fr, 0.5)), N.math("DIVIDE", N.math("MULTIPLY", r, 2 * math.pi), n_c))
        ray = N.math("MULTIPLY", L.ramp(arc, 0.012, 0.003, 0.0, 1.0),
                     N.math("MULTIPLY", L.ramp(r, N.math("MULTIPLY", Rt, 0.95), Rt, 0.0, 1.0), L.ramp(r, N.math("MULTIPLY", R, 1.4), N.math("MULTIPLY", R, 2.1))))
        ray = N.math("MULTIPLY", ray, alive)
        halo = N.math("MULTIPLY", N.math("MULTIPLY", L.ramp(r, N.math("MULTIPLY", R, 2.0), R), N.math("SUBTRACT", 1.0, inside)), alive)
        lit = N.math("ADD", 1.0, N.math("ADD", N.math("MULTIPLY", inside, P["lighter"] - 1.0), N.math("MULTIPLY", halo, 0.1)))
        dark = N.math("SUBTRACT", 1.0, N.math("MAXIMUM", N.math("MULTIPLY", pit, 1.0 - P["pit"]), N.math("MULTIPLY", ray, 0.7)))
        L.color(lambda b: _rgb_mul(L, b, N.math("MULTIPLY", lit, dark)))
        L.rough(lambda r_: N.math("ADD", r_, N.math("MULTIPLY", inside, P["rough"] - 0.6)))
        L.bump(N.math("SUBTRACT", N.math("SUBTRACT", 1.0, depth), N.math("MULTIPLY", ray, 0.15)), P["bump"], 0.01)
    else:
        raise KeyError(f"bl_states: no kind {kind} (available: {', '.join(KINDS)})")
    m[f"st_{kind}"] = mask if isinstance(mask, str) else float(mask) if isinstance(mask, (int, float)) else "nest"
    return m


def layer_objects(objs, kind, mask, strength=1.0, **p):
    """The layer in all materials of the objects (each material once). → number of materials."""
    done = set()
    for ob in objs:
        for sl in getattr(ob, "material_slots", []):
            m = sl.material
            if m is None or m.name in done or not m.use_nodes:
                continue
            if not any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes):
                continue
            layer(m, kind, mask, strength, **p); done.add(m.name)
    return len(done)
