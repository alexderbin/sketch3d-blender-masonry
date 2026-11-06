"""Covering substance — the "textures" step of the Constructions component: roofing, paving and timber get the substance of
a piece by the same method as masonry stone: for a tile, a sheet, a board, a paving stone, a beam — its own piece of a
scan (shift and rotation by the piece number sk_stone), the tone is the layout's Col (substance × spread), the scan
pattern is normalized to 1. Knowledge — covers.yaml → covers.json (substance → scan, rotation, strength, pattern color,
roughness, metallic; kind of a part → substance, piece role, second substance). Differences from masonry stone:
  grain direction: grain — along v of the unrolling (roof: wood, reed, streaks on metal — along the slope), long — along the long
  axis of the piece (a log of a log house lies, a half-timber post stands, a floor board); piece rotation 0 or π and a
  jitter of ±3°;
  pattern color (chroma) — for metal the pattern has no scan color: reddish rust spots do not lie on copper and gold;
  metallic and roughness are the substance's own (copper patina is not metal, gold is metal, glaze is smooth);
  faces without a number (sk_stone = 0: a far row as a strip, the tent cap, the ridge bed, the valley trough, the paving
  bedding) and pieces longer than 0.75 of the scan square (a 4 m board of boards, a log of a log house on a 0.5 m scan)
  — Mikkelsen tiling with the field shifted by the piece number: a repeat of the scan along the piece would be visible;
  far view (roof shell, the paving bedding pattern) — its pattern and tone stay, roughness and metallic are the
  substances' averages: near and far do not differ in gloss;
  faces without the Col tone (paving bedding) — the substance tone from the part's knowledge.
Parts — by the object property: sk_roofing (the substance is in the property), sk_paving, sk_timber (the substance by
kind). Face roles per the contract (core/contract): the role of a piece of a kind — the substance of the kind; the
second substance — the mortar of paving (bedding), the finish of half-timbering (infill); on a non-metal roof the metal
parts (valleys, caps) are lead, and the mortar role (the ridge bed) is the substance mortar.
  import bl_cover as CV; CV.run(scene, over={substance: {scan: …}}) → counts"""
import json
import math
import os

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from core.contract.contract import ROLES  # noqa: E402
from core.surface import bl_matlib as ML  # noqa: E402
from core.surface import bl_surface as SF  # noqa: E402  (surface engine: scan, Mikkelsen tiling, piece number)

COVERS = os.path.join(HERE, "covers.json")
PROPS = ("sk_roofing", "sk_paving", "sk_timber")
LONG = 0.75            # a piece longer than 0.75 of the scan square gets Mikkelsen tiling (the scan would repeat along the piece)
GRAIN = "sk_grain"     # face attribute: 1 — the piece's long axis is along u of the unrolling (long grain is rotated by 90°)
_MEAN, _MATS = {}, {}


def knowledge(over=None):
    """covers.json: substances (+ over {substance: fields | scan} — the building description, takes priority; new ones start
    from the default) and parts."""
    d = json.load(open(COVERS))
    t = {k: dict(v) for k, v in d["substances"].items()}
    for k, v in (over or {}).items():
        t[k] = {**d["defaults"], **t.get(k, {}), **(v if isinstance(v, dict) else {"scan": v})}
    return t, d["parts"]


def _mean(im, srgb):
    """Average color of an image (linear; 8-bit sRGB converted to linear), over every 7th pixel."""
    key = (im.name, srgb)
    if key not in _MEAN:
        px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px)
        g = px.reshape(-1, im.channels)[::7, :3]
        if srgb: g = np.where(g <= 0.04045, g / 12.92, ((g + 0.055) / 1.055) ** 2.4)
        _MEAN[key] = g.mean(0)
    return _MEAN[key]


def _finish(N, bsdf, col, rough, nor01, spec, mean, k, height=None, uv="sk_uv", tone=None):
    """Tone × pattern (color / average color; chroma — the share of color, k — the piece strength), roughness from…to by the
    map, metallic, the piece normal (+ relief by a height image). The tone is the layout's Col attribute, tone (rgb) is for
    faces without it."""
    m = [max(float(c), 1e-4) for c in mean]
    norm = N.mix_rgb("DIVIDE", 1.0, col, (m[0], m[1], m[2], 1.0))
    if spec["chroma"] < 1.0:
        lum = N.new("ShaderNodeRGBToBW"); N.put(lum.inputs[0], norm)
        gc = N.new("ShaderNodeCombineColor")
        for ch in ("Red", "Green", "Blue"): N.put(gc.inputs[ch], lum.outputs[0])
        norm = N.mix_rgb("MIX", float(spec["chroma"]), gc.outputs[0], norm)
    pat = N.mix_rgb("MIX", k, (1.0, 1.0, 1.0, 1.0), norm)
    base = N.attr("Col", "Color") if tone is None else (*[float(c) for c in tone[:3]], 1.0)
    N.put(bsdf.inputs["Base Color"], N.mix_rgb("MULTIPLY", 1.0, base, pat))
    bw = N.new("ShaderNodeRGBToBW"); N.put(bw.inputs[0], rough)
    lo, hi = (float(x) for x in spec["rough"])
    N.put(bsdf.inputs["Roughness"], N.math("ADD", lo, N.math("MULTIPLY", bw.outputs[0], hi - lo)))
    bsdf.inputs["Metallic"].default_value = float(spec["metal"])
    nm = N.new("ShaderNodeNormalMap"); nm.uv_map = uv; N.put(nm.inputs["Color"], nor01); nm.inputs["Strength"].default_value = 1.0
    if height is not None and float(spec["bump"]) > 0:
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = float(spec["bump"]); bp.inputs["Distance"].default_value = 0.003
        N.put(bp.inputs["Height"], height); N.put(bp.inputs["Normal"], nm.outputs["Normal"]); N.put(bsdf.inputs["Normal"], bp.outputs["Normal"])
    else:
        N.put(bsdf.inputs["Normal"], nm.outputs["Normal"])


def _rnd(N):
    r = N.new("ShaderNodeSeparateXYZ"); N.put(r.inputs[0], N.attr(SF.RND))
    sh = N.new("ShaderNodeCombineXYZ"); N.put(sh.inputs["X"], N.math("MULTIPLY", r.outputs["Y"], 37.0))
    N.put(sh.inputs["Y"], N.math("MULTIPLY", r.outputs["Z"], 37.0))
    k = N.math("ADD", 0.75, N.math("MULTIPLY", N.math("FRACT", N.math("MULTIPLY", N.math("ADD", r.outputs["Y"], r.outputs["Z"]), 5.37)), 0.5))
    return r, sh.outputs[0], k                                           # the piece's random value, the scan piece shift, strength 0.75…1.25


def _axis(N, spec):
    """Grain rotation along the piece's long axis: 90° for pieces long along u (rot: long), otherwise 0."""
    return N.math("MULTIPLY", N.attr(GRAIN, "Fac"), 0.5 * math.pi) if spec["rot"] == "long" else 0.0


def _rot_normal(N, n01, ang):
    n2 = N.vmath("SUBTRACT", N.vmath("SCALE", n01, s=2.0), (1.0, 1.0, 1.0))
    nr = N.new("ShaderNodeVectorRotate"); nr.rotation_type = "Z_AXIS"; N.put(nr.inputs["Vector"], n2); N.put(nr.inputs["Angle"], N.math("MULTIPLY", ang, -1.0))
    return N.vmath("ADD", N.vmath("SCALE", nr.outputs[0], s=0.5), (0.5, 0.5, 0.5))


def piece(name, spec, uv="sk_uv", tone=None):
    """A piece of a scan per item (tile, sheet, board, paving stone): shift and rotation by the piece number (sk_rnd); grain —
    rotation 0 or π, jitter ±jitter° (+ 90° for pieces long along u, rot: long); the piece's pattern strength ×0.75…1.25."""
    mt = SF.scan_meta(spec["scan"]); size = float(mt["size_m"][0])
    m = bpy.data.materials.new(name); nt = m.node_tree; N = SF._N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    r, shift, k = _rnd(N)
    if spec["rot"] in ("grain", "long"):
        x2 = N.math("MULTIPLY", r.outputs["X"], 2.0); flip = N.math("FLOOR", x2)
        jit = N.math("MULTIPLY", N.math("SUBTRACT", N.math("SUBTRACT", x2, flip), 0.5), 2.0 * math.radians(float(spec["jitter"])))
        ang = N.math("ADD", N.math("ADD", N.math("MULTIPLY", flip, math.pi), jit), _axis(N, spec))
    else:
        ang = N.math("MULTIPLY", r.outputs["X"], 2.0 * math.pi)
    uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = uv
    rot = N.new("ShaderNodeVectorRotate"); rot.rotation_type = "Z_AXIS"
    N.put(rot.inputs["Vector"], N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / size)); N.put(rot.inputs["Angle"], ang)
    vec = N.vmath("ADD", rot.outputs[0], shift)

    def tex(key, color):
        t = N.new("ShaderNodeTexImage"); t.image = SF._img(mt["maps"][key], color); N.put(t.inputs["Vector"], vec)
        return t
    dif, nor, rgh = tex("Diffuse", True), tex("nor_gl", False), tex("Rough", False)
    dsp = tex("Displacement", False) if "Displacement" in mt["maps"] else None
    _finish(N, bsdf, dif.outputs["Color"], rgh.outputs["Color"], _rot_normal(N, nor.outputs["Color"], ang), spec, _mean(dif.image, True),
            N.math("MULTIPLY", float(spec["strength"]), k), height=dsp.outputs["Color"] if dsp else None, uv=uv, tone=tone)
    m["sk_cover"] = {"scan": spec["scan"], "path": "piece", "square_m": size, "turn": spec["rot"]}
    return m


STRETCH = 4.0          # for fibrous materials the tiling cell is longer along the grain (×4): the seam between pieces runs along the grain, not across it


def _hex_grain(N, st, maps, rot, stretch):
    """Mikkelsen tiling (like core/surface/bl_surface.hex_maps: three pieces on a triangular grid, weights — barycentrics⁷ by
    brightness, Gain3) with a cell grid stretched along v by a factor of stretch: with wood and reed a 14 cm cell tore the
    grain into blotches. The pieces are taken at the original st (the scan scale does not change), the cells at st with a
    compressed v. → {kind: output}."""
    S3 = 2.0 * math.sqrt(3.0)
    sep = N.new("ShaderNodeSeparateXYZ"); N.put(sep.inputs[0], st)
    x, y = N.math("MULTIPLY", sep.outputs["X"], S3), N.math("MULTIPLY", sep.outputs["Y"], S3 / stretch)
    sx = N.math("SUBTRACT", x, N.math("MULTIPLY", y, 0.57735027)); sy = N.math("MULTIPLY", y, 1.15470054)
    bx, by = N.math("FLOOR", sx), N.math("FLOOR", sy)
    fx, fy = N.math("SUBTRACT", sx, bx), N.math("SUBTRACT", sy, by)
    z = N.math("SUBTRACT", N.math("SUBTRACT", 1.0, fx), fy)
    s = N.math("LESS_THAN", z, 1e-7); s2 = N.math("SUBTRACT", N.math("MULTIPLY", s, 2.0), 1.0); ns = N.math("SUBTRACT", 1.0, s)
    ws = [N.math("MULTIPLY", N.math("MULTIPLY", z, -1.0), s2), N.math("SUBTRACT", s, N.math("MULTIPLY", fy, s2)),
          N.math("SUBTRACT", s, N.math("MULTIPLY", fx, s2))]
    verts = [(N.math("ADD", bx, s), N.math("ADD", by, s)), (N.math("ADD", bx, s), N.math("ADD", by, ns)), (N.math("ADD", bx, ns), N.math("ADD", by, s))]
    H = []
    for (vx, vy), w in zip(verts, ws):
        cv = N.new("ShaderNodeCombineXYZ"); N.put(cv.inputs["X"], vx); N.put(cv.inputs["Y"], vy)
        wn = N.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"; N.put(wn.inputs["Vector"], cv.outputs[0])
        rs = N.new("ShaderNodeSeparateColor"); N.put(rs.inputs[0], wn.outputs["Color"])
        ang = N.math("MULTIPLY", N.math("SUBTRACT", wn.outputs["Value"], 0.5), 2.0 * math.pi * rot)
        cen = N.new("ShaderNodeCombineXYZ")                                     # the cell center is in piece coordinates (v stretched back)
        N.put(cen.inputs["X"], N.math("DIVIDE", N.math("ADD", vx, N.math("MULTIPLY", vy, 0.5)), S3))
        N.put(cen.inputs["Y"], N.math("MULTIPLY", N.math("DIVIDE", N.math("MULTIPLY", vy, 0.8660254), S3), stretch))
        rv = N.new("ShaderNodeVectorRotate"); rv.rotation_type = "Z_AXIS"
        N.put(rv.inputs["Vector"], st); N.put(rv.inputs["Center"], cen.outputs[0]); N.put(rv.inputs["Angle"], ang)
        off = N.new("ShaderNodeCombineXYZ"); N.put(off.inputs["X"], N.math("MULTIPLY", rs.outputs["Red"], 13.0))
        N.put(off.inputs["Y"], N.math("MULTIPLY", rs.outputs["Green"], 13.0))
        H.append((N.vmath("ADD", rv.outputs[0], off.outputs[0]), N.math("POWER", N.math("MAXIMUM", w, 0.0), 7.0), ang))
    smp = {k: [] for k in maps}
    for vec, _w, _a in H:
        for k, (im, _c) in maps.items():
            t = N.new("ShaderNodeTexImage"); t.image = im; N.put(t.inputs["Vector"], vec); smp[k].append(t.outputs["Color"])
    ws = []
    for (vec, w7, _a), c in zip(H, smp["Diffuse"]):
        lum = N.new("ShaderNodeRGBToBW"); N.put(lum.inputs[0], c)
        ws.append(N.math("MULTIPLY", N.math("ADD", 0.4, N.math("MULTIPLY", lum.outputs[0], 0.6)), w7))
    tot = N.math("ADD", N.math("ADD", ws[0], ws[1]), ws[2])
    W = SF._gain3(N, [N.math("DIVIDE", x_, N.math("MAXIMUM", tot, 1e-6)) for x_ in ws])
    out = {}
    for k, (im, color) in maps.items():
        parts = []
        for (vec, _w, ang), c, w in zip(H, smp[k], W):
            if k.startswith("nor"):
                n2 = N.vmath("SUBTRACT", N.vmath("SCALE", c, s=2.0), (1.0, 1.0, 1.0))
                rv = N.new("ShaderNodeVectorRotate"); rv.rotation_type = "Z_AXIS"; N.put(rv.inputs["Vector"], n2)
                N.put(rv.inputs["Angle"], N.math("MULTIPLY", ang, -1.0)); c = rv.outputs[0]
            parts.append(N.vmath("SCALE", c, s=w))
        acc = N.vmath("ADD", N.vmath("ADD", parts[0], parts[1]), parts[2])
        if k.startswith("nor"):
            acc = N.vmath("ADD", N.vmath("SCALE", N.vmath("NORMALIZE", acc), s=0.5), (0.5, 0.5, 0.5))
        out[k] = acc
    return out


def plane(name, spec, uv="sk_uv", tone=None):
    """Mikkelsen tiling by the unrolling — without repetition: faces without a piece number (a row as a strip, a cap, a bed, a
    trough, bedding) and long pieces (a 4 m board of boards, a log on a 0.5 m scan — a piece of the scan would repeat along
    it). The field shift is by the piece number: neighboring boards get their own pattern instead of one field across the
    joints. Grain — the tiling pieces are rotated only within ±jitter° (+ 90° for pieces long along u, rot: long)."""
    mt = SF.scan_meta(spec["scan"]); size = float(mt["size_m"][0])
    m = bpy.data.materials.new(name); nt = m.node_tree; N = SF._N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    _r, shift, k = _rnd(N)
    uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = uv
    ax = _axis(N, spec)
    rot = N.new("ShaderNodeVectorRotate"); rot.rotation_type = "Z_AXIS"
    N.put(rot.inputs["Vector"], N.vmath("SCALE", uvn.outputs["UV"], s=1.0 / size)); N.put(rot.inputs["Angle"], ax)
    st = N.vmath("ADD", rot.outputs[0], shift)
    maps = {"Diffuse": (SF._img(mt["maps"]["Diffuse"], True), True), "nor_gl": (SF._img(mt["maps"]["nor_gl"], False), False),
            "Rough": (SF._img(mt["maps"]["Rough"], False), False)}
    if spec["rot"] in ("grain", "long"):                                       # grain: the cell is elongated along it, rotation ±jitter°
        o = _hex_grain(N, st, maps, float(spec["jitter"]) / 180.0, STRETCH)
    else:
        o = SF.hex_maps(N, st, maps, rot=0.5)
    _finish(N, bsdf, o["Diffuse"], o["Rough"], _rot_normal(N, o["nor_gl"], ax) if spec["rot"] == "long" else o["nor_gl"], spec,
            _mean(maps["Diffuse"][0], True), N.math("MULTIPLY", float(spec["strength"]), k), uv=uv, tone=tone)
    m["sk_cover"] = {"scan": spec["scan"], "path": "Mikkelsen tiling", "square_m": size}
    return m


def far(mat, spec):
    """Far-view material with the roughness and metallic of the substance (pattern and tone belong to the part and do not
    change)."""
    key = f"{mat.name}_tx"
    if key in bpy.data.materials: return bpy.data.materials[key]
    m = mat.copy(); m.name = key
    bs = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    mt = SF.scan_meta(spec["scan"]); im = SF._img(mt["maps"]["Rough"], False)
    lo, hi = (float(x) for x in spec["rough"])
    bs.inputs["Roughness"].default_value = lo + (hi - lo) * float(_mean(im, False).mean())
    bs.inputs["Metallic"].default_value = float(spec["metal"])
    return m


def material(sub, mode, spec, tone=None):
    key = (sub, mode, None if tone is None else tuple(round(float(c), 4) for c in tone[:3]))
    if key not in _MATS:
        _MATS[key] = (piece if mode == "piece" else plane)(f"sk_cover_{sub}_{mode}" + ("_t" if tone is not None else ""), spec, tone=tone)
    return _MATS[key]


def _face_ints(me, name):
    a = me.attributes.get(name)
    if a is None or a.domain != "FACE": return None
    v = np.empty(len(me.polygons), np.int32); a.data.foreach_get("value", v)
    return v


def _extent(me, sid, uv="sk_uv"):
    """Extent of a piece along u and v of the unrolling, m: per face its own (faces of one piece with different normals are
    unrolled in different axes — do not mix them), per piece the largest of its faces. → (eu, ev) by the length of the
    faces."""
    n = len(me.polygons)
    a = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[uv].data.foreach_get("uv", a); a = a.reshape(-1, 2)
    ls = np.empty(n, np.int32); me.polygons.foreach_get("loop_start", ls)
    ext = np.maximum.reduceat(a, ls) - np.minimum.reduceat(a, ls)
    _u, inv = np.unique(sid, return_inverse=True)
    top = np.zeros((len(_u), 2), np.float32); np.maximum.at(top, inv, ext)
    return top[inv, 0], top[inv, 1]


def _second(prop):
    """Role of the second substance of a part: paving — mortar (bedding), half-timbering — finish (infill)."""
    return {"sk_paving": ROLES["mortar"], "sk_timber": ROLES["finish"]}.get(prop)


def run(scene, over=None, log=print):
    """Substance on the scene's covering objects (properties sk_roofing, sk_paving, sk_timber): slots by (face substance, piece
    | tiling), piece number sk_rnd, unrolling by the face plane, the grain axis; far view — the substance gloss. → counts."""
    tab, parts = knowledge(over); _MATS.clear()
    st = {"n_objects": 0, "far": 0, "n_faces_piece": 0, "n_faces_tile": 0, "substances": {}, "without_substance": []}
    metal, mortar = ROLES["metal"], ROLES["mortar"]
    for ob in list(scene.objects):
        prop = next((p for p in PROPS if ob.type == "MESH" and p in ob), None)
        if prop is None: continue
        info = json.loads(ob[prop]); part = parts[prop]; kd = part["kinds"].get(info.get("kind")) or {}
        sub = info.get("substance") or kd.get("substance"); second = kd.get("second")
        if sub not in tab:
            st["without_substance"].append(sub); continue
        me = ob.data; n = len(me.polygons)
        if info.get("far") or ob.name.startswith("pave_bed_"):                 # far view: the roof shell; the paving bedding with a far-view pattern
            for sl in ob.material_slots:
                if sl.material is None: continue
                if sl.material.name.startswith("paving_bedding") and second in tab:   # the bedding under the stones — the bedding substance
                    SF.block_random(ob); ML.plane_uv(ob)
                    sl.material = material(second, "plane", tab[second], tone=part["tones"].get(second))
                else:
                    sl.material = far(sl.material, tab[sub])
            st["far"] += 1; continue
        role, sid = _face_ints(me, "sk_role"), _face_ints(me, "sk_stone")
        if role is None or sid is None or not n: continue
        main = ROLES.get(kd.get("role") or "", -1)
        if main < 0:                                                            # the role of the kind's pieces is unknown — by the majority of pieces
            own = role[sid != 0] if (sid != 0).any() else role; main = int(np.bincount(own).argmax())
        sec = _second(prop)
        if prop == "sk_roofing":
            other = np.where(role == mortar, "mortar", np.where(role == metal, "lead", sub))
        else:
            other = np.where((role == sec) & (second is not None), second or sub, sub)
        subs = np.where(role == main, sub, other)
        miss = sorted(set(subs.tolist()) - set(tab)); st["without_substance"] += miss
        subs = np.where(np.isin(subs, list(tab)), subs, sub)
        SF.block_random(ob); ML.plane_uv(ob)
        eu, ev = _extent(me, sid)
        us, ui = np.unique(subs, return_inverse=True)                          # the scan square size — once per substance
        size = np.asarray([float(SF.scan_meta(tab[s_]["scan"])["size_m"][0]) for s_ in us.tolist()], np.float32)[ui]
        modes = np.where((sid == 0) | (np.maximum(eu, ev) > LONG * size), "plane", "piece")
        ga = me.attributes.get(GRAIN) or me.attributes.new(GRAIN, "FLOAT", "FACE")
        ga.data.foreach_set("value", (eu > ev).astype(np.float32))
        ukeys, slot = np.unique(np.char.add(np.char.add(subs.astype(str), "|"), modes.astype(str)), return_inverse=True)   # face slot — numpy (hundreds of thousands of faces)
        has_col = me.color_attributes.get("Col") is not None
        me.materials.clear()
        for key in ukeys.tolist():
            s_, md = key.split("|"); me.materials.append(material(s_, md, tab[s_], tone=None if has_col else part["tones"].get(s_)))
        me.polygons.foreach_set("material_index", slot.astype(np.int32)); me.update()
        st["n_objects"] += 1; st["n_faces_piece"] += int((modes == "piece").sum()); st["n_faces_tile"] += int((modes == "plane").sum())
        for s_ in set(subs.tolist()): st["substances"][s_] = st["substances"].get(s_, 0) + 1
    st["without_substance"] = sorted(set(x for x in st["without_substance"] if x))
    log("coverings:", json.dumps(st, ensure_ascii=False))
    return st
