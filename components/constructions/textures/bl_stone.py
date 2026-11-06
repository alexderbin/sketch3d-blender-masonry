"""Stone v2 — substance on masonry blocks on top of the engine (core/surface/bl_surface.scan_stone: a scan per block, tone
Col): roughness, masks, color. The masks are bl_masks (the face frame, chamfers, the stone's height above the ground)
and the stone number (sk_rnd); it has no images of its own. The stone's average tone does not change: blocks up close
match the rows and the masonry slab far away, and a light or dark stone is the masonry's decision.

The weathering w of a stone (0…1) is a single knob per stone: chance (stones from one quarry, but one lay in the rain,
another was replaced during repairs) + the place in the wall (the bottom up to ~1.5 m — splashes and damp). By w:
  the substance pattern is more visible (×0.5…1.5 of the scan strength), the scan relief is deeper, the roughness is
  higher (not below 0.45 — the stone is not "plastic"), the edges are worn: a wider wear band, more chips. A fresh stone
  is even, smoother, with sharp edges.
Edges (edges > 0): a band at the edge of the face and chamfers is lighter by a few percent (wear removes the tan), the
  relief falls off toward the edge (rounding catches the light); chips — noise in the edge band: a fresh stone is
  lighter and rougher, a shallow dent.
Edging (for finishes: quoins, voussoirs, sills): a smooth dressed band along the face edge — the scan relief is weaker,
  smoother.
  import bl_stone as ST; m = ST.stone("rock_surface", edges=1, weathering=0.6, edging=0.0)"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
from core.surface import bl_surface as SF  # noqa: E402  (surface engine)

R_EDGE = 0.012          # m — the wear band at the face edge of a fresh stone (up to ×2.5 for a weathered one)
R_CHIP = 0.04           # m — where chips occur
CHIP_SCALE = 38.0       # 1/m — chip grain
BASE_ZONE = (0.4, 1.6)  # m above the ground — the splash and damp zone (above it, no increment)


def _src(sock):
    return sock.links[0].from_socket if sock.is_linked else None


def _ramp(N, x, a, b, va=1.0, vb=0.0):
    """Smooth transition of x: a → va, b → vb (Map Range, smoothstep)."""
    mr = N.new("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"; mr.clamp = True
    N.put(mr.inputs["Value"], x); N.put(mr.inputs["From Min"], a); N.put(mr.inputs["From Max"], b)
    mr.inputs["To Min"].default_value = va; mr.inputs["To Max"].default_value = vb
    return mr.outputs["Result"]


def _edge_dist(N, uv="sk_uv"):
    """Distance to the edge of a stone face (m): min(u − u0, u1 − u, v − v0, v1 − v) by the face frame (bl_masks.frame)."""
    uvn = N.new("ShaderNodeUVMap"); uvn.uv_map = uv
    p = N.new("ShaderNodeSeparateXYZ"); N.put(p.inputs[0], uvn.outputs["UV"])
    lo = N.new("ShaderNodeSeparateXYZ"); N.put(lo.inputs[0], N.attr("sk_lo"))
    hi = N.new("ShaderNodeSeparateXYZ"); N.put(hi.inputs[0], N.attr("sk_hi"))
    d = [N.math("SUBTRACT", p.outputs["X"], lo.outputs["X"]), N.math("SUBTRACT", hi.outputs["X"], p.outputs["X"]),
         N.math("SUBTRACT", p.outputs["Y"], lo.outputs["Y"]), N.math("SUBTRACT", hi.outputs["Y"], p.outputs["Y"])]
    return N.math("MINIMUM", N.math("MINIMUM", d[0], d[1]), N.math("MINIMUM", d[2], d[3]))


def _weather(N, rnd, spread, place):
    """w = 0.5 ± spread/2 by the stone's chance + place · the wall-bottom zone (sk_h), 0…1."""
    w = N.math("ADD", 0.5 - spread / 2, N.math("MULTIPLY", rnd.outputs["Y"], spread))
    if place > 0:
        low = _ramp(N, N.attr("sk_h", "Fac"), BASE_ZONE[0], BASE_ZONE[1])   # 1 at the ground, 0 above the zone
        w = N.math("ADD", w, N.math("MULTIPLY", low, 0.35 * place))
    return N.math("MINIMUM", N.math("MAXIMUM", w, 0.0), 1.0)


def stone(sid, strength=1.0, edges=1.0, weathering=0.6, edging=0.0, place=1.0, name=None):
    """Stone v2 on blocks: the engine's scan_stone + weathering per stone, edges and chips, edging. All 0 — same as v1."""
    m = SF.scan_stone(sid, strength=strength, name=name)
    nt = m.node_tree; N = SF._N(nt)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    rnd = N.new("ShaderNodeSeparateXYZ"); N.put(rnd.inputs[0], N.attr(SF.RND))
    w = _weather(N, rnd, weathering, place) if weathering > 0 else 0.5
    info = {"edges": edges, "weathering": weathering, "edging": edging, "place": place}

    if weathering > 0:                                                      # substance pattern: more visible on a weathered stone (tone ~1)
        pat = next((n for n in nt.nodes if n.type == "MIX" and n.blend_type == "MIX" and n.data_type == "RGBA"
                    and any(i.identifier == "B_Color" and i.is_linked and i.links[0].from_node.type == "MIX"
                            and i.links[0].from_node.blend_type == "DIVIDE" for i in n.inputs)), None)
        if pat is not None:
            f = next(i for i in pat.inputs if i.identifier == "Factor_Float"); s0 = float(f.default_value)
            N.put(f, N.math("MULTIPLY", N.math("ADD", 0.5, w), s0))

    d = _edge_dist(N)
    bev = N.attr("sk_bevel", "Fac")
    r_e = N.math("MULTIPLY", R_EDGE, N.math("ADD", 1.0, N.math("MULTIPLY", w, 1.5))) if weathering > 0 else R_EDGE
    band = N.math("MAXIMUM", _ramp(N, d, 0.0, r_e), bev) if edges > 0 else 0.0
    chip = 0.0
    if edges > 0:
        pos = N.new("ShaderNodeNewGeometry")
        nz = N.new("ShaderNodeTexNoise", Scale=CHIP_SCALE, Detail=2.0, Roughness=0.6); N.put(nz.inputs["Vector"], pos.outputs["Position"])
        near = N.math("MAXIMUM", _ramp(N, d, R_CHIP * 0.5, R_CHIP), bev)
        thr = N.math("SUBTRACT", 0.70, N.math("MULTIPLY", w, 0.10))     # a weathered stone has more chips
        chip = N.math("MULTIPLY", _ramp(N, N.math("SUBTRACT", nz.outputs["Fac"], thr), 0.0, 0.05, 0.0, 1.0), near)
        chip = N.math("MULTIPLY", chip, edges)
        bc = _src(bsdf.inputs["Base Color"])                            # color: wear is lighter, a chip shows fresh stone
        k = N.math("ADD", 1.0, N.math("ADD", N.math("MULTIPLY", band, 0.06 * edges), N.math("MULTIPLY", chip, 0.14)))
        kc = N.new("ShaderNodeCombineColor"); N.put(kc.inputs["Red"], k); N.put(kc.inputs["Green"], k); N.put(kc.inputs["Blue"], k)
        N.put(bsdf.inputs["Base Color"], N.mix_rgb("MULTIPLY", 1.0, bc, kc.outputs[0]))

    r_src = _src(bsdf.inputs["Roughness"])                              # roughness
    if r_src is not None and (weathering > 0 or edges > 0 or edging > 0):
        bw = N.new("ShaderNodeRGBToBW"); N.put(bw.inputs[0], r_src); r = bw.outputs[0]
        if weathering > 0:
            r = N.math("MULTIPLY", r, N.math("ADD", 0.85, N.math("MULTIPLY", w, 0.3)))
        if edges > 0:
            r = N.math("ADD", N.math("SUBTRACT", r, N.math("MULTIPLY", band, 0.05 * edges)), N.math("MULTIPLY", chip, 0.15))
        if edging > 0:
            r = N.math("SUBTRACT", r, N.math("MULTIPLY", _ramp(N, d, edging * 0.8, edging), 0.08))
        N.put(bsdf.inputs["Roughness"], N.math("MAXIMUM", N.math("MINIMUM", r, 1.0), 0.45))

    nm = next((n for n in nt.nodes if n.type == "NORMAL_MAP"), None)
    if nm is not None and (weathering > 0 or edging > 0):                     # scan relief: deeper for a weathered stone, weaker in the edging
        st_ = N.math("ADD", 0.5, w) if weathering > 0 else 1.0
        if edging > 0:
            st_ = N.math("MULTIPLY", st_, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", _ramp(N, d, edging * 0.8, edging), 0.7)))
        N.put(nm.inputs["Strength"], st_)
    if edges > 0:                                                       # relief: falls off toward the edge, a chip is a dent
        n_src = _src(bsdf.inputs["Normal"])
        h = N.math("SUBTRACT", _ramp(N, d, 0.0, r_e, 0.0, 1.0), N.math("MULTIPLY", chip, 0.6))
        bp = N.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.5 * edges; bp.inputs["Distance"].default_value = 0.003
        N.put(bp.inputs["Height"], h)
        if n_src is not None:
            N.put(bp.inputs["Normal"], n_src)
        N.put(bsdf.inputs["Normal"], bp.outputs["Normal"])
    m["sk_stone_v2"] = info
    return m


def mortar(rgb, strength=1.0, scan="grey_plaster_03", name="sk_surf_mortar"):
    """Mortar in the joints and holes: a scan of coarse plaster with Mikkelsen tiling (scan=None — the engine's sandy noise: on
    an open putlog hole it read as a smooth gray patch), the tone is the masonry substance (packs.yaml "substances: mortar"),
    not the Col of the building mass (the mass has the palette's stone color)."""
    m = SF.scan_plane(scan, strength=strength, name=name) if scan else SF.mortar(strength=strength, name=name)
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type == "ATTRIBUTE" and n.attribute_name == "Col":
            rgbn = nt.nodes.new("ShaderNodeRGB"); rgbn.outputs[0].default_value = (*[float(c) for c in rgb[:3]], 1.0)
            for l in list(n.outputs["Color"].links):
                nt.links.new(rgbn.outputs[0], l.to_socket)
            nt.nodes.remove(n)
    m["sk_mortar_rgb"] = [float(c) for c in rgb[:3]]; m["sk_mortar_scan"] = scan or "noise"
    return m
