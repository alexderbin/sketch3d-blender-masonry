"""The masonry step in Blender — the Masonry component.

The step input is run(items, plan, coll, …): masonry surfaces (a side passport masonry.Strip.spec + a pack, openings, gaps) and the plan
(components/constructions/masonry/masonry.py resolve: kind, thresholds, packs, substances). The layout and the maps are computed by the project's Python
(masonry.py build — Blender has no OpenCV and YAML), here — the geometry:
  blocks — stones in a batch: a prism with a chamfer, the face flat / "pillow" / rusticated / rough; a stone's randomness comes from its number;
  rows — a prism per row along the stone edges (the bed shadow is real, headers and tone come from a map); a slab with a map goes under everything;
  body — mortar with openings (for walls without the building mass).
The output contract (read by the "surface" step, core/surface/bl_surface.py): on faces `sk_role` (FACE INT: 0 wall, 1 finish, 2 roof, 3 wood, 4 metal, 5 glass — shared
by all buildings; 6 mortar, 7 rubble — masonry), `sk_stone` (the block number), the stone's tone is the colour `Col`; materials come in slots by role `sk_<role>`
(not one for everything: the shader's limit is 16 textures); rows and the slab use the segment's map material. Objects: masonry_<tag> (blocks), plate_<tag> (rows and slab),
body_<tag> (body).
Used by: components/constructions/masonry/bl_masonry.py (a castle by steps).
"""
import json
import math
import os
import subprocess
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
from components.constructions.masonry import masonry as M  # noqa: E402

from core.contract.contract import ROLES   # object contract — a single table (no copies)
ROLE_ROUGH = {0: 0.85, 1: 0.75, 6: 0.95, 7: 0.9, 8: 0.85}
RNG = np.random.default_rng(5)                                                   # a fallback for calls without a number


def stone_rng(sid):
    """A stone's randomness comes from its number (sk_stone): the same stone is identical in any mode (all as blocks, by camera), run and build order."""
    return np.random.default_rng([int(sid) & 0x7FFFFFFF, int(sid) >> 31 & 0xFFFF, 1013])


class Acc:
    """A mesh accumulator: faces with colour (previs), role and block number → an object with material slots by role."""

    def __init__(self): self.v, self.f, self.c, self.r, self.s, self.n, self.vn = [], [], [], [], [], 0, []

    def add(self, pts, col, role, sid, vn=None):
        pts = np.asarray(pts, float); self.v.append(pts); self.f.append(len(pts))
        self.c.append(col); self.r.append(role); self.s.append(sid); self.n += len(pts); self.vn.append(vn)

    def obj(self, name, coll):
        V = np.concatenate(self.v); cnt = np.asarray(self.f, np.int32); starts = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int32)
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V)); me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.loops.add(len(V)); me.loops.foreach_set("vertex_index", np.arange(len(V), dtype=np.int32))
        me.polygons.add(len(cnt)); me.polygons.foreach_set("loop_start", starts); me.update(calc_edges=True)
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        cols = np.repeat(np.asarray(self.c, np.float32), cnt, axis=0)
        ca.data.foreach_set("color", np.column_stack([cols, np.ones(len(cols), np.float32)]).ravel())
        me.color_attributes.active_color = ca
        me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.asarray(self.r, np.int32))
        me.attributes.new("sk_stone", "INT", "FACE").data.foreach_set("value", np.asarray(self.s, np.int32))
        roles = sorted(set(self.r)); slot = {r: i for i, r in enumerate(roles)}
        for r in roles: me.materials.append(mat_role(r))
        me.polygons.foreach_set("material_index", np.asarray([slot[r] for r in self.r], np.int32)); me.update()
        if any(x is not None for x in self.vn):                                  # a smooth face (a cobblestone dome): the corners have their own normals,
            out = []                                                             # other faces use the face normal (flat shading)
            for P_, N_ in zip(self.v, self.vn):
                if N_ is None:
                    n = np.sum(np.cross(P_, np.roll(P_, -1, 0)), axis=0); out.append(np.repeat((n / max(np.linalg.norm(n), 1e-12))[None], len(P_), 0))
                else:
                    N_ = np.asarray(N_, float); out.append(N_ / np.maximum(np.linalg.norm(N_, axis=1, keepdims=True), 1e-12))
            me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool)); me.normals_split_custom_set(np.concatenate(out).tolist())
        ob = bpy.data.objects.new(name, me); coll.objects.link(ob); return ob


_MATS = {}


def mat_role(role):
    """A role's material (no textures yet): the colour is the attribute Col (previs), the roughness depends on the role."""
    if role not in _MATS:
        name = next(k for k, v in ROLES.items() if v == role)
        m = bpy.data.materials.new(f"sk_{name}"); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"; nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
        bs.inputs["Roughness"].default_value = ROLE_ROUGH.get(role, 0.85); _MATS[role] = m
    return _MATS[role]


def inset(p, d):
    """Shrink a polygon by d (the vertices shift along the bisector; for sharp corners — no more than 3d)."""
    p = np.asarray(p, float); n = len(p)
    if n < 3 or abs(d) < 1e-9: return p
    s = 1.0 if 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) > 0 else -1.0
    out = []
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e1 = (b - a) / (np.linalg.norm(b - a) + 1e-12); e2 = (c - b) / (np.linalg.norm(c - b) + 1e-12)
        n1 = s * np.array([-e1[1], e1[0]]); n2 = s * np.array([-e2[1], e2[0]]); bis = n1 + n2; L = np.linalg.norm(bis)
        out.append(b + n1 * d if L < 1e-6 else b + bis / L * d / max(0.33, float(bis / L @ n1)))
    return np.array(out)


def insert_folds(p, folds):
    """Polygon vertices (u, v) on the fold lines u = f (path stations with a kink): an edge crossing a fold is split."""
    out, n = [], len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]; out.append(a)
        cross = [f for f in folds if (a[0] - f) * (b[0] - f) < -1e-12]
        for f in sorted(cross, key=lambda f: (f - a[0]) / (b[0] - a[0])):
            out.append(a + (f - a[0]) / (b[0] - a[0]) * (b - a))
    return np.array(out)


def split_by_folds(q, folds, eps=1e-6):
    """A ring of vertices (u, v) → pieces by folds: each fold with two vertices on it splits a piece in two. → [indices]."""
    pieces = [list(range(len(q)))]
    for f in folds:
        nxt = []
        for pc in pieces:
            on = [k for k, i in enumerate(pc) if abs(q[i][0] - f) < eps]
            if len(on) != 2: nxt.append(pc); continue
            a, b = on
            nxt += [pc[a:b + 1], pc[b:] + pc[:a + 1]]
        pieces = nxt
    return [pc for pc in pieces if len(pc) >= 3]


def _convex(p):
    return len(p) <= 14 and np.all(np.cross(np.roll(p, -1, 0) - p, np.roll(p, -2, 0) - np.roll(p, -1, 0)) >= -1e-9)


def stones_batch(acc, S, items, live=1.0):
    """The stones of a side in a batch: a prism with a chamfer, the mapping (u, v, d) → world — one S.world call for all vertices.
    items — dict(poly, d0, d1, col, role, sid, cf, cb, finish, pillow, boss, margin, rough). Face finish:
      flat / pillow — a flat face or a "pillow" (the centre forward by pillow; from 1 cm — as a dome: a ring + the centre);
      boss — rustication: a flat border of margin along the edge, a convex rough boss of height boss in the middle;
      rough — a rough (split) face: an uneven dome — the edge wanders in depth, the ring and the centre go forward by a fraction rough
      (finer for a small stone: rough × size / 0.4 m, at least a third); a stone smaller than 0.3 m is a plane tilted by ±5°.
    The chamfer is cf, but no more than 7 % of the stone size (√area). live — the "liveliness" of hand-laid masonry: a face shift of ±6 mm, rotation of ±1°, tilt of ±0.7°
    (0 — ruler-straight). A stone across a path kink (a quoin at a corner) gets its face in flat pieces along the folds, without finish, rotation or tilt."""
    U, V_, D, recs = [], [], [], []

    def put(q, d):
        i0 = len(U); U.extend(q[:, 0]); V_.extend(q[:, 1]); D.extend(np.broadcast_to(d, len(q))); return i0

    for it in items:
        p = np.asarray(it["poly"], float)
        if len(p) < 3: continue
        RNG = stone_rng(it.get("sid", 0))
        if 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) < 0: p = p[::-1]
        folds = [f for f in getattr(S, "folds", []) if p[:, 0].min() + 0.02 < f < p[:, 0].max() - 0.02]
        if folds: p = insert_folds(p, folds)
        n = len(p); c2 = p.mean(0); cb = it.get("cb", 0.0)
        sz = math.sqrt(max(abs(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])), 1e-6))
        cf = min(it.get("cf", 0.014), max(0.004, 0.07 * sz))                      # a small stone's chamfer is smaller (otherwise it looks like a "crystal")
        lv = float(it.get("live", live))                                         # the "liveliness" is per item (pack) or global
        a = math.radians(RNG.uniform(-1.0, 1.0) * lv); ca, sa = math.cos(a), math.sin(a)
        d0 = it["d0"] + RNG.uniform(-0.006, 0.006) * lv; tu, tv = RNG.uniform(-0.012, 0.012, 2) * lv
        if folds: ca, sa, tu, tv = 1.0, 0.0, 0.0, 0.0

        def tr(q):
            q = np.atleast_2d(q) - c2; dd = tu * q[:, 0] + tv * q[:, 1]
            return np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1) + c2, dd

        rec = dict(n=n, col=it["col"], role=it["role"], sid=it["sid"], cb=cb, finish="flat")
        pin = inset(p, cf)
        rec["pieces"] = split_by_folds(pin, folds) if folds else None
        fin = it.get("finish", "flat") if not folds and (_convex(p) or it.get("finish") == "dome") else "flat"   # a dome — also for an almost convex cell
        if fin == "rough" and sz < 0.3:                                          # small rubble: the face is a tilted plane
            fin = "tilt"; tu, tv = RNG.uniform(-0.09, 0.09, 2)                    # ±5° (a dome 10 cm high reads as faceting)
        q, dd = tr(pin)
        rgh = it.get("rough", 0.02) * min(max(sz / 0.4, 0.35), 1.0)                # a small stone gets a finer chip
        if fin == "rough":                                                       # a split face: the edge varies randomly in depth
            dd = dd + RNG.uniform(-0.25, 0.25, len(q)) * rgh
        rec["r0"] = put(q, d0 + dd)
        q, dd = tr(p); rec["r1"] = put(q, d0 + cf + dd); rec["r2"] = put(q, it["d1"] - cb + dd)
        if cb > 0: q, dd = tr(inset(p, cb)); rec["r3"] = put(q, it["d1"] + dd)
        w_min = min(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        if fin == "boss" and w_min > 2 * (cf + it.get("margin", 0.035)) + 0.08:   # rustication: the border, the boss, the top of the boss
            m, h = it.get("margin", 0.035), it.get("boss", 0.05)
            q, dd = tr(inset(p, cf + m)); rec["rm"] = put(q, d0 + dd)
            q, dd = tr(inset(p, cf + m + 0.02)); q = q + RNG.uniform(-0.01, 0.01, q.shape)
            rec["rb"] = put(q, d0 - h * 0.75 + dd + RNG.uniform(-0.25, 0.25, len(q)) * h)
            rec["C"] = put(np.atleast_2d(c2), d0 - h); rec["finish"] = "boss"
        elif fin == "dome":                                                       # a rounded stone: two rings and a centre, smooth
            h = it.get("dome", 0.04) * min(max(sz / 0.25, 0.5), 1.2); rec["D"] = []
            for s_, hh in ((0.8, 0.58), (0.46, 0.9)):
                q, dd = tr(c2 + (pin - c2) * s_); rec["D"].append(put(q, d0 + dd - h * hh))
            q, dd = tr(c2); rec["C"] = put(q, d0 + dd - h); rec["finish"] = "dome"
        elif fin == "rough" or (it.get("pillow", 0) >= 0.01 and not folds and _convex(p)):   # dome: a ring + a centre (not a pyramid)
            rough_ = fin == "rough"; h = rgh if rough_ else it["pillow"]
            ring = c2 + (pin - c2) * (RNG.uniform(0.45, 0.62) if rough_ else 0.62)
            if rough_: ring = ring + RNG.uniform(-1, 1, ring.shape) * 0.05 * np.ptp(pin, 0).min()
            q, dd = tr(ring); rec["I"] = put(q, d0 + dd - h * (RNG.uniform(0.45, 0.85, len(q)) if rough_ else 0.8))
            rec["C"] = put(np.atleast_2d(c2), d0 - h * (RNG.uniform(0.7, 1.0) if rough_ else 1.0)); rec["finish"] = fin if rough_ else "pillow"
        elif it.get("pillow", 0) > 0 and fin != "tilt" and not folds and _convex(p):
            rec["C"] = put(np.atleast_2d(c2), d0 - it["pillow"]); rec["finish"] = "pillow"
        rec["o0"] = put(np.atleast_2d(c2), d0); rec["o1"] = put(np.atleast_2d(c2), d0 + 0.01)
        recs.append(rec)
    if not recs: return
    W = S.world(np.asarray(U), np.asarray(V_), np.asarray(D))
    for rec in recs:
        n = rec["n"]; col, role, sid = rec["col"], rec["role"], rec["sid"]
        r0, r1, r2 = W[rec["r0"]:rec["r0"] + n], W[rec["r1"]:rec["r1"] + n], W[rec["r2"]:rec["r2"] + n]
        nrm = np.cross(r0, np.roll(r0, -1, 0)).sum(0); out = W[rec["o0"]] - W[rec["o1"]]
        Fo = (lambda q: q) if float(nrm @ out) >= 0 else (lambda q: q[::-1])
        if rec.get("pieces"):                                                    # the face in flat pieces between folds
            for pc in rec["pieces"]: acc.add(Fo(r0[pc]), col, role, sid)
        elif rec["finish"] == "boss":
            rm, rb, C = W[rec["rm"]:rec["rm"] + n], W[rec["rb"]:rec["rb"] + n], W[rec["C"]]
            colb = tuple(np.clip(np.asarray(col) * 0.96, 0, 1))
            for i in range(n):
                j = (i + 1) % n
                acc.add(Fo(np.array([rm[i], rm[j], r0[j], r0[i]])), col, role, sid)        # border
                acc.add(Fo(np.array([rb[i], rb[j], rm[j], rm[i]])), colb, role, sid)       # sides of the boss
                acc.add(Fo(np.array([rb[i], rb[j], C])), colb, role, sid)                  # top of the boss
        elif rec["finish"] == "dome":                                            # the corner normal is the mean over the dome faces at the vertex
            A_, B_, C = W[rec["D"][0]:rec["D"][0] + n], W[rec["D"][1]:rec["D"][1] + n], W[rec["C"]]
            fs, vn = [], {}
            for i in range(n):
                j = (i + 1) % n
                for key, pts in (([("r", i), ("r", j), ("a", j), ("a", i)], [r0[i], r0[j], A_[j], A_[i]]),
                                 ([("a", i), ("a", j), ("b", j), ("b", i)], [A_[i], A_[j], B_[j], B_[i]]),
                                 ([("b", i), ("b", j), ("c", 0)], [B_[i], B_[j], C])):
                    pts, key = Fo(np.asarray(pts)), Fo(key); fn = np.sum(np.cross(pts, np.roll(pts, -1, 0)), axis=0)
                    for k_ in key: vn[k_] = vn.get(k_, 0.0) + fn
                    fs.append((pts, key))
            for pts, key in fs: acc.add(pts, col, role, sid, vn=np.asarray([vn[k_] for k_ in key]))
        elif "I" in rec:
            I_, C = W[rec["I"]:rec["I"] + n], W[rec["C"]]
            for i in range(n):
                j = (i + 1) % n
                acc.add(Fo(np.array([r0[i], r0[j], I_[j], I_[i]])), col, role, sid)
                acc.add(Fo(np.array([I_[i], I_[j], C])), col, role, sid)
        elif "C" in rec:
            C = W[rec["C"]]
            for i in range(n): acc.add(Fo(np.array([r0[i], r0[(i + 1) % n], C])), col, role, sid)
        else:
            acc.add(Fo(r0), col, role, sid)
        for i in range(n):
            j = (i + 1) % n
            acc.add(Fo(np.array([r1[i], r1[j], r0[j], r0[i]])), col, role, sid)
            acc.add(Fo(np.array([r2[i], r2[j], r1[j], r1[i]])), col, role, sid)
        if "r3" in rec:
            r3 = W[rec["r3"]:rec["r3"] + n]
            for i in range(n):
                j = (i + 1) % n; acc.add(Fo(np.array([r3[i], r3[j], r2[j], r2[i]])), col, role, sid)
            acc.add(Fo(r3[::-1]), col, role, sid)



# ── the masonry step: surfaces → blocks, rows, slab, body ────────────────────────────────────────────
def mat_mask_img(name, alb_path, hgt_path, bump=0.03):
    """A segment map material: colour (8-bit PNG, sRGB) and height (8-bit PNG, Non-Color) → bump."""
    ia = bpy.data.images.load(alb_path, check_existing=True); ih = bpy.data.images.load(hgt_path, check_existing=True)
    ih.colorspace_settings.name = "Non-Color"
    m = bpy.data.materials.new(name); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bs.inputs["Roughness"].default_value = 0.85
    na = nt.nodes.new("ShaderNodeTexImage"); na.image = ia; nh = nt.nodes.new("ShaderNodeTexImage"); nh.image = ih
    for i in (na, nh): i.extension = "EXTEND"
    bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Distance"].default_value = bump
    nt.links.new(na.outputs["Color"], bs.inputs["Base Color"]); nt.links.new(nh.outputs["Color"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], bs.inputs["Normal"])
    return m


def cam_px(cams, fpx):
    """A function of px per row for a point and a row height: the maximum over the cameras (the frames of a shot) — no flicker within a shot."""
    locs = [np.asarray(c, float) for c in cams]
    def f(P, h):
        d = min(float(np.linalg.norm(P - L_)) for L_ in locs); return fpx * h / max(d, 0.1)
    return f


def seg_texels(S, pack, px_at, seg=10.0):
    """Texel size of the side's segments for the camera: course / (2 · px of the segment's nearest point), 6 mm … 8 cm."""
    course = float(np.mean(pack["course"])); out = []
    for i in range(int(math.ceil(S.L / seg))):
        uu = np.linspace(i * seg, min(S.L, (i + 1) * seg), 6)
        pts = S.world(np.concatenate([uu, uu]), np.concatenate([S.base_at(uu), S.top_at(uu)]), 0.0)
        out.append(float(np.clip(course / (2.0 * max(px_at(P_, course) for P_ in pts)), 0.006, 0.08)))
    return out


def row_faces(S, run, seg, pack, V, F, UV, MI):
    """A row: the row's stones in sequence (not blocks) — as one prism along their edges (the bottom left to right, the top right to left — as in the
    layout), the face at depth −bed, a chamfer (the bed shadow is real); headers and shades come from the map of the segment that holds the middle of the row.
    Adds faces to the slab's lists."""
    bed = pack.get("bed", 0.025); cf = min(pack.get("cf", 0.014), 0.014)
    bot = np.concatenate([np.asarray(p)[:len(p) // 2] for p in run]); top = np.concatenate([np.asarray(p)[len(p) // 2:] for p in run[::-1]])
    poly = np.vstack([bot, top]); jh = pack["joint"] / 2
    if len(poly) < 4: return
    um = float(poly[:, 0].mean()); i = min(int(um // seg), int(math.ceil(S.L / seg)) - 1)
    s0, s1 = i * seg, min(S.L, (i + 1) * seg); v0, v1 = M.seg_vrange(S, s0, s1)
    uvf = lambda q: [((u - s0 + M.MARGIN) / (s1 - s0 + 2 * M.MARGIN), (v - v0) / (v1 - v0)) for u, v in q]
    p1 = inset(poly, jh); p0 = inset(poly, jh + cf)
    r0 = S.world(p0[:, 0], p0[:, 1], -bed); r1 = S.world(p1[:, 0], p1[:, 1], -bed + cf)
    nrm = np.zeros(3); n = len(r0)
    for k in range(n): nrm += np.cross(r0[k], r0[(k + 1) % n])
    c = poly.mean(0); out = S.world([c[0]], [c[1]], [-bed])[0] - S.world([c[0]], [c[1]], [-bed + 0.01])[0]
    flip = float(nrm @ out) < 0
    def add(P, Q):
        idx = list(range(len(P)))[::-1] if flip else list(range(len(P)))
        i0 = len(V); V.extend(P[j] for j in idx); F.append(list(range(i0, i0 + len(P)))); UV.extend(uvf([Q[j] for j in idx])); MI.append(i)
    add(r0, p0)
    for k in range(n):
        j = (k + 1) % n
        add(np.array([r1[k], r1[j], r0[j], r0[k]]), np.array([p1[k], p1[j], p0[j], p0[k]]))


def plate(S, L, pack, coll, name, maps_dir, tag, texels, seg=10.0, d=-0.005, runs=None):
    """A side slab with the masonry map: columns along u (path stations, every ≤ 0.5 m, the segment edges), along the height — the bottom, the batter knee
    (outside), the top; at depth d in front of the mass (block stones — one more bed forward). The map comes in segments of seg m, the texel for the camera.
    The role is wall, the block number is in the map; rows go on top as prisms (row_faces)."""
    us = np.unique(np.concatenate([S.U, np.arange(0.0, S.L, 0.5), np.arange(0.0, S.L, seg), [S.L],
                                   [x for g in L.get("gaps", []) for x in g if 0 <= x <= S.L]]))
    V, F, UV, MI, mats = [], [], [], [], []
    nseg = int(math.ceil(S.L / seg))
    for i in range(nseg):
        s0, s1 = i * seg, min(S.L, (i + 1) * seg); v0, v1 = M.seg_vrange(S, s0, s1)
        mats.append(mat_mask_img(f"{name}_seg{i}", os.path.join(maps_dir, f"{tag}_seg{i}_alb.png"),
                                 os.path.join(maps_dir, f"{tag}_seg{i}_hgt.png")))
        cols = us[(us >= s0 - 1e-9) & (us <= s1 + 1e-9)]
        for ua, ub in zip(cols[:-1], cols[1:]):
            if ub - ua < 1e-4: continue
            if any(a < 0.5 * (ua + ub) < b for a, b in L.get("gaps", [])): continue   # a gap (a gate): no slab
            def column(u):
                b, t = float(S.base_at(u)), float(S.top_at(u))
                vs = [b, float(np.interp(u, S.U, S.knee)), t] if S.side > 0 else [b, t]
                return [v for v in vs if b - 1e-6 <= v <= t + 1e-6]
            ca, cb = column(ua), column(ub)
            if len(ca) != len(cb): ca, cb = [ca[0], ca[-1]], [cb[0], cb[-1]]
            for k in range(len(ca) - 1):
                q = [(ua, ca[k]), (ub, cb[k]), (ub, cb[k + 1]), (ua, ca[k + 1])]
                if abs(cb[k + 1] - cb[k]) < 1e-4 and abs(ca[k + 1] - ca[k]) < 1e-4: continue
                P = S.world(np.array([x for x, _ in q]), np.array([y for _, y in q]), d)
                out = S.world([0.5 * (ua + ub)], [0.5 * (ca[k] + ca[k + 1])], [d])[0] - S.world([0.5 * (ua + ub)], [0.5 * (ca[k] + ca[k + 1])], [d + 0.01])[0]
                nrm = np.cross(P[1] - P[0], P[3] - P[0])
                order = [0, 1, 2, 3] if float(nrm @ out) >= 0 else [3, 2, 1, 0]
                i0 = len(V); V += [P[j] for j in order]; F.append(list(range(i0, i0 + 4)))
                UV += [((q[j][0] - s0 + M.MARGIN) / (s1 - s0 + 2 * M.MARGIN), (q[j][1] - v0) / (v1 - v0)) for j in order]; MI.append(i)
    for k_run in (runs or []):                                                    # rows: a prism along the edges of the row's stones
        row_faces(S, k_run, seg, pack, V, F, UV, MI)
    me = bpy.data.meshes.new(name); me.from_pydata(np.asarray(V).tolist(), [], F); me.update()
    uvl = me.uv_layers.new(name="UVMap"); uvl.data.foreach_set("uv", np.asarray(UV, np.float32).ravel())
    me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.full(len(F), ROLES["wall"], np.int32))
    me.attributes.new("sk_stone", "INT", "FACE").data.foreach_set("value", np.zeros(len(F), np.int32))
    for m in mats: me.materials.append(m)
    me.polygons.foreach_set("material_index", np.asarray(MI, np.int32)); me.update()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    ob["sk_masonry"] = json.dumps(dict(kind="plate", tag=tag, pack=pack.get("name"), rows=len(runs or [])), ensure_ascii=False)
    return ob, dict(segments=nseg, texel_mm=[round(min(texels) * 1000, 1), round(max(texels) * 1000, 1)] if texels else [], quads=len(F))


def tone(tones, mat, key="stone"):
    """The substance tone (linear RGB) from the plan's "substances:" section; if absent — the stone's."""
    v = (tones.get(mat) or {}).get(key) or (tones.get("stone") or {}).get(key) or (0.52, 0.47, 0.40)
    return np.asarray(v, float)


def stone_items(S, L, pack, sid0, tones, pick=None):
    """A side layout → stones_batch items by pack: stones (the face is the pack's finish, the tone is the substance × the stone's shade), surrounds (finish: flat,
    slightly forward, through ones — to the back), putlog plugs (sunk by 4 cm). The stone face is at −bed (the mortar is the plane d = 0: the building mass or the step's body).
    pick(what, k, q) → False means not a block (a stone goes into a row, a surround and a plug go only into the slab map); what — stone | surround | plug.
    The block number is sid0 + the index in the layout (sid0 ≥ 1: 0 means "not a stone"). → (items, the skipped stones in order — for rows, the count)."""
    bed = pack.get("bed", 0.025); live = float(pack.get("live", 1.0)); out, skipped, n_st = [], [], 0
    for k, st in enumerate(L["stones"]):
        q = np.asarray(st["poly"], float); uc = float(q[:, 0].mean())
        if any(a <= uc <= b for a, b in L.get("gaps", [])): skipped.append(None); continue
        n_st += 1
        if pick is not None and not pick("stone", k, q): skipped.append((st, q)); continue
        skipped.append(None)
        if not st.get("pre_inset"): q = inset(q, pack["joint"] / 2)
        col = tuple(np.clip(tone(tones, st.get("mat", pack.get("mat", "stone"))) * st["tint"], 0, 1))
        fin = pack.get("finish", "flat") if st["kind"] != "spall" else "rough"
        out.append(dict(poly=q, d0=-bed, d1=st["d"][1] if st["d"][1] > 0 else max(S.t.max(), 0.45), col=col, role=ROLES["wall"],
                        sid=sid0 + k, cf=pack.get("cf", 0.014), finish=fin, pillow=pack.get("pillow", 0.003), boss=pack.get("boss", 0.05), dome=pack.get("dome", 0.04),
                        margin=pack.get("margin", 0.035), rough=pack.get("rough", 0.02) * (0.6 if st["kind"] == "spall" else 1.0), live=live))
    nf = 0
    for j, f in enumerate(L.get("features", [])):
        q = np.asarray(f["poly"], float)
        if pick is not None and not pick("surround", j, q): continue
        thr = f["d"][1] <= 0
        col = tuple(np.clip(tone(tones, f.get("mat", "stone")) * f["tint"] * 1.05, 0, 1))
        out.append(dict(poly=inset(q, pack["joint"] / 2), d0=-bed - 0.004, d1=max(S.t.max(), 0.45) if thr else f["d"][1], col=col,
                        role=ROLES["finish"], sid=sid0 + 500_000 + j, cf=min(pack.get("cf", 0.014), 0.012), cb=0.01 if thr else 0.0,
                        finish="flat", pillow=0.002, live=live)); nf += 1
    for h in L.get("holes", []):                                                 # a putlog-socket plug is sunk by 4 cm
        if not h.get("plug"): continue
        q = np.array([(h["u0"] + 0.01, h["z0"] + 0.01), (h["u1"] - 0.01, h["z0"] + 0.01), (h["u1"] - 0.01, h["z1"] - 0.01), (h["u0"] + 0.01, h["z1"] - 0.01)])
        if pick is not None and not pick("plug", 0, q): continue
        out.append(dict(poly=q, d0=-bed + 0.04, d1=0.3, col=tuple(np.clip(tone(tones, pack.get("mat", "stone")) * 0.8, 0, 1)),
                        role=ROLES["wall"], sid=0, cf=0.02, pillow=0.006, live=live))
    return out, skipped, dict(stones=n_st, features=nf)


def _hole_rings(item, S, e=0.02):
    """The item's openings in the wall body (u, v), wider by e: gates and windows — up to the springers + a semicircle; arrow slits — a rectangle."""
    out = []
    for g in item.get("gates", []):
        uc, r = g["u"], g["w"] / 2 + e; zs = g["z"] + g["h"] - g["w"] / 2
        out.append([(uc - r, g["z"] - e), (uc + r, g["z"] - e)] + [(uc + r * math.cos(a), zs + r * math.sin(a)) for a in np.linspace(0, math.pi, 24)])
    for o in item.get("openings", []):
        u0, u1, z0, z1 = o["u"] - o["w"] / 2 - e, o["u"] + o["w"] / 2 + e, o["z"] - o["h"] / 2 - e, o["z"] + o["h"] / 2 + e
        out.append([(u0, z0), (u1, z0), (u1, z1), (u0, z1)])
    return out


def body(S, item, col, coll, name):
    """The wall body (mortar) for walls without the building mass: the face is the plane d = 0, the back is at the side's thickness; pieces between path kinks
    (a flat face per piece), openings with reveals (the sides of the surround stones protrude into the opening: the body's opening is 2 cm wider). The role is mortar."""
    from mathutils import Vector, geometry
    acc = Acc(); t = float(S.t.max()) or 0.5
    cuts = [0.0] + sorted(f for f in S.folds if 0 < f < S.L) + [S.L]
    rings = _hole_rings(item, S)
    for u0, u1 in zip(cuts[:-1], cuts[1:]):
        us = np.unique(np.concatenate([[u0, u1], S.U[(S.U > u0) & (S.U < u1)]]))
        outer = [(u, float(S.base_at(u))) for u in us] + [(u, float(S.top_at(u))) for u in us[::-1]]
        holes = [r for r in rings if min(p[0] for p in r) > u0 + 0.02 and max(p[0] for p in r) < u1 - 0.02]
        loops = [np.asarray(outer, float)] + [np.asarray(h, float)[::-1] for h in holes]
        tri = geometry.tessellate_polygon([[Vector((x, y, 0.0)) for x, y in lp] for lp in loops])
        pts = np.concatenate(loops)
        for d, flip in ((0.0, False), (t, True)):
            W = S.world(pts[:, 0], pts[:, 1], np.full(len(pts), d))
            for a, b, c in tri:
                q = W[[a, b, c]]
                nrm = np.cross(q[1] - q[0], q[2] - q[0]); out = S.world([pts[a][0]], [pts[a][1]], [d - 0.01])[0] - S.world([pts[a][0]], [pts[a][1]], [d])[0]
                if (nrm @ out < 0) != flip: q = q[::-1]
                acc.add(q, col, ROLES["mortar"], 0)
        for li, lp in enumerate(loops):                                          # sides: openings, end faces, top, bottom (not on folds)
            n = len(lp); A = S.world(lp[:, 0], lp[:, 1], np.zeros(n)); B = S.world(lp[:, 0], lp[:, 1], np.full(n, t))
            for i in range(n):
                j = (i + 1) % n
                if li == 0 and abs(lp[i][0] - lp[j][0]) < 1e-9 and (u0 < lp[i][0] < S.L - 1e-9 and lp[i][0] > 1e-9) and lp[i][0] in (u0, u1) and 0 < lp[i][0] < S.L:
                    continue
                acc.add([A[i], A[j], B[j], B[i]], col, ROLES["mortar"], 0)
    ob = acc.obj(name, coll); ob["sk_masonry"] = json.dumps(dict(kind="body", tag=item["tag"]), ensure_ascii=False)
    return ob


def build_side(S, L, pack, item, pxf, coll, stats, maps_dir, texels, every, Tlo, Thi, rows_on, plate_on, tones, Tfar=0.0):
    """A side by the kind's plan: everything as blocks (every) or by camera — a block if the row's px ≥ Thi, in the band Tlo…Thi — by the stone's grain (a ragged
    boundary), the rest — rows over a slab with a map; far away (row px < Tfar) — only the slab with a map: a row of a pixel or two gives no bed shadow, the map
    carries it. → {blocks, slab, body: object}."""
    tag, sid0 = item["tag"], max(1, int(item.get("sid0", 1))); objs = {}
    def pick(what, k, q):
        uc, vc = q.mean(0); px = pxf(S.world([uc], [vc], [0.0])[0], float(np.ptp(q[:, 1])))
        if what != "stone": return px >= Tlo                                    # surrounds and plugs: as blocks while the row is visible
        x = min(1.0, max(0.0, (px - Tlo) / (Thi - Tlo))); p = x * x * (3 - 2 * x)
        return ((k * 2654435761) % 1000) / 1000.0 < p                            # the band Tlo…Thi — by the stone's grain
    items, skipped, cnt = stone_items(S, L, pack, sid0, tones, pick=None if every else pick)
    stats["stones"] += cnt["stones"]; stats["features"] += cnt["features"]; stats["blocks"] += sum(1 for it in items if it["role"] == ROLES["wall"] and it["sid"])
    runs, cur, cur_key, prev_b = [], [], None, 0.0
    far = 0
    for sk in skipped:                                                           # non-blocks go into rows: consecutive along u, one segment
        if sk is None:
            if cur: runs.append(cur); cur = []
            continue
        st, q = sk; uc = float(q[:, 0].mean()); key = (st["course"], int(uc // 10.0))
        if Tfar > 0 and pxf is not None and pxf(S.world([uc], [float(q[:, 1].mean())], [0.0])[0], float(np.ptp(q[:, 1]))) < Tfar:
            far += 1
            if cur: runs.append(cur); cur = []
            continue                                                             # far away: the stone only on the slab map
        if cur and (key != cur_key or q[:, 0].min() - prev_b > 0.05): runs.append(cur); cur = []
        cur.append(q); cur_key = key; prev_b = float(q[:, 0].max())
    if cur: runs.append(cur)
    stats["far"] = stats.get("far", 0) + far
    acc = Acc(); stones_batch(acc, S, items)
    if acc.f:
        ob = acc.obj(f"masonry_{tag}", coll); stats["tris"] += sum(len(p.vertices) - 2 for p in ob.data.polygons)
        ob["sk_masonry"] = json.dumps(dict(kind="blocks", tag=tag, pack=pack.get("name"), part=item.get("part")), ensure_ascii=False)
        objs["blocks"] = ob
    if plate_on and not every:                                                   # a slab with a map — under the blocks and in the distance
        pob, pinfo = plate(S, L, pack, coll, f"plate_{tag}", maps_dir, tag, texels, runs=runs if rows_on else None)
        pinfo["rows"] = len(runs) if rows_on else 0; pinfo["tag"] = tag
        stats.setdefault("plates", []).append(pinfo); stats["tris"] += sum(len(p.vertices) - 2 for p in pob.data.polygons)
        objs["plate"] = pob
    return objs


def run(items, plan, coll, cams=(), fpx=1000.0, work=None, tones=None, log=print):
    """The masonry step: items — surfaces [{tag, strip: the side passport, pack | part, seed, sid0, gates, openings, gaps,
    putlogs, body: bool}], plan — masonry.py resolve (masonry: kind, thresholds, rows, slab; packs; substances), cams — the camera points of the
    frames (the "camera" kind: px per row from the nearest; without cameras — everything as blocks), fpx — the focal length in pixels (frame width ·
    lens / 36), work — the folder of layouts and maps, tones — the building's own substances (the palette) over the plan.
    → (count, {tag: {blocks, slab, body: object}})."""
    k = plan["masonry"]; packs = plan["packs"]; Tlo, Thi = k["thresholds"]; Tfar = float(k.get("far", 0.0))
    every = k["detail"] == "all" or not cams
    pxf = cam_px(cams, fpx) if cams else None
    tn = dict(plan.get("substances") or {}); tn.update(tones or {})
    work = work or os.path.join(bpy.app.tempdir or "/tmp", "masonry_work"); os.makedirs(work, exist_ok=True)
    t0 = time.time(); prep, req = [], []
    for it in items:
        it = dict(it); it.setdefault("pack", k["parts"].get(it.get("part", "outer"), "ashlar")); pk = packs[it["pack"]]
        S = M.strip_from(it["strip"])
        tx = seg_texels(S, pk, pxf) if (not every and k.get("plate", True)) else []
        prep.append((it, S, pk, tx))
        req.append({key: val for key, val in it.items() if key not in ("body",)} | dict(texels=tx, seg=10.0))
    rp = os.path.join(work, "masonry_req.json")
    json.dump(dict(out=work, items=req, packs=packs, far=Tfar), open(rp, "w"), ensure_ascii=False)
    r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-m", "components.constructions.masonry.masonry", "build", rp], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("masonry.py build: " + r.stderr[-800:])
    t_layout = time.time() - t0
    stats = dict(stones=0, blocks=0, features=0, tris=0, items=len(items)); objs = {}
    for it, S, pk, tx in prep:
        L = json.load(open(os.path.join(work, it["tag"] + "_layout.json")))
        objs[it["tag"]] = build_side(S, L, pk, it, pxf, coll, stats, work, tx, every, Tlo, Thi, k.get("rows", True), k.get("plate", True), tn, Tfar)
        if it.get("body"):
            dry = pk.get("dry"); mt = "dry" if dry else pk.get("mat", "stone")
            objs[it["tag"]]["body"] = body(S, it, tuple(tone(tn, mt, "mortar")), coll, f"body_{it['tag']}")
    stats.update(layout_s=round(t_layout, 1), build_s=round(time.time() - t0, 1), detail="all" if every else f"cam {Tlo:g}–{Thi:g} px, far < {Tfar:g}")
    log("masonry:", json.dumps({k_: v for k_, v in stats.items() if k_ != "plates"}, ensure_ascii=False))
    return stats, objs
