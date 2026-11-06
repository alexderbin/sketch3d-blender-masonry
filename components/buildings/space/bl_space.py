"""The space scheme in Blender (the scheme is space/space.py): the faces of the build get the cell they look at (sk_space —
the scheme cell number, 0 — the world; sk_side — the class: 0 outside, 1 inside, 2 yard, 3 covered outside, 4 void, 5
underground), and the model is checked against the scheme by rays: a floor under each walkable cell, a vault over the
covered ones and sky over the open ones, the scheme's doors and arches are open in the model, the scheme's walls are
blank at human height (the scheme's windows are the exception).

A face looks to where the points slightly in front of it and slightly behind it (±6 cm along the normal) lie: both in one
cell — that cell; one in a void (an attic, a tower shaft) — the other (the visible side of a vault and a roof); one under
the sky, the other covered — the sky (the facade of a thin wall)."""
import math

import bpy
import numpy as np
from mathutils import Vector

from components.buildings.space.space import CLASSES, ENCLOSED, OPEN_CLS, PASS_KINDS, inside, inner_point, seg_dist

SIDE_RGB = {"exterior": (0.42, 0.58, 0.82), "interior": (0.93, 0.62, 0.32), "court": (0.45, 0.78, 0.40), "covered": (0.55, 0.85, 0.92),
            "void": (0.35, 0.35, 0.35), "underground": (0.60, 0.45, 0.75)}


def tag(S, objs, eps=0.06):
    """sk_space and sk_side on the objects' faces (world); the color layer "Side" — for the "inside/outside" view. → shares of
    faces by class."""
    code = {k: i for i, k in enumerate(CLASSES)}
    cls_ = np.array([code[c["cls"]] for c in S.cells]); void = np.array([c["cls"] == "void" for c in S.cells]); opn = np.array([c["cls"] in OPEN_CLS for c in S.cells])
    rgb = np.array([SIDE_RGB[c["cls"]] for c in S.cells], np.float32); cnt = np.zeros(len(CLASSES), int)
    for ob in objs:
        me = ob.data; n = len(me.polygons)
        if not n: continue
        C = np.empty(n * 3); N = np.empty(n * 3); me.polygons.foreach_get("center", C); me.polygons.foreach_get("normal", N)
        M = np.array(ob.matrix_world); C = C.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]; N = N.reshape(-1, 3) @ M[:3, :3].T
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        a = S.locate(C + N * eps); b = S.locate(C - N * eps); pick = a.copy(); d = a != b
        m = d & void[a] & ~void[b]; pick[m] = b[m]                               # the visible side: not a void
        m = d & ~void[a] & ~void[b] & ~opn[a] & opn[b]; pick[m] = b[m]           # a thin wall: the facade matters more than the back
        for nm_, val in (("sk_space", pick), ("sk_side", cls_[pick])):
            at = me.attributes.get(nm_) or me.attributes.new(nm_, "INT", "FACE"); at.data.foreach_set("value", val.astype(np.int32))
        ca = me.color_attributes.get("Side") or me.color_attributes.new("Side", "BYTE_COLOR", "CORNER")
        lt = np.empty(n, np.int32); me.polygons.foreach_get("loop_total", lt)
        col = np.repeat(rgb[pick], lt, axis=0); ca.data.foreach_set("color", np.column_stack([col, np.ones(len(col), np.float32)]).ravel())
        cnt += np.bincount(cls_[pick], minlength=len(CLASSES))
    tot = max(int(cnt.sum()), 1)
    return {CLASSES[i]: round(cnt[i] / tot, 3) for i in range(len(CLASSES)) if cnt[i]}


def show(layer):
    """The active color layer for a view (Col — the palette, Side — inside/outside)."""
    for ob in bpy.data.objects:
        if ob.type == "MESH" and ob.data.color_attributes.get(layer): ob.data.color_attributes.active_color = ob.data.color_attributes[layer]


def probe_points(S, i, k=3, spread=0.25):
    """Up to k points of a cell where it is its own (not covered by a cell above, not under a flight or a ramp leading to it),
    away from the edges."""
    c = S.cells[i]; P = S.poly(i); lo, hi = P.min(0), P.max(0)
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 24)[1:-1], np.linspace(lo[1], hi[1], 24)[1:-1]); X, Y = X.ravel(), Y.ravel()
    m = S.inside_cell(i, X, Y)
    for t in S.trans:                                                            # the footprint of a flight on a cell is not its floor
        if t["kind"] in ("stair", "ramp") and not t.get("hidden") and isinstance(t["p"][0], list) and c["id"] in (t["a"], t["b"]):
            Q = np.asarray(t["p"], float)
            for a_, b_ in zip(Q[:-1], Q[1:]):
                e = b_ - a_; L2 = max(float(e @ e), 1e-9); tt = np.clip(((X - a_[0]) * e[0] + (Y - a_[1]) * e[1]) / L2, 0, 1)
                m &= np.hypot(X - a_[0] - e[0] * tt, Y - a_[1] - e[1] * tt) > t["w"] / 2 + 1.0
    X, Y = X[m], Y[m]
    if not len(X): return []
    own = S.locate(np.column_stack([X, Y, np.full(len(X), c["z0"] + 1.0)])) == i; X, Y = X[own], Y[own]
    if not len(X): return []
    d = np.full(len(X), np.inf)
    for e in range(len(P)):
        a, b = P[e], P[(e + 1) % len(P)]; v = b - a; L2 = max(float(v @ v), 1e-12)
        t = np.clip(((X - a[0]) * v[0] + (Y - a[1]) * v[1]) / L2, 0, 1); d = np.minimum(d, np.hypot(X - a[0] - v[0] * t, Y - a[1] - v[1] * t))
    out = []
    for j in np.argsort(-d):
        q = np.array([X[j], Y[j]])
        if all(np.linalg.norm(q - o) > spread * max(hi - lo) for o in out): out.append(q)
        if len(out) >= k: break
    return out


def verify(S, wall_step=2.0):
    """The model against the scheme by rays. → a summary and the lists of failures."""
    deps = bpy.context.evaluated_depsgraph_get(); sc = bpy.context.scene
    last = {}
    def ray(o, d, dist):
        hit, loc, nor, idx, ob, _ = sc.ray_cast(deps, Vector(tuple(map(float, o))), Vector(tuple(map(float, d))), distance=float(dist))
        role = ""
        if hit and ob.type == "MESH" and ob.data.attributes.get("sk_role") is not None and idx < len(ob.data.polygons):
            role = {0: "wall", 1: "finish", 2: "roof", 3: "wood", 4: "metal", 5: "glass", 8: "floor", 9: "opening", 10: "water"}.get(ob.data.attributes["sk_role"].data[idx].value, "")
        last.update(ob=(ob.name + (":" + role if role else "")) if hit else "", d=(loc - Vector(tuple(map(float, o)))).length if hit else 0.0,
                    z=loc.z if hit else 0.0); return hit, loc
    fl, cov, sky, nof, noc, nos, blk = 0, 0, 0, [], [], [], {}
    for i, c in enumerate(S.cells):
        if c["poly"] is None or c["cls"] == "void" or not c["walk"] or c.get("solid"): continue   # a solid body (a castle house) — without a floor and a vault
        pts = probe_points(S, i, k=5); pts_s = probe_points(S, i, k=16, spread=0.08)   # the floor — by 5 points; sky and vault — by 16 (a share)
        if not pts: continue
        okf = 0; okc = 0
        why_f = ""
        for q in pts:
            hit, loc = ray((q[0], q[1], c["z0"] + 1.0), (0, 0, -1), 1.6); okf += bool(hit and abs(loc.z - c["z0"]) <= 0.35)
            if not (hit and abs(loc.z - c["z0"]) <= 0.35): why_f = f"({q[0]:.1f}, {q[1]:.1f}): " + (f"{last['ob']} at {loc.z - c['z0']:+.1f} m" if hit else "empty")
        for q in pts_s:
            if c["cls"] in ENCLOSED:
                hit, loc = ray((q[0], q[1], c["z0"] + 1.0), (0, 0, 1), 2.5 * (c["z1"] - c["z0"]) + 2.0 if c["z1"] is not None else 300.0); okc += bool(hit)   # a vault or a roof
            else:
                hit, loc = ray((q[0], q[1], c["z0"] + 1.0), (0, 0, 1), 400.0); okc += not hit
                if hit: blk.setdefault(c["id"], f"{loc.z - c['z0']:.1f} m above the floor")
        if okf * 2 >= len(pts): fl += 1
        else: nof.append(f"{c['id']} {why_f}")
        need = 0.6 if c["cls"] in ENCLOSED else 0.3                               # a vault — above 60 % of the points; sky — at least above 30 % (awnings, towers, bridges, ramps over a terrace)
        if c["cls"] in ENCLOSED:
            if okc >= need * len(pts_s): cov += 1
            else: noc.append(c["id"])
        else:
            if okc >= need * len(pts_s): sky += 1
            else: nos.append(c["id"])
    # the scheme's transitions are open in the model
    op, shut, patch_n = 0, [], 0
    for t in S.trans:
        if t.get("model") == "patch": patch_n += 1; continue                       # a decoy door (a dark face): not ray-tested, counted
        if t["kind"] not in PASS_KINDS or t.get("hidden") or isinstance(t["p"][0], list): continue
        n = t.get("n")
        if n is None:
            ca, cb = S.cells[S.idx[t["a"]]], S.cells[S.idx[t["b"]]]
            pa = inner_point(ca["poly"]) if ca["poly"] is not None else None; pb = inner_point(cb["poly"]) if cb["poly"] is not None else None
            if pa is None or pb is None: continue
            n = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
        n = np.asarray(n, float); p = np.asarray(t["p"], float); z = max(t["za"], t["zb"]) + min(1.0, 0.45 * (t["h"] or 2.2))
        if t["kind"] == "opening": hl = max(0.6, min(2.0, t["w"] / 2 - 0.6)); hit, _ = ray((*(p - n * hl), z), (*n, 0.0), hl + min(hl, 0.8))   # without a wall: up to the opening line and slightly beyond it
        else: hit, _ = ray((*(p - n * 2.0), z), (*n, 0.0), 4.0)
        if hit: shut.append(f"{t['kind']} {t['a']}→{t['b']} ({last['ob']} at {last['d']:.1f} m, {last['z'] - z:+.1f} m)")
        else: op += 1
    def on_stair(loc, z, me):
        """The point loc lies on another flight that starts or ends at elevation z (the flight continues)."""
        for t in S.trans:
            if t is me or t["kind"] not in ("stair", "ramp") or not isinstance(t["p"][0], list) or min(abs(t["za"] - z), abs(t["zb"] - z)) > 0.4: continue
            Q = np.asarray(t["p"], float)
            if any(seg_dist((loc.x, loc.y), a_, b_) <= t["w"] / 2 + 0.6 for a_, b_ in zip(Q[:-1], Q[1:])): return True
        return False
    # flights: the exit at the top and the entry at the bottom are clear
    # (a flight does not run into a wall or a parapet)
    st_ok, st_bad = 0, []
    for t in S.trans:
        if t["kind"] not in ("stair", "ramp") or t.get("hidden") or not isinstance(t["p"][0], list): continue
        P = [np.asarray(q, float) for q in t["p"]]; up = t["zb"] >= t["za"]; top, pre = (P[-1], P[-2]) if up else (P[0], P[1])
        bot, nxt = (P[0], P[1]) if up else (P[-1], P[-2]); zt, zl = max(t["za"], t["zb"]), min(t["za"], t["zb"])
        d1 = (top - pre) / max(np.linalg.norm(top - pre), 1e-9); d0 = (nxt - bot) / max(np.linalg.norm(nxt - bot), 1e-9)
        h1, l1 = ray((*(top - d1 * 0.8), zt + 1.1), (*d1, 0.0), 2.0); h1 = h1 and not on_stair(l1, zt, t)   # further on is the next flight: not a dead end
        h0, l0 = ray((*(bot - d0 * 1.5), zl + 1.1), (*d0, 0.0), 1.4); h0 = h0 and not on_stair(l0, zl, t)
        if h1 or h0: st_bad.append(f"{t['kind']} {t['a']}→{t['b']}" + (" (top)" if h1 else "") + (" (bottom)" if h0 else ""))
        else: st_ok += 1
    # the scheme's walls are blank at human height (except transitions and windows)
    holes = []; solid = 0; tot = 0
    near = [(np.asarray(t["p"], float), t["w"] / 2 + 0.6, {t["a"], t["b"]}) for t in S.trans if t["kind"] in PASS_KINDS and not isinstance(t["p"][0], list)]   # a passage: along the wall — half the width, across it — up to 5 m (a door in a tower face in front of a wall)
    near += [(np.asarray(w["p"], float), w["w"] / 2 + 0.6, {w["a"], w["b"]}) for w in S.wins if w["z0"] < 1.6 + 0.1]
    near += [(np.asarray(q, float), t["w"] / 2 + 0.8, {t["a"], t["b"]}) for t in S.trans if t["kind"] in ("stair", "ramp") and not t.get("hidden")
             and isinstance(t["p"][0], list) for q in (t["p"][0], t["p"][-1])]  # a flight enters through a wall (a gopura, a gate)
    for b in S.bounds:
        c = S.cells[S.idx[b["a"]]]
        if b["kind"] != "wall" or c["cls"] not in ENCLOSED or not c["walk"] or c.get("solid"): continue
        p0, p1 = np.asarray(b["p0"], float), np.asarray(b["p1"], float); L = float(np.linalg.norm(p1 - p0))
        if L < 0.6: continue
        d = (p1 - p0) / L; out = np.array([d[1], -d[0]]); k = max(1, int(L / wall_step))
        for s in (np.arange(k) + 0.5) / k * L:
            q = p0 + d * s
            if any(abs(float((q - pp) @ d)) <= r_ and abs(float((q - pp) @ out)) <= 5.0 and b["a"] in pr for pp, r_, pr in near): continue
            tot += 1; hit, _ = ray((*(q - out * 1.0), c["z0"] + 1.2), (*out, 0.0), 6.0)   # a wall up to 8 m thick (a hive, a forge)
            if hit: solid += 1
            else: holes.append(f"{b['a']}→{b['b']} ({q[0]:.1f}, {q[1]:.1f})")
    return {"floor_under_walkable": f"{fl}/{fl + len(nof)}", "no_floor": nof[:8], "vault_over_covered": f"{cov}/{cov + len(noc)}", "no_vault": noc[:8],
            "sky_over_open": f"{sky}/{sky + len(nos)}", "closed_sky": [f"{c} ({blk.get(c, '')})" for c in nos[:8]], "transitions_open": f"{op}/{op + len(shut)}", "closed": shut[:8],
            "flights_free": f"{st_ok}/{st_ok + len(st_bad)}", "flights_blocked": st_bad[:8], "doors_fake": patch_n,
            "hidden_n_stairs": sum(1 for t in S.trans if t.get("hidden")),
            "walls_blind": f"{solid}/{tot}", "holes": holes[:10], "ok": not (nof or noc or shut or holes or st_bad),
            "warnings": [f"sky is covered: {x}" for x in nos[:6]]}            # a terrace under a ramp, an arcade or an awning is not a failure


def cutaway(objs, z_cut):
    """A cut of the model at height z_cut (for the "inside/outside" view): copies of objects without the faces above the cut,
    with the vertices above it moved onto the cut; the originals are hidden from rendering. → (copies, a restore function)."""
    import bmesh
    made = []
    for ob in objs:
        bm = bmesh.new(); bm.from_mesh(ob.data); bm.transform(ob.matrix_world)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if min(v.co.z for v in f.verts) > z_cut], context="FACES")
        for v in bm.verts:
            if v.co.z > z_cut: v.co.z = z_cut
        me = bpy.data.meshes.new(ob.name + "_cut"); bm.to_mesh(me); bm.free()
        cp = bpy.data.objects.new(ob.name + "_cut", me); bpy.context.scene.collection.objects.link(cp); made.append(cp); ob.hide_render = True
    def restore():
        for ob in objs: ob.hide_render = False
        for cp in made: cp.hide_render = True
    return made, restore
