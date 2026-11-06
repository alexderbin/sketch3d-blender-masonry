"""Settlement as a mass in Blender: from the plan ensemble.json — pavements over the adjusted terrain (triangulation of the
plan paving.npz), stairs on steep streets (15 cm rise), a city wall following the ground with merlons, towers and gates
(an arch in the body of the wall), a sanctuary (temenos with steps and an altar), hive tiers (slabs with shafts,
railings, bridges over the shafts, supports, a shell with openings), houses — the "house" type via the entry
(buildings.module), the host castle — via its builder through the entry (buildings.build_in).
build_ensemble(folder, sc=…) — step by step and in a foreign scene; CLI — views for checking:
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/ensembles/bl_ensemble.py -- <plan folder> [views=aerial,street,plan,district_trade,power_0] [res=1600x1000] [blend=1]"""
import os
import sys
from components.buildings import arch_parts
import json
import math
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils import geometry as MG

from components.buildings.base import bl_arch as BA
from components.buildings import buildings as B
from components.buildings.influences import field as FD

T0 = time.time()


def C(rgb, role): return BA.rc(tuple(rgb), role)


def _dense(P, step, closed=True):
    """Polyline with points spaced no more than step apart."""
    P = [np.asarray(p, float) for p in P]; out = []
    n = len(P) if closed else len(P) - 1
    for i in range(n):
        a, b = P[i], P[(i + 1) % len(P)]; k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * j / k for j in range(k)]
    if not closed: out.append(P[-1])
    return out


def tag_field(ob, folder, R):
    """Faces of an object → influence field at their centers (the plan's influences.npz): sk_P, sk_T, sk_W (FLOAT;
    hive — tier by the face's height)."""
    f = os.path.join(folder, (R.get("influence") or {}).get("field") or FD.FILE)
    if ob is None or not os.path.exists(f) or not len(ob.data.polygons): return
    F = FD.load(f); me = ob.data; n = len(me.polygons); C_ = np.empty(n * 3, np.float32); me.polygons.foreach_get("center", C_); C_ = C_.reshape(-1, 3)
    P, T, W = FD.sample(F, C_[:, :2], z=C_[:, 2] if F["P"].shape[0] > 1 else None)
    for nm, v in (("sk_P", P), ("sk_T", T), ("sk_W", W)): me.attributes.new(nm, "FLOAT", "FACE").data.foreach_set("value", v.astype(np.float32))


def paving(folder, R, rel, z_off=0.06):
    """Pavement: the plan's triangles, vertices on the adjusted terrain (streets level across, squares — as a plane)."""
    f = os.path.join(folder, R.get("paving_file") or "paving.npz")
    if not os.path.exists(f): return None
    Z = np.load(f); V = Z["V"].astype(float); T = Z["T"]
    if not len(T): return None
    zz = rel.height(V[:, 0], V[:, 1]) + z_off
    a = BA.Acc("floor"); a.tri([(x, y, z) for (x, y), z in zip(V, zz)], [tuple(map(int, t)) for t in T], C(R.get("paving") or (0.5, 0.47, 0.42), "floor"))
    return a.obj("pavement", BA.coll("streets"))


def water_bridges(R, rel, col=(0.6, 0.57, 0.52)):
    """Street bridges over water: a deck at bank level, parapets along, supports at the channel edges."""
    a = BA.Acc("finish"); cb = C(col, "finish"); n = 0
    for P in R.get("water_bridges") or []:
        P = np.asarray(P, float)
        if len(P) < 3: continue
        zs = rel.height(P[:, 0], P[:, 1]); zd = float(np.max(zs)) + 0.25; zb = float(np.min(zs))
        a.prism([tuple(p) for p in P], zd - 1.1, zd, cb)
        E_ = np.roll(P, -1, 0) - P; L = np.linalg.norm(E_, axis=1); i = int(np.argmax(L)); u = E_[i] / L[i]
        for k in range(len(P)):                                                 # parapets — along the edges of the bridge
            if L[k] < 2.5 or abs(float(E_[k] @ u)) / L[k] < 0.8: continue
            a.box(((P[k] + P[(k + 1) % len(P)]) / 2).tolist(), (E_[k] / L[k]).tolist(), 0.45, L[k], zd, zd + 1.0, cb)
        c = P.mean(0); w = float(np.max(np.abs((P - c) @ np.array([-u[1], u[0]])))); Lb = float(np.max(np.abs((P - c) @ u)))
        for sgn in (-1, 1):                                                     # supports at the channel edges
            q = c + u * sgn * (Lb - 1.6); a.box(q.tolist(), u.tolist(), 2 * w + 0.6, 2.2, zb - 1.5, zd - 1.0, cb)
        n += 1
    print(f"bridges over water: {n}")
    return a.obj("bridges", BA.coll("streets")) if a.f else None


def stairs(R, rel, rise=0.15):
    """Steps on steep street segments: rise ~15 cm, tread — by the slope (0.3–1.4 m), a stone across the full width."""
    a = BA.Acc("finish"); col = C(tuple(min(1.0, c * 1.12) for c in (R.get("paving") or (0.5, 0.47, 0.42))), "finish"); n = 0
    for e in R["streets"]["edges"]:
        if not e.get("stairs"): continue
        P = np.asarray(e["pts"], float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
        def at(s):
            i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)); t = (s - cum[i]) / max(seg[i], 1e-9)
            return P[i] + (P[i + 1] - P[i]) * t, (P[i + 1] - P[i]) / max(seg[i], 1e-9)
        for s0, s1 in e["stairs"]:
            s = s0
            while s < s1 - 0.2:
                p, d = at(s); z = float(rel.height(p[0], p[1])); q, _ = at(min(s1, s + 0.5)); g = abs(float(rel.height(q[0], q[1])) - z) / 0.5
                tread = float(np.clip(rise / max(g, 1e-3), 0.32, 1.4)); s2 = min(s1, s + tread)
                pa, _ = at(s); pb, _ = at(s2); m = (pa + pb) / 2
                za = float(rel.height(pa[0], pa[1])); zb = float(rel.height(pb[0], pb[1])); top = max(za, zb) + 0.1
                a.box(m.tolist(), d.tolist(), e["w"] * 0.98, s2 - s + 0.02, top - 0.55, top, col); n += 1
                s = s2
    print(f"steps: {n}")
    return a.obj("stairs", BA.coll("streets")) if a.f else None


def city_wall(R, rel, col=(0.62, 0.58, 0.52), W=None, gates=None, gap=None, every=42.0, name="wall", towers=True, merlons=True):
    """Wall following the ground: 6 m segments (top — ground + height), the gate is an arch in the body of the wall, merlons,
    towers every `every` m and at the gates; gap — (point, radius): the wall segment near the propylon is skipped (the
    passage is the propylon itself); without towers and merlons — a retaining wall with a smooth parapet (acropolis)."""
    W = W or R["wall"]; t, h = W.get("t", 2.4), W.get("h", 8.5); a = BA.Acc("wall"); cw = C(col, "wall"); ct = C(tuple(c * 0.92 for c in col), "finish")
    ring = _dense(W["line"], 6.0); gates = [np.asarray(g, float) for g in (R.get("gates") or [] if gates is None else gates)]; n = len(ring)
    cen = np.mean(np.asarray(W["line"], float), 0)
    def gz(p): return float(rel.height(p[0], p[1]))
    skip = lambda m: gap is not None and float(np.linalg.norm(m - np.asarray(gap[0], float))) < gap[1]
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]; L = float(np.linalg.norm(q - p))
        if L < 0.3 or skip((p + q) / 2): continue
        d = (q - p) / L; nrm = np.array([d[1], -d[0]])
        if float(nrm @ ((p + q) / 2 - cen)) < 0: nrm = -nrm
        c0 = p + nrm * 0.0; c1 = q + nrm * 0.0
        base = min(gz(p), gz(q)) - 1.2; holes = []
        for g in gates:
            s = float((g - p) @ d)
            if -0.5 < s < L + 0.5 and abs(float((g - p) @ nrm)) < 3.0:
                s = float(np.clip(s, 2.2, L - 2.2)) if L > 4.6 else L / 2
                holes.append((s, gz(p + d * s) + 0.02, 3.6, 4.6, "round"))
        BA.wall_panel(a, c0, c1, base, gz(p) + h, t, [hh for hh in holes if L > 4.6], cw, col_jamb=ct, z1_end=gz(q) + h)
        k = max(1, int(L / 1.7)) if merlons else 0                               # merlons — on the outer half
        for j in range(k):
            s = (j + 0.5) * L / k; m = p + d * s + nrm * (t * 0.25); zt = gz(p) + h + (gz(q) - gz(p)) * s / L
            a.box(m.tolist(), d.tolist(), t * 0.45, min(0.95, L / k * 0.55), zt, zt + 1.05, cw)
    # towers
    cum = 0.0; last = -1e9; spots = []
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        if cum - last > every and not skip(p): spots.append(p); last = cum
        cum += float(np.linalg.norm(q - p))
    for g in gates:
        k = int(np.argmin([np.linalg.norm(r - g) for r in ring])); d = ring[(k + 1) % n] - ring[k - 1]; d /= max(np.linalg.norm(d), 1e-9)
        spots += [g + d * 4.4, g - d * 4.4]
    spots = [s_ for s_ in spots if all(np.linalg.norm(s_ - g) > 3.5 for g in gates)] if towers else []
    for s_ in spots:
        z0 = gz(s_) - 1.2; zt = gz(s_) + h + 4.0; r = 3.2
        k = int(np.argmin([np.linalg.norm(r_ - s_) for r_ in ring])); d = ring[(k + 1) % n] - ring[k - 1]; d /= max(np.linalg.norm(d), 1e-9)
        nrm = np.array([d[1], -d[0]])
        if float(nrm @ (s_ - cen)) < 0: nrm = -nrm
        c = s_ + nrm * 1.2
        sq = [c + d * r - nrm * r, c + d * r + nrm * r, c - d * r + nrm * r, c - d * r - nrm * r]
        a.prism(sq, z0, zt, cw)
        for j in range(4):
            u = sq[(j + 1) % 4] - sq[j]; L = float(np.linalg.norm(u)); u /= L; m_ = 3
            for kk in range(m_):
                m = sq[j] + u * (L * (kk + 0.5) / m_); a.box(m.tolist(), u.tolist(), 0.7, 1.0, zt, zt + 1.1, cw)
    return a.obj(name, BA.coll("walls"))


def sanctuary(R, rel):
    """Temenos on the highest point: a stepped platform and an altar (the temple — as a type, once a classical one exists)."""
    P = np.asarray(R["sanctuary"], float); c = P.mean(0); a = BA.Acc("finish"); col = C((0.86, 0.84, 0.78), "finish")
    zs = rel.height(P[:, 0], P[:, 1]); top = float(np.max(zs)) + 0.9
    for k in range(3):
        Q = c + (P - c) * (0.92 - 0.035 * (2 - k)); a.prism(Q.tolist(), float(np.min(zs)) - 0.5, top - 0.3 * (2 - k), col)
    BA.altar(a, c.tolist(), 2.2, top, col)
    return a.obj("sanctuary", BA.coll("landmarks"))


def hive(R, rel):
    """Hive tiers: slabs (shafts are holes), railings along the shaft edges, bridges, supports at the shafts, a shell with
    openings on each tier."""
    D = R["decks"]; n, Hd, th, z0 = D["count"], D["height"], D["slab"], D["z0"]; foot = [tuple(p) for p in D["outline"]]
    shafts = [[tuple(p) for p in s] for s in R.get("shafts") or []]
    a = BA.Acc("floor"); cs = C((0.36, 0.36, 0.38), "floor"); cu = C((0.24, 0.24, 0.26), "wall"); cm = C((0.30, 0.29, 0.27), "metal")
    for k in range(1, n):
        zt = z0 + k * Hd + th; zb = zt - th
        loops = [foot] + shafts
        tris = MG.tessellate_polygon([[Vector((x, y, 0.0)) for x, y in lp] for lp in loops])
        pts = [(x, y, zt) for lp in loops for x, y in lp]; a.tri(pts, [(i, j, k_) for i, j, k_ in tris], cs)
        pts = [(x, y, zb) for lp in loops for x, y in lp]; a.tri(pts, [(i, k_, j) for i, j, k_ in tris], cu)
        for lp in loops:
            for i in range(len(lp)):
                p, q = lp[i], lp[(i + 1) % len(lp)]; a.face([(p[0], p[1], zb), (q[0], q[1], zb), (q[0], q[1], zt), (p[0], p[1], zt)], cu)
        for sh in shafts:                                                        # railings along the shaft edge
            for p in _dense(sh, 2.0):
                a.box([float(p[0]), float(p[1])], [1.0, 0.0], 0.12, 0.12, zt, zt + 1.15, cm)
            for i in range(len(sh)):
                p, q = np.asarray(sh[i]), np.asarray(sh[(i + 1) % len(sh)]); d = q - p; L = float(np.linalg.norm(d))
                if L > 0.5: a.box(((p + q) / 2).tolist(), (d / L).tolist(), 0.14, L, zt + 1.05, zt + 1.18, cm)
        for br in R.get("bridges") or []:                                        # bridges over the shaft on each tier
            a.prism([tuple(p) for p in br], zt - 1.6, zt, cs)
    for sh in shafts:                                                            # supports at the shaft corners
        P = np.asarray(sh, float); c = P.mean(0)
        for p in P[::max(1, len(P) // 4)][:4]:
            q = p + (c - p) / max(np.linalg.norm(c - p), 1e-9) * 3.0
            a.prism([tuple(q + v) for v in ((-1.6, -1.6), (1.6, -1.6), (1.6, 1.6), (-1.6, 1.6))], z0 - 1, z0 + (n - 1) * Hd + th + 6, cu)
    ztop = z0 + (n - 1) * Hd + th
    for i in range(len(foot)):                                                   # shell: openings per tier, buttresses
        p, q = np.asarray(foot[i], float), np.asarray(foot[(i + 1) % len(foot)], float); L = float(np.linalg.norm(q - p)); d = (q - p) / L
        nrm = np.array([d[1], -d[0]]); c0 = p - nrm * 1.6; c1 = q - nrm * 1.6
        holes = []
        for k in range(n):
            zb_ = z0 + k * Hd + (th if k else 0) + (8.0 if k == 0 else 4.0)
            for s in np.arange(14.0, L - 10.0, 22.0): holes.append((float(s), zb_, 9.0, min(Hd - th - 10.0, 14.0), "segment"))
        BA.wall_panel(a, c0, c1, z0 - 2.0, ztop + 3.5, 3.2, holes, cu)
        for s in np.arange(0.0, L, 22.0):
            m = p + d * (s + 3.0) + nrm * 1.5
            a.box(m.tolist(), nrm.tolist(), 2.6, 4.2, z0 - 2.0, ztop + 1.0 - 6.0 * (s / L), cu)
    return a.obj("tiers", BA.coll("tiers"))


def build_ensemble(folder, sc=None, tag=True, with_ground=False, host=True, **kw):
    """Settlement over an open scene (step by step) or with its own ground (CLI). → (objects, plan)."""
    R = json.load(open(os.path.join(folder, "ensemble.json"))); BA.SCENE[0] = sc
    rel = BA.Relief(os.path.join(folder, "relief.npz")); obs = []; c = np.asarray(R["site"]["centre"], float)
    if with_ground: obs.append(BA.ground(rel, c, R["site"]["R"] + 70, 1.0))
    if R.get("host") and host and R["host"].get("folder"):
        try:
            o, _ = B.build_in(R["host"]["folder"], sc=sc)
            obs += [x for v in (o.values() if isinstance(o, dict) else [o]) for x in (v if isinstance(v, (list, tuple)) else [v]) if hasattr(x, "name")]
        except Exception as e: print("host not built:", repr(e)[:300])
    o = paving(folder, R, rel)
    if o: tag_field(o, folder, R); obs.append(o)
    o = stairs(R, rel)
    if o: obs.append(o)
    o = water_bridges(R, rel)
    if o: obs.append(o)
    if R.get("wall"): obs.append(city_wall(R, rel))
    Ac = R.get("acropolis")
    if Ac:                                                                       # acropolis wall along the platform edge; the passage is the propylon
        obs.append(city_wall(R, rel, col=(0.7, 0.66, 0.58), W={"line": Ac["wall"], "t": 1.6, "h": 1.6}, gates=[], every=34.0,
                             gap=(Ac["gate"], 10.0) if Ac.get("gate") else None, name="wall_acropolis", towers=False, merlons=False))
    if R.get("landmarks"):
        obs += B.module("classics", "builder").build(R["landmarks"], BA.coll("landmarks"))
    elif R.get("sanctuary"): obs.append(sanctuary(R, rel))
    if R.get("decks"): obs.append(hive(R, rel)); tag_field(obs[-1], folder, R)
    HB = B.module("house", "builder")
    progs = R["houses"]
    groups = []
    for dk in sorted({h.get("deck") for h in progs}, key=lambda v: -1 if v is None else v):
        hs = [h for h in progs if h.get("deck") == dk]; groups += [hs[i:i + 60] for i in range(0, len(hs), 60)]
    obs += HB.build_houses(progs, BA.coll("houses"), name="houses", groups=groups)
    for ob in obs:
        if ob is not None and "sk_layer" not in ob: ob["sk_layer"] = "buildings"
    BA.SCENE[0] = None
    return obs, R


# ---------------------------------------------------------------- views for checking
def _seen(loc, tgt, ids):
    """Does the ray from loc to tgt first hit a face of a house from ids (the house number on the face is sk_house)? Supports,
    hulls and slabs do not count."""
    sc = bpy.context.scene; dg = bpy.context.evaluated_depsgraph_get(); o = Vector(loc); d = Vector(tgt) - o; L = d.length
    hit, _, _, idx, ob, _ = sc.ray_cast(dg, o, d.normalized(), distance=L + 30.0)
    if not hit or ob is None or ob.type != "MESH" or "sk_house" not in ob.data.attributes or idx >= len(ob.data.polygons): return False
    return int(ob.data.attributes["sk_house"].data[idx].value) in ids


def _deck_view(R, x, p, s_, z0, top):
    """Compound of the hive's lowest tier: 18 spots (two sides of the street × three distances × three heights from the
    `from_street` point); the farthest one from which the compound house is visible (ray) is taken; otherwise — point-blank
    from the street."""
    q = np.asarray(x["from_street"], float); best = None
    for ed in R["streets"]["edges"]:
        if ed["kind"] in ("edge", "road", "wall"): continue
        P_ = np.asarray(ed["pts"], float)
        for a_, b_ in zip(P_[:-1], P_[1:]):
            ab = b_ - a_; L_ = float(np.linalg.norm(ab))
            if L_ < 1e-6: continue
            t_ = float(np.clip((q - a_) @ ab / L_ ** 2, 0, 1)); dd = float(np.linalg.norm(a_ + ab * t_ - q))
            if best is None or dd < best[0]: best = (dd, ab / L_)
    u_ = best[1] if best else np.array([1.0, 0.0])
    ids = {int(h["id"]) for h in R["houses"] if h.get("compound") is not None and h.get("deck") == x["deck"]
           and np.linalg.norm(np.asarray(h["F"], float).mean(0) - p) < s_ + 10.0}
    tz = z0 + min(12.0, 0.35 * (top - z0)); tgt = (p[0], p[1], tz); pick = None
    for dist in (1.4 * s_ + 16.0, 0.9 * s_ + 12.0, 0.5 * s_ + 8.0):
        for sg in (1.0, -1.0):
            for h_ in (7.0, 3.0, 14.0):
                loc = q + sg * u_ * dist
                if _seen((loc[0], loc[1], z0 + h_), tgt, ids): pick = {"loc": (loc[0], loc[1], z0 + h_), "tgt": tgt, "lens": 22}; break
            if pick: break
        if pick: break
    return pick or {"loc": (q[0], q[1], z0 + 2.2), "tgt": tgt, "lens": 20}


def views(R, rel):
    c = np.asarray(R["site"]["centre"], float); Rr = R["site"]["R"]; zc = float(rel.height(c[0], c[1]))
    if R.get("decks"): zc = R["decks"]["z0"] + R["decks"]["count"] * R["decks"]["height"] * 0.5
    V = {"aerial": {"loc": (c[0] - Rr * 1.05, c[1] - Rr * 1.25, zc + Rr * 0.9), "tgt": (c[0], c[1], zc), "lens": 32},
         "plan": {"loc": (c[0], c[1], zc + 900), "tgt": (c[0], c[1], zc), "ortho": 2 * Rr + 40}}
    an = [x for x in R["spaces"]["anchors"] if "street" in (x.get("tags") or [])]
    if an:
        x = an[0]; p = np.asarray(x["p"], float); f = np.asarray(x.get("facing") or [1, 0], float); f /= max(np.linalg.norm(f), 1e-9)
        q = p - f * 14.0; zq = float(rel.height(q[0], q[1]))
        V["street"] = {"loc": (q[0], q[1], zq + 1.7), "tgt": (p[0] + f[0] * 30, p[1] + f[1] * 30, zq + 3.0), "lens": 22}
    sq = [x for x in R["spaces"]["anchors"] if x["name"].startswith("area")]
    if sq:
        p = np.asarray(sq[0]["p"], float); d = c - p if np.linalg.norm(c - p) > 1 else np.array([1.0, 0.3]); d /= np.linalg.norm(d)
        q = p - d * 18.0; zq = float(rel.height(q[0], q[1]))
        V["square"] = {"loc": (q[0], q[1], zq + 6.0), "tgt": (p[0] + d[0] * 40, p[1] + d[1] * 40, zq + 2.0), "lens": 20}
    lm = {x["tags"][1]: x for x in R["spaces"]["anchors"] if "landmark" in (x.get("tags") or [])}
    if "theatre" in lm:                                                           # theatre: from the skene into the cavea
        x = lm["theatre"]; p = np.asarray(x["p"], float); f = np.asarray(x["facing"], float); f /= np.linalg.norm(f); s_ = x.get("size", 40)
        V["theatre"] = {"loc": (*(p - f * s_ * 0.08), x["z"] + 4.5), "tgt": (*(p + f * s_ * 0.45), x["z"] + s_ * 0.18), "lens": 20}
    tm = lm.get("peripteros") or lm.get("prostyle") or lm.get("podium")
    if tm and R.get("acropolis"):                                               # acropolis: from the agora up to the temple
        sq_ = [x for x in R["spaces"]["anchors"] if x["name"].startswith("area")]
        p = np.asarray(sq_[0]["p"] if sq_ else c, float); q = np.asarray(tm["p"], float); d = q - p; d /= max(np.linalg.norm(d), 1e-9)
        zq = float(rel.height(p[0], p[1]))
        V["acropolis"] = {"loc": (*(p - d * 10.0), zq + 2.2), "tgt": (q[0], q[1], tm["z"] + 6.0), "lens": 26}
    if tm:                                                                       # temple in three-quarter view, from height — over the neighbors
        q = np.asarray(tm["p"], float); f = np.asarray(tm["facing"], float); f /= np.linalg.norm(f); L_, W_ = tm.get("L", 30), tm.get("W", 15)
        e = np.array([-f[1], f[0]]); loc = q + f * (L_ / 2 + 0.9 * L_) + e * (W_ / 2 + 0.6 * L_)
        V["temple"] = {"loc": (loc[0], loc[1], tm["z"] + 0.35 * L_), "tgt": (q[0], q[1], tm["z"] + 0.12 * L_), "lens": 28}
    st_ = [x for x in R["spaces"]["anchors"] if (x.get("tags") or [None, None])[1:2] == ["stoa"]]
    if st_:                                                                      # agora: from the square toward the main stoa (in front of it — at the depth of the square)
        x = max(st_, key=lambda y: y.get("L", 0)); q = np.asarray(x["p"], float); f = np.asarray(x["facing"], float); f /= max(np.linalg.norm(f), 1e-9)
        e = np.array([-f[1], f[0]]); loc = q + f * (x.get("L", 10) / 2 + 24.0) + e * x.get("W", 40) * 0.18
        V["agora"] = {"loc": (loc[0], loc[1], x["z"] + 1.8), "tgt": (q[0], q[1], x["z"] + 4.0), "lens": 22}
    hs = [h for h in R["houses"] if h.get("deck") in (None, 0)]
    if hs:                                                                       # block: houses from above, close up (near the middle of the settlement)
        P = np.array([np.mean(np.asarray(h["F"], float), 0) for h in hs]); k = int(np.argmin(np.linalg.norm(P - (c + np.array([Rr * 0.35, -Rr * 0.3])), axis=1)))
        q = P[k]; zq = hs[k]["z0"]
        V["block"] = {"loc": (q[0] - 30.0, q[1] - 38.0, zq + 32.0), "tgt": (q[0], q[1], zq + 2.0), "lens": 30}
    k_ = 0
    for x in R["spaces"]["anchors"]:                                              # districts (reading of the influences) and the power compounds
        tg = x.get("tags") or []
        if tg[:1] == ["district"]:                                                    # district — obliquely from above, one scale for all of them (to compare houses, yards, gardens)
            p = np.asarray(x["p"], float); f = np.asarray(x["facing"], float); f /= max(np.linalg.norm(f), 1e-9); e = np.array([-f[1], f[0]])
            zq = float(rel.height(p[0], p[1])); loc = p - f * 50.0 + e * 30.0
            V[f"district_{tg[1]}"] = {"loc": (loc[0], loc[1], zq + 52.0), "tgt": (p[0] + f[0] * 8, p[1] + f[1] * 8, zq + 2.0), "lens": 30}
        elif tg[:1] == ["power"]:                                                 # compound from height, against the background of the settlement
            p = np.asarray(x["p"], float); s_ = max(float(x.get("size", 20.0)), 12.0); z0 = float(x["z"]); top = float(x.get("top", z0 + 10.0))
            d = p - c; d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1.0 else np.array([0.6, -0.8])
            e = np.array([-d[1], d[0]]); loc = p + d * (s_ * 1.3 + 48.0) + e * 18.0
            V[f"power_{k_}"] = {"loc": (loc[0], loc[1], z0 + 30.0 + 0.5 * (top - z0)), "tgt": (p[0], p[1], z0 + 0.4 * (top - z0)), "lens": 26}
            D_ = R.get("decks")
            if D_ and x.get("from_street") and x.get("deck") is not None and x["deck"] < D_["count"] - 1:   # lowest hive tier — from the street inside the tier:
                V[f"power_{k_}"] = _deck_view(R, x, p, s_, z0, top)              # the spot from which the ray first hits the compound house
            k_ += 1
    if R.get("decks"):
        D = R["decks"]; sh = R.get("shafts") or []
        if sh:
            P = np.asarray(sh[0], float); m = P.mean(0); zz = D["z0"] + 1 * D["height"] + D["slab"] + 1.7
            e = P[0] - m; e /= np.linalg.norm(e)
            V["shaft"] = {"loc": (*(m + e * 20.0), zz + 30), "tgt": (*(m - e * 20.0), zz - 10), "lens": 18}
        p = np.asarray(R["streets"]["nodes"][0], float)
        mains = [x for x in R["spaces"]["anchors"] if "street" in (x.get("tags") or [])]
        if mains:
            x = mains[0]; p = np.asarray(x["p"], float); f = np.asarray(x["facing"], float); f /= np.linalg.norm(f)
            zz = D["z0"] + 1 * D["height"] + D["slab"] + 1.7
            V["deck"] = {"loc": (*(p - f * 30), zz), "tgt": (*(p + f * 30), zz + 4.0), "lens": 20}
    return V


def render(R, rel, folder, which, res):
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"; shd = sc.display.shading
    shd.light = "STUDIO"; shd.color_type = "VERTEX"; shd.show_cavity = True; shd.cavity_type = "WORLD"
    shd.shadow_intensity = 0.55; shd.show_shadows = True; sc.display.light_direction = (0.45, -0.35, 0.82)
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    V = views(R, rel); out = {}
    for k in which:
        if k not in V: continue
        v = V[k]; cam.location = v["loc"]; d = Vector(v["tgt"]) - Vector(v["loc"]); cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam.data.type = "ORTHO" if "ortho" in v else "PERSP"
        if "ortho" in v: cam.data.ortho_scale = v["ortho"]
        else: cam.data.lens = v["lens"]
        cam.data.clip_start = 0.2; cam.data.clip_end = 5000
        sc.render.filepath = os.path.join(folder, f"view_{k}.png"); bpy.ops.render.render(write_still=True); out[k] = sc.render.filepath
    return out


VIEWS = ("aerial,street,square,plan,shaft,deck,acropolis,theatre,temple,agora,block,"
         "district_trade,district_craft,district_nobility,district_middle,district_poor,power_0,power_1,power_2,power_3")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]; folder = os.path.abspath(argv[0]); KW = dict(x.split("=", 1) for x in argv[1:] if "=" in x)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obs, R = build_ensemble(folder, with_ground=True, host=KW.get("host", "1") == "1")
    t_b = time.time() - T0
    rel = BA.Relief(os.path.join(folder, "relief.npz"))
    res = tuple(int(v) for v in KW.get("res", "1600x1000").split("x"))
    out = render(R, rel, folder, KW.get("views", VIEWS).split(","), res)
    dg = bpy.context.evaluated_depsgraph_get()
    tris = sum(len(o.evaluated_get(dg).to_mesh().loop_triangles) for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground"))
    if KW.get("blend", "0") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "ensemble.blend"))
    print("__RESULT__", json.dumps({"build_s": round(t_b, 1), "total_s": round(time.time() - T0, 1), "n_objects": len(obs), "n_triangles": tris,
                                    "kinds": list(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
