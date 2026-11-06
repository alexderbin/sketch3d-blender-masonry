"""The "roofing" step in Blender — a part of the Constructions component.

Roofs are taken by the object contract, without building code: faces with the role "roof" (sk_role = 2) or roof colors
from the building palette (castles write vertex color without roles) → roofing.analyse: tent cones and flat slopes with
ridges, hips, finials → layout of the tiles of the kind (roofing.lay) → geometry:
  flat tile — a thin slab; the butt is raised by the thickness of the layers under it (the row shadow is real),
              "liveliness" by number;
  trough tile — a cross profile (S of the Dutch tile, the trough of the nun and channel tiles, the roll of the monk and
              capping tiles) with an end edge;
  ridge and hip — links as a half-circle (triangle) with a lap from the lower end, for trough tiles on a mortar bed
              (covers the ends of the troughs); finial — a cone; the tent top above the last ring — a cap (lead; for
              metal, the same metal); valley — an open trough: tiles trimmed 6 cm from the line, a strip of metal
              (lead) under them.
Camera-dependent view (as in masonry): px per row ≥ Thi — tiles, ≤ Tlo — the row as one strip (with a butt step,
average tone), in between — chosen by the tile seed: some tiles are full, the others are simple rectangles with a butt
step (there is no slab under the tile, as in masonry: a gap would reveal the roof mass). Output contract: objects
roof_<n> in the step's collection; on faces sk_role (roof | wood | metal by kind), sk_stone — the tile number (roof n:
6·10⁷ + n·10⁵ + number; ridges — + 90 000), tone — the Col color; slots by role sk_<role>.

  import bl_roofing as BT; stats = BT.run(sc, plan, coll, objs=…, roof_rgb=…, cams=…, fpx=…)
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.roofing import roofing as R  # noqa: E402
from components.constructions.masonry.bl_stones import ROLES, Acc  # noqa: E402  (accumulator of faces with a role, number and tone; slots by role)

ROOF_BASE = 60_000_000
PARTS = {"towers": "towers", "buildings": "buildings", "walls": "gates"}         # mass object name → building part


# ── roofs from the scene ────────────────────────────────────────────────────────────────────────────────────────
def roof_faces(objs, roof_rgb=None, tol=0.06):
    """Roof faces of objects (world): role sk_role = roof, otherwise color Col ≈ roof_rgb (the building palette). → [(vertices,
    part)], part is the object name (towers → towers, buildings → buildings); a shared roofs object is "all" (the part is
    given by support_part)."""
    out = []
    for ob in objs:
        if ob.type != "MESH": continue
        me = ob.data; nP = len(me.polygons)
        if not nP: continue
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        Mw = np.asarray(ob.matrix_world, float); co = co @ Mw[:3, :3].T + Mw[:3, 3]
        ls = np.empty(nP, np.int32); me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(nP, np.int32); me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
        sel = np.zeros(nP, bool)
        ra = me.attributes.get("sk_role")
        if ra is not None and ra.domain == "FACE":
            r = np.empty(nP, np.int32); ra.data.foreach_get("value", r); sel |= r == ROLES["roof"]
        ca = me.color_attributes.get("Col")
        if roof_rgb is not None and ca is not None and ca.domain == "CORNER":
            c = np.empty(len(me.loops) * 4, np.float32); ca.data.foreach_get("color", c); c = c.reshape(-1, 4)[:, :3]
            fc = np.add.reduceat(c, ls) / lt[:, None]
            sel |= np.abs(fc - np.asarray(roof_rgb, float)[None]).max(1) < tol
        part = PARTS.get(ob.name.split(".")[0], "all")
        for i in np.flatnonzero(sel):
            out.append((co[lv[ls[i]:ls[i] + lt[i]]], part))
    return out


def roof_points(rf):
    """Reference points of a roof (world): the slope vertices; for a cone — 12 eaves points along the arc and the apex."""
    def cone(S):
        th = np.linspace(S.th0, S.th1, 12, endpoint=not S.full)
        return np.vstack([S.C + S.R * np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1), S.A[None]])
    return np.vstack([np.asarray([S.world(x, v, 0.0) for x, v in S.poly]) if S.kind == "slope" else cone(S) for S in rf["surfaces"]])


def support_part(sc, rf, objs):
    """Part of a roof from the shared roofs object (the castle puts all roofs into "roofs"): a ray down from the roof volume
    (the middle in height: below the eaves there is already mass, while the castle prisms have no bottom — the ray would go
    through it) to the first building mass with a known part (towers → towers, buildings → buildings, walls → gates);
    masonry, the roof itself and the rest are passed through. If the ray went into the void (a tower is a pipe without a
    floor) — the mass nearest to the eaves (the lower quartile of the distances of eaves points ≤ 1.5 m: a turret on a wall
    touches it with one side)."""
    P = roof_points(rf); c = P.mean(0); o = Vector((float(c[0]), float(c[1]), float(c[2])))
    dg = bpy.context.evaluated_depsgraph_get(); keep = {ob.name for ob in objs}
    for _ in range(8):
        hit, loc, _n, _i, ob, _m = sc.ray_cast(dg, o, Vector((0.0, 0.0, -1.0)), distance=300.0)
        if not hit: break                                                        # void — go to the nearest mass
        part = PARTS.get(ob.name.split(".")[0]) if ob.name in keep else None
        if part: return part
        o = loc - Vector((0.0, 0.0, 0.01))
    eave = P[P[:, 2] <= P[:, 2].min() + 0.3][:16]; best = (1.5, None)
    for ob in objs:
        part = PARTS.get(ob.name.split(".")[0])
        if not part or ob.type != "MESH": continue
        Mi = ob.matrix_world.inverted(); Mw = ob.matrix_world; d = []
        for q in eave:
            ok, loc, _n, _i = ob.closest_point_on_mesh(Mi @ Vector(tuple(float(x) for x in q)))
            if ok: d.append(((Mw @ loc) - Vector(tuple(float(x) for x in q))).length)
        if d and float(np.percentile(d, 25)) < best[0]: best = (float(np.percentile(d, 25)), part)
    return best[1]


def find_roofs(sc, objs, roof_rgb=None):
    """Roofs of the building in the scene: faces by contract → roofing.analyse; the part of the roof within a shared object is
    determined by the mass under it."""
    roofs = R.analyse(roof_faces(objs, roof_rgb))
    for rf in roofs:
        if rf["part"] == "all" and rf["surfaces"]: rf["part"] = support_part(sc, rf, objs) or "all"
    return roofs


class RoofAcc(Acc):
    """Masonry accumulator + its own corner normals: the top of a tile, a sheet, a cap on a tent and a dome — the surface
    normal at each vertex (a curved tile is shaded smoothly, without fan facets); side walls and slopes — the face normal
    (flat)."""

    def __init__(self):
        super().__init__(); self.nm = []

    def add(self, pts, col, role, sid, nrm=None):
        super().add(pts, col, role, sid); self.nm.append(None if nrm is None else np.asarray(nrm, float))

    def obj(self, name, coll):
        ob = super().obj(name, coll)
        if any(x is not None for x in self.nm):
            out = []
            for P, N in zip(self.v, self.nm):
                if N is None:
                    n = np.sum(np.cross(P, np.roll(P, -1, 0)), axis=0); n = n / max(np.linalg.norm(n), 1e-12); out.append(np.repeat(n[None], len(P), 0))
                else:
                    out.append(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
            me = ob.data; me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
            me.normals_split_custom_set(np.concatenate(out).tolist())
        return ob


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────────
def _lift(v, v0, L, lam, t):
    return lam * t * np.clip(1.0 - (np.asarray(v, float) - v0) / max(L, 1e-6), 0.0, 1.0)


def _jitter(tile, live):
    """Shift (±3 mm), rotation (±0.8°) and lift (±1 mm) of a tile — from its number (the same in any run)."""
    rng = np.random.default_rng([int(tile["sid"]) & 0x7FFFFFFF, 2039])
    return (math.radians(rng.uniform(-0.8, 0.8)) * live, rng.uniform(-0.003, 0.003, 2) * live, rng.uniform(-0.001, 0.001) * live)


def _dense(P, S, r0, step_deg=6.0, step_v=0.06):
    """Tile outline on a cone and a dome: edges are subdivided every step_deg of arc (on a dome — also every step_v along the
    generatrix) — the tile follows the curvature (otherwise a large slab on a narrow ring is a flat chord through which the
    roof mass shows). On a slope — as is."""
    if S.kind == "slope" or r0 is None: return P
    lim = math.radians(step_deg) * r0; out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]; n = max(1, int(math.ceil(abs(b[0] - a[0]) / lim)))
        if S.kind == "revolve": n = max(n, int(math.ceil(abs(b[1] - a[1]) / step_v)))
        out += [a + (b - a) * (j / n) for j in range(n)]
    return np.asarray(out, float)


def _top(acc, S, P, h, r0, col, role, sid):
    """Tile top: a polygon; when subdivided on a cone and a dome — a fan from the middle (a curved surface, not a flat n-gon),
    corner normals are the surface normal (smooth)."""
    top = S.world(P[:, 0], P[:, 1], h, r0)
    if S.kind != "slope":
        N = S.nrm(P[:, 0], P[:, 1], r0)
        if len(P) > 6:
            c = P.mean(0); hc = float(np.mean(h)) if np.ndim(h) else float(h); ctr = S.world(c[0], c[1], hc, r0); nc = S.nrm(c[0], c[1], r0)
            for i in range(len(P)): j = (i + 1) % len(P); acc.add([ctr, top[i], top[j]], col, role, sid, nrm=[nc, N[i], N[j]])
        else:
            acc.add(top, col, role, sid, nrm=N)
    else:
        acc.add(top, col, role, sid)
    return top


def flat_tile(acc, S, tile, col, role, live):
    """Flat tile: the top is the outline polygon at the lift height + thickness, the sides go down to the bottom (the bottom is
    not needed from outside)."""
    P = np.asarray(tile["poly"], float)
    if len(P) < 3: return
    if R.area(P) < 0: P = P[::-1]
    P = _dense(P, S, tile["r0"])
    a, (dx, dv), dh = _jitter(tile, live)
    c = P.mean(0); ca, sa = math.cos(a), math.sin(a)
    Q = np.stack([c[0] + (P[:, 0] - c[0]) * ca - (P[:, 1] - c[1]) * sa + dx, c[1] + (P[:, 0] - c[0]) * sa + (P[:, 1] - c[1]) * ca + dv], 1)
    lam = tile["layers"]; t = tile["t"]
    hb = _lift(P[:, 1], tile["v0"], tile["L"], lam, t) + dh + (0.004 if S.kind == "revolve" else 0.002) + float(tile.get("h0", 0.0))
    sid = tile["sid"]; top = _top(acc, S, Q, hb + t, tile["r0"], col, role, sid); bot = S.world(Q[:, 0], Q[:, 1], hb, tile["r0"])
    n = len(P); vis = tile.get("g"); vtop = tile["v0"] + (vis + 0.01 if vis else 1e9)
    for i in range(n):                                                           # side walls only on the exposed part (above that it is under the row above)
        j = (i + 1) % n
        if min(P[i, 1], P[j, 1]) < vtop: acc.add([bot[i], bot[j], top[j], top[i]], col, role, sid)


def _vgrid(S, va, vb, step=0.08):
    """Mesh rows along the generatrix: every step and at the kinks of the dome profile (a chord across a kink would go into the
    mass)."""
    vs = np.linspace(va, vb, max(1, int(math.ceil((vb - va) / step))) + 1)
    if S.kind == "revolve": vs = np.unique(np.concatenate([vs, S.s[(S.s > va + 1e-4) & (S.s < vb - 1e-4)]]))
    return vs


def sheet_item(acc, S, tile, col, role):
    """Sheet, board, seam, joint on a tent and a dome: a mesh along the arc and the generatrix (a long strip as a fan would be a
    chord through the mass; rows are also placed at profile kinks), the lift includes an allowance for the sag of the chord
    along the arc; thick ones (seam, board) get side walls."""
    P = np.asarray(tile["poly"], float); xa, xb = float(P[:, 0].min()), float(P[:, 0].max()); va, vb = float(P[:, 1].min()), float(P[:, 1].max())
    r0 = tile["r0"]; nx = max(1, int(math.ceil((xb - xa) / (math.radians(6.0) * r0)))); vs = _vgrid(S, va, vb); nv = len(vs) - 1
    X, Vv = np.meshgrid(np.linspace(xa, xb, nx + 1), vs)
    rmax = S.R if S.kind == "cone" else S.Rmax; sag = rmax * (1.0 - math.cos(0.5 * (xb - xa) / nx / r0))
    hb = float(tile.get("h0", 0.0)) + 0.003 + sag; T = S.world(X, Vv, hb + float(tile["t"]), r0); sid = tile["sid"]; N = S.nrm(X, Vv, r0)
    for j in range(nv):
        for i in range(nx): acc.add([T[j, i], T[j, i + 1], T[j + 1, i + 1], T[j + 1, i]], col, role, sid, nrm=[N[j, i], N[j, i + 1], N[j + 1, i + 1], N[j + 1, i]])
    if tile["t"] > 0.003:
        B = S.world(X, Vv, hb, r0)
        for j in range(nv):
            for i in (0, nx): acc.add([B[j, i], B[j + 1, i], T[j + 1, i], T[j, i]], col, role, sid)
        for j in (0, nv):
            for i in range(nx): acc.add([B[j, i], B[j, i + 1], T[j, i + 1], T[j, i]], col, role, sid)


def ring_cap(acc, S, v0, h, col, role, n=32):
    """Dome cap: from v0 to the top along the profile (rings every 8 cm), at height h above the surface; an edge at the bottom."""
    if v0 >= S.V - 0.02: return
    th = np.linspace(S.th0, S.th1, n + 1); vs = _vgrid(S, v0, S.V)
    X, Vv = np.meshgrid(th, vs); G = S.world(X, Vv, h, 1.0); N = S.nrm(X, Vv, 1.0); lip = S.world(th, np.full(n + 1, v0), np.full(n + 1, h * 0.4), 1.0)
    for j in range(len(vs) - 1):
        for i in range(n): acc.add([G[j, i], G[j, i + 1], G[j + 1, i + 1], G[j + 1, i]], col, role, 0, nrm=[N[j, i], N[j, i + 1], N[j + 1, i + 1], N[j + 1, i]])
    for i in range(n): acc.add([lip[i], lip[i + 1], G[0, i + 1], G[0, i]], col, role, 0)


def simple_tile(acc, S, tile, col, role):
    """A tile of the Tlo…Thi band that was not picked: a rectangle of the tile width (without the shaped bottom, profile and
    liveliness) with the butt lift and an end face — there is no hole down to the roof mass, and far fewer triangles.
    Capping tiles (monk) — above the troughs."""
    Pp = np.asarray(tile["poly"], float); v0 = tile["v0"]; L = tile["L"]; lam = tile["layers"]; t = tile["t"]
    xa, xb = float(Pp[:, 0].min()), float(Pp[:, 0].max())
    P = [(xa, v0), (xb, v0), (xb, v0 + L), (xa, v0 + L)]
    if S.kind != "cone" and tile.get("cut"): P = R.clip_convex(P, S.poly)
    if len(P) < 3: return
    P = _dense(np.asarray(P, float), S, tile["r0"])
    up = float(tile["depth"]) if tile["prof"] in ("monk", "capping") else 0.0
    hb = _lift(P[:, 1], v0, L, lam, t) + (0.004 if S.kind == "revolve" else 0.002) + up
    top = _top(acc, S, P, hb + t, tile["r0"], col, role, tile["sid"]); bot = S.world(P[:, 0], P[:, 1], hb, tile["r0"])
    for i in range(len(P)):                                                      # butt end — the row shadow (edges along the bottom)
        j = (i + 1) % len(P)
        if abs(P[i, 1] - v0) < 1e-6 and abs(P[j, 1] - v0) < 1e-6:
            acc.add([bot[i], bot[j], top[j], top[i]], col, role, tile["sid"])


def _profile(prof, d, x):
    """Height of the outer surface across the tile (x ∈ 0…1) above the lift: s — trough and roll, nun and channel — trough,
    monk and capping — a semicircular roll (the height depends on the width, computed in profiled_tile)."""
    if prof == "s":
        return np.where(x < 0.62, d * (1.0 - np.sin(np.pi * np.clip(x / 0.62, 0, 1))), d + 0.6 * d * np.sin(np.pi * np.clip((x - 0.62) / 0.38, 0, 1)))
    if prof in ("nun", "channel"):
        return d * (1.0 - np.sin(np.pi * x))
    return np.sin(np.pi * x)                                                     # roll — height multiplier


def profiled_tile(acc, S, tile, col, role, live, m=11):
    """Trough tile: a mesh across (m points) × along (bottom, top; per column — inside the slope outline), height — profile +
    butt lift; the end face is a strip of the thickness along the bottom."""
    P = np.asarray(tile["poly"], float)
    if len(P) < 3: return
    x0, x1 = float(P[:, 0].min()), float(P[:, 0].max()); xs = np.linspace(x0, x1, m)
    lo = np.full(m, tile["v0"]); hi = np.full(m, tile["v0"] + tile["L"])
    if S.kind != "cone" and tile.get("cut", True):                               # at a hip the tile is trimmed: per column — inside the outline
        for i, x in enumerate(xs):
            q = R.clip_convex(P, [(x - 1e-4, -1e6), (x + 1e-4, -1e6), (x + 1e-4, 1e6), (x - 1e-4, 1e6)])
            if len(q) < 3: lo[i] = hi[i] = np.nan; continue
            q = np.asarray(q); lo[i], hi[i] = q[:, 1].min(), q[:, 1].max()
    ok = np.isfinite(lo) & (hi - lo > 0.01)
    if ok.sum() < 2: return
    xs, lo, hi = xs[ok], lo[ok], hi[ok]
    xn = (xs - x0) / max(x1 - x0, 1e-6); d = float(tile["depth"]); prof = tile["prof"]
    if prof in ("monk", "capping"):
        hgt = d + 0.5 * (x1 - x0) * 0.9 * _profile(prof, d, xn)                  # the roll lies on the edges of the troughs
    else:
        hgt = _profile(prof, d, xn)
    a, (dx, dv), dh = _jitter(tile, live)
    lam = tile["layers"]; t = tile["t"]
    def surf(v, off):
        return S.world(xs + dx, v + dv, _lift(v, tile["v0"], tile["L"], lam, t) + hgt + dh + off + 0.002, tile["r0"])
    B, T, Bi = surf(lo, t), surf(hi, t), surf(lo, 0.0)
    sid = tile["sid"]
    for i in range(len(xs) - 1):
        acc.add([B[i], B[i + 1], T[i + 1], T[i]], col, role, sid)
        acc.add([Bi[i], Bi[i + 1], B[i + 1], B[i]], col, role, sid)            # butt end — thickness


def cap(acc, c, col, role, m=7):
    """Ridge (hip) link: a half-circle (triangle) of radius r along a → b above the edge line, an end face at the lower end."""
    a = np.asarray(c["a"], float); b = np.asarray(c["b"], float); up = np.asarray(c["up"], float)
    d = b - a; L = np.linalg.norm(d)
    if L < 1e-4: return
    d /= L; e1 = up - (up @ d) * d; e1 /= max(np.linalg.norm(e1), 1e-9); e2 = np.cross(d, e1); r = float(c["r"])
    ang = np.linspace(0, math.pi, m) if c["prof"] == "semicircular" else np.array([0.0, math.pi / 2, math.pi])
    k = 1.0 if c["prof"] == "semicircular" else 0.75
    prof = [e2 * r * math.cos(t_) + e1 * r * k * math.sin(t_) for t_ in ang]
    lift = e1 * 0.012
    A = [a + lift + p for p in prof]; Bq = [b + lift + p for p in prof]
    for i in range(len(prof) - 1):
        acc.add([A[i], A[i + 1], Bq[i + 1], Bq[i]], col, role, c["sid"])
    acc.add([a + lift] + A[::-1], col, role, c["sid"])                          # end face at the lower end


def ridge_bed(acc, S, a, b, h, w, col, role):
    """Mortar bed under the ridge (hip) of a trough-tile roof on slope S: a strip from the edge down the slope by w at profile
    height h and an end face to the decking — covers the upper ends of the troughs under the links."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; d /= max(np.linalg.norm(d), 1e-12)
    e = np.cross(S.n, d); c = S.world(*S.poly.mean(0), 0.0)
    if (c - a) @ e < 0: e = -e
    H = S.n * h; E = e * w
    acc.add([a + H, b + H, b + E + H, a + E + H], col, role, 0)
    acc.add([a + E, b + E, b + E + H, a + E + H], col, role, 0)


def cone_cap(acc, S, v0, h, col, role, n=32):
    """Tent cap: from v0 (where the next row would lie) to the apex — a smooth cone at height h above the decking, an edge at
    the bottom."""
    if v0 >= S.V - 0.02: return
    th = np.linspace(S.th0, S.th1, n + 1); r0 = max(float(S.r(v0)), 1e-3); xs = th * r0
    lo = S.world(xs, np.full(n + 1, v0), np.full(n + 1, h), r0); lo0 = S.world(xs, np.full(n + 1, v0), np.full(n + 1, h * 0.4), r0)
    top = S.A + np.array([0.0, 0.0, h])                                          # the tip is above the apex (on a spire an offset along the normal would go above the finial)
    for i in range(n):
        acc.add([lo[i], lo[i + 1], top], col, role, 0)
        acc.add([lo0[i], lo0[i + 1], lo[i + 1], lo[i]], col, role, 0)


def valley_strip(acc, S, vl, col, role, w=0.2):
    """Valley trough: a strip in the slope plane from the valley line by w inward, slightly above the decking — under the
    tiles, visible in the trough."""
    a = np.asarray(vl["a"], float); b = np.asarray(vl["b"], float); d = b - a; d /= max(np.linalg.norm(d), 1e-12)
    e = np.cross(S.n, d); c = S.world(*S.poly.mean(0), 0.0)
    if (c - a) @ e < 0: e = -e
    h = S.n * 0.0015
    acc.add([a + h, b + h, b + e * w + h, a + e * w + h], col, role, 0)


def finial(acc, p, up, col, role, sid, r=0.12, h=0.4, n=10):
    p = np.asarray(p, float); up = np.asarray(up, float); up /= max(np.linalg.norm(up), 1e-9)
    e1 = np.cross(up, [1.0, 0.0, 0.0]); e1 = e1 if np.linalg.norm(e1) > 1e-3 else np.cross(up, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    ring = [p - up * 0.05 + (e1 * math.cos(t_) + e2 * math.sin(t_)) * r for t_ in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    top = p + up * h
    for i in range(n): acc.add([ring[i], ring[(i + 1) % n], top], col, role, sid)


def course_strip(acc, S, tiles, col, role, k):
    """A row as one strip (far away): the outline is the union of the row's tiles (for a slope — a strip along the chord
    trimmed by the outline), the butt lift as for a tile; a cone ring is a belt of segments."""
    t0 = tiles[0]; v0 = t0["v0"]; f = v0 / max(S.top, 1e-6); g = R.gauge(k, f); L = g + R.lerp(k["lap"], f); lam = t0["layers"]; t = t0["t"]
    if S.kind in ("cone", "revolve"):
        r0 = t0["r0"]; n = 24; xs = np.linspace(S.th0 * r0, S.th1 * r0, n + 1)
        m = max(1, int(math.ceil(L / 0.08))) if S.kind == "revolve" else 1         # dome: the strip follows the profile
        vv = v0 + L * np.arange(m + 1) / m; hh = lam * t * (1.0 - (vv - v0) / L) + t
        G = [S.world(xs, np.full(n + 1, vv[j]), np.full(n + 1, hh[j]), r0) for j in range(m + 1)]
        NN = [S.nrm(xs, np.full(n + 1, vv[j]), r0) for j in range(m + 1)]
        Bi = S.world(xs, np.full(n + 1, v0), np.full(n + 1, lam * t), r0)
        for i in range(n):
            for j in range(m): acc.add([G[j][i], G[j][i + 1], G[j + 1][i + 1], G[j + 1][i]], col, role, 0, nrm=[NN[j][i], NN[j][i + 1], NN[j + 1][i + 1], NN[j + 1][i]])
            acc.add([Bi[i], Bi[i + 1], G[0][i + 1], G[0][i]], col, role, 0)
        return
    xs = [p[0] for tl in tiles for p in tl["poly"]]
    q = R.clip_convex([(min(xs), v0), (max(xs), v0), (max(xs), v0 + L), (min(xs), v0 + L)], S.poly)
    if len(q) < 3: return
    tile = dict(poly=q, v0=v0, L=L, t=t, layers=lam, r0=None, sid=0)
    flat_tile(acc, S, tile, col, role, 0.0)


# ── far view: a shell with a row pattern in the shader ──────────────────────────────────────────────────────────
_FAR = {}


def far_material(plan, k):
    """Far-view roof material (per kind): a row pattern in the shader by the unrolling (UV in meters) — Blender's brick
    texture: tile width, row step, shift every other row (staggered; stacked — no shift), tile tones vary randomly; the
    shadow under the butt and the relief — a sawtooth by the step (highest at the butt, lowest under the butt of the next
    row). No images and no row geometry."""
    key = k["name"]
    if key in _FAR: return _FAR[key]
    m = bpy.data.materials.new(f"roof_far_{key}"); m.use_nodes = True; nt = m.node_tree; N, Lk = nt.nodes, nt.links
    bs = next(n for n in N if n.type == "BSDF_PRINCIPLED"); bs.inputs["Roughness"].default_value = 0.55 if k["role"] == "metal" else 0.85
    uv = N.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"
    br = N.new("ShaderNodeTexBrick"); g = R.gauge(k); w = R.width(k) + float(k["gap"]); spread = float(k["spread"])
    br.offset = {"staggered": 0.5, "stacked": 0.0}.get(k["bond"], 0.37); br.offset_frequency = 2; br.squash = 1.0; br.squash_frequency = 2
    base = np.asarray(tone(plan, k, 1.0), float)
    for name_, val in (("Scale", 1.0), ("Mortar Size", max(0.002, float(k["gap"]))), ("Mortar Smooth", 0.3), ("Bias", 0.0),
                       ("Brick Width", w), ("Row Height", g)):
        br.inputs[name_].default_value = val
    br.inputs["Color1"].default_value = (*np.clip(base * (1 - spread), 0, 1), 1.0); br.inputs["Color2"].default_value = (*np.clip(base * (1 + spread), 0, 1), 1.0)
    br.inputs["Mortar"].default_value = (*(base * 0.35), 1.0)
    Lk.new(uv.outputs["UV"], br.inputs["Vector"])
    sep = N.new("ShaderNodeSeparateXYZ"); Lk.new(uv.outputs["UV"], sep.inputs["Vector"])
    dv = N.new("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[1].default_value = g; Lk.new(sep.outputs["Y"], dv.inputs[0])
    fr = N.new("ShaderNodeMath"); fr.operation = "FRACT"; Lk.new(dv.outputs[0], fr.inputs[0])
    hgt = N.new("ShaderNodeMath"); hgt.operation = "SUBTRACT"; hgt.inputs[0].default_value = 1.0; Lk.new(fr.outputs[0], hgt.inputs[1])
    ramp = N.new("ShaderNodeValToRGB"); el = ramp.color_ramp.elements
    el[0].position = 0.0; el[0].color = (0.38, 0.38, 0.38, 1.0); el[1].position = 0.3; el[1].color = (1.0, 1.0, 1.0, 1.0)
    Lk.new(hgt.outputs[0], ramp.inputs["Fac"])
    mul = N.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
    Lk.new(br.outputs["Color"], mul.inputs[0]); Lk.new(ramp.outputs["Color"], mul.inputs[1]); Lk.new(mul.outputs["Vector"], bs.inputs["Base Color"])
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.7
    bump.inputs["Distance"].default_value = max(0.005, int(k["n_layers"]) * R.lerp(k["t"], 0.5))
    Lk.new(hgt.outputs[0], bump.inputs["Height"]); Lk.new(bump.outputs["Normal"], bs.inputs["Normal"])
    _FAR[key] = m
    return m


def far_shell(rf, coll, name, mat, h=0.003, n=32):
    """Far-view roof shell: a slope — the outline (polygon), a tent and a dome — a mesh of rings along the profile, at h above
    the mass; UV — (u, v) in meters of the unrolling (for rings u is the arc at the ring radius). It is not visible under
    the tiles and strips; far away it is all there is."""
    V, F, UV = [], [], []
    for S in rf["surfaces"]:
        if S.kind == "slope":
            P = np.asarray(S.poly, float); i0 = len(V)
            V += S.world(P[:, 0], P[:, 1], h).tolist(); F.append(list(range(i0, i0 + len(P)))); UV += P.tolist()
            continue
        vs = _vgrid(S, 0.0, S.V, step=0.5) if S.kind == "revolve" else np.linspace(0.0, S.V, max(2, int(math.ceil(S.V / 0.5))) + 1)
        rmax = S.R if S.kind == "cone" else S.Rmax; span = S.th1 - S.th0
        n = max(n, int(math.ceil(span * rmax / 0.3)))                            # face arc ≤ 0.3 m
        hr = h + rmax * (1.0 - math.cos(0.5 * span / n))                          # the lift includes an allowance for the sag of the chord: the mass edges do not show through
        th = np.linspace(S.th0, S.th1, n + 1); i0 = len(V)
        for v in vs:
            V += S.world(th, np.full(n + 1, v), np.full(n + 1, hr), 1.0).tolist(); UV += [(float(t_ * S.r(v)), float(v)) for t_ in th]
        for j in range(len(vs) - 1):
            for i in range(n):
                a = i0 + j * (n + 1) + i; F.append([a, a + 1, a + n + 2, a + n + 1])
    if not F: return None
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update()
    loops = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", loops)
    uvl = me.uv_layers.new(name="UVMap"); uvl.data.foreach_set("uv", np.asarray(UV, np.float32)[loops].ravel())
    me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.full(len(F), ROLES["roof"], np.int32))
    me.attributes.new("sk_stone", "INT", "FACE").data.foreach_set("value", np.zeros(len(F), np.int32))
    me.materials.append(mat)
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))                 # tent and dome are smooth (a slope is flat anyway)
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    return ob


# ── step ────────────────────────────────────────────────────────────────────────────────────────────────────────
def roof_far(rf, locs, D, margin=0.5):
    """The whole roof farther than D from every camera — far view, decided before the layout. The middle of any row (the step
    judges the row's px by it) lies within the surface frame: a slope — the rectangle of the unrolling (x along the outline,
    v from the eaves to the top), a tent and a dome — a cylinder of the largest radius from bottom to top; the frame of all
    surfaces + margin is farther than D from every camera → the px of any row < the far threshold: the result is a shell
    with a pattern, no tile layout needed (the check has a margin: a row that would go to the far view is sometimes laid out
    anyway)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for S in rf["surfaces"]:
        if S.kind == "slope":
            x0, x1 = float(S.poly[:, 0].min()), float(S.poly[:, 0].max())
            P = S.world(np.array([x0, x1, x0, x1]), np.array([0.0, 0.0, S.top, S.top]), 0.0)
        else:
            rm = S.R if S.kind == "cone" else S.Rmax
            zs = (S.C[2], S.A[2]) if S.kind == "cone" else (S.P[:, 1].min(), S.P[:, 1].max())
            P = np.array([[S.C[0] - rm, S.C[1] - rm, min(zs)], [S.C[0] + rm, S.C[1] + rm, max(zs)]], float)
        lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    lo, hi = lo - margin, hi + margin
    return bool(rf["surfaces"]) and all(float(np.linalg.norm(np.maximum(np.maximum(lo - L_, L_ - hi), 0.0))) > D for L_ in locs)


def tone(plan, k, tint):
    v = (plan.get("substances") or {}).get(k["substance"]) or [0.5, 0.22, 0.12]
    return tuple(float(x) for x in np.clip(np.asarray(v, float) * tint, 0, 1))


def run(sc, plan, coll, objs=None, roof_rgb=None, cams=(), fpx=1000.0, log=print):
    """The "roofing" step: roofs of the objects objs (none — all scene meshes outside the collection coll) → tiles by the kind
    of the part (plan: roofing.resolve). cams — camera points of the frames (the "camera" detail mode; without cameras — all
    tiles), fpx — px per frame focus. → counts: roofs, slopes, cones, tiles, rows as a strip, rows far away, slopes far away
    (the whole roof is beyond the far threshold — the slopes are not laid out, roof_far), ridge and hip links, triangles,
    seconds."""
    t0 = time.time(); c = plan["roofing"]; kinds = plan["kinds"]; parts = c["parts"]
    every0 = c["detail"] == "all" or not cams; Tlo, Thi = c["thresholds"]; Tfar = float(c.get("far", 0.0))
    locs = [np.asarray(p, float) for p in cams]
    if objs is None: objs = [o for o in sc.objects if o.type == "MESH" and o.name not in coll.objects]
    roofs = find_roofs(sc, objs, roof_rgb)
    st = dict(n_roofs=len(roofs), n_pitches=0, n_cones=0, n_cupolas=0, n_spires=0, n_tiles=0, n_simple=0, n_rows=0, afar=0, n_pitches_afar=0, n_links=0, valleys=0,
              tris=0, skipped=0)
    by_part = {}
    for ri, rf in enumerate(roofs):
        if rf["kind"] == "skip": st["skipped"] += 1; continue               # onion, cupola — mass only for now
        k = R.kind_for(parts, kinds, rf)
        key = f'{rf["part"]}:{k["name"]}'; by_part[key] = by_part.get(key, 0) + 1
        sheets = k["generator"] == "sheets"; every = every0 or sheets                # sheets and boards — in full (there are few of them and they are large)
        metal = k["role"] == "metal"; capcol = tone(plan, k if metal else dict(substance="lead"), 0.95)
        role = ROLES.get(k["role"], ROLES["roof"]); live = float(k["liveliness"]); base = ROOF_BASE + ri * 100_000
        acc = RoofAcc(); sid = base + 1
        R.open_valleys(rf)                                                       # valleys: the slope outlines are set back — a trough
        far = not every and Tfar > 0 and roof_far(rf, locs, fpx * R.gauge(k) / Tfar)   # the whole roof is in the far view: do not lay out the slopes
        tiny = False
        for S in rf["surfaces"]:
            st["n_pitches" if S.kind == "slope" else ("n_cones" if S.kind == "cone" else "n_cupolas")] += 1
            if S.kind == "cone" and S.R < max(0.5, 1.2 * R.width(k)):            # a decorative spire of the building — the cap as a whole
                cone_cap(acc, S, 0.0, 0.006, capcol, ROLES["metal"]); tiny = True; st["n_spires"] += 1; continue
            if S.kind == "revolve" and S.Rmax < 0.5:                               # a decorative dome top — the cap as a whole
                ring_cap(acc, S, 0.0, 0.006, capcol, ROLES["metal"]); tiny = True; st["n_spires"] += 1; continue
            if far and S.kind == "slope":                                        # every row would go to the far view — the same result without tiles;
                st["n_pitches_afar"] += 1; continue                                # tent and dome are laid out: the cap height follows from the rows
            tiles = R.lay(S, k, seed=int(c["seed"]) * 7919 + ri * 31 + st["n_pitches"] + st["n_cones"], sid0=sid)
            sid += len(tiles) + 1
            if S.kind != "slope" and tiles:                                      # the top of a tent and a dome above the last ring — a cap
                top_v = (max(t_["v0"] + t_["L"] for t_ in tiles) if sheets else max(t_["v0"] for t_ in tiles if not t_["under"]) + R.gauge(k, 1.0))
                hc = (int(k["n_layers"]) + 1) * R.lerp(k["t"], 1.0) + 0.004 if not sheets else max(R.lerp(k["t"], 1.0) * 3, 0.03)
                (cone_cap if S.kind == "cone" else ring_cap)(acc, S, top_v, hc, capcol, ROLES["metal"])
            by = {}
            for tl in tiles: by.setdefault(tl["course"], []).append(tl)
            for course, tl_ in sorted(by.items()):
                pick = None
                if not every:
                    v0 = tl_[0]["v0"]; xs = [p[0] for tl in tl_ for p in tl["poly"]]
                    mid = S.world(0.5 * (min(xs) + max(xs)), v0, 0.0, tl_[0]["r0"])
                    px = fpx * R.gauge(k) / max(min(float(np.linalg.norm(mid - L_)) for L_ in locs), 0.1)
                    if px < Tfar:                                                # far view: a row is only the shell pattern
                        st["afar"] += 1; continue
                    if px <= Tlo:                                                # far away: a row as a strip; the underlay row below the first is not visible
                        if not tl_[0]["under"]:
                            course_strip(acc, S, tl_, tone(plan, k, float(np.mean([t_["tint"] for t_ in tl_]))), role, k); st["n_rows"] += 1
                        continue
                    x = min(1.0, max(0.0, (px - Tlo) / (Thi - Tlo))); pick = x * x * (3 - 2 * x)
                    if px >= Thi: pick = None
                for tl in tl_:
                    col = tone(plan, k, tl["tint"])
                    if pick is not None and ((tl["sid"] * 2654435761) % 1000) / 1000.0 >= pick:
                        simple_tile(acc, S, tl, col, role); st["n_simple"] += 1; continue   # the Tlo…Thi band: some tiles are simpler
                    if tl["prof"] in ("sheet", "plank", "seam", "joint"):
                        if S.kind == "slope": flat_tile(acc, S, tl, col, role, live if tl["prof"] == "plank" else 0.0)
                        else: sheet_item(acc, S, tl, col, role)
                    elif tl["prof"] == "flat": flat_tile(acc, S, tl, col, role, live)
                    else: profiled_tile(acc, S, tl, col, role, live)
                    st["n_tiles"] += 1
        caps_ = []
        hb = R.relief(k)
        for a, b, up in rf["ridges"] + rf["hips"]:
            caps_ += R.caps(a, b, up, k, sid0=base + 90_000 + len(caps_))
            if hb > 0:                                                           # trough tiles: links on a mortar bed
                for S in rf["surfaces"]:
                    if S.kind == "slope" and max(abs(float((np.asarray(p_, float) - S.O) @ S.n)) for p_ in (a, b)) < 0.03:
                        ridge_bed(acc, S, a, b, hb, caps_[-1]["r"] + 0.03, tone(plan, dict(substance="mortar"), 1.0), ROLES["mortar"])
        for cp in caps_: cap(acc, cp, tone(plan, k, cp["tint"] * 0.92), role)
        st["n_links"] += len(caps_)
        lead = tone(plan, dict(substance="lead"), 1.0)
        for vl in rf.get("valleys", []):
            for si in vl["s"]: valley_strip(acc, rf["surfaces"][si], vl, lead, ROLES["metal"])
        st["valleys"] += len(rf.get("valleys", []))
        if rf["apex"] is not None and not tiny and rf["kind"] != "revolve":       # a dome's finial is the building's cross, not ours
            finial(acc, rf["apex"], (0.0, 0.0, 1.0), tone(plan, k, 0.85), role, base + 99_000)
        if Tfar > 0 and not every and rf["surfaces"] and not tiny:              # far view: a shell with a pattern under everything
            fo = far_shell(rf, coll, f"roof_far_{ri}", far_material(plan, k))
            if fo is not None:
                st["tris"] += sum(len(p.vertices) - 2 for p in fo.data.polygons)
                fo["sk_roofing"] = json.dumps(dict(kind=k["name"], substance=k["substance"], part=rf["part"], shape=rf["kind"], far=True), ensure_ascii=False)
        if acc.f:
            ob = acc.obj(f"roof_{ri}", coll); st["tris"] += sum(len(p.vertices) - 2 for p in ob.data.polygons)
            ob["sk_roofing"] = json.dumps(dict(kind=k["name"], substance=k["substance"], part=rf["part"], shape=rf["kind"], numbers=[base, base + 99_999]),
                                          ensure_ascii=False)
    st["by_parts"] = ", ".join(f"{k_} {n}" for k_, n in sorted(by_part.items()))      # part:kind number of roofs
    st["seconds"] = round(time.time() - t0, 1)
    log("roof:", json.dumps(st, ensure_ascii=False))
    return st
