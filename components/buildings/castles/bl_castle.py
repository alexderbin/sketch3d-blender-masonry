"""Castle in Blender from the plan castle.json (castles/castle_plan.py): ground with edits (terraces, moat),
water, walls following the relief (foundation down to the ground, batter, walkway, parapet, merlons by
style), towers (round, D-shaped, square, polygonal; cone, tent, pyramid, spire, spike, platform with
merlons; machicolations), a gate with an arched passage, the main tower, buildings with roofs, a chapel with
an apse, a bridge, a road. Colors are a vertex attribute (the Workbench "by vertices" view). No ornaments.

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/castles/bl_castle.py -- <plan folder> [views=aerial,road,top,back] [res=1280x800] [blend=1]
→ <folder>/view_<view>.png, castle.blend (if blend=1), the __RESULT__ line

As a function over an open scene (the import reads and builds nothing):
    from components.buildings.castles import bl_castle as BC
    objs, C = BC.build_castle(folder, sc=None, with_ground=False, tag=True)     # load + build(fresh=False) + face markup
or in parts: BC.load(folder, kw=None) → BC.build(sc=None, with_ground=True, fresh=True) → (c, half, objs).
The plan globals (C, HN, HF, ST, COL, K, KC, LEVEL, Z, TWO_LEVEL, ARCH_TOP, MERLON …) are filled by load();
after it footprint, win_blocked, height and the rest are available, as before. The paths castle.json,
relief.npz, location.json (if present) are in the plan folder."""
import json, math, os, sys, time

import bmesh
import bpy
import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.base import arch_elems as AE                                                          # noqa: E402

# Plan state — filled by load(folder); on import the module reads and builds nothing.
ARGV = []; OUT = None; KW = {}; C = None; Z = None; LOC = None
HN = HF = LEVEL = None; NX0 = NY0 = NRES = FX0 = FY0 = FRES = None; NB = None
ST = COL = K = None
T0 = time.time()
_SC = [None]                                                                     # the scene in which build runs (coll() puts the collections into it)


def _bil(A, x0, y0, res, x, y):
    gx = np.clip((np.asarray(x) - x0) / res, 0, A.shape[1] - 1.001); gy = np.clip((np.asarray(y) - y0) / res, 0, A.shape[0] - 1.001)
    ix, iy = gx.astype(int), gy.astype(int); fx, fy = gx - ix, gy - iy
    return A[iy, ix] * (1 - fx) * (1 - fy) + A[iy, ix + 1] * fx * (1 - fy) + A[iy + 1, ix] * (1 - fx) * fy + A[iy + 1, ix + 1] * fx * fy


def height(x, y):
    """Site height: the near plot (with the castle's edits), beyond it the belt (like Ground: a 12 m seam)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    zf = _bil(HF, FX0, FY0, FRES, x, y); zn = _bil(HN, NX0, NY0, NRES, x, y)
    d = np.minimum(np.minimum(x - NB[0], NB[2] - x), np.minimum(y - NB[1], NB[3] - y)); w = np.clip(d / 12.0, 0, 1); w = w * w * (3 - 2 * w)
    return np.where(d > 0, zn * w + zf * (1 - w), zf)


class Acc:
    """Mesh accumulator: vertices, faces, face color → one object."""

    def __init__(self): self.v = []; self.f = []; self.c = []

    def face(self, pts, col):
        i0 = len(self.v); self.v += [tuple(map(float, p)) for p in pts]; self.f.append(list(range(i0, i0 + len(pts)))); self.c.append(col)

    def quad_strip(self, A, B, col, closed=False):
        """Faces between two polylines A and B of equal length."""
        n = len(A); m = n if closed else n - 1
        for i in range(m):
            j = (i + 1) % n; self.face([A[i], A[j], B[j], B[i]], col)

    def prism(self, poly, z0, z1, col, top=True, bottom=False, scale_top=1.0, scale_bot=1.0, centre=None):
        P = np.asarray(poly, float); c = P.mean(0) if centre is None else np.asarray(centre)
        lo = [(*(c + (p - c) * scale_bot), z0) for p in P]; hi = [(*(c + (p - c) * scale_top), z1) for p in P]
        self.quad_strip(lo, hi, col, closed=True)
        if top: self.face(hi, col)
        if bottom: self.face(lo[::-1], col)
        return hi

    def box(self, c, d, w, l, z0, z1, col):
        """Bar: center c (x, y), axis d, length l along d, width w across, from z0 to z1."""
        d = np.asarray(d, float); d /= max(np.linalg.norm(d), 1e-9); n = np.array([-d[1], d[0]]); c = np.asarray(c, float)
        P = [c - d * l / 2 - n * w / 2, c + d * l / 2 - n * w / 2, c + d * l / 2 + n * w / 2, c - d * l / 2 + n * w / 2]
        self.prism(P, z0, z1, col)

    def obj(self, name, coll):
        V = np.asarray(self.v, float); keep = []                                 # without degenerate faces (a wall ledge: two stations at one
        for i, f in enumerate(self.f):                                           # point — the section face collapses into a segment)
            P = V[f]; n = np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0)
            if float(n @ n) > 1e-10: keep.append(i)
        if len(keep) < len(self.f):
            self.f = [self.f[i] for i in keep]; self.c = [self.c[i] for i in keep]
        me = bpy.data.meshes.new(name); me.from_pydata(self.v, [], self.f); me.validate(); me.update()
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        cols = np.repeat(np.array(self.c, np.float32), [len(f) for f in self.f], axis=0)
        ca.data.foreach_set("color", np.column_stack([cols, np.ones(len(cols), np.float32)]).ravel())
        me.color_attributes.active_color = ca
        ob = bpy.data.objects.new(name, me); coll.objects.link(ob); return ob


ARK = Acc(); ARB = Acc(); ARC = Acc()                                            # castle rock: plinths, strips under the walls, boulders


def coll(name):
    sc = _SC[0] or bpy.context.scene
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in sc.collection.children: sc.collection.children.link(c)
    return c


def shade(c, k): return tuple(float(min(1, max(0, v * k))) for v in c)


STONE = STONE_D = ROOF = PLASTER = STONE2 = KC = None                             # plan colors — load()
WOOD = (0.42, 0.30, 0.20); ROAD = (0.55, 0.47, 0.36); WATER = (0.22, 0.38, 0.55); HOLE = (0.07, 0.06, 0.06)
IRON = (0.16, 0.16, 0.17)
FEAT = []; SECS = {}; TWO_LEVEL = False; ARCH_TOP = "round"


# ---------------------------------------------------------------- ground and water
def ground(centre, half, step=1.0, hole=None, name="ground"):
    if hole is None:                                                             # nodes follow the relief nodes (terrace ledges without a "sawtooth")
        x0 = NX0 + math.ceil((centre[0] - half - NX0) / step) * step; y0 = NY0 + math.ceil((centre[1] - half - NY0) / step) * step
        xs = np.arange(x0, centre[0] + half + 1e-6, step); ys = np.arange(y0, centre[1] + half + 1e-6, step)
    else:
        xs = np.arange(centre[0] - half, centre[0] + half + 1e-6, step); ys = np.arange(centre[1] - half, centre[1] + half + 1e-6, step)
    X, Y = np.meshgrid(xs, ys); Zg = height(X, Y); ny, nx = X.shape
    if hole is not None:                                                         # far ring under the near ground (on cliffs the coarse
        Zg = np.where((np.abs(X - centre[0]) < hole + 4.5) & (np.abs(Y - centre[1]) < hole + 4.5), Zg - 3.0, Zg)   # mesh stuck out as a strip)
    gy, gx = np.gradient(Zg, step); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    grass = np.array([0.36, 0.42, 0.25]); dirt = np.array([0.45, 0.40, 0.30]); rock = np.array([0.50, 0.48, 0.45])
    t = np.clip((sl - 20) / 20, 0, 1)[..., None]; colv = grass * (1 - t) + rock * t
    flat = (sl < 3)[..., None]; colv = np.where(flat, colv * 0.6 + dirt * 0.4, colv)
    me = bpy.data.meshes.new("ground")
    V = np.column_stack([X.ravel(), Y.ravel(), Zg.ravel()])
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]; F = np.stack([i, i + 1, i + nx + 1, i + nx], -1).reshape(-1, 4)
    if hole is not None:                                                         # far ring: without the middle (the near ground is there)
        cx = (X[:-1, :-1] + X[1:, 1:]) / 2; cy = (Y[:-1, :-1] + Y[1:, 1:]) / 2
        keep = (np.abs(cx - centre[0]) > hole) | (np.abs(cy - centre[1]) > hole); F = F[keep.ravel()]
    me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 4)); me.update()
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    ca.data.foreach_set("color", np.column_stack([colv.reshape(-1, 3), np.ones(nx * ny)]).astype(np.float32).ravel())
    me.color_attributes.active_color = ca
    ob = bpy.data.objects.new(name, me); coll("terrain").objects.link(ob)
    if hole is not None: return ob
    return ob, water_of(X, Y, Zg, xs, ys)


def water_of(X, Y, Zg, xs, ys):
    """Water: the cells where the level is above the ground (the water object in terrain) — or None."""
    L = np.full(X.shape, np.nan)
    gxi = np.clip(np.round((X - NX0) / NRES).astype(int), 0, LEVEL.shape[1] - 1); gyi = np.clip(np.round((Y - NY0) / NRES).astype(int), 0, LEVEL.shape[0] - 1)
    lv = LEVEL[gyi, gxi]; wet = np.isfinite(lv) & (lv > Zg + 0.03)
    if wet.any():
        a = Acc(); ii, jj = np.nonzero(wet[:-1, :-1])
        for i_, j_ in zip(ii, jj):
            z = float(lv[i_, j_]); a.face([(xs[j_], ys[i_], z), (xs[j_ + 1], ys[i_], z), (xs[j_ + 1], ys[i_ + 1], z), (xs[j_], ys[i_ + 1], z)], WATER)
        return a.obj("water", coll("terrain"))
    return None


# ---------------------------------------------------------------- rock: plinths, strips under the walls, boulders (a castle out of rock)
ROCK = (0.47, 0.45, 0.42); KN = {}; NCRAG = [0]


def rnoise(p, s=0.32, seed=0.0):
    """Rock noise −1…1 (large ridges + fine detail), as for the site landmarks (bl_relief._lathe)."""
    from mathutils import Vector, noise
    v = Vector((p[0] * s + seed * 3.1, p[1] * s + seed * 1.7, p[2] * s + seed * 0.9))
    return 0.6 * noise.noise(v) + 0.3 * noise.noise(v * 2.6) + 0.1 * noise.noise(v * 6.1)


def rock_col(p, seed):
    return shade(ROCK, 0.9 + 0.22 * rnoise(p, 0.21, seed + 5))


def resample(ring, n):
    P = np.asarray(ring, float); P2 = np.vstack([P, P[:1]]); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P2, axis=0).T))])
    u = np.linspace(0, s[-1], n, endpoint=False); return np.column_stack([np.interp(u, s, P2[:, 0]), np.interp(u, s, P2[:, 1])])


def rock_mass(a, ring, z_top, skirt, seed, n=34, rings=7):
    """Rock plinth under the masonry: rings from the top (the masonry footprint + the offset skirt[0]) down to the
    ground (offset skirt[1]), noise along the normal — ridges and ledges; the bottom sinks 0.8 m into the
    ground; the top is under the masonry."""
    R = resample(ring, n); c = R.mean(0); u = R - c; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    top = R + u * skirt[0]; bot = R + u * skirt[1]
    zb = np.minimum(height(bot[:, 0], bot[:, 1]) - 0.8, z_top - 0.4)
    loops = []
    for k in range(rings + 1):
        f = k / rings; Pk = top + (bot - top) * f ** 0.8; zk = z_top + (zb - z_top) * f; amp = 0.25 + 0.85 * f
        d = np.array([amp * (0.55 + rnoise((p[0], p[1], z), 0.3, seed)) for p, z in zip(Pk, zk)])
        Pk = Pk + u * d[:, None]; loops.append([(*p, z) for p, z in zip(Pk, zk)])
    for k in range(rings):
        A, B = loops[k], loops[k + 1]
        for i in range(n):
            j = (i + 1) % n; a.face([A[i], A[j], B[j], B[i]], rock_col(A[i], seed))
    a.face(loops[0], shade(ROCK, 0.85))


def rock_band(a, P, N, T, rt, seed, skirt=(0.5, 1.8)):
    """Rock under a wall on the edge: from the bottom of the masonry (rt) down and outward to the ground; noise —
    ledges; the ends are closed."""
    ok = np.array([v is not None for v in rt]); j = 0
    while j < len(P):
        if not ok[j]: j += 1; continue
        k = j
        while k < len(P) and ok[k]: k += 1
        if k - j >= 2:
            prof = []
            for i in range(j, k):
                q, n, t, z1 = P[i], N[i], float(T[i]), float(rt[i]); pr = [(*(q + n * (t / 2 - 0.25)), z1 + 0.25)]
                qb = q + n * (t / 2 + skirt[1] * (0.8 + 0.5 * rnoise((q[0], q[1], 0.0), 0.15, seed)))
                zb = min(float(height(*qb)) - 0.8, z1 - 0.5)
                for f in (0.0, 0.3, 0.62, 1.0):
                    off = t / 2 + skirt[0] + (skirt[1] - skirt[0]) * f ** 0.8; z = z1 + (zb - z1) * f
                    off += (0.2 + 0.6 * f) * (0.5 + rnoise((q[0], q[1], z), 0.3, seed))
                    pr.append((*(q + n * off), z))
                pr.append((*(q + n * (t / 2 - 0.25)), zb)); prof.append(pr)
            for i in range(len(prof) - 1):
                A, B = prof[i], prof[i + 1]
                for m in range(len(A) - 1): a.face([A[m], A[m + 1], B[m + 1], B[m]], rock_col(A[m + 1], seed))
            a.face(prof[0][::-1], shade(ROCK, 0.85)); a.face(prof[-1], shade(ROCK, 0.85))
        j = k


CRAG_SHAPES = None


def crag_shapes(n=8, seed=3):
    """Boulder shapes: an icosphere distorted by noise (ridges), a flat bottom (like rock_shapes in bl_relief.py)."""
    from mathutils import Vector, noise
    out = []
    for k in range(n):
        bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0); o = Vector((k * 13.7 + seed, k * 7.1, k * 3.3))
        for v in bm.verts:
            p = v.co * 1.1 + o; r = 1 + 0.55 * noise.ridged_multi_fractal(p * 0.8, 1.0, 2.0, 2, 1.0, 2.0) - 0.35
            v.co = v.co * r; v.co.z = max(v.co.z * 1.4, -0.35)
        V = np.array([v.co[:] for v in bm.verts]); F = [[v.index for v in f.verts] for f in bm.faces]; bm.free()
        V -= (V.max(0) + V.min(0)) / 2; V /= np.maximum(np.ptp(V, 0) / 2, 1e-6); out.append((V, F))
    return out


def crags(a, n_max, seed=7):
    """Boulders at cliffs under the walls — the rock is woven into the castle: steep ground (> 38°) 3–40 m outside
    the walls, not near the road, bridge or moat; one per 9 m cell; 2.5–7 m, sunk by 40 %."""
    global CRAG_SHAPES
    if n_max <= 0: return 0
    rng = np.random.default_rng(seed); W = [np.asarray(w["pts"], float) for w in C["walls"] if w["kind"] != "retaining"]
    if not W: return 0
    allw = np.vstack(W); lo = allw.min(0) - 40; hi = allw.max(0) + 40
    xs = np.arange(lo[0], hi[0], 3.0); ys = np.arange(lo[1], hi[1], 3.0); X, Y = np.meshgrid(xs, ys); X = X.ravel(); Y = Y.ravel()
    zx = (height(X + 1.0, Y) - height(X - 1.0, Y)) / 2; zy = (height(X, Y + 1.0) - height(X, Y - 1.0)) / 2
    sl = np.degrees(np.arctan(np.hypot(zx, zy))); m = sl > 38
    X, Y, zx, zy = X[m], Y[m], zx[m], zy[m]
    if not len(X): return 0
    dw = np.full(len(X), 1e9)
    for k0 in range(0, len(allw), 400):
        Wk = allw[k0:k0 + 400]; dw = np.minimum(dw, np.hypot(X[:, None] - Wk[None, :, 0], Y[:, None] - Wk[None, :, 1]).min(1))
    m = (dw > 3.0) & (dw < 40.0)
    for w in C["wards"]: m &= ~inside_v(np.asarray(w["poly"], float), X, Y)
    for L_, rr in ([(np.asarray(C["road"], float)[:, :2], 7.0)] if C.get("road") else []) + \
                  [(np.asarray(D["pts"], float), D["w"] / 2 + D["depth"] + 2) for D in C["ditches"]] + \
                  [(np.asarray(br["pts"], float), 6.0) for br in C["bridges"]]:
        for p_ in L_: m &= np.hypot(X - p_[0], Y - p_[1]) > rr
    X, Y, zx, zy = X[m], Y[m], zx[m], zy[m]
    if not len(X): return 0
    o = rng.permutation(len(X)); X, Y, zx, zy = X[o], Y[o], zx[o], zy[o]
    _, first = np.unique(np.floor(X / 9.0).astype(np.int64) * 100003 + np.floor(Y / 9.0).astype(np.int64), return_index=True)
    first = np.sort(first)[:n_max]
    if CRAG_SHAPES is None: CRAG_SHAPES = crag_shapes()
    rk = float(KN.get("rock", 0.5))
    for i in first:
        V, Fc = CRAG_SHAPES[int(rng.integers(len(CRAG_SHAPES)))]; size = float(rng.uniform(2.5, 7.0)) * (0.7 + 0.6 * rk)
        sx, sy, sz = size * rng.uniform(0.8, 1.2), size * rng.uniform(0.6, 0.9), size * rng.uniform(0.55, 0.95)
        ang = math.atan2(-zy[i], -zx[i]); ca, sa = math.cos(ang), math.sin(ang)        # the long side across the fall line
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]); Pv = (V * [sx / 2, sy / 2, sz / 2]) @ Rz.T
        z0 = float(height(X[i], Y[i])) - sz * 0.4                                     # sunk by 40 %, tilted along the slope,
        gx_, gy_ = zx[i], zy[i]; gn_ = math.hypot(gx_, gy_)                             # but not steeper than 35° (on a sheer face the boulder was stretched like a splinter)
        if gn_ > 0.7: gx_, gy_ = gx_ * 0.7 / gn_, gy_ * 0.7 / gn_
        Pw = Pv + [X[i], Y[i], z0]; Pw[:, 2] += gx_ * (Pw[:, 0] - X[i]) + gy_ * (Pw[:, 1] - Y[i])
        for f in Fc: a.face([Pw[q] for q in f], rock_col(Pw[f[0]], 11))
    return len(first)


# ---------------------------------------------------------------- walls
MERLON = dict(w=1.7, gap=0.8, h=1.2, parapet=1.1, pt=0.6)
_MERLON0 = dict(MERLON)                                                          # load(): _MERLON0 + ST["crenel"] (castles.yaml canon.crenel)


def merlon_box(a, c, d, nm, z, kind, col):
    """One merlon: c is the middle (at the parapet's outer face, inward by half the thickness), d runs along the
    wall with the merlon's length, nm points outward."""
    h = MERLON["h"]
    if kind == "swallowtail":
        for f in (-0.3, 0.3):
            a.box(c + d * f, d, MERLON["pt"], MERLON["w"] * 0.4, z, z + h * 1.25, col)
    elif kind == "stepped":
        a.box(c, d, MERLON["pt"], MERLON["w"], z, z + h * 0.6, col); a.box(c, d, MERLON["pt"], MERLON["w"] * 0.55, z + h * 0.6, z + h * 1.1, col)
    elif kind == "jagged":
        q = [c - d / 2 * 0.9 - nm * 0.3, c + d / 2 * 0.9 - nm * 0.3, c + d / 2 * 0.9 + nm * 0.3, c - d / 2 * 0.9 + nm * 0.3]
        for i in range(4): a.face([(*q[i], z), (*q[(i + 1) % 4], z), (*c, z + h * 2.2)], col)
    elif kind != "none":
        a.box(c, d, MERLON["pt"], MERLON["w"], z, z + h, col)


def merlons(a, P, N, zt, kind, col):
    """Merlons along the parapet in an even row at the center (towers, gates, walls without a row in the plan): P
    is the outer face of the parapet, N the outward normals, zt the top of the parapet."""
    seg = np.hypot(*np.diff(P, axis=0).T); s = np.concatenate([[0], np.cumsum(seg)]); L = s[-1]
    per = MERLON["w"] + MERLON["gap"]; n = int(L // per)
    if n < 1 or kind == "none": return
    off = (L - n * per + MERLON["gap"]) / 2
    for k in range(n):
        u0 = off + k * per; u1 = u0 + MERLON["w"]; um = (u0 + u1) / 2
        def at(u, A): return np.array([np.interp(u, s, A[:, i]) for i in range(A.shape[1])])
        p0, p1, pm = at(u0, P), at(u1, P), at(um, P); nm = at(um, N); nm /= max(np.linalg.norm(nm), 1e-9)
        merlon_box(a, pm - nm * MERLON["pt"] / 2, p1 - p0, nm, float(np.interp(um, s, zt)), kind, col)


def grid_merlons(a, w, P, T, zp, ok, kind, col):
    """Merlons along the plan's row (crenels — [[start, count], …] measured along the wall): the pitch and
    position are the same as for the stairstep's ledges, a ledge falls on a clearance; there is no merlon where
    there is no wall under it (a gate, a house as a wall) or machicolations."""
    if kind == "none": return
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]); L = float(s[-1]); p = MERLON["w"] + MERLON["gap"]
    for g0, n in w.get("crenels", []):
        for m in range(int(n)):
            u = g0 + (m + 0.5) * p
            if w["closed"]: u %= L
            j0 = max(0, int(np.searchsorted(s, u - MERLON["w"] / 2 - 0.15, side="right")) - 1)
            j1 = min(len(P) - 1, int(np.searchsorted(s, u + MERLON["w"] / 2 + 0.15, side="left")))
            if not ok[j0:j1 + 1].all(): continue
            j = int(np.clip(np.searchsorted(s, u, side="right") - 1, 0, len(P) - 2))
            while j < len(P) - 2 and s[j + 1] - s[j] < 1e-6: j += 1                 # not a ledge segment (two stations at one point)
            seg = P[j + 1] - P[j]; ln = float(np.linalg.norm(seg))
            if ln < 1e-6: continue
            dv = seg / ln; nm = np.array([dv[1], -dv[0]]); q = P[j] + seg * float(np.clip((u - s[j]) / ln, 0, 1))
            t_ = float(np.interp(u, s, T)); merlon_box(a, q + nm * (t_ / 2 - MERLON["pt"] / 2), dv * MERLON["w"], nm, float(np.interp(u, s, zp)), kind, col)


def wall_flags(wi, n):
    """Flags of the wall stations by element: machicolations — their own parapet, stepped walkway."""
    nopar = np.zeros(n, bool); step = np.zeros(n, bool)
    for f in FEAT:
        if f.get("wall") != wi or "ii" not in f: continue
        if f["kind"] == "machicolation": nopar[f["ii"]] = True
        if f["kind"] == "stepped_walk": step[f["ii"]] = True
    return nopar, step


def dirs(Q, closed):
    """Directions to the next and from the previous station; at a ledge (two stations at one point) — from the
    neighboring station."""
    tn = (np.roll(Q, -1, 0) - Q) if closed else np.vstack([Q[1:] - Q[:-1], Q[-1:] - Q[-2:-1]])
    tp = (Q - np.roll(Q, 1, 0)) if closed else np.vstack([Q[1:2] - Q[:1], Q[1:] - Q[:-1]])
    m = len(Q)
    for D, step_ in ((tn, 1), (tp, -1)):
        bad = np.linalg.norm(D, axis=1) < 1e-6
        for _ in range(3):
            if not bad.any(): break
            for j in (np.flatnonzero(bad)[::-1] if step_ > 0 else np.flatnonzero(bad)):
                jj = j + step_
                if closed: jj %= m
                elif not 0 <= jj < m: jj = j - step_
                if 0 <= jj < m and not bad[jj]: D[j] = D[jj]; bad[j] = False
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9); tp /= np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-9)
    return tn, tp


def wall(a, w, gaps, wi=-1):
    """Wall: a section (batter, walkway, parapet) along the points; breaks under gates and where a house has become
    the wall; a stairstep wall — a ledge in the plan (two stations at one point), merlons follow the plan's row,
    the ledge in a clearance; on a rock edge — masonry starting from the rock (rock_top); a terrace retaining
    wall — a parapet without merlons."""
    P = np.asarray(w["pts"], float); walk = np.asarray(w["walk"], float).copy(); base = np.asarray(w["base"], float).copy()
    nopar, step = wall_flags(wi, len(P)); ret = w["kind"] == "retaining"
    zp_all = np.where(nopar, walk + 0.02, walk + (1.0 if ret else MERLON["parapet"]))   # parapet — following the plan's walkway
    if not w.get("steps"): walk = np.where(step, np.floor(walk / 1.1) * 1.1 + 0.55, walk)   # walkway steps on a steep stretch — only the walkway itself
    rt = w.get("rock_top")
    if rt: base = np.array([max(b_, v) if v is not None else b_ for b_, v in zip(base, rt)])   # masonry — from the top of the rock
    T = np.asarray(w["t"], float) if isinstance(w["t"], list) else np.full(len(P), float(w["t"]))
    keep = np.ones(len(P), bool)
    for g, half in gaps:
        keep &= np.hypot(P[:, 0] - g[0], P[:, 1] - g[1]) > half
    if w.get("absorbed"): keep[np.asarray(w["absorbed"], int)] = False         # house as a wall: here the wall is the outer wall of the house
    runs = []; cur = []
    for i in range(len(P)):
        if keep[i]: cur.append(i)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    closed = w["closed"] and keep.all()
    if w["closed"] and not keep.all() and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(P) - 1:
        runs[0] = runs[-1] + runs[0]; runs.pop()
    for r in runs:
        if len(r) < 2: continue
        Q = P[r]
        tn, tp = dirs(Q, closed)
        nn = np.column_stack([tn[:, 1], -tn[:, 0]]); npv = np.column_stack([tp[:, 1], -tp[:, 0]])
        N = nn + npv; N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        N = N / np.clip((N * nn).sum(1), 0.35, 1.0)[:, None]                      # mitered joint at corners (like an outline inset)
        t = T[r]; wk = walk[r]; b = base[r]; zp = zp_all[r]
        h = wk - b
        bat = ST["batter"] * np.minimum(h, 12.0) * 0.5
        knee = b + np.minimum(h * 0.4, 1.0 + h * 0.25)
        sec = [Q + N * (t / 2 + bat)[:, None], Q + N * (t / 2)[:, None], Q + N * (t / 2)[:, None], Q + N * (t / 2 - MERLON["pt"])[:, None],
               Q + N * (t / 2 - MERLON["pt"])[:, None], Q - N * (t / 2)[:, None], Q - N * (t / 2)[:, None]]
        zs = [b, knee, zp, zp, wk, wk, b]
        loops = [[(*sec[k][i], zs[k][i]) for i in range(len(Q))] for k in range(7)]
        for k in range(7):
            a.quad_strip(loops[k], loops[(k + 1) % 7], STONE, closed=closed)
        if not closed:
            for e in (0, len(Q) - 1):
                cap = [loops[k][e] for k in range(7)]
                a.face(cap if e else cap[::-1], STONE)
        if ret or w.get("crenels") is not None: continue                         # retaining wall — no merlons; the plan's row — below, along the whole wall
        keepm = ~nopar[r]; j = 0
        while j < len(Q):                                                        # merlons — where there are no machicolations
            if not keepm[j]: j += 1; continue
            k = j + 1
            while k < len(Q) and keepm[k]: k += 1
            if k - j >= 2: merlons(a, (Q + N * (t / 2)[:, None])[j:k], N[j:k], zp[j:k], ST["merlon"], STONE)
            j = k
    if not ret and w.get("crenels") is not None:
        grid_merlons(a, w, P, T, zp_all, keep & ~nopar, ST["merlon"], STONE)


# ---------------------------------------------------------------- towers
def footprint(plan, c, r, d, n_round=24, prow=False):
    P = _footprint(plan, c, r, d, n_round)
    if not prow: return P
    c = np.asarray(c, float); d = np.asarray(d if d is not None else (1, 0), float); d /= max(np.linalg.norm(d), 1e-9); o = np.array([d[1], -d[0]])
    ang = np.arctan2(*(P - c).T[::-1]); a0 = math.atan2(o[1], o[0]); da = (ang - a0 + math.pi) % (2 * math.pi) - math.pi
    keep = np.abs(da) > math.radians(52)                                         # wedge toward the threat: the sharp edge outward
    Q = [(float(dd), p) for dd, p, k in zip(da, P, keep) if k]
    rr = float(np.hypot(*(P - c).T).mean())
    Q += [(math.radians(51), c + rr * np.array([math.cos(a0 + math.radians(51)), math.sin(a0 + math.radians(51))])),
          (0.0, c + o * rr * 1.75), (math.radians(-51), c + rr * np.array([math.cos(a0 - math.radians(51)), math.sin(a0 - math.radians(51))]))]
    Q.sort(key=lambda q: (q[0] + 2 * math.pi) % (2 * math.pi))
    return np.array([q[1] for q in Q])


def _footprint(plan, c, r, d, n_round=24):
    c = np.asarray(c, float); d = np.asarray(d if d is not None else (1, 0), float); d /= max(np.linalg.norm(d), 1e-9)
    o = np.array([d[1], -d[0]])                                                 # outward
    if plan == "round":
        return np.array([c + r * np.array([math.cos(2 * math.pi * i / n_round), math.sin(2 * math.pi * i / n_round)]) for i in range(n_round)])
    if plan == "D":                                                              # semicircle outward + rectangle inward
        pts = [c + d * r * math.cos(t) + o * r * math.sin(t) for t in np.linspace(0, math.pi, 13)]
        return np.array(pts + [c - d * r - o * r, c + d * r - o * r])
    k = {"square": 4, "poly6": 6, "poly8": 8}.get(plan, 8)
    a0 = math.atan2(d[1], d[0]) + (math.pi / 4 if k == 4 else math.pi / k)
    rr = r / math.cos(math.pi / k)
    return np.array([c + rr * np.array([math.cos(a0 + 2 * math.pi * i / k), math.sin(a0 + 2 * math.pi * i / k)]) for i in range(k)])


def roof(a, ring, z, kind, r, col, h=None):
    """Roof over the top ring: cone/pyramid/tent/spire/spike — up to the apex; the height h comes from the plan
    (roofscape), otherwise from the slope."""
    P = np.asarray(ring, float); c = P.mean(0)
    if h is None:
        pitch = math.radians(min(max(ST["roof_pitch"], 35), 55)); h = r * math.tan(pitch)
    eave = c + (P - c) * 1.12; apex = (*c, z + 0.4 + float(h))
    lo = [(*p, z - 0.2) for p in eave]; hi = [(*p, z + 0.4) for p in eave]
    a.quad_strip(lo, hi, col, closed=True); a.face(lo[::-1], col)
    for i in range(len(eave)):
        a.face([hi[i], hi[(i + 1) % len(eave)], apex], col)


def tower(a, ar, t, is_keep=False):
    plan = t["plan"]; r = max(float(t["r"]), 1.5); c = np.asarray(t["xy"], float); d = t.get("dir")
    base, top = float(t["base"]), float(t["top"])
    P = footprint(plan, c, r, d, prow=bool(t.get("prow")))
    if t.get("rock"):                                                            # on a rock plinth: masonry starts from the top of the rock
        rk = t["rock"]; rock_mass(ARK, P, float(rk["z1"]), rk.get("skirt", (0.5, 1.8)), seed=float(c[0] * 0.13 + c[1] * 0.07))
    h = top - base; plinth = base + min(h * 0.3, 2.5 + h * 0.15)
    lean = math.radians(t.get("lean", 0.0)); o = np.array([d[1], -d[0]]) if d else np.zeros(2)
    def sh(z): return o * math.tan(lean) * (z - base)                           # tilt (hell) — outward
    bat = 1.0 + ST["batter"] * 0.8
    a.quad_strip([(*(c + (p - c) * bat), base) for p in P], [(*(p + sh(plinth)), plinth) for p in P], STONE, closed=True)
    roofk = "flat" if t.get("butterfass") else t.get("roof", "flat")
    body_top = top
    gal = TWO_LEVEL and roofk != "flat" and not is_keep                          # Pierrefonds: machicolations and a gallery under the roof
    mach = ST.get("machicolation") and (roofk in ("flat",) or gal) and not is_keep
    if mach:
        z1 = top - 1.6; a.quad_strip([(*(p + sh(plinth)), plinth) for p in P], [(*(p + sh(z1)), z1) for p in P], STONE, closed=True)
        Pm = c + (P - c) * (1 + 0.9 / r)
        a.quad_strip([(*(p + sh(z1)), z1) for p in P], [(*(q + sh(z1 + 0.6)), z1 + 0.6) for q in Pm], STONE_D, closed=True)
        a.quad_strip([(*(q + sh(z1 + 0.6)), z1 + 0.6) for q in Pm], [(*(q + sh(top)), top) for q in Pm], STONE, closed=True)
        if gal and plan == "round":                                              # gallery openings around the circle
            cc_ = c + sh(z1); n_o = max(6, int(2 * math.pi * r / 2.4))
            for k_ in range(n_o):
                u_ = np.array([math.cos(2 * math.pi * k_ / n_o), math.sin(2 * math.pi * k_ / n_o)])
                opening(a, cc_ + u_ * (r + 0.9), u_, z1 + 0.75, 0.6, 0.75, "rect")
        P = Pm
    else:
        a.quad_strip([(*(p + sh(plinth)), plinth) for p in P], [(*(p + sh(top)), top) for p in P], STONE, closed=True)
    ring = [p + sh(top) for p in P]
    oo = o if d else np.array([1.0, 0.0]); dd = np.asarray(d, float) if d else np.array([0.0, 1.0])
    for zz in np.arange(plinth + 2.0, top - 2.2, 4.2 if not is_keep else 4.6):    # loopholes; on the main tower, higher up — windows
        for ang in ((-0.9, 0.0, 0.9) if not is_keep else (-1.4, -0.5, 0.5, 1.4)):
            v = oo * math.cos(ang) + dd * math.sin(ang); v /= np.linalg.norm(v)
            q = c + v * (r * (1.0 if plan in ("round", "D") else 1.0 / math.cos(math.pi / 8) * 0.92)) + sh(zz)
            if win_blocked(q, v, zz): continue
            if is_keep and zz > plinth + 8: opening(a, q, v, zz, 1.0, 1.8, ARCH_TOP)
            else: opening(a, q, v, zz, 0.22, 1.5, "slit")
    if roofk == "flat":
        a.face([(*p, top) for p in ring], STONE_D)
        cc = np.mean(ring, 0); inner = [cc + (p - cc) * (1 - MERLON["pt"] / r) for p in ring]
        a.quad_strip([(*p, top) for p in ring], [(*p, top + MERLON["parapet"]) for p in ring], STONE, closed=True)
        a.quad_strip([(*p, top + MERLON["parapet"]) for p in ring], [(*p, top + MERLON["parapet"]) for p in inner], STONE, closed=True)
        Q = np.asarray(ring + [ring[0]]); Nn = np.asarray([(q - cc) / max(np.linalg.norm(q - cc), 1e-9) for q in Q])
        merlons(a, Q, Nn, np.full(len(Q), top + MERLON["parapet"]), ST["merlon"], STONE)
    else:
        a.face([(*p, top) for p in ring], STONE_D)
        roof(ar, ring, top, roofk, r * 1.05, ROOF, h=t.get("roof_h"))


def lathe_ring(c, prof, z0, n=24):
    """Rings of revolution: prof — [(radius, height above z0)] → [[(x, y, z)…]…]."""
    return [[(c[0] + r_ * math.cos(2 * math.pi * i / n), c[1] + r_ * math.sin(2 * math.pi * i / n), z0 + h_) for i in range(n)] for r_, h_ in prof]


def lathe(a, c, prof, z0, col, n=24):
    """Solid of revolution by a profile (onion, dome, drum); the top is a point if the radius is 0."""
    R = lathe_ring(c, prof, z0, n)
    for A, B in zip(R[:-1], R[1:]): a.quad_strip(A, B, col, closed=True)


def stage_windows(a, sg, P, c, k, t, is_keep):
    """Tier windows by floors: on the faces of a polygonal prism (for a circle — 8 sides), staggered by floor and
    by face — no grid."""
    z0, z1 = float(sg["z0"]), float(sg["z1"]); sh = float(ST["storey"]); kind = sg.get("win", "slit")
    r = float(sg["r"]); plan = sg["plan"]
    if plan in ("round", "D"): dirs8 = [np.array([math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)]) for i in range(8)]; pts = [c + u * r for u in dirs8]
    else:
        pts = [(P[i] + P[(i + 1) % len(P)]) / 2 for i in range(len(P))]; dirs8 = [(q - c) / max(np.linalg.norm(q - c), 1e-9) for q in pts]
    zmin = float(t.get("walk", -1e9)) + 0.6 if not is_keep else -1e9               # wall tower — windows above the wall walkway
    for j, zz in enumerate(np.arange(z0 + (2.0 if k == 0 else 1.0), z1 - 2.2, sh)):
        if zz < zmin: continue
        for i, (q, u) in enumerate(zip(pts, dirs8)):
            if (i + j + k) % 2: continue
            if win_blocked(q, u, zz, self_xy=c): continue
            if kind == "lancet": opening(a, q, u, zz, min(0.9, r * 0.25), min(2.8, sh - 0.8), "pointed")
            elif kind == "arched": opening(a, q, u, zz, min(1.2, r * 0.3), min(2.3, sh - 1.0), "round")
            else: opening(a, q, u, zz, 0.22, 1.4, "slit")


def stage_parapet(a, P, c, r, z, merl=True):
    """Parapet along the edge of a platform (ledge, top): a low wall and the style's merlons."""
    inner = [c + (p - c) * (1 - MERLON["pt"] / max(r, 1.0)) for p in P]
    a.quad_strip([(*p, z) for p in P], [(*p, z + MERLON["parapet"]) for p in P], STONE, closed=True)
    a.quad_strip([(*p, z + MERLON["parapet"]) for p in P], [(*p, z + MERLON["parapet"]) for p in inner], STONE, closed=True)
    if merl:
        Q = np.asarray(list(P) + [P[0]]); Nn = np.asarray([(q - c) / max(np.linalg.norm(q - c), 1e-9) for q in Q])
        merlons(a, Q, Nn, np.full(len(Q), z + MERLON["parapet"]), ST["merlon"], STONE)


def pinnacle(a, ar, p, z0, w, h):
    """Pinnacle (Roriczer): a square shaft and a small pyramid three times as tall as the width."""
    a.box(p, (1, 0), w, w, z0, z0 + h, STONE)
    q = [p + np.array(v) * w / 2 for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4): ar.face([(*q[i], z0 + h), (*q[(i + 1) % 4], z0 + h), (*p, z0 + h + 3.2 * w)], STONE_D)


def staged_tower(a, ar, t, is_keep=False):
    """Tower by tiers (castle_tower.py): prism tiers (a batter at the base), transitions (a ledge with a parapet,
    pinnacles at the corners of a square under a polygon, a balcony gallery on corbels, a corbel-out), windows
    by floors, a crown (merlons, machicolations, spikes, horns, a crown), a roof (needle, cone, tent, pyramid,
    onion, dome), corner turrets."""
    c = np.asarray(t["xy"], float); d = np.asarray(t.get("dir") or [1.0, 0.0], float); d /= max(np.linalg.norm(d), 1e-9); G = t["stages"]
    if t.get("rock"):
        rk = t["rock"]; rock_mass(ARK, _footprint(G[0]["plan"], c, float(G[0]["r"]), d), float(rk["z1"]), rk.get("skirt", (0.5, 1.8)), seed=float(c[0] * 0.13 + c[1] * 0.07))
    for k, sg in enumerate(G):
        r = float(sg["r"]); P = _footprint(sg["plan"], c, r, d, 28); z0, z1 = float(sg["z0"]), float(sg["z1"]); zs = z0
        if k == 0:                                                               # batter of the base
            zs = z0 + min(0.22 * (z1 - z0), 3.5); bat = 1 + ST["batter"] * 0.8 + 0.05
            a.quad_strip([(*(c + (p - c) * bat), z0) for p in P], [(*p, zs) for p in P], STONE, closed=True)
        a.quad_strip([(*p, zs) for p in P], [(*p, z1) for p in P], STONE, closed=True)
        stage_windows(a, sg, P, c, k, t, is_keep)
        if k == len(G) - 1: break
        nx = G[k + 1]; rn = float(nx["r"]); Pn = _footprint(nx["plan"], c, rn, d, 28); tr = sg.get("trans")
        if tr == "corbel":                                                       # corbel-out on corbels: the upper tier is wider
            Pl = [c + (q - c) * (r / max(rn, 1e-6)) for q in Pn]                  # the lower ring of corbels follows the plan of the upper tier
            a.quad_strip([(*p, z1 - 1.3) for p in Pl], [(*q, z1) for q in Pn], STONE_D, closed=True); a.face([(*q, z1) for q in Pn], STONE_D)
            continue
        a.face([(*p, z1) for p in P], STONE_D)                                    # ledge platform
        if tr == "gallery":                                                      # balcony on corbels around the tier
            Pg = [c + (p - c) * (1 + 1.5 / max(r, 1.0)) for p in P]
            a.quad_strip([(*p, z1 - 1.1) for p in P], [(*q, z1 - 0.3) for q in Pg], STONE_D, closed=True)
            a.quad_strip([(*q, z1 - 0.3) for q in Pg], [(*q, z1) for q in Pg], STONE, closed=True)
            a.face([(*q, z1) for q in Pg], STONE_D); stage_parapet(a, Pg, c, r + 1.5, z1, merl=False)
        elif r - rn > 0.9:
            stage_parapet(a, P, c, r, z1, merl=(tr != "pinnacles"))
        if tr == "pinnacles" and sg["plan"] == "square":                         # pinnacles at the corners of the square under the octagon
            w = max(0.7, 0.12 * r); hp = min(0.4 * (float(nx["z1"]) - float(nx["z0"])), 12.0)
            for p in P: pinnacle(a, ar, c + (p - c) * (1 - 0.9 * w / max(np.linalg.norm(p - c), 1e-9)), z1, w, hp)
    top = G[-1]; r = float(top["r"]); Pt = _footprint(top["plan"], c, r, d, 28); zt = float(top["z1"]); cr = t["crown"]["kind"]; sp = t["spire"]
    a.face([(*p, zt) for p in Pt], STONE_D)
    if cr == "merlons": stage_parapet(a, Pt, c, r, zt)
    elif cr == "machicolation":
        Pm = [c + (p - c) * (1 + 0.9 / r) for p in Pt]
        a.quad_strip([(*p, zt - 1.6) for p in Pt], [(*q, zt - 1.0) for q in Pm], STONE_D, closed=True)
        a.quad_strip([(*q, zt - 1.0) for q in Pm], [(*q, zt) for q in Pm], STONE, closed=True); stage_parapet(a, Pm, c, r + 0.9, zt)
    elif cr == "spikes":                                                         # spikes along the edge
        Q = np.asarray(list(Pt) + [Pt[0]]); Nn = np.asarray([(q - c) / max(np.linalg.norm(q - c), 1e-9) for q in Q])
        merlons(a, Q, Nn, np.full(len(Q), zt), "jagged", STONE); merlons(a, Q, Nn, np.full(len(Q), zt + 1.2), "jagged", STONE_D)
    elif cr == "horns":                                                          # four horns (Orthanc): faceted, tapering, slightly outward
        corners = [Pt[i] for i in range(0, len(Pt), max(1, len(Pt) // 4))][:4]
        for q in corners:
            u = (q - c) / max(np.linalg.norm(q - c), 1e-9); bw = max(1.2, 0.32 * r); hh = r * float(1.1 + 0.3 * ((q[0] * 7 + q[1] * 3) % 1))
            rings = []
            for j, f in enumerate(np.linspace(0, 1, 6)):
                cc = q - u * bw * 0.6 + u * (0.35 * r * math.sin(math.pi * f * 0.8)); ww = bw * (1 - 0.85 * f)
                rings.append([(*(cc + np.array([math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)]) * ww), zt + hh * f) for i in range(5)])
            for A, B in zip(rings[:-1], rings[1:]): a.quad_strip(A, B, STONE, closed=True)
    elif cr == "crown":                                                          # crown of blades (Barad-dur)
        nb = max(8, int(2 * math.pi * r / 2.6))
        for i in range(nb):
            u = np.array([math.cos(2 * math.pi * i / nb), math.sin(2 * math.pi * i / nb)]); tg = np.array([-u[1], u[0]]); q = c + u * r * 0.97
            hb = r * (0.55 + 0.25 * (i % 2)); tip = q + u * hb * 0.15
            for s_ in (-0.15, 0.15):
                a.face([(*(q - tg * 0.9 + u * s_), zt), (*(q + tg * 0.9 + u * s_), zt), (*(tip + u * s_), zt + hb)], STONE_D)
    hs = float(sp.get("h", 0.0)); kind = sp["kind"]
    if kind != "flat" and hs > 0.5:
        inset = cr in ("merlons", "machicolation")
        rb = r * (0.8 if inset else 1.0); ring = [c + (p - c) * (rb / r) for p in Pt]
        if kind in ("needle", "cone", "pyramid", "tent"):
            if kind == "needle": ring = [c + rb * np.array([math.cos(math.pi / 8 + math.pi * i / 4), math.sin(math.pi / 8 + math.pi * i / 4)]) for i in range(8)]
            roof(ar, ring, zt, "cone", rb, ROOF, h=hs)
        elif kind == "onion":
            lathe(ar, c, [(rb * 0.85, 0.0), (rb * 1.15, 0.22 * hs), (rb * 1.05, 0.45 * hs), (rb * 0.6, 0.7 * hs), (rb * 0.18, 0.88 * hs), (0.02, hs)], zt, ROOF)
        elif kind == "dome":                                                     # observatory: a drum and a dome with a slit
            lathe(a, c, [(rb * 0.92, 0.0), (rb * 0.92, 0.45 * rb)], zt, STONE)
            lathe(ar, c, [(rb * 0.95 * math.cos(math.radians(a_)), 0.45 * rb + rb * 0.95 * math.sin(math.radians(a_))) for a_ in range(0, 91, 15)], zt, ROOF)
            opening(ar, c + np.array([rb * 0.6, 0.0]), np.array([1.0, 0.0]), zt + 0.45 * rb + rb * 0.1, 0.5, rb * 0.55, "rect")
    if t.get("turrets"):                                                         # turrets at the corners of the top tier
        tt = t["turrets"]; cs_ = [Pt[i] for i in range(0, len(Pt), max(1, len(Pt) // tt["n"]))][: tt["n"]]
        for q in cs_:
            u = (q - c) / max(np.linalg.norm(q - c), 1e-9)
            AE.bartizan(ar, q + u * float(tt["r"]) * 0.3, zt + (MERLON["parapet"] if cr == "merlons" else 0.0), float(tt["r"]), KC,
                        roof="cone" if kind != "flat" else "flat", pitch=max(60, ST["roof_pitch"]), out=u)


def keep_parts(a, ar, k):
    """Parts of the main tower by type: donjon — corner pilaster strips and an entrance annex; bergfried — a high
    entrance; spire — ribs."""
    c = np.asarray(k["xy"], float); r = float(k["r"]); base, top = float(k["base"]), float(k["top"]); kind = k["kind"]
    d = np.asarray(k.get("dir") or (1, 0), float); d /= max(np.linalg.norm(d), 1e-9); o = np.array([d[1], -d[0]])
    if kind in ("donjon", "norman", "homage") and k["plan"] == "square":
        P = footprint("square", c, r, d)
        for p in P:                                                              # corner pilaster strips
            u = (p - c) / max(np.linalg.norm(p - c), 1e-9)
            a.box(p - u * 0.4, d, 1.8, 1.8, base, top - 0.4, STONE)
        if kind in ("donjon", "norman"):                                         # entrance annex on the court side
            q = c - o * (r + 2.6); a.box(q, d, 5.0, r * 1.1, base, base + (top - base) * 0.45, STONE)
            a.face([(*(q - d * r * 0.55 - o * 2.5), base + (top - base) * 0.45), (*(q + d * r * 0.55 - o * 2.5), base + (top - base) * 0.45),
                    (*(q + d * r * 0.55 + o * 2.5), base + (top - base) * 0.45 + 2.2), (*(q - d * r * 0.55 + o * 2.5), base + (top - base) * 0.45 + 2.2)], ROOF)
    if kind == "bergfried":                                                      # entrance at a height of ~ 9 m, from the court
        opening(a, c - o * r, -o, base + min(9.0, (top - base) * 0.4), 1.3, 2.3, ARCH_TOP)
    if kind == "spire":                                                          # buttress ribs of the spire
        for i in range(8):
            u = np.array([math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)])
            a.prism([c + u * r * 0.95 + np.array([-u[1], u[0]]) * 0.8, c + u * (r + 3.5) + np.array([-u[1], u[0]]) * 0.8,
                     c + u * (r + 3.5) - np.array([-u[1], u[0]]) * 0.8, c + u * r * 0.95 - np.array([-u[1], u[0]]) * 0.8],
                    base, base + (top - base) * 0.45, STONE_D, scale_top=0.75, centre=c + u * r)


# ---------------------------------------------------------------- gate: a block with an arched passage
def arch_pts(s, h, n=10):
    r = (h * h + s * s / 4) / s; xc = r - s / 2; a1 = math.atan2(h, -xc)
    left = [(xc + r * math.cos(math.pi + (a1 - math.pi) * i / n), r * math.sin(math.pi + (a1 - math.pi) * i / n)) for i in range(n + 1)]
    return left + [(-x, z) for x, z in reversed(left[:-1])]


def gate_block(a, g, t_wall, half):
    p = np.asarray(g["xy"], float); d = np.asarray(g["dir"], float); o = np.asarray(g["out"], float)
    z0 = float(g["z"]) - 1.0; zw = float(g["walk"]); wd = float(g["w"]); hg = float(g["h"])
    depth = t_wall * 1.25; L = 2 * half
    tower_gate = g.get("kind") == "tower" and g.get("r")
    if tower_gate:
        r = float(g["r"]); L = 2 * r * 1.05; depth = max(depth, 2 * r * 0.95); zw = zw + float(g["up"])
    rise = wd * (0.5 if ST.get("era_round") else 0.62)
    prof = [(-wd / 2, z0 + 1.0)] + [(x, z0 + 1.0 + hg - rise + zz) for x, zz in arch_pts(wd, rise)] + [(wd / 2, z0 + 1.0)]
    # facade walls with an arch: the block outline minus the opening; built from strips: the left pier, the right
    # one, above the arch
    top = zw + MERLON["parapet"]
    for s_ in (-1, 1):                                                          # wall piers
        cx = p + d * s_ * (wd / 2 + (L / 2 - wd / 2) / 2)
        a.box(cx, d, depth, L / 2 - wd / 2, z0, top, STONE)
    pts = np.array(prof[1:-1])                                                   # above the arch — strips along the arc
    for (x1, z1), (x2, z2) in zip(pts[:-1], pts[1:]):
        xm = (x1 + x2) / 2; zm = min(z1, z2)
        a.box(p + d * xm, d, depth, abs(x2 - x1) + 0.02, zm, top, STONE)
    q = np.asarray([p - d * L / 2, p + d * L / 2]); merlons(a, q + o * depth / 2, np.array([o, o]), np.full(2, top), ST["merlon"], STONE)
    if tower_gate:
        merlons(a, q - o * depth / 2, np.array([-o, -o]), np.full(2, top), ST["merlon"], STONE)
        a.box(p, d, depth, L, top - 0.3, top, STONE_D)
    a.box(p, d, depth * 0.9, wd, z0 + 1.0 - 0.05, z0 + 1.0, ROAD)              # threshold


# ---------------------------------------------------------------- openings: windows, doors, loopholes (dark openings on a face)
def opening(a, c, n, z0, w, h, top="rect", eps=0.04):
    """Opening on a vertical face: c is a point of the face (x, y), n points outward, bottom z0, width w, height h;
    the top is rect, round, pointed, horseshoe, slit."""
    n = np.asarray(n, float); n /= max(np.linalg.norm(n), 1e-9); t = np.array([-n[1], n[0]]); c = np.asarray(c, float) + n * eps
    if top == "rect" or top == "slit" or w < 0.5:
        prof = [(-w / 2, 0), (w / 2, 0), (w / 2, h), (-w / 2, h)]
    else:
        rise = w / 2 if top == "round" else (w * 0.75 if top == "pointed" else w * 0.6)
        sp = h - rise; arc = [(x, sp + z) for x, z in arch_pts(w, rise, 6)]
        if top == "horseshoe": arc = [(x * (1 + 0.15 * math.sin(math.pi * i / (len(arc) - 1))), z) for i, (x, z) in enumerate(arc)]
        prof = [(-w / 2, 0), (w / 2, 0)] + arc[::-1]
    a.face([(*(c + t * x), z0 + z) for x, z in prof], HOLE)


def facade_windows(a, p0, p1, n, floor, storeys, sh, w, h, rhythm, top, door=False, tr=False, tr_storey=None, self_poly=None, zmin=None):
    """Windows by floors on the face p0–p1 (outward n): pitch rhythm, inset from the corners; a door in the middle
    of the first floor; zmin — no windows below it (a house as a wall: outward — only above the walkway)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); L = float(np.hypot(*(p1 - p0))); d = (p1 - p0) / max(L, 1e-9)
    k = int((L - 1.2) // rhythm)
    if k < 1: return
    xs = [(L - (k - 1) * rhythm) / 2 + i * rhythm for i in range(k)]
    for st_ in range(storeys):
        for i, x in enumerate(xs):
            if door and st_ == 0 and i == len(xs) // 2:
                opening(a, p0 + d * x, n, floor, max(1.4, w * 1.2), min(sh - 0.6, 2.7), top); continue
            z_ = floor + st_ * sh + (0.9 if st_ == 0 else 0.8)
            if zmin is not None and z_ < zmin: continue
            if win_blocked(p0 + d * x, n, z_, self_poly=self_poly): continue
            top_ = top if (tr_storey is None or st_ == tr_storey) else ("round" if top == "pointed" else top)
            opening(a, p0 + d * x, n, z_, w, min(h, sh - 1.3), top_)
            if tr and top_ == "pointed" and w >= 1.0 and (tr_storey is None or st_ == tr_storey): AE.tracery(a, p0 + d * x, n, z_, w, min(h, sh - 1.3), KC)


# ---------------------------------------------------------------- buildings
def building(a, ar, b):
    R = np.asarray(b["poly"], float); z0, eave = float(b["base"]), float(b["eave"]); d = np.asarray(b["axis"], float)
    n_in = np.array([-d[1], d[0]]); D = float(b["D"]); L = float(b["L"])
    pitch = math.radians(max(float(b.get("pitch", 45)), 0.0)); kind = b["roof"]
    c = R.mean(0); sh = float(ST["storey"]); fl = float(b.get("floor", z0)); ns = int(b["storeys"])
    big = b["kind"] in ("palas", "palas_big", "governor", "hall", "wing", "tower_house")
    w_, h_ = (1.3, 2.4) if big else (0.9, 1.5)
    tr = bool(b.get("tracery")); hall_tr = b.get("tracery") == "hall"
    fl_w = b.get("flush") if isinstance(b.get("flush"), dict) else None
    if b.get("rock"):                                                            # house on a rock: the rock up to the floor
        rock_mass(ARK, R, float(b["rock"]["z1"]), b["rock"].get("skirt", (0.5, 1.8)), seed=float(c[0] * 0.11 + c[1] * 0.05), n=24)
    z_st = max(fl - 0.05, float(fl_w["walk"]) + 0.4 if fl_w else fl - 0.05)       # stone bottom: the plinth up to the floor, for a "house-wall" — up to the walkway
    if z_st - z0 > 0.6: a.prism(R, z0, z_st, STONE_D if not fl_w else STONE, top=False)
    else: z_st = z0
    if fl_w:                                                                     # outward — windows only above the wall walkway (defense)
        facade_windows(a, R[1], R[0], -n_in, fl, ns, sh, w_ * 0.85, h_, 5.2, ARCH_TOP if big else "rect", self_poly=R, zmin=float(fl_w["walk"]) + 1.2)
    stair = None
    if big and L > 14 and ns >= 2:                                             # the stair tower's place is decided before the windows (windows avoid it)
        rs = 1.7 * K ** 0.4; stair = (R[3] + (R[2] - R[3]) * 0.3 + n_in * rs * 0.8, rs); STAIR_T.append((stair[0].copy(), rs))
    if b["kind"].startswith("chapel"):                                         # chapel — tall windows in one tier (stained glass)
        facade_windows(a, R[2], R[3], n_in, fl, 1, eave - fl, 1.3 if tr else 1.1, (eave - fl) * 0.6, 4.5, "pointed" if ARCH_TOP != "round" else "round",
                       tr=tr, self_poly=R)
    else:
        facade_windows(a, R[2], R[3], n_in, fl, ns, sh, w_, h_, 4.6 if big else 5.4,
                       ("pointed" if hall_tr and ARCH_TOP != "round" else ARCH_TOP) if big else "rect", door=True, tr=hall_tr,
                       tr_storey=1 if hall_tr else None, self_poly=R)
    if b.get("fachwerk") and kind != "flat":                                    # half-timbering of the upper floors
        zf0 = fl + sh * int(b["fachwerk"]); zf1 = eave - 0.05
        if zf1 - zf0 > 1.5:
            AE.fachwerk(a, R[3], R[2], n_in, zf0, zf1, KC)
            AE.fachwerk(a, R[2], R[1], d, zf0, zf1, KC); AE.fachwerk(a, R[0], R[3], -d, zf0, zf1, KC)
        if D >= 7: facade_windows(a, R[1], R[2], d, fl, ns, sh, w_ * 0.8, h_, 5.0, "rect", self_poly=R); facade_windows(a, R[3], R[0], -d, fl, ns, sh, w_ * 0.8, h_, 5.0, "rect", self_poly=R)
    if stair is not None:                                                      # stair tower on the court facade
        q, rs = stair; zt = eave + 2.5
        P8 = footprint("poly8", q, rs, d); top8 = a.prism(P8, z0, zt, PLASTER, top=False)
        roof(ar, [p[:2] for p in [np.array(v[:2]) for v in top8]], zt, "pyramid" if not ST["flat_roofs"] else "flat", rs, ROOF) if not ST["flat_roofs"] else a.face(top8, PLASTER)
        for zz in np.arange(fl + 2.0, zt - 1.5, 2.8): opening(a, q + n_in * rs, n_in, zz, 0.5, 1.1, "rect")
    if b["kind"].startswith("chapel"):                                         # chapel buttresses between the windows
        for side, n_ in ((0, n_in), (1, -n_in)):
            p0, p1 = (R[3], R[2]) if side == 0 else (R[0], R[1])
            for f in np.linspace(0.0, 1.0, max(3, int(L / 4.0) + 1)):
                q = p0 + (p1 - p0) * f + n_ * 0.6
                a.box(q, d, 1.2, 0.9, z0, z0 + (eave - z0) * 0.8, PLASTER)
    if kind == "lean":
        hi = eave + D * math.tan(pitch)
        back = [R[0], R[1]]; front = [R[2], R[3]]
        a.quad_strip([(*p, z_st) for p in R], [(*R[0], hi), (*R[1], hi), (*R[2], eave), (*R[3], eave)], PLASTER, closed=True)
        a.face([(*(R[0] - d * 0.4 - n_in * 0.2), hi + 0.2), (*(R[1] + d * 0.4 - n_in * 0.2), hi + 0.2),
                (*(R[2] + d * 0.4 + n_in * 0.6), eave - 0.3), (*(R[3] - d * 0.4 + n_in * 0.6), eave - 0.3)], ROOF)
        return
    a.prism(R, z_st, eave, PLASTER, top=(kind == "flat"))
    if b.get("turrets"):                                                        # turrets at the corners (Eltz): the outer corners, above the cornice
        tt = b["turrets"]; cs = [R[0], R[1]] if tt["n"] == 2 else list(R)
        for q in cs:
            u = (q - c) / max(np.linalg.norm(q - c), 1e-9)
            AE.bartizan(ar, q + u * float(tt["r"]) * 0.3, eave + 0.6, float(tt["r"]), KC, roof="flat" if ST["flat_roofs"] else "cone",
                        pitch=max(55, ST["roof_pitch"]), out=u)
    if kind == "flat":
        a.prism(R, eave, eave + 0.9, PLASTER, top=False); return
    rh = D / 2 * math.tan(pitch); m0 = (R[0] + R[3]) / 2; m1 = (R[1] + R[2]) / 2
    ov = 0.5
    e = [R[0] - d * ov - n_in * ov, R[1] + d * ov - n_in * ov, R[2] + d * ov + n_in * ov, R[3] - d * ov + n_in * ov]
    if kind == "hip":
        r0 = m0 + d * min(D / 2, L / 2 - 0.5); r1 = m1 - d * min(D / 2, L / 2 - 0.5)
    else:
        r0 = m0 - d * ov; r1 = m1 + d * ov
        a.face([(*R[3], eave), (*R[0], eave), (*m0, eave + rh)], PLASTER); a.face([(*R[1], eave), (*R[2], eave), (*m1, eave + rh)], PLASTER)
    zt = eave + rh
    ar.face([(*e[0], eave - 0.25), (*e[1], eave - 0.25), (*r1, zt), (*r0, zt)], ROOF)
    ar.face([(*e[2], eave - 0.25), (*e[3], eave - 0.25), (*r0, zt), (*r1, zt)], ROOF)
    if kind == "hip":
        ar.face([(*e[1], eave - 0.25), (*e[2], eave - 0.25), (*r1, zt)], ROOF); ar.face([(*e[3], eave - 0.25), (*e[0], eave - 0.25), (*r0, zt)], ROOF)
    if b.get("stepped_gable") and kind == "gable":                              # stepped gables
        AE.stepped_gable(a, R[3], R[0], -d, eave, zt, KC); AE.stepped_gable(a, R[1], R[2], d, eave, zt, KC)
    if b.get("chimneys") and kind in ("gable", "hip"):                          # chimneys on the ridge closer to the wall
        for f_ in np.linspace(0.22, 0.78, int(b["chimneys"])):
            q = R[0] + (R[1] - R[0]) * f_ + n_in * (D * 0.3); AE.chimney(a, q, eave, zt + 1.3, KC, axis=d)
    if b.get("dormers") and kind in ("gable", "hip") and L > 10:                 # dormers on the roof slope toward the court
        for f_ in np.linspace(0.2, 0.8, max(2, int(L / 6))):
            q = R[3] + (R[2] - R[3]) * f_ - n_in * (D * 0.2); AE.dormer(a, q, n_in, eave + D * 0.2 * math.tan(pitch) - 0.6, KC, pitch=max(40, math.degrees(pitch)))
    if b.get("roof_turret") and kind in ("gable", "hip"):                       # turret on the chapel ridge
        AE.roof_turret(a, (r0 + r1) / 2, zt - 0.3, KC)
    if b.get("apse") and not b.get("apse_out"):                                # apse — to the east (not through the wall)
        east = d if d[0] >= 0 else -d; pe = (m1 if d[0] >= 0 else m0); rr = D / 2 * 0.85
        pts = [pe + east * rr * math.sin(t) + n_in * rr * math.cos(t) for t in np.linspace(0, math.pi, 12)]
        pts = [pe + n_in * rr] + pts + [pe - n_in * rr]
        za = eave * 0.85 + z0 * 0.15
        a.quad_strip([(*p, z0) for p in pts], [(*p, za) for p in pts], PLASTER)
        ape = (*pe, za + rr * math.tan(pitch) * 0.8)
        for i in range(len(pts) - 1): ar.face([(*pts[i], za), (*pts[i + 1], za), ape], ROOF)


# ---------------------------------------------------------------- bridge and road
def bridge(a, br):
    p0, p1 = np.asarray(br["pts"][0], float), np.asarray(br["pts"][1], float); w = float(br["w"]); z = float(br["deck"])
    if br.get("kind") == "chasm":                                                # bridge over a chasm — on arches, the piers reach the bottom
        AE.arcade(a, p0, p1, z + 0.6, w, max(2, int(round(float(br["len"]) / 13.0))), lambda x, y: float(height(x, y)), KC,
                  pointed=ARCH_TOP != "round", roof=False, walls=1.1)
        return
    d = p1 - p0; L = float(np.linalg.norm(d)); d /= max(L, 1e-9)
    a.box((p0 + p1) / 2, d, w, L, z - 0.9, z, WOOD)
    n = max(1, int(L / 7))
    for k in range(1, n + 1):
        q = p0 + d * (L * k / (n + 1)); zb = float(height(*q)) - 1.0
        if z - 1.0 - zb > 0.5: a.box(q, d, w * 0.9, 1.6, zb, z - 0.9, STONE_D)


def road(a, R, w=4.0):
    R = np.asarray(R, float)[:, :2]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(R, axis=0).T))]); u = np.arange(0, s[-1], 1.5)
    P = np.column_stack([np.interp(u, s, R[:, 0]), np.interp(u, s, R[:, 1])])
    if len(P) < 2: return
    T = np.gradient(P, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9); N = np.column_stack([-T[:, 1], T[:, 0]])
    A = P + N * w / 2; B = P - N * w / 2
    a.quad_strip([(*p, float(height(*p)) + 0.12) for p in A], [(*p, float(height(*p)) + 0.12) for p in B], ROAD)


# ---------------------------------------------------------------- unique elements (castle_features.py → arch_elems.py)
def wall_frame(wi):
    w = C["walls"][wi]; P = np.asarray(w["pts"], float); closed = bool(w["closed"])
    tn, tp = dirs(P, closed)                                                     # at a ledge — the direction of the neighboring station
    N = np.column_stack([tn[:, 1], -tn[:, 0]]) + np.column_stack([tp[:, 1], -tp[:, 0]]); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    T = np.asarray(w["t"], float) if isinstance(w["t"], list) else np.full(len(P), float(w["t"]))
    return P, N, np.asarray(w["walk"], float), np.asarray(w["base"], float), T


def features(aw, at_, ar, ab):
    """Plan elements: wall stretches, wall points, gates, towers, buildings."""
    for f in FEAT:                                                               # first — where the point elements stand
        if f["kind"] in ("oriel", "garderobe", "chapel_bay", "postern", "water_gate", "blocked_gate") and f.get("wall", 99) < len(C["walls"]):
            P_, N_, _, _, T_ = wall_frame(f["wall"]); i_ = int(f["i"])
            FEAT_PTS.append((P_[i_] + N_[i_] * T_[i_] / 2, {"oriel": 2.6, "chapel_bay": 4.0, "garderobe": 1.6}.get(f["kind"], 2.0)))
    for f in FEAT:
        k = f["kind"]; sec = SECS.get(f.get("sec"))
        if "wall" in f and f["wall"] < len(C["walls"]):
            P, N, walk, base, T = wall_frame(f["wall"])
            if sec and sec.get("flip"): N = -N
        if k == "machicolation":
            ii = f["ii"]; AE.machicolation(aw, P[ii], N[ii], walk[ii], T[ii], KC, tiers=int(f.get("tiers", 3)))
        elif k == "hoarding":
            ii = f["ii"]; AE.hoarding(aw, P[ii], N[ii], walk[ii], T[ii], KC, bay=float(f.get("bay", 1.3)))
        elif k == "covered_walk":                                                # over machicolations — an overhang up to their parapet (Pierrefonds)
            ii = f["ii"]; AE.covered_walk(aw, P[ii], N[ii], walk[ii], T[ii], KC, **({"out": 1.1, "eave": 2.75} if f.get("over_mach") else {}))
        elif k == "arrow_loops":
            ii = f["ii"]; AE.arrow_loops(aw, P[ii], N[ii], walk[ii], T[ii], KC, step=float(f.get("step", 3.6)), cross=bool(f.get("cross")))
        elif k == "putlogs":
            ii = f["ii"]; AE.putlogs(aw, P[ii], N[ii], base[ii], walk[ii], T[ii], KC, rows=int(f.get("rows", 3)), step=float(f.get("step", 2.0)))
        elif k == "buttresses":
            ii = f["ii"]; AE.buttresses(aw, P[ii], N[ii], base[ii], walk[ii], T[ii], KC, spacing=float(f.get("spacing", 6.0)))
        elif k == "pinnacle_buttresses":                                         # fantasy Gothic: buttresses with pinnacles
            ii = f["ii"]; AE.pinnacle_buttresses(aw, P[ii], N[ii], base[ii], walk[ii], T[ii], KC, spacing=float(f.get("spacing", 8.0)), parapet=MERLON["parapet"])
        elif k == "sky_bridge":                                                  # bridge between tower tops
            AE.skybridge(at_, f["p0"], f["p1"], float(f["z"]), float(f["w"]), KC, pointed=ARCH_TOP != "round")
        elif k == "raised":                                                      # filled-in merlons at the former height
            ii = f["ii"]; old = np.asarray(f["old"], float); Po = P[ii] + N[ii] * (T[ii] / 2 + 0.03)[:, None]
            AE.merlon_row(aw, Po, N[ii], old + MERLON["parapet"], KC, th=0.08, col=STONE2)
            a0 = [(*p, z + MERLON["parapet"] - 0.02) for p, z in zip(Po + N[ii] * 0.02, old)]
            a1 = [(*p, z + MERLON["parapet"] + 0.12) for p, z in zip(Po + N[ii] * 0.02, old)]
            aw.quad_strip(a0, a1, STONE_D)
        elif k == "palace_windows":
            ii = f["ii"]
            if len(ii) < 3: continue
            b = C["buildings"][f["building"]]; sh = float(ST["storey"]); fl = float(b["floor"])
            zo = height(P[ii[len(ii) // 2]][0] + N[ii[len(ii) // 2]][0] * 4, P[ii[len(ii) // 2]][1] + N[ii[len(ii) // 2]][1] * 4)
            wk = float(np.min(walk[ii]))
            for st_ in range(1, int(b["storeys"]) + 1):
                z0 = fl + st_ * sh + 0.7
                if z0 < zo + 3.0: continue
                if z0 + 2.4 < wk - 0.8:                                         # windows in the curtain
                    pts, Tg, u = AE.along(P[ii] + N[ii] * (T[ii] / 2)[:, None], 5.5, margin=2.0)
                    for q, tg in zip(pts, Tg):
                        nn = np.array([tg[1], -tg[0]]); top_ = "pointed" if (f.get("tracery") and ARCH_TOP != "round") else ("round" if ARCH_TOP == "round" else "rect")
                        if win_blocked(q, nn, z0): continue
                        AE.opening(aw, q, nn, z0, 1.1, min(2.2, sh - 1.4), KC, top_)
                        if f.get("tracery") and top_ == "pointed": AE.tracery(aw, q, nn, z0, 1.35, min(2.8, sh - 1.1), KC)
                elif z0 > wk + 0.3 and z0 + 2.0 < float(b["eave"]):                # above the wall — windows of the palas's back wall
                    R = np.asarray(b["poly"], float); nn = -np.array([-np.asarray(b["axis"])[1], np.asarray(b["axis"])[0]])
                    pts, Tg, u = AE.along(np.array([R[0], R[1]]), 3.8, margin=1.5)
                    for q, tg in zip(pts, Tg):
                        top_ = "pointed" if (f.get("tracery") and ARCH_TOP != "round") else ("round" if ARCH_TOP == "round" else "rect")
                        AE.opening(ab, q, nn, z0, 1.35, min(2.6, sh - 1.1), KC, top_)
                        if f.get("tracery") and top_ == "pointed": AE.tracery(ab, q, nn, z0, 1.35, min(2.6, sh - 1.1), KC)
        elif k in ("oriel", "garderobe", "postern", "water_gate", "blocked_gate", "stairs", "chapel_bay"):
            i = int(f["i"]); p = P[i]; n = N[i]; t = float(T[i]); po = p + n * t / 2
            zo = float(height(*(p + n * (t / 2 + 1.0)))); zi = float(height(*(p - n * (t / 2 + 1.5)))); wk = float(walk[i])
            if k == "oriel":
                z0 = max(float(f["z0"]), zo + 3.5); z1 = min(float(f["z1"]), z0 + float(ST["storey"]) * 1.7)
                if z1 - z0 > 2.2: AE.oriel(aw, po, n, z0, z1, KC, faces=int(f.get("faces", 3)), w=float(f.get("w", 3.0)), pitch=max(40, ST["roof_pitch"]))
            elif k == "garderobe":
                z = min(float(f["z"]), wk - 3.0); z = max(z, zo + 3.0)
                if z < wk - 2.6: AE.garderobe(aw, p, n, z, t, KC)
            elif k == "postern":
                AE.postern(aw, p, n, zo + 0.05, t, KC)
            elif k == "water_gate":
                lv = float(np.nanmax([LEVEL[min(LEVEL.shape[0] - 1, max(0, int(round(((p + n * 4)[1] - NY0) / NRES)))),
                                            min(LEVEL.shape[1] - 1, max(0, int(round(((p + n * 4)[0] - NX0) / NRES))))], zo]))
                AE.water_gate(aw, p, n, lv + 0.1, t, KC)
            elif k == "blocked_gate":
                AE.blocked_gate(aw, p, n, max(zo, zi - 0.5), t, KC)
            elif k == "stairs":
                tg = np.array([-n[1], n[0]]) * (1 if f.get("left") else -1)
                if 1.5 <= wk - zi <= 10.5: AE.stairs(aw, p, tg, -n, zi, wk, t, KC)
            elif k == "chapel_bay":
                b = C["buildings"][f["building"]]; fl = float(b["floor"]); ev = float(b["eave"])
                corb = zo < fl - 2.5
                AE.chapel_bay(aw, po, n, (fl + 0.6) if corb else zo - 0.4, max(ev, fl + 5.0), KC, r=min(3.4, float(b["D"]) * 0.42), corbelled=corb,
                              pitch=max(45, ST["roof_pitch"]))
        elif k in ("portcullis", "breteche", "drawbridge"):
            g = C["gates"][f["gate"]]; gp = np.asarray(g["xy"], float); d = np.asarray(g["dir"], float); o = np.asarray(g["out"], float)
            zg = float(g["z"]); wd = float(g["w"]); hg = float(g["h"])
            if k == "portcullis": AE.portcullis(aw, gp + o * 0.4, d, zg, wd, hg, KC)
            elif k == "breteche": AE.breteche(aw, gp + o * (float(ST["wall_t"]) * 0.62), o, float(g["walk"]) - 0.9, KC, w=wd + 0.6)
            elif k == "drawbridge" and C["bridges"]:
                br = C["bridges"][0]; AE.drawbridge(aw, gp + o * (float(ST["wall_t"]) * 0.62), d, o, float(br["deck"]), wd, 4.5, KC,
                                                   beams=bool(f.get("beams")), zgate_top=zg + hg)
        elif k == "dansker":                                                     # projecting tower on an arcade (Danzker)
            p0, p1 = np.asarray(f["p0"], float), np.asarray(f["p1"], float); dd = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            side = float(f["side"]); AE.arcade(aw, p0, p1 - dd * side / 2, float(f["deck"]), float(f["w"]), int(f["arches"]),
                                              lambda x, y: float(height(x, y)), KC, pointed=ARCH_TOP != "round")
            sq = footprint("square", p1, side / 2, [dd[1], -dd[0]]); top = float(f["top"])
            at_.prism(sq, float(f["base"]), top, STONE, top=True)
            for zz in np.arange(float(f["deck"]) + 0.8, top - 1.2, 3.4):
                for v in (dd, np.array([-dd[1], dd[0]]), -np.array([-dd[1], dd[0]])): opening(at_, p1 + v * side / 2, v, zz, 0.9, 1.6, ARCH_TOP)
            if f.get("roof") == "flat":
                Q = np.vstack([sq, sq[:1]]); cc = sq.mean(0); Nn = np.asarray([(q - cc) / max(np.linalg.norm(q - cc), 1e-9) for q in Q])
                merlons(at_, Q, Nn, np.full(len(Q), top), ST["merlon"], STONE)
            else: roof(ar, list(sq), top, "pyramid", side / 2, ROOF, h=side * 0.42)
        elif k == "bartizans":
            if f.get("keep") and C.get("keep"):
                kp = C["keep"]; P4 = footprint(kp["plan"], kp["xy"], float(kp["r"]), C["threat"]); cc = np.asarray(kp["xy"], float); top = float(kp["top"])
            elif "tower" in f:
                tw = C["towers"][f["tower"]]; P4 = footprint(tw["plan"], tw["xy"], float(tw["r"]), tw.get("dir")); cc = np.asarray(tw["xy"], float); top = float(tw["top"])
            else: continue
            rb = float(f.get("r", 1.4))
            for q in P4[:: max(1, len(P4) // 4)][:4]:
                u = (q - cc) / max(np.linalg.norm(q - cc), 1e-9)
                AE.bartizan(at_, q + u * rb * 0.35, top, rb, KC, roof="flat" if ST["flat_roofs"] else "cone", pitch=max(50, ST["roof_pitch"]), out=u)


# ---------------------------------------------------------------- build
def load(folder, kw=None):
    """Read the castle plan from a folder (castle.json, relief.npz, location.json if present) and fill the module
    globals: C, Z, HN/HF/LEVEL with their origins and steps, NB, ST, COL, K, the colors (STONE…, KC), FEAT,
    SECS, TWO_LEVEL, ARCH_TOP, ROCK, KN, MERLON. kw — the keys of the line (views, res, look, exposure, hide,
    blend …), as after "--" on the command line. Does not touch the scene."""
    global OUT, KW, C, Z, LOC, HN, HF, LEVEL, NX0, NY0, NRES, FX0, FY0, FRES, NB, ST, COL, K, T0
    global STONE, STONE_D, ROOF, PLASTER, STONE2, KC, FEAT, SECS, TWO_LEVEL, ARCH_TOP, ROCK, KN
    global WALLS_G
    OUT = os.path.abspath(folder); KW = dict(kw or {})
    C = json.load(open(os.path.join(OUT, "castle.json")))
    Z = np.load(os.path.join(OUT, "relief.npz"))
    LOC = json.load(open(os.path.join(OUT, "location.json"))) if os.path.exists(os.path.join(OUT, "location.json")) else None
    HN = Z["near"].astype(np.float64); NX0, NY0, NRES = (float(v) for v in Z["near_o"]); LEVEL = Z["level"].astype(np.float64)
    HF = Z["far"].astype(np.float64); FX0, FY0, FRES = (float(v) for v in Z["far_o"])
    ST = C["style"]; COL = C["colors"]; K = ST["k"]
    NB = (NX0, NY0, NX0 + (HN.shape[1] - 1) * NRES, NY0 + (HN.shape[0] - 1) * NRES)
    STONE = tuple(COL["stone"]); STONE_D = shade(STONE, 0.82); ROOF = tuple(COL["roof"]); PLASTER = tuple(COL["plaster"])
    STONE2 = shade(STONE, 1.12)
    KC = {"STONE": STONE, "STONE_D": STONE_D, "STONE2": STONE2, "ROOF": ROOF, "PLASTER": PLASTER, "WOOD": (0.30, 0.21, 0.14), "HOLE": HOLE, "IRON": IRON}
    FEAT = C.get("features", []); SECS = {s_["id"]: s_ for s_ in C.get("sections", [])}
    TWO_LEVEL = any(f.get("over_mach") for f in FEAT)                            # two-tier defense (Pierrefonds) at the castle
    ARCH_TOP = {"german": "round", "french": "pointed", "english": "pointed", "crusader": "pointed", "moorish": "horseshoe",
                "kremlin": "round", "imperial_scifi": "pointed", "hell": "pointed", "gothic": "pointed", "vampire": "pointed",
                "sorcerer": "pointed"}.get(C["spec"]["culture"], "round")
    if C["spec"]["era"] == "romanesque": ARCH_TOP = "round" if ARCH_TOP != "horseshoe" else ARCH_TOP
    ROCK = tuple(COL.get("rock", (0.47, 0.45, 0.42))); KN = C.get("knobs", {})
    MERLON.clear(); MERLON.update(_MERLON0); MERLON.update(ST.get("crenel") or {})  # plan and build use the same dimensions
    WALLS_G = None                                                               # line-of-sight cache — depends on the plan; build resets the rest
    T0 = time.time()
    return C


def build(sc=None, with_ground=True, fresh=True):
    """Build the castle from the plan filled by load(). fresh=True — an empty scene first (read_factory_settings,
    sc is ignored); fresh=False — into the scene sc (the current one by default), collections castle and
    terrain. with_ground=False — without ground, ground_far and water (the ground already exists, e.g. from
    bl_relief); the rock (rock_plinth, rock_band, rock_crag) is always built.
    → (c, half, objs): the center, the half-size and a dict of objects — walls, towers, roofs, buildings, rocks (a list), and
    also ground, ground_far, water if created."""
    global ARK, ARB, ARC
    if fresh: bpy.ops.wm.read_factory_settings(use_empty=True)
    _SC[0] = sc if (sc is not None and not fresh) else bpy.context.scene
    ARK = Acc(); ARB = Acc(); ARC = Acc(); STAIR_T.clear(); FEAT_PTS.clear(); NCRAG[0] = 0
    objs = {"rocks": []}
    c = np.asarray(C["site"]["centre"], float)
    allp = np.vstack([np.asarray(w["poly"]) for w in C["wards"]])
    half = float(np.abs(allp - c).max()) + 90 * K ** 0.5
    if with_ground:
        objs["ground"], objs["water"] = ground(c, half, NRES if half <= 200 else (2 * NRES if half < 300 else 4 * NRES))
        objs["ground_far"] = ground(c, half * 3.2, 8.0, hole=half - 4, name="ground_far")
        if objs["water"] is None: del objs["water"]
    aw, at_, ar, ab = Acc(), Acc(), Acc(), Acc()
    gaps = [(np.asarray(g["xy"]), (float(g["r"]) * 1.05 if g.get("kind") == "tower" else float(g["w"]) / 2 + 1.4)) for g in C["gates"]]
    for wi, w in enumerate(C["walls"]):
        wall(aw, w, gaps, wi)
        if w.get("rock_top"):                                                    # wall on the edge: rock under the masonry
            P_, N_, _, _, T_ = wall_frame(wi); rt = list(w["rock_top"])
            for g, half_g in gaps:
                for j in np.flatnonzero(np.hypot(P_[:, 0] - g[0], P_[:, 1] - g[1]) <= half_g + 1.0): rt[j] = None
            rock_band(ARB, P_, N_, T_, rt, seed=float(wi) * 1.37)
        if w["kind"] == "retaining" and w.get("stair") is not None:            # stair from the lower terrace to the upper one
            P_, N_, wk_, bs_, T_ = wall_frame(wi); j = int(w["stair"]); n_ = N_[j]; tg = np.array([-n_[1], n_[0]])
            zl = float(height(*(P_[j] + n_ * (T_[j] / 2 + 1.5)))); AE.stairs(aw, P_[j] + tg * 2.0, tg, n_, zl, float(wk_[j]), float(T_[j]), KC, w=1.6)
    for g, (gp, half_g) in zip(C["gates"], gaps):
        tw = max((max(w["t"]) if isinstance(w["t"], list) else w["t"]) for w in C["walls"])
        gate_block(aw, g, float(ST["wall_t"]), half_g)
    for t in C["towers"]:
        if t.get("stages"): staged_tower(at_, ar, t)                             # tower by tiers (fantasy)
        else: tower(at_, ar, t)
    if C.get("keep"):
        kp = dict(C["keep"], dir=C["threat"])
        if kp.get("stages"): staged_tower(at_, ar, kp, is_keep=True)
        else: tower(at_, ar, kp, is_keep=True); keep_parts(at_, ar, kp)
        if kp.get("butterfass"):
            bf = kp["butterfass"]; AE.butterfass(at_, kp["xy"], float(kp["r"]) * bf["k"], float(kp["top"]), float(bf["h"]), KC,
                                                 roof=kp.get("roof", "cone") if kp.get("roof") != "flat" else "cone", pitch=ST["roof_pitch"],
                                                 plan="round" if kp["plan"] == "round" else "square")
    for b in C["buildings"]: building(ab, ar, b)
    for br in C["bridges"]: bridge(aw, br)
    features(aw, at_, ar, ab)
    for w in C["wards"]:                                                         # ravelins in the moat — low triangular works
        for rv in w.get("ravelins", []):
            R = np.asarray(rv, float); zg = float(np.max(height(R[:, 0], R[:, 1]))); zb = float(np.min(height(R[:, 0], R[:, 1]))) - 0.5
            cc = R.mean(0); aw.prism(R, zb, zg + 2.5, STONE, scale_top=0.92, centre=cc)
    if C.get("road"): road(ab, C["road"], 4.0 * K ** 0.5)
    NCRAG[0] = crags(ARC, int(C.get("crags", round(float(KN.get("rock", 0.5)) * 30))), seed=int(abs(c[0] * 7 + c[1] * 3)) % 9973)
    for a, name in ((aw, "walls"), (at_, "towers"), (ar, "roofs"), (ab, "buildings"), (ARK, "rock_plinth"), (ARB, "rock_band"), (ARC, "rock_crag")):
        if a.f:
            ob = a.obj(name, coll("castle") if not name.startswith("rock") else coll("terrain"))
            if name.startswith("rock"): objs["rocks"].append(ob)
            else: objs[name] = ob
    return c, half, objs


SP_LAST = None                                                                  # the face markup result of the last build_castle(tag=True)


def build_castle(folder, sc=None, with_ground=False, tag=True):
    """load(folder) + build(sc, with_ground, fresh=False) + face markup by the space scheme (bl_space.tag, if the
    plan has spaces).
    → (objs, C): the dict of build objects and the plan. The result of the markup and its checks is SP_LAST."""
    global SP_LAST
    load(folder)
    c, half, objs = build(sc, with_ground, fresh=False); SP_LAST = None
    if tag and C.get("spaces"):
        from components.buildings.space import bl_space as BS
        from components.buildings.space.space import Space
        S_ = Space.from_dict(C["spaces"])
        obs_ = [o for k_, v_ in objs.items() if k_ not in ("ground", "ground_far") for o in (v_ if isinstance(v_, list) else [v_])]
        SP_LAST = {"faces": BS.tag(S_, obs_), "check": BS.verify(S_)}
    return objs, C


def cameras(c, half):
    sc = bpy.context.scene; cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)
    con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    allp = np.vstack([np.asarray(w["poly"]) for w in C["wards"]]); R = float(np.hypot(*(allp - c).T).max())
    zc = float(height(*c)); top = max([t["top"] for t in C["towers"]] + ([C["keep"]["top"]] if C.get("keep") else [zc + 20]))
    g = C["gates"][0]; o = np.asarray(g["out"], float)
    app = o if not C.get("road") else (np.asarray(C["road"][max(0, len(C["road"]) - 30)][:2]) - c)
    app = app / max(np.linalg.norm(app), 1e-9)
    side = np.array([-app[1], app[0]])
    views = {}
    d = max(R * 2.6 + 40, (top - zc) * 2.6)
    views["aerial"] = dict(loc=(*(c + (app * 0.85 + side * 0.5) * d), zc + d * 0.55), tgt=(*c, (zc + top) / 2 - 4), lens=40)
    views["back"] = dict(loc=(*(c + (-app * 0.85 - side * 0.4) * d), zc + d * 0.45), tgt=(*c, (zc + top) / 2 - 4), lens=40)
    gp = np.asarray(g["xy"], float); gz = float(g["z"]) + 3.0
    WL = [(np.asarray(w["pts"], float), np.asarray(w["walk"], float)) for w in C["walls"]]
    TWR = [(np.asarray(t["xy"], float), float(t["r"]), float(t["top"])) for t in C["towers"]]
    def clear_view(e):
        """The view of the gate is not blocked by the ground, walls or towers; the camera is not right up against
        buildings."""
        q = np.array([*gp, gz])
        for P, wk in WL:
            if np.hypot(P[:, 0] - e[0], P[:, 1] - e[1]).min() < 18: return False
        for c_, r_, _ in TWR:
            if np.hypot(*(c_ - e[:2])) < r_ + 14: return False
        for t_ in np.linspace(0.04, 0.9, 30):
            m = e + (q - e) * t_
            if float(height(m[0], m[1])) > m[2] - 0.4: return False
            if np.hypot(*(m[:2] - gp)) < 12: continue
            for P, wk in WL:
                d = np.hypot(P[:, 0] - m[0], P[:, 1] - m[1]); i = int(np.argmin(d))
                if d[i] < 2.5 and wk[i] + 1.5 > m[2]: return False
            for c_, r_, tp in TWR:
                if np.hypot(*(c_ - m[:2])) < r_ and tp > m[2]: return False
        return True
    cands = []
    if C.get("road"):
        for p in np.asarray(C["road"])[::-1]:
            dd = float(np.hypot(*(p[:2] - gp)))
            if 45 <= dd <= 120: cands.append(np.array([p[0], p[1], float(height(p[0], p[1])) + 3.5]))
    o = np.asarray(g["out"], float)
    for ang in (0, 0.35, -0.35, 0.7, -0.7):
        for dd in (55, 75, 95):
            v = np.array([o[0] * math.cos(ang) - o[1] * math.sin(ang), o[0] * math.sin(ang) + o[1] * math.cos(ang)])
            q = gp + v * dd; cands.append(np.array([q[0], q[1], float(height(*q)) + 3.5]))
    best = next((e for e in cands if clear_view(e)), cands[-1] if cands else None)
    if best is not None: views["road"] = dict(loc=tuple(best), tgt=(*gp, gz + 4), lens=30)
    views["top"] = dict(loc=(*c, top + 400), tgt=(*c, zc), ortho=R * 2.4 + 30)
    if C.get("spaces"):                                                          # section: inside — warm, court — green, outside — blue
        zw = min(float(w["level"]) for w in C["wards"]); dk = 2.2 * R + 20
        views["cutaway"] = dict(loc=(*(c + (app * 0.35 + side * 0.62) * dk), zw + 0.75 * dk), tgt=(*c, zw), lens=32, side=True, cut=zw + 3.0)
    return cam, tgt, views


FEAT_RU = {"machicolation": "machicolations", "hoarding": "hoarding gallery", "covered_walk": "covered walk", "zwinger": "zwinger",
           "garderobe": "garderobe", "palace_windows": "residential windows in the curtain", "oriel": "oriel", "chapel_bay": "apse through the wall",
           "postern": "wicket", "water_gate": "water gate", "blocked_gate": "bricked-up gate", "buttresses": "buttresses",
           "stairs": "stair to the wall", "stepped_walk": "stepped walkway", "raised": "superstructure, filled-in merlons",
           "butterfass": "'butter churn' on the bergfried", "prow": "tower-wedge", "bartizans": "bartizans", "barbican": "barbican",
           "drawbridge": "drawbridge", "portcullis": "portcullis", "breteche": "breteche", "arrow_loops": "loopholes in the curtain",
           "putlogs": "putlog holes", "dansker": "projecting tower on an arcade", "pinnacle_buttresses": "buttresses with pinnacles",
           "sky_bridge": "bridge between towers", "landmark_spire": "main tower in tiers"}
FEAT_ORDER = ["dansker", "chapel_bay", "oriel", "butterfass", "barbican", "garderobe", "machicolation", "hoarding", "bartizans", "drawbridge",
              "zwinger", "raised", "blocked_gate", "postern", "water_gate", "prow", "palace_windows", "covered_walk", "buttresses", "stairs", "breteche"]


WALLS_G = None


STAIR_T = []                                                                    # stair towers of buildings (for window occlusion)
FEAT_PTS = []                                                                   # oriels, garderobes, apses, wicket gates on the walls (for the curtain windows)


def win_blocked(q, n, z, self_poly=None, self_xy=None):
    """The opening is blocked: right behind it there is another building, a tower, the main tower, a wall (below
    the walkway) or a stair tower; self_xy — the own tower (its tiers are narrower than the base — do not count
    itself)."""
    q = np.asarray(q, float); n = np.asarray(n, float); n = n / max(np.linalg.norm(n), 1e-9); m = q + n * 0.7
    for b in C["buildings"]:
        R = np.asarray(b["poly"], float)
        if self_poly is not None and np.allclose(R, self_poly): continue
        if z < float(b["eave"]) + 2.0 and AE_inside(R, m[0], m[1]): return True
    for t in C["towers"] + ([C["keep"]] if C.get("keep") else []):
        if self_xy is not None and np.allclose(np.asarray(t["xy"], float), self_xy): continue
        if np.hypot(*(np.asarray(t["xy"]) - m)) < float(t["r"]) + 0.4 and z < float(t["top"]): return True
    for w in C["walls"]:
        P = np.asarray(w["pts"], float); d = np.hypot(P[:, 0] - m[0], P[:, 1] - m[1]); i = int(np.argmin(d))
        tw = (w["t"][i] if isinstance(w["t"], list) else w["t"]) / 2 + 0.3
        if d[i] < tw and z < float(w["walk"][i]) + 1.2: return True
    for c_, r_ in STAIR_T:
        if np.hypot(*(c_ - m)) < r_ + 0.8: return True
    for c_, r_ in FEAT_PTS:
        if np.hypot(*(c_ - q)) < r_: return True
    return False


def los_ok(e, q):
    """Is q visible from e: the ground, walls (walkway + parapet) and towers do not block it; the camera is not
    inside a wall, tower or building."""
    global WALLS_G
    if WALLS_G is None:
        WALLS_G = [(np.asarray(w["pts"], float), np.asarray(w["walk"], float) + 2.4,
                    (np.asarray(w["t"], float) if isinstance(w["t"], list) else np.full(len(w["pts"]), float(w["t"]))) / 2 + 0.4) for w in C["walls"]]
    TW_ = [(np.asarray(t["xy"], float), float(t["r"]) + 0.4, float(t["top"]) + 3.0) for t in C["towers"]]
    if C.get("keep"): TW_.append((np.asarray(C["keep"]["xy"], float), float(C["keep"]["r"]) + 0.4, float(C["keep"]["top"]) + 8.0))
    BL_ = [(np.asarray(b["poly"], float), float(b["eave"]) + float(b["D"]) * (1.0 if b["roof"] == "lean" else 0.5)
            * math.tan(math.radians(max(float(b.get("pitch", 45)), 20))) + 1.0) for b in C["buildings"]]
    e = np.asarray(e, float); q = np.asarray(q, float)
    if float(height(e[0], e[1])) > e[2] - 0.8: return False
    for R_, zt in BL_:                                                           # the camera is not right up against a roof (4 m from the house, unless it is higher)
        c_ = R_.mean(0); Rx = c_ + (R_ - c_) * (1 + 4.0 / max(float(np.linalg.norm(R_ - c_, axis=1).min()) * 2, 1.0))
        if e[2] < zt + 2.0 and AE_inside(Rx, e[0], e[1]): return False
    for P, zt, hw in WALLS_G:
        d = np.hypot(P[:, 0] - e[0], P[:, 1] - e[1]); i = int(np.argmin(d))
        if d[i] < hw[i] + 1.2 and e[2] < zt[i] + 0.5: return False
    for c_, r_, zt in TW_:
        if np.hypot(*(c_ - e[:2])) < r_ + 1.2 and e[2] < zt: return False
    L = float(np.linalg.norm(q - e)); n_ = max(12, int(L / 0.8))
    for t_ in np.linspace(0.03, 1.0 - min(0.5, 3.5 / max(L, 1e-6)), n_):
        m = e + (q - e) * t_
        if float(height(m[0], m[1])) > m[2] - 0.2: return False
        for P, zt, hw in WALLS_G:
            d = np.hypot(P[:, 0] - m[0], P[:, 1] - m[1]); i = int(np.argmin(d))
            if d[i] < hw[i] and m[2] < zt[i]: return False
        for c_, r_, zt in TW_:
            if np.hypot(*(c_ - m[:2])) < r_ and m[2] < zt: return False
        for R_, zt in BL_:
            if m[2] < zt and AE_inside(R_, m[0], m[1]): return False
    return True


def inside_v(P, x, y):
    """Points inside a polygon, vectorized (like castle_plan.inside)."""
    x = np.asarray(x, float); y = np.asarray(y, float); r = np.zeros(np.broadcast(x, y).shape, bool)
    for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, 0)):
        r ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1)
    return r


def los_why(e, q):
    """Why it is not visible (view debugging): ground at the camera, wall, tower, ground/wall/tower/house along the
    ray."""
    e = np.asarray(e, float); q = np.asarray(q, float)
    if float(height(e[0], e[1])) > e[2] - 0.8: return "camera in the ground"
    L = float(np.linalg.norm(q - e)); TW_ = [(np.asarray(t["xy"], float), float(t["r"]) + 0.4, float(t["top"]) + 3.0) for t in C["towers"]]
    for t_ in np.linspace(0.03, 1.0 - min(0.5, 3.5 / max(L, 1e-6)), max(12, int(L / 0.8))):
        m = e + (q - e) * t_
        if float(height(m[0], m[1])) > m[2] - 0.2: return f"ground {t_:.2f}"
        for P, zt, hw in WALLS_G:
            d = np.hypot(P[:, 0] - m[0], P[:, 1] - m[1]); i = int(np.argmin(d))
            if d[i] < hw[i] and m[2] < zt[i]: return f"wall {t_:.2f}"
        for c_, r_, zt in TW_:
            if np.hypot(*(c_ - m[:2])) < r_ and m[2] < zt: return f"tower {t_:.2f}"
        for b in C["buildings"]:
            if m[2] < float(b["eave"]) + float(b["D"]) / 2 * 1.5 + 1.0 and AE_inside(np.asarray(b["poly"], float), m[0], m[1]): return f"house {t_:.2f}"
    return "at the camera: wall/tower"


def AE_inside(P, x, y):
    c = False
    for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, 0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1: c = not c
    return c


def feature_views(limit=8):
    """Cameras "element close-up": one per element kind (the order is landmarks, function, defense, history)."""
    out = []; seen = set()
    for kind in FEAT_ORDER:
        f = next((x for x in FEAT if x["kind"] == kind), None)
        if f is None or kind in seen: continue
        seen.add(kind); sec = SECS.get(f.get("sec")); tgt = loc = None; lens = 32
        if kind == "dansker":                                                     # arcade from the side: the middle of the span, both sides
            p0, p1 = np.asarray(f["p0"], float), np.asarray(f["p1"], float); mid = (p0 + p1) / 2; dd = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            nr = np.array([-dd[1], dd[0]]); zt = (float(f["deck"]) + float(f["base"])) / 2; tgt = np.array([*mid, zt]); L_ = float(f["len"])
            cands = [np.array([*(mid + nr * s_ * dd_ - dd * L_ * 0.2), zt + h_]) for dd_ in (L_ * 1.3 + 15, L_ * 1.8 + 20) for s_ in (1, -1) for h_ in (4, 12, -2)]
        elif "wall" in f and f["wall"] < len(C["walls"]):
            P, N, walk, base, T = wall_frame(f["wall"])
            if sec and sec.get("flip"): N = -N
            if "i" in f: i = int(f["i"])
            elif "ii" in f:                                                      # the point of the stretch farthest from the gate (the middle part)
                mid = f["ii"][len(f["ii"]) // 4: max(len(f["ii"]) // 4 + 1, 3 * len(f["ii"]) // 4)]
                GP = np.array([g["xy"] for g in C["gates"]], float)
                i = int(max(mid, key=lambda j: float(np.min(np.hypot(*(GP - P[j]).T)))))
            else: continue
            p, n, t, wk = P[i], N[i], float(T[i]), float(walk[i]); tg = np.array([-n[1], n[0]])
            zo = float(height(*(p + n * (t / 2 + 3)))); zi = float(height(*(p - n * (t / 2 + 3))))
            if kind == "stairs":
                tgt = np.array([*p, (zi + wk) / 2]); cands = [np.array([*(p - n * dd + tg * s_), zi + 1.7 + dd * 0.25]) for dd in (14, 18) for s_ in (6, -6)]
            else:
                zt = {"garderobe": float(f.get("z", wk - 4)) + 1.0, "postern": zo + 1.6, "water_gate": zo + 1.6, "blocked_gate": zo + 2.5,
                      "oriel": float(f.get("z0", wk - 6)) + 2.0, "chapel_bay": wk - 4.0, "buttresses": (float(base[i]) + wk) / 2,
                      "zwinger": zo + 2.5}.get(kind, wk - 0.5)
                tgt = np.array([*(p + n * t / 2), zt]); dist = 22.0 if kind not in ("zwinger", "buttresses", "raised", "covered_walk") else 30.0
                cands = [np.array([*(p + n * dd + tg * s_), max(zo + 2.0, zt + h_)]) for dd in (dist, dist * 0.75, dist * 1.4, 12.0, 9.0, 40.0, 55.0)
                         for s_ in (dist * 0.35, -dist * 0.35, 0) for h_ in (2.0, 6.0, -4.0, 12.0, 20.0, 30.0)]
        elif f.get("keep") and C.get("keep"):
            kp = C["keep"]; c_ = np.asarray(kp["xy"], float); v = np.asarray(C["threat"], float)
            ztop = float(kp["top"]) + (float(kp["butterfass"]["h"]) * 0.6 if kp.get("butterfass") else -1.0)
            tgt = np.array([*c_, ztop])
            cands = [np.array([*(c_ + (v * np.cos(a_) + np.array([-v[1], v[0]]) * np.sin(a_)) * dd), ztop - 3.0])
                     for dd in (40, 55, 70) for a_ in (0.3, -0.3, 0.9, -0.9, 1.6)]
        elif "tower" in f:
            tw = C["towers"][f["tower"]]; c_ = np.asarray(tw["xy"], float); d_ = tw.get("dir") or [1, 0]; v = np.array([d_[1], -d_[0]])
            tgt = np.array([*c_, float(tw["top"]) - 1.0])
            cands = [np.array([*(c_ + (v * np.cos(a_) + np.array([-v[1], v[0]]) * np.sin(a_)) * dd), float(tw["top"])]) for dd in (28, 40) for a_ in (0.4, -0.4, 0.0)]
        elif "gate" in f:
            g = C["gates"][f["gate"]]; gp = np.asarray(g["xy"], float); o = np.asarray(g["out"], float); d_ = np.asarray(g["dir"], float)
            tgt = np.array([*gp, float(g["z"]) + float(g["h"]) * 0.8])
            cands = [np.array([*(gp + o * dd + d_ * s_), float(g["z"]) + h_]) for dd in (24, 32, 18) for s_ in (5, -5, 0) for h_ in (5, 9, 2)]
        else:
            continue
        e = next((c_ for c_ in cands if los_ok(c_, tgt)), None)
        if e is None: continue                                                   # not visible from anywhere — the element has no close-up view
        out.append((kind, tuple(e), tuple(tgt), lens, f.get("why", "")))
        if len(out) >= limit: break
    return out


def complex_views(limit=8):
    """Cameras "complex close-up": rock plinth, stairstep wall, wall on the rock edge, terrace, house as a wall,
    tower house, projecting tower — one each, from where it is visible."""
    out = []
    def ring_c(c_, tgt, dists, hs, u0=None, spread=(0.0, 0.5, -0.5, 1.0, -1.0)):
        u0 = np.asarray(u0 if u0 is not None else (1.0, 0.0), float); u0 /= max(np.linalg.norm(u0), 1e-9)
        return [np.array([*(c_ + np.array([u0[0] * math.cos(a_) - u0[1] * math.sin(a_), u0[0] * math.sin(a_) + u0[1] * math.cos(a_)]) * d_), tgt[2] + h_])
                for d_ in dists for a_ in spread for h_ in hs]
    def pick(name, tgt, cands, why):
        e = next((c_ for c_ in cands if los_ok(c_, tgt)), None)
        if e is not None:
            out.append((name, tuple(e), tuple(tgt), 32, why))
            if KW.get("debug"): print("kind", name, np.round(e, 1).tolist(), "→", np.round(tgt, 1).tolist())
        elif KW.get("debug"): print("no view", name, len(cands), [los_why(c_, tgt) for c_ in cands[:12]])
    WP = [np.asarray(w["poly"], float) for w in C["wards"]]
    def outside(cs): return [c_ for c_ in cs if not any(AE_inside(P_, c_[0], c_[1]) for P_ in WP)]   # outside the castle
    def low_dir(t):
        c_ = np.asarray(t["xy"], float); ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        R_ = c_ + (float(t["r"]) + 3) * np.column_stack([np.cos(ang), np.sin(ang)]); u = R_[int(np.argmin(height(R_[:, 0], R_[:, 1])))] - c_
        u /= max(np.linalg.norm(u), 1e-9); return u, not any(AE_inside(P_, *(c_ + u * (float(t["r"]) + 6))) for P_ in WP)
    pl = sorted([t for t in C["towers"] + ([C["keep"]] if C.get("keep") else []) if t.get("rock")],
                key=lambda q: -(q["rock"]["drop"] * (2.0 if low_dir(q)[1] else 1.0)))          # rock — from the slope side, outward
    for t in pl[:6]:
        if any(o[0] == "rock" for o in out): break
        c_ = np.asarray(t["xy"], float); u, outw = low_dir(t); q_ = c_ + u * (float(t["r"]) + 1.2)   # target — on the rock face
        tgt = np.array([*q_, (float(t["rock"]["z0"]) + float(t["rock"]["z1"])) / 2 + 2.0])
        cs = ring_c(q_, tgt, (30, 42, 22, 60), (3, 9, 16, -2), u)
        if outw: pick("rock", tgt, outside(cs), f"rock plinth: drop {t['rock']['drop']} m, slope {t['rock']['slope']}°")
    stw = [(wi, w) for wi, w in enumerate(C["walls"]) if w.get("steps")]
    best = None                                                                   # the whole stairstep: the stretch between towers with the largest
    for wi, w in stw:                                                             # number of ledges, the camera outside across the stretch
        P, N, wk, bs, T = wall_frame(wi); s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        TWp = [(np.asarray(t_["xy"], float), float(t_["r"])) for t_ in C["towers"]]
        near = np.array([min([np.hypot(*(P[j] - c_)) - r_ for c_, r_ in TWp] or [99.0]) for j in range(len(P))]) < 1.0
        j = 0
        while j < len(P):
            if near[j]: j += 1; continue
            k = j
            while k < len(P) and not near[k]: k += 1
            ris = sum(1 for q in range(j, k - 1) if s_[q + 1] - s_[q] < 1e-3 and abs(wk[q + 1] - wk[q]) > 0.05)
            if ris and (best is None or ris > best[0]): best = (ris, wi, j, k, w)
            j = k
    if best is not None:
        ris, wi, j, k, w = best; P, N, wk, bs, T = wall_frame(wi); s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        um = (s_[j] + s_[k - 1]) / 2; jm = int(np.clip(np.searchsorted(s_, um), 0, len(P) - 1)); Ls = s_[k - 1] - s_[j]
        n_ = N[jm] / max(np.linalg.norm(N[jm]), 1e-9); tg = np.array([-n_[1], n_[0]]); zt = float(np.mean(wk[j:k]))
        tgt = np.array([*P[jm], zt])
        cands = [np.array([*(P[jm] + n_ * d_ + tg * s2), zt + h_]) for d_ in (Ls * 0.9 + 10, Ls * 1.2 + 14, Ls * 0.7 + 8, Ls * 1.6 + 20)
                 for s2 in (0.0, Ls * 0.15, -Ls * 0.15) for h_ in (1.0, 4.0, 8.0, -2.0)]
        pick("steps", tgt, outside(cands) or cands, f"stairstep wall: {ris} ledges of {w['steps']['h']} m, a step is a whole number of merlons")
    rkw = [(wi, w) for wi, w in enumerate(C["walls"]) if w.get("rock_top") and w["closed"]] or \
          [(wi, w) for wi, w in enumerate(C["walls"]) if w.get("rock_top")]
    if rkw:
        wi, w = max(rkw, key=lambda q: sum(v is not None for v in q[1]["rock_top"])); P, N, wk, bs, T = wall_frame(wi)
        J = [j for j, v in enumerate(w["rock_top"]) if v is not None]; j = J[len(J) // 2]; tgt = np.array([*P[j], (float(w["rock_top"][j]) + wk[j]) / 2 - 1.0])
        tg = np.array([-N[j][1], N[j][0]])
        cands = [np.array([*(P[j] + N[j] * sg * d_ + tg * s_), tgt[2] + h_]) for sg in (1, -1) for d_ in (30, 42, 22, 60) for s_ in (10, -10, 0, 25, -25)
                 for h_ in (0, 6, 12, -6)]
        pick("rock_wall", tgt, outside(cands), f"wall on the rock edge: {len(J) * 1.5:.0f} m — masonry starting from the rock")
    rw = [(wi, w) for wi, w in enumerate(C["walls"]) if w["kind"] == "retaining"]
    if rw:
        wi, w = max(rw, key=lambda q: float(np.max(np.asarray(q[1]["walk"]) - np.asarray(q[1]["base"])))); P, N, wk, bs, T = wall_frame(wi); j = len(P) // 2
        tgt = np.array([*P[j], (wk[j] + bs[j]) / 2]); tg = np.array([-N[j][1], N[j][0]])
        cands = [np.array([*(P[j] + N[j] * d_ + tg * s_), wk[j] + h_]) for d_ in (16, 24, 12, 32) for s_ in (6, -6, 0, 14, -14) for h_ in (3, 7, 12)]
        pick("terrace", tgt, cands, f"court terrace: retaining wall {float(np.max(wk - bs)):.1f} m, a stair")
    fl = [b for b in C["buildings"] if isinstance(b.get("flush"), dict)]
    if fl:
        b = max(fl, key=lambda q: q["eave"] - q["base"]); R_ = np.asarray(b["poly"], float); d_ = np.asarray(b["axis"], float); n_o = np.array([d_[1], -d_[0]])
        m_ = (R_[0] + R_[1]) / 2; tgt = np.array([*m_, (float(b["base"]) + float(b["eave"])) / 2])
        cands = [np.array([*(m_ + n_o * dd + d_ * s_), tgt[2] + h_]) for dd in (28, 40, 20, 55) for s_ in (8, -8, 0, 18, -18) for h_ in (0, 6, 12, -5)]
        pick("flush", tgt, outside(cands), f"house as a wall: {b['kind']}, {b['storeys']} floors, outward windows only above the walkway")
    th = [b for b in C["buildings"] if b["kind"] == "tower_house"]
    if th:
        b = max(th, key=lambda q: q["storeys"]); R_ = np.asarray(b["poly"], float); d_ = np.asarray(b["axis"], float); n_in = np.array([-d_[1], d_[0]])
        cands = []
        for q_, n_ in (((R_[0] + R_[1]) / 2, -n_in), ((R_[2] + R_[3]) / 2, n_in)):          # facade outward (outside the castle), then into the court
            tgt = np.array([*q_, float(b["eave"]) - 3.0]); cs = ring_c(q_, tgt, (30, 44, 60, 22), (2, 8, 16), n_, spread=(0.0, 0.5, -0.5, 1.0, -1.0))
            cands += [(tgt, c_) for c_ in (outside(cs) if n_ is not n_in else cs)]
        e = next(((c_, tg_) for tg_, c_ in cands if los_ok(c_, tg_)), None)
        if e is not None: out.append(("tower_house", tuple(e[0]), tuple(e[1]), 32, f"tower house: {b['storeys']} floors"))
    kp = C.get("keep")                                                            # fantasy: the main tower as a whole from outside
    if kp and kp.get("stages"):
        c_ = np.asarray(kp["xy"], float); Ht = float(kp["stages"][-1]["z1"]) + float(kp["spire"]["h"]) - float(kp["base"])
        zc_ = float(kp["base"]) + Ht * 0.5; u0 = np.asarray(C["gates"][0]["xy"], float) - c_
        cs = outside(ring_c(c_, np.array([*c_, zc_]), (Ht * 1.05 + 20, Ht * 1.4 + 30, Ht * 0.8 + 15), (-Ht * 0.25, 0.0, Ht * 0.2), u0,
                            spread=(0.0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0, 3.14)))
        for e in cs:                                                              # target — the face of the tower toward the camera (not its middle)
            v = (e[:2] - c_) / max(np.linalg.norm(e[:2] - c_), 1e-9); tg_ = np.array([*(c_ + v * (float(kp["stages"][0]["r"]) + 0.5)), zc_])
            if los_ok(e, tg_):
                out.append(("keep", tuple(e), tuple(np.array([*c_, zc_])), 32, f"main tower in tiers: {Ht:.0f} m — {kp.get('why', '')}"[:110])); break
    cb = [b_ for b_ in C["bridges"] if b_.get("kind") == "chasm"]
    if cb:
        b_ = cb[0]; p0, p1 = np.asarray(b_["pts"][0], float), np.asarray(b_["pts"][1], float); mid = (p0 + p1) / 2; dv = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        nr = np.array([-dv[1], dv[0]]); zt = float(b_["deck"]) - float(b_["depth"]) * 0.35; tgt = np.array([*mid, zt]); L_ = float(b_["len"])
        pick("chasm", tgt, [np.array([*(mid + nr * s_ * d2 + dv * L_ * 0.15), zt + h_]) for d2 in (L_ * 0.9 + 15, L_ * 1.3 + 20) for s_ in (1, -1) for h_ in (0, 10, -5)],
             f"bridge over a chasm: {L_:.0f} m above {b_['depth']:.0f} m")
    sb = [f for f in FEAT if f["kind"] == "sky_bridge"]
    if sb:
        f = sb[0]; p0, p1 = np.asarray(f["p0"], float), np.asarray(f["p1"], float); mid = (p0 + p1) / 2; dv = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        nr = np.array([-dv[1], dv[0]]); tgt = np.array([*mid, float(f["z"])])
        pick("sky", tgt, [np.array([*(mid + nr * s_ * d2), float(f["z"]) + h_]) for d2 in (30, 45, 22, 60) for s_ in (1, -1) for h_ in (-8, 0, 8, -16)],
             f"bridge between towers: span {f['span']:.0f} m")
    tiers = [w for w in C["wards"] if w.get("tier")]
    if len(tiers) > 1:
        c_ = np.asarray(C["site"]["top"][:2], float); g0 = np.asarray(C["gates"][0]["xy"], float); u0 = g0 - c_; Rr = float(np.linalg.norm(u0))
        tgt = np.array([*c_, float(height(*c_)) + 8.0])
        pick("tiers", tgt, outside(ring_c(c_, tgt, (Rr + 60, Rr + 90, Rr + 40), (-6.0, 4.0, 15.0), u0, spread=(0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.14))) or [],
             f"ring tiers: {len(tiers)} — gates on different sides")
    dk = [f for f in FEAT if f["kind"] == "dansker"]
    if dk:
        f = dk[0]; p0, p1 = np.asarray(f["p0"], float), np.asarray(f["p1"], float); mid = (p0 + p1) / 2; dd = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        nr = np.array([-dd[1], dd[0]]); zt = (float(f["deck"]) + float(f["base"])) / 2; tgt = np.array([*mid, zt]); L_ = float(f["len"])
        pick("dansker", tgt, [np.array([*(mid + nr * s_ * d2 - dd * L_ * 0.2), zt + h_]) for d2 in (L_ * 1.3 + 15, L_ * 1.8 + 20) for s_ in (1, -1) for h_ in (4, 12, -2)],
             f"projecting tower on an arcade: {f['arches']} arches, {f['len']:.0f} m")
    prio = ["keep", "tiers", "chasm", "sky", "dansker", "rock", "steps", "rock_wall", "terrace", "flush", "tower_house"]
    out.sort(key=lambda o: prio.index(o[0]) if o[0] in prio else 99)             # fantasy and landmarks first
    return out[:limit]


CX_RU = {"rock": "rock plinth", "steps": "stairstep wall", "rock_wall": "wall on the rock edge", "terrace": "court terrace",
         "flush": "house as a wall", "tower_house": "house-tower", "dansker": "projecting tower on an arcade", "keep": "main tower in tiers",
         "chasm": "bridge over a chasm", "sky": "bridge between towers", "tiers": "tiers-rings"}


def eevee_look(sc, sun_dir=(0.45, -0.35, 0.82)):
    """View for the eyes: Eevee, sun and sky, a material from the vertex color (roughness 0.85), water — glossy."""
    sc.render.engine = "BLENDER_EEVEE"; sc.view_settings.view_transform = "AgX"; sc.view_settings.exposure = float(KW.get("exposure", -1.6))
    ee = sc.eevee; ee.taa_render_samples = 32
    try: ee.use_raytracing = True
    except Exception: pass
    mat = bpy.data.materials.new("col"); nt = mat.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"; nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.85
    wat = bpy.data.materials.new("water"); bw = next(n for n in wat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bw.inputs["Base Color"].default_value = (0.03, 0.07, 0.10, 1); bw.inputs["Roughness"].default_value = 0.08
    for ob in bpy.data.objects:
        if ob.type == "MESH": ob.data.materials.clear(); ob.data.materials.append(wat if ob.name == "water" else mat)
    w = sc.world; w.use_nodes = True; nt = w.node_tree; bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try: sky.sky_type = "MULTIPLE_SCATTERING"
    except Exception: pass
    el = math.asin(sun_dir[2] / math.sqrt(sum(v * v for v in sun_dir)))
    try: sky.sun_elevation = el; sky.sun_rotation = math.atan2(sun_dir[0], sun_dir[1])
    except Exception: pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.15
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 3.0; sun.data.angle = math.radians(1.5)
    d = np.asarray(sun_dir, float); d /= np.linalg.norm(d)
    sun.rotation_euler = (math.acos(d[2]), 0, math.atan2(d[0], -d[1]) + math.pi)


def render(cam, tgt, views, which, res):
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"; shd = sc.display.shading
    shd.light = "STUDIO"; shd.color_type = "VERTEX"; shd.show_cavity = True; shd.cavity_type = "WORLD"; shd.show_shadows = True
    shd.shadow_intensity = 0.55; sc.display.light_direction = (0.45, -0.35, 0.82)
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    if KW.get("look") == "eevee": eevee_look(sc)
    for o in bpy.data.objects:                                                   # debugging: hide=rock,walls — hide parts
        if o.name in KW.get("hide", "").split(","): o.hide_render = True
    out = {}
    if "feat" in which:
        for f_ in os.listdir(OUT):                                               # remove the shots of the previous run
            if f_.startswith("view_feat_"): os.remove(os.path.join(OUT, f_))
        for j, (kind, loc, tg, lens, why) in enumerate(feature_views(int(KW.get("feat_n", 8)))):
            views[f"feat_{j}_{kind}"] = dict(loc=loc, tgt=tg, lens=lens); which = list(which) + [f"feat_{j}_{kind}"]
        json.dump([{"kind": k_, "ru": FEAT_RU.get(k_, k_), "why": w_} for k_, _, _, _, w_ in feature_views(int(KW.get("feat_n", 8)))],
                  open(os.path.join(OUT, "feat_views.json"), "w"), ensure_ascii=False)
    if "cx" in which:
        for f_ in os.listdir(OUT):
            if f_.startswith("view_cx_"): os.remove(os.path.join(OUT, f_))
        cv = complex_views()
        for j, (kind, loc, tg, lens, why) in enumerate(cv):
            views[f"cx_{j}_{kind}"] = dict(loc=loc, tgt=tg, lens=lens); which = list(which) + [f"cx_{j}_{kind}"]
        json.dump([{"kind": k_, "ru": CX_RU.get(k_, k_), "why": w_} for k_, _, _, _, w_ in cv], open(os.path.join(OUT, "cx_views.json"), "w"), ensure_ascii=False)
    for k in which:
        if k not in views: continue
        v = views[k]; cam.location = v["loc"]; tgt.location = v["tgt"]
        cam.data.type = "ORTHO" if "ortho" in v else "PERSP"
        if "ortho" in v: cam.data.ortho_scale = v["ortho"]
        else: cam.data.lens = v["lens"]
        cam.data.clip_start = 0.5; cam.data.clip_end = 6000; back = None
        if v.get("side"):                                                        # the "inside/outside" view by the face markup (bl_space)
            from components.buildings.space import bl_space as BS
            obs_ = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground") and o.data.color_attributes.get("Side")]
            _, back = BS.cutaway(obs_, v["cut"]); BS.show("Side"); shd.show_shadows = False
        sc.render.filepath = os.path.join(OUT, f"view_{k}.png"); bpy.ops.render.render(write_still=True); out[k] = sc.render.filepath
        if back: back(); from components.buildings.space import bl_space as BS; BS.show("Col"); shd.show_shadows = True
    return out


def setup_views(c, half, res=None):
    """Camera cam, target tgt and Workbench settings, as in main with views=none (no shots): this is how the
    "masonry" step (components/constructions/masonry/bl_masonry.py) prepares the scene before its own Eevee
    view."""
    cam, tgt, views = cameras(c, half)
    res = res or tuple(int(v) for v in KW.get("res", "1280x800").split("x"))
    return render(cam, tgt, views, ["none"], res)


def main(argv=None):
    global ARGV
    ARGV = list(argv) if argv is not None else (sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    load(ARGV[0], dict(a.split("=", 1) for a in ARGV[1:] if "=" in a))
    c, half, _ = build(fresh=True); t_build = time.time() - T0; SP = None
    if C.get("spaces") and KW.get("views", "") != "none":                         # faces — cell and side by the scheme; model against the scheme
        from components.buildings.space import bl_space as BS
        from components.buildings.space.space import Space
        S_ = Space.from_dict(C["spaces"]); obs_ = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground")]
        SP = {"faces": BS.tag(S_, obs_), "check": BS.verify(S_)}
    cam, tgt, views = cameras(c, half)
    res = tuple(int(v) for v in KW.get("res", "1280x800").split("x"))
    out = render(cam, tgt, views, KW.get("views", "aerial,road,back,top").split(","), res)
    dg = bpy.context.evaluated_depsgraph_get()
    tris = sum(len(o.evaluated_get(dg).to_mesh().loop_triangles) for o in bpy.data.objects if o.type == "MESH")
    tris_c = sum(len(o.evaluated_get(dg).to_mesh().loop_triangles) for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground"))
    if KW.get("blend", "0") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "castle.blend"))
    print("__RESULT__", json.dumps({"build_s": round(t_build, 2), "total_s": round(time.time() - T0, 1), "n_triangles": tris, "n_triangles_castle": tris_c, "kinds": list(out),
                                    "n_crags": NCRAG[0], "rock_n_faces": len(ARK.f) + len(ARB.f) + len(ARC.f), "space": SP}, ensure_ascii=False))


if __name__ == "__main__":
    main()
