"""Paving — layout of coverings of horizontal surfaces: street and courtyard pavements, building floors (the paving part of
the Constructions component; knowledge — paving.yaml). Pure numpy: works in the project's Python and in Blender (YAML
only in the project's Python; in Blender the kinds arrive in the JSON plan).

The logic is masonry laid flat: rows with bonding (setts, slabs, boards, half_logs — masonry.fill_row), cells
(cobblestone, rubble slabs — a Voronoi diagram on a jittered grid), patterns (herringbone, basketweave, grid with
checkers).
Area:
  Path — a strip along a path (street, road): u is the length along the axis, v is across (−w/2…+w/2);
  Area — a polygon (courtyard, square, floor): u, v are the row axes (u along the axis rotated by an angle).
Layout in chunks of CHUNK m: a stone belongs to the chunk of its middle, rows and cells of the cell patterns use shared
numbers; a stone is the same for any camera and order; at the area edge it is trimmed by the contour.
  units(region, kind, chunk, seed) → stones [{poly (u, v), sid, tint, sub (checker substance), course}]
Stone number: 7·10⁸ + area·10⁶ + chunk·10⁴ + number within the chunk.
Step plan: resolve(folder, keys) — the "paving:" section of the building description constructions.yaml.

  .venv/bin/python -m components.constructions.paving.paving resolve <folder> [court=cobble] [road=setts_large] [floor=planks] [areas=flagstone] [width=…] [detail=cam|all] out=<plan.json>
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "paving.yaml")
from components.constructions.masonry import masonry as M  # noqa: E402

CHUNK = 4.0
SID_BASE = 700_000_000


def load_library(path=LIB):
    """Paving knowledge: kinds (base + own; name, title), substances. Needs PyYAML — the project's Python."""
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    base, kinds = d["base"], {}
    for name, kw in d["kinds"].items():
        k = dict(base); k.update({a: b for a, b in kw.items() if a != "title"}); k["name"] = name; k["title"] = kw.get("title", name)
        kinds[name] = k
    return kinds, d


try:
    KINDS, LIBRARY = load_library()
except ImportError:                                                              # Blender's Python: the kinds come from the plan
    KINDS, LIBRARY = {}, {}


def course_w(k):
    w = k["w"]; return [float(x) for x in w] if isinstance(w, (list, tuple)) else [float(w)]


def unit_size(k):
    """Characteristic stone size (for the camera-based mode): the row width or the cell diameter."""
    if k["generator"] in ("cells", "grid"): return float(np.mean(k["L"]))
    return float(np.mean(course_w(k)))


# ── plane geometry ──────────────────────────────────────────────────────────────────────────────────────────────
def area(P):
    P = np.asarray(P, float)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) if len(P) >= 3 else 0.0


def clip_half(P, A, nrm):
    """Polygon P ∩ half-plane (p − A)·nrm ≥ 0."""
    P = [tuple(map(float, p)) for p in P]; out = []
    f = lambda p: (p[0] - A[0]) * nrm[0] + (p[1] - A[1]) * nrm[1]
    for j in range(len(P)):
        p, q = P[j - 1], P[j]; sp, sq = f(p), f(q)
        if sq >= 0:
            if sp < 0: t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            out.append(q)
        elif sp >= 0:
            t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def clip_box(P, u0, u1, v0, v1):
    for A, n in (((u0, 0), (1, 0)), ((u1, 0), (-1, 0)), ((0, v0), (0, 1)), ((0, v1), (0, -1))):
        P = clip_half(P, A, n)
        if len(P) < 3: return []
    return P


def inset(P, d):
    """Convex polygon P shrunk by d (each edge moved inward by d)."""
    P = np.asarray(P, float)
    if area(P) < 0: P = P[::-1]
    Q = [tuple(p) for p in P]
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]; e = b - a; L = float(np.hypot(*e))
        if L < 1e-9: continue
        n = np.array([-e[1], e[0]]) / L
        Q = clip_half(Q, a + n * d, n)
        if len(Q) < 3: return []
    return Q


def _seg_hits_poly(a, b, Q, eps=1e-9):
    """Segment a–b enters the interior of the convex polygon Q (touching with an end or along the boundary does not count): an
    end strictly inside or a proper intersection with an edge; the segment is shortened at the ends."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = float(np.hypot(*d))
    if L < 1e-12 or len(Q) < 3: return False
    a, b = a + d * (1e-6 / L), b - d * (1e-6 / L)
    Q = [np.asarray(q, float) for q in Q]
    def cross(o, p, q): return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    def inside(p):
        cs = [cross(Q[i], Q[(i + 1) % len(Q)], p) for i in range(len(Q))]
        return all(c > eps for c in cs) or all(c < -eps for c in cs)
    if inside(a) or inside(b): return True
    for i in range(len(Q)):
        c, e = Q[i], Q[(i + 1) % len(Q)]
        d1, d2, d3, d4 = cross(a, b, c), cross(a, b, e), cross(c, e, a), cross(c, e, b)
        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)): return True
    return False


def densify(P, step, axis=None):
    """Polygon edges are subdivided every step (along u — axis=0, along the whole length — None): a long plank at a bend of the
    path follows the axis, not a chord."""
    P = np.asarray(P, float); out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]; L = abs(b[axis] - a[axis]) if axis is not None else float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(L / step)))
        out += [a + (b - a) * (j / n) for j in range(n)]
    return np.asarray(out, float)


def chaikin(P, it=2):
    P = np.asarray(P, float)
    for _ in range(it):
        Q = np.roll(P, -1, 0); P = np.stack([0.75 * P + 0.25 * Q, 0.25 * P + 0.75 * Q], 1).reshape(-1, 2)
    return P


def _rng(*key):
    return np.random.default_rng([int(abs(k)) & 0x7FFFFFFF for k in key])


# ── areas ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Path:
    """A strip along a path: pts — the axis (x, y[, z]), width — the width. u is the length along the axis, v is across (positive
    to the left)."""
    kind = "path"

    def __init__(self, pts, width, name="path"):
        P = np.asarray(pts, float); P = P[np.concatenate([[True], np.hypot(*np.diff(P[:, :2], axis=0).T) > 1e-6])]
        self.P = P[:, :2]; self.Z = P[:, 2] if P.shape[1] > 2 else None; self.w = float(width); self.name = name
        d = np.diff(self.P, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); T = d / seg[:, None]
        Tv = np.vstack([T[:1], T[:-1] + T[1:], T[-1:]]); Tv /= np.linalg.norm(Tv, axis=1, keepdims=True)
        self.N = np.stack([-Tv[:, 1], Tv[:, 0]], 1); self.s = np.concatenate([[0.0], np.cumsum(seg)]); self.L = float(self.s[-1])
        self.box = (0.0, self.L, -self.w / 2, self.w / 2)

    def xy(self, u, v):
        u = np.clip(np.asarray(u, float), 0.0, self.L); v = np.asarray(v, float)
        x = np.interp(u, self.s, self.P[:, 0]); y = np.interp(u, self.s, self.P[:, 1])
        nx = np.interp(u, self.s, self.N[:, 0]); ny = np.interp(u, self.s, self.N[:, 1]); m = np.hypot(nx, ny) + 1e-12
        return np.stack([x + v * nx / m, y + v * ny / m], -1)

    def clip(self, poly):
        c = clip_box(poly, *self.box); return [c] if len(c) >= 3 else []

    def chunks(self):
        n = max(1, int(math.ceil(self.L / CHUNK)))
        return [(i, i * CHUNK, min(self.L, (i + 1) * CHUNK), -self.w / 2, self.w / 2) for i in range(n)]

    def outline(self, step=1.0):
        """Strip contour (x, y) — for the bedding."""
        us = np.unique(np.concatenate([np.arange(0.0, self.L, step), [self.L], self.s]))
        return np.vstack([self.xy(us, -self.w / 2), self.xy(us[::-1], self.w / 2)])


class Area:
    """Polygon: poly — the contour (x, y); angle — the row direction (rad; none — along the longest edge)."""
    kind = "area"

    def __init__(self, poly, angle=None, name="areas"):
        P = np.asarray(poly, float)[:, :2]
        if area(P) < 0: P = P[::-1]
        if angle is None:
            e = np.roll(P, -1, 0) - P; i = int(np.argmax(np.hypot(e[:, 0], e[:, 1]))); angle = math.atan2(e[i, 1], e[i, 0])
        self.angle = float(angle); c, s = math.cos(angle), math.sin(angle)
        self.Rm = np.array([[c, s], [-s, c]]); self.P = P; self.name = name
        self.Q = P @ self.Rm.T
        self.box = (float(self.Q[:, 0].min()), float(self.Q[:, 0].max()), float(self.Q[:, 1].min()), float(self.Q[:, 1].max()))
        e0 = self.Q - np.roll(self.Q, 1, 0); e1 = np.roll(self.Q, -1, 0) - self.Q   # concave vertices of the contour (counter-clockwise)
        self.reflex = self.Q[(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) < -1e-9]
        self.w = self.box[3] - self.box[2]

    def xy(self, u, v):
        U = np.stack([np.asarray(u, float), np.asarray(v, float)], -1); return U @ self.Rm

    def inside(self, pts):
        """Points (u, v) inside the contour (ray to the right)."""
        pts = np.atleast_2d(np.asarray(pts, float)); x, y = pts[:, 0][:, None], pts[:, 1][:, None]
        A, B = self.Q, np.roll(self.Q, -1, 0)
        c = ((A[:, 1] > y) != (B[:, 1] > y)) & (x < (B[:, 0] - A[:, 0]) * (y - A[:, 1]) / (B[:, 1] - A[:, 1] + 1e-12) + A[:, 0])
        return np.count_nonzero(c, axis=1) % 2 == 1

    def clip(self, poly):
        """Stone ∩ contour → pieces: all vertices inside — as is; a concave vertex of the contour inside the stone — the stone is
        cut along its lines (a plank at a corner — in two pieces), the pieces — by the half-planes of the contour edges that they
        touch."""
        q = np.asarray(poly, float)
        if self.inside(q).all(): return [[tuple(p) for p in q]]
        pieces = [q]
        lo, hi = q.min(0), q.max(0)
        for r in self.reflex:
            if not (lo[0] < r[0] < hi[0] and lo[1] < r[1] < hi[1]): continue
            nxt = []
            for pc in pieces:
                for A, n in (((r[0], 0.0), (1.0, 0.0)), ((r[0], 0.0), (-1.0, 0.0))):
                    h = clip_half(pc, A, n)
                    if len(h) >= 3: nxt.append(np.asarray(h, float))
            pieces = nxt
        out = []
        for pc in pieces:
            c = self._clip1(pc)
            if len(c) >= 3: out.append(c)
        return out

    def _clip1(self, q):
        q = np.asarray(q, float); inn = self.inside(q)
        if inn.all(): return [tuple(p) for p in q]
        out = [tuple(p) for p in q]; hit = False
        for a, b in zip(self.Q, np.roll(self.Q, -1, 0)):
            if not _seg_hits_poly(a, b, out): continue
            e = b - a; L = float(np.hypot(*e))
            if L < 1e-9: continue
            hit = True; out = clip_half(out, a, np.array([-e[1], e[0]]) / L)
            if len(out) < 3: return []
        if not hit and not inn.any(): return []
        return out

    def chunks(self):
        u0, u1, v0, v1 = self.box; out = []
        nu, nv = max(1, int(math.ceil((u1 - u0) / CHUNK))), max(1, int(math.ceil((v1 - v0) / CHUNK)))
        for i in range(nu):
            for j in range(nv):
                a, b, c, d = u0 + i * CHUNK, min(u1, u0 + (i + 1) * CHUNK), v0 + j * CHUNK, min(v1, v0 + (j + 1) * CHUNK)
                corners = np.array([(a, c), (b, c), (b, d), (a, d), ((a + b) / 2, (c + d) / 2)])
                if self.inside(corners).any() or ((self.Q[:, 0] >= a) & (self.Q[:, 0] <= b) & (self.Q[:, 1] >= c) & (self.Q[:, 1] <= d)).any():
                    out.append((i * 1000 + j, a, b, c, d))
        return out

    def outline(self, step=1.0):
        return self.P


# ── generators: stones of the chunk [u0, u1) × [v0, v1) — by the stone's middle (clamped to the area's frame) ───
def own(region, chunk, c):
    """A stone with middle c belongs to this chunk: the middle is clamped to the area's frame (an edge stone whose middle is
    outside is not lost); the right and top edge of the area belong to the last chunk."""
    ci, u0, u1, v0, v1 = chunk; b = region.box
    x = min(max(c[0], b[0]), b[1]); y = min(max(c[1], b[2]), b[3])
    return (u0 <= x < u1 or (u1 >= b[1] - 1e-9 and x >= u0)) and (v0 <= y < v1 or (v1 >= b[3] - 1e-9 and y >= v0))



_COURSES = {}


def _courses(region, k, seed, along_v):
    """Rows of an area as a chain from the start of the axis (cached per area and kind): [(start, width, [(a, b)] — stones
    along the row)]. The row width comes from the kind's list (half_logs — random within limits); staggered — a half-stone
    shift every other row; randomly — lengths vary, a joint no closer than from_side to the joints of the row below
    (masonry.fill_row); half_logs — one stone per row, ends staggered randomly."""
    key = (id(region), k["name"], seed, along_v)
    if key in _COURSES: return _COURSES[key]
    lo, hi = (region.box[2], region.box[3]) if along_v else (region.box[0], region.box[1])
    s0, s1 = (region.box[0], region.box[1]) if along_v else (region.box[2], region.box[3])
    ws = course_w(k); j = float(k["joint"]); lens = tuple(float(x) for x in k["L"]); rng = _rng(seed, 17); out = []; x = lo; prev = []; n = 0
    while x < hi:
        if k["generator"] == "half_logs": w = float(rng.uniform(ws[0], ws[-1]))
        else: w = ws[int(rng.integers(len(ws)))] if len(ws) > 1 else ws[0]
        r = _rng(seed, n, 31)
        if k["generator"] == "half_logs":
            spans = [(s0 - r.uniform(0.0, 0.08), s1 + r.uniform(0.0, 0.08))]
        elif k["bond"] == "staggered":
            Lm = float(np.mean(lens)); st = Lm + j; ph = st / 2 if n % 2 else 0.0
            a0 = ph + math.floor((s0 - ph) / st) * st
            spans = [(a, a + Lm) for a in np.arange(a0, s1, st)]
        else:
            f = M.fill_row(s0 - r.uniform(0, lens[0]), s1, lens, float(k["from_side"]), prev, r)
            prev = [b for a, b in f]; spans = [(a, b - j) for a, b in f if b - j > a + 0.01]
        out.append((x, w, spans)); x += w + j; n += 1
    _COURSES[key] = out
    return out


def _rows(region, k, chunk, seed):
    """Rows with bonding: across (rows along u, stones along v) or along (rows along v, stones along u); a stone belongs to the
    chunk of its middle."""
    along_v = k["direction"] == "along"; out = []; pl = k["generator"] == "half_logs"
    for n, (x, w, spans) in enumerate(_courses(region, k, seed, along_v)):
        for i, (a, b) in enumerate(spans):
            c = (a + b) / 2
            if along_v:
                if not own(region, chunk, (c, x + w / 2)): continue
                poly = [(a, x), (b, x), (b, x + w), (a, x + w)]
            else:
                if not own(region, chunk, (x + w / 2, (region.box[2] + region.box[3]) / 2 if pl else c)): continue
                poly = [(x, a), (x + w, a), (x + w, b), (x, b)]
            out.append(dict(poly=poly, course=n, key=(n, i)))
    return out


def _lattice(region, k, chunk, seed, cells):
    """Pattern on a lattice (herringbone, basketweave, grid): cells(i, j) → [(poly, sub)] in its own axes (a, b); rotated by
    angle."""
    ci, u0, u1, v0, v1 = chunk; th = math.radians(float(k.get("angle", 0))); c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])                                              # (a, b) → (u, v)
    corners = np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]) @ R             # (u, v) → (a, b): R⁻¹ = Rᵀ
    lo, hi = corners.min(0) - 1.0, corners.max(0) + 1.0
    out = []
    for (i, j), items in cells(lo, hi):
        for n, (poly, sub) in enumerate(items):
            P = np.asarray(poly, float) @ R.T; m = P.mean(0)
            if own(region, chunk, m): out.append(dict(poly=[tuple(p) for p in P], sub=sub, course=int(j), key=(int(i), int(j), n)))
    return out


def _herring(region, k, chunk, seed):
    W = float(k["w"]) if not isinstance(k["w"], list) else float(k["w"][0]); L = 2 * W; j = float(k["joint"]) / 2
    def cells(lo, hi):
        res = []                                                                 # pair: lying [0, 2W]×[0, W], standing [2W, 3W]×[0, 2W]; t1 = (W, −W), t2 = (2W, 2W)
        T = np.array([[W, 2 * W], [-W, 2 * W]]); Ti = np.linalg.inv(T)            # (a, b) → the start of the pair; window corners → (a, b)
        ab = np.array([(x, y) for x in (lo[0] - 3 * W, hi[0] + W) for y in (lo[1] - 3 * W, hi[1] + W)]) @ Ti.T
        for a in range(int(math.floor(ab[:, 0].min())) - 1, int(math.ceil(ab[:, 0].max())) + 2):
            for b in range(int(math.floor(ab[:, 1].min())) - 1, int(math.ceil(ab[:, 1].max())) + 2):
                ox, oy = a * W + b * 2 * W, -a * W + b * 2 * W
                if not (lo[0] - 3 * W <= ox <= hi[0] + W and lo[1] - 3 * W <= oy <= hi[1] + W): continue
                H = [(ox + j, oy + j), (ox + L - j, oy + j), (ox + L - j, oy + W - j), (ox + j, oy + W - j)]
                V = [(ox + L + j, oy + j), (ox + L + W - j, oy + j), (ox + L + W - j, oy + L - j), (ox + L + j, oy + L - j)]
                res.append(((a, b), [(H, None), (V, None)]))
        return res
    return _lattice(region, k, chunk, seed, cells)


def _basket(region, k, chunk, seed):
    W = float(k["w"]) if not isinstance(k["w"], list) else float(k["w"][0]); S = 2 * W; j = float(k["joint"]) / 2
    def cells(lo, hi):
        res = []
        for a in range(int(math.floor(lo[0] / S)) - 1, int(math.ceil(hi[0] / S)) + 1):
            for b in range(int(math.floor(lo[1] / S)) - 1, int(math.ceil(hi[1] / S)) + 1):
                x, y = a * S, b * S
                if (a + b) % 2 == 0: items = [[(x + j, y + q * W + j), (x + S - j, y + q * W + j), (x + S - j, y + (q + 1) * W - j), (x + j, y + (q + 1) * W - j)] for q in (0, 1)]
                else: items = [[(x + q * W + j, y + j), (x + (q + 1) * W - j, y + j), (x + (q + 1) * W - j, y + S - j), (x + q * W + j, y + S - j)] for q in (0, 1)]
                res.append(((a, b), [(p, None) for p in items]))
        return res
    return _lattice(region, k, chunk, seed, cells)


def _grid(region, k, chunk, seed):
    S = float(k["L"][0]); j = float(k["joint"]) / 2; ch = k.get("checker")
    def cells(lo, hi):
        res = []
        for a in range(int(math.floor(lo[0] / S)) - 1, int(math.ceil(hi[0] / S)) + 1):
            for b in range(int(math.floor(lo[1] / S)) - 1, int(math.ceil(hi[1] / S)) + 1):
                x, y = a * S, b * S; sub = ch[(a + b) % 2] if ch else None
                res.append(((a, b), [([(x + j, y + j), (x + S - j, y + j), (x + S - j, y + S - j), (x + j, y + S - j)], sub)]))
        return res
    return _lattice(region, k, chunk, seed, cells)


def _cells(region, k, chunk, seed):
    """Cells: points are a grid jittered by shared cell numbers (step d, rows shifted by half a step), a stone is a Voronoi cell
    (half-planes toward the neighbors) shrunk by half a joint; for a cobblestone (top = dome) it is rounded."""
    ci, u0, u1, v0, v1 = chunk; d = float(np.mean(k["L"])); dj = float(k.get("jitter", 0.21)) * d; j = float(k["joint"]) / 2
    hy = d * 0.866; wv = float(k.get("weight", 0.15))                                 # point jitter (fraction of the step), spread of the stone's "weight"
    def pt(a, b):
        r = _rng(seed, a + 100_000, b + 100_000, 7); return np.array([a * d + (0.5 * d if b % 2 else 0.0) + r.uniform(-dj, dj),
                                                                     b * hy + r.uniform(-dj, dj)]), float(r.uniform(1.0 - wv, 1.0 + wv))
    out = []
    ia0, ia1 = int(math.floor(u0 / d)) - 3, int(math.ceil(u1 / d)) + 3; ib0, ib1 = int(math.floor(v0 / hy)) - 3, int(math.ceil(v1 / hy)) + 3   # with a margin: at the area edge there are stones whose middle is outside
    cache = {}
    def get(a, b):
        if (a, b) not in cache: cache[(a, b)] = pt(a, b)
        return cache[(a, b)]
    for a in range(ia0, ia1 + 1):
        for b in range(ib0, ib1 + 1):
            p, wt = get(a, b)
            if not own(region, chunk, p): continue
            cell = [(p[0] - 1.5 * d, p[1] - 1.5 * d), (p[0] + 1.5 * d, p[1] - 1.5 * d), (p[0] + 1.5 * d, p[1] + 1.5 * d), (p[0] - 1.5 * d, p[1] + 1.5 * d)]
            for da in range(-2, 3):
                for db in range(-2, 3):
                    if da == 0 and db == 0: continue
                    q, wq = get(a + da, b + db); e = q - p; L2 = float(e @ e)
                    if L2 > (2.6 * d) ** 2: continue
                    t = 0.5 * (1.0 + (wt - wq) * 0.35)                           # stone "weight": neighbors larger and smaller
                    cell = clip_half(cell, p + e * t, -e / math.sqrt(L2))
                    if len(cell) < 3: break
                if len(cell) < 3: break
            if len(cell) < 3: continue
            cell = inset(cell, j)
            if len(cell) < 3: continue
            if k["top"] == "dome": cell = [tuple(x) for x in chaikin(cell, 1)]
            out.append(dict(poly=cell, course=b, key=(a, b)))
    return out


GENERATORS = {"rows": _rows, "half_logs": _rows, "herringbone": _herring, "basketweave": _basket, "grid": _grid, "cells": _cells}


def units(region, k, chunk, seed=1, ri=0):
    """Stones of an area chunk: the kind's generator → trimming by the contour → number and tint from the number."""
    raw = GENERATORS[k["generator"]](region, k, chunk, seed)
    out = []; ci = chunk[0]
    for n, u in enumerate(raw):
        A0 = abs(area(u["poly"])); r = _rng(seed, *u["key"], 3); tint = round(float(r.normal(1.0, float(k["spread"]))), 4)
        for pi, q in enumerate(region.clip(u["poly"])):
            if abs(area(q)) < max(0.0015, 0.08 * A0 if A0 < 0.02 else 0.0015): continue
            sid = SID_BASE + (ri % 1400) * 1_000_000 + (ci % 100) * 10_000 + ((n + 5000 * pi) % 10_000)
            out.append(dict(poly=np.round(np.asarray(q, float), 5).tolist(), sid=sid, tint=tint, sub=u.get("sub"), course=u["course"],
                            cut=len(q) != len(u["poly"])))
    return out


def lay(region, k, seed=1, ri=0):
    """All stones of an area (all chunks) — for the reference test and the kinds sheet."""
    out = []
    for ch in region.chunks(): out += units(region, k, ch, seed, ri)
    return out


# ── step plan ───────────────────────────────────────────────────────────────────────────────────────────────────
DESC_FILES = ("constructions.yaml", "masonry.yaml")
PART_KEYS = ("court", "road", "floor", "areas")


def _detail(v):
    if v not in ("cam", "all"): raise SystemExit(f"detail level '{v}': cam | all")
    return v


def resolve(folder, kw=None):
    """Paving plan of a building: <folder>/constructions.yaml — the "paving:" section: the covering kind per part (court — the
    castle courtyard, road — the approach, floor — building floors, area — areas of the description), road width, mode
    (camera | blocks), px-per-stone thresholds [Tlo, Thi], far view (below it — only bedding with a pattern), seed, areas (a
    list of {path: [[x, y(, z)]…], width} or {outline: [[x, y]…], angle}). Command-line keys override the description. Kinds
    — only the needed ones (kinds=all — all of them: sheet, checks)."""
    import yaml
    kw = dict(kw or {}); path = next((os.path.join(folder, f) for f in DESC_FILES if os.path.exists(os.path.join(folder, f))), "") if folder else ""
    d = (yaml.safe_load(open(path, encoding="utf-8")) or {}) if path else {}
    c = dict(d.get("paving") or {})
    parts = {"court": "cobble", "road": "setts_large", "floor": "planks", "areas": "flagstone"}
    parts.update({p_: c[p_] for p_ in PART_KEYS if c.get(p_)}); parts.update({p_: kw[p_] for p_ in PART_KEYS if kw.get(p_)})
    bad = sorted({v for v in parts.values() if v not in KINDS})
    if bad: raise SystemExit(f"paving: no kind {bad}; available: {', '.join(KINDS)} (components/constructions/paving/paving.yaml)")
    lo, hi = c.get("thresholds") or [8, 14]
    plan = dict(paving=dict(parts=parts, detail=_detail(kw.get("detail") or c.get("detail") or "cam"), thresholds=[float(kw.get("Tlo", lo)), float(kw.get("Thi", hi))],
                             far=float(kw.get("Tfar", c.get("far", 4.0))), seed=int(kw.get("seed", c.get("seed", 1))),
                             road_width=float(kw.get("width", c.get("road_width", 4.0))), regions=c.get("regions") or [],
                             skip=[s for s in str(kw.get("skip", ",".join(c.get("skip") or []))).split(",") if s]),
                kinds={n: KINDS[n] for n in sorted(set(parts.values()))} if kw.get("kinds") != "all" else dict(KINDS),
                substances=LIBRARY.get("substances", {}), version=LIBRARY.get("version"), description=path or None)
    return plan


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "resolve":
    KW = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    plan = resolve(sys.argv[2], KW)
    if KW.get("out"): json.dump(plan, open(KW["out"], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False))
