"""Castle plan from a place and an intent: courts, walls, towers, gates, the main tower, buildings, moat,
bridge, road, relief edits, checks. The knowledge is castles/castles.yaml (place → layout, purpose →
program, era and culture → elements and numbers); the place is read by siting/castle_site.py.

.venv/bin/python -m components.buildings.castles.castle_plan <place folder> [site=auto|spur|summit|…] [purpose=lord_seat] [era=gothic]
  [culture=german] [scale=medium] [mood=grim] [seed=1] [at=anchor|x,y] [out=folder]
  → <out>/castle.json (markup), <out>/relief.npz + location.json (the place with relief edits), <out>/plan.png (__RESULT__)
In code: plan(folder, **intent) → the plan dict."""
import json, math, os, shutil, sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
from components.buildings.siting import castle_site as CS                                                       # noqa: E402
from core.surface.ground import Ground                                                      # noqa: E402

K = yaml.safe_load(open(os.path.join(HERE, "castles.yaml")))
DEFAULT = dict(site="auto", purpose="lord_seat", era="gothic", culture="german", scale="medium", mood="grim", seed=1, goal="")
KNOBS = ("density", "vertical", "rock", "age", "wealth", "terraces", "steps")   # character knobs 0–1
CX = K["complex"]
RES = 0.5                                                                      # m: the plan raster


def lerp(ab, t):
    """Value of a mechanism by a knob: [at 0, at 1]."""
    return float(ab[0] + (ab[1] - ab[0]) * float(np.clip(t, 0, 1)))


# ---------------------------------------------------------------- polygons (meters of the place, counter-clockwise)
def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def area(P):
    P = np.asarray(P, float); return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def ccw(P):
    P = np.asarray(P, float); return P if area(P) > 0 else P[::-1].copy()


def inside(P, x, y):
    """Points (x, y) inside the polygon P — by the parity of crossings, vectorized."""
    P = np.asarray(P, float); x = np.asarray(x, float); y = np.asarray(y, float); r = np.zeros(np.broadcast(x, y).shape, bool)
    for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, 0)):
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
        r ^= c
    return r


def seg_proj(P, p):
    """Nearest point of the outline P to p: (point, edge index, fraction)."""
    P = np.asarray(P, float); A, B = P, np.roll(P, -1, 0); d = B - A
    t = np.clip(((p[0] - A[:, 0]) * d[:, 0] + (p[1] - A[:, 1]) * d[:, 1]) / np.maximum((d ** 2).sum(1), 1e-12), 0, 1)
    Q = A + d * t[:, None]; k = int(np.argmin(np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1])))
    return Q[k], k, float(t[k])


def split_line(P, c, a, t):
    """Cut of a polygon by the line (p − c)·a = t: (the "less" part, the "greater" part, [A, B] — the cut line) or
    None."""
    P = np.asarray(P, float); s = (P - c) @ a - t; lo, hi, cut = [], [], []
    for i in range(len(P)):
        p, q, sp, sq = P[i], P[(i + 1) % len(P)], s[i], s[(i + 1) % len(P)]
        (lo if sp < 0 else hi).append(p)
        if (sp < 0) != (sq < 0):
            x = p + (q - p) * (sp / (sp - sq)); lo.append(x); hi.append(x); cut.append(x)
    if len(cut) != 2 or len(lo) < 3 or len(hi) < 3: return None
    return ccw(np.array(lo)), ccw(np.array(hi)), np.array(cut)


class Frame:
    """Raster around the castle (RES m) for operations on areas: fill, offset, outline, occupancy check."""

    def __init__(self, c, half):
        self.x0, self.y0 = c[0] - half, c[1] - half; self.n = int(2 * half / RES) + 1

    def px(self, P):
        P = np.asarray(P, float); return np.column_stack([(P[:, 0] - self.x0) / RES, (P[:, 1] - self.y0) / RES])

    def fill(self, P, val=1, M=None):
        M = np.zeros((self.n, self.n), np.uint8) if M is None else M
        cv2.fillPoly(M, [np.round(self.px(P) * 8).astype(np.int32)], int(val), shift=3); return M

    def fill_win(self, P, pad=3):
        """Fill in a window by the bounding box of P (fast for small figures): (the window mask, the window slice in
        the raster)."""
        Q = self.px(P); j0, i0 = np.floor(Q.min(0)).astype(int) - pad; j1, i1 = np.ceil(Q.max(0)).astype(int) + pad
        j0, i0 = max(j0, 0), max(i0, 0); j1, i1 = min(j1, self.n), min(i1, self.n)
        if j1 <= j0 or i1 <= i0: return np.zeros((1, 1), np.uint8), (slice(0, 1), slice(0, 1))
        M = np.zeros((i1 - i0, j1 - j0), np.uint8)
        cv2.fillPoly(M, [np.round((Q - [j0, i0]) * 8).astype(np.int32)], 1, shift=3); return M, (slice(i0, i1), slice(j0, j1))

    def contour(self, M, eps=0.0):
        cs, _ = cv2.findContours((M > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: return None
        c = max(cs, key=cv2.contourArea)
        if eps > 0: c = cv2.approxPolyDP(c, eps / RES, True)
        c = c[:, 0, :].astype(float)
        return ccw(np.column_stack([self.x0 + c[:, 0] * RES, self.y0 + c[:, 1] * RES]))

    def offset(self, P, d, eps=0.6):
        """Offset of a polygon by d m (> 0 — outward) via the distance on a raster; the corners are slightly rounded."""
        M = self.fill(P)
        if d < 0: M = (cv2.distanceTransform(M, cv2.DIST_L2, 5) * RES > -d).astype(np.uint8)
        else: M = (cv2.distanceTransform(1 - M, cv2.DIST_L2, 5) * RES <= d).astype(np.uint8)
        return self.contour(M, eps)


def clip_area(A, B):
    """Area of the intersection of convex polygons A and B (Sutherland–Hodgman)."""
    out = [np.asarray(p, float) for p in ccw(A)]; Bc = ccw(B)
    for e0, e1 in zip(Bc, np.roll(Bc, -1, 0)):
        if not out: return 0.0
        inp, out = out, []; d = e1 - e0
        for i in range(len(inp)):
            p, q = inp[i], inp[(i + 1) % len(inp)]; sp, sq = cross2(d, p - e0), cross2(d, q - e0)
            if sp >= 0: out.append(p)
            if (sp >= 0) != (sq >= 0): out.append(p + (q - p) * (sp / (sp - sq)))
    return abs(area(np.array(out))) if len(out) >= 3 else 0.0


def pl_dist(L, p):
    """Distance from a point to a polyline."""
    L = np.asarray(L, float); a, d = L[:-1], np.diff(L, axis=0)
    t = np.clip(((p[0] - a[:, 0]) * d[:, 0] + (p[1] - a[:, 1]) * d[:, 1]) / np.maximum((d ** 2).sum(1), 1e-12), 0, 1)
    return float(np.min(np.hypot(a[:, 0] + d[:, 0] * t - p[0], a[:, 1] + d[:, 1] * t - p[1])))


def road_route(G, a, b, grade=0.16, cell_slope=34.0, step=2.0, margin=60.0, k_turn=0.6):
    """A* road over a grid with spacing step: the path slope per step ≤ grade (a switchback across the slope is
    allowed), cells steeper than cell_slope and water are forbidden; the cost is length × (1 + 6·slope²).
    Returns (N, 3) or None."""
    import heapq
    a = np.asarray(a, float)[:2]; b = np.asarray(b, float)[:2]; nb = G.near_bounds()
    x0, y0 = max(min(a[0], b[0]) - margin, nb[0]), max(min(a[1], b[1]) - margin, nb[1])
    x1, y1 = min(max(a[0], b[0]) + margin, nb[2]), min(max(a[1], b[1]) + margin, nb[3])
    nx, ny = int((x1 - x0) / step) + 1, int((y1 - y0) / step) + 1
    X, Y = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step)
    Z = G.height(X, Y); bad = (G.slope(X, Y) > cell_slope) | (G.water(X, Y) > 0.3)
    idx = lambda p: (int(np.clip(round((p[1] - y0) / step), 0, ny - 1)), int(np.clip(round((p[0] - x0) / step), 0, nx - 1)))
    s0, s1 = idx(a), idx(b); bad[s0] = bad[s1] = False; g = {s0: 0.0}; prev = {}; pq = [(0.0, s0)]
    nbrs = [(dj, di, math.hypot(dj, di) * step) for dj in (-2, -1, 0, 1, 2) for di in (-2, -1, 0, 1, 2) if (dj or di) and math.gcd(abs(dj), abs(di)) == 1]
    while pq:
        f_, u = heapq.heappop(pq)
        if u == s1: break
        if f_ > g.get(u, 1e18) + math.hypot(u[0] - s1[0], u[1] - s1[1]) * step + 1e-6: continue
        for dj, di, L in nbrs:
            v = (u[0] + dj, u[1] + di)
            if not (0 <= v[0] < ny and 0 <= v[1] < nx) or bad[v]: continue
            gr = abs(Z[v] - Z[u]) / L
            if gr > grade: continue
            c = g[u] + L * (1 + 6 * gr * gr)
            if c < g.get(v, 1e18):
                g[v] = c; prev[v] = u; heapq.heappush(pq, (c + math.hypot(v[0] - s1[0], v[1] - s1[1]) * step, v))
    if s1 not in prev: return None
    cells, u = [s1], s1
    while u != s0: u = prev[u]; cells.append(u)
    P = np.array([(x0 + c[1] * step, y0 + c[0] * step) for c in cells[::-1]])
    return np.column_stack([P, G.height(P[:, 0], P[:, 1])])


def reach_map(G, F, src, ditches, wards_, grade=0.16, cell_slope=34.0):
    """Road cost from src to every cell of the window (Dijkstra, the path slope per step ≤ grade — a switchback
    across the slope is allowed); into courts, the moat, water and onto steep slopes — not allowed."""
    import heapq
    half = (F.n - 1) * RES / 2; step = 2.0 if half < 260 else 3.0; n = int(2 * half / step) + 1
    xs = F.x0 + np.arange(n) * step; ys = F.y0 + np.arange(n) * step; X, Y = np.meshgrid(xs, ys)
    Z = G.height(X, Y); bad = (G.slope(X, Y) > cell_slope) | (G.water(X, Y) > 0.3)
    for w in wards_: bad |= inside(np.asarray(w["poly"]), X, Y)
    for D in ditches:
        L = np.asarray(D["pts"]); dd = np.full(X.shape, 1e9)
        for a, b in zip(L[:-1], L[1:]):
            d = b - a; t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(float(d @ d), 1e-9), 0, 1)
            dd = np.minimum(dd, np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]))
        bad |= dd < D["w"] / 2 + 1.0
    sx = np.clip(np.asarray(src, float), [xs[0], ys[0]], [xs[-1], ys[-1]])
    free = np.argwhere(~bad)
    if not len(free): return None
    k = int(np.argmin((X[~bad] - sx[0]) ** 2 + (Y[~bad] - sx[1]) ** 2)); s0 = tuple(free[k])
    C = np.full(X.shape, np.inf); prev = np.full(X.shape + (2,), -1, np.int32); C[s0] = 0.0; pq = [(0.0, s0)]
    nbrs = [(dj, di, math.hypot(dj, di) * step) for dj in (-2, -1, 0, 1, 2) for di in (-2, -1, 0, 1, 2) if (dj or di) and math.gcd(abs(dj), abs(di)) == 1]
    while pq:
        cu_, u = heapq.heappop(pq)
        if cu_ > C[u]: continue
        for dj, di, L in nbrs:
            v = (u[0] + dj, u[1] + di)
            if not (0 <= v[0] < n and 0 <= v[1] < n) or bad[v]: continue
            gr = abs(Z[v] - Z[u]) / L
            if gr > grade: continue
            c_ = cu_ + L * (1 + 6 * gr * gr)
            if c_ < C[v]: C[v] = c_; prev[v] = u; heapq.heappush(pq, (c_, v))
    return {"x0": float(xs[0]), "y0": float(ys[0]), "step": step, "n": n, "C": C, "prev": prev, "src": [float(X[s0]), float(Y[s0])]}


def reach_cell(RM, p):
    j = int(round((p[0] - RM["x0"]) / RM["step"])); i = int(round((p[1] - RM["y0"]) / RM["step"]))
    return (i, j) if 0 <= i < RM["n"] and 0 <= j < RM["n"] else None


def reach_path(RM, p):
    """Path from the start of the approach to the cell p by the reachability map (smoothed)."""
    u = reach_cell(RM, p)
    if u is None or not np.isfinite(RM["C"][u]): return None
    cells = [u]
    while RM["prev"][u][0] >= 0: u = tuple(RM["prev"][u]); cells.append(u)
    P = np.array([(RM["x0"] + c[1] * RM["step"], RM["y0"] + c[0] * RM["step"]) for c in cells[::-1]])
    for _ in range(2):
        if len(P) < 3: break
        Q = np.empty((2 * len(P) - 2, 2)); Q[0::2] = 0.75 * P[:-1] + 0.25 * P[1:]; Q[1::2] = 0.25 * P[:-1] + 0.75 * P[1:]
        P = np.concatenate([P[:1], Q[1:-1], P[-1:]])
    return P


def choose_gate(poly, RM, G, st, ditches, arrive, min_corner, max_ramp=0.30, reach_m=32.0):
    """Gates on the outline: where the road is cheapest at the foot of the ramp (across the moat — by a bridge),
    ramp ≤ 30 %. Candidates every 3 m along the whole outline, away from corner towers; the direction is a chord
    of ±8 m."""
    P = np.asarray(poly); best = None; T = turn(P)
    corners = [P[i] for i in range(len(P)) if abs(T[i]) >= 32]
    seg = np.hypot(*(np.roll(P, -1, 0) - P).T); cum = np.concatenate([[0], np.cumsum(seg)]); per = cum[-1]
    def at(u):
        u = u % per; i = int(np.searchsorted(cum, u, side="right") - 1); f = (u - cum[i]) / max(seg[i], 1e-9)
        return P[i] + (P[(i + 1) % len(P)] - P[i]) * f, i
    keep_off = st["tower_r"] + 4.0
    for u in np.arange(0, per, 3.0):
        g, i = at(u)
        if any(np.hypot(*(g - q)) < keep_off for q in corners): continue
        a_, _ = at(u - 8); b_, _ = at(u + 8); d = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9); o = np.array([d[1], -d[0]])
        dmin = st["wall_t"] / 2 + 3.0
        for D in ditches:
            if pl_dist(D["pts"], g) < st["wall_t"] / 2 + 4 + D["w"]: dmin = max(dmin, st["wall_t"] / 2 + 2.5 + D["w"] + 2.0)
        zg = float(G.height(g[0], g[1]))
        for dd in np.arange(dmin, dmin + reach_m, 2.0):
            p = g + o * dd
            if RM is not None:
                c_ = reach_cell(RM, p)
                if c_ is None or not np.isfinite(RM["C"][c_]): continue
                cost = float(RM["C"][c_])
            else:
                cost = float(np.hypot(*(p - arrive)))
            gr = abs(zg - float(G.height(p[0], p[1]))) / dd
            if gr > max_ramp: continue
            sc = cost + 40 * max(0.0, gr - 0.10) * dd + 2.0 * dd
            if best is None or sc < best[0]: best = (sc, g, d, o, p, gr, dd, i)
            break
    if best is None:
        gp = gate_point(P, arrive, min_corner); gp["foot"] = None; return gp
    sc, g, d, o, p, gr, dd, i = best
    return {"xy": g.round(3).tolist(), "edge": i, "dir": d.round(4).tolist(), "out": o.round(4).tolist(),
            "foot": p.round(3).tolist(), "ramp": round(gr, 3), "ramp_m": round(dd, 1)}


def perim(P):
    P = np.asarray(P, float); return float(np.hypot(*(np.roll(P, -1, 0) - P).T).sum())


def simple_ok(F, P):
    """Outline without self-intersections: the area on the raster ≈ the polygon area (corrected for the perimeter
    strip)."""
    ra = F.fill(P).sum() * RES * RES; sa = abs(area(P))
    return abs(ra - sa - 0.5 * perim(P) * RES) <= 0.04 * sa + 0.35 * perim(P) * RES


def grade_road(G2, out, P, width, win=9.0, keep_out=None):
    """Road profile — the ground smoothed along the path (window win m); the ground under the roadbed follows the
    profile, blending smoothly at the edge. Writes relief.npz; returns points (N, 3) every 1 m."""
    P = np.asarray(P, float)[:, :2]; s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    if s[-1] < 2: return np.column_stack([P, G2.height(P[:, 0], P[:, 1])])
    u = np.arange(0, s[-1] + 1e-6, 1.0); Q = np.column_stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])])
    z = G2.height(Q[:, 0], Q[:, 1]); k = max(1, int(win / 2)); pad = np.pad(z, k, mode="edge")
    zr = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    Hn = G2.Hn; b = G2.near_bounds(); x0, y0, r = b[0], b[1], G2.nres; band = width / 2 + 2.5
    j0 = max(0, int((Q[:, 0].min() - band - x0) / r)); j1 = min(Hn.shape[1], int((Q[:, 0].max() + band - x0) / r) + 2)
    i0 = max(0, int((Q[:, 1].min() - band - y0) / r)); i1 = min(Hn.shape[0], int((Q[:, 1].max() + band - y0) / r) + 2)
    if j1 > j0 and i1 > i0:
        X, Y = np.meshgrid(x0 + np.arange(j0, j1) * r, y0 + np.arange(i0, i1) * r)
        D = np.full(X.shape, 1e9); Z = np.zeros(X.shape)
        for a, c, za, zc in zip(Q[:-1], Q[1:], zr[:-1], zr[1:]):
            d = c - a; t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(float(d @ d), 1e-9), 0, 1)
            dd = np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]); m = dd < D; D = np.where(m, dd, D); Z = np.where(m, za + (zc - za) * t, Z)
        w = np.clip((band - D) / 2.5, 0, 1); w = w * w * (3 - 2 * w)
        for poly, pad_ in (keep_out or []):                                       # courts and the wall strip — do not touch
            Pp = np.asarray(poly); m = inside(Pp, X, Y)
            Lp = np.vstack([Pp, Pp[:1]]); dd = np.full(X.shape, 1e9)
            for a2, c2 in zip(Lp[:-1], Lp[1:]):
                d2 = c2 - a2; t2 = np.clip(((X - a2[0]) * d2[0] + (Y - a2[1]) * d2[1]) / max(float(d2 @ d2), 1e-9), 0, 1)
                dd = np.minimum(dd, np.hypot(X - a2[0] - t2 * d2[0], Y - a2[1] - t2 * d2[1]))
            w = np.where(m | (dd < pad_), 0.0, w)
        H = Hn[i0:i1, j0:j1]; Hn[i0:i1, j0:j1] = H * (1 - w) + Z * w
        z_ = dict(np.load(os.path.join(out, "relief.npz"))); z_["near"] = Hn.astype(z_["near"].dtype)
        np.savez_compressed(os.path.join(out, "relief.npz"), **z_)
    return np.column_stack([Q, zr])


def simplify(P, eps):
    c = cv2.approxPolyDP(np.asarray(P, np.float32).reshape(-1, 1, 2), eps, True)[:, 0, :]
    return ccw(c.astype(float)) if len(c) >= 3 else ccw(P)


def turn(P):
    """Turn angle of the outline at each vertex, degrees (0 — straight)."""
    P = np.asarray(P, float); a = P - np.roll(P, 1, 0); b = np.roll(P, -1, 0) - P
    ang = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))); return ang


def pick(w, rng):
    k = list(w); p = np.array([w[x] for x in k], float); return k[int(rng.choice(len(k), p=p / p.sum()))]


# ---------------------------------------------------------------- intent → numbers
def knobs_of(s):
    """Knobs: castles.yaml knobs → goal (goals) → explicit ones in the intent. Returns (goal, knobs, small things
    yes/no)."""
    kn = dict(K["knobs"]); g = K["goal_alias"].get(s.get("goal") or "", s.get("goal") or ""); small = True
    if g:
        if g not in K["goals"]: raise ValueError(f"unknown goal {g}; available: {', '.join(K['goals'])}")
        gv = dict(K["goals"][g]); small = bool(gv.pop("small", True)); kn.update(gv)
    for k in KNOBS:
        if s.get(k) not in (None, ""): kn[k] = float(s[k])
    return g, {k: round(float(np.clip(v, 0, 1)), 3) for k, v in kn.items()}, small


def resolve(spec):
    s = dict(DEFAULT); s.update({k: v for k, v in spec.items() if v is not None})
    pu = K["purposes"][s["purpose"]]
    for key in ("scale", "culture", "mood"):                                     # the purpose sets its own default (the sorcerer — his own stone)
        if pu.get(key) and key not in spec: s[key] = pu[key]
    sc, er, cu, mo = K["scales"][s["scale"]], K["eras"][s["era"]], K["cultures"][s["culture"]], K["moods"][s["mood"]]
    s["goal"], kn, small = knobs_of(s)
    vk = 1 + 0.6 * (kn["vertical"] - K["knobs"]["vertical"])                    # vertical: towers taller or lower than usual
    k = cu.get("scale", 1.0) * mo.get("size", 1.0)                               # sci-fi, hell — larger
    hk = sc["h"] * mo["h"]
    st = dict(k=k, size=sc["size"] * k, area=sc["area"] * k ** 1.3, wards=max(sc["wards"], pu.get("wards_min", 1)),
              wall_h=er["wall_h"] * hk * k * pu.get("wall_k", 1.0) * cu.get("wall_h_k", 1.0), wall_t=er["wall_t"] * cu.get("wall_t_k", 1.0) * k,
              tower_up=er["tower_up"] * hk * k * mo["slender"] * pu.get("tower_up_k", 1.0) * vk * cu.get("tower_up_k", 1.0),
              spacing=er["spacing"] * k / (mo["towers"] * pu.get("towers_k", 1.0) * cu.get("towers_k", 1.0)), vk=vk,
              tower_r=(3.6 if s["era"] == "romanesque" else 4.4 if s["era"] == "gothic" else 5.2) * k * sc["size"] ** 0.5 / mo["slender"] ** 0.3,
              gate=er["gate"], machicolation=er["machicolation"], concentric=er["concentric"], corners_only=er.get("corners_only", False),
              bastions=er.get("bastions", False), merlon=er.get("merlon", cu["merlon"]), batter=cu["batter"], roof_pitch=max(0, cu["roof_pitch"] + mo.get("pitch", 0)),
              flat_roofs=cu.get("flat_roofs", False), regular=min(1.0, cu["regular"] * pu.get("regular_k", 1.0)),
              lean=cu.get("lean", 0.0), colors=cu["colors"], storey=3.8 * k ** 0.6, keep_k=pu.get("keep_k", 1.0) * (1 + 0.5 * (vk - 1)),
              knobs=kn, small=small, storeys_add=int(round(2 * (kn["vertical"] - K["knobs"]["vertical"]))),
              free_min=round(min(0.35, lerp(CX["infill"]["free_min"], kn["density"])), 3))
    return s, st


# ---------------------------------------------------------------- layout
def template(S, st, rng, cu, centre, face):
    """A regular figure on a plain: a rectangle, a polygon, a triangle — with a side toward the approach; the
    irregularity comes from the culture."""
    A = st["area"]; shape = rng.choice(cu["plan_shape"])
    if shape == "rect":
        asp = rng.uniform(1.0, 1.5); w = math.sqrt(A * asp); h = A / w
        Q = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    elif shape == "tri":
        s = math.sqrt(4 * A / math.sqrt(3)); Q = np.array([[-s / 2, -s * math.sqrt(3) / 6], [s / 2, -s * math.sqrt(3) / 6], [0, s * math.sqrt(3) / 3]])
    else:
        n = int(rng.integers(5, 9)); r = math.sqrt(2 * A / (n * math.sin(2 * math.pi / n)))
        Q = np.array([[r * math.cos(2 * math.pi * i / n - math.pi / 2 - math.pi / n), r * math.sin(2 * math.pi * i / n - math.pi / 2 - math.pi / n)] for i in range(n)])
    j = (1 - st["regular"]) * 0.18 * math.sqrt(A)                                # irregularity
    Q = Q + rng.normal(0, j / 2, Q.shape)
    ang = math.atan2(face[1], face[0]) + math.pi / 2                             # the "−y" side faces the approach
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    return ccw(Q @ R.T + centre)


def bastion_trace(V):
    """Vauban's first system (Britannica, 1842): a bastioned circuit over the polygon V (counter-clockwise).
    The perpendicular from the middle of a side inward: 1/6 of the side (6+ sides), 1/7 (5), 1/8 (4); the lines
    of defense run from the corners through its end; the faces are 2/7 of the side (shoulders S); the flanks run
    from the shoulders to the curtain (the curtain ends F lie on the extension of another's line of defense,
    |S_B F_A| = |S_A S_B|). Returns (outline, ravelins)."""
    V = ccw(np.asarray(V, float)); n = len(V); k = 1 / 6 if n >= 6 else (1 / 7 if n == 5 else 1 / 8); out = []; rav = []
    for i in range(n):
        A, B = V[i], V[(i + 1) % n]; s = float(np.hypot(*(B - A))); u = (B - A) / s; nu = np.array([-u[1], u[0]])
        M = (A + B) / 2; P = M + nu * s * k
        la = (P - A) / np.linalg.norm(P - A); lb = (P - B) / np.linalg.norm(P - B)
        SA = A + la * s * 2 / 7; SB = B + lb * s * 2 / 7; d0 = float(np.hypot(*(SA - SB)))
        FA = SB + lb * d0; FB = SA + la * d0
        out += [A, SA, FA, FB, SB]
        rav.append([M - nu * s * 0.06 - u * s * 0.2, M - nu * s * 0.34, M - nu * s * 0.06 + u * s * 0.2])
    return ccw(np.array(out)), [np.array(r) for r in rav]


def threat_dir(S):
    """Where the threat comes from: the middle of the longest gentle/uphill arc of the edge, otherwise the approach
    direction. A unit vector from the center."""
    c = np.asarray(S["centre"]); P = np.asarray(S["outline"]); best = None
    for a in S["arcs"]:
        if a["cls"] in ("gentle", "uphill") and (best is None or a["n"] > best["n"]): best = a
    if best is not None and best["n"] < 0.9 * len(P):
        p = P[(best["i0"] + best["n"] // 2) % len(P)]
    elif S["approach"].get("arrive"):
        p = np.asarray(S["approach"]["arrive"])
    else:
        p = c + np.array([0, -1.0])
    d = p - c; return d / max(np.linalg.norm(d), 1e-9)


def exposure(S, p):
    """Class of the place edge at point p (what is outside the wall): cliff, steep, water, uphill, gentle."""
    P = np.asarray(S["outline"]); i = int(np.argmin(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])))
    d = float(np.hypot(*(P[i] - p)))
    return CS.CLASSES[S["edge_cls"][i]] if d < 25 else "gentle"


def barrier_site(G, st, rise=14.0, max_half=140.0):
    """Barrier: the narrowest point of the place's road between slopes (across the road in both directions — up to
    a rise of rise m or to water). Returns a dict (point, road direction, wall ends, width) or None."""
    best = None
    for name, pth in (G.loc.get("paths") or {}).items():
        if pth.get("kind") != "road": continue
        P = np.asarray(pth["pts"], float)[:, :2]; s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]); L = s_[-1]
        for u in np.arange(0.22 * L, 0.78 * L, 4.0):
            p = np.array([np.interp(u, s_, P[:, 0]), np.interp(u, s_, P[:, 1])])
            a = np.array([np.interp(u - 8, s_, P[:, 0]), np.interp(u - 8, s_, P[:, 1])]); b = np.array([np.interp(u + 8, s_, P[:, 0]), np.interp(u + 8, s_, P[:, 1])])
            t = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-t[1], t[0]]); z0 = float(G.height(*p)); ends = []
            for sg in (1, -1):
                e = None
                for r in np.arange(4.0, max_half, 2.0):
                    q = p + n * sg * r
                    if float(G.height(*q)) - z0 >= rise or float(G.water(*q)) > 0.5: e = (r, q); break
                ends.append(e)
            if ends[0] is None or ends[1] is None: continue
            w = ends[0][0] + ends[1][0]
            if best is None or w < best["width"]:
                best = {"road": name, "u": float(u), "p": p, "t": t, "n": n, "left": ends[1][1], "right": ends[0][1], "width": float(w),
                        "start": P[0]}
    return best


def rock_bailey(G, S, st, F, top, thr):
    """Lower court at the foot of the rock on the approach side: level (≥ 80 % of the points flatter than 22°),
    dry; smaller if it does not fit."""
    c = np.asarray(top).mean(0); Rr = math.sqrt(abs(area(top)) / math.pi)
    for k in (1.0, 0.75, 0.55, 0.4):
        Ab = max(500.0, 0.55 * st["area"] * k); rb = math.sqrt(Ab / math.pi)
        for ang in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6):
            d = np.array([thr[0] * math.cos(ang) - thr[1] * math.sin(ang), thr[0] * math.sin(ang) + thr[1] * math.cos(ang)])
            ctr = c + d * (Rr + rb + 3)
            B = ccw(np.array([ctr + rb * 1.08 * np.array([math.cos(math.atan2(d[1], d[0]) + math.pi / 6 + i * math.pi / 3),
                                                          math.sin(math.atan2(d[1], d[0]) + math.pi / 6 + i * math.pi / 3)]) for i in range(6)]))
            xs = np.linspace(B[:, 0].min(), B[:, 0].max(), 14); ys = np.linspace(B[:, 1].min(), B[:, 1].max(), 14); X, Y = np.meshgrid(xs, ys)
            m = inside(B, X, Y) & ~inside(np.asarray(top), X, Y)
            if m.sum() < 20: continue
            ok = (G.slope(X[m], Y[m]) < 22).mean() >= 0.8 and (G.water(X[m], Y[m]) < 0.05).all()
            nb = G.near_bounds()
            if ok and B[:, 0].min() > nb[0] + 8 and B[:, 0].max() < nb[2] - 8 and B[:, 1].min() > nb[1] + 8 and B[:, 1].max() < nb[3] - 8:
                return B
    return None


def tier_wards(S, st, F, G, n):
    """Ring tiers upward (Minas Tirith, Mont-Saint-Michel, Potala): rings along the contours of the mountain — the
    areas above the marks z_top − k·Δz, nested one in another; a citadel on top. Δz is the drop from the summit
    to the foot within the circle of the intent, divided by the number of tiers (6–30 m); a ring narrower than
    12 m — the tier is skipped. Returns (courts, info) or None."""
    c = np.asarray(S["top"][:2], float); R_lim = 1.25 * math.sqrt(st["area"] * (1 + 0.9 * (n - 1)) / math.pi)
    xs = F.x0 + np.arange(F.n) * RES; X, Y = np.meshgrid(xs, F.y0 + np.arange(F.n) * RES)
    disk = np.hypot(X - c[0], Y - c[1]) <= R_lim
    if disk.sum() < 100: return None
    H = G.height(X, Y); z_top = float(G.height(*c)); ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    z_rim = float(np.median(G.height(c[0] + R_lim * np.cos(ang), c[1] + R_lim * np.sin(ang))))
    dz = float(np.clip((z_top - z_rim) / n, 6.0, 30.0))
    if z_top - z_rim < 2 * 6.0: return None                                        # a mountain lower than two tiers — no tiers
    polys, prev = [], None; jx, jy = (np.round(F.px([c])[0])).astype(int)
    for k in range(1, n + 1):
        M = ((H >= z_top - k * dz) & disk).astype(np.uint8)
        nl, lab = cv2.connectedComponents(M)
        if not (0 <= jy < F.n and 0 <= jx < F.n) or lab[jy, jx] == 0: continue
        M = (lab == lab[jy, jx]).astype(np.uint8); P = F.contour(M, eps=2.0)
        if P is None or len(P) < 4: continue
        P = simplify(P, max(1.5, 0.03 * math.sqrt(abs(area(P)))))
        if prev is not None:
            Mi = F.fill(prev); ring_w = cv2.distanceTransform(((F.fill(P) > 0) & (Mi == 0)).astype(np.uint8), cv2.DIST_L2, 5) * RES
            if not (ring_w > 0).any() or float(np.percentile(ring_w[ring_w > 0], 90)) * 2 < 12.0: continue    # the ring is empty or narrow — on to the next tier
        polys.append(P); prev = P
    if len(polys) < 2: return None
    out = [{"id": "inner", "poly": polys[0], "tier": 1}]
    for k, P in enumerate(polys[1:], 2):
        out.append({"id": f"tier{k}", "poly": P, "ring": True, "hole": polys[k - 2], "tier": k})
    return out, {"dz": round(dz, 1), "n": len(polys), "z_top": round(z_top, 1), "R": round(R_lim, 1)}


def ring_ward(S, st, F):
    """Sorcerer's ring (Isengard): a round court with a tower in the middle — a circle on a plain and in a valley,
    otherwise along the brow of the platform."""
    c = np.asarray(S["centre"], float); R = math.sqrt(st["area"] / math.pi); n = 36
    return ccw(np.array([c + R * np.array([math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)]) for i in range(n)]))


def wards(S, st, spec, rng, F, G=None):
    """Courts: the main (inner) one and outer ones — by the place rule: an axis (cape, ridge, cliff edge), terraces
    (summit, slope), rings (plain), one (rock, island)."""
    site = K["sites"][S["type"]]; cu = K["cultures"][spec["culture"]]
    c = np.asarray(S["centre"]); thr = threat_dir(S)
    if site["walls"] == "template" or S["type"] in ("plain",):
        main = template(S, st, rng, cu, c, thr)
    else:
        main = F.offset(S["outline"], -(st["wall_t"] / 2 + 1.0))                  # wall at the brow, inside the platform
        if main is None or area(main) < 0.15 * st["area"]:
            main = F.offset(S["outline"], -0.5)
            if main is None: main = ccw(S["outline"])
        raw = main; eps = float(np.clip(0.05 * math.sqrt(abs(area(main))), 2.0, 6.0 * st["k"]))
        for _ in range(5):
            main = simplify(raw, eps)
            if simple_ok(F, main): break
            eps *= 0.5
    pu_ = K["purposes"][spec["purpose"]]
    if pu_.get("layout") == "ring" and (site["walls"] == "template" or S["type"] in ("plain", "valley", "shore")):
        P = ring_ward(S, st, F); return [{"id": "inner", "poly": P, "area_m2": round(abs(area(P)), 1), "ring_tower": True}], None
    nt = int(spec.get("tiers") or 0) or (int(np.clip(round(lerp(pu_["tiers"], 0.5)), *pu_["tiers"])) if pu_.get("tiers") else 0)
    if nt >= 2 and G is not None and S["type"] in ("summit", "hillside", "spur", "ridge", "rock", "cliff_edge"):
        r_ = tier_wards(S, st, F, G, nt)
        if r_ is not None:
            W, info = r_; st["tiers"] = info
            for w in W: w["area_m2"] = round(abs(area(w["poly"])) - (abs(area(w["hole"])) if w.get("hole") is not None else 0), 1)
            return W, None
    if st.get("bastions"):                                                       # artillery era: the circuit is bastions
        n_b = {"watch": 4, "small": 4, "medium": 5, "large": 6, "royal": 7}[spec["scale"]]
        if site["walls"] == "template" or S["type"] == "plain":
            A = st["area"] * 1.15; r = math.sqrt(2 * A / (n_b * math.sin(2 * math.pi / n_b))); a0 = math.atan2(thr[1], thr[0])
            V = np.array([c + r * np.array([math.cos(a0 + math.pi / n_b + 2 * math.pi * i / n_b), math.sin(a0 + math.pi / n_b + 2 * math.pi * i / n_b)]) for i in range(n_b)])
        else:
            eps = math.sqrt(abs(area(main))) * 0.12; V = simplify(main, eps)
            while len(V) > 7: eps *= 1.3; V = simplify(main, eps)
            if len(V) < 4: V = simplify(main, eps * 0.4)
        trace, rav = bastion_trace(V)
        return [{"id": "inner", "poly": trace, "area_m2": round(abs(area(trace)), 1), "bastions": True,
                 "ravelins": [r.round(3).tolist() for r in rav]}], None
    if S["type"] == "rock" or site.get("tight"):                              # castle on a rock: the top of the rock is a building, the court below
        top = F.offset(S["outline"], -0.8)
        if top is None: top = ccw(S["outline"])
        top = simplify(top, max(0.8, 0.03 * math.sqrt(abs(area(top)))))
        out = [{"id": "inner", "poly": top, "rock": True}]
        if spec["purpose"] != "watchtower" and G is not None:
            b = rock_bailey(G, S, st, F, top, thr)
            if b is not None: out.append({"id": "outer", "poly": b, "bailey": True})
        for w in out: w["area_m2"] = round(abs(area(w["poly"])), 1)
        return out, None
    out = [{"id": "inner", "poly": main}]; cross = None
    n = st["wards"] if spec["purpose"] != "watchtower" else 1
    mode = site["wards"]
    if n >= 2 and mode in ("axis", "terraces") and area(main) > 1500 * st["k"] ** 2:
        a = thr if mode == "axis" else -np.asarray(S.get("down", thr))             # the outer court faces the threat (for terraces — lower)
        proj = (main - c) @ a; t = proj.min() + 0.55 * (proj.max() - proj.min())   # ~40 % of the area is the outer court
        r = split_line(main, c, a, t)
        if r is not None:
            lo, hi, cut = r; out = [{"id": "inner", "poly": lo}, {"id": "outer", "poly": hi}]; cross = cut
    elif n >= 2 and mode == "concentric" and st["concentric"]:
        gap = 14.0 * st["k"] ** 0.7
        outer = F.offset(main, gap, eps=1.0)
        if outer is not None: out.append({"id": "outer", "poly": simplify(outer, 1.5), "ring": True})
    elif n >= 2 and mode == "concentric":                                        # Romanesque era: a bailey court on the approach side
        ctr = c + thr * math.sqrt(st["area"]) * 0.95
        b = template(S, dict(st, area=st["area"] * 0.7, regular=0.3), rng, cu, ctr, thr)
        out.append({"id": "outer", "poly": b, "bailey": True})
    for w in out: w["area_m2"] = round(abs(area(w["poly"])), 1)
    return out, cross


def towers_on(poly, S, st, rng, kind_main, gate_pts, skip_edges=()):
    """Towers along the outline: at corners (a turn ≥ 25°) and along walls at a spacing set by the edge's
    protection; close ones merge."""
    P = np.asarray(poly); T = turn(P); pts = []
    for i, p in enumerate(P):
        if abs(T[i]) >= 32: pts.append((p, "corner"))
    sp = st["spacing"]
    for i in range(len(P)):
        if i in skip_edges: continue
        a, b = P[i], P[(i + 1) % len(P)]; L = float(np.hypot(*(b - a))); m = (a + b) / 2
        ex = exposure(S, m); k = {"gentle": 1.0, "uphill": 0.85, "steep": 1.6, "water": 2.2, "cliff": 2.6}[ex]
        if st["corners_only"]: k *= 1.8
        n = int(L // (sp * k))
        for j in range(1, n + 1): pts.append((a + (b - a) * j / (n + 1), "flank"))
    out = []; gap = max(4.0 * st["tower_r"], 18.0 * st["k"] ** 0.6)
    for p, role in sorted(pts, key=lambda q: q[1] != "corner"):
        if any(np.hypot(*(p - g)) < gap * 1.1 for g in gate_pts): continue
        if all(np.hypot(*(p - q["xy"])) >= gap for q in out): out.append({"xy": p, "role": role})
    return out


def place_tower(t, poly, st, plan, roof, base_r, up, lean=0.0):
    """Tower at an outline point: projects outward by 0.4 r (flanking)."""
    p = np.asarray(t["xy"]); q, k, f = seg_proj(poly, p); P = np.asarray(poly)
    a, b = P[k], P[(k + 1) % len(P)]; d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([d[1], -d[0]])
    if t["role"] == "corner":                                                    # at a corner — along the bisector outward
        i = int(np.argmin(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]))); u = P[i] - P[i - 1]; v = P[(i + 1) % len(P)] - P[i]
        nu = np.array([u[1], -u[0]]) / max(np.linalg.norm(u), 1e-9); nv = np.array([v[1], -v[0]]) / max(np.linalg.norm(v), 1e-9)
        n = nu + nv; n /= max(np.linalg.norm(n), 1e-9); d = np.array([-n[1], n[0]])
    c = p + n * base_r * 0.4
    return {"xy": c.round(3).tolist(), "on": p.round(3).tolist(), "plan": plan, "r": round(base_r, 2), "up": round(up, 2),
            "roof": roof, "dir": d.round(4).tolist(), "role": t["role"], "lean": lean}


def gate_point(poly, p_to, min_corner):
    """Gate: the wall point nearest to p_to, no closer than min_corner to the corners."""
    P = np.asarray(poly); q, k, t = seg_proj(P, p_to); a, b = P[k], P[(k + 1) % len(P)]; L = float(np.hypot(*(b - a)))
    if L < 2 * min_corner:
        best = max(range(len(P)), key=lambda i: np.hypot(*(P[(i + 1) % len(P)] - P[i]))); k = best; a, b = P[k], P[(k + 1) % len(P)]
        L = float(np.hypot(*(b - a))); t = 0.5
    t = float(np.clip(t, min_corner / L, 1 - min_corner / L)) if L > 2 * min_corner else 0.5
    d = (b - a) / max(L, 1e-9)
    return {"xy": (a + (b - a) * t).round(3).tolist(), "edge": k, "dir": d.round(4).tolist(), "out": [round(d[1], 4), round(-d[0], 4)]}


def gate_toward(poly, c, v, min_corner):
    """Gate of a ring toward a given side: an edge no shorter than 2·min_corner whose middle is nearest to the
    direction v from c (tiers: the gate of each ring is on a different side)."""
    P = np.asarray(poly, float); v = np.asarray(v, float) / max(np.linalg.norm(v), 1e-9); best = None
    for need_ in (2 * min_corner, min_corner, 0.6 * min_corner):                   # a long edge toward v; if none — a shorter one
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]; L = float(np.hypot(*(b - a)))
            if L < need_: continue
            m = (a + b) / 2 - c; dev = 1 - float(m @ v) / max(np.linalg.norm(m), 1e-9)
            if best is None or dev < best[0]: best = (dev, k)
        if best is not None and best[0] < 0.5: break
    if best is None: return gate_point(P, c + v * 2000, min_corner)
    k = best[1]; a, b = P[k], P[(k + 1) % len(P)]; L = float(np.hypot(*(b - a))); d = (b - a) / max(L, 1e-9)
    mc = min(min_corner, 0.45 * L); t = float(np.clip(((c + v * 2000 - a) @ d) / max(L, 1e-9), mc / L, 1 - mc / L))
    return {"xy": (a + (b - a) * t).round(3).tolist(), "edge": k, "dir": d.round(4).tolist(), "out": [round(d[1], 4), round(-d[0], 4)]}


def keep_spot(rule, ward, S, st, r, thr, H_at, F):
    """Place of the main tower by a rule: highest — the highest point of the court; threat — at the wall toward the
    threat; corner_threat — the wall corner toward the threat (the tower within the circuit); view — the edge
    above the deepest cliff; centre — farthest from the walls."""
    P = np.asarray(ward)
    if rule == "corner_threat":
        return P[int(np.argmax((P - P.mean(0)) @ thr))]
    M = F.fill(P); D = cv2.distanceTransform(M, cv2.DIST_L2, 5) * RES
    ok = D >= r + 2.0
    if not ok.any(): ok = D >= D.max() * 0.8
    ii, jj = np.nonzero(ok); X = F.x0 + jj * RES; Y = F.y0 + ii * RES
    if rule == "highest": v = H_at(X, Y) + 0.02 * D[ii, jj]
    elif rule == "threat": v = (np.column_stack([X, Y]) - P.mean(0)) @ thr
    elif rule == "view":
        Po = np.asarray(S["outline"]); dr = np.asarray(S["edge_drop"]); q = Po[int(np.argmax(dr))]
        v = -np.hypot(X - q[0], Y - q[1])
    else: v = D[ii, jj]
    k = int(np.argmax(v)); return np.array([X[k], Y[k]])


def fill_rect(F, P, keep, margin):
    """The largest rectangle for the platform's shape (the minimal bounding box shrunk to an inscribed one) — a
    building spanning the whole rock."""
    M = F.fill(P); k = max(1, int(margin / RES)); M = cv2.erode(M, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    if keep is not None: cv2.circle(M, tuple(np.round(F.px([keep["xy"]])[0]).astype(int)), int((keep["r"] + 2.0) / RES), 0, -1)
    pts = np.argwhere(M > 0)[:, ::-1].astype(np.float32)
    if len(pts) < 60: return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    for f in np.linspace(1.0, 0.35, 14):
        box = cv2.boxPoints(((cx, cy), (w * f, h * f), ang)); Mb = np.zeros_like(M); cv2.fillPoly(Mb, [np.round(box).astype(np.int32)], 1)
        if Mb.sum() and (Mb & (1 - M)).sum() / Mb.sum() < 0.03:
            Rw = np.column_stack([F.x0 + box[:, 0] * RES, F.y0 + box[:, 1] * RES])
            e0 = np.hypot(*(Rw[1] - Rw[0])); e1 = np.hypot(*(Rw[2] - Rw[1]))
            if e1 > e0: Rw = np.roll(Rw, -1, 0)                                       # R0–R1 is the long side
            return ccw(Rw) if area(Rw) > 0 else Rw
    return None


def buildings(items, wards_, st, gate_xy, keep, S, F, rng, towers=None, pre=None):
    """Buildings along the inner side of the walls: the side by a rule (safe — above a cliff/water, gate — at the
    gate, near_palas — next to the palas), not overlapping, the court stays ≥ 35 % free, with a passage at the
    gate and the main tower."""
    occ = np.zeros((F.n, F.n), np.uint8)
    for t in towers or []: cv2.circle(occ, tuple(np.round(F.px([t["xy"]])[0]).astype(int)), int((t["r"] + 0.8) / RES), 1, -1)
    if keep is not None: cv2.circle(occ, tuple(np.round(F.px([keep["xy"]])[0]).astype(int)), int((keep["r"] + 2.5) / RES), 1, -1)
    for g in gate_xy: cv2.circle(occ, tuple(np.round(F.px([g])[0]).astype(int)), int(9 * st["k"] ** 0.5 / RES), 1, -1)
    T = st["wall_t"]; masks = {}
    for w in wards_:
        M = F.fill(w["poly"])
        for h in w.get("holes", []):                                             # ring: without the inner court and its wall
            M[F.fill(F.offset(h, T + 1.0) if F.offset(h, T + 1.0) is not None else h) > 0] = 0
        masks[w["id"]] = M
    placed = list(pre or [])
    for b in placed: occ |= F.fill(np.asarray(b["poly"]))
    for it in items:
        B = K["buildings"][it]; L0 = B["L"] * st["size"]; D = B["D"] * st["size"] ** 0.8; best = None
        for w in wards_:
            P = np.asarray(w["poly"]); M = masks[w["id"]]; free_area = M.sum() * RES * RES
            seg = np.hypot(*(np.roll(P, -1, 0) - P).T); cum = np.concatenate([[0], np.cumsum(seg)]); per = cum[-1]
            def at(u):                                                              # outline point at length u
                u = u % per; i = int(np.searchsorted(cum, u, side="right") - 1); f = (u - cum[i]) / max(seg[i], 1e-9)
                return P[i] + (P[(i + 1) % len(P)] - P[i]) * f
            for Lk in (1.0, 0.8, 0.62):
                Lb = L0 * Lk; found_any = None
                for u in np.arange(0, per, 3.0):
                    a = at(u); b = at(u + Lb * 1.04); ch = float(np.hypot(*(b - a)))
                    if ch < 0.9 * Lb: continue
                    d = (b - a) / ch; n_in = np.array([-d[1], d[0]]); m = (a + b) / 2
                    sag = max(0.0, float(max(((at(u + Lb * f_) - a) @ n_in) for f_ in (0.25, 0.5, 0.75))))
                    o = m + n_in * (T / 2 + 0.3 + sag)
                    R = np.array([o - d * Lb / 2, o + d * Lb / 2, o + d * Lb / 2 + n_in * D, o - d * Lb / 2 + n_in * D])
                    Mb = F.fill(R)
                    if (Mb & (1 - M)).sum() > 0.02 * Mb.sum() or (Mb & occ).any(): continue
                    if ((occ & M).sum() + Mb.sum()) * RES * RES > 0.65 * free_area: continue
                    ex = exposure(S, m - n_in * 3)
                    gd = min(np.hypot(*(m - np.asarray(g))) for g in gate_xy) if gate_xy else 50.0
                    sc = {"safe": (ex in ("cliff", "steep", "water")) * 2.0 + gd / 60, "gate": 2.0 - gd / 40, "near_palas": 0.5,
                          "any": 0.4, "ring": 1.0}[B["side"]]
                    if B["side"] == "near_palas" and placed: sc = 2.0 - np.hypot(*(m - np.asarray(placed[0]["c"]))) / 30
                    if B["side"] == "ring" and placed:                         # wings — along different walls
                        sc = 1.0 + min(np.hypot(*(m - np.asarray(q["c"]))) for q in placed) / 20
                    if B.get("east"): sc += abs(d[0]) * 0.8                      # chapel — a west–east axis
                    if w["id"] == "inner": sc += 0.5 if B["side"] in ("safe", "near_palas") else (-0.4 if B["side"] == "gate" else 0.0)
                    cand = (sc + 0.5 * Lk, R, w["id"], d, Lb)
                    if found_any is None or cand[0] > found_any[0]: found_any = cand
                if found_any is not None:
                    if best is None or found_any[0] > best[0]: best = found_any
                    break
        if best is None: continue
        _, R, wid, d, Lb = best
        occ |= F.fill(R); placed.append({"kind": it, "ward": wid, "poly": R.round(3).tolist(), "c": R.mean(0).round(3).tolist(),
                                         "axis": d.round(4).tolist(), "L": round(Lb, 2), "D": round(D, 2),
                                         "storeys": max(1, B["storeys"] + st.get("storeys_add", 0)), "roof": "flat" if st["flat_roofs"] else B["roof"],
                                         "apse": bool(B.get("apse"))})
    return placed, occ, masks


def infill(wards_, masks, occ, placed, st, gate_xy, S, F, rx, towers, culture, tight):
    """Growth by annexes (Taufers, Eltz, Bran): after the program — houses along the walls from inside up to a
    share of the wall frontage (knob density), tight against the neighbors (shared walls; a row stops at a
    corner and the next one starts from the corner — an L-shaped joint), of different heights and roofs; in
    cramped castles some houses become the wall (the outer wall of a house = the circuit); tower houses of 5–8
    floors (knob vertical × culture). Returns (new buildings, summary)."""
    fx = CX["infill"]; th = CX["tower_house"]; kn = st["knobs"]; T = st["wall_t"]; sz = st["size"]
    occ_b = np.zeros_like(occ); occ_nb = occ.copy()                            # houses are handled separately (exact joint, by geometry); towers and gates — by raster
    for b in placed: F.fill(np.asarray(b["poly"]), M=occ_b)
    occ_nb[occ_b > 0] = 0
    kinds = dict(fx["kinds"]); kinds["tower_house"] *= th["cultures"].get(culture, 0.5) * kn["vertical"] / 0.4
    p_aw = lerp(fx["as_wall"], kn["density"]) * tight
    new = []; ker = np.ones((3, 3), np.uint8); kk = list(kinds); pk = np.array([kinds[x] for x in kk], float); pk /= pk.sum()
    dil = int(round((T / 2 + 0.6) / RES)); summary = {"front_m": 0.0, "needed_m": 0.0, "like_wall": 0, "n_tower_houses": 0}
    rej = {"outside the court": 0, "towers and gates": 0, "lapped": 0, "court full": 0, "tiny": 0}
    for w in wards_:
        P = np.asarray(w["poly"]); M = masks[w["id"]]; free_px = float(M.sum())
        Mw = cv2.dilate(M, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))       # with the wall thickness — for "house as a wall"
        DT = cv2.distanceTransform(M, cv2.DIST_L2, 5) * RES                       # in a ring the house is no deeper than its width (p90 — not at the corners)
        D_cap = 2.0 * float(np.percentile(DT[M > 0], 90)) - T / 2 - 1.2 if M.any() else 0.0
        seg = np.hypot(*(np.roll(P, -1, 0) - P).T); cum = np.concatenate([[0], np.cumsum(seg)]); per = float(cum[-1])

        def at(u):
            u = u % per; i = int(np.searchsorted(cum, u, side="right") - 1); f = (u - cum[i]) / max(seg[i], 1e-9)
            return P[i] + (P[(i + 1) % len(P)] - P[i]) * f

        def u_of(p):
            q, k, f = seg_proj(P, p); return float(cum[k] + f * seg[k])
        gts = [g for g in gate_xy if pl_dist(np.vstack([P, P[:1]]), np.asarray(g)) < 4.0]
        tws = [t for t in towers if t.get("ward") == w["id"]]
        per_free = max(0.0, per - len(gts) * 18 * st["k"] ** 0.5 - sum(2 * t["r"] + 1.6 for t in tws))
        need = kn["density"] * per_free - sum(b["L"] for b in placed if b["ward"] == w["id"])
        summary["needed_m"] += max(0.0, need)
        for _ in range(40):
            if need < 5.0: break
            kind = kk[int(rx.choice(len(kk), p=pk))]
            if kind == "tower_house": L = rx.uniform(*th["L"]) * sz ** 0.5; D = rx.uniform(*th["D"]) * sz ** 0.5
            else:
                L = rx.uniform(*fx["L"]) * sz ** 0.5 * {"stable": 1.4, "wing": 1.6}.get(kind, 1.0); D = rx.uniform(*fx["D"]) * sz ** 0.5   # a house of human scale
            D = min(D, D_cap)
            if D < 5.0: rej["tiny"] += 1; break
            flush_ok = rx.random() < p_aw; used_px = float((occ & M).sum())
            ends = []
            for b in placed + new:
                if b["ward"] != w["id"]: continue
                R0 = np.asarray(b["poly"]); ends += [u_of(R0[1]) + 0.15, u_of(R0[0]) - L * 1.04 - 0.15, u_of(R0[1]) + 0.15 + D * 0.0]
            grid = list(rx.permutation(np.arange(0, per, 2.0))[:70])
            best = None
            for u in ends + grid:
                for Lk in (1.0, 0.8, 0.62):
                    Lb = L * Lk
                    if Lb < 6.0: continue                                           # under 6 m — not a house
                    a = at(u); b_ = at(u + Lb * 1.04); ch = float(np.hypot(*(b_ - a)))
                    if ch < 0.9 * Lb: continue
                    d = (b_ - a) / ch; n_in = np.array([-d[1], d[0]]); m = (a + b_) / 2
                    dev = [float((at(u + Lb * f_) - a) @ n_in) for f_ in (0.25, 0.5, 0.75)]; sag = max(0.0, max(dev))
                    for flush in ((True, False) if flush_ok else (False,)):
                        if flush and (sag > 0.6 or min(dev) < -0.6): continue
                        o = m + n_in * ((-T / 2 + 0.02) if flush else (T / 2 + 0.3 + sag))
                        R = np.array([o - d * Lb / 2, o + d * Lb / 2, o + d * Lb / 2 + n_in * D, o - d * Lb / 2 + n_in * D])
                        mb, win = F.fill_win(R); s_ = float(mb.sum())
                        if s_ < 4: continue
                        Mref = Mw if flush else M
                        if (mb & (1 - Mref[win])).sum() > 0.02 * s_: rej["outside the court"] += 1; continue
                        if (cv2.erode(mb, ker) & occ_nb[win]).any(): rej["towers and gates"] += 1; continue
                        if any(clip_area(R, np.asarray(q["poly"])) > 0.3 for q in placed + new if np.hypot(*(np.asarray(q["c"]) - m)) < 40):
                            rej["lapped"] += 1; continue
                        if used_px + s_ > (1 - st["free_min"]) * free_px: rej["court full"] += 1; continue
                        touch = bool((cv2.dilate(mb, ker, iterations=2) & occ_b[win]).any())
                        ex = exposure(S, m - n_in * 3)
                        gd = min((np.hypot(*(m - np.asarray(g))) for g in gate_xy), default=50.0)
                        sc = 2.0 * touch + 0.6 * (ex in ("cliff", "steep", "water")) + min(gd, 60) / 60 + 0.25 * Lk + 0.5 * flush + 0.2 * rx.random()
                        if best is None or sc > best[0]: best = (sc, R, d, Lb, flush, touch, mb, win)
                if best is not None and best[5] and u in ends: break               # tight against a neighbor — nothing better will be found
            if best is None: break
            _, R, d, Lb, flush, touch, mb, win = best
            occ[win] |= mb; occ_b[win] |= mb
            if kind == "tower_house": ns = int(rx.integers(th["storeys"][0], th["storeys"][1] + 1)) + max(0, st.get("storeys_add", 0)); summary["n_tower_houses"] += 1
            else: ns = max(1, {"house": int(rx.integers(2, 4)), "store": 2, "stable": 1, "wing": int(rx.integers(2, 4))}[kind] + st.get("storeys_add", 0))
            if touch and new and new[-1]["storeys"] == ns and kind != "tower_house": ns = max(1, ns + (1 if rx.random() < 0.5 else -1))
            roof = ("flat" if st["flat_roofs"] else ("hip" if rx.random() < 0.5 else "gable") if kind == "tower_house" else
                    "lean" if kind == "stable" else "gable" if kind == "wing" else pick(fx["roof"], rx))
            if st["flat_roofs"]: roof = "flat"
            new.append({"kind": kind, "ward": w["id"], "poly": R.round(3).tolist(), "c": R.mean(0).round(3).tolist(), "axis": d.round(4).tolist(),
                        "L": round(Lb, 2), "D": round(float(D), 2), "storeys": ns, "roof": roof, "apse": False, "infill": True,
                        **({"flush": True} if flush else {})})
            need -= Lb; summary["front_m"] += Lb; summary["like_wall"] += int(flush)
    summary = {k: round(v, 1) if isinstance(v, float) else v for k, v in summary.items()}
    summary["rejections"] = {k: v for k, v in rej.items() if v}
    return new, summary


def tower_houses(B, st, rx, culture, tight):
    """Tower houses (Eltz): in a cramped castle with a high vertical the program's dwellings rise upward — 5–8
    floors; a long block is divided into 2–3 houses of different heights built tight together (the houses of the
    family branches). Returns the number of tower houses."""
    th = CX["tower_house"]; kn = st["knobs"]; p = max(0.0, (kn["vertical"] - 0.55) * 2.2) * th["cultures"].get(culture, 0.5) * tight
    if p <= 0: return 0
    out, n = [], 0
    for b in B:
        if b["kind"] not in ("palas", "palas_big", "hall", "wing", "governor") or b.get("infill") or rx.random() >= p:
            out.append(b); continue
        R = np.asarray(b["poly"], float); parts = int(np.clip(round(b["L"] / 12.0), 1, 3))
        for k in range(parts):
            f0, f1 = k / parts, (k + 1) / parts; Q = np.array([R[0] + (R[1] - R[0]) * f0, R[0] + (R[1] - R[0]) * f1,
                                                             R[3] + (R[2] - R[3]) * f1, R[3] + (R[2] - R[3]) * f0])
            ns = int(rx.integers(th["storeys"][0], th["storeys"][1] + 1)) + max(0, st.get("storeys_add", 0))
            out.append(dict(b, kind="tower_house", poly=Q.round(3).tolist(), c=Q.mean(0).round(3).tolist(), L=round(b["L"] / parts, 2),
                            storeys=ns, roof="flat" if st["flat_roofs"] else ("hip" if rx.random() < 0.5 else "gable"),
                            **({"orig": b["kind"]} if k == 0 else {})))
            n += 1
    B[:] = out
    return n


def chasm_bridge(G2, g, W_, TW, st, deep=12.0):
    """Bridge over a chasm (Hunyadi: a bridge on four piers across a ravine; Ravenloft — over a chasm): from the
    main gate outward (±50°) — the nearest edge at gate level (a drop ≤ 4 m and ≤ 8 % of the span) with a depth
    below it greater than deep m; the path does not touch courts or towers. Returns the bridge or None."""
    gp = np.asarray(g["xy"], float); o = np.asarray(g["out"], float); z0 = float(g["z"]); best = None; t = st["wall_t"]
    for ang in np.radians(np.arange(-50, 51, 10)):
        u = np.array([o[0] * math.cos(ang) - o[1] * math.sin(ang), o[0] * math.sin(ang) + o[1] * math.cos(ang)])
        for L in np.arange(22.0, 160.0, 3.0) * st["k"] ** 0.5:
            q = gp + u * (t / 2 + L); zq = float(G2.height(*q))
            if abs(zq - z0) > max(4.0, 0.08 * L): continue
            line = gp + u * (t / 2 + np.linspace(2.0, L - 2.0, 36))[:, None]; zl = G2.height(line[:, 0], line[:, 1]); depth = float(z0 - zl.min())
            if depth < deep: break                                                # the near edge has no chasm: no bridge in this direction
            if any(inside(np.asarray(w["poly"]), line[4:, 0], line[4:, 1]).any() for w in W_): break
            if any(np.hypot(line[:, 0] - tw["xy"][0], line[:, 1] - tw["xy"][1]).min() < tw["r"] + 2.0 for tw in TW): break
            sc = depth / L - abs(ang) * 0.1
            if best is None or sc > best[0]: best = (sc, u, q, L, depth, zq)
            break
    if best is None: return None
    _, u, q, L, depth, zq = best
    return {"pts": [(gp + u * (t / 2 + 0.3)).round(3).tolist(), q.round(3).tolist()], "w": round(float(g["w"]) + 1.6, 2), "deck": round(z0, 2),
            "kind": "chasm", "len": round(float(L), 1), "depth": round(depth, 1), "land": q.round(3).tolist()}


def flush_program(B, W_, st, TW, gates, S, rx, p_aw):
    """Houses as a wall in cramped castles (Eltz, Bran): a program building at a safe wall (cliff, steep slope,
    water) is moved out with its outer wall onto the circuit — the side at the wall shifts to the wall's outer
    face, the inner wall of the house stays."""
    T = st["wall_t"]; n = 0
    for b in B:
        if b.get("infill") or b.get("flush") or b["kind"] in ("chapel", "chapel_big", "stable", "store", "cells"): continue
        if rx.random() >= p_aw: continue
        R = np.asarray(b["poly"], float); d = np.asarray(b["axis"], float); n_in = np.array([-d[1], d[0]])
        if exposure(S, (R[0] + R[1]) / 2 - n_in * 4) not in ("cliff", "steep", "water"): continue
        P = np.asarray(next(w for w in W_ if w["id"] == b["ward"])["poly"])
        dd = [float((R[k] - seg_proj(P, R[k])[0]) @ n_in) for k in (0, 1)]
        if max(dd) - min(dd) > 0.8 or min(dd) < 0: continue                      # the wall at the house is not straight
        sh = min(dd) + T / 2 - 0.02; R2 = R.copy(); R2[0] -= n_in * sh; R2[1] -= n_in * sh
        if any(pl_dist(np.vstack([R2, R2[:1]]), np.asarray(t["xy"])) < t["r"] + 0.3 or inside(R2, *t["xy"]) for t in TW): continue
        if any(np.hypot(*(R2.mean(0) - np.asarray(g["xy"]))) < 6.0 + b["L"] / 2 for g in gates): continue
        b.update(poly=R2.round(3).tolist(), D=round(b["D"] + sh, 2), c=R2.mean(0).round(3).tolist(), flush=True); n += 1
    return n


def ditch_runs(poly, S, rule, st):
    """Wall stretches with a moat in front of them: by the place rule (neck, gentle, uphill, ends, ring)."""
    P = np.asarray(poly); runs = []; cur = []
    want = {"neck": ("gentle", "uphill"), "neck_wet": ("gentle", "uphill"), "gentle": ("gentle",), "uphill": ("uphill", "gentle"),
            "ends": ("gentle",), "ring": None, "ring_wet": None}.get(rule, ())
    if want == ():
        return []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        ok = want is None or exposure(S, (a + b) / 2) in want
        if ok: cur.append(i)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    if want is None: return [list(range(len(P)))]
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(P) - 1: runs[0] = runs[-1] + runs[0]; runs.pop()
    return runs


def offset_chain(P, idx, d, closed):
    """A line parallel to the edges idx of the outline P at a distance d outward."""
    P = np.asarray(P); pts = []
    for i in idx:
        a, b = P[i], P[(i + 1) % len(P)]; t = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([t[1], -t[0]])
        pts += [a + n * d, b + n * d]
    Q = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - Q[-1])) > 0.5: Q.append(p)
    return np.array(Q)


# ---------------------------------------------------------------- court terraces
def clip_line(P, p0, t):
    """Segments of the line p0 + s·t inside the polygon P: [(s0, s1)] in ascending order."""
    P = np.asarray(P, float); ss = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        d = b - a; den = cross2(t, d)
        if abs(den) < 1e-12: continue
        w = a - p0; s_ = cross2(w, d) / den; u = cross2(w, t) / den
        if 0 <= u < 1: ss.append(s_)
    ss = sorted(ss); return [(ss[i], ss[i + 1]) for i in range(0, len(ss) - 1, 2)]


def terraces(G, F, w, st, kn, rng):
    """Court terraces (Bran, Pena): the ground drop of the court (p5–p95) is at least the threshold → 2–3 levels
    across the slope of the plane fitted to the court ground; the boundaries are at fractions of the area, the
    level is the median ground of a strip; a boundary is a retaining wall. Returns (a description or None, the
    drop, the reason)."""
    tp = CX["terraces"]; P = np.asarray(w["poly"]); M = F.fill(P)
    ii, jj = np.nonzero(M)
    if len(ii) < 400: return None, 0.0, "court too small"
    sub = rng.choice(len(ii), min(len(ii), 5000), replace=False); X = F.x0 + jj[sub] * RES; Y = F.y0 + ii[sub] * RES
    z = G.height(X, Y); R = float(np.percentile(z, 95) - np.percentile(z, 5)); rmin = lerp(tp["range_min"], kn["terraces"])
    if R < rmin: return None, R, f"court drop {R:.1f} m < threshold {rmin:.1f} m"
    coef = np.linalg.lstsq(np.column_stack([X, Y, np.ones(len(X))]), z, rcond=None)[0]; g = coef[:2]; gn = float(np.linalg.norm(g))
    if gn < 0.03: return None, R, f"court without a common slope (bumps), plane slope {gn:.2f}"
    g = g / gn; u = np.column_stack([X, Y]) @ g
    n = int(min(tp["levels_max"], max(2, round(R / tp["level_m"]))))
    while n >= 2:
        cuts = [float(np.percentile(u, 100 * k / n)) for k in range(1, n)]; band = np.searchsorted(cuts, u)
        lv = [float(np.median(z[band == k])) for k in range(n)]
        area_k = [float((band == k).sum()) * len(ii) / len(sub) * RES * RES for k in range(n)]
        if min(area_k) >= tp["area_min"] and all(b_ - a_ >= 1.2 for a_, b_ in zip(lv[:-1], lv[1:])): break
        n -= 1
    if n < 2: return None, R, "the terraces come out below the norm"
    c = P.mean(0); tl = np.array([g[1], -g[0]])                                   # along the retaining wall: on the right — down the slope
    lines = []
    for k, uk in enumerate(cuts):
        p0 = c + g * (uk - c @ g)
        for s0, s1 in clip_line(P, p0, tl):
            s0 += st["wall_t"] / 2 - 0.1; s1 -= st["wall_t"] / 2 - 0.1                   # up to the inner face of the circuit
            if s1 - s0 >= 5.0: lines.append({"k": k, "pts": [(p0 + tl * s0).round(3).tolist(), (p0 + tl * s1).round(3).tolist()]})
    return {"dir": g.round(4).tolist(), "cuts": [round(v, 3) for v in cuts], "levels": [round(v, 2) for v in lv], "lines": lines}, R, \
        f"court drop {R:.1f} m ≥ {rmin:.1f} m → {n} levels at {', '.join(f'{v:.1f}' for v in lv)} m"


# ---------------------------------------------------------------- relief edits and wall heights
def terrain_edits(G, F, wards_, ditches, st, keep=None):
    """Courts: level to the court mark (the share is the knob rock; cut and fill up to 6 m) or to the terrace
    levels (retaining walls at the boundaries); the rocky hill of the main tower is not leveled; the moat (a
    trapezoid, 1:1 batters)."""
    Hn = G.Hn.copy(); b = G.near_bounds(); ny, nx = Hn.shape
    X, Y = np.meshgrid(b[0] + np.arange(nx) * G.nres, b[1] + np.arange(ny) * G.nres)
    levels = [None] * len(wards_); inner_m = inside(np.asarray(wards_[0]["poly"]), X, Y)
    s_fl = lerp(CX["rock"]["flatten"], st["knobs"]["rock"]) if st.get("rocky") else 0.8
    protect = np.zeros(Hn.shape)
    if keep is not None and keep.get("knoll"):                                     # rocky hill: around the main tower — as is
        dk = np.hypot(X - keep["xy"][0], Y - keep["xy"][1]); r0 = keep["r"] + 2.0; r1 = keep["r"] + 2.0 + keep["knoll"]
        protect = np.clip((r1 - dk) / max(r1 - r0, 1e-6), 0, 1)
    for k in range(len(wards_) - 1, -1, -1):                                       # the outer ones first, the inner one on top
        w = wards_[k]; P = np.asarray(w["poly"]); m = inside(P, X, Y)
        if w.get("ring"): m &= ~(inside(np.asarray(w["hole"]), X, Y) if w.get("hole") is not None else inner_m)   # ring: without its own "inner" part
        if not m.any(): continue
        z = Hn[m]; zw = float(np.percentile(z, 55)); levels[k] = zw
        tr = w.get("terraces")
        if tr:                                                                    # terraces: each strip has its own level; the transition —
            g = np.asarray(tr["dir"]); u = X * g[0] + Y * g[1]; lvs = np.asarray(tr["levels"]); lv = np.full(Hn.shape, lvs[0])
            for k_, cu_ in enumerate(tr["cuts"]):                                 # inside the retaining wall (1.5 m thick), not in front of it
                f_ = np.clip((u - cu_ + 0.4) / 1.05, 0, 1); f_ = f_ * f_ * (3 - 2 * f_); lv = lv + (lvs[k_ + 1] - lvs[k_]) * f_
            flat = lv + 0.06 * (Hn - lv)
        else:
            flat = zw + (1 - s_fl) * (Hn - zw)
        dz = np.clip(flat - Hn, -6, 6) * (1 - protect)
        Hn = np.where(m, Hn + dz, Hn)
    level = G.level.copy()
    for D in ditches:
        L = np.asarray(D["pts"]); w = D["w"]; dep = D["depth"]
        dist = np.full(Hn.shape, 1e9)
        for a, c in zip(L[:-1], L[1:]):
            d = c - a; t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
            dist = np.minimum(dist, np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]))
        near = dist < w / 2 + dep + 1
        if not near.any(): continue
        brow = float(np.percentile(Hn[dist < w / 2 + 2], 70)) if (dist < w / 2 + 2).any() else float(Hn[near].mean())
        bottom = brow - dep; prof = bottom + np.maximum(0.0, dist - w / 2 + dep * 0.25)   # bottom, batters ~1:1
        Hn = np.where(near, np.minimum(Hn, prof), Hn)
        D["bottom"] = round(bottom, 2)
        if D.get("wet"):
            wl = bottom + 0.55 * dep; D["water"] = round(wl, 2)
            level = np.where(near & (Hn < wl), wl, level)
    return Hn, level, levels


def wall_profile(G2, P, i0, i1, st, inner_level, mult=1.0, closed=True, rock=False, grade=0.35):
    """Wall along the outline edges: points every ~1.5 m, the bottom (down to the ground outside and under the
    wall), the walkway (not lower than wall_h above the ground outside and 2.5 m above the court), smoothed,
    walkway slope ≤ grade (a level walkway — 35 %, with ledges — steeper)."""
    P = np.asarray(P); idx = list(range(i0, i1)); pts = []
    for i in idx:
        a, b = P[i % len(P)], P[(i + 1) % len(P)]; n = max(2, int(np.hypot(*(b - a)) / 1.5) + 1)
        for k in range(n - 1): pts.append(a + (b - a) * k / (n - 1))
    pts.append(P[i1 % len(P)]); pts = np.array(pts)
    T = np.gradient(pts, axis=0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9); N = np.column_stack([T[:, 1], -T[:, 0]])
    t = st["wall_t"]
    zo = G2.height(pts[:, 0] + N[:, 0] * (t / 2 + 1.5), pts[:, 1] + N[:, 1] * (t / 2 + 1.5))
    zi = G2.height(pts[:, 0] - N[:, 0] * (t / 2 + 1.5), pts[:, 1] - N[:, 1] * (t / 2 + 1.5))
    zc = G2.height(pts[:, 0], pts[:, 1])
    walk = np.maximum(zo + st["wall_h"] * mult, np.maximum(zi, zc) + 2.6) if not rock else np.maximum(zi, zc) + 3.4
    k = 4; pad = np.pad(walk, k, mode="edge"); walk = np.array([pad[j:j + 2 * k + 1].max() for j in range(len(walk))])
    for _ in range(3): walk[1:-1] = 0.5 * walk[1:-1] + 0.25 * (walk[:-2] + walk[2:])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    for _ in range(2):                                                            # walkway slope ≤ grade
        for j in range(1, len(walk)): walk[j] = max(walk[j], walk[j - 1] - grade * seg[j - 1])
        for j in range(len(walk) - 2, -1, -1): walk[j] = max(walk[j], walk[j + 1] - grade * seg[j])
    base = (np.minimum(np.minimum(zo, zi), zc) if not rock else np.minimum(zi, zc)) - 0.8   # on rock — the wall stands on the top of the rock
    return pts, walk, base


def plinth(G2, ring, kn, min_body=6.0, top=None, G0=None):
    """Rock plinth (a castle out of rock — Bran, Eltz, Taufers): the ground drop under the footprint (the ring) is
    at least the threshold and it is steep on the lower side → masonry starts at the upper side (0.8 m into the
    ground), below it — rock with an offset. Returns (bottom of the masonry, rock or None)."""
    kr = CX["rock"]; R = np.asarray(ring, float); zz = G2.height(R[:, 0], R[:, 1]); lo, hi = float(zz.min()), float(zz.max())
    c = R.mean(0); q = R[int(np.argmin(zz))]; u = (q - c) / max(np.linalg.norm(q - c), 1e-9)
    sl = float((G0 or G2).slope(*(q + u * 1.2))); dmin = lerp(kr["drop_min"], kn["rock"]); zm = hi - 0.8   # rock — by the original relief (not the moat batter)
    if hi - lo < dmin or sl < kr["slope_min"] or (top is not None and top - zm < min_body):
        return lo - 1.0, None
    sk = [kr["skirt"][0], round(min(6.0, kr["skirt"][1] + 0.3 * (hi - lo)), 2)]   # the taller the rock, the wider its foot
    return zm, {"z1": round(zm + 0.25, 2), "z0": round(lo - 1.0, 2), "skirt": sk, "drop": round(hi - lo, 1), "slope": round(sl)}


def arc_of(P, p):
    """Length along the polyline P to the point nearest to p, and the distance to it."""
    P = np.asarray(P, float); a, d = P[:-1], np.diff(P, axis=0); l2 = (d ** 2).sum(1)
    t = np.clip(((p[0] - a[:, 0]) * d[:, 0] + (p[1] - a[:, 1]) * d[:, 1]) / np.maximum(l2, 1e-12), 0, 1)
    Q = a + d * t[:, None]; dist = np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1]); k = int(np.argmin(dist))
    cum = np.concatenate([[0], np.cumsum(np.sqrt(l2))]); return float(cum[k] + t[k] * np.sqrt(l2[k])), float(dist[k])


def wall_cuts(P, walk, towers, gates, gap):
    """Where the row of merlons breaks: towers on the wall (their "on" at the wall line), gates, a former walkway
    drop greater than 1.5 m (the edge of a shield wall) — [(length along the wall, half-width)]."""
    P = np.asarray(P, float); out = []; cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    for t in towers:
        s_, d = arc_of(P, t["on"])
        if d < max(2.5, 1.2 * t["r"]): out.append((s_, float(t["r"]) + 0.3))
    for g in gates:
        s_, d = arc_of(P, g["xy"])
        if d < 3.0: out.append((s_, (float(g["r"]) * 1.05 if g.get("kind") == "tower" and g.get("r") else float(g["w"]) / 2 + 1.4) + 0.3))
    for j in np.flatnonzero(np.abs(np.diff(np.asarray(walk, float))) > 1.5):
        out.append(((cum[j] + cum[j + 1]) / 2, max(gap / 2, (cum[j + 1] - cum[j]) / 2 + 0.05)))   # the row — beyond the drop stations
    return sorted(out)


def crenel_layout(P, walk, base, T, closed, cuts, p, h=None, each=False):
    """The row of merlons and the wall's steps (a tidy stairstep instead of merlons following their own logic,
    which clashed with the ledges). Between towers, gates and former drops (cuts) — an even row of merlons with
    pitch p at the center of the stretch. For a stairstep wall (ledge h) the level of each merlon is the walkway
    level needed at its position, raised to the ladder of levels at multiples of h (common to the whole wall):
    all ledges equal h, a step is a whole number of merlons, a ledge is in the middle of a clearance. A ledge is
    two stations at one point (the lower and the upper step); for a ring the start is at the first tower (the
    ring seam is inside the tower). A wall ledge is no smaller than the largest rise between neighboring merlons
    — otherwise on a steep stretch the ledge doubled. each — "merlon by merlon" (kremlin): the level of each
    merlon follows its own needed walkway, the ledges follow the slope smoothly.
    Returns (points, walkway, bottom, thicknesses, the row of merlons [[start, count], …], ledge info or None)."""
    P = np.asarray(P, float); walk = np.asarray(walk, float); base = np.asarray(base, float); T = np.asarray(T, float)
    def arc(Q): return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    def at(u, Q, cum, *arrs):
        j = int(np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(Q) - 2)); f = float(np.clip((u - cum[j]) / max(cum[j + 1] - cum[j], 1e-9), 0, 1))
        return Q[j] + (Q[j + 1] - Q[j]) * f, [A[j] + (A[j + 1] - A[j]) * f for A in arrs], j, f
    if closed and cuts:                                                          # a ring — starting from the first tower
        cum = arc(P); L = float(cum[-1]); s0 = cuts[0][0]; q, (w0, b0, t0), j, f = at(s0, P, cum, walk, base, T)
        ring = [(P[i], walk[i], base[i], T[i]) for i in range(len(P) - 1)]
        if f < 0.03: j0 = j
        elif f > 0.97: j0 = j + 1
        else: ring.insert(j + 1, (q, w0, b0, t0)); j0 = j + 1
        j0 %= len(ring); ring = ring[j0:] + ring[:j0]; ring.append(ring[0])
        P = np.array([r_[0] for r_ in ring]); walk = np.array([r_[1] for r_ in ring]); base = np.array([r_[2] for r_ in ring]); T = np.array([r_[3] for r_ in ring])
        cuts = sorted(((c - s0) % L, hw) for c, hw in cuts)
    cum = arc(P); L = float(cum[-1]); cs = list(cuts)
    if not closed: cs = [(0.0, 0.3)] + [c for c in cs if 0.5 < c[0] < L - 0.5] + [(L, 0.3)]
    elif not cs: cs = [(0.0, 0.0)]
    if closed: cs = cs + [(cs[0][0] + L, cs[0][1])]
    z0 = float(walk.min()) - 1e-3
    def need(u0, u1, ends=True):                                                 # the walkway needed on a wall segment (without the ends — not across a drop)
        m = (cum >= u0 - 1e-6) & (cum <= u1 + 1e-6); v = list(walk[m])
        if ends or not v: v += [at(min(max(u, 0), L), P, cum, walk)[1][0] for u in ((u0, u1) if ends else ((u0 + u1) / 2,))]
        return float(max(v))
    secs = []                                                                    # stretches: (c0, c1, row start, merlons, the walkway needed for the merlons)
    for (c0, h0), (c1, h1) in zip(cs[:-1], cs[1:]):
        a_, b_ = c0 + h0, c1 - h1; n = int(np.floor((b_ - a_) / p + 1e-9)) if b_ > a_ else 0
        g0 = a_ + ((b_ - a_) - n * p) / 2
        nd = [need(g0 + m * p, g0 + (m + 1) * p) for m in range(n)] if h is not None else []
        if n > 0 and h is not None:
            nd[0] = max(nd[0], need(c0 + h0, g0, ends=False)); nd[-1] = max(nd[-1], need(g0 + n * p, c1 - h1, ends=False))   # at the edges — like the outermost merlon
        secs.append((c0, c1, g0, n, nd, h0, h1))
    if h is not None and not each:                                               # a ledge is no smaller than the rise between neighboring merlons
        h = max([h] + [abs(b - a) for sec in secs for a, b in zip(sec[4][:-1], sec[4][1:])])
    def rung(z):                                                                 # ladder of levels; "merlon by merlon" — its own level (to 5 cm)
        q = 0.05 if each else h
        return z0 + math.ceil((z - z0) / q - 1e-6) * q
    grid, steps = [], []                                                         # steps: [from, to, level]
    for c0, c1, g0, n, nd, h0, h1 in secs:
        if n > 0: grid.append([round(float(g0), 3), n])
        if h is None: continue
        if n == 0: steps.append([c0, c1, rung(need(c0 + h0, c1 - h1, ends=False))]); continue
        lv = [rung(v) for v in nd]
        e = [c0] + [g0 + m * p for m in range(1, n)] + [c1]
        for m in range(n):
            if steps and steps[-1][1] >= e[m] - 1e-6 and abs(steps[-1][2] - lv[m]) < 1e-6 and m > 0: steps[-1][1] = e[m + 1]
            else: steps.append([e[m], e[m + 1], lv[m]])
    if h is None: return P, walk, base, T, grid, None
    def lev(u):
        for st_ in steps:
            if st_[0] - 1e-6 <= u < st_[1] - 1e-6: return st_[2]
        return steps[-1][2]
    ris = [(steps[j][0], steps[j - 1][2], steps[j][2]) for j in range(1, len(steps)) if abs(steps[j][2] - steps[j - 1][2]) > 1e-6]
    out = []; ri = 0
    for i in range(len(P)):
        while ri < len(ris) and ris[ri][0] <= cum[i] + 1e-6 and ris[ri][0] < L - 1e-6:
            u, lo_, hi_ = ris[ri]; q, (b_, t_), _, _ = at(u, P, cum, base, T)
            out += [(q, lo_, b_, t_), (q, hi_, b_, t_)]; ri += 1
        out.append((P[i], lev(cum[i]) if i < len(P) - 1 or not closed else steps[-1][2], base[i], T[i]))
    cut_s = [c for c, _ in cs]
    inner = [abs(r_[2] - r_[1]) for r_ in ris if min(abs(r_[0] - c) for c in cut_s) > 0.5]   # visible ledges (not inside towers)
    d = np.array(inner) if inner else np.zeros(1)
    lens = [st_[1] - st_[0] for st_ in steps]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]), np.array([o[3] for o in out]), grid,
            {"n": len(inner), "max": round(float(d.max()), 2), "mean": round(float(d.mean()), 2), "h": round(h, 2), "each": bool(each),
             "k": round(float(np.median(lens)) / p, 1) if lens else 0})


# ---------------------------------------------------------------- all together
def plan(folder, out=None, **spec):
    s, st = resolve(spec); rng = np.random.default_rng(int(s["seed"]))
    cu = K["cultures"][s["culture"]]; pu = K["purposes"][s["purpose"]]; kn = st["knobs"]
    logic = []                                                                   # markup: decision, value, why, what it will change
    import zlib
    rx = np.random.default_rng(int(s["seed"]) * 7907 + zlib.crc32(os.path.basename(folder.rstrip("/")).encode()) + 3)   # decisions of the complex geometry

    def note(step, what, value, why, alt=None):
        logic.append({"step": step, "what": what, "value": value, "why": why, **({"otherwise": alt} if alt else {})})
    note("intent", "goal and knobs", dict(kn, target=s["goal"] or "—"), "castles.yaml knobs → goal (goals) → explicit intent knobs",
         "goal=battle|frame|detour|background|history; density= vertical= rock= age= wealth= terraces= steps=")
    G = Ground(folder)
    want = None if s["site"] == "auto" else s["site"]
    S = CS.read(folder, area=st["area"], at=s.get("at"), want=want, g=G, approach_from=s.get("approach"))   # approach — where the approach comes from (the gate goes there)
    ef = S.get("edge_frac", {})
    note("place", "place type", S["type_ru"], "edge: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(ef.items(), key=lambda q: -q[1]) if v >= 0.05)
         + f"; platform {S['area_m2']:.0f} m², {S.get('rel_m', 0):.0f} m above the surroundings", "site=<type> or at=<anchor|x,y> — another platform place")
    st["rocky"] = S["type"] == "rock" or ef.get("cliff", 0) + ef.get("steep", 0) >= 0.3
    BR = barrier_site(G, st) if pu.get("barrier") else None
    if BR is None and S["area_m2"] < 20.0:                                       # smaller than the base of a small tower (a circle of r 2.5 m): the court degenerates
        raise ValueError(f"place {os.path.basename(folder.rstrip('/'))} is not suitable for a castle: platform {S['area_m2']:.0f} m² ({S['type_ru']}) "
                         "smaller than the base of a small tower; choose another place or at=x,y")
    if BR is not None:                                                           # barrier: an outpost on the road at the narrowest place
        tb = BR["t"] if (BR["p"] - BR["start"]) @ BR["t"] > 0 else -BR["t"]        # "behind the wall" — farther from the start of the road
        Wc = min(46.0 * st["size"], BR["width"] * 0.55); Dc = 32.0 * st["size"]; p0 = BR["p"]; n = BR["n"]
        court = ccw(np.array([p0 - n * Wc / 2, p0 + n * Wc / 2, p0 + n * Wc / 2 + tb * Dc, p0 - n * Wc / 2 + tb * Dc]))
        S["type"], S["type_ru"], S["centre"] = "valley", "gorge (barrier)", (p0 + tb * Dc / 2).round(2).tolist()
        BR["court"] = court; BR["tb"] = tb
    site = K["sites"][S["type"]]
    c = np.asarray(S["centre"]); half = math.sqrt(st["area"]) * 1.6 + 70 + 30 * st["k"] + (BR["width"] if BR else 0)
    F = Frame(c, half)
    gx, gy = G.grad(c[0], c[1]); down = -np.array([gx, gy]); S["down"] = (down / max(np.linalg.norm(down), 1e-9)).tolist()
    thr = threat_dir(S) if BR is None else -BR["tb"]
    if BR is not None:
        W_, cross = [{"id": "inner", "poly": BR["court"], "area_m2": round(abs(area(BR["court"])), 1)}], None
        nn = BR["n"]; p0 = BR["p"]; Wc = float(np.hypot(*(BR["court"][1] - BR["court"][0])))
        cl, cr = p0 - nn * Wc / 2, p0 + nn * Wc / 2                              # wall wings — from the outpost up the slopes
        el = BR["left"] - nn * 6.0; er = BR["right"] + nn * 6.0
        BR["wings"] = [np.array([el, cl]), np.array([cr, er])]
    else:
        W_, cross = wards(S, st, s, rng, F, G)
    inner = W_[0]; outer = W_[1] if len(W_) > 1 else None
    enclosure = outer if (outer is not None and (outer.get("ring") or outer.get("bailey"))) else None
    tiered = sum(1 for w in W_ if w.get("tier")) > 1
    if tiered: outer = enclosure = W_[-1]                                        # tiers: outside — the lower ring
    wmode = ("barrier across the gorge" if BR is not None else "bastions (Vauban)" if inner.get("bastions") else
             f"ring tiers ({len(W_)})" if sum(1 for w in W_ if w.get("tier")) > 1 else "ring with a tower in the middle" if inner.get("ring_tower") else
             "a building spanning the whole rock" + (" + lower court" if outer is not None else "") if inner.get("rock") else
             "ring (concentric)" if enclosure is not None and enclosure.get("ring") else
             "bailey court at the approach" if enclosure is not None else "court after court (cut)" if cross is not None else "one court")
    note("courts", "floorplan", wmode, f"place '{S['type_ru']}' → rule {site['wards']}; courts by scale {st['wards']}",
         "scale= (more courts), era=artillery (bastions), purpose=pass_fort (barrier), site=rock (rock)")
    A_have = sum(abs(area(w["poly"])) for w in W_ if not w.get("ring")) + (abs(area(enclosure["poly"])) - abs(area(inner["poly"])) if enclosure is not None and enclosure.get("ring") else 0)
    fit = float(np.clip(math.sqrt(A_have / st["area"]), 0.55, 1.0))                 # fit the elements into the platform the place gives
    st["fit"] = round(fit, 3); st["tower_r"] *= fit; st["size"] *= fit; st["tower_up"] *= fit ** 0.5
    st["A_inner"] = abs(area(inner["poly"]))

    # moat
    ditches = []
    rule = site["ditch"] if not st.get("bastions") else ("ring_wet" if site["ditch"].endswith("_wet") else "ring")
    ring = enclosure if enclosure is not None else (outer if outer is not None else inner)
    if BR is not None:
        rule = "none"; w_ = 10.0 * st["k"] ** 0.8; dep = 4.5 * st["k"] ** 0.7
        L = np.array([BR["left"], BR["right"]]) - BR["tb"] * (st["wall_t"] / 2 + 2.5 + w_ / 2)
        ditches.append({"pts": L.round(3).tolist(), "w": round(w_, 2), "depth": round(dep, 2), "wet": False, "ward": "inner"})
    if rule != "none":
        wet = rule.endswith("_wet") or (rule == "ring" and S["edge_frac"].get("water", 0) > 0.02)
        w_ = {"romanesque": 9.0, "gothic": 12.0, "late": 15.0, "artillery": 26.0, "future": 18.0}[s["era"]] * st["k"] ** 0.8
        dep = {"romanesque": 4.5, "gothic": 6.0, "late": 7.0, "artillery": 6.0, "future": 9.0}[s["era"]] * st["k"] ** 0.7
        for run in ditch_runs(ring["poly"], S, rule, st):
            closed = len(run) == len(ring["poly"])
            L = offset_chain(ring["poly"], run, st["wall_t"] / 2 + 2.5 + w_ / 2, closed)
            if not closed and len(L) >= 2:                                        # extend the ends to the slope
                e0 = L[0] - L[1]; e1 = L[-1] - L[-2]
                L = np.vstack([L[0] + e0 / max(np.linalg.norm(e0), 1e-9) * 8, L, L[-1] + e1 / max(np.linalg.norm(e1), 1e-9) * 8])
            if closed: L = np.vstack([L, L[:1]])
            if len(L) >= 2 and np.hypot(*np.diff(L, axis=0).T).sum() > 12:
                ditches.append({"pts": L.round(3).tolist(), "w": round(w_, 2), "depth": round(dep, 2), "wet": bool(wet), "ward": ring["id"]})

    # gates: where a road can climb (a reachability map from the start of the approach); inner ones — with a shift
    arrive = np.asarray(S["approach"].get("arrive") or (c + thr * 50))
    gate_r = st["tower_r"] * (1.25 if cu["keep"] == "gatehouse" else 1.0)
    if BR is not None:
        Pc = BR["court"]; tb = BR["tb"]; d_ = BR["n"] / np.linalg.norm(BR["n"])
        gf = {"xy": BR["p"].round(3).tolist(), "edge": 0, "dir": d_.round(4).tolist(), "out": (-tb).round(4).tolist(), "foot": None, "ward": "inner"}
        Rp = np.asarray(G.loc["paths"][BR["road"]]["pts"], float)[:, :2]; back = BR["p"] + tb * float(np.hypot(*(Pc[2] - Pc[1])))
        k_ = int(np.argmin(np.hypot(Rp[:, 0] - back[0], Rp[:, 1] - back[1]))); gb_p = seg_proj(Pc, Rp[k_])[0]
        gr = {"xy": gb_p.round(3).tolist(), "edge": 2, "dir": (-d_).round(4).tolist(), "out": tb.round(4).tolist(), "foot": None, "ward": "inner"}
        S["approach"]["from"] = None
    src = S["approach"].get("from")
    if src is not None and any(inside(np.asarray(w["poly"]), src[0], src[1]) or pl_dist(np.vstack([w["poly"], w["poly"][:1]]), src) < 25 for w in W_):
        nb = G.near_bounds(); far_ = (nb[2] - nb[0]) / 2 - 12
        src = [float(np.clip(c[0] + thr[0] * 400, nb[0] + 12, nb[2] - 12)), float(np.clip(c[1] + thr[1] * 400, nb[1] + 12, nb[3] - 12))]
        S["approach"]["from"] = src
    gw0 = enclosure if enclosure is not None else (outer if outer is not None else inner)
    RM = None; g_main = None; approach_kind = None
    for kind_, gr_, cs_ in (("road", 0.16, 34.0), ("trail", 0.35, 45.0), ("stair", 0.9, 62.0)):
        if src is None: break                                                    # a road if possible; otherwise a path, a stair in the rock
        RM_ = reach_map(G, F, src, ditches, W_, grade=gr_, cell_slope=cs_)
        if RM_ is None: continue
        for mr, rm in ((0.30, 32.0), (0.45, 50.0), (9.0, 80.0)):                 # a steeper and longer ramp — if there is no other way
            g_try = choose_gate(gw0["poly"], RM_, G, st, ditches, arrive, gate_r * 2 + 3, mr, rm)
            if g_try.get("foot") is not None: break
        if g_try.get("foot") is not None and reach_path(RM_, g_try["foot"]) is not None:
            g_main, RM, approach_kind = g_try, RM_, kind_; break
    if g_main is None and BR is not None:
        g_main = gf; approach_kind = "road of the place"
    if g_main is None:
        g_main = gate_point(gw0["poly"], arrive, gate_r * 2 + 3); g_main["foot"] = None
    g_main["ward"] = gw0["id"]
    if tiered:                                                                   # tiers: the gate of each ring is on a different side (zigzag)
        gates = [g_main]; v = np.asarray(g_main["xy"]) - np.asarray(S["top"][:2]); sgn = 1.0
        for w in reversed(W_[:-1]):
            a_ = math.pi * 0.85 * sgn + rx.uniform(-0.25, 0.25); sgn = -sgn
            v = np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]]) @ v
            g_ = gate_toward(w["poly"], np.asarray(S["top"][:2]), v, gate_r * 2 + 3)    # toward v — an edge, not the nearest one
            g_["ward"] = w["id"]; gates.append(g_)
        note("courts", "tiers-rings", f"{len(W_)} tiers every {st['tiers']['dz']} m, summit {st['tiers']['z_top']} m",
             "rings along the mountain's contours (the area above a mark is nested), a citadel on top; the gate of each ring is on a different "
             "side, the road zigzags (Minas Tirith)", "tiers=N (2–7), purpose=citadel")
    elif outer is not None and enclosure is not None:
        to_in = np.asarray(g_main["xy"]); rot = np.array([[math.cos(0.9), -math.sin(0.9)], [math.sin(0.9), math.cos(0.9)]])
        g_in = gate_point(inner["poly"], c + rot @ (to_in - c), gate_r * 2 + 3); g_in["ward"] = "inner"; gates = [g_main, g_in]
    elif outer is not None:                                                      # court after court: the gate into the outer court, then through the cut
        cut = np.asarray(cross); mid = cut.mean(0)
        g_in = {"xy": (mid + (cut[1] - cut[0]) * 0.18).round(3).tolist(), "edge": -1,
                "dir": ((cut[1] - cut[0]) / max(np.linalg.norm(cut[1] - cut[0]), 1e-9)).round(4).tolist(), "ward": "inner"}
        d = np.asarray(g_in["dir"]); g_in["out"] = [round(d[1], 4), round(-d[0], 4)]
        gates = [g_main, g_in]
    else:
        gates = [g_main] if BR is None else [g_main, dict(gr, ward="inner")]
    for g in gates:
        g["kind"] = "gatehouse" if (cu["keep"] == "gatehouse" and g is gates[0]) else st["gate"]
        if pu.get("no_towers"): g["kind"] = "simple"                                # watchtower: a simple opening in the wall
        g["w"] = round(float(np.clip(3.4 * st["size"] ** 0.5, 3.0, 4.4 * st["k"] ** 0.5)), 2)
        g["h"] = round(float(np.clip(4.6 * st["size"] ** 0.4, 4.0, 5.8 * st["k"] ** 0.5)), 2)
    note("gate", "main gate", f"{g_main['kind'] if 'kind' in g_main else st['gate']}, approach — {approach_kind or 'not found'}",
         "the gate is where a road climbs from the start of the approach (reachability map): a road ≤ 16 %, otherwise a path, otherwise a stair"
         + (f"; ramp {g_main.get('ramp', 0):.0%} over {g_main.get('ramp_m', 0):.0f} m" if g_main.get("ramp") is not None else ""),
         "at= (another platform), culture= (gate kind: tower, a pair of towers, a gate-donjon)")

    # main tower
    keep = None
    if pu["keep"] and cu["keep"] not in ("none", "gatehouse") and not st.get("bastions"):
        rule = cu["keep_at"] if cu["keep"] in ("spire", "donjon_round") else site.get("keep_at", cu["keep_at"])
        kk = {"bergfried": (5.0, "square" if rng.random() < 0.5 else "round", 20), "donjon": (10.0, "square", 18),
              "donjon_round": (8.5, "round", 22), "norman": (11.0, "square", 16), "homage": (6.5, "square", 15),
              "spire": (9.0, "poly8", 34)}[cu["keep"]]
        r = kk[0] * st["size"] ** 0.5 * st["keep_k"] ** 0.4 * (1.0 if cu["keep"] != "spire" else st["k"] ** 0.2)
        r = max(3.0, min(r, 0.2 * math.sqrt(st["A_inner"])))                        # no thicker than a fifth of the court
        p = keep_spot(rule, inner["poly"], S, st, r, thr, G.height, F)
        keep = {"kind": cu["keep"], "xy": np.asarray(p).round(3).tolist(), "plan": kk[1], "r": round(r, 2),
                "up": round(min(kk[2] * st["size"] ** 0.5 * K["moods"][s["mood"]]["h"] * st["keep_k"] * (st["k"] ** 0.5), r * 6.5), 2),
                "roof": pick(cu["tower_roof"], rng) if cu["keep"] != "spire" else "spire", "rule": rule, "lean": st["lean"]}
        ks = pu.get("keep_style") or cu.get("tower_style")                     # fantasy: the main tower by tiers
        if ks:
            sty = K["tower_styles"][ks]
            if pu.get("keep_h"):                                                 # the tower of a mage, a sorcerer, a vampire — has its own measure
                r = float(min(rx.uniform(*pu["keep_r"]) * st["k"] ** 0.5, 0.22 * math.sqrt(st["A_inner"])))
                H = lerp(pu["keep_h"], 0.5 + (kn["vertical"] - K["knobs"]["vertical"])) * st["k"] ** 0.5
                p = keep_spot(rule, inner["poly"], S, st, r, thr, G.height, F); keep.update(xy=np.asarray(p).round(3).tolist(), r=round(r, 2))
            else:
                H = keep["up"] * cu.get("keep_up_k", 1.0) * st["vk"]
            keep["up"] = round(min(H, sty["slender"] * 2 * keep["r"]), 2); keep["style"] = ks
            if keep["plan"] not in sty["chain"]: keep["plan"] = next(iter(sty["chain"]))
        Pk = np.asarray(inner["poly"]); xs_ = rx.uniform(Pk[:, 0].min(), Pk[:, 0].max(), 600); ys_ = rx.uniform(Pk[:, 1].min(), Pk[:, 1].max(), 600)
        m_ = inside(Pk, xs_, ys_); zmed = float(np.median(G.height(xs_[m_], ys_[m_]))) if m_.sum() > 20 else float(G.height(*p))
        rise_k = float(G.height(*p)) - zmed
        if not inner.get("rock") and (st["rocky"] or rise_k >= 1.5) and rx.random() < CX["rock"]["knoll"] * kn["rock"]:
            keep["knoll"] = round(3.0 + 6.0 * kn["rock"], 1)                       # the court around the main tower is not leveled
        note("main tower", "place", f"{cu['keep']} by the rule {rule}" + (", on a rocky hill" if keep.get("knoll") else ""),
             {"highest": "the highest point of the court", "threat": "at the wall toward the threat", "corner_threat": "a corner toward the threat, within the circuit",
              "view": "above the deepest cliff", "centre": "farthest from the walls"}.get(rule, rule)
             + f"; {rise_k:+.1f} m above the middle of the court" + (f"; the {keep['knoll']} m hill is not leveled (knob rock {kn['rock']})" if keep.get("knoll") else ""),
             "culture= (kind and rule of the main tower), rock= (hill), vertical= (height)")

    # towers
    plan_main = pick(cu["tower_plan"], rng); roof_main = pick(cu["tower_roof"], rng)
    TW = []
    for w in W_:
        gp = [np.asarray(g["xy"]) for g in gates]
        skip = ()
        if w.get("bastions") or pu.get("no_towers"): continue                    # bastions flank by themselves; a watchtower has no towers
        for t in towers_on(w["poly"], S, st, rng, plan_main, gp, skip):
            if keep is not None and keep["rule"] == "corner_threat" and np.hypot(*(np.asarray(t["xy"]) - np.asarray(keep["xy"]))) < keep["r"] + st["tower_r"]:
                continue
            if keep is not None and np.hypot(*(np.asarray(t["xy"]) - np.asarray(keep["xy"]))) < keep["r"] + st["tower_r"] * 1.5:
                continue
            if w is outer and cross is not None and min(np.hypot(*(np.asarray(t["xy"]) - q)) for q in cross) < 1.0:
                continue                                                         # the ends of the cut — at the inner court
            role_k = 1.0 if w is inner else 0.8
            plan_w = plan_main
            if len(W_) > 1 and w is inner and s["era"] in ("gothic", "late") and "square" in cu["tower_plan"] and not cu.get("keep") == "spire" \
                    and kn["age"] >= 0.3:
                plan_w = "square"                                                # the core is older than the rings: square Romanesque towers
            if t["role"] == "flank" and plan_w == "round" and "D" in cu["tower_plan"]: plan_w = "D"
            tw = place_tower(t, w["poly"], st, plan_w, roof_main, st["tower_r"] * role_k * (1.1 if t["role"] == "corner" else 0.92),
                             st["tower_up"] * role_k * (1.15 if t["role"] == "corner" else 0.9), st["lean"])
            tw["ward"] = w["id"]; TW.append(tw)
    for wing in (BR["wings"] if BR is not None else []):
        a_, b_ = wing[0], wing[-1]; L_ = float(np.hypot(*(b_ - a_))); d_ = (b_ - a_) / max(L_, 1e-9); nk = int(L_ // st["spacing"])
        for j in range(0, nk + 2):
            q = a_ + d_ * (L_ * j / (nk + 1)) if nk >= 0 else a_
            if any(np.hypot(*(q - np.asarray(t_["on"]))) < st["tower_r"] * 3 for t_ in TW): continue
            o_ = -BR["tb"]; tw = {"xy": (q + o_ * st["tower_r"] * 0.4).round(3).tolist(), "on": q.round(3).tolist(), "plan": plan_main,
                                  "r": round(st["tower_r"], 2), "up": round(st["tower_up"], 2), "roof": roof_main, "dir": d_.round(4).tolist(),
                                  "role": "corner" if j in (0, nk + 1) else "flank", "lean": st["lean"], "ward": "inner"}
            TW.append(tw)
    for g in gates:                                                              # gate towers
        d = np.asarray(g["dir"]); p = np.asarray(g["xy"]); o = np.asarray(g["out"])
        if g["kind"] == "simple": continue
        if g["kind"] == "tower":                                                 # gatehouse tower: the passage goes through the tower
            g["r"] = round(st["tower_r"] * 1.2, 2); g["up"] = round(st["tower_up"] * 1.25, 2); g["roof"] = roof_main
        else:
            big = g["kind"] == "gatehouse"; r = st["tower_r"] * (1.25 if big else 0.95)
            for sgn in (-1, 1):
                q = p + d * sgn * (g["w"] / 2 + 1.4 + r) + o * r * (0.55 if not big else 0.35)
                TW.append({"xy": q.round(3).tolist(), "on": (p + d * sgn * (g["w"] / 2 + 1.4 + r)).round(3).tolist(),
                           "plan": "D" if plan_main in ("round", "D") or big else plan_main, "r": round(r, 2),
                           "up": round(st["tower_up"] * (1.35 if big else 1.1), 2), "roof": roof_main, "dir": d.tolist(),
                           "role": "gate", "ward": g["ward"], "lean": 0.0, "face": o.round(4).tolist()})

    ex_n = {}
    for t in TW: ex_n[exposure(S, np.asarray(t["on"]))] = ex_n.get(exposure(S, np.asarray(t["on"])), 0) + 1
    note("towers", "count and spacing", f"{len(TW)} towers, spacing {st['spacing']:.0f} m", "spacing — by era and culture; ×2.6 at a cliff, ×2.2 at water, ×1.6 on a steep slope"
         + "; at the edge: " + ", ".join(f"{k} {v}" for k, v in sorted(ex_n.items(), key=lambda q: -q[1]))
         + ("; the core is square (older than the rings)" if len(W_) > 1 and kn["age"] >= 0.3 and "square" in cu["tower_plan"] else ""),
         "mood= (more frequent towers), era= (spacing), age= (an old core)")

    # court terraces: the ground drop of the court → levels with retaining walls (Bran, Pena)
    for w in W_:
        if w.get("ring") or w.get("rock") or w.get("bastions") or BR is not None: continue
        tr, R_, why = terraces(G, F, w, st, kn, rx)
        if tr is not None: w["terraces"] = tr
        note("terraces", f"court {w['id']}", f"{len(tr['levels'])} levels" if tr else "one level", why,
             f"terraces= (drop threshold: at 0 — from {CX['terraces']['range_min'][0]} m, at 1 — from {CX['terraces']['range_min'][1]} m)")

    # relief edits → the castle's place (for wall heights, buildings and the road — already on the new relief)
    Hn, level, levels = terrain_edits(G, F, W_, ditches, st, keep)
    for w, z in zip(W_, levels): w["level"] = None if z is None else round(z, 2)
    if st["rocky"]:
        note("court", "alignment", f"{lerp(CX['rock']['flatten'], kn['rock']):.0%} toward the court level",
             "rocky place: part of the rock relief stays in the court", "rock= (less leveling — more rock)")
    out = out or os.path.join(ROOT, "out/castle", os.path.basename(folder.rstrip("/")) + f"_{s['culture']}_{s['seed']}")
    os.makedirs(out, exist_ok=True)
    z = dict(np.load(os.path.join(folder, "relief.npz"))); z["near"] = Hn.astype(z["near"].dtype); z["level"] = level.astype(z["level"].dtype)
    np.savez_compressed(os.path.join(out, "relief.npz"), **z)
    shutil.copy(os.path.join(folder, "location.json"), os.path.join(out, "location.json"))
    G2 = Ground(out)

    # walls: the court outlines (at the cut — an inner wall), a shield wall on the threat side
    walls = []
    shield = cu.get("shield_wall") and site.get("shield") is not None
    for w in W_:
        P = np.asarray(w["poly"]); n = len(P)
        if w is outer and cross is not None:                                     # outer court: no wall along the cut
            cut = np.asarray(cross); keep_e = []
            for i in range(n):
                a, b = P[i], P[(i + 1) % n]
                on = all(min(np.hypot(*(q - cut[0])), np.hypot(*(q - cut[1]))) < 0.6 or abs(cross2(cut[1] - cut[0], q - cut[0])) / max(np.linalg.norm(cut[1] - cut[0]), 1e-9) < 0.6 for q in (a, b))
                keep_e.append(not on)
            i = 0
            while i < n:
                if not keep_e[i]: i += 1; continue
                j = i
                while j < n and keep_e[j]: j += 1
                pts, walk, base = wall_profile(G2, P, i, j, st, w.get("level"), 0.85)
                walls.append({"ward": w["id"], "kind": "curtain", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                              "base": base.round(2).tolist(), "t": round(st["wall_t"], 2), "closed": False})
                i = j
            continue
        mult = 0.8 if (w.get("ring") or w.get("bailey")) else 1.0
        pts, walk, base = wall_profile(G2, P, 0, n, st, w.get("level"), mult, rock=bool(w.get("rock")))
        T = np.full(len(pts), st["wall_t"])
        if shield and w is inner:
            dd = ((pts - c) @ thr); m = dd > np.percentile(dd, 78)                 # shield wall: thicker and taller toward the threat
            T = np.where(m, st["wall_t"] * 2.2, T); walk = np.where(m, walk + st["wall_h"] * 0.6, walk)
        if site.get("cliff_wall") == "low":
            ex = np.array([exposure(S, p) == "cliff" for p in pts]); walk = np.where(ex, walk - st["wall_h"] * 0.35, walk)
        walls.append({"ward": w["id"], "kind": "ring" if mult < 1 else "curtain", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                      "base": base.round(2).tolist(), "t": [round(float(x), 2) for x in T], "closed": True})
    for wing in (BR["wings"] if BR is not None else []):                         # barrier: wall wings up the slopes
        pts, walk, base = wall_profile(G2, wing, 0, len(wing) - 1, st, None, 1.0)
        walls.append({"ward": "inner", "kind": "barrier", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                      "base": base.round(2).tolist(), "t": round(st["wall_t"], 2), "closed": False})
    if cross is not None:                                                         # wall between courts
        cut = np.asarray(cross); Pc = np.vstack([cut[0], cut[1]])
        pts, walk, base = wall_profile(G2, Pc, 0, 1, st, None, 0.9)
        walls.append({"ward": "inner", "kind": "cross", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                      "base": base.round(2).tolist(), "t": round(st["wall_t"], 2), "closed": False})

    # merlons and stairstep walls (Bran, Taufers; a tidy stairstep rather than a mess of merlons and ledges):
    # between the towers — an even row of merlons; a wall on a slope goes in ledges — a step is a whole number of
    # merlons, a ledge is in a clearance
    sp = CX["steps"]; cd = K["canon"]["crenel"]; p_c = cd["w"] + cd["gap"]; st["crenel"] = dict(cd)
    p_st = min(1.0, sp["cultures"].get(s["culture"], 0.5) * kn["steps"] * 1.4); n_st = 0; n_r = 0; skip_st = 0; ks = []
    for wl in walls:
        if wl["kind"] not in ("curtain", "ring", "cross", "barrier"): continue
        Pw = np.asarray(wl["pts"]); wk = np.asarray(wl["walk"], float); bs = np.asarray(wl["base"], float); seg = np.hypot(*np.diff(Pw, axis=0).T)
        Tw = np.asarray(wl["t"], float) if isinstance(wl["t"], list) else np.full(len(Pw), float(wl["t"]))
        gm = float((np.abs(np.diff(wk)) * (np.abs(np.diff(wk)) < 1.5) * (seg > 0.1)).sum() / max(seg.sum(), 1e-6))   # excluding the drops of a shield wall
        h_ = None
        if gm >= sp["grade_min"] and len(Pw) >= 6:
            if rx.random() < p_st:
                if wl["kind"] == "barrier":                                       # the barrier climbs the slope: a steeper walkway, in ledges
                    Pw, wk, bs = wall_profile(G2, Pw[[0, -1]], 0, 1, st, None, 1.0, grade=sp["grade_max"]); Tw = np.full(len(Pw), Tw[0])
                h_ = lerp(sp["h"], (gm - sp["grade_min"]) / 0.25)                    # ledge — by slope; for the kremlin — merlon by merlon
                kc = sp.get("k_culture", {}).get(s["culture"])
                if kc is not None: h_ = max(sp["riser_min"], gm * p_c * kc)
            else: skip_st += 1
        each = h_ is not None and sp.get("k_culture", {}).get(s["culture"]) == 1
        cuts = wall_cuts(Pw, wk, TW, gates, cd["gap"])
        Pw, wk, bs, Tw, grid, info = crenel_layout(Pw, wk, bs, Tw, bool(wl["closed"]), cuts, p_c, h_, each)
        wl.update(pts=Pw.round(3).tolist(), walk=wk.round(2).tolist(), base=bs.round(2).tolist(), t=[round(float(x), 2) for x in Tw], crenels=grid)
        if info is not None and info["n"]:
            wl["steps"] = dict(info, grade=round(gm, 3)); n_st += 1; n_r += info["n"]; ks.append(info["h"])
    note("walls", "stairstep", f"{n_st} walls, {n_r} ledges" + (f" {min(ks):.1f}–{max(ks):.1f} m each" if ks else "") if n_st else "no",
         f"a wall on a slope (average walkway slope ≥ {sp['grade_min']:.0%}) goes in ledges with probability {p_st:.0%} "
         f"(culture {sp['cultures'].get(s['culture'], 0.5)} × knob steps {kn['steps']}); all ledges of a wall are equal (1.1–2.4 m by slope, no smaller than "
         "the rise between neighboring merlons; kremlin — merlon by merlon), "
         "a step is a whole number of merlons, a ledge is in a clearance between merlons; between towers — an even row of merlons"
         + (f"; {skip_st} walls on a slope — with a level walkway" if skip_st else ""), "steps= (share of walls with ledges), culture=")

    # wall on the rock edge: masonry begins where the rock ends (Bran, Eltz) — a rock strip under the wall
    kr = CX["rock"]; wdrop = lerp(kr["wall_drop_min"], kn["rock"]); n_rk = 0
    for wl in walls:
        Pw = np.asarray(wl["pts"]); wk = np.asarray(wl["walk"], float); bs = np.asarray(wl["base"], float)
        Tw = np.asarray(wl["t"], float) if isinstance(wl["t"], list) else np.full(len(Pw), float(wl["t"]))
        Tg = np.gradient(Pw, axis=0); Tg /= np.maximum(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-9); Nw = np.column_stack([Tg[:, 1], -Tg[:, 0]])
        zf = G2.height(Pw[:, 0] + Nw[:, 0] * (Tw / 2 + 0.3), Pw[:, 1] + Nw[:, 1] * (Tw / 2 + 0.3))
        zi = G2.height(Pw[:, 0] - Nw[:, 0] * (Tw / 2 + 1.5), Pw[:, 1] - Nw[:, 1] * (Tw / 2 + 1.5))
        sl = G.slope(Pw[:, 0] + Nw[:, 0] * (Tw / 2 + 1.2), Pw[:, 1] + Nw[:, 1] * (Tw / 2 + 1.2))   # rock — by the original relief, not the moat batter
        nz = np.convolve(rx.normal(0, 1, len(Pw) + 8), np.ones(5) / 5, "same")[4:-4] * 1.6   # the boundary between masonry and rock is uneven
        rt = np.minimum(np.minimum(zi - 0.8, wk - 3.0), zi - 0.8 + nz); rt = np.maximum(rt, zf + 0.6)
        ok = (zi - zf >= wdrop) & (sl >= kr["slope_min"]) & (rt > bs + 0.8)
        ok = np.convolve(ok.astype(float), np.ones(3) / 3, "same") > 0.5                  # without isolated stations
        if ok.sum() >= 3:
            wl["rock_top"] = [round(float(v), 2) if o else None for v, o in zip(rt, ok)]; n_rk += int(ok.sum())
    note("rock", "walls on the edge", f"{n_rk * 1.5:.0f} m of walls" if n_rk else "no",
         f"drop court − ground at the outer face ≥ {wdrop:.1f} m and slope ≥ {kr['slope_min']}°: masonry from the edge, rock below",
         "rock= (drop threshold: at 0 — 3.5 m, at 1 — 1.2 m)")

    def walk_at(p):
        best = (1e9, None)
        for wl in walls:
            Pw = np.asarray(wl["pts"]); d = np.hypot(Pw[:, 0] - p[0], Pw[:, 1] - p[1]); i = int(np.argmin(d))
            if d[i] < best[0]: best = (d[i], wl["walk"][i])
        return best[1] if best[1] is not None else float(G2.height(p[0], p[1]) + st["wall_h"])

    n_pl = 0; ang16 = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for t in TW:
        p = np.asarray(t["xy"]); r = t["r"]; ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        zz = G2.height(p[0] + r * np.cos(ang), p[1] + r * np.sin(ang))
        t["base"] = round(float(min(zz.min(), G2.height(p[0], p[1]))) - 1.0, 2); t["walk"] = round(float(walk_at(t["on"])), 2)
        t["top"] = round(t["walk"] + t["up"], 2)
        zm, rk = plinth(G2, p + (r + 0.4) * np.column_stack([np.cos(ang16), np.sin(ang16)]), kn, top=t["top"], G0=G)
        if rk: t["base"] = round(zm, 2); t["rock"] = rk; n_pl += 1
    if keep is not None:
        p = np.asarray(keep["xy"]); r = keep["r"]; ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        zz = G2.height(p[0] + r * np.cos(ang), p[1] + r * np.sin(ang)); keep["base"] = round(float(zz.min()) - 1.0, 2)
        keep["top"] = round(float(max(zz.max(), walk_at(p) if keep["rule"] == "corner_threat" else zz.max())) + keep["up"], 2)
        zm, rk = plinth(G2, p + (r + 0.5) * np.column_stack([np.cos(ang16), np.sin(ang16)]), kn, top=keep["top"], G0=G)
        if rk: keep["base"] = round(zm, 2); keep["rock"] = rk; n_pl += 1
    for g in gates:
        p = np.asarray(g["xy"]); g["z"] = round(float(G2.height(p[0], p[1])), 2); g["walk"] = round(float(walk_at(p)), 2)

    # fantasy: towers by tiers — the main one by the style of the purpose or culture, the wall ones by the
    # culture's style
    from components.buildings.castles import castle_tower as CT
    for t in ([keep] if keep is not None and keep.get("style") else []) + ([t for t in TW if t["role"] in ("corner", "flank", "gate")] if cu.get("tower_style") else []):
        mh = K["tower_styles"][t.get("style") or cu["tower_style"]]["slender"] * 2 * float(t["r"])   # slenderness — from the bottom of the masonry (on a slope)
        if t["top"] - t["base"] > mh: t["top"] = round(t["base"] + mh, 2)
    if keep is not None and keep.get("style"):
        keep.update(CT.stages(keep, K["tower_styles"][keep["style"]], rx, st["storey"], "keep"))
        note("towers", "main tower by tiers", f"{keep['style']}: {keep['top'] - keep['base'] + keep['spire']['h']:.0f} m with the roof", keep["why"],
             "purpose= (mage, sorcerer, vampire), culture= (gothic, vampire, sorcerer), vertical=")
    if cu.get("tower_style"):
        for t in TW:
            if t["role"] in ("corner", "flank", "gate"):
                t.update(CT.stages(t, K["tower_styles"][cu["tower_style"]], rx, st["storey"], t["role"])); t["style"] = cu["tower_style"]
        if TW: note("towers", "wall towers by tiers", f"{sum(1 for t in TW if t.get('stages'))} towers, style {cu['tower_style']}",
                    "the culture builds towers in tiers: " + TW[0].get("why", ""), "culture=")

    # buildings
    items = []
    for i in pu["items"]: items += [i] * (4 if i == "convent" else 1)          # Order castle: four wings around the court
    gate_xy = [g["xy"] for g in gates]
    for w in W_:
        if w.get("ring"): w["holes"] = [w["hole"] if w.get("hole") is not None else inner["poly"]]
    pre = []
    if inner.get("rock"):                                                        # on a rock: the main building spans the whole rock
        core = next((i for i in items if K["buildings"][i]["side"] == "safe"), items[0] if items else None)
        R = fill_rect(F, inner["poly"], keep, st["wall_t"] / 2 + 0.4) if core else None
        if R is not None:
            Bd = K["buildings"][core]; d = (R[1] - R[0]) / max(np.linalg.norm(R[1] - R[0]), 1e-9)
            pre.append({"kind": core, "ward": "inner", "poly": R.round(3).tolist(), "c": R.mean(0).round(3).tolist(), "axis": d.round(4).tolist(),
                        "L": round(float(np.hypot(*(R[1] - R[0]))), 2), "D": round(float(np.hypot(*(R[2] - R[1]))), 2),
                        "storeys": Bd["storeys"] + 1, "roof": "flat" if st["flat_roofs"] else ("hip" if Bd["roof"] == "lean" else Bd["roof"]), "apse": False})
            items = [i for i in items]; items.remove(core)
    wb = [w for w in W_ if not w.get("rock")] or W_
    B, occ, masks = buildings(items, wb, st, gate_xy, keep, S, F, rng, TW, pre)
    tight = float(np.clip(1.4 - st["A_inner"] / CX["infill"]["tight_area"], 0.3, 1.0)) if not (st["rocky"] or S["type"] in ("spur", "summit", "rock")) else 1.0
    wi_ = [w for w in wb if not w.get("bastions")] if BR is None else []
    NB, fs = infill(wi_, masks, occ, B, st, gate_xy, S, F, rx, TW, s["culture"], tight) if kn["density"] > 0.02 and wi_ else ([], {})
    B += NB
    n_th = tower_houses(B, st, rx, s["culture"], tight)
    if n_th: note("buildings", "houses-towers", f"{n_th}", f"a cramped castle and vertical {kn['vertical']}: dwellings rise 5–8 floors; a long "
                  "block — 2–3 houses of different heights built tight together (Eltz)", "vertical= (from 0.55), culture=")
    n_fp = flush_program(B, W_, st, TW, gates, S, rx, lerp(CX["infill"]["as_wall"], kn["density"]) * tight) if tight >= 0.8 and BR is None else 0
    if n_fp: note("buildings", "houses as a wall", f"{n_fp} buildings of the program", "a cramped castle: the palas and other houses at a safe wall "
                  "(cliff, steep slope, water) form the outer wall of the circuit — windows outward only above the walkway", "density= (share), site= (tightness)")
    note("buildings", "growth by annexes", f"{len(NB)} houses along the walls" + (f", {fs.get('like_wall', 0)} as a wall" if fs.get("like_wall") else "")
         + (f", {fs.get('n_tower_houses', 0)} tower houses" if fs.get("n_tower_houses") else ""),
         f"program '{s['purpose']}' — {len(B) - len(NB)} buildings; then houses up to the share of the wall frontage {kn['density']:.0%} (knob density), "
         f"tight against the neighbors; 'house as a wall' — {lerp(CX['infill']['as_wall'], kn['density']) * tight:.0%} (tightness {tight:.1f}); "
         f"the frontage fell short by {fs.get('needed_m', 0):.0f} m" + (f"; refusals: {fs['rejections']}" if fs.get("rejections") else ""),
         "density= (building density), vertical= (floors, tower houses)")
    for w in W_: w.pop("holes", None)
    # "houses as a wall": a wall stretch that has become the outer wall of a house is not built in the build; the
    # bottom of the house goes down to the bottom of the wall
    for bi, b in enumerate(B):
        if not b.get("flush"): continue
        Rb = np.asarray(b["poly"]); cb = Rb.mean(0); Rx = cb + (Rb - cb) * (1 + 0.8 / max(min(b["L"], b["D"]), 1.0))
        for wi, wl in enumerate(walls):
            if wl["ward"] != b["ward"] or wl["kind"] not in ("curtain", "ring"): continue
            Pw = np.asarray(wl["pts"]); m_ = inside(Rx, Pw[:, 0], Pw[:, 1])
            if m_.sum() < 2: continue
            ii = [int(i) for i in np.flatnonzero(m_)]
            wl["absorbed"] = sorted(set(wl.get("absorbed", [])) | set(ii))
            b["flush"] = {"wall": wi, "ii": ii, "walk": round(float(np.max(np.asarray(wl["walk"])[ii])), 2),
                          "base": round(float(np.min(np.asarray(wl["base"])[ii])), 2)}
            break
        if b["flush"] is True: b.pop("flush")                                    # no wall found — an ordinary house
    for b in B:
        R = np.asarray(b["poly"]); zz = G2.height(R[:, 0], R[:, 1]); zc = float(G2.height(*R.mean(0)))
        b["base"] = round(float(min(zz.min(), zc)) - 0.5, 2); b["floor"] = round(float(np.median(np.append(zz, zc))), 2)
        Rr = np.vstack([R[i] + (R[(i + 1) % 4] - R[i]) * f for i in range(4) for f in (0.0, 0.5)]); cc = R.mean(0)
        Rr = cc + (Rr - cc) * (1 + 0.6 / max(np.linalg.norm(Rr - cc, axis=1).min(), 1.0))
        zm, rk = plinth(G2, Rr, kn, min_body=0.0, G0=G) if not b.get("flush") else (0.0, None)
        if rk:                                                                   # house on a rock: the floor is at the top of the rock
            b["base"] = round(zm, 2); b["rock"] = rk; b["floor"] = round(max(b["floor"], zm + 0.4), 2); n_pl += 1
        if b.get("flush"):                                                       # house as a wall: the bottom is like the wall's (or the top of the rock under it)
            fw = walls[b["flush"]["wall"]]; rt = [fw.get("rock_top", [None] * len(fw["pts"]))[i] for i in b["flush"]["ii"]]
            b["base"] = round(min(b["base"], min(v for v in rt if v is not None) if any(v is not None for v in rt) else b["flush"]["base"]), 2)
        b["eave"] = round(b["floor"] + b["storeys"] * st["storey"], 2)
        b["pitch"] = st["roof_pitch"] if b["roof"] != "lean" else max(20.0, st["roof_pitch"] * 0.6)
    # retaining walls of the terraces: along the level boundaries, except stretches under houses and towers; the
    # stair is on the path from the gate inward
    tt_ = CX["terraces"]["t"]; n_ret = 0; target = np.asarray(keep["xy"]) if keep is not None else np.asarray(inner["poly"]).mean(0)
    route = np.array([gates[0]["xy"], target], float)
    for w in W_:
        tr = w.get("terraces")
        if not tr: continue
        g = np.asarray(tr["dir"], float)
        for ln in tr["lines"]:
            a_, b_ = np.asarray(ln["pts"][0], float), np.asarray(ln["pts"][1], float); L_ = float(np.hypot(*(b_ - a_)))
            pts = a_ + (b_ - a_) * np.linspace(0, 1, max(2, int(L_ / 1.5) + 1))[:, None]
            zu = G2.height(pts[:, 0] + g[0] * (tt_ / 2 + 1.0), pts[:, 1] + g[1] * (tt_ / 2 + 1.0))
            zl = G2.height(pts[:, 0] - g[0] * (tt_ / 2 + 1.0), pts[:, 1] - g[1] * (tt_ / 2 + 1.0))
            ok = zu - zl > 0.8
            for b in B:
                Rb = np.asarray(b["poly"]); cb = Rb.mean(0); ok &= ~inside(cb + (Rb - cb) * 1.08, pts[:, 0], pts[:, 1])
            for t in TW + ([keep] if keep is not None else []):
                ok &= np.hypot(pts[:, 0] - t["xy"][0], pts[:, 1] - t["xy"][1]) > t["r"] + 0.6
            j = 0
            while j < len(pts):
                if not ok[j]: j += 1; continue
                k = j
                while k < len(pts) and ok[k]: k += 1
                if k - j >= 3:
                    Q = pts[j:k]; dq = [pl_dist(route, q) for q in Q]
                    walls.append({"ward": w["id"], "kind": "retaining", "pts": Q.round(3).tolist(), "walk": (zu[j:k] + 0.05).round(2).tolist(),
                                  "base": (zl[j:k] - 0.6).round(2).tolist(), "t": tt_, "closed": False, "level": ln["k"],
                                  "stair": int(np.argmin(dq))})
                    n_ret += 1
                j = k
    if n_ret:
        note("terraces", "retaining walls", f"{n_ret}", "boundary of the court levels; under houses and towers their role is played by the walls of buildings; "
             "the stair is at the point nearest to the path from the main gate to the main tower", "terraces=")
    note("rock", "plinths", f"{n_pl} (towers, the main tower, houses)",
         f"ground drop under the footprint ≥ {lerp(CX['rock']['drop_min'], kn['rock']):.1f} m and at the lower side ≥ {CX['rock']['slope_min']}°: "
         "masonry from the upper side, rock below", "rock= (at 0 — from 4.5 m, at 1 — from 1.2 m)")

    # bridge and road: the approach keeps its right side to the wall (Vitruvius) — the last stretch runs along the
    # wall clockwise
    g0 = gates[0]; gp = np.asarray(g0["xy"]); o = np.asarray(g0["out"]); bridges = []; span = 0.0
    for D in ditches:
        if pl_dist(D["pts"], gp) < st["wall_t"] / 2 + 4 + D["w"]: span = max(span, st["wall_t"] / 2 + 2.5 + D["w"] + 1.5)
    if span > 0:
        a_ = gp + o * (st["wall_t"] / 2 + 0.5); b_ = gp + o * (st["wall_t"] / 2 + span)
        bridges.append({"pts": [a_.round(3).tolist(), b_.round(3).tolist()], "w": round(g0["w"] + 0.6, 2), "deck": g0["z"]})
    front = gp + o * (st["wall_t"] / 2 + span + 4.0)
    road = None; along = 0.0; n_path = 0
    if BR is not None:
        Rp = np.asarray(G.loc["paths"][BR["road"]]["pts"], float); road = np.column_stack([Rp[:, :2], G2.height(Rp[:, 0], Rp[:, 1])]); n_path = len(road)
    if RM is not None and g0.get("foot") is not None:
        P1 = reach_path(RM, g0["foot"])
        if P1 is not None:
            end = gp + o * (st["wall_t"] / 2 + span * 0.5)
            road = np.vstack([P1, np.asarray(g0["foot"])[None], end[None]])
            road = np.column_stack([road, G2.height(road[:, 0], road[:, 1])])
            Rg = grade_road(G2, out, np.vstack([P1, np.asarray(g0["foot"])[None]]), 4.0 * st["k"] ** 0.5,
                            keep_out=[(w["poly"], st["wall_t"] * 1.2 + st["tower_r"] + 2) for w in W_])
            ramp = np.array([np.asarray(g0["foot"]), end]); zr = [Rg[-1, 2], g0["z"] if not bridges else bridges[0]["deck"]]
            road = np.vstack([Rg, np.column_stack([ramp, zr])[1:]]); n_path = len(Rg)
            G2 = Ground(out)
            wl_ring = np.asarray(gw0["poly"]); dd = np.array([pl_dist(np.vstack([wl_ring, wl_ring[:1]]), q) for q in road[-25:, :2]])
            along = float((dd < st["wall_t"] / 2 + span + 12).sum() * 2.0)

    # bridge over a chasm: for the vampire — always, if there is a chasm; for the others — when otherwise there
    # would be only a stair
    via_bridge = False
    if BR is None and not bridges and (pu.get("bridge") or approach_kind in ("stair", None)):
        cb = chasm_bridge(G2, g0, W_, TW, st)
        if cb is not None:
            bridges.append(cb); land = np.asarray(cb["land"])
            if RM is not None:                                                    # the road goes to the other end of the bridge
                P2 = reach_path(RM, land)
                if P2 is not None:                                                # the approach — the same kind of path, but to the end of the bridge
                    road = np.column_stack([np.vstack([P2, land[None]]), G2.height(*np.vstack([P2, land[None]]).T)]); n_path = len(road)
                    via_bridge = True
            note("gate", "bridge over a chasm", f"{cb['len']:.0f} m over a chasm {cb['depth']:.0f} m deep",
                 "the gate is separated from the approach by a ravine: a straight span at gate level to the near edge, piers down to the bottom (Hunyadi, Ravenloft)",
                 "purpose=vampire_castle (a bridge always, if there is a chasm)")

    # unique elements: wall stretches and their surroundings → elements by the rules (castle_features.py)
    from components.buildings.castles import castle_features as CF
    part = {"walls": walls, "towers": TW, "gates": gates, "keep": keep, "buildings": B, "bridges": bridges, "ditches": ditches,
            "wards": [dict(w, poly=np.asarray(w["poly"])) for w in W_], "road": None if road is None else road.tolist()}
    fseed = int(s["seed"]) * 7919 + zlib.crc32(os.path.basename(folder.rstrip("/")).encode())     # each place gets its own decisions
    fctx = dict(G2=G2, S=S, st=st, spec=s, thr=thr, rng=np.random.default_rng(fseed), exposure=exposure, inside=inside,
                wall_profile=wall_profile)
    features = CF.place(part, fctx) if not s.get("plain_walls") else []
    part["features"] = features
    fk = {}
    for f in features: fk.setdefault(f["kind"], []).append(f["why"])
    for kind_, whys in sorted(fk.items(), key=lambda q: -len(q[1])):
        note("elements", kind_, len(whys), whys[0], "features.yaml — the conditions; seed= — different decisions under the same rules")
    if part.get("roof_share") is not None:
        note("roofs", "share of roofed towers", part["roof_share"], "culture share × wealth (knob wealth), a cap of 65 %; one tall roof is a landmark",
             "wealth=, culture=")

    # checks
    checks = []
    def chk(what, ok, **kw): checks.append(dict({"what": what, "ok": bool(ok)}, **kw))
    for w in W_:
        P = np.asarray(w["poly"]); sa = abs(area(P))
        chk(f"court {w['id']}: outline without self-intersections", simple_ok(F, P), area_m2=round(sa))
    chk("the main court is no smaller than 25 % of the intent and 150 m²", st["A_inner"] >= max(150.0, 0.25 * st["area"] * st["fit"] ** 2), area_m2=round(st["A_inner"]))
    deep = max((max(np.asarray(wl["walk"]) - np.asarray(wl["base"])) for wl in walls), default=0)
    chk("walls: height with the foundation ≤ 2.5 × wall height + 12 m", deep <= st["wall_h"] * 2.5 + 12, max_m=round(float(deep), 1))
    hang = []
    for wl in walls:
        Pw = np.asarray(wl["pts"]); zg = G2.height(Pw[:, 0], Pw[:, 1]); hang.append(float((np.asarray(wl["base"]) - zg).max()))
    chk("walls stand on the ground (the bottom is not above the ground)", max(hang) <= 0.01, max_m=round(max(hang), 2))
    wet_t = [t for t in TW if G2.water(*t["xy"]) > 1.5]
    chk("towers not in deep water (> 1.5 m)", not wet_t, within_water=len(wet_t))
    if road is not None:
        rp = road[:max(2, n_path)]; sl = np.abs(np.diff(rp[:, 2])) / np.maximum(np.hypot(*np.diff(rp[:, :2], axis=0).T), 1e-6)
        lim = {"road": K["canon"]["road_slope_max"], "trail": 0.4, "stair": 1.0, "road of the place": 0.4}[approach_kind]
        chk(f"approach to the gate: {approach_kind}, slope p95 ≤ {lim:.0%}", np.percentile(sl, 95) <= lim, p95=round(float(np.percentile(sl, 95)), 3),
            length_m=round(float(np.hypot(*np.diff(road[:, :2], axis=0).T).sum())))
        if BR is not None: chk("the road passes through the outpost (both gateways)", all(pl_dist(road[:, :2], np.asarray(g["xy"])) < 6 for g in gates[:2]))
        elif not via_bridge: chk("ramp to the gate: slope ≤ 30 %", g0.get("ramp", 1) <= 0.3, slope=g0.get("ramp"), length_m=g0.get("ramp_m"))
        cbs = [b_ for b_ in bridges if b_.get("kind") == "chasm"]
        if cbs: chk("bridge over a chasm: the deck is level (≤ 10 %), deeper than 12 m beneath it", all(abs(float(G2.height(*b_["land"])) - b_["deck"]) <= max(4.0, 0.1 * b_["len"])
                                                                                        and b_["depth"] >= 12 for b_ in cbs), length_m=cbs[0]["len"], depth_m=cbs[0]["depth"])
    else:
        chk("approach to the gate found (a road, a path or a stair)", False)
    for w in W_:
        P = np.asarray(w["poly"]); Mw = F.fill(P); used = (occ & Mw).sum() / max(Mw.sum(), 1)
        chk(f"court {w['id']}: free ≥ {st['free_min']:.0%}", used <= 1 - st["free_min"] + 1e-6, occupied=round(float(used), 2))
    if keep is not None:
        kp = np.asarray(keep["xy"]); P = np.asarray(inner["poly"])
        chk("the main tower is in the inner court", bool(inside(P, kp[0], kp[1])) or keep["rule"] in ("corner_threat", "view"), rule=keep["rule"])
    chk("canon: gate", K["canon"]["gate_w"][0] <= gates[0]["w"] <= K["canon"]["gate_w"][1] * st["k"], width=gates[0]["w"])
    chk("canon: wall walkway ≥ 1.6 m", st["wall_t"] - 0.6 >= K["canon"]["walk_min"] - 1e-6, walk_m=round(st["wall_t"] - 0.6, 2))
    checks += CF.checks(part, fctx) if features else []
    stp = [wl["steps"]["max"] for wl in walls if wl.get("steps")]                # complex geometry
    if stp: chk("stairstep walls: ledge ≤ 4 m (two steps on a steep stretch)", max(stp) <= 4.0, max_m=max(stp), walls=len(stp))
    bad_r = 0; n_r = 0                                                            # tidy stairstep: a visible ledge = h, in a clearance of the row
    for wl in walls:
        if not wl.get("steps"): continue
        Pw = np.asarray(wl["pts"]); wk = np.asarray(wl["walk"]); sw = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Pw, axis=0).T))])
        for j in np.flatnonzero((np.hypot(*np.diff(Pw, axis=0).T) < 1e-3) & (np.abs(np.diff(wk)) > 0.05)):
            u = sw[j]; sec = [g for g in wl["crenels"] if g[0] - 1e-3 <= u <= g[0] + g[1] * p_c + 1e-3]
            if not sec: continue                                                  # at a tower or the edge of a shield wall — not visible
            n_r += 1; f_ = (u - sec[0][0]) / p_c
            if abs(f_ - round(f_)) > 0.01 or (not wl["steps"].get("each") and abs(abs(wk[j + 1] - wk[j]) - wl["steps"]["h"]) > 0.03): bad_r += 1
    if n_r: chk("stairstep walls: the visible ledges are equal and stand in the clearances of the row of merlons", bad_r == 0, n_ledges=n_r, non_conforming=bad_r)
    ret = [float(np.max(np.asarray(wl["walk"]) - np.asarray(wl["base"]))) for wl in walls if wl["kind"] == "retaining"]
    lim_r = max(8.0 * st["k"] ** 0.5, 0.9 * st["wall_h"])                         # a retaining wall is not taller than the castle's fighting wall
    if ret: chk(f"terraces: retaining wall ≤ {lim_r:.0f} m (not taller than the fighting wall)", max(ret) <= lim_r, max_m=round(max(ret), 1), walls=len(ret))
    pl = [t for t in TW + ([keep] if keep else []) if t.get("rock")]
    if pl: chk("rock plinths: masonry above the rock ≥ 6 m", min(t["top"] - t["base"] for t in pl) >= 6.0 - 1e-6,
               min_m=round(min(t["top"] - t["base"] for t in pl), 1), n_plinths=len(pl))
    fl = [b for b in B if isinstance(b.get("flush"), dict)]
    if fl: chk("houses as a wall: there is no wall under the house, the bottom of the house is not above the bottom of the wall",
               all(set(b["flush"]["ii"]) <= set(walls[b["flush"]["wall"]].get("absorbed", [])) and b["base"] <= b["flush"]["base"] + 1e-6
                   or any(walls[b["flush"]["wall"]].get("rock_top", [None])[i] is not None for i in b["flush"]["ii"]) for b in fl), n_houses=len(fl))
    stg = [t for t in TW + ([keep] if keep else []) if t.get("stages")]          # fantasy: towers by tiers
    if stg:
        bad_s = [t for t in stg if (t["top"] - t["base"]) / (2 * float(t["stages"][0]["r"])) > K["tower_styles"][t.get("style") or keep.get("style")]["slender"] + 0.05
                 or any(float(b2["r"]) > float(b1["r"]) * 1.07 + 1e-6 for b1, b2 in zip(t["stages"][:-1], t["stages"][1:]))]
        chk("towers by tiers: slenderness within the norm of the style, the upper tier rests on the lower one", not bad_s, n_towers=len(stg), non_conforming=len(bad_s))
    tw_ = [w for w in W_ if w.get("tier")]
    if len(tw_) > 1:
        wk_m = [float(np.mean([np.mean(wl["walk"]) for wl in walls if wl["ward"] == w["id"] and wl["kind"] != "retaining"] or [np.nan])) for w in tw_]
        chk("ring tiers: the wall of each ring is higher than the outer one", all(a_ > b_ for a_, b_ in zip(wk_m[:-1], wk_m[1:])), walk_m=[round(v, 1) for v in wk_m])
        top_ = np.asarray(S["top"][:2]); ang_ = [math.atan2(*(np.asarray(g["xy"]) - top_)[::-1]) for g in gates[:len(tw_)]]
        dif = [abs((a_ - b_ + math.pi) % (2 * math.pi) - math.pi) for a_, b_ in zip(ang_[:-1], ang_[1:])]
        chk("ring tiers: the gates of adjacent tiers are on different sides (≥ 75°)", all(d_ >= math.radians(75) for d_ in dif), angles=[round(math.degrees(d_)) for d_ in dif])
    dk = [f for f in features if f["kind"] == "dansker"]
    for f in dk: chk("projecting tower: an arcade 6–26 m above the ground, not in deep water", 6 <= f["deck"] - f["base"] - 1 <= 27 and G2.water(*f["p1"]) <= 2.5,
                     height_m=round(f["deck"] - f["base"] - 1, 1), length_m=f["len"])
    prog = [b.get("orig", b["kind"]) for b in B]; core = items[:2 if len(items) <= 3 else 3]
    miss = [i for i in core if prog.count(i) < core.count(i)]
    chk("main buildings of the program are placed", not miss, no=sorted(set(miss)), total=f"{len(B)} of {len(items)}")

    res = {"approach_kind": approach_kind + (" + bridge over a chasm" if via_bridge else "") if approach_kind else approach_kind, "spec": s, "style": {k: v for k, v in st.items() if k != "colors"},
           "colors": dict(st["colors"], rock=CX["rock"]["color"]), "crags": int(round(lerp(CX["rock"]["crags"], kn["rock"]) * (1.0 if st["rocky"] else 0.0))),
           "site": {k: S[k] for k in ("type", "type_ru", "q", "centre", "top", "area_m2", "rel_m", "edge_frac", "outline", "edge_cls", "approach")},
           "threat": thr.round(4).tolist(), "wards": [dict(w, poly=np.asarray(w["poly"]).round(3).tolist(),
                                                         **({"hole": np.asarray(w["hole"]).round(3).tolist()} if w.get("hole") is not None else {})) for w in W_],
           "cross": None if cross is None else np.asarray(cross).round(3).tolist(),
           "walls": walls, "towers": TW, "gates": gates, "keep": keep, "buildings": B, "ditches": ditches, "bridges": bridges,
           "features": features, "sections": part.get("sections", []), "logic": logic, "knobs": kn, "goal": s["goal"],
           "road": None if road is None else road.round(2).tolist(), "checks": checks, "place": folder, "out": out}
    res["stats"] = {"n_towers": len(TW), "n_buildings": len(B), "n_courts": len(W_), "n_elements": len(features), "walls_m": round(sum(np.hypot(*np.diff(np.asarray(w["pts"]), axis=0).T).sum() for w in walls if w["kind"] != "retaining")),
                    "n_ledges": sum(wl["steps"]["n"] for wl in walls if wl.get("steps")), "n_terraces": sum(len(w["terraces"]["levels"]) for w in W_ if w.get("terraces")),
                    "n_plinths": len(pl) + sum(1 for b in B if b.get("rock")), "n_annexes": len(NB), "like_wall": len(fl),
                    "checks": f"{sum(c['ok'] for c in checks)}/{len(checks)}"}
    try:                                                                         # space: courts, gates, wall walkways — as a summary (does not touch the castle checks)
        from components.buildings.castles import castle_space as CSP
        res["spaces"] = CSP.spaces(res, Ground(out).height).to_dict(); sc_ = res["spaces"]["checks"]
        res["stats"]["space"] = f"{sum(c_['ok'] for c_ in sc_)}/{len(sc_)}"; res["stats"]["n_rooms"] = res["spaces"]["stats"]["n_cells"]
    except Exception as e: res["stats"]["space"] = f"failure: {e}"
    json.dump(res, open(os.path.join(out, "castle.json"), "w"), ensure_ascii=False)
    return res


# ---------------------------------------------------------------- plan sheet
def draw(res, path, size=900, half=None):
    from PIL import Image, ImageDraw, ImageFont
    G = Ground(res["out"]); c = np.asarray(res["site"]["centre"])
    half = half or max(80.0, max(np.abs(np.asarray(res["wards"][-1]["poly"]) - c).max() for _ in [0]) + 45)
    xs = np.linspace(c[0] - half, c[0] + half, size); ys = np.linspace(c[1] + half, c[1] - half, size); X, Y = np.meshgrid(xs, ys)
    Z = G.height(X, Y); gy_, gx_ = np.gradient(Z, 2 * half / size); n = np.dstack([-gx_, gy_, np.ones_like(Z)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = np.array([-0.5, 0.6, 0.62]); L /= np.linalg.norm(L); sh = np.clip(n @ L, 0, 1)
    z = (Z - Z.min()) / max(np.ptp(Z), 1e-6); rgb = np.dstack([0.38 + 0.4 * z, 0.45 + 0.3 * z, 0.32 + 0.2 * z]) * (0.4 + 0.7 * sh[..., None])
    rgb[G.water(X, Y) > 0.05] = (0.25, 0.42, 0.68)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(im, "RGBA")
    px = lambda p: ((p[0] - c[0] + half) / (2 * half) * size, (c[1] + half - p[1]) / (2 * half) * size)
    col = res["colors"]; stone = tuple(int(255 * v) for v in col["stone"]); roof = tuple(int(255 * v) for v in col["roof"])
    for w in res["wards"]: d.polygon([px(p) for p in w["poly"]], fill=(235, 225, 200, 70))
    for D in res["ditches"]: d.line([px(p) for p in D["pts"]], fill=(70, 110, 170, 170) if D["wet"] else (90, 70, 50, 150), width=max(2, int(D["w"] / (2 * half) * size)))
    if res["road"]: d.line([px(p) for p in res["road"]], fill=(240, 220, 120, 230), width=3)
    for b in res["bridges"]: d.line([px(p) for p in b["pts"]], fill=(150, 120, 90, 255), width=max(3, int(b["w"] / (2 * half) * size)))
    for wl in res["walls"]:
        tw = wl["t"] if isinstance(wl["t"], float) else max(wl["t"])
        d.line([px(p) for p in wl["pts"]] + ([px(wl["pts"][0])] if wl["closed"] else []), fill=stone + (255,), width=max(3, int(tw / (2 * half) * size)))
    for b in res["buildings"]: d.polygon([px(p) for p in b["poly"]], fill=roof + (255,), outline=(30, 20, 15, 255))
    def shape(t, fill):
        p = np.asarray(t["xy"]); r = t["r"]; k = {"round": 20, "D": 20, "square": 4, "poly6": 6, "poly8": 8}.get(t["plan"], 12)
        a0 = math.atan2(t["dir"][1], t["dir"][0]) + (math.pi / 4 if k == 4 else 0) if t.get("dir") else 0
        rr = r / math.cos(math.pi / k) if k <= 8 else r
        d.polygon([px(p + rr * np.array([math.cos(a0 + 2 * math.pi * i / k), math.sin(a0 + 2 * math.pi * i / k)])) for i in range(k)], fill=fill, outline=(20, 20, 20, 255))
    for t in res["towers"]: shape(t, (150, 140, 125, 255) if t["role"] != "gate" else (200, 160, 90, 255))
    if res["keep"]: shape(res["keep"], (90, 60, 50, 255))
    for g in res["gates"]:
        x, y = px(g["xy"]); d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 40, 255))
    F_ = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    s = res["spec"]; d.rectangle([0, 0, size, 48], fill=(0, 0, 0, 200))
    d.text((8, 4), f"{res['site']['type_ru']} · {s['purpose']} · {s['era']} · {s['culture']} · {s['scale']} · {s['mood']}", fill="white", font=F_)
    d.text((8, 26), f"towers {res['stats']['n_towers']}, buildings {res['stats']['n_buildings']}, courts {res['stats']['n_courts']}, walls {res['stats']['walls_m']} m · checks {res['stats']['checks']}",
           fill=(210, 210, 210), font=F_)
    im.save(path)
    return path


def main():
    a = sys.argv[1:]; folder = a[0]; kw = dict(x.split("=", 1) for x in a[1:])
    at = kw.pop("at", None)
    if at and "," in at: at = [float(v) for v in at.split(",")]
    out = kw.pop("out", None)
    res = plan(folder, out=out, at=at, **kw)
    draw(res, os.path.join(res["out"], "plan.png"))
    print("__RESULT__", json.dumps({"out": res["out"], "place": res["site"]["type_ru"], **res["stats"],
                                    "non_passing": [c["what"] for c in res["checks"] if not c["ok"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
