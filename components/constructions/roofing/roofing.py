"""Roofing — layout of roof coverings: clay tile, slate, lemekh, shingle, seamed sheets, boards, stone slab (the roofing
part of the Constructions component; knowledge — coverings.yaml). Pure numpy: works in the project's Python and in
Blender (YAML only in the project's Python; in Blender the kinds arrive in the JSON plan).

The logic is masonry with differences: rows along the eaves from the bottom up (masonry goes by height), bonding
staggered / stacked / randomly (as for brick and rubble; random widths — masonry.fill_row), edges are features (masonry
has quoins and jambs, here the ridge, hip, finial). Differences: tiles overlap (row step = (length − lap) / layers), lie
at a slope (the bottom is raised by the thickness of the layers under it), the bottom is shaped (beaver tail, pointed,
fishscale), trough tiles go in columns.

Surfaces:
  Slope — a flat slope: a convex polygon in the world → a frame (O, eu along the eaves, ev up the slope, n outward), an
          outline (u, v), v = 0 is the eaves. world(x, v, h) — a point of the slope, h above the slope along the normal.
  Cone  — a tower cone: apex, eaves center, radius; x is the arc length at the radius of the row's bottom (r0), v is
          along the generatrix from the eaves. world(x, v, h, r0).
  Revolve — a body of revolution (onion, dome, helmet, bell, half-dome): a profile (r, z) around a vertical axis; v is
          the length along the generatrix from the bottom, x is the arc at r0; rings in zones (the tile count holds
          while the width stays within 0.7…1.35 w).
Generators: rows (flat and shaped tiles; diminishing rows — L, lap, t as a pair "eaves → ridge"), columns (trough
tiles), sheets (seam — panels from the eaves to the ridge, seams and transverse joints staggered; boards — boards in two
layers); on a tent and a dome sheets and boards are wedges toward the crown.
Layout: lay(surface, kind, seed, sid0) → tiles [{poly (x, v), v0, L, t, layers, r0, prof, depth, sid, tint, course}].
Roofs from scene faces: analyse(faces) → [{kind: cone|revolve|slopes, surfaces, ridges, hips, valleys, apex}].
Step plan: resolve(folder, keys) — the "roofing:" section of the run scene's constructions.yaml description.

  .venv/bin/python -m components.constructions.roofing.roofing resolve <folder> [covering=beaver] [towers=slate] [domes=lemekh] [tents=…] [detail=cam|all] out=<plan.json>
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "coverings.yaml")
from components.constructions.masonry import masonry as M  # noqa: E402


def load_library(path=LIB):
    """Roofing knowledge: kinds (base + own; name, title), substances. Needs PyYAML — the project's Python."""
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


def lerp(v, f):
    """A kind's value: a number or a pair [at the eaves, at the ridge] (diminishing rows) → at the fraction f of the slope height."""
    return float(v[0] + (v[1] - v[0]) * f) if isinstance(v, (list, tuple)) else float(v)


def gauge(k, f=0.5):
    """Row step: single — L − lap, double — (L − lap)/2, triple — /3 (diminishing rows — at the fraction f of the slope)."""
    return (lerp(k["L"], f) - lerp(k["lap"], f)) / max(int(k["n_layers"]), 1)


def width(k):
    w = k["w"]; return float(np.mean(w)) if isinstance(w, (list, tuple)) else float(w)


def relief(k):
    """Profile height of a trough tile above the slope: the S wave, the roll of the monk and capping tiles (as in
    bl_roofing.profiled_tile); flat — 0."""
    d = float(k["depth"]); w = width(k); p = k["profile"]
    if p == "s": return 1.6 * d
    if p == "monk_nun": return d + 0.288 * w                                # roll 0.64 of the column step
    if p == "troughs": return d + 0.36 * w                                        # capping tile 0.8 of the step
    return 0.0


# ── plane geometry ──────────────────────────────────────────────────────────────────────────────────────────────
def area(P):
    P = np.asarray(P, float)
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) if len(P) >= 3 else 0.0


def clip_convex(P, C):
    """Polygon P ∩ convex C (counter-clockwise) — Sutherland–Hodgman."""
    out = [tuple(map(float, p)) for p in P]
    for i in range(len(C)):
        if not out: break
        a, b = C[i], C[(i + 1) % len(C)]
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]; sp, sq = side(p), side(q)
            if sq >= -1e-12:
                if sp < -1e-12: t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
                out.append(q)
            elif sp >= -1e-12:
                t = sp / (sp - sq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


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


def outline(x0, x1, v0, L, low):
    """Tile outline (x, v) counter-clockwise: the bottom (butt) at v0 — by the bottom kind, the top — straight at v0 + L."""
    w = x1 - x0; xm = 0.5 * (x0 + x1)
    if low == "circular":                                                         # beaver tail: a segment, rise 0.3 w
        s = 0.3 * w; R = (w * w / 4 + s * s) / (2 * s); yc = v0 + R
        a0 = math.asin(min(1.0, (w / 2) / R))
        arc = [(xm + R * math.sin(a), yc - R * math.cos(a)) for a in np.linspace(-a0, a0, 9)]
        low_pts = arc
    elif low == "fishscale":                                                         # a semicircle across the full width
        r = w / 2; low_pts = [(xm - r * math.cos(a), v0 + r - r * math.sin(a)) for a in np.linspace(0, math.pi, 11)]
    elif low == "sharp":
        low_pts = [(x0, v0 + 0.35 * w), (xm, v0), (x1, v0 + 0.35 * w)]
    elif low == "hexagonal":
        c = 0.25 * w; low_pts = [(x0, v0 + c), (x0 + c, v0), (x1 - c, v0), (x1, v0 + c)]
    elif low == "crenellated":                                                    # lemekh with steps (stepped rectangular)
        a, b = 0.30 * w, 0.14 * w
        low_pts = [(x0, v0 + a), (x0 + 0.18 * w, v0 + a), (x0 + 0.18 * w, v0 + b), (x0 + 0.34 * w, v0 + b), (x0 + 0.34 * w, v0),
                   (x1 - 0.34 * w, v0), (x1 - 0.34 * w, v0 + b), (x1 - 0.18 * w, v0 + b), (x1 - 0.18 * w, v0 + a), (x1, v0 + a)]
    elif low == "tufts":                                                         # thatch: a ragged butt — the ends of the bundles
        rr = np.random.default_rng([int(abs(x0) * 1000) & 0x7FFFFFFF, int(abs(v0) * 1000) & 0x7FFFFFFF, 77])
        n = max(3, int(w / 0.06)); xs = np.linspace(x0, x1, n + 1)
        low_pts = [(float(x), v0 + float(rr.uniform(0.0, 0.05))) for x in xs]
    else:
        low_pts = [(x0, v0), (x1, v0)]
    return low_pts + [(x1, v0 + L), (x0, v0 + L)]


def _rng(sid):
    return np.random.default_rng([int(sid) & 0x7FFFFFFF, 2029])


# ── surfaces ────────────────────────────────────────────────────────────────────────────────────────────────────
class Slope:
    """Flat slope: pts — the vertices of a convex polygon in the world (3D), normal — outward (up). The eaves are the lowest
    horizontal edge (none — the lowest point): eu along it, ev up the slope, v = 0 is the eaves, u ≥ 0."""
    kind = "slope"

    def __init__(self, pts, normal, name="pitch"):
        P = np.asarray(pts, float); n = np.asarray(normal, float); n = n / max(np.linalg.norm(n), 1e-12)
        if n[2] < 0: n = -n
        ez = np.array([0.0, 0.0, 1.0]); ev = ez - (ez @ n) * n
        ev = ev / max(np.linalg.norm(ev), 1e-12) if np.linalg.norm(ev) > 1e-6 else np.array([0.0, 1.0, 0.0])
        eu = np.cross(ev, n); eu /= max(np.linalg.norm(eu), 1e-12)
        O = P[int(np.argmin(P[:, 2]))]
        uv = np.stack([(P - O) @ eu, (P - O) @ ev], 1)
        self.O = O + eu * uv[:, 0].min() + ev * uv[:, 1].min(); uv -= uv.min(0)
        hull = _hull(uv)
        self.eu, self.ev, self.n, self.poly, self.name = eu, ev, n, hull, name
        self.top = float(hull[:, 1].max()); self.width = float(hull[:, 0].max())

    def world(self, x, v, h, r0=None):
        x, v, h = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float), np.asarray(h, float))
        return self.O + x[..., None] * self.eu + v[..., None] * self.ev + h[..., None] * self.n

    def normal(self, x, v):
        return np.broadcast_to(self.n, np.broadcast(np.asarray(x), np.asarray(v)).shape + (3,))

    def nrm(self, x, v, r0=None):
        return np.array(self.normal(x, v), float)

    def uv(self, p):
        p = np.asarray(p, float) - self.O; return np.array([p @ self.eu, p @ self.ev])

    def cut_back(self, a, b, gap):
        """Valley a → b (world): the slope outline is set back from its line by gap inward — an open trough between the trimmed
        tiles."""
        A, B = self.uv(a), self.uv(b); d = (B - A) / max(np.linalg.norm(B - A), 1e-12); nrm = np.array([-d[1], d[0]])
        if (self.poly.mean(0) - A) @ nrm < 0: nrm = -nrm
        q = clip_half(self.poly, A + nrm * gap, nrm)
        if len(q) >= 3: self.poly = np.asarray(q, float)

    def chord(self, v0, v1):
        """The segment along u that the strip v0…v1 occupies in the slope outline (the union)."""
        q = clip_convex(self.poly, [(-1e6, v0), (1e6, v0), (1e6, v1), (-1e6, v1)])
        if len(q) < 3: return None
        q = np.asarray(q); return float(q[:, 0].min()), float(q[:, 0].max())


class Cone:
    """Tower cone: apex — the apex (3D), centre — the eaves center (3D), R — the eaves radius. x is the arc length at the radius
    r0 (the row's bottom), v is along the generatrix from the eaves; the tile narrows toward the apex together with the ring."""
    kind = "cone"

    def __init__(self, apex, centre, R, name="cone", th0=0.0, th1=2 * math.pi):
        self.A = np.asarray(apex, float); self.C = np.asarray(centre, float); self.R = float(R); self.name = name
        self.H = float(self.A[2] - self.C[2]); self.V = math.hypot(self.R, self.H); self.top = self.V
        self.th0, self.th1 = float(th0), float(th1)                              # eaves arc: a full circle or a fan (apse)
        self.full = self.th1 - self.th0 > 2 * math.pi - 1e-6

    def r(self, v):
        return self.R * (1.0 - np.asarray(v, float) / self.V)

    def world(self, x, v, h, r0=None):
        x, v, h = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float), np.asarray(h, float))
        th = x / max(float(r0 if r0 is not None else self.R), 1e-6)
        c, s = np.cos(th), np.sin(th)
        rr = self.r(v); z = self.C[2] + self.H * v / self.V
        nx, nz = self.H / self.V, self.R / self.V                                # outward normal: (H cosθ, H sinθ, R) / V
        return np.stack([self.C[0] + rr * c + h * nx * c, self.C[1] + rr * s + h * nx * s, z + h * nz], -1)

    def nrm(self, x, v, r0=None):
        x, v = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float)); th = x / max(float(r0 if r0 is not None else self.R), 1e-6)
        nx, nz = self.H / self.V, self.R / self.V
        return np.stack([nx * np.cos(th), nx * np.sin(th), np.full_like(th, nz)], -1)


class Revolve:
    """Body of revolution (onion, dome, helmet, bell; a half-dome is an arc th0…th1): the profile prof [(r, z)] from the bottom
    up around a vertical axis through centre (x, y). v is the length along the generatrix from the bottom, x is the arc at
    the radius r0 (the row's bottom). Points follow the profile polyline (at the vertices of the mass faces: the tile does
    not sink into the mass), the normal follows the smoothed tangent."""
    kind = "revolve"

    def __init__(self, centre, prof, name="cupola", th0=0.0, th1=2 * math.pi):
        P = np.asarray(prof, float); self.P = P; self.cx, self.cy = float(centre[0]), float(centre[1])
        self.s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
        self.V = float(self.s[-1]); self.top = self.V; self.name = name
        self.th0, self.th1 = float(th0), float(th1); self.full = self.th1 - self.th0 > 2 * math.pi - 1e-6
        self.R = float(P[0, 0]); self.Rmax = float(P[:, 0].max())
        self.C = np.array([self.cx, self.cy, P[0, 1]]); self.A = np.array([self.cx, self.cy, P[-1, 1]])
        self.e = max(0.02, 0.5 * float(np.min(np.diff(self.s)))) if len(self.s) > 1 else 0.02   # tangent window

    def r(self, v):
        return np.interp(v, self.s, self.P[:, 0])

    def z(self, v):
        return np.interp(v, self.s, self.P[:, 1])

    def world(self, x, v, h, r0=None):
        x, v, h = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float), np.asarray(h, float))
        th = x / max(float(r0 if r0 is not None else self.R), 1e-6)
        a, b = np.clip(v - self.e, 0, self.V), np.clip(v + self.e, 0, self.V)
        dr, dz = self.r(b) - self.r(a), self.z(b) - self.z(a); m = np.hypot(dr, dz) + 1e-12
        nr, nz = dz / m, -dr / m                                                 # outward normal in the meridian plane
        rr = self.r(v) + h * nr; c, s_ = np.cos(th), np.sin(th)
        return np.stack([self.cx + rr * c, self.cy + rr * s_, self.z(v) + h * nz], -1)

    def nrm(self, x, v, r0=None):
        x, v = np.broadcast_arrays(np.asarray(x, float), np.asarray(v, float)); th = x / max(float(r0 if r0 is not None else self.R), 1e-6)
        a, b = np.clip(v - self.e, 0, self.V), np.clip(v + self.e, 0, self.V)
        dr, dz = self.r(b) - self.r(a), self.z(b) - self.z(a); m = np.hypot(dr, dz) + 1e-12
        return np.stack([dz / m * np.cos(th), dz / m * np.sin(th), -dr / m], -1)


def _circle(P):
    """Circle through the points xy (Kasa fit): → (cx, cy, r, largest deviation)."""
    x, y = P[:, 0], P[:, 1]; A = np.stack([x, y, np.ones_like(x)], 1)
    c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]; cx, cy = c[0] / 2, c[1] / 2; r = math.sqrt(max(c[2] + cx * cx + cy * cy, 0.0))
    return cx, cy, r, float(np.abs(np.hypot(x - cx, y - cy) - r).max())


def revolve_of(P, ring_min=16):
    """Vertices of a connected group of faces (k×3) → Revolve or None. A body of revolution: ≥ 2 rings of ≥ ring_min vertices
    on circles with a common vertical axis and ≥ 3 levels (rings and an apex). A polyhedron with fewer than ring_min faces
    is not a body of revolution: a circle around an octagon would hang over the middles of its faces (it is slopes with
    hips)."""
    P = np.unique(np.round(np.asarray(P, float), 3), axis=0); P = P[np.argsort(P[:, 2])]
    if len(P) < 2 * ring_min: return None
    cut = np.flatnonzero(np.diff(P[:, 2]) > 0.005) + 1; levels = np.split(P, cut)
    rings = [L_ for L_ in levels if len(L_) >= ring_min]
    if len(rings) < 2 or len(levels) < 3: return None
    fits = [_circle(L_) for L_ in rings]; rmax = max(f[2] for f in fits)
    if any(f[3] > max(0.02, 0.04 * f[2]) for f in fits): return None
    cxs, cys = np.array([f[0] for f in fits]), np.array([f[1] for f in fits]); cx, cy = float(np.median(cxs)), float(np.median(cys))
    if np.hypot(cxs - cx, cys - cy).max() > max(0.05, 0.03 * rmax): return None
    prof = []
    for L_ in levels:
        d = np.hypot(L_[:, 0] - cx, L_[:, 1] - cy)
        if len(L_) >= ring_min: prof.append((float(d.mean()), float(L_[:, 2].mean())))
        elif len(L_) <= 2 and d.max() < max(0.05, 0.05 * rmax): prof.append((0.0, float(L_[:, 2].mean())))   # apex on the axis
        else: return None                                                         # a level that is neither a ring nor an apex — not a body of revolution
    big = max(rings, key=len); ang = np.sort(np.mod(np.arctan2(big[:, 1] - cy, big[:, 0] - cx), 2 * math.pi))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]])); gi = int(np.argmax(gaps))
    if gaps[gi] > 3 * float(np.median(gaps)):                                    # apse half-dome — an arc without a gap
        th0 = float(ang[(gi + 1) % len(ang)]); th1 = th0 + 2 * math.pi - float(gaps[gi])
    else:
        th0, th1 = 0.0, 2 * math.pi
    return Revolve((cx, cy), prof, th0=th0, th1=th1)


def _hull(P):
    """Convex hull (counter-clockwise), monotone chain."""
    P = sorted(set(map(tuple, np.round(np.asarray(P, float), 6))))
    if len(P) < 3: return np.asarray(P, float)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 1e-12: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 1e-12: up.pop()
        up.append(p)
    return np.asarray(lo[:-1] + up[:-1], float)


# ── layout ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _row_spans(k, x_lo, x_hi, course, prev, rng, period=None):
    """Tiles of a row by the kind's bond: [(x0, x1)] on the segment x_lo…x_hi (a ring — period: a whole number of tiles)."""
    gap = float(k["gap"]); bond = k["bond"]
    if bond == "randomly":                                                       # widths vary, row joints ≥ from_side (like rubble)
        lens = tuple(k["w"]) if isinstance(k["w"], (list, tuple)) else (k["w"] * 0.8, k["w"] * 1.2)
        start = x_lo - (rng.uniform(0, lens[0]) if period is None else 0.0)
        return [(a, b - gap) for a, b in M.fill_row(start, x_hi, lens, float(k["from_side"]), prev, rng) if b - gap > a + 0.01]
    w = width(k)
    if period is not None:                                                       # cone ring: a whole number of tiles
        n = max(4, int(round(period / (w + gap)))); step = period / n
        ph = 0.5 * step if (bond == "staggered" and course % 2) else 0.0
        return [(ph + i * step, ph + (i + 1) * step - gap) for i in range(n)]
    step = w + gap; ph = 0.5 * step if (bond == "staggered" and course % 2) else 0.0
    i0 = int(math.floor((x_lo - ph) / step)) - 1
    out = []
    for i in range(i0, i0 + int((x_hi - x_lo) / step) + 3):
        a = ph + i * step
        if a + w > x_lo and a < x_hi: out.append((a, a + w))
    return out


def _courses(S, k, g, L, t):
    """Rows from the bottom up: (number, v0, step, L, lap, t, width fraction). Number −1 is the underlay row of double and
    triple layering. Diminishing rows (L, lap, t are pairs "at the eaves, at the ridge"): each row takes its own fraction of
    the slope, the step decreases."""
    layers = int(k["n_layers"]); lap = k["lap"]
    if not isinstance(k["L"], (list, tuple)):
        n_courses = int(math.ceil((S.top - 1e-6) / g)) + 1
        for kk in ([-1] if layers >= 2 else []) + list(range(n_courses)):
            yield kk, (0.0 if kk < 0 else kk * g), g, L, float(lap), t, 1.0
        return
    L0 = lerp(k["L"], 0.0)
    if layers >= 2:
        yield -1, 0.0, gauge(k, 0.0), L0, lerp(lap, 0.0), lerp(k["t"], 0.0), 1.0
    v, kk = 0.0, 0
    while v < S.top - 0.02:
        f = min(1.0, v / max(S.top, 1e-6)); Lc, lc = lerp(k["L"], f), lerp(lap, f)
        gc = (Lc - lc) / max(layers, 1)
        yield kk, v, gc, Lc, lc, lerp(k["t"], f), Lc / L0
        v += gc; kk += 1


def lay(S, k, seed=1, sid0=1, r_stop=None):
    """Tiles on surface S (Slope | Cone | Revolve) by kind k. Rows: the step is gauge, the first row starts at the eaves (under
    it, with double and triple layering, an underlay row that is shorter, with a bond shift), the top one is cut by the
    ridge (by the slope outline). Columns (trough tiles): the column step is w, for monk-and-nun and Spanish — two elements
    per column. Tent rings — a whole number of tiles; on a dome — in zones. Sheets and boards — lay_sheets. → tiles."""
    if k["generator"] in ("sheets",):
        return lay_sheets(S, k, seed=seed, sid0=sid0)
    rng = np.random.default_rng(seed); g = gauge(k); L = lerp(k["L"], 0.0); t = lerp(k["t"], 0.0); layers = int(k["n_layers"])
    cone = S.kind in ("cone", "revolve"); tiles = []; sid = sid0; prev = []
    w0 = width(k); zone = [None]
    r_stop = r_stop if r_stop is not None else max(0.18, 0.9 * w0)
    for kk, v0, gk, Lc, lap_k, tk, sc in _courses(S, k, g, L, t):
        under = kk < 0
        if v0 >= S.top - 0.02: break
        Lk = (gk + lap_k) if under else Lc                                       # underlay row is shorter: step + lap
        sw = math.sqrt(sc)                                                       # toward the ridge the width diminishes more gently than the length; the joint shift is the same
        kc = k if sc == 1.0 else dict(k, w=[x * sw for x in k["w"]] if isinstance(k["w"], (list, tuple)) else k["w"] * sw)
        course = (kk + 1) if under else kk
        if cone:
            r0 = float(S.r(v0))
            if r0 < r_stop:
                if S.kind == "cone": break
                continue                                                         # dome: the neck is narrower than a tile — it may widen higher up
            rm = float(S.r(v0 + 0.5 * gk)); period = (S.th1 - S.th0) * rm
            if S.full and S.kind == "revolve" and k["bond"] != "randomly":   # dome: the tile count holds within a zone
                wg = w0 + float(k["gap"])
                if zone[0] is None or not (0.7 * wg <= period / zone[0] <= 1.35 * wg): zone[0] = max(4, int(round(period / wg)))
                step = period / zone[0]; ph = 0.5 * step if (k["bond"] == "staggered" and course % 2) else 0.0
                spans = [(ph + i * step, ph + (i + 1) * step - float(k["gap"])) for i in range(zone[0])]
            elif S.full:
                spans = _row_spans(kc, 0.0, period, course, prev, rng, period=period)
            else:                                                                # fan: a row from edge to edge, the end tiles are trimmed
                spans = [(max(a, 0.0), min(b, period)) for a, b in _row_spans(kc, 0.0, period, course, prev, rng)
                         if min(b, period) - max(a, 0.0) > 0.25 * (b - a)]
            spans = [((a + S.th0 * rm) * r0 / max(rm, 1e-6), (b + S.th0 * rm) * r0 / max(rm, 1e-6)) for a, b in spans]
        else:
            ch = S.chord(v0, v0 + gk)
            if ch is None: continue
            r0 = None
            spans = _row_spans(kc, ch[0] - 0.02, ch[1] + 0.02, course, prev, rng)
        cur = []
        for a, b in spans:
            prof = k["profile"]
            if k["generator"] == "columns":
                items = _column_items(k, a, b, v0, Lk, prof, cone)
            else:
                items = [(outline(a, b, v0, Lk, "straight" if under else k["bottom"]), "flat")]
            for poly, pr in items:
                if cone:
                    q = [(x, min(v, S.V * 0.985)) for x, v in poly]; cut = False
                else:
                    q = clip_convex(poly, S.poly)
                    if len(q) < 3: continue
                    cut = len(q) != len(poly) or not np.allclose(np.sort(np.asarray(q), 0), np.sort(np.asarray(poly, float), 0), atol=1e-6)
                    vis = clip_convex(q, [(-1e6, v0), (1e6, v0), (1e6, v0 + gk), (-1e6, v0 + gk)])
                    if len(vis) < 3 or abs(area(vis)) < min(0.18 * (b - a) * gk, 0.012): continue   # a sliver at a hip — no tile
                tint = float(rng.normal(1.0, float(k["spread"])))
                tiles.append(dict(poly=np.round(np.asarray(q, float), 5).tolist(), v0=v0, L=Lk, t=tk, layers=1 if under else layers,
                                  r0=r0, prof=pr, depth=float(k["depth"]), sid=sid, tint=round(tint, 4),
                                  course=kk, under=under, x0=a, x1=b, cut=bool(cut), g=round(gk, 5)))
                sid += 1
            cur += [a, b]
        if not under: prev = cur
    return tiles


def lay_sheets(S, k, seed=1, sid0=1):
    """Sheets and boards — strips from the eaves to the ridge (masonry has nothing like this: a strip across the whole slope).
    profile "seam": panels of width w, sheets of length L with transverse joints staggered between neighboring panels, a
    standing seam (height "depth") along the panel edge. profile "boards": boards of width w in two layers — the lower ones
    with a half-board gap, the upper ones over the gaps with a quarter-board lap. On a tent and a dome — wedges: the number
    of panels (boards) is by the largest radius, a wedge narrows toward the crown and ends where it is narrower than 12 cm
    (above that is a cap, as with copper on domes). → tiles (like lay)."""
    rng = np.random.default_rng(seed); w = width(k); L = float(k["L"]); t = float(k["t"]); d = float(k["depth"])
    tiles = []; sid = [sid0]; boards = k["profile"] == "boards"
    ring = S.kind in ("cone", "revolve")
    def add(poly, prof, h0, tt, course, x0, x1):
        if ring:
            q = [(x, min(v, S.V * 0.99)) for x, v in poly]; cut = False
        else:
            q = clip_convex(poly, S.poly)
            if len(q) < 3 or abs(area(q)) < 2e-4: return
            cut = len(q) != len(poly)
        v0 = float(min(v for _, v in q)); v1 = float(max(v for _, v in q))
        tiles.append(dict(poly=np.round(np.asarray(q, float), 5).tolist(), v0=v0, L=max(v1 - v0, 1e-3), t=tt, layers=0,
                          r0=(1.0 if ring else None), prof=prof, depth=d, sid=sid[0], tint=round(float(rng.normal(1.0, float(k["spread"]))), 4),
                          course=course, under=False, x0=x0, x1=x1, cut=bool(cut), h0=h0))
        sid[0] += 1
    if ring:                                                                     # wedges: x is the angle (r0 = 1)
        span = S.th1 - S.th0; rref = S.R if S.kind == "cone" else S.Rmax
        n = max(8, int(round(span * rref / (1.5 * w if boards else w)))); dth = span / n
        vs = np.linspace(0.0, S.V, 400); wide = S.r(vs) * dth >= 0.12
        top = float(vs[np.flatnonzero(wide)].max()) if wide.any() else 0.0
        cols = [(S.th0 + i * dth, S.th0 + (i + 1) * dth) for i in range(n)]
        rib = lambda b: 0.006 / max(rref, 1e-3)
    else:
        x_lo, x_hi = float(S.poly[:, 0].min()), float(S.poly[:, 0].max()); top = S.top
        p = 1.5 * w if boards else w
        cols = [(x_lo - 0.25 * w + i * p, x_lo - 0.25 * w + i * p + p) for i in range(int(math.ceil((x_hi - x_lo + 0.5 * w) / p)) + 1)]
        rib = lambda b: 0.006
    for i, (a, b) in enumerate(cols):
        if boards:                                                               # boards: a lower board and an upper one over the gap
            wa = (b - a) / 1.5
            for lay_, (xa, xb) in ((0, (a, a + wa)), (1, (a + 0.75 * wa, a + 1.75 * wa))):
                v = 0.0; j = 0
                while v < top - 0.02:
                    v1 = min(v + L, top); add([(xa, v), (xb, v), (xb, v1), (xa, v1)], "plank", lay_ * t, t, j, xa, xb); v = v1 + 0.01; j += 1
            continue
        off = 0.5 * L if i % 2 else 0.0; v = -off; j = 0                          # seam: sheets staggered, joints and the rib
        while v < top - 0.02:
            v0, v1 = max(v, 0.0), min(v + L, top)
            if v1 - v0 > 0.02: add([(a, v0), (b, v0), (b, v1), (a, v1)], "sheet", 0.0, max(t, 0.001), j, a, b)
            if v0 > 0.0: add([(a, v0 - 0.008), (b, v0 - 0.008), (b, v0 + 0.008), (a, v0 + 0.008)], "joint", 0.001, 0.004, j, a, b)
            v += L; j += 1
        e = rib(b); add([(b - e, 0.0), (b + e, 0.0), (b + e, top), (b - e, top)], "seam", 0.0, d, -2, b - e, b + e)
    return tiles


def _column_items(k, a, b, v0, L, prof, cone):
    """Trough tiles of column a…b: s — one tile with an S profile; monk-and-nun — a trough (nun) and a roll over the joint
    (monk, half a column to the right); troughs (Spanish) — a channel and a capping tile in alternate columns (the same
    trough, turned over)."""
    if prof == "s":
        return [(outline(a, b, v0, L, "straight"), "s")]
    if prof == "monk_nun":
        w = b - a; nun = (a + 0.05 * w, b - 0.05 * w); monk = (b - 0.32 * w, b + 0.32 * w)
        return [(outline(nun[0], nun[1], v0, L, "straight"), "nun"), (outline(monk[0], monk[1], v0 + 0.02, L, "straight"), "monk")]
    if prof == "troughs":                                                         # channel 0.8 of the step, the capping tile over the gap
        w = b - a; return [(outline(a + 0.1 * w, b - 0.1 * w, v0, L, "straight"), "channel"),
                           (outline(b - 0.4 * w, b + 0.4 * w, v0 + 0.03, L, "straight"), "capping")]
    return [(outline(a, b, v0, L, "straight"), "flat")]


# ── edges: ridge, hip, finial ───────────────────────────────────────────────────────────────────────────────────
def caps(a, b, up, k, sid0, seg=0.33, lap=0.04, r=None):
    """Ridge (hip) tile along the edge a → b (world): links of length seg with a lap lap, from the lower end to the upper one
    (the upper link lies on the lower one), the cross section is a half-circle (or a triangle) of radius r; for trough tiles
    r is above the profile (the link lies on mortar above the rolls). → [{a, b, up, r, sid, prof}]."""
    a = np.asarray(a, float); b = np.asarray(b, float); up = np.asarray(up, float); up = up / max(np.linalg.norm(up), 1e-12)
    if a[2] > b[2]: a, b = b, a
    L = float(np.linalg.norm(b - a)); d = (b - a) / max(L, 1e-12); r = r if r is not None else (float(k["ridge_r"]) if k.get("ridge_r") else max(0.09, min(0.16, 0.6 * width(k)), relief(k) + 0.04))
    out, s, i = [], 0.0, 0
    rng = _rng(sid0)
    while s < L - 0.02:
        e = min(L, s + seg)
        out.append(dict(a=(a + d * s).tolist(), b=(a + d * e).tolist(), up=up.tolist(), r=r, sid=sid0 + i,
                        prof="triangular" if k.get("ridge") == "triangular" else "semicircular", tint=round(float(rng.normal(1.0, float(k["spread"]))), 4)))
        if e >= L - 1e-9: break                                                  # the last link has reached the end of the edge
        s = e - lap; i += 1
    return out


VALLEY_GAP = 0.06                                                               # open valley: the tile stops 6 cm short of the line on each side


def open_valleys(rf, gap=VALLEY_GAP):
    """Roof valleys: the outlines of both slopes are set back from the valley line by gap (the trough is visible between the
    trimmed tiles). A repeated call does not cut twice."""
    for vl in rf.get("valleys", []):
        if vl.get("open"): continue
        for si in vl["s"]: rf["surfaces"][si].cut_back(vl["a"], vl["b"], gap)
        vl["open"] = gap


# ── roofs from scene faces ──────────────────────────────────────────────────────────────────────────────────────
def analyse(faces, weld=0.003, min_z=0.15, cone_min=9):
    """Roof faces (world) → roofs. faces — [(vertices k×3, part)]; faces looking down or sideways (n.z < min_z) are not slopes
    (an overhang, the underside). Groups connected by shared vertices are roofs; ≥ cone_min triangles with a common vertex
    are a cone (a tent), otherwise flat slopes (faces of one plane are one slope); an edge of two slopes: horizontal — a
    ridge, inclined — a hip; a vertex common to all slopes, without a ridge (a pyramid), is a finial; a concave junction (a
    slope above the plane of its neighbor) is a valley: no ridge, an open trough on both slopes (open_valleys). A fan (an
    apse half-cone) is a cone along an arc; a kink of one slope (slopes almost in one plane, < 25°) has no ridge; an onion
    and a dome (many faces at several inclinations) get kind "skip" and are not laid yet. A body of revolution (onion, dome,
    helmet, half-dome — revolve_of) is looked for first, over all faces of the group: the bottom of an onion faces down, and
    the n.z filter would tear it apart. → [{kind, surfaces, ridges, hips, valleys, apex, part}]."""
    roofs, faces = _revolves(faces, weld, cone_min)
    F = []
    for pts, part in faces:
        P = np.asarray(pts, float)
        if len(P) < 3: continue
        n = np.sum(np.cross(P, np.roll(P, -1, 0)), axis=0)                    # polygon normal (Newell)
        nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n = n / nn
        if n[2] < min_z: continue
        F.append((P, n, part))
    if not F: return roofs
    keymap = {}; fv = []
    for P, n, part in F:
        ids = []
        for p in P:
            key = tuple(np.round(p / weld).astype(np.int64)); ids.append(keymap.setdefault(key, len(keymap)))
        fv.append(ids)
    parent = list(range(len(F)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = {}
    for fi, ids in enumerate(fv):
        for vi in ids:
            if vi in owner: parent[find(fi)] = find(owner[vi])
            else: owner[vi] = fi
    comps = {}
    for fi in range(len(F)): comps.setdefault(find(fi), []).append(fi)
    for fis in comps.values():
        part = F[fis[0]][2]
        tri = [fi for fi in fis if len(F[fi][0]) == 3]
        if len(tri) >= cone_min and len(tri) == len(fis):                        # tent: all triangles share a vertex
            common = set(fv[tri[0]])
            for fi in tri[1:]: common &= set(fv[fi])
            if len(common) == 1:
                vid = common.pop()
                apex = next(F[fi][0][fv[fi].index(vid)] for fi in tri[:1])
                ring = np.asarray([p for fi in tri for j, p in enumerate(F[fi][0]) if fv[fi][j] != vid])
                cxy = apex[:2]                                                   # the axis passes through the apex (for a fan the ring center is offset)
                R = float(np.linalg.norm(ring[:, :2] - cxy, axis=1).mean()); z0 = float(ring[:, 2].mean())
                ang = np.sort(np.mod(np.arctan2(ring[:, 1] - cxy[1], ring[:, 0] - cxy[0]), 2 * math.pi))
                gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]])); gi = int(np.argmax(gaps))
                step = float(np.median(gaps))
                if gaps[gi] > 3 * step:                                          # fan (apse): an arc without the largest gap
                    th0 = float(ang[(gi + 1) % len(ang)]); th1 = th0 + 2 * math.pi - float(gaps[gi])
                else:
                    th0, th1 = 0.0, 2 * math.pi
                roofs.append(dict(kind="cone", surfaces=[Cone(apex, (cxy[0], cxy[1], z0), R, th0=th0, th1=th1)], ridges=[], hips=[], valleys=[],
                                  apex=apex.tolist() if th1 - th0 > 2 * math.pi - 1e-6 else None, part=part))
                continue
        groups = {}                                                              # planes: normal and offset
        for fi in fis:
            P, n, _ = F[fi]; d = float(n @ P[0])
            key = (tuple(np.round(n / 0.02).astype(int)), int(round(d / 0.03)))
            groups.setdefault(key, []).append(fi)
        levels = {round(float(F[fi][1][2]) / 0.05) for fi in fis}
        if len(groups) > 12 and len(levels) > 2:                                 # onion, dome (body of revolution) — not yet
            Pk = np.vstack([F[fi][0] for fi in fis])
            roofs.append(dict(kind="skip", surfaces=[], ridges=[], hips=[], valleys=[], apex=None, part=part, n_faces=len(fis),
                              box=[np.round(Pk.min(0), 2).tolist(), np.round(Pk.max(0), 2).tolist()]))
            continue
        surfaces, vsets, cen = [], [], []
        for key, gf in groups.items():
            pts = np.vstack([F[fi][0] for fi in gf]); n = np.mean([F[fi][1] for fi in gf], 0)
            S = Slope(pts, n); surfaces.append(S); vsets.append(set(v for fi in gf for v in fv[fi])); cen.append(pts.mean(0))
        ridges, hips, valleys = [], [], []
        rev = {v: k for k, v in keymap.items()}
        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                sh = vsets[i] & vsets[j]
                if len(sh) < 2: continue
                if float(surfaces[i].n @ surfaces[j].n) > math.cos(math.radians(25)):
                    continue                                                     # a kink of one slope (a walkway roof on a curved wall) — no ridge
                pts = np.asarray([np.asarray(rev[v], float) * weld for v in sh])
                if float((cen[i] - surfaces[j].O) @ surfaces[j].n + (cen[j] - surfaces[i].O) @ surfaces[i].n) > 0.01:
                    D = np.linalg.norm(pts[:, None] - pts[None], axis=2); a_, b_ = np.unravel_index(int(np.argmax(D)), D.shape)
                    valleys.append(dict(a=pts[a_].tolist(), b=pts[b_].tolist(), s=[i, j])); continue   # concave junction — a valley
                lo, hi = pts[np.argmin(pts[:, 2])], pts[np.argmax(pts[:, 2])]
                if abs(hi[2] - lo[2]) < 0.05 * max(np.linalg.norm(hi - lo), 1e-6) + 0.01:
                    e = pts[np.argsort(pts @ (pts[-1] - pts[0]))]; ridges.append((e[0].tolist(), e[-1].tolist(), (surfaces[i].n + surfaces[j].n).tolist()))
                else:
                    hips.append((lo.tolist(), hi.tolist(), (surfaces[i].n + surfaces[j].n).tolist()))
        cnt = {}
        for vs in vsets:
            for v in vs: cnt[v] = cnt.get(v, 0) + 1
        top = [v for v, c in cnt.items() if c >= max(3, len(surfaces))] if not ridges else []   # finial — a pyramid: all slopes meet at a point, no ridge
        apex = None
        if top:
            vz = max(top, key=lambda v: rev[v][2]); apex = (np.asarray(rev[vz], float) * weld).tolist()
        roofs.append(dict(kind="slopes", surfaces=surfaces, ridges=ridges, hips=hips, valleys=valleys, apex=apex, part=part))
    return roofs


def _revolves(faces, weld, cone_min):
    """Bodies of revolution among the roof faces: groups connected by shared vertices (all faces, without filtering by the
    normal) → Revolve. A tent (all triangles meeting at one vertex) is not here (a cone). → (roofs, remaining faces)."""
    keymap = {}; fv = []
    for pts, _ in faces:
        fv.append([keymap.setdefault(tuple(np.round(np.asarray(p, float) / weld).astype(np.int64)), len(keymap)) for p in pts])
    parent = list(range(len(faces)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = {}
    for fi, ids in enumerate(fv):
        for vi in ids:
            if vi in owner: parent[find(fi)] = find(owner[vi])
            else: owner[vi] = fi
    comps = {}
    for fi in range(len(faces)): comps.setdefault(find(fi), []).append(fi)
    roofs, used = [], set()
    for fis in comps.values():
        if len(fis) < cone_min: continue
        if all(len(faces[fi][0]) == 3 for fi in fis):
            common = set(fv[fis[0]])
            for fi in fis[1:]: common &= set(fv[fi])
            if len(common) == 1: continue                                        # tent — a cone
        S = revolve_of(np.vstack([np.asarray(faces[fi][0], float) for fi in fis]))
        if S is None: continue
        apex = S.A.tolist() if (S.P[-1, 0] < 0.05 and S.full) else None
        roofs.append(dict(kind="revolve", surfaces=[S], ridges=[], hips=[], valleys=[], apex=apex, part=faces[fis[0]][1]))
        used.update(fis)
    return roofs, [f for i, f in enumerate(faces) if i not in used]


# ── step plan ───────────────────────────────────────────────────────────────────────────────────────────────────
DESC_FILES = ("constructions.yaml", "masonry.yaml")


def _detail(v):
    if v not in ("cam", "all"): raise SystemExit(f"detail level '{v}': cam | all")
    return v


def resolve(folder, kw=None):
    """Roofing plan of a building (like masonry.resolve): <folder>/constructions.yaml — the "roofing:" section: covering — the
    kind of all roofs (the former key roofing); a part of the building has its own kind (towers, buildings, gates) and so
    does a shape (tents — cones, domes — onions, cupolas, helmets); a shape overrides a part. detail — cam | all, as for
    masonry; px-per-row thresholds (thresholds), far view (far), seed. Command-line keys override the description. Kinds —
    only the needed ones, in full (kinds=all — all of them: kinds sheet, checks)."""
    import yaml
    kw = dict(kw or {}); path = next((os.path.join(folder, f) for f in DESC_FILES if os.path.exists(os.path.join(folder, f))), "")
    d = (yaml.safe_load(open(path, encoding="utf-8")) or {}) if path else {}
    c = dict(d.get("roofing") or {})
    parts = {"all": kw.get("covering") or kw.get("roofing") or c.get("covering") or c.get("roofing") or "beaver"}
    for p_ in PART_KEYS:
        v = kw.get(p_) or c.get(p_)
        if v: parts[p_] = v
    bad = sorted({v for v in parts.values() if v not in KINDS})
    if bad: raise SystemExit(f"roofing: no kind {bad}; available: {', '.join(KINDS)} (components/constructions/roofing/coverings.yaml)")
    lo, hi = c.get("thresholds") or [5, 12]
    plan = dict(roofing=dict(parts=parts, detail=_detail(kw.get("detail") or c.get("detail") or "cam"),
                            thresholds=[float(kw.get("Tlo", lo)), float(kw.get("Thi", hi))], seed=int(kw.get("seed", c.get("seed", 1))),
                            far=float(kw.get("Tfar", c.get("far", 2.5)))),       # px per row: below it only a shell with a pattern
                kinds={n: KINDS[n] for n in sorted(set(parts.values()))} if kw.get("kinds") != "all" else dict(KINDS),
                substances=LIBRARY.get("substances", {}), version=LIBRARY.get("version"), description=path or None)
    return plan


PART_KEYS = ("towers", "buildings", "gates", "tents", "domes")       # building parts and roof shapes (a shape overrides a part)
FORM_KEY = {"cone": "tents", "revolve": "domes"}


def kind_for(parts, kinds, rf):
    """Kind of roof rf: by shape (tents, domes), otherwise by building part, otherwise the covering of all roofs."""
    name = parts.get(FORM_KEY.get(rf["kind"], "")) or parts.get(rf["part"]) or parts["all"]
    return kinds.get(name) or kinds[parts["all"]]


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "resolve":
    KW = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    plan = resolve(sys.argv[2], KW)
    if KW.get("out"): json.dump(plan, open(KW["out"], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False))
