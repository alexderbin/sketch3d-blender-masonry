"""Settlement landmarks: the acropolis, a theatre in the slope, the ascent to the sanctuary, temples on the platform,
stoas around the agora, civic buildings in the blocks — where they stand (by terrain and plan) and the programs of the
"classic" type (via the buildings.py entry).
Knowledge — settlements.yaml (keys `acropolis`, `landmarks`). Project Python: numpy, OpenCV, shapely, networkx."""
import os
import sys
from components.buildings import arch_parts
import math

import cv2
import networkx as nx
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from components.buildings import buildings as B
from components.buildings.ensembles import plots as PL

EAST = np.array([1.0, 0.0])


def _unit(v): v = np.asarray(v, float); return v / max(float(np.linalg.norm(v)), 1e-9)


def _contour(M, xs, ys, step, at=None):
    """Grid mask → polygon (the one containing point at, otherwise the largest)."""
    cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); polys = []
    for cn in cs:
        if len(cn) < 4: continue
        P = Polygon([(xs[0] + x * step, ys[0] + y * step) for x, y in cn[:, 0, :]]).buffer(0)
        if not P.is_empty: polys += PL.parts(P, 1.0)
    if not polys: return None
    if at is not None:
        for P in polys:
            if P.buffer(step).contains(Point(at)): return P
    return max(polys, key=lambda g: g.area)


# ---------------------------------------------------------------- acropolis
def acropolis(G, c, R, spec, step=2.0):
    """A hill within the settlement, higher than the place's median by the `projection` in m, and compact → an acropolis: the
    foot (the contour at 12 % of the projection — the street grid goes around it, a ring road along the contour), the
    platform (the top, 22 % of the projection), the platform elevation. Otherwise None."""
    xs = np.arange(c[0] - R, c[0] + R + 1e-6, step); ys = np.arange(c[1] - R, c[1] + R + 1e-6, step); X, Y = np.meshgrid(xs, ys)
    Z = G.height(X, Y); inside = (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= (0.92 * R) ** 2
    zmed = float(np.median(Z[inside])); zmax = float(Z[inside].max()); prom = zmax - zmed
    if prom < float(spec.get("projection", 8.0)): return None
    iy, ix = np.unravel_index(int(np.argmax(np.where(inside, Z, -1e9))), Z.shape); p = (float(xs[ix]), float(ys[iy]))
    foot = _contour((Z > zmed + 0.12 * prom) & inside, xs, ys, step, at=p)
    top = _contour((Z > zmax - 0.22 * prom) & inside, xs, ys, step, at=p)
    if foot is None or top is None or foot.area > 0.3 * math.pi * R * R or top.area < 600: return None
    foot = foot.simplify(1.0).buffer(3.0, join_style="round"); top = top.simplify(0.8)
    zt = Z[shapely.contains_xy(top.buffer(-1.0), X, Y)]
    z_top = float(np.median(zt)) if zt.size else zmax
    return {"foot": foot, "top": top, "z_top": z_top, "p": [float(top.centroid.x), float(top.centroid.y)], "prom": prom, "z_med": zmed}


# ---------------------------------------------------------------- theatre
def site_theatre(G, CL, acro, c, Bd, S, rng, size=1.0):
    """Theatre in a hillside: a search over directions (every 15°), the offset of the orchestra center from the foot, and the
    cavea radius. Cost — fill under the rows (the cavea must lie into the slope), a deep cut, entering the acropolis
    platform (the temples are there), going outside the wall; better — a larger cavea and a view of the town.
    → site {c, u, R, z, shape, zone, foot_pts} or None."""
    if acro is None: return None
    Hc = np.asarray(acro["p"], float); best = None; top_in = acro["top"].buffer(-4.0)
    toward = _unit(np.asarray(c) - Hc) if np.linalg.norm(np.asarray(c) - Hc) > 10 else None
    fb = acro["foot"].exterior
    for k in range(24):
        th = 2 * math.pi * k / 24; uo = np.array([math.cos(th), math.sin(th)])
        ray = LineString([Hc, Hc + uo * 400]); hit = ray.intersection(fb)
        if hit.is_empty: continue
        rf = min(float(np.linalg.norm(np.asarray(q.coords[0]) - Hc)) for q in getattr(hit, "geoms", [hit]))
        for Rc in np.array([20.0, 25.0, 30.0, 35.0]) * size:
            T = CL.theatre_shape(S, Rc, np.random.default_rng(7))
            for dd in (-6.0, 0.0, 6.0, 12.0):
                O = Hc + uo * (rf + dd); zo = float(G.height(*O))
                ang = th + math.pi + np.linspace(-1, 1, 9) * math.radians(T["span"]) / 2
                rr = np.linspace(T["ro"] + 1.5, Rc, 7); A_, R_ = np.meshgrid(ang, rr)
                PX = O[0] + R_ * np.cos(A_); PY = O[1] + R_ * np.sin(A_)
                zc = zo + CL.cavea_z(T, R_); zt = G.height(PX, PY)
                fill = float(np.mean(np.clip(zc - zt, 0, None))); deep = float(np.mean(np.clip(zt - zc - 3.0, 0, None)))
                on_top = float(np.mean(shapely.contains_xy(top_in, PX, PY)))
                out_wall = 0.0 if Bd is None else float(np.mean(~shapely.contains_xy(Bd.buffer(-4.0), PX, PY)))
                sk = O + uo * (T["ro"] + 6.0); flat = abs(float(G.height(*sk)) - zo)
                score = fill * 1.5 + deep * 1.2 + on_top * 25.0 + out_wall * 30.0 + flat * 0.8 - 0.08 * Rc
                if toward is not None: score -= 1.2 * float(uo @ toward)
                if best is None or score < best[0]: best = (score, O, -uo, Rc, zo, T)
    if best is None: return None
    sc, O, u, Rc, zo, T = best
    fr = {"c": [float(O[0]), float(O[1])], "u": [float(u[0]), float(u[1])], "R": float(Rc), "z": zo, "shape": T}
    P = CL.program("theatre", fr, {"height": None, "z": zo}, np.random.default_rng(3))
    fp = Polygon(CL.footprint(P)).buffer(0)
    return dict(fr, zone=fp.buffer(2.0), score=round(float(sc), 2))


def cavea_cut(CL, site):
    """Slope cut for the cavea: (geometry, z(x, y)) — the ground is no higher than the rows; the orchestra and skene are a flat
    platform."""
    T = site["shape"]; O = np.asarray(site["c"], float); u = _unit(site["u"]); th = math.atan2(u[1], u[0]); sp = math.radians(T["span"])
    Rl = T["ro"] + 1.0 + T["rows"] * T["dep"] + (1.4 if T["dia"] else 0.0) + 1.2      # up to the middle of the wall behind the top row
    arc = [O + Rl * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(th - sp / 2, th + sp / 2, 40)]
    sector = Polygon([O] + arc).buffer(0)
    def zf(X, Y):
        r = np.hypot(X - O[0], Y - O[1]); return site["z"] + CL.cavea_z(T, r) - 0.25
    v = np.array([-u[1], u[0]]); ws = T["ro"] + 6.0; back = T["ro"] + 13.0
    flat = Polygon([O + v * ws + u * 2.0, O - v * ws + u * 2.0, O - v * ws - u * back, O + v * ws - u * back]).union(Point(O).buffer(T["ro"] + 1.5))
    return sector, zf, flat


# ---------------------------------------------------------------- ascent
def path_up(G, acro, streets_g, avoid, step=2.0, margin=40.0):
    """Ascent to the acropolis: the shortest path over a 2 m lattice from the platform edge to the nearest street; slope cost —
    steeper than 30 % is expensive (a looping stair rather than a straight one), going around the theatre and other
    buildings. → (line from the top to the street, point at the edge, outward direction) or None."""
    x0, y0, x1, y1 = acro["foot"].buffer(margin).bounds
    xs = np.arange(x0, x1 + 1e-6, step); ys = np.arange(y0, y1 + 1e-6, step); X, Y = np.meshgrid(xs, ys); Z = G.height(X, Y)
    bad = shapely.contains_xy(avoid, X, Y) if avoid is not None and not avoid.is_empty else np.zeros(X.shape, bool)
    rim = acro["top"].buffer(-3.0)
    src = shapely.contains_xy(rim, X, Y) & ~shapely.contains_xy(rim.buffer(-step * 1.5), X, Y)
    dst = shapely.contains_xy(streets_g.buffer(1.5), X, Y) & ~shapely.contains_xy(acro["foot"].buffer(-1.0), X, Y)
    if not src.any() or not dst.any(): return None
    Gx = nx.Graph(); ny_, nx_ = X.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for iy in range(ny_ - dy):
            for ix in range(max(0, -dx), nx_ - max(0, dx)):
                jy, jx = iy + dy, ix + dx
                if bad[iy, ix] or bad[jy, jx]: continue
                L = step * math.hypot(dx, dy); s = abs(Z[jy, jx] - Z[iy, ix]) / L
                Gx.add_edge((iy, ix), (jy, jx), weight=L * (1.0 + 40.0 * max(0.0, s - 0.3) ** 2 + 2.0 * max(0.0, s - 0.6)))
    for iy, ix in zip(*np.nonzero(src)):
        if (iy, ix) in Gx: Gx.add_edge("S", (iy, ix), weight=0.0)
    for iy, ix in zip(*np.nonzero(dst)):
        if (iy, ix) in Gx: Gx.add_edge((iy, ix), "T", weight=0.0)
    if "S" not in Gx or "T" not in Gx: return None
    try: path = nx.shortest_path(Gx, "S", "T", weight="weight")
    except nx.NetworkXNoPath: return None
    pts = [(float(xs[ix]), float(ys[iy])) for iy, ix in path[1:-1]]
    if len(pts) < 3: return None
    g = LineString(pts).simplify(1.2)
    p0 = np.asarray(g.coords[0]); out = _unit(np.asarray(g.interpolate(min(8.0, g.length)).coords[0]) - p0)
    return g, p0, out


# ---------------------------------------------------------------- rectangle in a polygon
def fit_rect(poly, u, aspects, step=2.0, max_side=None):
    """Largest rectangle along axis u inside a polygon: centers on a grid with step `step`, the length (along u) to width ratio
    taken from aspects. → (center, L, W) or None."""
    if poly is None or poly.is_empty: return None
    u = _unit(u); ang = math.atan2(u[1], u[0])
    Pr = affinity.rotate(poly, -ang, origin=(0, 0), use_radians=True); pp = shapely.prepared.prep(Pr)
    x0, y0, x1, y1 = Pr.bounds; best = None
    gx, gy = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(y0 + step / 2, y1, step))
    ok = shapely.contains_xy(Pr, gx, gy)
    for cx, cy in zip(gx[ok], gy[ok]):
        for a in aspects:
            lo, hi = 0.0, min(x1 - x0, y1 - y0)
            if max_side: hi = min(hi, max_side / 2)
            for _ in range(10):
                m = (lo + hi) / 2
                if pp.contains(box(cx - a * m, cy - m, cx + a * m, cy + m)): lo = m
                else: hi = m
            area = 4 * a * lo * lo
            if lo > 0.5 and (best is None or area > best[0]): best = (area, cx, cy, 2 * a * lo, 2 * lo)
    if best is None: return None
    _, cx, cy, L, W = best; c = np.array([cx, cy]); c = np.array([c[0] * math.cos(ang) - c[1] * math.sin(ang), c[0] * math.sin(ang) + c[1] * math.cos(ang)])
    return c, L, W


def block_frame(poly, ang, face=None):
    """Block frame along the grid (angle ang): center, axis u (toward face — the point the facade looks at), length along u,
    width."""
    u = np.array([math.cos(ang), math.sin(ang)]); v = np.array([-u[1], u[0]])
    C = np.asarray(poly.exterior.coords); c = np.asarray(poly.centroid.coords[0])
    a = (C - c) @ u; b = (C - c) @ v; cc = c + u * (a.max() + a.min()) / 2 + v * (b.max() + b.min()) / 2
    Lu, Lv = float(np.ptp(a)), float(np.ptp(b))
    if face is not None:
        d = np.asarray(face, float) - cc
        if abs(d @ v) > abs(d @ u): u, v, Lu, Lv = (v if d @ v > 0 else -v), u, Lv, Lu
        elif d @ u < 0: u = -u
    return cc, u, Lu, Lv


# ---------------------------------------------------------------- landmark programs
def acropolis_set(CL, acro, gate, G2, rng, avoid=None):
    """Acropolis platform: a propylon at the edge where the ascent ends; the main temple (a peripteros on the east–west axis,
    the largest that fits); an altar to the east; a second temple in the remaining space (prostyle, amphiprostyle or
    tholos). → programs."""
    out = []; ctx = {"height": G2.height, "z": acro["z_top"]}
    room = acro["top"].buffer(-3.0)
    if avoid is not None and not avoid.is_empty: room = room.difference(avoid)
    if gate is not None:                                                            # propylon: back to the platform, facing the ascent
        p0, uo = gate; Lp, Wp = 12.0, 16.0
        cp = np.asarray(p0) - uo * (Lp / 2 + 1.0)                                   # facing the start of the ascent, entirely on the platform
        P = CL.program("propylon", {"c": cp.tolist(), "u": uo.tolist(), "L": Lp, "W": Wp}, ctx, rng)
        if P:
            out.append(P); room = room.difference(Polygon(CL.footprint(P)).buffer(4.0))
    for kind, aspects, share in (("peripteros", (2.1, 2.3), 1.0), (None, None, 0.85)):
        if kind is None:
            kind = str(rng.choice(["prostyle", "amphiprostyle", "tholos", "in_antis"], p=[0.4, 0.25, 0.2, 0.15]))
            aspects = (1.0,) if kind == "tholos" else (1.7, 1.9)
            if rng.random() > share: continue
        best = None
        for da in (0, -6, 6, -12, 12):                                            # axis — east ± up to 12°
            a = math.radians(da); u = np.array([math.cos(a), math.sin(a)])
            parts_ = PL.parts(room, 30.0)
            f = fit_rect(max(parts_, key=lambda g: g.area) if parts_ else None, u, aspects, step=2.0, max_side=60 if kind == "peripteros" else 26)
            if f and (best is None or f[1] * f[2] > best[1] * best[2] + 1.0): best = (u, f[1], f[2], f[0])
        if best is None: continue
        u, L, W, cc = best
        P = CL.program(kind, {"c": cc.tolist(), "u": u.tolist(), "L": L, "W": W}, ctx, rng)
        if P is None and kind == "peripteros": P = CL.program("prostyle", {"c": cc.tolist(), "u": u.tolist(), "L": L, "W": W}, ctx, rng)
        if P is None: continue
        out.append(P); fp = Polygon(CL.footprint(P))
        room = room.difference(fp.buffer(4.0))
        if P["kind"] in ("peripteros", "prostyle", "podium"):                        # altar — to the east of the facade
            front = np.asarray(P["c"]) + np.asarray(P["u"]) * (P["L"] / 2 + 9.0)
            fa = {"c": front.tolist(), "u": P["u"], "L": 5.0, "W": min(10.0, P["W"] * 0.6)}
            if room.contains(Polygon(Point(front).buffer(5.5))):
                A = CL.program("altar", fa, ctx, rng)
                if A: out.append(A); room = room.difference(Polygon(CL.footprint(A)).buffer(3.0))
    return out


def agora_stoas(CL, sq, ang, open_to, G2, rng):
    """Stoas around the agora in a U shape: the main one (deeper, with two aisles) opposite the open side, the side ones — from
    its facade to the edge; the open side faces the ascent to the acropolis or the main street. The altar is in the square.
    → programs."""
    u = np.array([math.cos(ang), math.sin(ang)])                                  # frame — the largest grid-aligned rectangle inside the square
    f = fit_rect(sq.buffer(-0.5), u, (0.6, 0.8, 1.0, 1.25, 1.6), step=3.0)
    if f is None: return []
    c, Lu, Lv = f
    v = np.array([-u[1], u[0]]); sides = [(u, Lv, Lu), (-u, Lv, Lu), (v, Lu, Lv), (-v, Lu, Lv)]   # (outward, side length, square depth)
    if open_to is not None: k_open = int(np.argmax([float(_unit(np.asarray(open_to) - c) @ s[0]) for s in sides]))
    else: k_open = int(rng.integers(4))
    n_open = sides[k_open][0]; k_main = next(i for i, s in enumerate(sides) if float(s[0] @ n_open) < -0.9)
    ctx = {"height": G2.height}; out = []
    nm, Lm, Dm = sides[k_main]; dm = float(np.clip(0.24 * Dm, 9.0, 13.0))
    pm = c + nm * (Dm / 2 - dm / 2)
    P = CL.program("stoa", {"c": pm.tolist(), "u": (-nm).tolist(), "L": dm, "W": Lm - 0.6}, ctx, rng)
    if P: out.append(P)
    for i, (ns, Ls, Ds) in enumerate(sides):
        if i in (k_open, k_main): continue
        ds_ = float(np.clip(0.18 * Ds, 7.0, 9.5)); Lw = Ls - dm - 1.0
        pc = c + ns * (Ds / 2 - ds_ / 2) - nm * (dm + 1.0) / 2
        P = CL.program("stoa", {"c": pc.tolist(), "u": (-ns).tolist(), "L": ds_, "W": Lw}, ctx, rng)
        if P: out.append(P)
    q = c - nm * (Dm * 0.1)
    A = CL.program("altar", {"c": q.tolist(), "u": (-nm).tolist(), "L": 4.0, "W": 6.0}, ctx, rng)
    if A: out.append(A)
    return out


def civic(CL, kind, blk, ang, face, G2, rng):
    """Civic block: a bouleuterion (and a small temple next to it) or a palaestra — within the block frame, facade toward the
    square."""
    cc, u, Lu, Lv = block_frame(blk, ang, face); ctx = {"height": G2.height}; out = []
    if kind == "palaestra":
        P = CL.program("palaestra", {"c": cc.tolist(), "u": u.tolist(), "L": Lu - 1.0, "W": Lv - 1.0}, ctx, rng)
        return [P] if P else []
    v = np.array([-u[1], u[0]])
    if Lv > 34:                                                                     # a long block — the hall and the small temple side by side
        wb = min(26.0, Lv * 0.55); wt = Lv - wb - 3.0
        P = CL.program("bouleuterion", {"c": (cc + v * (Lv / 2 - wb / 2)).tolist(), "u": u.tolist(), "L": Lu - 1.0, "W": wb}, ctx, rng)
        if P: out.append(P)
        Lt = min(Lu - 2.0, 2.2 * wt)                                               # the temple kind depends on the space: peripteros, prostyle, in antis
        kind = "peripteros" if Lt >= 26 and wt >= 14.5 else ("prostyle" if Lt >= 15 and wt >= 8.5 else "in_antis")
        fr = {"c": (cc - v * (Lv / 2 - wt / 2)).tolist(), "u": u.tolist(), "L": Lt, "W": wt}
        T = CL.program(kind, fr, ctx, rng) or CL.program("in_antis", fr, ctx, rng)
        if T: out.append(T)
    else:
        P = CL.program("bouleuterion", {"c": cc.tolist(), "u": u.tolist(), "L": Lu - 1.0, "W": Lv - 1.0}, ctx, rng)
        if P: out.append(P)
    return out


def temenos(CL, blk, G2, rng):
    """Sanctuary in a block (a polis without a hill): a peripteros on the east–west axis and an altar in front of it."""
    f = fit_rect(blk.buffer(-2.0), EAST, (2.0, 2.2), step=2.0, max_side=60)
    if f is None: return []
    cc, L, W = f; ctx = {"height": G2.height}; out = []
    L2 = L - 12.0; c2 = cc - EAST * 6.0
    P = CL.program("peripteros", {"c": c2.tolist(), "u": EAST.tolist(), "L": L2, "W": W}, ctx, rng) or CL.program("prostyle", {"c": c2.tolist(), "u": EAST.tolist(), "L": L2, "W": W}, ctx, rng)
    if P:
        out.append(P); fa = np.asarray(P["c"]) + EAST * (P["L"] / 2 + 6.0)
        A = CL.program("altar", {"c": fa.tolist(), "u": EAST.tolist(), "L": 4.0, "W": min(8.0, W * 0.6)}, ctx, rng)
        if A: out.append(A)
    return out
