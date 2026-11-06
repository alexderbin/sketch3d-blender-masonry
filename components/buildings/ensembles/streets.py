"""Street network of an ensemble. The generators produce lines with a rank and a width:
  grid    — Hippodamus, castrum, a hive tier: lines at the block step, main axes through the center, a ring along the wall,
            distortion by a field;
  organic — Voronoi blocks on a spiral, node welding, a wall along the outline, gates, arterial streets by the
            shortest path from the gates to the square with a slope cost, big blocks are cut by lanes (chaotic splitting);
  growth  — Parish–Müller, Emilien: segments from the gates or edges, direction with a slope cost, branches, snapping to
            nodes and edges.
Common: the settlement boundary, stitching the lines into a graph (nodes, edges with points), the street ribbon. Project
Python: numpy, scipy, shapely, networkx."""
import os
import sys
from components.buildings import arch_parts
import heapq
import math

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import split, unary_union

RANKS = ("artery", "street", "lane", "passage")
WALK = RANKS + ("road", "bridge")


def width(W, rank):
    """Width of a rank; if the archetype lacks it — the nearest lower (or higher) rank."""
    if rank in W: return float(W[rank])
    i = RANKS.index(rank) if rank in RANKS else 1
    for r in RANKS[i:] + RANKS[:i][::-1]:
        if r in W: return float(W[r])
    return 4.0


def lower(W, rank):
    """The rank one step lower, if the archetype has it."""
    i = RANKS.index(rank)
    for r in RANKS[i + 1:]:
        if r in W: return r
    return rank


class Field:
    """Smooth displacement field (a sum of sines, three waves) — a distortion of the grid: a point shifts the same way on any line."""

    def __init__(self, amp, rng, wave=(70.0, 160.0)):
        self.amp = amp; self.k = []; self.ph = []
        for _ in range(3):
            a = rng.uniform(0, 2 * math.pi); lam = rng.uniform(*wave)
            self.k.append(np.array([math.cos(a), math.sin(a)]) * 2 * math.pi / lam); self.ph.append(rng.uniform(0, 2 * math.pi, 2))

    def __call__(self, P):
        P = np.asarray(P, float); d = np.zeros_like(P)
        for k, ph in zip(self.k, self.ph): t = P @ k; d[:, 0] += np.sin(t + ph[0]); d[:, 1] += np.sin(t + ph[1])
        return P + d * self.amp / 3.0


# ---------------------------------------------------------------- boundary
def boundary(form, c, R, rough, rng, ang=0.0):
    """Settlement outline: a circle (with irregularity), a castrum (a square with rounded corners), an octagon, a polyline
    (Greek wall)."""
    c = np.asarray(c, float)
    if form == "castrum":
        h = R / math.sqrt(2) * 1.0; P = box(c[0] - h, c[1] - h, c[0] + h, c[1] + h).buffer(-0.12 * h).buffer(0.12 * h, quad_segs=6)
        return shapely.affinity.rotate(P, ang, origin=tuple(c), use_radians=True)
    if form == "octagon":
        return Polygon([(c[0] + R * math.cos(ang + math.pi / 8 + i * math.pi / 4), c[1] + R * math.sin(ang + math.pi / 8 + i * math.pi / 4)) for i in range(8)])
    if form == "polyline":                                                         # Greek wall: straight segments of 50–90 m, towers at the corners
        n = int(rng.integers(9, 14)); a0 = rng.uniform(0, 2 * math.pi)
        A = np.sort(a0 + 2 * math.pi * (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n)
        return Polygon([(c[0] + R * (1 - rough * rng.random()) * math.cos(a), c[1] + R * (1 - rough * rng.random()) * math.sin(a)) for a in A]).buffer(0)
    n = 72; ks = [(k, rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0)) for k in (2, 3, 5)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n; r = R * (1 + rough * sum(w * math.sin(k * a + ph) for k, ph, w in ks) / 2.2)
        pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    return Polygon(pts)


def line(geom, rank, w, kind="street"):
    return {"geom": geom, "rank": rank, "w": float(w), "kind": kind}


# ---------------------------------------------------------------- grid
def grid(c, B, ang, blk, W, axes, ring=0.0, jitter=0.0, rng=None):
    """Grid inside outline B: lines along u (the step across — the block depth) and along v (the step — the length); main axes
    through the center; a ring — a street along the boundary; distortion — a displacement field (the grid of Pompeii).
    → lines, axes (the lines of the main streets)."""
    c = np.asarray(c, float); u = np.array([math.cos(ang), math.sin(ang)]); v = np.array([-u[1], u[0]])
    a, b = blk; w0 = width(W, "street"); wm = width(W, "artery")
    pu, pv = a + w0, b + w0
    x0, y0, x1, y1 = B.bounds; R = math.hypot(x1 - x0, y1 - y0)
    inner = B.buffer(-(ring / 2 + 0.6)) if ring > 0 else B
    if inner.geom_type != "Polygon": inner = max(inner.geoms, key=lambda g: g.area)
    F = Field(jitter, rng) if jitter > 0 and rng is not None else None
    lines, axes_l = [], []

    def main_ks(n, step):
        if n <= 0: return set()
        if n == 1: return {0}
        m = max(1, int(round(0.45 * R / 2 / step)))
        return {-m, m} if n == 2 else {-m, 0, m}
    for along, across, step, n_ax in ((u, v, pv, axes.get("u", 1)), (v, u, pu, axes.get("v", 0))):
        K = int(R / step) + 2; mk = main_ks(n_ax, step)
        for k in range(-K, K + 1):
            o = c + across * (k * step)
            g = LineString([o - along * R, o + along * R]).segmentize(8.0)
            if F is not None: g = LineString(F(np.asarray(g.coords)))
            main = k in mk
            g = g.intersection(B if main else inner)                              # main ones — up to the wall (gate), the others — up to the ring
            for part in getattr(g, "geoms", [g]):
                if part.length < 3.0 or part.geom_type != "LineString": continue
                lines.append(line(part, "artery" if main else "street", wm if main else w0))
                if main: axes_l.append(part)
    if ring > 0: lines.append(line(inner.exterior, "street", ring))
    lines.append(line(B.exterior, "edge", 0.0, kind="edge"))
    return lines, axes_l


# ---------------------------------------------------------------- organic
def _weld(cells, d):
    """Welding of vertices closer than d (8 units) → a shared vertex array and cells as lists of indices."""
    pts = []; own = []
    for i, P in enumerate(cells):
        for q in P: pts.append(q); own.append(i)
    pts = np.asarray(pts, float); par = list(range(len(pts)))

    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in cKDTree(pts).query_pairs(d): par[f(i)] = f(j)
    roots = sorted({f(i) for i in range(len(pts))}); rid = {r: k for k, r in enumerate(roots)}
    V = np.zeros((len(roots), 2)); cnt = np.zeros(len(roots))
    for i in range(len(pts)): k = rid[f(i)]; V[k] += pts[i]; cnt[k] += 1
    V /= cnt[:, None]
    out = []; k = 0
    for P in cells:
        ids = [rid[f(k + j)] for j in range(len(P))]; k += len(P)
        ids = [x for j, x in enumerate(ids) if x != ids[j - 1]]
        out.append(ids if len(set(ids)) >= 3 else [])
    return V, out


def bisect(P, chaos, rng, gap=0.0):
    """An alley cut: a cut across the long side (fraction 0.5 ± spread, tilt up to 30° × chaos).
    → (halves, cut line)."""
    C = np.asarray(P.exterior.coords)[:-1]; L = np.linalg.norm(np.roll(C, -1, 0) - C, axis=1); i = int(np.argmax(L))
    p0, p1 = C[i], C[(i + 1) % len(C)]; spread = 0.8 * chaos; ratio = (1 - spread) / 2 + rng.random() * spread
    ang = (rng.random() - 0.5) * math.pi / 6 * chaos
    p = p0 + (p1 - p0) * ratio; d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    n = np.array([-d[1], d[0]]); ca, sa = math.cos(ang), math.sin(ang); n = np.array([n[0] * ca - n[1] * sa, n[0] * sa + n[1] * ca])
    big = 4 * math.sqrt(P.area) + 10
    cut = LineString([p - n * big, p + n * big]).intersection(P)
    parts = [g for g in getattr(cut, "geoms", [cut]) if g.geom_type == "LineString" and g.length > 0.5]
    if not parts: return [P], None
    seg = min(parts, key=lambda g: g.distance(Point(p)))
    a_, b_ = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1]); t_ = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
    seg = LineString([a_ - t_ * 0.05, b_ + t_ * 0.05])                          # cut ends go past the edge: otherwise split does not cut (an end exactly on
    halves = [g for g in split(P, seg).geoms if g.area > 1.0]                   # the boundary — 3 of 4 divisions failed), and the lane does not end at a street node
    if gap > 0: halves = [h.buffer(-gap / 2, join_style="mitre") for h in halves]; halves = [h for h in halves if not h.is_empty]
    return halves, seg


def organic(c, R, spec, W, rng, height=None, wall=True, n_gates=3, plaza=True, cost_slope=4.0):
    """An organic town. → a dict: lines, outline, wall, gates, square,
    cells (the inner ones are town blocks)."""
    c = np.asarray(c, float); patch = float(spec.get("city_block", 60)); chaos = float(spec.get("chaos", 0.6))
    n_in = max(5, int(round(math.pi * R * R / patch ** 2 * 1.05)))
    # spiral (angle √i·5); radius ∝ i^0.6 instead of ∝ i: with ∝ i the streets in the center are
    # fractions of a meter, ours are 5–7 m, and small central cells would be eaten by the streets;
    # the core stays slightly denser than the edge
    k = R / n_in ** 0.6; sa = rng.random() * 2 * math.pi; n_pts = n_in * 3
    rad = [0.0] + [k * i ** 0.6 * (1 + 0.18 * (rng.random() - 0.5)) for i in range(1, n_pts)]
    pts = np.array([[math.cos(sa + math.sqrt(i) * 5) * rad[i], math.sin(sa + math.sqrt(i) * 5) * rad[i]] for i in range(n_pts)]) + c
    env = box(*(c - 4 * R), *(c + 4 * R))

    def cells_of(P):
        vp = shapely.voronoi_polygons(MultiPoint(P), extend_to=env, ordered=True)
        return [g.intersection(env) for g in vp.geoms]
    cells = cells_of(pts)
    for _ in range(3):                                                           # Lloyd relaxation for the central ones (0, 1, 2)
        for j in (0, 1, 2): pts[j] = np.asarray(cells[j].centroid.coords[0])
        cells = cells_of(pts)
    keep = [i for i in range(n_pts) if cells[i].distance(Point(c)) < 2.2 * R]
    V, ids = _weld([list(cells[i].exterior.coords)[:-1] for i in keep], min(7.0, 0.12 * patch))
    inner = set(range(min(n_in, len(keep))))
    # wall — the outline of the inner cells, smoothed (strength min(1, 40/n))
    city = unary_union([Polygon(V[ids[i]]) for i in inner if ids[i]]).buffer(0)
    if city.geom_type != "Polygon": city = max(city.geoms, key=lambda g: g.area)
    city = Polygon(city.exterior)
    vid = {(round(V[i][0], 4), round(V[i][1], 4)): i for i in range(len(V))}
    ring = [vid.get((round(x, 4), round(y, 4))) for x, y in list(city.exterior.coords)[:-1]]
    ring = [r for r in ring if r is not None]
    if wall and len(ring) > 4:
        f = min(1.0, 40.0 / n_in); Vn = V.copy()
        for j, r in enumerate(ring):
            a, b = ring[j - 1], ring[(j + 1) % len(ring)]; Vn[r] = V[r] + f * ((V[a] + V[b]) / 2 - V[r]) * 0.5
        V = Vn
        city = Polygon(V[ring])
    # gates — wall vertices where two inner cells meet
    share = {}
    for i in inner:
        for r in ids[i]: share[r] = share.get(r, 0) + 1
    cand = [r for r in ring if share.get(r, 0) >= 2]
    gates = []
    if cand:
        ang = {r: math.atan2(V[r][1] - c[1], V[r][0] - c[0]) for r in cand}
        g0 = cand[int(rng.integers(len(cand)))]; gates = [g0]
        while len(gates) < min(n_gates, len(cand)):
            def sep(r): return min(abs((ang[r] - ang[g] + math.pi) % (2 * math.pi) - math.pi) for g in gates)
            best = max(cand, key=sep)
            if sep(best) < 0.5: break
            gates.append(best)
    # graph of cell edges: cost = length × (1 + slope × price); the wall — only through the gates
    G = nx.Graph(); wall_v = set(ring) - set(gates) if wall else set()
    for i, L in enumerate(ids):
        for j in range(len(L)):
            a, b = L[j], L[(j + 1) % len(L)]
            if a == b: continue
            dl = float(np.linalg.norm(V[a] - V[b])); sl = 0.0
            if height is not None and dl > 0: sl = abs(float(height(V[a][0], V[a][1]) - height(V[b][0], V[b][1]))) / dl
            G.add_edge(a, b, w=dl * (1 + cost_slope * sl), cells=G.get_edge_data(a, b, {}).get("cells", set()) | {i})
    Gin = G.subgraph([n for n in G.nodes if n not in wall_v and Point(V[n]).distance(city) < 0.5]).copy()
    plaza_poly = Polygon(V[ids[0]]) if plaza and ids[0] else None
    targets = set(ids[0]) if plaza_poly is not None else {min(Gin.nodes, key=lambda n: np.linalg.norm(V[n] - c))}
    art = set()
    for g in gates:
        if g not in Gin: continue
        try: path = nx.multi_source_dijkstra(Gin, targets & set(Gin.nodes), target=g, weight="w")[1]
        except (nx.NetworkXNoPath, ValueError): continue
        for a, b in zip(path[:-1], path[1:]): art.add(frozenset((a, b)))
    wall_e = {frozenset((ring[j], ring[(j + 1) % len(ring)])) for j in range(len(ring))}
    lines = []; seen = set()
    for i in inner:
        L = ids[i]
        for j in range(len(L)):
            e = frozenset((L[j], L[(j + 1) % len(L)]))
            if len(e) < 2 or e in seen: continue
            seen.add(e)
            if e in wall_e: continue
            a, b = tuple(e); g = LineString([V[a], V[b]])
            if e in art: lines.append(line(g, "artery", width(W, "artery")))
            else: lines.append(line(g, "street", width(W, "street")))
    lines.append(line(city.exterior, "edge", 0.0, kind="edge"))
    # big blocks — lanes (chaotic splitting)
    lim = float(spec.get("lanes", 1e9)); thick = float(spec.get("thickness", 26.0)); lanes = []
    def side(h):
        C = np.asarray(h.minimum_rotated_rectangle.exterior.coords); return min(np.linalg.norm(C[1] - C[0]), np.linalg.norm(C[2] - C[1]))
    def cut(P, depth=0):
        if P.area <= lim or depth > 2: return
        halves, seg = bisect(P, chaos * 0.6, rng)
        if seg is None or len(halves) < 2 or min(side(h) for h in halves) < thick: return   # the halves must hold houses on both sides
        lanes.append(seg)
        for h in halves: cut(h, depth + 1)
    for i in inner:
        if i == 0 and plaza_poly is not None: continue
        if ids[i]: cut(Polygon(V[ids[i]]).buffer(0))
    for s in lanes: lines.append(line(s, lower(W, "street"), width(W, lower(W, "street"))))
    return {"lines": lines, "city": city, "wall": city if wall else None, "gates": [V[g].tolist() for g in gates],
            "plaza": plaza_poly, "cells": [Polygon(V[ids[i]]) for i in inner if ids[i]]}


# ---------------------------------------------------------------- growth (Parish–Müller, Emilien)
def _seg_x(p, q, a, b):
    """Intersection of segments pq and ab → (t on pq, u on ab) or None."""
    r = q - p; s = b - a; den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9: return None
    w = a - p; t = (w[0] * s[1] - w[1] * s[0]) / den; u = (w[0] * r[1] - w[1] * r[0]) / den
    return (t, u) if 1e-6 < t < 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6 else None


def growth(seeds, B, spec, W, rng, height=None, blocked=None, wet=None):
    """Network growth: seeds — [(point, direction (rad), rank)]; B — the outline; blocked(p) → True if building there is not
    allowed (moat, castle, cliff). Constraints (Parish–Müller): an intersection — a node and a stop; a node or edge nearby —
    snapping and a stop; beyond the boundary — clipping and a stop. Goals: direction with a slope cost (contour lines),
    branches at almost a right angle, the branch rank is lower."""
    L = float(spec.get("step", 20)); pb = float(spec.get("branching", 0.3)); turn = math.radians(float(spec.get("turn", 15)))
    snap = float(spec.get("snap", 8)); cs = float(spec.get("slope", 2.0)); nmax = int(spec.get("n_segments", 100))
    N = []; S = []                                                                # nodes; edges [a, b, rank]
    def node(p): N.append(np.asarray(p, float)); return len(N) - 1
    Q = []; cnt = 0
    for p, a, rk in seeds:
        heapq.heappush(Q, (0.0, cnt, node(p), a, rk)); cnt += 1
    delay = {"artery": 0.0, "street": 2.0, "lane": 4.0, "passage": 6.0}
    Bb = B.buffer(0)

    def slope(p, q):
        if height is None: return 0.0
        return abs(float(height(q[0], q[1]) - height(p[0], p[1]))) / max(np.linalg.norm(q - p), 1e-6)

    def angle_ok(n, a):
        for e in S:
            if n in (e[0], e[1]):
                o = e[1] if e[0] == n else e[0]; d = N[o] - N[n]; b = math.atan2(d[1], d[0])
                if abs((a - b + math.pi) % (2 * math.pi) - math.pi) < math.radians(32): return False
        return True
    while Q and len(S) < nmax:
        t, _, n0, a0, rk = heapq.heappop(Q)
        p0 = N[n0]; Lr = L * (1.0 if rk == "artery" else 0.8 if rk == "street" else 0.6)
        best = None
        for da in np.linspace(-turn, turn, 7):
            a = a0 + da; q = p0 + Lr * np.array([math.cos(a), math.sin(a)])
            cost = cs * slope(p0, q) + 0.25 * abs(da) / max(turn, 1e-6) + 0.1 * rng.random()
            if best is None or cost < best[0]: best = (cost, a, q)
        _, a, q = best
        if not angle_ok(n0, a): continue
        seg = LineString([p0, q])
        if not Bb.contains(Point(q)):                                              # boundary: clip and stop
            cut = seg.intersection(Bb)
            if cut.is_empty or cut.length < Lr * 0.3: continue
            q = np.asarray(cut.coords[-1] if cut.geom_type == "LineString" else list(cut.geoms)[0].coords[-1]); stop = True
        else: stop = False
        if blocked is not None and (blocked(q) or blocked((p0 + q) / 2)): continue
        if wet is not None and wet(q): continue                                    # the end is on the bank; a middle over water — a bridge
        hit = None                                                                 # intersection with the existing edges
        for k, (ea, eb, _) in enumerate(S):
            if n0 in (ea, eb): continue
            x = _seg_x(p0, q, N[ea], N[eb])
            if x and (hit is None or x[0] < hit[0]): hit = (x[0], x[1], k)
        if hit:
            t_, u_, k = hit; pi = p0 + (q - p0) * t_
            if np.linalg.norm(pi - p0) < 0.3 * Lr: continue
            ea, eb, er = S[k]; ni = node(pi); S[k] = [ea, ni, er]; S.append([ni, eb, er]); S.append([n0, ni, rk]); continue
        near = [i for i in range(len(N)) if i != n0 and np.linalg.norm(N[i] - q) < snap]
        if near:                                                                   # snap to a node
            ni = min(near, key=lambda i: np.linalg.norm(N[i] - q))
            if any({n0, ni} == {e[0], e[1]} for e in S): continue
            S.append([n0, ni, rk]); continue
        ni = node(q); S.append([n0, ni, rk])
        if stop: continue
        heapq.heappush(Q, (t + 1.0 + 0.3 * rng.random(), cnt, ni, a, rk)); cnt += 1
        for side in (-1, 1):
            if rng.random() < pb:
                r2 = lower(W, rk) if rk != "artery" or rng.random() < 0.7 else rk
                heapq.heappush(Q, (t + 1.0 + delay.get(r2, 3.0) + rng.random(), cnt, ni, a + side * (math.pi / 2 + rng.normal(0, 0.15)), r2)); cnt += 1
    lines = [line(LineString([N[a], N[b]]), r, width(W, r)) for a, b, r in S if np.linalg.norm(N[a] - N[b]) > 0.5]
    return lines


def snap_ends(lines_new, net, reach=15.0):
    """Passages and cuts end at the street edge — extend the ends to the axis of the nearest street (otherwise the graph breaks)."""
    axes = unary_union([l["geom"] for l in net if l["kind"] != "edge" and l["w"] > 0])
    out = []
    for g in lines_new:
        C = [np.asarray(q, float) for q in g.coords]
        for end in (0, -1):
            p = C[end]; q = np.asarray(shapely.ops.nearest_points(axes, Point(p))[0].coords[0])
            if 0.05 < np.linalg.norm(q - p) < reach:
                d = (q - p) / np.linalg.norm(q - p); qq = q + d * 0.05
                if end == 0: C = [qq] + C
                else: C = C + [qq]
        out.append(LineString(C))
    return out


def connect(lines, W, max_d=90.0, blocked=None):
    """Link the parts of the network (Parish–Müller: extend a dead end to a neighboring street): the nearest pair of nodes of
    the small part and the main one — by a segment of rank "street", if shorter than max_d and not crossing a forbidden
    area; if the nearest one hits a forbidden area — the next ones by distance (up to 40 pairs). Water is not forbidden —
    a bridge will go over it."""
    lines = list(lines)
    for _ in range(12):
        P, E = build_graph(lines); G = nx.Graph()
        for e in E: G.add_edge(e["a"], e["b"])
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(cc) < 2: break
        main = np.array([P[i] for i in cc[0]]); added = False
        for comp in cc[1:]:
            Q = np.array([P[i] for i in comp]); tree = cKDTree(main); kk = min(8, len(main))
            d, j = tree.query(Q, k=kk); d = np.asarray(d).reshape(len(Q), -1); j = np.asarray(j).reshape(len(Q), -1)
            pairs = sorted((float(d[a_, b_]), a_, int(j[a_, b_])) for a_ in range(len(Q)) for b_ in range(d.shape[1]))[:40]
            for dd, a_, b_ in pairs:
                if dd > max_d: break
                g = LineString([Q[a_], main[b_]])
                if blocked is not None and any(blocked(np.asarray(g.interpolate(t, normalized=True).coords[0])) for t in (0.25, 0.5, 0.75)): continue
                lines.append(line(g, "street", width(W, "street"))); added = True; break
        if not added: break
    return lines


# ---------------------------------------------------------------- graph and ribbon
def build_graph(lines):
    """Stitching of lines: intersections are nodes; the rank of a piece is the highest of the coinciding lines. → nodes
    [[x, y]], edges [{a, b, rank, w, kind, pts}] (pts — the piece's points, a and b — its ends)."""
    walk = [l for l in lines if l["kind"] != "edge"]
    U = unary_union([l["geom"] for l in walk])
    pieces = [g for g in getattr(U, "geoms", [U]) if g.geom_type == "LineString" and g.length > 0.2]
    tree = shapely.STRtree([l["geom"] for l in walk])
    P = []; key = {}
    def nid(p):
        k = (round(p[0], 2), round(p[1], 2))
        if k not in key: key[k] = len(P); P.append([round(p[0], 3), round(p[1], 3)])
        return key[k]
    E = []
    for g in pieces:
        m = g.interpolate(0.5, normalized=True)
        cand = [i for i in tree.query(m.buffer(0.05)) if walk[i]["geom"].distance(m) < 0.05]
        if not cand: continue
        best = min(cand, key=lambda i: (RANKS.index(walk[i]["rank"]) if walk[i]["rank"] in RANKS else 9, -walk[i]["w"]))
        l = walk[best]; C = list(g.coords)
        E.append({"a": nid(C[0]), "b": nid(C[-1]), "rank": l["rank"], "w": max(walk[i]["w"] for i in cand), "kind": l["kind"],
                  "pts": [[round(x, 3), round(y, 3)] for x, y in C]})
    return P, E


def ribbon(E, extra=0.0):
    """Street ribbon: the union of edge strips (round ends give rounded intersection corners)."""
    polys = [LineString(e["pts"]).buffer(e["w"] / 2 + extra, quad_segs=4) for e in E if e["w"] > 0]
    return unary_union(polys) if polys else Polygon()
