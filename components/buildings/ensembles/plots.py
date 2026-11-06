"""Plots of an ensemble: free land = outline − street ribbon − squares − forbidden; the blocks are its pieces.
Subdivision by archetype kind:
  strips     — along streets on both sides: facade × depth, the senior rank takes the facade first; deeper than half a block — no
               (Vanegas 2012: strips by the skeleton; burgage ~1 : 6);
  rows       — a block into two rows along a passage on the long axis, each row into houses of equal width (Olynthus: 2 × 5);
  whole      — a block is one plot, some of the blocks halved (Timgad: a domus or insula over the whole block);
  division   — recursive division with chaos; a piece without street access goes to a neighbor;
  built_up   — strips, then the block's yard is divided into back plots, each with a passage to the street (slum, Kowloon);
  farmsteads — strips with a large facade and gaps, sparser toward the edge (suburbs are thinned out);
  blocks     — a block into 1–3 hub blocks with a gap (hive).
Plot: outline, facades (a segment on the street, rank, outward normal to the street, street flow), area, purpose.
Influence adjustments (wealth, trade) — multipliers of facade and depth and the share of empty plots for strips (`mod`);
for the other kinds — in the ensemble."""
import os
import sys
from components.buildings import arch_parts
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from components.buildings.ensembles import streets as ST


def parts(g, min_area=1.0):
    """Polygons of a geometry (without empty and tiny ones)."""
    if g.is_empty: return []
    gs = getattr(g, "geoms", [g])
    return [p for p in gs if p.geom_type == "Polygon" and p.area >= min_area]


def fronts(P, E, tol=0.8):
    """Plot facades: outline edges lying at the street edge (distance to the axis ≈ half the width) → [{p0, p1, rank, w, n}]."""
    C = np.asarray(P.exterior.coords); out = []
    lines = [(LineString(e["pts"]), e) for e in E if e["w"] > 0]
    tree = shapely.STRtree([g for g, _ in lines])
    for i in range(len(C) - 1):
        p0, p1 = C[i], C[i + 1]; L = float(np.linalg.norm(p1 - p0))
        if L < 0.8: continue
        m = Point((p0 + p1) / 2); best = None
        for k in tree.query(m.buffer(12.0)):
            g, e = lines[k]; d = g.distance(m)
            if abs(d - e["w"] / 2) < tol + 0.05 * e["w"]:
                r = ST.RANKS.index(e["rank"]) if e["rank"] in ST.RANKS else (0 if e["kind"] == "bridge" else 1)
                if best is None or r < best[0]: best = (r, e)
        if best is None: continue
        d = (p1 - p0) / L; n = np.array([d[1], -d[0]])                               # the outline is counterclockwise: outward is on the right
        if not P.exterior.is_ccw: n = -n
        out.append({"p0": [round(float(p0[0]), 3), round(float(p0[1]), 3)], "p1": [round(float(p1[0]), 3), round(float(p1[1]), 3)],
                    "rank": best[1]["rank"], "w": best[1]["w"], "n": [round(float(n[0]), 4), round(float(n[1]), 4)], "len": round(L, 2),
                    **({"flow": best[1]["flow"]} if "flow" in best[1] else {})})
    return out


def _ray(block, p, n, far):
    """Length of the ray from p along n inside the block (up to the first exit)."""
    g = LineString([p, p + n * far]).intersection(block)
    best = 0.0
    for s in getattr(g, "geoms", [g]):
        if s.geom_type == "LineString" and s.distance(Point(p)) < 0.6: best = max(best, s.length)
    return best


def _chains(E, max_turn=40.0):
    """Whole streets: edges of the same rank and kind are stitched through nodes without a fork if the turn is less than
    max_turn (the flow on such a piece is one) — a plot may be wider than the network step (growth: 12–26 m), street
    corners remain corners."""
    groups = {}
    for e in E: groups.setdefault((e["rank"], e["kind"]), []).append(e)
    out = []; ct = math.cos(math.radians(max_turn))
    key = lambda p: (round(float(p[0]), 2), round(float(p[1]), 2))
    for (rk, kd), es in groups.items():
        ends = {}
        for i, e in enumerate(es): ends.setdefault(key(e["pts"][0]), []).append((i, 0)); ends.setdefault(key(e["pts"][-1]), []).append((i, 1))
        nxt = {}
        for lst in ends.values():
            if len(lst) != 2 or lst[0][0] == lst[1][0]: continue
            (i, a), (j, b) = lst; Pi = np.asarray(es[i]["pts"], float); Pj = np.asarray(es[j]["pts"], float)
            di = Pi[1] - Pi[0] if a == 0 else Pi[-2] - Pi[-1]; dj = Pj[1] - Pj[0] if b == 0 else Pj[-2] - Pj[-1]   # from the node along the edge
            if -(di @ dj) / max(np.linalg.norm(di) * np.linalg.norm(dj), 1e-9) < ct: continue
            nxt[(i, a)] = (j, b); nxt[(j, b)] = (i, a)
        seen = set()
        for i in range(len(es)):
            if i in seen: continue
            cur, end, n = i, 0, 0
            while (cur, end) in nxt and n < len(es):                              # to the start of the chain
                j, b = nxt[(cur, end)]
                if j == i: break
                cur, end, n = j, 1 - b, n + 1
            pts = []; mem = []; c_, e_ = cur, end
            while c_ not in seen:
                seen.add(c_); mem.append(es[c_]); P = es[c_]["pts"] if e_ == 0 else es[c_]["pts"][::-1]
                pts += [list(q) for q in (P if not pts else P[1:])]
                if (c_, 1 - e_) not in nxt: break
                c_, e_ = nxt[(c_, 1 - e_)]
            out.append(mem[0] if len(mem) == 1 else {"pts": pts, "rank": rk, "kind": kd, "w": max(e["w"] for e in mem),
                                                     **({"flow": max(e.get("flow", 0.0) for e in mem)} if any("flow" in e for e in mem) else {})})
    return out


def strips(avail, E, f_rng, d_rng, rng, empty=0.0, empty_fn=None, ranks=None, min_share=0.45, min_area=None, mod=None):
    """Strips along streets: facade-width steps on both sides, depth no more than half a block; mod(point at the street, edge
    flow) → (× facade, × depth, + share of empty ones) — the influence adjustments (influence.py). → (plots, remainder)."""
    out = []
    order = sorted(_chains([e for e in E if e["w"] > 0 and e["kind"] != "bridge" and (ranks is None or e["rank"] in ranks)]),
                   key=lambda e: (ST.RANKS.index(e["rank"]) if e["rank"] in ST.RANKS else 1, -LineString(e["pts"]).length))
    for e in order:
        g = LineString(e["pts"]); Lt = g.length; hw = e["w"] / 2
        if Lt < f_rng[0] * 0.7: continue
        for side in (1, -1):
            s = rng.uniform(0.0, 0.4) * f_rng[0]
            while s < Lt - 0.5 * f_rng[0]:
                fm = dm = 1.0; em = 0.0
                if mod is not None:                                              # influences: adjustments at a point by the street
                    a_ = np.asarray(g.interpolate(min(Lt, s + f_rng[0] / 2)).coords[0]); b_ = np.asarray(g.interpolate(min(Lt, s + f_rng[0] / 2 + 0.5)).coords[0])
                    t_ = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
                    fm, dm, em = mod(a_ + np.array([-t_[1], t_[0]]) * side * (hw + 0.5 * float(np.mean(d_rng))), e.get("flow", 0.0))
                f = rng.uniform(*f_rng) * fm; s1 = min(Lt, s + f)
                if Lt - s1 < 0.45 * f_rng[0] * fm: s1 = Lt                          # the tail goes to the last one
                sub = substring(g, s, s1)
                s = s1
                if sub.length < 0.5 * f_rng[0]: continue
                C = np.asarray(sub.coords); d = C[-1] - C[0]; d = d / max(np.linalg.norm(d), 1e-9); nrm = np.array([-d[1], d[0]]) * side
                m = (C[0] + C[-1]) / 2 + nrm * (hw + 0.3)
                comp = [q for q in parts(avail, 4.0) if q.distance(Point(m)) < 0.8]
                if not comp: continue
                blk = min(comp, key=lambda q: q.distance(Point(m)))
                depth = rng.uniform(*d_rng) * dm; ray = _ray(blk, m, nrm, depth * 2.6 + 10)
                if ray < 2.0: continue
                if ray < 2.15 * depth: depth = ray / 2 + 0.2                        # narrow block — up to the middle
                front = sub.offset_curve(side * (hw + 0.05)); back = sub.offset_curve(side * (hw + depth))
                front, back = (max(x.geoms, key=lambda y: y.length) if x.geom_type == "MultiLineString" else x for x in (front, back))   # a bend — take the long piece
                if front.is_empty or back.is_empty or front.geom_type != "LineString" or back.geom_type != "LineString": continue
                Fc = list(front.coords); Bc = list(back.coords)
                if np.dot(np.asarray(Fc[-1]) - Fc[0], np.asarray(Bc[-1]) - Bc[0]) < 0: Bc = Bc[::-1]
                quad = Polygon(Fc + Bc[::-1]).buffer(0)
                piece = quad.intersection(blk)
                cand = [q for q in parts(piece, 2.0) if q.distance(front) < 0.6]
                if not cand: continue
                q = max(cand, key=lambda x: x.area)
                if q.area < max(min_share * f * depth, min_area if min_area is not None else 0.75 * f_rng[0] * min(d_rng[0], 5.0)): continue
                if q.intersection(front.buffer(0.7)).area < 0.4 * f * 0.7: continue                  # a facade on the street, not just a corner
                pe = (empty_fn(q) if empty_fn else empty) + em
                avail = avail.difference(q.buffer(0.01))
                if rng.random() < pe: continue                                   # empty plot (garden, wasteland)
                out.append({"poly": q, "street": e["rank"]})
    return out, avail


def rows(block, f_rng, gap, rng):
    """Olynthus: a block into two rows along a passage on the long axis; each row into houses of equal width."""
    R = block.minimum_rotated_rectangle; C = np.asarray(R.exterior.coords)[:4]
    if np.linalg.norm(C[1] - C[0]) < np.linalg.norm(C[2] - C[1]): C = np.roll(C, -1, 0)   # C0 → C1 is the long side
    u = (C[1] - C[0]); Lu = np.linalg.norm(u); u = u / Lu; v = C[3] - C[0]; Lv = np.linalg.norm(v); v = v / max(Lv, 1e-9)
    halves = []
    if Lv > 2 * 7.0 + gap:
        for lo, hi in ((0.0, Lv / 2 - gap / 2), (Lv / 2 + gap / 2, Lv)):
            halves.append((lo, hi))
    else: halves.append((0.0, Lv))
    f = rng.uniform(*f_rng); n = max(1, int(round(Lu / f))); out = []
    for lo, hi in halves:
        for i in range(n):
            a, b = Lu * i / n, Lu * (i + 1) / n
            q = Polygon([C[0] + u * a + v * lo, C[0] + u * b + v * lo, C[0] + u * b + v * hi, C[0] + u * a + v * hi]).intersection(block)
            for p in parts(q, 6.0): out.append({"poly": p})
    return out


def whole(block, share, rng, told=None):
    """A block is a plot; a share of the blocks is halved across the long side. told — a list: whether splitting was ordered
    (plan check: ordered — and the cut was made)."""
    cut = rng.random() < share and block.area > 260
    if told is not None: told.append(cut)
    if cut:
        halves, _ = ST.bisect(block, 0.1, rng)
        return [{"poly": h} for h in halves if h.area > 6]
    return [{"poly": block}]


def divide(block, area, chaos, rng, gap=0.0, depth=0):
    """Alleys without the lanes: splitting across the long side with chaos while the piece is larger than the
    target area (± spread)."""
    if block.area < area * 2 ** (4 * 0.5 * (rng.random() - 0.5)) or depth > 8: return [{"poly": block}]
    halves, _ = ST.bisect(block, chaos, rng, gap=gap)
    if len(halves) < 2: return [{"poly": block}]
    out = []
    for h in halves:
        for p in parts(h, 6.0): out += divide(p, area, chaos, rng, gap, depth + 1)
    return out


def divide_cuts(block, area, chaos, rng, depth=0):
    """Block division with cuts (the hive's service passages): → (plots, cut lines)."""
    if block.area < area or depth > 6: return [{"poly": block}], []
    halves, seg = ST.bisect(block, chaos, rng)
    if len(halves) < 2 or seg is None: return [{"poly": block}], []
    out, cuts = [], [seg]
    for h in halves:
        for p in parts(h, 6.0):
            o, c = divide_cuts(p, area, chaos, rng, depth + 1); out += o; cuts += c
    return out, cuts


def infill(rest, ribbon, E, area, pw, rng):
    """Yard infill: the remainder of the block → back plots; each one without a facade gets a passage of width pw to the
    nearest street (it cuts the front plots, as in a slum). → (plots, passage lines)."""
    out = []; lanes = []
    edge = ribbon.boundary
    for blk in parts(rest, 30.0):
        for d in divide(blk, area, 0.7, rng):
            P = d["poly"]
            if P.area < 25: continue
            if P.distance(ribbon) < 0.6: out.append(d); continue
            c = np.asarray(P.representative_point().coords[0]); q = np.asarray(shapely.ops.nearest_points(edge, Point(c))[0].coords[0])
            ln = LineString([c, q])
            if ln.length > 80: continue
            lanes.append(ln); out.append(d)
    return out, lanes


def subtract(plots, geom):
    """Subtract a geometry from the plots (passages cut the front houses) → plots (pieces with an area)."""
    out = []
    for p in plots:
        g = p["poly"].difference(geom)
        for q in parts(g, 6.0): out.append(dict(p, poly=q))
    return out
