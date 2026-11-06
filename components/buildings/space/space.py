"""The space scheme of a building: cells — rooms and platforms (inside, yard, outside, covered outside, void, underground),
boundaries (a wall, an arcade, a virtual one — IFC VIRTUAL; internal or external), transitions (a door, a gate, an arch,
an opening, a stair, a ramp, a bridge, a crawlway, a stair in the thickness of a wall) and windows, anchors for the scene;
a graph of the cells (IndoorGML NRG): reachability from the world, depth, entrances, bottlenecks (graph bridges), paths;
the checks and the plan drawing — without Blender. The scheme is built by the plans (temples/temple_space.py …); the
builds take the cell and the face side from it (space/bl_space.py).
Outlines are world xy, elevations are world z; frame — the building's frame (for the drawing: u right, v up).

.venv/bin/python -m components.buildings.space.space <plan.json> [draw=drawing.png] [path=from,to]   → a summary, checks, __RESULT__"""
import heapq, json, math, os, sys

import numpy as np

CLASSES = ("exterior", "interior", "court", "covered", "void", "underground")    # the class index is the sk_side code
CLASS_RU = {"interior": "inside", "court": "court", "exterior": "outside", "covered": "covered outside", "void": "void", "underground": "underground"}
OPEN_CLS = ("exterior", "court")                                                 # under the sky: the boundary with them is external
ENCLOSED = ("interior", "covered", "underground")
WALK_KINDS = ("door", "gate", "arch", "opening", "stair", "ramp", "bridge", "ladder", "spiral")
PASS_KINDS = ("door", "gate", "arch", "opening")                                 # across a boundary (not up-down)
COLORS = {"interior": (236, 200, 160), "court": (196, 226, 178), "exterior": (232, 226, 206), "covered": (184, 208, 232),
          "void": (205, 205, 205), "underground": (170, 150, 190)}


def area(P):
    P = np.asarray(P, float); x, y = P[:, 0], P[:, 1]; return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def signed_area(P):
    P = np.asarray(P, float); x, y = P[:, 0], P[:, 1]; return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside(P, x, y):
    """Points (arrays x, y) inside the polygon P — even-odd."""
    P = np.asarray(P, float); x = np.asarray(x, float); y = np.asarray(y, float); r = np.zeros(np.broadcast(x, y).shape, bool)
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % n]
        if y1 == y2: continue
        r ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return r


def seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b)); d = b - a; L2 = float(d @ d)
    t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ d / L2, 0, 1)); return float(np.linalg.norm(p - (a + d * t)))


def poly_dist(P, p):
    """The distance from a point to a polygon (0 — inside)."""
    if P is None or bool(inside(P, p[0], p[1])): return 0.0
    return min(seg_dist(p, P[i], P[(i + 1) % len(P)]) for i in range(len(P)))


def inner_point(P, n=24, holes=()):
    """The point inside a polygon (not in the holes) farthest from the edges (on a grid) — for a label, an anchor, a check ray."""
    P = np.asarray(P, float); lo, hi = P.min(0), P.max(0); Hs = [np.asarray(H, float) for H in holes or ()]
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], n + 2)[1:-1], np.linspace(lo[1], hi[1], n + 2)[1:-1]); X, Y = X.ravel(), Y.ravel()
    m = inside(P, X, Y)
    for H in Hs: m &= ~inside(H, X, Y)
    if not m.any(): return P.mean(0)
    X, Y = X[m], Y[m]; d = np.full(len(X), np.inf)
    for Q in [P] + Hs:
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]; e = b - a; L2 = max(float(e @ e), 1e-12)
            t = np.clip(((X - a[0]) * e[0] + (Y - a[1]) * e[1]) / L2, 0, 1); d = np.minimum(d, np.hypot(X - a[0] - e[0] * t, Y - a[1] - e[1] * t))
    k = int(np.argmax(d)); return np.array([X[k], Y[k]])


def r2(v): return round(float(v), 2)


def offset_convex(P, d):
    """A convex polygon inset by d (counterclockwise): a vertex is the intersection of the shifted edges (like
    bl_arch.offset_poly)."""
    P = [np.asarray(p, float) for p in P]
    if signed_area(P) < 0: P = P[::-1]
    n = len(P); out = []
    for i in range(n):
        a_, b_, c_ = P[i - 1], P[i], P[(i + 1) % n]; e1 = b_ - a_; e2 = c_ - b_
        n1 = np.array([-e1[1], e1[0]]) / max(np.linalg.norm(e1), 1e-9); n2 = np.array([-e2[1], e2[0]]) / max(np.linalg.norm(e2), 1e-9)
        out.append(b_ + d * (n1 + n2) / max(1.0 + float(n1 @ n2), 1e-3))
    return out


class Space:
    """The space scheme: cells, boundaries, transitions, windows, anchors."""

    def __init__(self, frame=None, human=1.8):
        self.cells = []; self.idx = {}; self.bounds = []; self.trans = []; self.wins = []; self.anchors = []; self.rel = {}
        self.frame = frame; self.human = human; self._np = {}
        self.cell("outside", "world", None, -1e4, None, "exterior", why="everything that is not a building")

    # ---- construction
    def cell(self, id_, kind, poly, z0, z1=None, cls="interior", walk=True, sky=None, use=None, access="public", parent=None, edge="wall", why="", holes=None):
        """A cell: the outline (world, xy; None — the surrounding world), the floor z0, the ceiling z1 (None — the sky), the class;
        edge — the default boundary; holes — holes (a gallery ring, the terrace around the tier above)."""
        if cls not in CLASSES: raise ValueError(f"class {cls}")
        if id_ in self.idx: raise ValueError(f"duplicate cell {id_}")
        P = None if poly is None else [[r2(p[0]), r2(p[1])] for p in poly]
        if P is not None and signed_area(P) < 0: P = P[::-1]                     # counterclockwise: outward is to the right of the edge
        c = {"id": id_, "kind": kind, "poly": P, "z0": r2(z0), "z1": None if z1 is None else r2(z1), "cls": cls, "walk": bool(walk),
             "sky": bool(cls in OPEN_CLS if sky is None else sky), "use": use, "access": access, "parent": parent, "edge": edge}
        if holes:                                                                # holes are clockwise: outward from the cell is into the hole
            c["holes"] = [[[r2(p[0]), r2(p[1])] for p in (H if signed_area(H) < 0 else H[::-1])] for H in holes]
        if why: c["why"] = why
        self.idx[id_] = len(self.cells); self.cells.append(c); return id_

    def relate(self, a, b, kind):
        """A boundary between cells a and b: wall, arcade (a wall with arches), open (virtual), parapet, edge (a drop), rail."""
        self.rel[frozenset((a, b))] = kind

    def link(self, a, b, kind, p, za, zb, w, h=None, length=None, n=None, hidden=False, ground=None, why="", **kw):
        """A transition a → b: a point p (a door, an arch, an opening) or a polyline [p0, …, p1] from a to b (a stair, a ramp, a
        bridge); n — the direction a → b across the boundary; hidden — a stair in the thickness of a wall (not visible in the
        model); ground — the ground at the entrance outside."""
        seg = isinstance(p[0], (list, tuple, np.ndarray))
        L = length if length is not None else (sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1)) if seg else 0.0)
        d = {"a": a, "b": b, "kind": kind, "p": [[r2(q[0]), r2(q[1])] for q in p] if seg else [r2(p[0]), r2(p[1])],
             "za": r2(za), "zb": r2(zb), "w": r2(w), "h": None if h is None else r2(h), "len": r2(L)}
        if kind in ("stair", "ramp", "spiral", "ladder"): d["slope"] = round(math.degrees(math.atan2(abs(zb - za), max(L, 1e-6))), 1)
        if n is not None: nn = np.asarray(n, float); nn = nn / max(np.linalg.norm(nn), 1e-9); d["n"] = [round(float(nn[0]), 4), round(float(nn[1]), 4)]
        if hidden: d["hidden"] = True
        if ground is not None: d["ground"] = r2(ground)
        if why: d["why"] = why
        d.update(kw); self.trans.append(d); return d

    def window(self, a, b, p, z0, w, h):
        self.wins.append({"a": a, "b": b, "p": [r2(p[0]), r2(p[1])], "z0": r2(z0), "w": r2(w), "h": r2(h)})

    def anchor(self, name, cell, p, z, facing=None, tags=None):
        a = {"name": name, "cell": cell, "p": [r2(p[0]), r2(p[1])], "z": r2(z)}
        if facing is not None: f = np.asarray(facing, float); f = f / max(np.linalg.norm(f), 1e-9); a["facing"] = [round(float(f[0]), 3), round(float(f[1]), 3)]
        if tags: a["tags"] = tags
        self.anchors.append(a)

    # ---- cell geometry
    def poly(self, i):
        if i not in self._np: c = self.cells[i]; self._np[i] = None if c["poly"] is None else np.asarray(c["poly"], float)
        return self._np[i]

    def rings(self, i):
        """The outline and holes of a cell (arrays)."""
        return [self.poly(i)] + [np.asarray(H, float) for H in self.cells[i].get("holes", [])]

    def inside_cell(self, i, x, y):
        m = inside(self.poly(i), x, y)
        for H in self.cells[i].get("holes", []): m &= ~inside(H, x, y)
        return m

    def area_cell(self, i):
        return area(self.poly(i)) - sum(area(H) for H in self.cells[i].get("holes", []))

    def locate(self, P, tol_lo=0.12, tol_hi=0.12, walk_only=False, skip=None):
        """The cell of points P (N×3, world): among the cells where the point is inside the outline and between the floor and the
        ceiling — the one with the highest floor, and among equals the smaller by area (a sanctuary on a terrace); a void (a tower
        body, an attic) takes precedence if the point is in it; outside all — 0 (the world)."""
        P = np.asarray(P, float).reshape(-1, 3); n = len(P); best = np.zeros(n, int); bz = np.full(n, -np.inf); ba = np.full(n, np.inf)
        vd = np.zeros(n, bool)
        order = sorted(range(len(self.cells)), key=lambda i: self.cells[i]["cls"] == "void" or bool(self.cells[i].get("solid")))   # voids and bodies last
        for i in order:
            c = self.cells[i]
            Q = self.poly(i)
            if Q is None or (walk_only and not c["walk"]) or i == skip: continue
            lo, hi = Q.min(0), Q.max(0); z1 = np.inf if c["z1"] is None else c["z1"] + tol_hi; zl = c.get("zb", c["z0"])   # a solid body — from the bottom
            m = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]) & (P[:, 2] >= zl - tol_lo) & (P[:, 2] <= z1)
            if not m.any(): continue
            ix = np.nonzero(m)[0]; ix = ix[self.inside_cell(i, P[ix, 0], P[ix, 1])]
            if not len(ix): continue
            a = self.area_cell(i); z = c["z0"]
            if c["cls"] == "void" or c.get("solid"): better = ~vd[ix] | (z > bz[ix] + 1e-6); vd[ix[better]] = True
            else: better = (z > bz[ix] + 1e-6) | ((np.abs(z - bz[ix]) <= 1e-6) & (a < ba[ix]))
            ix = ix[better]; best[ix] = i; bz[ix] = z; ba[ix] = a
        return best

    def auto_bounds(self, step=1.0, probe=0.35):
        """Cell boundaries by adjacency: along each edge — points every ~1 m, the neighbor is the cell beyond the edge at human
        height; the boundary kind is by the pair (relate) or the cell's default. Each cell has its own boundary (IFC: the first
        level)."""
        self.bounds = []
        for i, c in enumerate(self.cells):
            Q = self.poly(i)
            if Q is None or c["cls"] == "void": continue
            zc = c["z0"] + min(1.0, 0.45 * ((c["z1"] - c["z0"]) if c["z1"] is not None else 2.0))
            for Q, e in [(Q_, e_) for Q_ in self.rings(i) for e_ in range(len(Q_))]:
                a_, b_ = Q[e], Q[(e + 1) % len(Q)]; L = float(np.linalg.norm(b_ - a_))
                if L < 0.05: continue
                d = (b_ - a_) / L; out = np.array([d[1], -d[0]]); k = max(1, int(math.ceil(L / step)))
                ts = (np.arange(k) + 0.5) / k * L; pts = a_ + np.outer(ts, d) + out * probe
                nb = self.locate(np.column_stack([pts, np.full(k, zc)]), tol_lo=1.0, skip=i)
                s0 = 0
                for j in range(1, k + 1):
                    if j < k and nb[j] == nb[s0]: continue
                    o = self.cells[nb[s0]]; kind = self.rel.get(frozenset((c["id"], o["id"])), c["edge"])
                    ext = "external" if (c["cls"] in OPEN_CLS) != (o["cls"] in OPEN_CLS) else "internal"
                    self.bounds.append({"a": c["id"], "b": o["id"], "p0": [r2(v) for v in a_ + d * (s0 * L / k)], "p1": [r2(v) for v in a_ + d * (j * L / k)],
                                        "z0": c["z0"], "z1": c["z1"] if c["z1"] is not None else r2(c["z0"] + 1.1), "kind": kind,
                                        "phys": kind not in ("open", "edge"), "side": ext})
                    s0 = j

    # ---- graph
    def edges(self):
        E = {}
        for i, t in enumerate(self.trans):
            if t["kind"] not in WALK_KINDS: continue
            if not (self.cells[self.idx[t["a"]]]["walk"] and self.cells[self.idx[t["b"]]]["walk"]): continue
            cost = max(t["len"], 1.0) + abs(t["zb"] - t["za"]) * 2.0
            E.setdefault(t["a"], []).append((t["b"], cost, i)); E.setdefault(t["b"], []).append((t["a"], cost, i))
        return E

    def dijkstra(self, src="outside", skip=None):
        E = self.edges(); dist = {src: 0.0}; depth = {src: 0}; prev = {}; q = [(0.0, src)]
        while q:
            d, u = heapq.heappop(q)
            if d > dist.get(u, 1e18): continue
            for v, c, i in E.get(u, []):
                if i == skip: continue
                nd = d + c
                if nd < dist.get(v, 1e18): dist[v] = nd; depth[v] = depth[u] + 1; prev[v] = (u, i); heapq.heappush(q, (nd, v))
        return dist, depth, prev

    def path(self, a, b):
        """A path a → b: [(cell, the index of the transition used to enter)…] or None."""
        dist, depth, prev = self.dijkstra(a)
        if b not in dist: return None
        out = []; u = b
        while u != a: p, i = prev[u]; out.append((u, i)); u = p
        return [(a, None)] + out[::-1]

    def entrances(self):
        return [i for i, t in enumerate(self.trans) if t["kind"] in WALK_KINDS and
                ((self.cells[self.idx[t["a"]]]["cls"] in OPEN_CLS) != (self.cells[self.idx[t["b"]]]["cls"] in OPEN_CLS))]

    def chokepoints(self):
        """Graph bridges (Tarjan): the transitions without which some walkable cells are cut off from the world — the bottlenecks of
        defense and story; cuts — how many cells lie beyond the bridge."""
        E = self.edges(); tin, low, size = {}, {}, {}; out = []; t_ = [0]
        if "outside" not in E: return out
        stack = [("outside", None, iter(E["outside"]))]; tin["outside"] = low["outside"] = 0; size["outside"] = 1
        while stack:
            u, pe, it = stack[-1]
            for v, _, i in it:
                if i == pe: continue
                if v in tin: low[u] = min(low[u], tin[v]); continue
                t_[0] += 1; tin[v] = low[v] = t_[0]; size[v] = 1; stack.append((v, i, iter(E.get(v, [])))); break
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]; low[p] = min(low[p], low[u]); size[p] += size[u]
                    if low[u] > tin[p]:
                        t = self.trans[pe]; out.append({"trans": pe, "kind": t["kind"], "between": [t["a"], t["b"]], "cuts": size[u]})
        return out

    # ---- checks
    def checks(self, steep=None, headroom=2.2, door_h=2.0):
        out = []; steep = steep or float((self.frame or {}).get("steep", 45.0))
        def chk(what, ok, **kw): out.append(dict({"what": what, "ok": bool(ok)}, **kw))
        walkc = [c for c in self.cells if c["walk"] and c["cls"] != "void" and c["id"] != "outside"]
        dist, depth, _ = self.dijkstra(); unreach = [c["id"] for c in walkc if c["id"] not in dist]
        chk("every walkable cell is reachable from the world", not unreach, unreachable=unreach[:8], total=len(walkc))
        chk("there is an entrance from outside to inside", bool(self.entrances()) or not any(c["cls"] in ENCLOSED and c["walk"] for c in self.cells), n_entrances=len(self.entrances()))
        bad = []
        for t in self.trans:
            for side, z in (("a", t["za"]), ("b", t["zb"])):
                c = self.cells[self.idx[t[side]]]; p = (t["p"][0] if side == "a" else t["p"][-1]) if isinstance(t["p"][0], list) else t["p"]
                if c["poly"] is None: continue
                if poly_dist(c["poly"], p) > 1.5 + t["w"] / 2: bad.append(f"{t['kind']} {t['a']}→{t['b']}: not at {t[side]}")
                elif abs(z - c["z0"]) > 1.2 and c["cls"] != "void": bad.append(f"{t['kind']} {t['a']}→{t['b']}: elevation {z} at {t[side]} {c['z0']}")
        chk("transitions join the cells (in plan and in elevation)", not bad, fails=bad[:6])
        nb = []                                                                  # a door and an arch lie within the boundary of their cells
        for t in self.trans:
            if t["kind"] not in PASS_KINDS or isinstance(t["p"][0], list): continue
            pair = {t["a"], t["b"]}
            if not any({b["a"], b["b"]} == pair and seg_dist(t["p"], b["p0"], b["p1"]) <= t["w"] / 2 + 0.8 for b in self.bounds):
                nb.append(f"{t['kind']} {t['a']}→{t['b']}")
        chk("doors and arches lie within the boundary between their cells", not nb, beyond_bounds=nb[:6])
        jump = [f"{t['kind']} {t['a']}→{t['b']} {abs(t['za'] - t['zb']):.2f} m" for t in self.trans if t["kind"] in PASS_KINDS and abs(t["za"] - t["zb"]) > 1.0]
        chk("passages with no level change over 1 m (otherwise it is a stair)", not jump, height_drops=jump[:6])
        st = [t for t in self.trans if t["kind"] in ("stair", "ramp") and not t.get("hidden")]
        steep_ = [f"{t['a']}→{t['b']} {t['slope']}°" for t in st if t["slope"] > steep + 1e-6]
        chk(f"stairs and ramps are passable (≤ {steep:.0f}°)", not steep_, steep=steep_[:6], total=len(st))
        gr = [f"{t['a']}→{t['b']}: threshold {t['za'] if self.cells[self.idx[t['a']]]['poly'] is None else t['zb']}, ground {t['ground']}"
              for t in self.trans if "ground" in t and abs((t["za"] if self.cells[self.idx[t["a"]]]["poly"] is None else t["zb"]) - t["ground"]) > 0.6]
        chk("entrances from outside are from the ground (the bottom of the steps at the ground, ±0.6 m)", not gr, hanging=gr[:6], n_entrances_on_ground=sum("ground" in t for t in self.trans))
        low = [c["id"] for c in self.cells if c["cls"] in ENCLOSED and c["walk"] and c["z1"] is not None and c["z1"] - c["z0"] < headroom]
        chk(f"headroom no lower than {headroom} m", not low, low=low[:6])
        dl = [f"{t['a']}→{t['b']} {t['h']}" for t in self.trans if t["kind"] in ("door", "gate", "arch") and t.get("h") is not None and t["h"] < door_h]
        chk(f"doors and arches sized for a person (≥ {door_h} m)", not dl, low=dl[:6])
        leak = []                                                                # envelope: the edges of the interior cells are closed by boundaries
        for i, c in enumerate(self.cells):
            if c["cls"] not in ENCLOSED or c["poly"] is None: continue
            mine = [b for b in self.bounds if b["a"] == c["id"]]
            for Q, e in [(Q_, e_) for Q_ in self.rings(i) for e_ in range(len(Q_))]:
                a_, b_ = Q[e], Q[(e + 1) % len(Q)]; Le = float(np.linalg.norm(b_ - a_))
                if Le < 0.3: continue
                d = (b_ - a_) / Le; cov = []
                for b in mine:
                    q0, q1 = np.asarray(b["p0"], float), np.asarray(b["p1"], float)
                    if max(seg_dist(q0, a_, b_), seg_dist(q1, a_, b_)) > 0.6: continue
                    s0, s1 = sorted(((q0 - a_) @ d, (q1 - a_) @ d)); cov.append((max(0.0, s0), min(Le, s1)))
                cov.sort(); got = 0.0; cur = 0.0
                for s0, s1 in cov:
                    if s1 <= cur: continue
                    got += s1 - max(s0, cur); cur = s1
                if got < 0.95 * Le: leak.append(f"{c['id']} edge {e}: closed {got / Le:.0%}")
        chk("the envelope of the interior cells is closed (the edges are a wall, an arcade or a virtual boundary)", not leak, leaks=leak[:6])
        open_out = [f"{b['a']} {math.dist(b['p0'], b['p1']):.1f} m" for b in self.bounds
                    if b["b"] == "outside" and b["kind"] == "open" and self.cells[self.idx[b["a"]]]["cls"] == "interior"]
        chk("the interior is not opened to the world by a virtual boundary", not open_out, exposed=open_out[:6])
        over = []                                                                # covered cells at one elevation do not overlap
        cs = [i for i, c in enumerate(self.cells) if c["cls"] in ENCLOSED and c["poly"] is not None and c["walk"]]
        for x in range(len(cs)):
            for y in range(x + 1, len(cs)):
                a_, b_ = self.cells[cs[x]], self.cells[cs[y]]
                if abs(a_["z0"] - b_["z0"]) > 1.5 or a_["parent"] == b_["id"] or b_["parent"] == a_["id"]: continue
                A, Bp = self.poly(cs[x]), self.poly(cs[y])
                lo = np.maximum(A.min(0), Bp.min(0)); hi = np.minimum(A.max(0), Bp.max(0))
                if (hi - lo).min() <= 0.05: continue
                X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 16), np.linspace(lo[1], hi[1], 16))
                frac = float((self.inside_cell(cs[x], X, Y) & self.inside_cell(cs[y], X, Y)).mean()) * float(np.prod(hi - lo)) / max(min(self.area_cell(cs[x]), self.area_cell(cs[y])), 1e-6)
                if frac > 0.05: over.append(f"{a_['id']}×{b_['id']} {frac:.0%}")
        chk("covered walkable cells at one elevation do not overlap", not over, overlapping=over[:6])
        return out

    # ---- output
    def to_dict(self, checks=True):
        if not self.bounds: self.auto_bounds()
        dist, depth, _ = self.dijkstra()
        walk_area = sum(self.area_cell(i) for i, c in enumerate(self.cells) if c["walk"] and c["poly"] is not None and c["cls"] in ENCLOSED + ("court",))
        st = {"n_cells": len(self.cells) - 1, "by_classes": {CLASS_RU[k]: sum(c["cls"] == k for c in self.cells[1:]) for k in CLASSES},
              "n_transitions": len(self.trans), "n_entrances": len(self.entrances()), "n_windows": len(self.wins), "n_anchors": len(self.anchors),
              "depth_max": max([d for k, d in depth.items() if k != "outside"] or [0]), "area_inside_m2": round(walk_area, 0),
              "capacity_ppl": int(walk_area / 0.6)}
        d = {"frame": self.frame, "cells": self.cells, "rel": [[*sorted(k), v] for k, v in self.rel.items()], "bounds": self.bounds, "trans": self.trans,
             "windows": self.wins, "anchors": self.anchors, "graph": {"depth": depth, "dist": {k: round(v, 1) for k, v in dist.items()},
                                                                      "entrances": self.entrances(), "choke": self.chokepoints()}, "stats": st}
        if checks: d["checks"] = self.checks()
        return d

    @classmethod
    def from_dict(cls, d):
        S = cls(frame=d.get("frame")); S.cells = d["cells"]; S.idx = {c["id"]: i for i, c in enumerate(S.cells)}
        S.rel = {frozenset(r[:2]): r[2] for r in d.get("rel", [])}
        S.bounds, S.trans, S.wins, S.anchors = d["bounds"], d["trans"], d.get("windows", []), d.get("anchors", []); return S

    def draw(self, path, title="", px=1500, route=None):
        """A space plan (the building's frame: u right, v up): cells by class (higher ones on top, voids are not drawn), walls,
        arcades, virtual boundaries, doors, arches, stairs, anchors, the depth from the world (a number in the cell); route — a
        path."""
        from PIL import Image, ImageDraw, ImageFont
        fr = self.frame or {}; O = np.asarray(fr.get("origin", [0.0, 0.0]), float); AX = np.asarray(fr.get("ax", [1.0, 0.0]), float); PV = np.array([-AX[1], AX[0]])
        def loc(p): q = np.asarray(p, float) - O; return np.array([q @ AX, q @ PV])
        cs = [c for c in self.cells if c["poly"] is not None and c["cls"] != "void"]
        if not cs: return
        allp = np.vstack([np.array([loc(p) for p in c["poly"]]) for c in cs]); lo = allp.min(0) - 4; hi = allp.max(0) + 4
        sc = px / max(hi - lo); W_, H_ = int((hi - lo)[0] * sc) + 1, int((hi - lo)[1] * sc) + 1
        im = Image.new("RGB", (W_ + 380, max(H_, 560) + 70), "white"); d = ImageDraw.Draw(im)
        F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14); F2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        def T(p): q = loc(p); return (float((q[0] - lo[0]) * sc), float(64 + (hi[1] - q[1]) * sc))
        _, depth, _ = self.dijkstra(); zs = sorted({c["z0"] for c in cs})
        for c in sorted(cs, key=lambda c: (c["z0"], -area(c["poly"]))):
            k = 1.0 - 0.35 * (zs.index(c["z0"]) / max(1, len(zs) - 1)) if len(zs) > 1 else 1.0
            col = tuple(int(v * k) for v in COLORS[c["cls"]]); d.polygon([T(p) for p in c["poly"]], fill=col, outline=(140, 140, 140))
            for H in c.get("holes", []): d.polygon([T(p) for p in H], fill=(255, 255, 255), outline=(140, 140, 140))
        for b in self.bounds:
            q0, q1 = T(b["p0"]), T(b["p1"])
            if b["kind"] == "wall": d.line([q0, q1], fill=(30, 30, 30), width=3)
            elif b["kind"] == "arcade": d.line([q0, q1], fill=(90, 90, 90), width=2)
            elif b["kind"] in ("parapet", "rail"): d.line([q0, q1], fill=(120, 80, 40), width=2)
            elif b["kind"] == "edge": d.line([q0, q1], fill=(190, 60, 60), width=1)
        for w in self.wins:
            q = T(w["p"]); d.rectangle([q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2], fill=(40, 170, 220))
        for t in self.trans:
            if isinstance(t["p"][0], list):
                Q = [T(q) for q in t["p"]]; col = (40, 90, 210) if t["kind"] in ("stair", "ramp", "spiral", "ladder") else (140, 60, 170)
                if t.get("hidden"): col = (120, 150, 220)
                d.line(Q, fill=col, width=2 if t.get("hidden") else 4); a_ = math.atan2(Q[-1][1] - Q[-2][1], Q[-1][0] - Q[-2][0]); q1 = Q[-1]
                d.polygon([q1, (q1[0] - 10 * math.cos(a_ - 0.4), q1[1] - 10 * math.sin(a_ - 0.4)), (q1[0] - 10 * math.cos(a_ + 0.4), q1[1] - 10 * math.sin(a_ + 0.4))], fill=col)
            else:
                q = T(t["p"]); r_ = 5 if t["kind"] in ("door", "gate") else 3
                d.ellipse([q[0] - r_, q[1] - r_, q[0] + r_, q[1] + r_], fill=(210, 40, 40) if t["kind"] in ("door", "gate") else (230, 130, 30))
        if route:                                                                # path: through the passages themselves (a door, an arch, a flight — along the polyline)
            pts = []
            for cid, ti in route:
                if ti is None: continue
                t = self.trans[ti]; Q = t["p"] if isinstance(t["p"][0], list) else [t["p"]]
                if isinstance(t["p"][0], list) and t["b"] != cid: Q = Q[::-1]        # the flight is traversed from top to bottom
                pts += [T(q) for q in Q]
            if len(pts) > 1: d.line(pts, fill=(220, 0, 160), width=4)
            for q in pts[:1] + pts[-1:]: d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=(220, 0, 160), width=3)
        for c in cs:
            if area(c["poly"]) * sc * sc < 250: continue
            q = T(inner_point(c["poly"], holes=c.get("holes"))); dep = depth.get(c["id"])
            d.text((q[0] - 5, q[1] - 8), "×" if dep is None and c["walk"] else (str(dep) if dep is not None else "·"), fill=(20, 20, 20), font=F)
        for a in self.anchors:
            q = T(a["p"]); d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=(0, 0, 0)); d.text((q[0] + 6, q[1] - 7), a["name"], fill=(0, 0, 0), font=F)
        d.text((10, 10), title, fill="black", font=F2)
        x0 = W_ + 20; y = 70
        for k in CLASSES:
            if k == "void": continue
            d.rectangle([x0, y, x0 + 24, y + 16], fill=COLORS[k], outline=(120, 120, 120)); d.text((x0 + 32, y), CLASS_RU[k], fill="black", font=F); y += 24
        for txt, col, kind in (("wall", (30, 30, 30), "l3"), ("arcade", (90, 90, 90), "l2"), ("parapet, railings", (120, 80, 40), "l2"), ("cliff", (190, 60, 60), "l1"),
                               ("door, gate", (210, 40, 40), "o"), ("arch, opening", (230, 130, 30), "o"), ("stair, ramp", (40, 90, 210), "l4"),
                               ("stair in the wall thickness", (120, 150, 220), "l2"), ("bridge", (140, 60, 170), "l4"), ("window", (40, 170, 220), "s"),
                               ("anchor", (0, 0, 0), "o"), ("path", (220, 0, 160), "l4")):
            if kind == "o": d.ellipse([x0 + 6, y + 3, x0 + 16, y + 13], fill=col)
            elif kind == "s": d.rectangle([x0 + 8, y + 5, x0 + 14, y + 11], fill=col)
            else: d.line([(x0, y + 8), (x0 + 24, y + 8)], fill=col, width=int(kind[1]))
            d.text((x0 + 32, y), txt, fill="black", font=F); y += 24
        for line in ("number — depth: transitions from the world", "higher — darker; voids (attics,", "tower shafts) are not drawn"):
            d.text((x0, y + 8), line, fill=(60, 60, 60), font=F); y += 20
        im.save(path)


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:] if "=" in x); R = json.load(open(a[0]))
    S = Space.from_dict(R["spaces"]); D = S.to_dict(); route = None
    res = {"stats": D["stats"], "non_passing": [c["what"] for c in D["checks"] if not c["ok"]]}
    if kw.get("path"):
        f, t = kw["path"].split(","); route = S.path(f, t)
        res["path"] = None if route is None else [f"{S.trans[i]['kind']}→{c}" if i is not None else c for c, i in route]
    if kw.get("draw"): S.draw(kw["draw"], title=os.path.basename(os.path.dirname(os.path.abspath(a[0]))), route=route)
    print("__RESULT__", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
