"""Walkable surfaces and connectivity: knowing by calculation, before the model, where one can get to — for checks and for
the neighbors (the take, the crowd, a camera by tiers). The project's Python only.

Nodes: ground — the cells of the top layer that are passable by slope (neighboring cells with a level change ≤ climb merge;
houses are cut out where they stand two meters above the ground: a passage under a house on old walls is walkable),
decks (slabs, tiers, bridges, sidewalks; they merge where they touch at one elevation), building storeys, terrace roofs.
Links: a door (a storey ↔ the surface at the door, a porch too), a relate bridge (a storey ↔ the surface at the other end),
a stair inside a building (storey ↔ storey), an entrance from inside (via: an upper step ↔ the lower one), a deck ↔ ground
(the deck edge at its elevation), explicit world links W.links (a lift — by elevations, a ramp, a gate). The result —
connected parts, the share of the biggest, what is unreachable from the start (W.origin — a point of ground or deck)."""
import json
import math

import numpy as np
import shapely
from shapely.geometry import Point


WALK, CLIMB = 1.3, 0.4                                                       # walking on the level and climbing a stair, m/s


def link_cost(kind, a, b):
    """The cost of a link in seconds (for a route): a lift — waiting and the ride, a funicular — waiting and the trip, a stair —
    a climb slower than walking, a ramp — walking. a, b — the ends (x, y, z)."""
    h = float(np.hypot(b[0] - a[0], b[1] - a[1])); dz = abs(b[2] - a[2])
    if kind == "lift": return 10.0 + dz / 1.5
    if kind == "tunnel_lift": return h / WALK + 30.0 + dz / 1.5
    if kind == "funicular": return 45.0 + float(np.hypot(h, dz)) / 3.0
    if kind == "stair": return h / 1.0 + dz / CLIMB
    if kind == "ramp": return float(np.hypot(h, dz)) / 1.1
    return h / WALK + 5.0


class DSU:
    """Connected parts; edges — edges with the link points and a cost in seconds (for a route): (u, v, point_u, point_v, cost)."""
    def __init__(self): self.p = {}; self.edges = []

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b, pa=None, pb=None, cost=1.0):
        ra, rb = self.find(a), self.find(b); self.p[ra] = rb if ra != rb else rb; self.edges.append((a, b, pa, pb, cost))


def graph(W, step=1.0, climb=0.6):
    """→ (DSU, nodes with descriptions, the labeling of ground cells). A node is a string: g:<i>,<j> (a cell), d:<k> (a deck),
    f:<house>:<storey>."""
    D = DSU(); info = {}
    X = np.arange(W.block[0] + step / 2, W.block[2], step); Y = np.arange(W.block[1] + step / 2, W.block[3], step)
    Xg, Yg = np.meshgrid(X, Y); Z = np.asarray(W.layer(-1)(Xg, Yg), float)
    free = np.ones_like(Z, bool)                                             # a cell is occupied where a building stands within [ground, ground + 2 m]:
    for b in W.buildings:                                                    # what was cut down underground and a house above a passage do not obstruct walking
        if b["state"] not in ("whole", "raised", "cut"): continue
        top = b["z_cut"] if b["z_cut"] is not None else b["z_top"]
        bx = b["poly"].bounds; m = (Xg >= bx[0]) & (Xg <= bx[2]) & (Yg >= bx[1]) & (Yg <= bx[3])
        if not m.any(): continue
        ins = np.zeros_like(m); ins[m] = shapely.contains_xy(b["poly"], Xg[m], Yg[m])
        free &= ~(ins & (b["z_found"] < Z + 2.0) & (top > Z + 0.3))
    for j in range(len(Y)):                                                  # ground: neighbors with a level change ≤ climb
        for i in range(len(X)):
            if not free[j, i]: continue
            n = f"g:{i},{j}"; D.find(n)
            if i + 1 < len(X) and free[j, i + 1] and abs(Z[j, i + 1] - Z[j, i]) <= climb:
                D.union(n, f"g:{i + 1},{j}", cost=step / WALK + abs(Z[j, i + 1] - Z[j, i]) / 0.5)
            if j + 1 < len(Y) and free[j + 1, i] and abs(Z[j + 1, i] - Z[j, i]) <= climb:
                D.union(n, f"g:{i},{j + 1}", cost=step / WALK + abs(Z[j + 1, i] - Z[j, i]) / 0.5)
    grid = (X, Y, Z, free, step)

    def at(x, y, z, tol=0.5, ground_only=False):
        """The surface node at a point at elevation z: a deck, then a ground cell."""
        p = Point(x, y)
        for k, d in enumerate(W.decks if not ground_only else []):
            if abs(d["z"] - z) <= tol and d["poly"].buffer(0.3).contains(p): return f"d:{k}"
        for b in (W.buildings if not ground_only else []):                     # the terrace roof of another building
            if b.get("roof") in ("terrace", "street") and b["z_cut"] is None and abs(b["z_top"] - z) <= tol and b["poly"].contains(p):
                return f"r:{b['id']}"
        i = int((x - X[0]) / step + 0.5); j = int((y - Y[0]) / step + 0.5)
        for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < len(X) and 0 <= jj < len(Y) and free[jj, ii] and abs(Z[jj, ii] - z) <= tol + 0.8: return f"g:{ii},{jj}"
        return None
    for k, d in enumerate(W.decks):                                          # decks: those touching at one elevation are one
        D.find(f"d:{k}"); info[f"d:{k}"] = d.get("name", "deck")
        for m in range(k):
            e = W.decks[m]
            if abs(e["z"] - d["z"]) <= 0.5 and e["poly"].buffer(0.2).intersects(d["poly"]): D.union(f"d:{k}", f"d:{m}")
        P = d["poly"]                                                         # a deck edge at ground elevation → ground
        for geom in getattr(P, "geoms", [P]):
            ring = geom.exterior
            for t in np.arange(0.0, ring.length, 1.5):                            # along the whole edge every 1.5 m
                q = ring.interpolate(t)
                n = at(q.x, q.y, d["z"], tol=0.4, ground_only=True)
                if n: D.union(f"d:{k}", n)
    for b in W.buildings:                                                    # storeys: the inner stair, doors, bridges, via
        if b["state"] not in ("whole", "raised") or "open" not in b: continue  # the link was not computed — no verdict
        top = b["z_cut"] if b["z_cut"] is not None else b["z_top"]
        fl = [fi for fi, f in enumerate(b["floors"]) if f["z"] < top - 0.5]
        for fi in fl: D.find(f"f:{b['id']}:{fi}"); info[f"f:{b['id']}:{fi}"] = f"house {b['id']} storey {fi}"
        for a, c in zip(fl[:-1], fl[1:]):                                    # house stair: the climb to the next storey
            D.union(f"f:{b['id']}:{a}", f"f:{b['id']}:{c}", cost=(b["floors"][c]["z"] - b["floors"][a]["z"]) / CLIMB)
        if b.get("via") is not None:
            o = next((q for q in W.buildings if q["id"] == b["via"]), None)
            if o is not None and fl: D.union(f"f:{b['id']}:{fl[0]}", f"f:{o['id']}:{len(o['floors']) - 1}", cost=5.0)
        for dd in b.get("doors", []):
            if dd["side"] == "party": continue                                 # a door into a neighbor — via a W.links link
            from components.buildings.levels.strata import side_line
            a_, c_, nrm = side_line(b, dd["side"]); a_ = np.asarray(a_, float); c_ = np.asarray(c_, float); nrm = np.asarray(nrm, float)
            dv = (c_ - a_) / max(np.linalg.norm(c_ - a_), 1e-9); p = a_ + dv * dd["u"] + nrm * (b["t"] + 0.6)
            z = b["floors"][dd["floor"]]["z"] - dd.get("stoop", 0.0)
            n = at(p[0], p[1], z, tol=0.6 if not dd.get("stoop") else 0.8)
            if n:
                pin = p - nrm * (b["t"] + 1.2)
                D.union(f"f:{b['id']}:{dd['floor']}", n, (pin[0], pin[1], b["floors"][dd["floor"]]["z"]), (p[0], p[1], z),
                        cost=2.0 + dd.get("stoop", 0.0) / CLIMB)
        for br in b.get("bridges", []):
            n = at(br["b"][0], br["b"][1], br["z"])
            fz = min(fl, key=lambda fi: abs(b["floors"][fi]["z"] - br["z"])) if fl else None
            if n and fz is not None:
                D.union(f"f:{b['id']}:{fz}", n, tuple(br["a"]) + (br["z"],), tuple(br["b"]) + (br["z"],),
                        cost=float(np.hypot(br["b"][0] - br["a"][0], br["b"][1] - br["a"][1])) / WALK + 2.0)
        if b.get("roof") in ("terrace", "street") and b["z_cut"] is None:      # a terrace roof is walkable, linked to the top storey
            c = b["poly"].centroid; n = f"r:{b['id']}"; D.find(n); info[n] = f"roof {b['id']}"
            if fl: D.union(n, f"f:{b['id']}:{fl[-1]}", cost=8.0)
            for q in np.asarray(b["poly"].buffer(0.8).exterior.coords)[::2]:
                m = at(q[0], q[1], b["z_top"], tol=0.5)
                if m: D.union(n, m)
    for L in getattr(W, "links", []):                                        # lifts, ramps, gates, doors into a neighbor
        if L["kind"] == "party":
            (ba, fa), (bb, fb) = L["floors"]
            if f"f:{ba}:{fa}" in D.p and f"f:{bb}:{fb}" in D.p: D.union(f"f:{ba}:{fa}", f"f:{bb}:{fb}", cost=2.0)
            continue
        pts = [(L["x"], L["y"], z) for z in L["z"]] if L["kind"] == "lift" else [tuple(L["a"]), tuple(L["b"])]
        ns = [(at(x, y, z, tol=0.8), (x, y, z)) for x, y, z in pts]; ns = [t for t in ns if t[0]]
        for (a, pa), (c, pc) in zip(ns[:-1], ns[1:]): D.union(a, c, pa, pc, cost=link_cost(L["kind"], pa, pc))
    return D, info, grid


def report(W, step=1.0):
    """Connected parts of the walkable surface: how many, the share of the big one, the storeys and decks unreachable from the
    start W.origin (or from the big one)."""
    D, info, grid = graph(W, step)
    keys = list(D.p.keys()); roots = {}
    for k in keys: roots.setdefault(D.find(k), []).append(k)
    comps = sorted(roots.values(), key=len, reverse=True)
    weight = lambda c: sum(1 for k in c if not k.startswith("g:")) + 0.02 * sum(1 for k in c if k.startswith("g:"))
    comps = sorted(comps, key=weight, reverse=True)
    o = getattr(W, "origin", None)
    main = None
    if o is not None:
        r = None
        X, Y, Z, free, st = grid
        i = int((o[0] - X[0]) / st + 0.5); j = int((o[1] - Y[0]) / st + 0.5)
        r = D.find(f"g:{i},{j}") if f"g:{i},{j}" in D.p else None
        main = next((c for c in comps if r and D.find(c[0]) == r), None)
    main = main or comps[0]
    floors = [k for k in keys if k.startswith("f:")]; decks = [k for k in keys if k.startswith("d:")]
    mr = D.find(main[0])
    lost_f = [info[k] for k in floors if D.find(k) != mr]
    lost_d = sorted({info[k] for k in decks if D.find(k) != mr})
    n_big = sum(1 for c in comps if any(not k.startswith("g:") for k in c))     # parts with buildings or decks
    islands = sum(1 for c in comps if all(k.startswith("g:") for k in c) and len(c) >= 4)
    lost_k = [k for k in keys if (k.startswith("d:") or k.startswith("f:")) and D.find(k) != mr]
    out = {"n_parts": n_big, "islets_ground": islands, "n_storeys": len(floors), "unreachable_n_storeys": len(lost_f), "decks": len(decks),
           "unreachable_decks": lost_d, "examples": lost_f[:8], "nodes": lost_k[:12]}
    return out, D, info, grid


def draw(W, D, grid, path, scale=4):
    """A connectivity plan: ground cells and decks colored by their connected part (the main one is gray), storeys as dots by
    height."""
    from PIL import Image, ImageDraw
    X, Y, Z, free, st = grid
    Wd, Ht = int((W.block[2] - W.block[0]) * scale), int((W.block[3] - W.block[1]) * scale)
    im = Image.new("RGB", (Wd, Ht), (250, 250, 248)); dr = ImageDraw.Draw(im)
    roots = {}
    for k in D.p: roots.setdefault(D.find(k), 0); roots[D.find(k)] += 1
    big = max(roots, key=roots.get)
    pal = [(214, 96, 77), (67, 147, 195), (90, 174, 97), (153, 112, 171), (241, 163, 64), (140, 81, 10), (1, 102, 94)]
    col = {}
    f = lambda x, y: ((x - W.block[0]) * scale, Ht - (y - W.block[1]) * scale)
    for j in range(len(Y)):
        for i in range(len(X)):
            if not free[j, i]: continue
            r = D.find(f"g:{i},{j}")
            c = (200, 200, 196) if r == big else col.setdefault(r, pal[len(col) % len(pal)])
            x0, y0 = f(X[i] - st / 2, Y[j] + st / 2); dr.rectangle([x0, y0, x0 + st * scale, y0 + st * scale], fill=c)
    for k, d in enumerate(W.decks):
        r = D.find(f"d:{k}"); c = (120, 120, 116) if r == big else col.setdefault(r, pal[len(col) % len(pal)])
        for g in getattr(d["poly"], "geoms", [d["poly"]]):
            dr.polygon([f(*q) for q in np.asarray(g.exterior.coords)], outline=c)
    for b in W.buildings:
        if b["state"] not in ("whole", "raised"): continue
        ok = any(D.find(f"f:{b['id']}:{fi}") == big for fi in range(len(b["floors"])) if f"f:{b['id']}:{fi}" in D.p)
        dr.polygon([f(*q) for q in np.asarray(b["poly"].exterior.coords)], outline=(40, 40, 40) if ok else (220, 30, 30),
                   width=1 if ok else 3)
    im.save(path); return path


def prune(W, step=1.0):
    """The unreachable is not built: remove the buildings none of whose storeys is linked to the main part, and the bridge decks
    not linked to it (a bridge to nowhere). → (how many buildings, how many decks were removed)."""
    rep, D, info, grid = report(W, step)
    roots = {}
    for k in D.p: roots.setdefault(D.find(k), []).append(k)
    o = getattr(W, "origin", None); mr = None
    if o is not None:
        X, Y, Z, free, st = grid; i = int((o[0] - X[0]) / st + 0.5); j = int((o[1] - Y[0]) / st + 0.5)
        mr = D.find(f"g:{i},{j}") if f"g:{i},{j}" in D.p else None
    if mr is None: mr = D.find(max(roots.values(), key=lambda c: sum(1 for k in c if not k.startswith("g:")))[0])
    gone_b = [b for b in W.buildings if "open" in b and b["state"] in ("whole", "raised") and
              not any(D.find(f"f:{b['id']}:{fi}") == mr for fi in range(len(b["floors"])) if f"f:{b['id']}:{fi}" in D.p)]
    gone_d = [k for k, d in enumerate(W.decks) if f"d:{k}" in D.p and D.find(f"d:{k}") != mr and d.get("prunable", True)]
    for b in gone_b: b["state"] = "gone"
    W.decks = [d for k, d in enumerate(W.decks) if k not in set(gone_d)]
    if gone_b or gone_d: W.notes.append(f"unreachable removed: buildings {len(gone_b)}, decks {len(gone_d)}")
    return len(gone_b), len(gone_d)


def route(W, src, dst, step=1.0, cost=False):
    """The fastest path by edge cost (seconds: walking, stairs, lifts, funicular — link_cost) from the ground point src (x, y) to
    the node dst (for example "f:<house>:<storey>" or "g:<i>,<j>") → [(x, y, z)] — the route line (the link points; ground
    cells — their centers); cost=True → (points, seconds). Walking inside a deck or a storey is not counted (a node)."""
    import heapq
    D, info, grid = graph(W, step)
    X, Y, Z, free, st = grid
    i = int((src[0] - X[0]) / st + 0.5); j = int((src[1] - Y[0]) / st + 0.5); s0 = f"g:{i},{j}"
    adj = {}
    for u, v, pu, pv, c in D.edges: adj.setdefault(u, []).append((v, pu, pv, c)); adj.setdefault(v, []).append((u, pv, pu, c))
    if s0 not in adj:                                                          # the start — the nearest cell that has links
        cand = [n for n in adj if n.startswith("g:")]
        s0 = min(cand, key=lambda n: (int(n[2:].split(",")[0]) - i) ** 2 + (int(n[2:].split(",")[1]) - j) ** 2 +
                 (0 if any(not m.startswith("g:") for m, *_ in adj[n]) else 4))
    dist = {s0: 0.0}; prev = {s0: None}; pq = [(0.0, s0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == dst: break
        for v, pu, pv, c in adj.get(u, []):
            if d + c < dist.get(v, float("inf")): dist[v] = d + c; prev[v] = (u, pu, pv); heapq.heappush(pq, (d + c, v))
    if dst not in prev: return ([], None) if cost else []
    cell = lambda n: (float(X[int(n[2:].split(",")[0])]), float(Y[int(n[2:].split(",")[1])]),
                      float(Z[int(n[2:].split(",")[1]), int(n[2:].split(",")[0])]))
    pts = []; n = dst
    while prev[n] is not None:
        u, pu, pv = prev[n]
        if pv is not None: pts.append(pv)
        elif n.startswith("g:"): pts.append(cell(n))
        if pu is not None: pts.append(pu)
        elif u.startswith("g:"): pts.append(cell(u))
        n = u
    return (pts[::-1], float(dist[dst])) if cost else pts[::-1]
