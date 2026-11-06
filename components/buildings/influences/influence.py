"""Influences — part of the "buildings" core (knowledge — influences.yaml). There are exactly three kinds of influence:
administration/force, trade/production, wealth/poverty (center–periphery); all the variety of districts follows from these
three; a huge town has a main center and a few small ones, placed at points; the mechanism reduces to calculations.
  power P      — centers: the main one (a landmark — acropolis, castle; otherwise at the main square, or on high ground near
                 the middle; the `center` knob) and small ones: their number follows the fabric area ⌊ha × per_ha + chance⌋
                 ≤ max, the place is a crossing (higher and more passable — likelier); P = max s·exp(−(d/r)²);
  trade T      — flow along the streets: shortest paths (length × rank cost) entrance ↔ main center, entrance ↔ entrance,
                 center ↔ squares, residents → main center (Hillier 1993: trade settles on through streets); an edge's
                 T = √(flow / q98); at squares, entrances, bridges and water — nodes s·exp(−(d/r)²); production — the same
                 trade where it is poorer;
  wealth W     — center · closeness to the center (1 − (d/D)²: the share of area farther away, D — the reach of the fabric)
                 + place · (higher up the slope; on level ground — the windward side, Heblich 2021), not lower than
                 at_power · P; in a hive — also the tier (the top is richer).
Settlement knobs (levels 0–1, 0.5 — as computed): power (number and radius of centers), trade (flow amplitude), wealth
(shift), inequality (amplitude of the deviation from the mean); wind, center. Plan corrections — one formula for all
archetypes: plot facade and depth, gardens, shops, storeys by the place value V = max(P, T, W), house kind by rank, tone,
cheap features (half-timbering, jetties…).
A district (power, trade, craft, nobility, middle, poor) is only a reading of the three numbers (field.label).
Input — the settlement geometry (street graph, squares, landmark, boundary, water, tiers); output — a model (fields at any
point, flow on the edges, centers, corrections) and the raster `influences.npz` (field.py — no shapely, for Blender and the
neighbors). It does not call the types and ensembles."""
import os
import sys
from components.buildings import arch_parts
import heapq
import math

import numpy as np
import shapely
import yaml
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from components.buildings.influences import field as FD
from components.buildings.influences.field import COLORS, LABELS  # noqa: F401 — district reading and colors (the ensemble map)

HERE = os.path.dirname(os.path.abspath(__file__))
_K = None
SKIP = ("edge", "wall", "road")                                              # not settlement streets: the edge of the place, the wall, the road beyond the wall


def knowledge():
    """Knowledge of the influences (influences.yaml)."""
    global _K
    if _K is None: _K = yaml.safe_load(open(os.path.join(HERE, "influences.yaml"), encoding="utf-8"))
    return _K


def levels(**kw):
    """Settlement knobs: power, trade, wealth, inequality (0–1; absent — 0.5)."""
    en = {"power": "power", "trade": "trade", "wealth": "wealth", "inequality": "inequality"}
    L = {k: 0.5 for k in en.values()}
    for k, v in kw.items():
        k = en.get(k, k)
        if k in L and v is not None: L[k] = float(np.clip(float(v), 0.0, 1.0))
    return L


def _lerp(r, t): return float(r[0]) + (float(r[1]) - float(r[0])) * t


def _kern(d, s, r): return s * np.exp(-(np.asarray(d, float) / max(float(r), 1e-6)) ** 2)


def _pick(wd, rng):
    ks = [k for k in wd if float(wd[k]) > 0]; w = np.array([float(wd[k]) for k in ks]); return ks[int(rng.choice(len(ks), p=w / w.sum()))]


# ---------------------------------------------------------------- house corrections from the three numbers
def shops(T, K=None):
    a, b = (K or knowledge())["corrections"]["shops"]; return float(min(1.0, a + b * T))


def tone(W, K=None):
    return round(float(2 * (K or knowledge())["corrections"]["tone"] * (W - 0.5)), 3)


def place(P, T, W, floors=None, K=None):
    """The house's context from the plot's three numbers: the place value V = max(P, T, W), tone, shops, storeys (between the edge
    and the center of the archetype, by V), the numbers themselves, and the palette: w — wealth between the thresholds of the
    poor and the nobility (0–1), the spread (the house kind takes its color from the houses.yaml list `by_wealth`: the poor —
    the first, the nobility — the last)."""
    K = K or knowledge(); V = max(P, T, W); L = K["districts"]
    w_ = float(np.clip((W - L["poor"]) / max(L["nobility"] - L["poor"], 1e-6), 0.0, 1.0))   # palette — from the poor to the nobility
    ctx = {"center": V, "tone": tone(W, K), "shops": shops(T, K), "influence": [round(P, 3), round(T, 3), round(W, 3)],
           "palette": {"w": round(w_, 3), "spread": float(K["corrections"]["palette"])}}
    if floors and "center" in floors:
        lo = floors["edge"][0] + (floors["center"][0] - floors["edge"][0]) * V; hi = floors["edge"][1] + (floors["center"][1] - floors["edge"][1]) * V
        ctx["storeys"] = [int(round(lo)), max(int(round(lo)), int(round(hi)))]
    return ctx


def cheap(KH_k, W, K=None):
    """Cheap features are more common among the poor and rarer among the rich (stone instead of wood, no accretions): the shares
    of the kind's features from `cheap` (houses.yaml; by default half-timbering and jetties) × (1 + k·(1 − 2W)). → the kind's
    corrections (a nested key replacement)."""
    k = float((K or knowledge())["corrections"]["wood"]); f = 1 + k * (1 - 2 * W); out = {}
    for name in KH_k.get("cheap", ("fachwerk", "reach")):
        v = KH_k.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v: out[name] = round(float(np.clip(v * f, 0.0, 1.0)), 3)
        elif isinstance(v, dict) and isinstance(v.get("frac"), (int, float)): out[name] = {"frac": round(float(np.clip(v["frac"] * f, 0.0, 1.0)), 3)}
    return out


# ---------------------------------------------------------------- trade: flow along the streets
def flows(P_, E, K, entries, main, others):
    """Flow along the edges E: shortest paths (length × rank cost) entrance ↔ main center, entrance ↔ entrance, main ↔ squares,
    and residents (all nodes) → the main center; the residents' mass = the paths' mass × `residents`. → an np.array over E."""
    n = len(P_); adj = [[] for _ in range(n)]
    for i, e in enumerate(E):
        if e["kind"] in SKIP: continue
        L = LineString(e["pts"]).length * float(K["cost"].get(e["rank"], 1.0))
        adj[e["a"]].append((e["b"], L, i)); adj[e["b"]].append((e["a"], L, i))
    trees = {}
    def tree(s):
        if s in trees: return trees[s]
        dist = np.full(n, np.inf); pe = np.full(n, -1); pn = np.full(n, -1); dist[s] = 0.0; h = [(0.0, s)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist[u]: continue
            for v, L, i in adj[u]:
                if d + L < dist[v]: dist[v] = d + L; pe[v] = i; pn[v] = u; heapq.heappush(h, (d + L, v))
        trees[s] = (dist, pe, pn); return trees[s]
    f = np.zeros(len(E))
    def add(s, t, w):
        dist, pe, pn = tree(s)
        if s == t or not np.isfinite(dist[t]): return 0.0
        v = t
        while v != s and pe[v] >= 0: f[pe[v]] += w; v = pn[v]
        return w
    Wp = K["pairs"]; mass = 0.0
    for i, a in enumerate(entries):
        if main is not None: mass += add(main, a, Wp["entrance_center"])
        for b in entries[i + 1:]: mass += add(a, b, Wp["entrance_entrance"])
    for o in others:
        if main is not None: mass += add(main, o, Wp["center_square"])
    if main is not None:
        dist = tree(main)[0]; nodes = [v for v in range(n) if adj[v] and np.isfinite(dist[v]) and v != main]
        w = float(Wp["residents"]) * max(mass, 1.0) / max(len(nodes), 1)
        for t in nodes: add(main, t, w)
    return f


def _entries(P_, E, pts, sep=15.0):
    """Entrances — network nodes at the gates and the road starts (ensemble points), within 25 m; those closer than sep to each
    other count as one."""
    used = sorted({v for e in E if e["kind"] not in SKIP for v in (e["a"], e["b"])})
    if not used: return []
    X = np.asarray([P_[i] for i in used], float); keep = []
    for g in pts or []:
        d = np.linalg.norm(X - np.asarray(g, float), axis=1); k = int(np.argmin(d))
        if d[k] < 25.0 and all(np.linalg.norm(X[k] - P_[j]) >= sep for j in keep): keep.append(used[k])
    return keep


def _nearest_node(P_, E, p):
    used = {e["a"] for e in E if e["kind"] not in SKIP} | {e["b"] for e in E if e["kind"] not in SKIP}
    if not used: return None
    ids = sorted(used); X = np.asarray([P_[i] for i in ids], float)
    return ids[int(np.argmin(np.linalg.norm(X - np.asarray(p, float), axis=1)))]


# ---------------------------------------------------------------- model
class Influence:
    """The three fields and the plan corrections. Points — an (N, 2) array; deck — the hive tier (None — without tiers)."""

    def __init__(self, K, height, centres, kern_T, wc, Rw, zq, s_h, up, n_decks, E, L=None):
        self.K = K; self.height = height; self.centres = centres; self.kern_T = kern_T; self.wc = wc; self.Rw = Rw
        self.zq = zq; self.s_h = s_h; self.up = up; self.n_decks = n_decks; self.L = L or levels(); self.Wm = 0.5
        self._lines = [(LineString(e["pts"]), e.get("flow", 0.0), float(e.get("w", 0.0))) for e in E if e["kind"] not in SKIP]
        self._tree = shapely.STRtree([g for g, _, _ in self._lines]) if self._lines else None

    # -- fields
    def P(self, X, deck=None):
        X = np.atleast_2d(np.asarray(X, float)); v = np.zeros(len(X))
        for ce in self.centres:
            if ce.get("deck") != deck: continue
            v = np.maximum(v, _kern(np.linalg.norm(X - np.asarray(ce["p"]), axis=1), ce["s"], ce["r"]))
        return v

    def Tk(self, X):
        X = np.atleast_2d(np.asarray(X, float)); v = np.zeros(len(X)); pts = shapely.points(X)
        for g, s, r in self.kern_T: v = np.maximum(v, _kern(shapely.distance(g, pts), s, r))
        return np.minimum(v, 1.0)

    def W0(self, X):
        """Wealth without the knobs, power and tier: center — outskirts, and place (slope or windward side)."""
        X = np.atleast_2d(np.asarray(X, float)); B = self.K["wealth"]
        d = shapely.distance(self.wc, shapely.points(X)); C = np.clip(1.0 - (d / self.Rw) ** 2, 0.0, 1.0)   # the share of area farther than d
        z = np.asarray(self.height(X[:, 0], X[:, 1]), float)
        H = np.clip((z - self.zq[0]) / max(self.zq[1] - self.zq[0], 1e-6), 0.0, 1.0)
        c = np.asarray(self.wc.centroid.coords[0]); dv = X - c; r = np.linalg.norm(dv, axis=1)
        sec = 0.5 + 0.5 * (dv @ self.up) / np.maximum(r, 1e-6) * np.minimum(1.0, r / self.Rw)
        return np.clip(B["center"] * C + B["place"] * (self.s_h * H + (1 - self.s_h) * sec), 0.0, 1.0)

    def W(self, X, deck=None, P=None):
        X = np.atleast_2d(np.asarray(X, float)); B = self.K["wealth"]; Wv = self.W0(X)
        b, u = self.L["wealth"], self.L["inequality"]
        if b != 0.5 or u != 0.5:                                                   # knobs: the whole town richer or poorer, the difference sharper or smoother
            U = self.K["levels"]
            Wv = np.clip(self.Wm + (Wv - self.Wm) * _lerp(U["inequality"]["amplitude"], u) + (b - 0.5) * float(U["wealth"]["shift"]), 0.0, 1.0)
        Wv = np.maximum(Wv, B["at_power"] * (self.P(X, deck) if P is None else P))
        if deck is not None and self.n_decks > 1:
            Wv = np.clip((1 - B["tier"]) * Wv + B["tier"] * deck / (self.n_decks - 1), 0.0, 1.0)
        return Wv

    def T_line(self, g):
        """The flow of the old edge for a street piece (the edges were rebuilt: slum passages, hive cuts)."""
        if self._tree is None: return 0.0
        m = g.interpolate(0.5, normalized=True); best = (2.0, 0.0)
        for k in self._tree.query(m.buffer(2.0)):
            d = self._lines[k][0].distance(m)
            if d < best[0]: best = (d, self._lines[k][1])
        return best[1]

    def T_block(self, poly, reach=8.0):
        """Trade of a block: the largest street flow along its edge and the nodes near the middle."""
        t = float(self.Tk([poly.representative_point().coords[0]])[0])
        if self._tree is None: return t
        for k in self._tree.query(poly.buffer(reach)):
            g, f, _ = self._lines[k]
            if g.distance(poly) < reach: t = max(t, f)
        return t

    def T_field(self, X):
        """Trade at any point (raster, neighbors): the nodes and the flow of nearby streets, × exp(−(d/field)²), d — from the street edge."""
        X = np.atleast_2d(np.asarray(X, float)); v = self.Tk(X)
        if self._tree is None or not len(X): return v
        r = float(self.K["trade"]["field"]); pts = shapely.points(X)
        ip, ig = self._tree.query(pts, predicate="dwithin", distance=3.0 * r + 12.0)
        if len(ip):
            geoms = np.array([g for g, _, _ in self._lines], dtype=object); f = np.array([t for _, t, _ in self._lines]); w = np.array([w_ for _, _, w_ in self._lines])
            d = np.maximum(shapely.distance(pts[ip], geoms[ig]) - w[ig] / 2.0, 0.0)
            np.maximum.at(v, ip, f[ig] * np.exp(-(d / r) ** 2))
        return v

    def at(self, X, T_front=None, deck=None):
        """(P, T, W) at points; T is the largest of the facade flow and the nodes."""
        P = self.P(X, deck); T = self.Tk(X)
        if T_front is not None: T = np.maximum(T, np.asarray(T_front, float))
        return P, T, self.W(X, deck, P)

    # -- plan corrections (one formula for all archetypes)
    def cut(self, X, T_edge):
        """Subdivision corrections at a point by the street: (× facade, × depth, + share of empty ones)."""
        P, T, W = self.at([X], [T_edge]); return self.cut_pw(float(W[0]), float(T[0]))

    def cut_pw(self, W, T):
        M = self.K["corrections"]; f = M["facade"]; d = M["depth"]; s = M["gardens"]
        fm = (f[0] + f[1] * W) * (1 + f[2] * T); dm = d[0] + d[1] * W + d[2] * T * (1 - W)
        em = s[1] * max(0.0, W - s[0]) / (1 - s[0]) * (1 - T)
        return fm, dm, em

    def shops(self, T): return shops(T, self.K)

    def halves(self, W, T, base):
        """A whole block (castrum): the share halved = the archetype's share + a·(0.5 − W) + b·T (splitting the poor into 3–4 is
        left to the caller)."""
        a, b = self.K["corrections"]["division"]; return float(np.clip(base + a * (0.5 - W) + b * T, 0.0, 0.95))

    def pieces_exp(self, W, T):
        """The expected number of hub-block parts by the formula (the mean over the spread ± d) — the yardstick for the plan check."""
        a, b, c, d = self.K["corrections"]["blocks"]
        return float(np.mean([2 ** int(np.clip(round(a * (1 - W) + b * T + c + x), 0, 2)) for x in np.linspace(-d, d, 7)]))

    def pieces(self, W, T, rng):
        """Hub blocks: a block into 2^m parts, m = a·(1 − W) + b·T + c ± d, 0–2: richer — whole, poorer and on a busy street — 2 and 4."""
        a, b, c, d = self.K["corrections"]["blocks"]; return 2 ** int(np.clip(round(a * (1 - W) + b * T + c + rng.uniform(-d, d)), 0, 2))

    def tone(self, W): return tone(W, self.K)

    def place(self, P, T, W, floors=None):
        return place(P, T, W, floors, self.K)

    def build(self, KH_k, W):
        return cheap(KH_k, W, self.K)

    def variant(self, weights, KH, W, T, rng):
        """House kind: the archetype's weight × the closeness of the kind's rank to the wealth × (1 + k · T · the kind's trade)."""
        M = self.K["corrections"]; names = list(weights)
        w = np.array([float(weights[k]) * math.exp(-((W - float(KH[k].get("rank", 0.5))) / M["rank"]) ** 2) *
                      (1 + M["trade_view"] * T * float(KH[k].get("trade", 0.0))) for k in names])
        return names[int(rng.choice(len(names), p=w / w.sum()))] if w.sum() > 0 else names[0]

    def label(self, P, T, W):
        return FD.label(P, T, W, self.K["districts"])

    def promote(self, E):
        """Trade raises the rank: a lane with the flow at or above the threshold becomes a street (shops, the house's main facade; the
        width stays). Call after the plot subdivision and the graph rebuild (edges come from the street lines, the rank is the
        original): it does not shift the layout. → the number promoted."""
        U = self.K["trade"].get("rank"); n = 0
        if not U: return 0
        for e in E:
            if e["kind"] not in SKIP and e["rank"] in U["of"] and float(e.get("flow", 0.0)) >= float(U["threshold"]):
                e["rank_was"] = e["rank"]; e["rank"] = U["to"]; n += 1
        return n

    # -- raster for the neighbors and for the build
    def raster(self, path, c, half, step=2.0, z0=0.0, dz=0.0):
        """The fields on a grid → influences.npz (field.py): P and W per tier (hive), T — a single layer."""
        n = int(math.ceil(2 * half / step)); x0 = c[0] - half + step / 2; y0 = c[1] - half + step / 2
        X, Y = np.meshgrid(x0 + np.arange(n) * step, y0 + np.arange(n) * step); pts = np.c_[X.ravel(), Y.ravel()]
        dks = list(range(self.n_decks)) if self.n_decks > 1 else [None]
        Ps = []; Ws = []
        for dk in dks:
            P = self.P(pts, dk); Ps.append(P.reshape(n, n)); Ws.append(self.W(pts, dk, P).reshape(n, n))
        FD.save(path, x0, y0, step, np.stack(Ps), self.T_field(pts).reshape(n, n), np.stack(Ws), self.K["districts"], z0, dz)
        return path


def model(K, height, c, R, Bd, E, P_, gates, squares, sq_main, acro, host, rib, water, decks, reserve, depth, rng, why=None,
          L=None, wind=None, centre=None):
    """Influence fields of a settlement: the flow along the streets (in E — the edge's `flow`), the power centers, the trade nodes,
    the center and the wealth measures. gates — entrances: gates (for growth — also the road starts from the edges of the
    place). K — knowledge (None — influences.yaml); L — knobs (levels); wind — where it blows from, °; centre — the point of
    the main power center (the scene's intent)."""
    K = K or knowledge(); L = L or levels()
    Kv = K["power"]; Kt = K["trade"]; Kb = K["wealth"]; U = K["levels"]
    n_decks = int(decks["count"]) if decks else 1
    fab = rib.buffer(depth)                                                       # the fabric — streets with the plot depth (for growth — not the place's circle)
    if Bd is not None: fab = fab.intersection(Bd)
    for z_ in reserve: fab = fab.difference(z_)
    A = float(fab.area) * n_decks; sA = math.sqrt(max(A / n_decks, 1.0))
    kp = 1.0 if L["power"] == 0.5 else float(U["power"]["n_centers"]) * L["power"]   # the power knob: the number of small centers
    kr = _lerp(U["power"]["radius"], L["power"])                                  # and the radius of the centers
    kt = 1.0 if L["trade"] == 0.5 else float(U["trade"]["amplitude"]) * L["trade"]       # the trade knob: the amplitude of the flow and the nodes
    # trade: flow
    ent = _entries(P_, E, gates)
    main_sq = sq_main["poly"] if sq_main is not None else None
    # power: the main center
    if centre is not None:
        pm = np.asarray(centre, float); seat = None
    elif acro is not None:
        pm = np.asarray(acro["top"].centroid.coords[0]); seat = "acropolis"
    elif host is not None:
        pm = np.asarray(host.centroid.coords[0]); seat = "castle"
    elif main_sq is not None:
        pm = np.asarray(main_sq.representative_point().coords[0]); seat = None
    else:
        X = np.asarray(P_, float); dd = np.linalg.norm(X - c, axis=1); ok = dd < 0.4 * R
        if not ok.any(): ok = dd <= dd.min() + 1e-6
        z = np.asarray(height(X[:, 0], X[:, 1]), float); zr = (z - z[ok].min()) / max(np.ptp(z[ok]), 1e-6)
        k = int(np.argmax(np.where(ok, zr - 0.8 * dd / R, -1e9))); pm = X[k]; seat = None
    main = _nearest_node(P_, E, main_sq.representative_point().coords[0] if main_sq is not None else pm)
    others = [n for n in (_nearest_node(P_, E, s_["poly"].representative_point().coords[0]) for s_ in squares if s_ is not sq_main) if n is not None]
    f = flows(P_, E, Kt, ent, main, others)
    q = float(np.quantile(f[f > 0], 0.98)) if (f > 0).any() else 1.0
    T_e = np.clip(np.sqrt(f / max(q, 1e-9)) * kt, 0.0, 1.0)
    for e, t in zip(E, T_e): e["flow"] = round(float(t), 3)
    top = int(n_decks - 1) if decks else None
    centres = [{"p": [round(float(pm[0]), 2), round(float(pm[1]), 2)], "s": float(Kv["main"]["s"]), "r": round(float(Kv["main"]["r"]) * sA * kr, 2),
                "main": True, "deck": top, "landmark": seat, "kind": seat or _pick(Kv["main"]["kind"], rng)}]
    # small centers: the number follows the area, the place is a crossing (higher and more passable — likelier)
    n_min = min(int(round(int(Kv["n_small"]["max"]) * kp)), int(math.floor(A / 1e4 * float(Kv["n_small"]["per_ha"]) * kp + rng.random())))
    deg = {}; tn = {}
    for e in E:
        if e["kind"] in SKIP: continue
        for v in (e["a"], e["b"]): deg[v] = deg.get(v, 0) + 1; tn[v] = max(tn.get(v, 0.0), e["flow"])
    inner = Bd.buffer(-10.0) if Bd is not None else fab
    blocked = unary_union(list(reserve) + ([water.buffer(8.0)] if water is not None and not water.is_empty else []))
    cand = [v for v, k in deg.items() if k >= 3 and inner.contains(Point(P_[v])) and (blocked.is_empty or not blocked.contains(Point(P_[v])))]
    Xc = np.asarray([P_[v] for v in cand], float).reshape(-1, 2)
    zc = np.asarray(height(Xc[:, 0], Xc[:, 1]), float) if len(cand) else np.zeros(0)
    Hc = (zc - zc.min()) / max(np.ptp(zc), 1e-6) if len(cand) else zc
    M_ = Kv["small"]
    for _ in range(n_min):
        ok = [i for i in range(len(cand)) if np.linalg.norm(Xc[i] - pm) >= M_["off_main"] * sA and
              all(np.linalg.norm(Xc[i] - np.asarray(ce["p"])) >= M_["from_each_other"] * sA for ce in centres[1:])]
        if not ok: break
        sc = np.array([(0.25 + 0.5 * Hc[i] + 0.5 * tn[cand[i]]) ** 2 for i in ok]); i = ok[int(rng.choice(len(ok), p=sc / sc.sum()))]
        dk = int(rng.integers(n_decks)) if decks else None
        vid = _pick(M_["kind"], rng)
        if vid == "tower" and decks and dk != top: vid = "chambers"                  # a tower only on the top tier (above it — the shell)
        centres.append({"p": [round(float(Xc[i][0]), 2), round(float(Xc[i][1]), 2)], "s": round(float(rng.uniform(*M_["s"])), 2),
                        "r": round(float(M_["r"]) * sA * kr, 2), "main": False, "deck": dk, "landmark": None, "kind": vid})
    # trade: nodes — squares, entrances, bridges, water
    Un = Kt["nodes"]; kern = []
    for s_ in squares:
        kern.append((s_["poly"], (Un["area"][0] if s_ is sq_main else Un["area"][0] * 0.85) * kt, max(15.0, Un["area"][1] * sA)))
    for i in ent: kern.append((Point(P_[i]), Un["entrance"][0] * kt, Un["entrance"][1]))
    br = [LineString(e["pts"]) for e in E if e.get("over_water")]
    if br: kern.append((unary_union(br), Un["bridge"][0] * kt, Un["bridge"][1]))
    if water is not None and not water.is_empty: kern.append((water, Un["water"][0] * kt, Un["water"][1]))
    # wealth: the center, the terrain measures, the windward side
    wc = main_sq if main_sq is not None else Point(pm)
    xs = np.linspace(c[0] - R, c[0] + R, 31); X, Y = np.meshgrid(xs, xs - c[0] + c[1]); pts = np.c_[X.ravel(), Y.ravel()]
    ins = shapely.contains(fab, shapely.points(pts)) if not fab.is_empty else np.ones(len(pts), bool)
    dw = shapely.distance(wc, shapely.points(pts[ins])) if ins.any() else np.array([R])
    Rw = max(float(np.quantile(dw, 0.95)), 40.0)                                    # the reach of the fabric from the wealth center
    zz = np.asarray(height(pts[ins, 0], pts[ins, 1]), float) if ins.any() else np.zeros(2)
    zq = (float(np.quantile(zz, 0.1)), float(np.quantile(zz, 0.9))); s_h = float(np.clip((zq[1] - zq[0]) / float(Kb["relief_m"]), 0.0, 1.0))
    if decks: s_h = 0.0
    a_n = rng.normal(0.0, 20.0)                                                    # the random draw is always taken: the wind knob does not shift the other random draws
    a_up = math.radians(float(wind) if wind is not None else float(Kb["wind"]) + a_n)   # where it blows from (0 — north, 90 — east): the windward side is richer
    up = np.array([math.sin(a_up), math.cos(a_up)])
    IF = Influence(K, height, centres, kern, wc, Rw, zq, s_h, up, n_decks, E, L)
    if ins.any(): IF.Wm = float(np.mean(IF.W0(pts[ins])))                         # the mean wealth of the fabric — the anchor of the knobs
    IF.area_ha = A / 1e4; IF.entries = [list(map(float, P_[i])) for i in ent]
    if why:
        why("influences", "power", {"main": centres[0]["kind"], "n_small": len(centres) - 1, "ha": round(A / 1e4, 2)},
            "the main center is a landmark (acropolis, castle), otherwise at the main square or on high ground near the middle; small ones — ⌊ha × "
            f"{Kv['n_small']['per_ha']} + chance⌋ ≤ {Kv['n_small']['max']}, at a crossing (higher and more passable — likelier); the kind is random")
        why("influences", "trade", {"n_entrances": len(ent), "n_edges": int((T_e > 0.5).sum())},
            "flow — shortest paths entrance ↔ center, entrance ↔ entrance, residents → center (Hillier 1993); nodes — squares, entrances, bridges, water")
        why("influences", "wealth", {"relief": round(s_h, 2), "windward": round(math.degrees(a_up) % 360)},
            "center — periphery; higher up the slope, on level ground — the windward side (Heblich 2021); near power; in a hive — the top tier")
        if any(v != 0.5 for v in L.values()) or wind is not None or centre is not None:
            why("influences", "knobs", dict(L, **({"wind": wind} if wind is not None else {}), **({"center": [round(float(v), 2) for v in centre]} if centre is not None else {})),
                "the scene's intent: power — the number and radius of the centers, trade — the amplitude of the flow, wealth — the shift, inequality — the amplitude of the difference")
    return IF
