"""Time — a town over a town (knowledge — levels.yaml). An era: a fill raises the ground (a layer); each old building gets a
fate (keep it: the entrance is higher; cut it down to the new ground: the lower part becomes empty rooms under the new
building; bury it: stumps in the layer), new plots cling to the old walls with probability reuse, and the new basements
cut into the old. The main building of a block (a temple) is cut down and repeated on its own walls (San Clemente). An
entrance that has sunk below ground deeper than the porch gets a sunken yard with steps.
Era snapshots — the same world at the moment of era k (snapshot)."""
import copy
import os

import numpy as np
import shapely
import yaml
from shapely.geometry import box as sbox

from components.buildings.levels import strata as S

HERE = os.path.dirname(os.path.abspath(__file__))
K = yaml.safe_load(open(os.path.join(HERE, "levels.yaml")))


def _split(x0, x1, rng, w, old=(), reuse=0.0, snap=2.0):
    """Plot boundaries along the street; with probability reuse a boundary clings to the old one (wall on wall)."""
    e = [x0]
    while e[-1] < x1 - w[0] * 0.8:
        nxt = e[-1] + rng.uniform(*w)
        near = [o for o in old if abs(o - nxt) <= snap and o - e[-1] >= w[0] * 0.7]
        if near and rng.random() < reuse: nxt = min(near, key=lambda o: abs(o - nxt))
        e.append(min(nxt, x1) if x1 - nxt >= w[0] * 0.7 else x1)
    return e


def _record(b, k):
    b.setdefault("hist", []).append((k, b["state"], b["z_cut"]))


def strata_town(seed=3, length=66.0, fills=(3.4, 2.4), eras=("ancient", "medieval", "modern"), y_back=-6.0, y_front=10.0,
                street=6.0, lane=5.0, reuse=None, temple=True):
    rng = np.random.default_rng(seed); E = K["eras"]; reuse = K["reuse"] if reuse is None else reuse; So = K["soil"]
    main = sbox(-1.0, y_front, length + 1.0, y_front + street + 1.0)
    xs0 = round(length * 0.22, 1); lane_p = sbox(xs0, y_back - 1.0, xs0 + lane, y_front + 0.01)   # a cross lane from era 1
    streets = [main] + [main.union(lane_p)] * len(fills)
    g0 = lambda X, Y: 0.35 * np.sin(np.asarray(X, float) / 7.0) + 0.015 * np.asarray(X, float) + 0.0 * np.asarray(Y, float)
    pave = lambda g, poly: (lambda X, Y: g(X, Y) + So["pave"] * shapely.contains_xy(poly, np.asarray(X, float), np.asarray(Y, float)))
    G = [g0]; P = [pave(g0, streets[0])]
    for k, d in enumerate(fills):
        G.append(lambda X, Y, prev=G[-1], d=d, k=k: prev(X, Y) + d * (1 + 0.12 * np.sin(np.asarray(X, float) / 9.0 + 2 * k)))   # the fill goes onto the era's ground: the pavement gets buried, the curb does not accumulate
        P.append(pave(G[-1], streets[k + 1]))
    layers = [f for pair in zip(G, P) for f in pair]
    W = S.World((0.0, 0.0, length, y_front + street), layers)
    W.ground_k = P; W.history = []; W.streets_k = streets
    era_t = [tuple(E[eras[min(k, len(eras) - 1)]]["tint"]) for k in range(len(G))]
    W.layer_cls = [c for k in range(len(G)) for c in (("rock" if k == 0 else "fill"), "street")]
    W.layer_tints = [t for k in range(len(G)) for t in ((tuple(So["rock"]) if k == 0 else tuple(So["fill"][min(k - 1, 2)])), era_t[k])]
    W.top_cls = lambda x, y: (("street", era_t[-1]) if streets[-1].contains(shapely.geometry.Point(x, y)) else ("ground", (1.0, 1.0, 1.0)))
    cx = length / 2; Tm = K["temple"]
    nid = [0]

    def new(x0, x1, y0, y1, k, era, kind="house", **kw):
        st = E[era]; z0 = float(W.ground_k[k](np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]))[0]) + 0.15
        if kind == "temple":
            t = Tm[era]; b = S.building(x0, x1, y0, y1, z0 + t["podium"], 1, fh=t["h"], t=st["t"] + 0.2, roof="gable", epoch=k,
                                        kind="temple", pitch=t["pitch"], ridge="y", tone=st["tone"], id=nid[0], stoop_max=2.2,
                                        z_found=z0 - 0.75, tint=st["tint"], **kw)
        else:
            fl = int(rng.integers(st["floors"][0], st["floors"][1] + 1)); ce = int(rng.integers(st["cellars"][0], st["cellars"][1] + 1))
            b = S.building(x0, x1, y0, y1, z0, fl, fh=st["fh"], t=st["t"], cellars=ce, roof=st["roof"], pitch=st.get("pitch", 40),
                           ridge="x", epoch=k, kind=kind, tone=float(st["tone"] * rng.uniform(0.96, 1.04)), id=nid[0],
                           tint=st["tint"], vaults=st.get("vaults", True), **kw)
        nid[0] += 1; b["era"] = era; W.buildings.append(b); _record(b, k); return b

    # era 0: plots along the street, a temple in the middle
    era = eras[0]; tw = Tm[era]["w"]
    tx0, tx1 = cx - tw / 2, cx + tw / 2
    for a, c in ((1.0, tx0 - 1.0), (tx1 + 1.0, length - 1.0)):
        e = _split(a, c, rng, E[era]["plot"])
        for x0, x1 in zip(e[:-1], e[1:]): new(x0, x1, y_back, y_front, 0, era)
    tmp = new(tx0, tx1, y_back + 1.0, y_back + 1.0 + Tm[era]["d"], 0, era, "temple") if temple else None
    # eras 1…: fill → fates → new plots on the free land (a lane is not a plot)
    for k in range(1, len(G)):
        era = eras[min(k, len(eras) - 1)]
        alive = [b for b in W.buildings if b["state"] in ("whole", "raised")]
        for b in alive:
            zc = float(W.ground_k[k](np.array([(b["x0"] + b["x1"]) / 2]), np.array([(b["y0"] + b["y1"]) / 2]))[0])
            d = zc - (b["z0"] - 0.15)
            if b["kind"] == "temple" and d > 1.5 and k == 1:
                b["state"] = "cut"; b["z_cut"] = zc
            elif b["kind"] != "temple" and rng.random() < K["renew"]:                 # rebuilding: demolished down to the ground, the lower part becomes a basement
                b["state"] = "cut"; b["z_cut"] = zc
            elif d < K["fate"]["keep_below"] * b["fh"] or b["kind"] == "temple":
                b["state"] = "raised"
            elif rng.random() < K["fate"]["cut"]:
                b["state"] = "cut"; b["z_cut"] = zc
            else:
                b["state"] = "buried"; b["z_cut"] = float(zc - rng.uniform(0.6, 2.0))
            _record(b, k)
        if temple and k == 1:                                                 # a church on the temple walls: the same outline, longer (up to the street)
            ch = new(tmp["x0"] - 0.5, tmp["x1"] + 0.5, tmp["y0"], min(tmp["y0"] + Tm[era]["d"], y_front - 3.5), k, era, "temple", on=tmp["id"])
        old_edges = sorted({v for b in W.buildings if b["state"] == "cut" and b["kind"] != "temple" for v in (b["x0"], b["x1"])})
        busy = [(b["x0"] - 0.6, b["x1"] + 0.6) for b in W.buildings if b["state"] in ("whole", "raised") and b["epoch"] < k or
                (b["epoch"] == k and b["kind"] == "temple")] + [(xs0, xs0 + lane)]
        free = _free(1.0, length - 1.0, busy)
        for a, c in free:
            if c - a < E[era]["plot"][0] * 0.8: continue
            e = _split(a, c, rng, E[era]["plot"], old_edges, reuse, K["snap"])
            for x0, x1 in zip(e[:-1], e[1:]):
                b = new(x0, x1, y_back, y_front, k, era)
                _settle(W, b, k)
    # what was cut down and has nothing standing on it: the basement stays underground (covered by the ground-floor slab),
    # with no basement — buried
    for b in W.buildings:
        if b["state"] == "cut" and not b.get("under") and b["kind"] != "temple":
            cel = [f for f in b["floors"] if f["kind"] == "cellar"]
            if cel and b["z_cut"] > cel[-1]["z"] + cel[-1]["h"] - 0.2: b["z_cut"] = cel[-1]["z"] + cel[-1]["h"]; b["covered"] = True
            else: b["state"] = "buried"; b["z_cut"] = float(b["z_cut"] - 0.8)
            _record(b, len(G) - 1)
    # result: links of the storeys with the ground; a sunken yard if the entrance went underground
    for b in W.buildings:
        if b["state"] in ("whole", "raised", "cut"): S.relate(W, b, stoop_max=b.get("stoop_max", 1.3))
    for b in W.buildings:
        if b["state"] in ("whole", "raised") and not b["doors"]: _sunk_court(W, b, y_front)
    W.notes.append(_summary(W))
    return W


def _free(a, c, busy):
    out = [(a, c)]
    for b0, b1 in sorted(busy):
        nxt = []
        for x0, x1 in out:
            if b1 <= x0 or b0 >= x1: nxt.append((x0, x1)); continue
            if b0 > x0: nxt.append((x0, b0))
            if b1 < x1: nxt.append((b1, x1))
        out = nxt
    return out


def _settle(W, b, k):
    """A new building on an old one: a cut-down house under the same outline (≥ 70 %) — its lower part becomes the new building's
    basement (without basements of its own, standing on the old walls); otherwise the new basements cut everything old under
    the outline below their own bottom."""
    for o in W.buildings:
        if o is b or o["epoch"] >= k or o["state"] in ("gone",): continue
        inter = o["poly"].intersection(b["poly"]).area
        if inter < 1.0: continue
        top = o["z_cut"] if o["z_cut"] is not None else o["z_top"]
        if o["state"] == "cut" and inter >= 0.7 * min(o["poly"].area, b["poly"].area) and not o.get("under"):
            o["under"] = b["id"]; b["on"] = o["id"]
            b["floors"] = [f for f in b["floors"] if f["kind"] == "floor"]; b["z_found"] = top - 0.3
            continue
        if top > b["z_found"] + 0.3:
            o["z_cut"] = float(max(b["z_found"], o["z_found"] + 0.6)); o["state"] = "cut" if o["state"] != "buried" else "buried"
            if o["z_cut"] <= o["z_found"] + 0.7: o["state"] = "gone"
            _record(o, k)


def _sunk_court(W, b, y_front, depth=None):
    """A sunken yard in front of the street side: an excavation down to the floor + steps from the street + retaining walls at
    the edges."""
    depth = depth or K["sunk_court"]; z = b["floors"][0]["z"] - 0.1 if b["floors"][0]["kind"] == "floor" else b["z0"] - 0.1
    xc = (b["x0"] + b["x1"]) / 2; w = min(b["x1"] - b["x0"] - 2.0, 9.0)
    poly = sbox(xc - w / 2, b["y1"], xc + w / 2, b["y1"] + depth); W.digs.append((poly, z))
    g = W.ground(xc, b["y1"] + depth + 1.5)
    S.stairs(W, b["y1"] + depth + 0.0, b["y1"] + 1.2, xc, g, z, width=3.0, axis="y")
    for x0, x1 in ((xc - w / 2 - 0.5, xc - w / 2), (xc + w / 2, xc + w / 2 + 0.5)):
        W.solids.append({"type": "prism", "poly": sbox(x0, b["y1"], x1, b["y1"] + depth), "z0": z - 0.5, "z1": g + 0.9, "cls": "retain"})
    S.relate(W, b, stoop_max=b.get("stoop_max", 1.3))
    W.notes.append(f"sunken yard at {b['kind']} {b['id']}: bottom {z:.1f} m, street {g:.1f} m")


def _summary(W):
    from collections import Counter
    c = Counter((b["era"], b["state"]) for b in W.buildings)
    return "; ".join(f"{e} {s} {n}" for (e, s), n in sorted(c.items()))


def snapshot(W, k):
    """The world at the moment of era k: the layers up to k, the buildings of eras ≤ k in their state at k (excavations — only
    the final ones, they are not attached).
    The world of events (without fill layers: a bridge, new over old) — all layers, bodies and links alive in era k (live)."""
    if getattr(W, "ground_k", None) is None:
        V = S.World(W.block, W.layers, W.tones)
        V.layer_cls, V.layer_tints, V.top_cls = W.layer_cls, W.layer_tints, W.top_cls
        V.digs = [d for d in W.digs if (d[2] if len(d) > 2 else 0) <= k]
        past = getattr(W, "past", {})
        for name in ("streets", "decks", "solids", "voids", "links"):
            setattr(V, name, [e for e in getattr(W, name) + past.get(name, []) if live(e, k)])
        for a in ("auto_retain", "retain_zone", "origin"):
            if hasattr(W, a): setattr(V, a, getattr(W, a))
    else:
        V = S.World(W.block, W.layers[:2 * k + 2])
        V.layer_cls, V.layer_tints = W.layer_cls[:2 * k + 2], W.layer_tints[:2 * k + 2]
        st_k = W.streets_k[k]; tk = W.layer_tints[2 * k + 1]
        V.top_cls = lambda x, y: (("street", tk) if st_k.contains(shapely.geometry.Point(x, y)) else ("ground", (1.0, 1.0, 1.0)))
    for b in W.buildings:
        if b["epoch"] > k: continue
        c = copy.deepcopy(b); st = [h for h in b["hist"] if h[0] <= k][-1]
        c["state"], c["z_cut"] = st[1], st[2]
        if c["state"] == "raised": c["state"] = "whole"
        if c["state"] != "gone" and c.get("on") is not None and k < b["epoch"]: continue
        V.buildings.append(c)
    for b in V.buildings:
        if b["state"] in ("whole", "cut"): S.relate(V, b, stoop_max=b.get("stoop_max", 1.3))
    return V


# ---------------------------------------------------------------- events: new over old without a fill
def live(e, k):
    """Whether a body, street, deck or link exists in era k: from epoch (default 0) up to until (exclusive)."""
    return e.get("epoch", 0) <= k < e.get("until", 1 << 30)


def _retire(W, name, pred, k):
    """Remove bodies or links from the world by a condition in era k: until = k, into W.past (the era snapshots see them up to k)."""
    keep, gone = [], []
    for e in getattr(W, name): (gone if pred(e) else keep).append(e)
    for e in gone: e["until"] = k
    setattr(W, name, keep)
    if not hasattr(W, "past"): W.past = {}
    W.past.setdefault(name, []).extend(gone)
    return len(gone)


def _clip_links(W, poly, k, z_roof, clear=2.2):
    """Stairs under a new body poly: the link breaks at its wall (the bottom is outside, the top is sealed), the steps remain
    while there is enough height above them up to z_roof (a passage under the house)."""
    from shapely.geometry import LineString
    cut = [L for L in W.links if L.get("kind") in ("stair", "ramp") and LineString([L["a"][:2], L["b"][:2]]).intersects(poly)]
    _retire(W, "links", lambda L: any(L is c for c in cut), k)
    for L in cut:
        lo, hi = sorted((np.asarray(L["a"], float), np.asarray(L["b"], float)), key=lambda p: p[2])
        ts = np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(hi[:2] - lo[:2]) / 0.25)) + 1)
        P = lo[None, :] + (hi - lo)[None, :] * ts[:, None]
        ins = shapely.contains_xy(poly, P[:, 0], P[:, 1])
        i = int(np.argmax(ins)) if ins.any() else len(P)
        if i > 1: W.links.append({**{q: v for q, v in L.items() if q != "until"}, "a": tuple(map(float, lo)), "b": tuple(map(float, P[i - 1])), "epoch": k})
    return _retire(W, "solids", lambda s_: s_.get("cls") == "stair" and s_["type"] == "prism" and s_["poly"].intersects(poly)
                   and s_["z1"] > z_roof - clear, k)


def overbuild(W, olds, k, floors, fh=4.2, z_floor=None, kind="hall", **kw):
    """New over old (Edinburgh: the Exchange over Mary King's Close, 1753–61): the olds buildings are cut down at the new floor —
    their storeys below stay as its basements, the passage between them — a street under the building (sealed at the top,
    entered from below); the new building takes the common extent and stands on the old walls. The stairs under it break off
    at its wall."""
    x0, x1 = min(o["x0"] for o in olds), max(o["x1"] for o in olds); y0, y1 = min(o["y0"] for o in olds), max(o["y1"] for o in olds)
    nid = max(b["id"] for b in W.buildings) + 1
    b = S.building(x0, x1, y0, y1, z_floor + 0.15, floors, fh=fh, epoch=k, kind=kind, id=nid, z_found=z_floor - 0.6,
                   on=[o["id"] for o in olds], **kw)
    for o in olds:
        o["state"] = "cut"; o["z_cut"] = float(z_floor - 0.3); o["under"] = nid; _record(o, k)
    W.buildings.append(b); _record(b, k)
    n = _clip_links(W, b["poly"], k, b["z_found"])
    W.notes.append(f"era {k}: {kind} {nid} over houses {', '.join(str(o['id']) for o in olds)} — their storeys below {z_floor:.1f} m "
                   f"became basements, steps under it removed: {n}")
    return b


def _minus(seg, cut):
    a, b = seg; c0, c1 = cut
    if c1 <= a or c0 >= b: return [seg]
    return [s_ for s_ in ((a, c0), (c1, b)) if s_[1] - s_[0] > 0.5]


def bridge_over(W, k, x0, x1, yc, z, width=12.0, depth=9.5, gap=0.2, span=7.0, pier=2.4, z_spring=None, keep_open=(),
                plot=(7.0, 10.0), over=(2, 4), t=1.2, fh=3.0, rng=None, tint=(1.0, 1.0, 1.0)):
    """A bridge over an old town (Edinburgh, South Bridge 1785–88) along x from x0 to x1 at elevation z: the strip of the bridge
    and of the two fronts is demolished (buildings, stairs, links); an arcade — a span over each street in keep_open at its
    middle, the others at the same step; a vault in the slope — solid masonry, an open vault over the slope — rooms (the floor
    leveled, floors above, end walls with a door); the front — houses from the ground up to the deck and above, with a floor
    both at the ground and on the bridge (strata.floor_fit). → the front's houses."""
    from components.buildings.levels import geom as G
    from shapely.geometry import LineString
    rng = rng or np.random.default_rng(k)
    yb0, yb1 = yc - width / 2, yc + width / 2
    band = sbox(x0, yb0 - gap - depth - 0.4, x1, yb1 + gap + depth + 0.4)
    gone = 0
    for b in W.buildings:                                                    # demolition of the strip
        if b["state"] != "gone" and b["epoch"] < k and b["poly"].intersection(band).area > 0.5:
            b["state"] = "gone"; b["z_cut"] = None; _record(b, k); gone += 1
    nst = _retire(W, "links", lambda L: L.get("kind") in ("stair", "ramp") and LineString([L["a"][:2], L["b"][:2]]).intersects(band), k)
    _retire(W, "solids", lambda s_: s_.get("cls") == "stair" and s_["type"] == "prism" and s_["poly"].intersects(band), k)
    # the arcade and the deck
    zs = z_spring if z_spring is not None else (z - t) - span / 2 - 1.0
    xs = np.arange(x0, x1 + 0.01, 0.5); zb = float(np.min(W.layer(-1)(xs, np.full_like(xs, yc)))) - 2.0
    step = span + pier; c0 = (keep_open[0][0] + keep_open[0][1]) / 2 if keep_open else x0 + pier + span / 2
    centers = [c for c in (c0 + step * j for j in range(-30, 31)) if x0 + pier + span / 2 - 1e-6 <= c <= x1 - pier - span / 2 + 1e-6]
    prof = sbox(x0, zb, x1, z - t); n_open = n_vault = 0
    for c in centers:
        xa, xb = c - span / 2, c + span / 2
        g = np.asarray(W.layer(-1)(np.linspace(xa, xb, 9), np.full(9, yc)), float)
        if float(g.mean()) > zs - 1.0: continue                               # a vault in the slope — solid masonry
        arch = G.arch_profile(xa, xb, zb - 1.0, zs); prof = prof.difference(arch)
        if any(a <= c <= b_ for a, b_ in keep_open): n_open += 1; continue   # a span over a street — open
        zf = float(g.min()); W.digs.append((sbox(xa, yb0, xb, yb1), zf, k))  # vaults — rooms: the floor at the lower ground of the span
        for zz in np.arange(zf + fh, zs + span / 2 - 1.0, fh):
            sl = arch.intersection(sbox(xa, zz - 0.25, xb, zz))
            if not sl.is_empty: W.solids.append({"type": "extrude", "prof": sl, "axis": "y", "a0": yb0 + 0.4, "a1": yb1 - 0.4, "cls": "slab", "tint": tint, "epoch": k})
        face = arch.intersection(sbox(xa, zf, xb, 1e3)).difference(sbox(c - 0.7, zf, c + 0.7, zf + 2.2))
        for ya in (yb0, yb1 - 0.4):
            W.solids.append({"type": "extrude", "prof": face, "axis": "y", "a0": ya, "a1": ya + 0.4, "cls": "wall", "tint": tint, "epoch": k})
        n_vault += 1
    W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": yb0, "a1": yb1, "cls": "wall", "tint": tint, "cut_ground": True, "epoch": k})
    W.decks.append({"poly": sbox(x0, yb0, x1, yb1), "z": z, "t": t, "name": "bridge", "epoch": k, "tint": tint})
    for a, b_ in keep_open:                                                  # parapets over the open spans
        for ya in (yb0, yb1 - 0.4):
            W.solids.append({"type": "prism", "poly": sbox(a - 1.0, ya, b_ + 1.0, ya + 0.4), "z0": z - 0.1, "z1": z + 1.1, "cls": "wall", "tint": tint, "epoch": k})
    # front: houses on both sides of the deck, with a floor both at the ground and on the bridge
    segs = [(x0 + 0.3, x1 - 0.3)]
    for a, b_ in keep_open: segs = [q for sg in segs for q in _minus(sg, (a - 0.4, b_ + 0.4))]
    nid = max(b["id"] for b in W.buildings) + 1; front = []
    for ya, yb_ in ((yb0 - gap - depth, yb0 - gap), (yb1 + gap, yb1 + gap + depth)):
        for a, c in segs:
            if c - a < plot[0] * 0.6: continue
            e = _split(a, c, rng, plot)
            for xa, xb in zip(e[:-1], e[1:]):
                if xb - xa < 4.0: continue
                Xg, Yg = np.meshgrid(np.linspace(xa, xb, 6), np.linspace(ya, yb_, 6)); zmin = float(np.min(W.layer(-1)(Xg, Yg)))
                z0, kf, fhf = S.floor_fit(zmin, z)
                b = S.building(xa + 0.1, xb - 0.1, ya, yb_, z0, kf + int(rng.integers(over[0], over[1] + 1)), fh=fhf, t=0.55, roof="gable",
                               pitch=32, ridge="x", epoch=k, kind="front", id=nid, tint=tint, z_found=min(z0 - 0.6, zmin - 0.3), bay=3.0,
                               below=kf)
                nid += 1; W.buildings.append(b); _record(b, k); front.append(b)
    W.notes.append(f"era {k}: bridge {z:.1f} m, buildings demolished {gone}, stairs {nst}; spans {len(centers)} — open over a street: "
                   f"{n_open}, vaulted rooms {n_vault}, in the slope {len(centers) - n_open - n_vault}; front {len(front)} houses")
    return front
