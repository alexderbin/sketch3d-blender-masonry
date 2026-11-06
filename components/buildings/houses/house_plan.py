"""The "house" type: a house program on a plot and its geometry.
  program(kind, plot, context, rng) — the plan (project Python, shapely): the outline on the plot (the whole plot or a
            rectangle from the facade), yard, storeys, roof (which edges are eaves), openings, shop, jetties, colors →
            a dict (JSON);
  edges(programs, street ribbon) — classes of the outline edges: street (rank), neighbor (shared wall, no windows), yard,
            rear;
  geometry(program) — numpy only (Blender and project Python): volumes, walls with openings (s, z, width, height, kind),
            roof by faces, jetties, pipes. The roof is an envelope of slope planes: hip and gable — the lower envelope of
            the planes from the eaves, slopes toward the yard — the upper one from the yard edges, a ring — the smaller of
            the two (a straight skeleton of a convex outline with a yard).
Separately: plan(place, out, kind=…, at=…, wealth=, trade=, power=) — one house on a site (house.json); the three numbers
are the same adjustments as in a settlement (part of the "influences" core: tone, shops, cheap features, color by wealth)."""
import os
import sys
from components.buildings import arch_parts
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_K = None


def kinds():
    """Knowledge of the house kinds (houses.yaml); used only in the project's Python."""
    global _K
    if _K is None:
        import yaml
        _K = yaml.safe_load(open(os.path.join(HERE, "houses.yaml")))["kinds"]
    return _K


def _u(r, rng):
    return float(r[0] + (r[1] - r[0]) * rng.random()) if isinstance(r, (list, tuple)) else float(r)


def _col(v, rng, k=0.05, w=None, sp=0.15):
    """A color from a list of variants (or a single one) with a tone spread of ±k; w — the plot's wealth: the list runs from
    poor to rich, the index is w·(n − 1) ± sp·n (influences, houses.yaml `by_wealth`), otherwise — random."""
    if isinstance(v[0], (list, tuple)):
        i = int(rng.integers(len(v)))
        if w is not None: i = int(np.clip(round(w * (len(v) - 1) + (rng.random() - 0.5) * 2 * sp * len(v)), 0, len(v) - 1))
        v = v[i]
    m = 1 + (rng.random() - 0.5) * 2 * k
    return [round(float(min(1, max(0, c * m))), 4) for c in v]


# ---------------------------------------------------------------- plan (shapely)
def _poly_ccw(P):
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    return orient(Polygon(P), 1.0)


def rect_on_front(plot, p0, p1, n_out, off, depth, width=None, min_depth=3.5):
    """A rectangle from the facade (segment p0–p1 on the street, n_out — toward the street): setback off, depth depth, width
    (centered on the facade; None — the whole facade). If it does not fit the plot, depth and width are squeezed.
    → a shapely Polygon or None."""
    from shapely.geometry import Polygon
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); n_out = np.asarray(n_out, float); v = -n_out
    L = float(np.linalg.norm(p1 - p0)); u = (p1 - p0) / max(L, 1e-9)
    W = L if width is None else min(width, L - 0.6); c = (p0 + p1) / 2
    zone = plot.buffer(0.05)
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6):
        w_ = W * shrink; a = c - u * w_ / 2; b = c + u * w_ / 2
        lo, hi = min_depth, depth
        R = lambda d: Polygon([a + v * off, b + v * off, b + v * (off + d), a + v * (off + d)])
        if not R(lo).within(zone): continue
        for _ in range(14):
            mid = (lo + hi) / 2
            if R(mid).within(zone): lo = mid
            else: hi = mid
        if R(hi).within(zone): lo = hi
        return R(lo)
    return None


def _convex_fit(P):
    """Plot outline for a house: convex — as is (simplified); nearly convex — the hull within a tolerance; otherwise None."""
    Q = P.simplify(0.25, preserve_topology=True)
    H = Q.convex_hull
    if H.area <= Q.area * 1.012: return _poly_ccw(list(H.exterior.coords)[:-1])
    return None


def _local(F, p0, p1):
    """A frame from the facade: u — along the facade, v — into the depth. → (o, u, v, umin, umax, vmin, vmax)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9); v = np.array([-u[1], u[0]])
    C = np.asarray(F.exterior.coords)[:-1]
    if np.mean((C - p0) @ v) < 0: v = -v
    U = (C - p0) @ u; V = (C - p0) @ v
    return p0, u, v, float(U.min()), float(U.max()), float(V.min()), float(V.max())


def _box(o, u, v, u0, u1, v0, v1):
    return [list(o + u * u0 + v * v0), list(o + u * u1 + v * v0), list(o + u * u1 + v * v1), list(o + u * u0 + v * v1)]


def _ccw(P):
    P = [list(map(float, p)) for p in P]
    a = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))
    return P if a > 0 else P[::-1]


def program(kind, plot, ctx, rng):
    """House program. plot: {poly (shapely), fronts [{p0, p1, rank, n, len}], id}; ctx: storeys [a, b], height limit,
    height(x, y), z_deck (hive tier), ceiling (tier height), color (overrides); from the influences (../influences/, place) —
    center (place value 0–1), shops (share), tone (± lighter), influence [P, T, W] and palette (color by wealth — the kind's
    `by_wealth`), corrections (a nested key replacement for the kind: cheap by wealth, compounds). → a list of programs or None."""
    from shapely.geometry import Polygon
    K = _adjust(kinds()[kind], ctx); P = plot["poly"]
    fr = sorted(plot.get("fronts") or [], key=lambda f: (["artery", "street", "lane", "passage", "bridge", "road"].index(f["rank"]) if f["rank"] in ("artery", "street", "lane", "passage", "bridge", "road") else 9, -f["len"]))
    if fr: main = fr[0]; p0, p1, n_out = main["p0"], main["p1"], main["n"]
    else:
        C = np.asarray(P.exterior.coords)[:-1]; L = np.linalg.norm(np.roll(C, -1, 0) - C, axis=1); i = int(np.argmax(L))
        p0, p1 = C[i].tolist(), C[(i + 1) % len(C)].tolist(); d = np.asarray(p1) - p0; d /= max(np.linalg.norm(d), 1e-9)
        n_out = [d[1], -d[0]]
        if P.contains(Polygon([p0, p1, list(np.asarray(p1) + np.asarray(n_out)), list(np.asarray(p0) + np.asarray(n_out))]).centroid): n_out = [-d[1], d[0]]
        main = {"rank": None}
    if K["outline"] in ("plot",):
        F = _convex_fit(P)
        if F is None:                                                             # non-convex — the largest rectangle from one of the facades
            best = None
            for f_ in ([main] if main.get("rank") else []) + [f for f in fr if f is not main and f["len"] >= 3.0] + ([] if fr else [{"p0": p0, "p1": p1, "n": n_out}]):
                R_ = rect_on_front(P, f_["p0"], f_["p1"], f_["n"], 0.0, 60.0)
                if R_ is not None and (best is None or R_.area > best[0].area * 1.15): best = (R_, f_)
            if best is not None:
                F, f_ = best
                if f_ is not main and f_.get("rank"): main = f_; p0, p1, n_out = f_["p0"], f_["p1"], f_["n"]
    elif K["outline"] == "given":
        F = _poly_ccw(list(P.exterior.coords)[:-1])
    else:
        off = _u(K.get("offset", [0.0, 0.0]), rng); dep = _u(K.get("depth", [10, 14]), rng)
        wid = _u(K["width"], rng) if "width" in K else None
        F = rect_on_front(P, p0, p1, n_out, off, dep, wid)
    if F is None or F.area < 12: return None
    F = _poly_ccw(list(F.exterior.coords)[:-1])
    C = np.asarray(F.exterior.coords)[:-1]
    # the facade edge of the outline — the one closest to the plot facade and parallel to it
    fm = (np.asarray(p0) + np.asarray(p1)) / 2; fd = np.asarray(p1) - p0; fd /= max(np.linalg.norm(fd), 1e-9)
    def score(i):
        a, b = C[i], C[(i + 1) % len(C)]; d = b - a; L = np.linalg.norm(d)
        return np.linalg.norm((a + b) / 2 - fm) - 3 * abs(float(d @ fd)) / max(L, 1e-9) - 0.05 * L
    front = int(min(range(len(C)), key=score))
    # storeys
    z_deck = ctx.get("z_deck"); fh = float(K["storey"]); h0 = float(K.get("first", fh))
    if ctx.get("ceiling"):
        n = max(1, int((ctx["ceiling"] - h0) / fh) + 1)
        if ctx.get("tower"): n = int(n * ctx["tower"])
    else:
        lo, hi = K.get("storeys") or ctx.get("storeys", [1, 2]); n = int(rng.integers(int(lo), int(hi) + 1))
    if ctx.get("limit"): n = max(1, min(n, int((ctx["limit"] - h0) / fh) + 1))
    height = ctx.get("height")
    fp = (C[front] + C[(front + 1) % len(C)]) / 2
    if z_deck is not None: z0 = float(z_deck); zb = z0
    else:
        zs = [float(height(x, y)) for x, y in list(C) + [np.asarray(F.centroid.coords[0])]]
        z0 = float(height(fp[0], fp[1])) + 0.15; zb = min(zs) - 0.35
        z0 = max(z0, min(zs) + 0.15)
    # court
    court = None; court2 = None; split = None
    D = K.get("court")
    if D and F.area > (D.get("off_squares", 0) or 0):
        o, u, v, u0, u1, v0, v1 = _local(F, C[front], C[(front + 1) % len(C)])
        wing = _u(D.get("wing", [4, 6]), rng); share = _u(D["frac"], rng)
        Wu, Wv = u1 - u0, v1 - v0
        if D.get("two") and Wv > 4 * wing + 6:                                # domus: atrium at the front, peristyle at the back
            sv = v0 + Wv * 0.45; split = sv
            a1 = max(3.0, min(Wu - 2 * wing, math.sqrt(share * 0.5 * F.area))); a2 = max(4.0, min(Wu - 2 * wing, Wv * 0.55 - 2 * wing))
            cv = (v0 + sv) / 2; court = _box(o, u, v, (u0 + u1) / 2 - a1 / 2, (u0 + u1) / 2 + a1 / 2, cv - min(a1, (sv - v0) - 2 * wing) / 2, cv + min(a1, (sv - v0) - 2 * wing) / 2)
            cv2 = (sv + v1) / 2; h2 = max(3.0, (v1 - sv) - 2 * wing)
            court2 = _box(o, u, v, u0 + wing, u1 - wing, cv2 - h2 / 2, cv2 + h2 / 2)
        elif Wu > 2 * wing + 3 and Wv > 2 * wing + 3:
            cu = max(3.0, min(Wu - 2 * wing, math.sqrt(share * F.area * Wu / Wv)))
            cvv = max(3.0, min(Wv - 2 * wing, share * F.area / cu))
            if D.get("where") == "south":                                         # the yard toward the south side of the plot (toward the street, if the south is there)
                south = np.array([0.0, -1.0]); at_back = float(v @ south) > 0
                vv = (v1 - cvv, v1) if at_back else (v0, v0 + cvv)
                court = _box(o, u, v, (u0 + u1) / 2 - cu / 2, (u0 + u1) / 2 + cu / 2, *vv)
            else:
                court = _box(o, u, v, (u0 + u1) / 2 - cu / 2, (u0 + u1) / 2 + cu / 2, (v0 + v1) / 2 - cvv / 2, (v0 + v1) / 2 + cvv / 2)
    # roof
    R = K["roof"]; rk = R["kind"]; pitch = _u(R.get("slope", [30, 30]), rng)
    if rk in ("inward", "ring") and court is None: rk = "hip"
    nE = len(C); eaves = [True] * nE
    if rk == "gable":
        gable_to_street = rng.random() < float(R.get("gable_to_street", 0.5))
        if nE != 4: rk = "hip"
        else:
            fdir = C[(front + 1) % nE] - C[front]; fdir /= np.linalg.norm(fdir)
            for i in range(nE):
                d = C[(i + 1) % nE] - C[i]; par = abs(float(d @ fdir)) / max(np.linalg.norm(d), 1e-9) > 0.7
                eaves[i] = (not par) if gable_to_street else par
            if not any(eaves): rk = "hip"; eaves = [True] * nE
    elif rk == "pitch":
        eaves = [False] * nE
        back = max(range(nE), key=lambda i: np.linalg.norm((C[i] + C[(i + 1) % nE]) / 2 - fp))
        eaves[back if R.get("high") == "facade" else front] = True
        if R.get("high") == "rear": eaves = [False] * nE; eaves[front] = True
    elif rk == "inward": eaves = [False] * nE
    elif rk == "flat": eaves = [False] * nE
    roof = {"kind": rk, "pitch": round(pitch, 2), "ov": float(R.get("overhang", 0.5)), "eaves": [bool(e) for e in eaves], "parapet": float(R.get("parapet", 0.0))}
    # openings, shop, jetties
    Wn = K["windows"]; win = {"bay": float(Wn["rhythm"]), "w": round(_u(Wn["w"], rng), 2), "h": round(_u(Wn["h"], rng), 2), "sill": float(Wn["sill"]),
                           "kind": Wn.get("kind", "rect"), "street": Wn.get("street", "yes")}
    Dd = K["door"]; door = {"w": round(_u(Dd["w"], rng), 2), "h": round(_u(Dd["h"], rng), 2), "kind": Dd.get("kind", "rect")}
    shop = None; S = K.get("shop")
    if S and main.get("rank") and rng.random() < S["frac"] and (not S.get("streets") or main["rank"] in S["streets"]):
        shop = {"w": round(_u(S["w"], rng), 2), "h": round(_u(S["h"], rng), 2), "kind": S.get("kind", "segment"), "bay": float(S["rhythm"])}
    jetty = 0.0
    if K.get("reach") and n >= 2 and rng.random() < K["reach"]["frac"]: jetty = round(_u(K["reach"]["m"], rng), 2)
    fach = bool(K.get("fachwerk") and n >= 2 and rng.random() < K["fachwerk"])
    if K.get("fachwerk") and n == 1 and kind == "annex": fach = rng.random() < K["fachwerk"]
    chim = int(bool(K.get("pipe")) and rng.random() < K["pipe"])
    balc = None
    if K.get("balcony") and n >= 3 and rng.random() < K["balcony"]["frac"]: balc = {"floor": int(K["balcony"]["storey"]), "d": round(_u(K["balcony"]["depth"], rng), 2)}
    pipes = None; gal = None
    if K.get("pipes"): pipes = {"step": round(_u(K["pipes"]["step"], rng), 2), "r": round(_u(K["pipes"]["r"], rng), 2)}
    if K.get("gallery") and rng.random() < 0.9: gal = {"every": int(K["gallery"]["every"]), "share": float(K["gallery"]["frac"]), "d": float(K["gallery"]["depth"])}
    CC = dict(K["color"], **(ctx.get("color") or {}))
    pal = ctx.get("palette"); Wv = float(pal["w"]) if pal else None                 # color by the plot's wealth (influences: the poor → the nobility)
    byw = set(K.get("by_wealth") or []) if Wv is not None else set()
    col = {k: _col(CC[k], rng, w=Wv if k in byw else None, sp=float(pal["spread"]) if pal else 0.0) for k in ("wall", "roof", "plinth", "finish", "wood") if k in CC}
    col["metal"] = _col(CC.get("metal", [0.26, 0.26, 0.28]), rng)
    if fach: col["wall_top"] = _col([[0.9, 0.87, 0.8], [0.86, 0.82, 0.72], [0.92, 0.9, 0.84]], rng, 0.03)
    t = float(ctx.get("tone") or 0.0)                                              # influences: richer — lighter and cleaner, poorer — darker
    if t:
        for k_, f_ in (("wall", 1.0), ("wall_top", 1.0), ("plinth", 0.6), ("roof", 0.5)):
            if k_ in col: col[k_] = [round(float(min(1.0, max(0.0, v + (1 - v) * t * f_ if t > 0 else v * (1 + t * f_)))), 4) for v in col[k_]]
    prog = {"kind": kind, "plot": plot.get("id"), "F": [[round(float(x), 3), round(float(y), 3)] for x, y in C], "front": front,
            "court": court and _ccw(court), "court2": court2 and _ccw(court2), "split": split and round(float(split), 3),
            "floors": n, "h0": h0, "fh": fh, "z0": round(z0, 3), "zb": round(zb, 3), "t": float(K.get("wall", 0.45)),
            "roof": roof, "win": win, "door": door, "shop": shop, "jetty": jetty, "fach": fach, "chimney": chim, "balcony": balc,
            "pipes": pipes, "gallery": gal, "col": col, "seed": int(rng.integers(1 << 30)), "street": main.get("rank")}
    if K.get("makeup") == "wings":                                               # a house with wings around a yard (Greek)
        wp = _wings(K, F, C, prog, n_out, rng)
        if wp: prog.update(wp)
    out = [prog]
    # yard buildings: a back house (burgage), a shed (izba, rural)
    for key, sub in (("back", "burgage"), ("shed", "shed")):
        X = K.get(key)
        if not X: continue
        share = X["frac"] if not isinstance(X["frac"], list) else X["frac"][0] + (X["frac"][1] - X["frac"][0]) * float(ctx.get("center", 0.5))
        if rng.random() >= share: continue
        o, u, v, u0, u1, v0, v1 = _local(F, C[front], C[(front + 1) % len(C)])
        plot_rest = P.difference(F.buffer(1.2))
        if plot_rest.is_empty: continue
        rest = max(getattr(plot_rest, "geoms", [plot_rest]), key=lambda g: g.area)
        dep = _u(X.get("depth", [5, 8]), rng); wid = _u(X.get("width", [u1 - u0, u1 - u0]), rng) if key == "shed" else (u1 - u0)
        q0 = o + u * u0 + v * (v1 + 1.5); q1 = o + u * u1 + v * (v1 + 1.5)
        G = rect_on_front(rest, q0, q1, -v, 0.6, dep, wid if key == "shed" else None, min_depth=3.0)
        if G is None or G.area < 12: continue
        sk = "shed" if key == "shed" else kind
        sub_plot = {"poly": G, "fronts": [], "id": plot.get("id")}
        c2 = dict(ctx, storeys=X.get("storeys", [1, 1]))
        sp = program("shed", sub_plot, c2, rng) if sk == "shed" else _back(prog, G, X, rng, ctx)
        if sp: out += sp if isinstance(sp, list) else [sp]
    return out


def _adjust(K, ctx):
    """Ensemble corrections to a kind (influences): ctx `corrections` — a nested key replacement (None — remove), `shops` — the
    share of shops (for a Greek house — also of the street wing). → K (a copy, if modified)."""
    if not ctx.get("corrections") and ctx.get("shops") is None: return K
    K = json.loads(json.dumps(K))
    def merge(a, b):
        for k, v in b.items():
            if v is None: a.pop(k, None)
            elif isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
            else: a[k] = v
    merge(K, ctx.get("corrections") or {})
    x = ctx.get("shops")
    if x is not None:
        if K.get("shop"): K["shop"]["frac"] = float(x)
        if K.get("wings"): K["wings"]["street"]["shops"] = float(x); K["wings"]["street"]["frac"] = max(float(K["wings"]["street"]["frac"]), float(x))
    return K


def _strip(F, a, u, v, s0, s1, t0, t1):
    """The part of the convex outline F within the strip s ∈ [s0, s1], t ∈ [t0, t1] of the frame (a, u, v) → a counterclockwise
    polygon or None."""
    P = [tuple(p) for p in F]
    for A, Bq, c in ((-u[0], -u[1], -float(u @ a) - s0), (u[0], u[1], float(u @ a) + s1), (-v[0], -v[1], -float(v @ a) - t0), (v[0], v[1], float(v @ a) + t1)):
        P = _clip(P, A, Bq, c)
        if len(P) < 3: return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in _ccw(P)] if abs(_area(P)) > 1.0 else None


def _wings(K, F, C, prog, n_out, rng):
    """A house with wings around a yard (Olynthus, Priene): the main wing along the north edge (1–2 storeys, a portico into the
    yard), the sides, the street wing with shops — at random; the yard no smaller than the measure (otherwise the wings are
    dropped: first those not at the street). → program fields or None."""
    W_ = K["wings"]; Cl = [tuple(map(float, p)) for p in C]; nE = len(Cl)
    iN = int(np.argmax([float(-_inward(Cl, i)[1]) for i in range(nE)]))
    a = np.asarray(Cl[iN]); b = np.asarray(Cl[(iN + 1) % nE]); u = (b - a) / max(np.linalg.norm(b - a), 1e-9); v = _inward(Cl, iN)
    S = (np.asarray(Cl) - a) @ u; T = (np.asarray(Cl) - a) @ v; s0, s1, T1 = float(S.min()), float(S.max()), float(T.max())
    M = W_["main"]; dB = _u(M["depth"], rng); dP = _u(M["portico"], rng); cmin, cshare = W_.get("court_min", [4.5, 0.18])
    if T1 - dB - dP < cmin: dP = max(0.0, T1 - dB - cmin)
    if T1 - dB - dP < cmin or s1 - s0 < cmin + 1.0: return None
    dM = dB + dP; area = abs(_area(Cl))
    no = np.asarray(n_out, float); side = "NSWE"[int(np.argmax([float(-v @ no), float(v @ no), float(-u @ no), float(u @ no)]))]
    have = {"S": rng.random() < (W_["street"]["frac"] if side == "S" else 0.2), "W": rng.random() < (0.85 if side == "W" else W_["flank"]["frac"]),
            "E": rng.random() < (0.85 if side == "E" else W_["flank"]["frac"])}
    d = {"S": _u(W_["street"]["depth"], rng), "W": _u(W_["flank"]["depth"], rng), "E": _u(W_["flank"]["depth"], rng)}
    box = lambda: (s0 + (d["W"] if have["W"] else 0.0), s1 - (d["E"] if have["E"] else 0.0), dM, T1 - (d["S"] if have["S"] else 0.0))
    for k in [x for x in "EWS" if x != side] + [side]:
        c0, c1, t0, t1 = box()
        if c1 - c0 >= cmin and t1 - t0 >= cmin and (c1 - c0) * (t1 - t0) >= cshare * area: break
        have[k] = False
    c0, c1, t0, t1 = box()
    if c1 - c0 < cmin or t1 - t0 < cmin: return None
    n = prog["floors"]; h = _u(M["height"], rng); gable = rng.random() < float(M.get("gable", 0.0)); wings = []
    st = lambda *q: _strip(Cl, a, u, v, *q)
    if n >= 2:
        wings.append({"P": st(s0 - 1, s1 + 1, -1, dB), "role": "main", "floors": n, "h0": h, "gable": gable})
        if dP > 0.5: wings.append({"P": st(s0 - 1, s1 + 1, dB, dM), "role": "portico", "floors": 1, "h0": h - 0.3, "open": True})
    else:
        wings.append({"P": st(s0 - 1, s1 + 1, -1, dM), "role": "main", "floors": 1, "h0": h, "gable": gable, "open": dP > 0.5, "inner": round(dB, 3)})
    for k, ss, ts in (("W", (s0 - 1, s0 + d["W"]), (dM, t1)), ("E", (s1 - d["E"], s1 + 1), (dM, t1)), ("S", (s0 - 1, s1 + 1), (t1, T1 + 1))):
        if not have[k]: continue
        Wk = W_["street"] if k == "S" else W_["flank"]
        wings.append({"P": st(*ss, *ts), "role": {"W": "west", "E": "east", "S": "south"}[k], "floors": 1, "h0": _u(Wk["height"], rng),
                      "shops": bool(k == side and rng.random() < float(W_["street"].get("shops", 0.0)))})
    wings = [w for w in wings if w["P"]]
    court = st(c0, c1, t0, t1)
    return {"wings": wings, "court": court, "fence_h": round(_u(W_.get("fence", [3.0, 3.0]), rng), 2), "street_side": side,
            "frame": {"a": [float(a[0]), float(a[1])], "u": [float(u[0]), float(u[1])], "v": [float(v[0]), float(v[1])], "s": [round(c0, 3), round(c1, 3)], "dB": round(dB, 3), "dM": round(dM, 3)},
            "h0": round(h, 3)}


def _back(prog, G, X, rng, ctx):
    """Back house of a burgage plot: smaller and lower, a mono-pitch roof sloping away from the neighbor."""
    q = json.loads(json.dumps(prog)); C = np.asarray(_poly_ccw(list(G.exterior.coords)[:-1]).exterior.coords)[:-1]
    n = int(rng.integers(int(X["storeys"][0]), int(X["storeys"][1]) + 1))
    q.update(F=[[round(float(x), 3), round(float(y), 3)] for x, y in C], front=0, court=None, court2=None, split=None, floors=n, jetty=0.0,
             shop=None, balcony=None, chimney=0, seed=int(rng.integers(1 << 30)), street=None, kind="back")
    q["roof"] = {"kind": "pitch", "pitch": 28.0, "ov": 0.4, "eaves": [i == 0 for i in range(len(C))], "parapet": 0.0}
    h = ctx.get("height")
    if h is not None:
        zs = [float(h(x, y)) for x, y in C]; q["z0"] = round(max(zs) + 0.1, 3); q["zb"] = round(min(zs) - 0.3, 3)
    return q


def edges(progs, ribbon, tol=0.7):
    """Classes of each house's outline edges: street (the rank of the nearest one), neighbor (a shared wall — no windows), yard
    (by a yard or garden). Writes prog["edge"] and prog["door_edge"] (the door — on the facade edge, otherwise on any street
    edge)."""
    import shapely
    from shapely.geometry import LineString, Polygon
    polys = [Polygon(p["F"]) for p in progs]; tree = shapely.STRtree(polys)
    rb = ribbon.boundary if not ribbon.is_empty else None
    for k, p in enumerate(progs):
        C = np.asarray(p["F"]); cls = []
        for i in range(len(C)):
            a, b = C[i], C[(i + 1) % len(C)]; seg = LineString([a, b]); m = seg.interpolate(0.5, normalized=True)
            c = "court"
            if rb is not None and m.distance(ribbon) < tol: c = "street"
            else:
                near = [j for j in tree.query(seg.buffer(0.45)) if j != k and polys[j].distance(m) < 0.45]
                if near: c = "neighbor"
            cls.append(c)
        p["edge"] = cls
        fe = p["front"]
        p["door_edge"] = fe if cls[fe] == "street" else next((i for i, c in enumerate(cls) if c == "street"), fe)
    return progs


# ---------------------------------------------------------------- geometry (numpy; Blender and project Python)
def _clip(P, a, b, c):
    """Sutherland–Hodgman: the part of a convex polygon where a·x + b·y ≤ c."""
    out = []; n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]
        fp = a * p[0] + b * p[1] - c; fq = a * q[0] + b * q[1] - c
        if fp <= 1e-9: out.append(p)
        if (fp < -1e-9 < fq) or (fq < -1e-9 < fp):
            t = fp / (fp - fq); out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(P):
    return 0.5 * sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))


def _inward(P, i):
    """Inward normal of edge i of a counterclockwise polygon."""
    a, b = np.asarray(P[i], float), np.asarray(P[(i + 1) % len(P)], float); d = b - a; d /= max(np.linalg.norm(d), 1e-9)
    return np.array([-d[1], d[0]])


def _offset(P, ds):
    """A convex polygon with the edges shifted outward by ds[i] (an intersection of half-planes)."""
    big = 1e4; Q = [(-big, -big), (big, -big), (big, big), (-big, big)]
    for i in range(len(P)):
        n = _inward(P, i); a = np.asarray(P[i], float) - n * ds[i]               # n·p ≥ n·a  ⇔  −n·p ≤ −n·a
        Q = _clip(Q, -n[0], -n[1], -float(n @ a))
    return Q


class Roof:
    """Roof slopes as planes z = g0 + gx·x + gy·y: outer (from the eaves inward) and yard ones (from the yard edges outward).
    z(p) = min(outer) | max(yard) | min(min of outer, max of yard) — hip and gable, slopes toward the yard, a ring."""

    def __init__(self, F, eaves, court, z1, pitch, kind):
        t = math.tan(math.radians(pitch)); self.kind = kind; self.z1 = z1; self.A = []; self.Cp = []; cap = None
        if kind != "flat":
            for i, e in enumerate(eaves):
                if e:
                    n = _inward(F, i); a = np.asarray(F[i], float); self.A.append((t * n[0], t * n[1], z1 - t * float(n @ a)))
            if court is not None:
                for j in range(len(court)):
                    m = -_inward(court, j); b = np.asarray(court[j], float); self.Cp.append((t * m[0], t * m[1], z1 - t * float(m @ b)))
        if kind == "inward":                                                       # slopes toward the yard: the top is not higher than the ridge of the narrowest wing
            ws = []                                                                # (otherwise the wall tops of different wings would climb toward the corners)
            for j in range(len(court or [])):
                m = -_inward(court, j); b = np.asarray(court[j], float); w = max(float((np.asarray(p, float) - b) @ m) for p in F)
                if w > 1.5: ws.append(w)
            self.A = [(0.0, 0.0, z1 + t * min(ws))] if ws else []

    @staticmethod
    def ev(g, x, y): return g[2] + g[0] * x + g[1] * y

    def z(self, x, y):
        if not self.A and not self.Cp: return self.z1
        a = min(self.ev(g, x, y) for g in self.A) if self.A else None
        c = max(self.ev(g, x, y) for g in self.Cp) if self.Cp else None
        if a is None: return c
        if c is None: return a
        return min(a, c)

    def faces(self, Fo, hole_d):
        """Roof faces over the outline with the overhang Fo: [(xy polygon, plane)]; the yard hole is deeper than hole_d from the edges."""
        out = []
        le = lambda g, h: (g[0] - h[0], g[1] - h[1], h[2] - g[2])                # g ≤ h
        if not self.A and not self.Cp: return [(Fo, (0.0, 0.0, self.z1))]
        if self.Cp:
            for j, g in enumerate(self.Cp):                                        # yard ones: g = max of the yard planes, ≤ all outer ones, outside the hole
                Q = list(Fo)
                for l, h in enumerate(self.Cp):
                    if l != j: Q = _clip(Q, *le(h, g))
                for h in self.A: Q = _clip(Q, *le(g, h))
                Q = _clip(Q, *le((0.0, 0.0, self.z1 - self.slope_hole(hole_d)), g))
                if len(Q) >= 3 and abs(_area(Q)) > 1e-4: out.append((Q, g))
        for i, g in enumerate(self.A):
            if self.Cp:
                for j, h in enumerate(self.Cp):                                    # outer g is the minimum of the outer planes, h the maximum of the yard ones, g ≤ h
                    Q = list(Fo)
                    for k, f in enumerate(self.A):
                        if k != i: Q = _clip(Q, *le(g, f))
                    for l, f in enumerate(self.Cp):
                        if l != j: Q = _clip(Q, *le(f, h))
                    Q = _clip(Q, *le(g, h))
                    if len(Q) >= 3 and abs(_area(Q)) > 1e-4: out.append((Q, g))
            else:
                Q = list(Fo)
                for k, f in enumerate(self.A):
                    if k != i: Q = _clip(Q, *le(g, f))
                if len(Q) >= 3 and abs(_area(Q)) > 1e-4: out.append((Q, g))
        return out

    def slope_hole(self, d):
        """Height at the edge of the yard hole: an overhang of d over the yard lowers the eave by d·tg."""
        g = self.Cp[0]; return math.hypot(g[0], g[1]) * d


def profile(roof, a, b, n=48):
    """Top of the wall along edge a → b under the roof: [(s, z)] from 0 to L (the kinks are kept)."""
    a = np.asarray(a, float); b = np.asarray(b, float); L = float(np.linalg.norm(b - a))
    S = np.linspace(0, L, n + 1); Z = [roof.z(*(a + (b - a) * (s / max(L, 1e-9)))) for s in S]
    keep = [0]
    for i in range(1, n):
        z_pred = Z[keep[-1]] + (Z[i + 1] - Z[keep[-1]]) * (S[i] - S[keep[-1]]) / max(S[i + 1] - S[keep[-1]], 1e-9)
        if abs(Z[i] - z_pred) > 0.01: keep.append(i)
    keep.append(n)
    return [(float(S[i]), float(Z[i])) for i in keep]


def _spots(L, bay, w, margin=0.75):
    """Opening centers at step bay along a wall of length L, with margins at the corners."""
    if bay <= 0 or L < w + 2 * margin: return []
    k = int((L - 2 * margin) // bay)
    if k <= 0: return [L / 2] if L > w + 2 * margin else []
    return [margin + (L - 2 * margin) * (i + 0.5) / k for i in range(k)]


def _inside(P, x, y):
    """Point in polygon (ray casting)."""
    ins = False; n = len(P)
    for i in range(n):
        (x1, y1), (x2, y2) = P[i], P[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1): ins = not ins
    return ins


def _wings_geometry(prog):
    """A house with wings: each wing has its own roof sloping toward the yard (the main one is sometimes gabled), walls by edge
    classes (on the outline — the house's edge class; toward the yard — doors and windows; toward the neighboring wing —
    blank), a portico — columns and a beam instead of a wall, with an inner wall behind it; where there is no wing — a yard
    wall with a door, if it is a street."""
    rng = np.random.default_rng(prog["seed"]); F = [tuple(p) for p in prog["F"]]; nE = len(F)
    z0, zb, t = prog["z0"], prog["zb"], prog["t"]; col = prog["col"]; R = prog["roof"]; Wn = prog["win"]; fh = prog["fh"]
    cls = prog.get("edge") or ["street" if i == prog["front"] else "court" for i in range(nE)]; door_e = prog.get("door_edge", prog["front"])
    court = prog.get("court"); walls = []; faces = []; boxes = []; cols = []; covered = {k: [] for k in range(nE)}; door_done = False; ztop = z0
    for Wg in prog["wings"]:
        P = [tuple(p) for p in Wg["P"]]; m = len(P); n = Wg["floors"]; h0 = Wg["h0"]
        zf = [z0] + [z0 + h0 + i * fh for i in range(n - 1)]; z1 = z0 + h0 + (n - 1) * fh; ztop = max(ztop, z1)
        ec = []; to_court = []
        for i in range(m):
            A, Bq = np.asarray(P[i], float), np.asarray(P[(i + 1) % m], float); k = _match_edge(F, A, Bq)
            if k is not None:
                f0 = np.asarray(F[k], float); dk = np.asarray(F[(k + 1) % nE], float) - f0; Lk = float(np.linalg.norm(dk)); dk /= max(Lk, 1e-9)
                covered[k].append(sorted((float((A - f0) @ dk), float((Bq - f0) @ dk)))); ec.append(cls[k])
            else:
                q = (A + Bq) / 2 - _inward(P, i) * 0.8
                if court and _inside(court, q[0], q[1]): ec.append("inner"); to_court.append(i)
                else: ec.append("neighbor")
        eaves = [i in to_court for i in range(m)]
        if Wg.get("gable") and to_court:                                           # gabled: the opposite edge is an eave too
            nc = -_inward(P, to_court[0]); opp = int(np.argmin([float(-_inward(P, i) @ nc) for i in range(m)])); eaves[opp] = True
        if not any(eaves): eaves = [True] * m
        rk = "gable" if Wg.get("gable") else "pitch"; rf = Roof(P, eaves, None, z1, R["pitch"], rk)
        ds = [R["ov"] if eaves[i] else (0.12 if ec[i] == "street" else 0.0) for i in range(m)]
        for Q, g in rf.faces(_offset(P, ds), R["ov"]):
            faces.append({"pts": [(x, y, Roof.ev(g, x, y)) for x, y in Q], "col": col["roof"], "role": "roof", "band": 0.14})
        V = {"floors": list(range(n)), "lo": zb, "hi": z1, "roof": True}; pr = prog if Wg.get("shops") else dict(prog, shop=None)
        for i in range(m):
            A, Bq = np.asarray(P[i], float), np.asarray(P[(i + 1) % m], float); L = float(np.linalg.norm(Bq - A))
            if L < 0.3: continue
            if Wg.get("open") and i in to_court: continue                         # portico — columns instead of a wall
            if Wg["role"] == "portico" and ec[i] == "neighbor": continue                # the wall behind the portico — at the main wing
            nr = _inward(P, i); top = profile(rf, A, Bq)
            k = _match_edge(F, A, Bq); is_door = k is not None and k == door_e and not door_done
            holes = _openings(pr, ec[i], is_door, L, V, zf, Wn, rng, top)
            if is_door and any(hh[5] == "wood" for hh in holes): door_done = True
            walls.append({"p0": (A + nr * t / 2).tolist(), "p1": (Bq + nr * t / 2).tolist(), "n": (-nr).tolist(), "z0": zb, "top": top, "t": t,
                          "holes": holes, "col": col["wall"], "role": "wall", "fach": [], "zf": zf, "fh": fh, "h0": h0, "edge": ec[i]})
            if zb < z0 - 0.05 and ec[i] != "neighbor":
                boxes.append({"seg": [A.tolist(), Bq.tolist()], "nout": (-nr).tolist(), "z0": zb, "z1": z0, "d": 0.06, "col": col.get("plinth", col["wall"]), "role": "finish"})
        if Wg.get("open"):                                                         # portico columns and the beam; portico floors
            for i in to_court:
                A, Bq = np.asarray(P[i], float), np.asarray(P[(i + 1) % m], float); L = float(np.linalg.norm(Bq - A)); dd = (Bq - A) / max(L, 1e-9); nr = _inward(P, i)
                for s_ in _spots(L, 2.6, 0.4, 0.9): cols.append({"c": (A + dd * s_ + nr * 0.35).tolist(), "r": 0.19, "z0": z0, "z1": z1 - 0.28})
                boxes.append({"bar": [(*(A + nr * 0.35), z1 - 0.14), (*(Bq + nr * 0.35), z1 - 0.14)], "w": 0.28, "col": col.get("finish", col["wall"]), "role": "finish"})
            fr = prog["frame"]; a_ = np.asarray(fr["a"]); u_ = np.asarray(fr["u"]); v_ = np.asarray(fr["v"]); c0, c1 = fr["s"]
            if Wg.get("inner") is not None:                                        # inner wall behind the portico (a single-storey main wing)
                ti = Wg["inner"]; p0 = a_ + u_ * c1 + v_ * (ti - t / 2); p1 = a_ + u_ * c0 + v_ * (ti - t / 2); L = float(np.linalg.norm(p1 - p0))
                top = profile(rf, p0 + v_ * t / 2, p1 + v_ * t / 2)
                holes = _openings(dict(prog, shop=None), "inner", False, L, V, zf, Wn, rng, top)
                walls.append({"p0": p0.tolist(), "p1": p1.tolist(), "n": v_.tolist(), "z0": z0 - 0.02, "top": top, "t": t, "holes": holes, "col": col["wall"],
                              "role": "wall", "fach": [], "zf": zf, "fh": fh, "h0": h0, "edge": "inner"})
            q0, q1 = a_ + u_ * c0 + v_ * fr["dB"], a_ + u_ * c1 + v_ * fr["dB"]
            faces.append({"loops": [_ccw([q0.tolist(), q1.tolist(), (q1 + v_ * (fr["dM"] - fr["dB"])).tolist(), (q0 + v_ * (fr["dM"] - fr["dB"])).tolist()])],
                          "z": z0 + 0.02, "col": col.get("plinth", col["wall"]), "role": "floor"})
    hf = prog.get("fence_h", 3.0)                                                  # yard wall — where there is no wing
    for k in range(nE):
        f0 = np.asarray(F[k], float); f1 = np.asarray(F[(k + 1) % nE], float); Lk = float(np.linalg.norm(f1 - f0)); dk = (f1 - f0) / max(Lk, 1e-9); nr = _inward(F, k)
        iv = sorted(covered[k]); gaps = []; cur = 0.0
        for x0_, x1_ in iv:
            if x0_ > cur + 0.6: gaps.append((cur, x0_))
            cur = max(cur, x1_)
        if Lk > cur + 0.6: gaps.append((cur, Lk))
        for g0, g1 in gaps:
            L = g1 - g0; p0 = f0 + dk * g0 + nr * 0.2; p1 = f0 + dk * g1 + nr * 0.2; holes = []
            if k == door_e and not door_done and L > prog["door"]["w"] + 1.2:
                holes = [(L / 2, z0, prog["door"]["w"], min(prog["door"]["h"], hf - 0.4), prog["door"]["kind"], "wood")]; door_done = True
            walls.append({"p0": p0.tolist(), "p1": p1.tolist(), "n": (-nr).tolist(), "z0": zb, "top": [(0.0, z0 + hf), (L, z0 + hf)], "t": 0.4, "holes": holes,
                          "col": col["wall"], "role": "wall", "fach": [], "zf": [z0], "fh": fh, "h0": hf, "edge": cls[k]})
            boxes.append({"box": [((p0 + p1) / 2).tolist(), dk.tolist(), 0.52, L, z0 + hf, z0 + hf + 0.12], "col": col.get("finish", col["wall"]), "role": "finish"})
    if court: faces.append({"loops": [_ccw(court)], "z": z0 - 0.01, "col": col.get("plinth", col["wall"]), "role": "floor"})
    return {"walls": walls, "faces": faces, "boxes": boxes, "tubes": [], "columns": cols, "z1": ztop}


def geometry(prog):
    """Program → {"walls": [...], "faces": [...], "boxes": [...], "tubes": [...]} (world, z up).
    A wall: p0, p1 (the line of the outer face), n (outward), z0, top [(s, z)], t, holes [(s, z0, w, h, kind, role)], col, role."""
    if prog.get("wings"): return _wings_geometry(prog)
    rng = np.random.default_rng(prog["seed"]); F = [tuple(p) for p in prog["F"]]; nE = len(F)
    n = prog["floors"]; h0, fh = prog["h0"], prog["fh"]; z0, zb = prog["z0"], prog["zb"]; t = prog["t"]
    zf = [z0] + [z0 + h0 + i * fh for i in range(n - 1)]; z1 = z0 + h0 + (n - 1) * fh
    R = prog["roof"]; cls = prog.get("edge") or ["street" if i == prog["front"] else "court" for i in range(nE)]
    col = prog["col"]; walls = []; faces = []; boxes = []; tubes = []
    W = prog["win"]; door_e = prog.get("door_edge", prog["front"])
    fach_floors = set(range(1, n)) if prog.get("fach") else set()
    jet = prog.get("jetty", 0.0) or 0.0
    volumes = []
    if prog.get("split") is not None and prog.get("court2"):                     # domus: two parts, each with its own yard
        f0 = np.asarray(F[prog["front"]]); f1 = np.asarray(F[(prog["front"] + 1) % nE]); u = (f1 - f0) / np.linalg.norm(f1 - f0); v = np.array([-u[1], u[0]])
        if np.mean([(np.asarray(p) - f0) @ v for p in F]) < 0: v = -v
        cut = float(prog["split"]); A = _clip(F, *(v[0], v[1], float(v @ f0) + cut)); B = _clip(F, *(-v[0], -v[1], -(float(v @ f0) + cut)))
        for P_, ct in ((A, prog["court"]), (B, prog["court2"])):
            if len(P_) >= 3: volumes.append({"P": _ccw(P_), "court": ct, "lo": zb, "hi": z1, "floors": list(range(n)), "roof": True, "cut": (v, float(v @ f0) + cut)})
    elif jet > 0 and n >= 2:                                                       # storey jetty over the street (a single step)
        fe = prog["front"]; nr = _inward(F, fe)
        F2 = [tuple(np.asarray(p) - nr * jet) if i in (fe, (fe + 1) % nE) else p for i, p in enumerate(F)]
        volumes.append({"P": F, "court": None, "lo": zb, "hi": z0 + h0, "floors": [0], "roof": False})
        volumes.append({"P": F2, "court": None, "lo": z0 + h0, "hi": z1, "floors": list(range(1, n)), "roof": True, "jetty_from": F})
    else:
        volumes.append({"P": F, "court": prog.get("court"), "lo": zb, "hi": z1, "floors": list(range(n)), "roof": True})
    for V in volumes:
        P = V["P"]; m = len(P); court = V.get("court")
        eaves = R["eaves"] if len(R["eaves"]) == m else [True] * m
        rf = Roof(P, eaves, court, V["hi"], R["pitch"], R["kind"]) if V["roof"] else None
        rkind = R["kind"] if V["roof"] else "flat"
        # walls
        for i in range(m):
            a, b = np.asarray(P[i], float), np.asarray(P[(i + 1) % m], float); L = float(np.linalg.norm(b - a))
            if L < 0.3: continue
            nr = _inward(P, i); nout = -nr
            k_e = _match_edge(F, a, b); c = cls[k_e] if k_e is not None else "court"
            if V.get("cut") is not None:                                           # the seam between the domus parts — a blank wall
                cv, cc = V["cut"]
                if abs(float(cv @ a) - cc) < 1e-3 and abs(float(cv @ b) - cc) < 1e-3: c = "neighbor"
            if V["roof"] and rf is not None and rkind != "flat": top = profile(rf, a, b)
            else: top = [(0.0, V["hi"] + (R.get("parapet", 0.0) if V["roof"] else 0.0)), (L, V["hi"] + (R.get("parapet", 0.0) if V["roof"] else 0.0))]
            holes = _openings(prog, c, k_e == door_e, L, V, zf, W, rng, top)
            walls.append({"p0": (a + nr * t / 2).tolist(), "p1": (b + nr * t / 2).tolist(), "n": nout.tolist(), "z0": V["lo"], "top": top, "t": t,
                          "holes": holes, "col": col["wall"], "role": "wall", "fach": [f for f in V["floors"] if f in fach_floors],
                          "zf": zf, "fh": fh, "h0": h0, "edge": c})
            if V["lo"] < z0 - 0.05 and V["lo"] == zb:                              # plinth
                boxes.append({"seg": [a.tolist(), b.tolist()], "nout": nout.tolist(), "z0": zb, "z1": z0, "d": 0.06, "col": col.get("plinth", col["wall"]), "role": "finish"})
        if court is not None:
            cm = len(court)
            for j in range(cm):
                a, b = np.asarray(court[j], float), np.asarray(court[(j + 1) % cm], float); L = float(np.linalg.norm(b - a))
                nin = _inward(court, j)                                            # toward the inside of the yard; the wall is outside the yard
                holes = _openings(prog, "inner", False, L, V, zf, W, rng, [(0.0, V["hi"]), (L, V["hi"])])
                walls.append({"p0": (a - nin * t / 2).tolist(), "p1": (b - nin * t / 2).tolist(), "n": nin.tolist(), "z0": z0 - 0.02, "top": [(0.0, V["hi"]), (L, V["hi"])],
                              "t": t, "holes": holes, "col": col["wall"], "role": "wall", "fach": [], "zf": zf, "fh": fh, "h0": h0, "edge": "inner"})
            faces.append({"loops": [_ccw(court)], "z": z0 - 0.01, "col": col.get("plinth", col["wall"]), "role": "floor"})
        # roof
        if rf is not None:
            ov = R["ov"]; ds = [ov if (rkind in ("hip", "ring", "gable", "pitch") and eaves[i]) else (0.25 if rkind != "flat" else 0.0) for i in range(m)]
            if rkind == "inward": ds = [0.15] * m
            for i in range(m):
                k_e = _match_edge(F, P[i], P[(i + 1) % m])
                if k_e is not None and cls[k_e] == "neighbor": ds[i] = min(ds[i], 0.0)
            Fo = _offset(P, ds)
            if rkind == "flat":
                loops = [_ccw(P)] + ([_ccw(court)[::-1]] if court else [])
                faces.append({"loops": loops, "z": V["hi"], "col": col["roof"], "role": "roof"})
            else:
                for Q, g in rf.faces(Fo, ov):
                    faces.append({"pts": [(x, y, Roof.ev(g, x, y)) for x, y in Q], "col": col["roof"], "role": "roof", "band": 0.16})
        if V.get("jetty_from"):                                                    # underside of the jetty and the beam ends
            J = V["jetty_from"]; fe = prog["front"]; a0, a1 = np.asarray(J[fe]), np.asarray(J[(fe + 1) % nE]); b0, b1 = np.asarray(P[fe]), np.asarray(P[(fe + 1) % nE])
            faces.append({"pts": [(*a1, V["lo"]), (*a0, V["lo"]), (*b0, V["lo"]), (*b1, V["lo"])], "col": col.get("wood", col["wall"]), "role": "wood"})
            L = float(np.linalg.norm(a1 - a0)); k = max(2, int(L / 0.9))
            for s in np.linspace(0.3, L - 0.3, k):
                p = a0 + (a1 - a0) * s / L; q = p + (b0 - a0)
                boxes.append({"bar": [(*p, V["lo"] - 0.12), (*q, V["lo"] - 0.12)], "w": 0.18, "col": col.get("wood", col["wall"]), "role": "wood"})
    # projections: balcony, galleries, pipes, chimney
    if prog.get("balcony"):
        Bc = prog["balcony"]; f = min(Bc["floor"], n - 1); zz = zf[f]
        for i, c in enumerate(cls):
            if c != "street": continue
            a, b = np.asarray(F[i], float), np.asarray(F[(i + 1) % nE], float); nr = _inward(F, i)
            boxes.append({"slab": [a.tolist(), b.tolist()], "nout": (-nr).tolist(), "d": Bc["d"], "z": zz, "th": 0.16, "rail": 1.0, "col": col.get("wood", col["wall"]), "role": "wood"})
    if prog.get("gallery"):
        G = prog["gallery"]
        for f in range(G["every"], n, G["every"]):
            for i, c in enumerate(cls):
                if c == "neighbor" or rng.random() > G["share"]: continue
                a, b = np.asarray(F[i], float), np.asarray(F[(i + 1) % nE], float); nr = _inward(F, i)
                boxes.append({"slab": [a.tolist(), b.tolist()], "nout": (-nr).tolist(), "d": G["d"], "z": zf[f], "th": 0.25, "rail": 1.1, "col": col["metal"], "role": "metal"})
    if prog.get("pipes"):
        Pp = prog["pipes"]
        for i, c in enumerate(cls):
            if c == "neighbor": continue
            a, b = np.asarray(F[i], float), np.asarray(F[(i + 1) % nE], float); L = float(np.linalg.norm(b - a)); nr = _inward(F, i)
            for s in np.arange(Pp["step"] * (0.3 + 0.4 * rng.random()), L - 1.0, Pp["step"]):
                if rng.random() < 0.35: continue
                p = a + (b - a) * s / L - nr * (Pp["r"] + 0.12); r = Pp["r"] * (0.6 + 0.6 * rng.random())
                tubes.append({"pts": [(*p, zb), (*p, z1 + 1.2 + 2.5 * rng.random())], "r": r, "col": col["metal"], "role": "metal"})
    if prog.get("chimney") and volumes[-1]["roof"]:
        P = volumes[-1]["P"]; rf = Roof(P, R["eaves"] if len(R["eaves"]) == len(P) else [True] * len(P), volumes[-1].get("court"), volumes[-1]["hi"], R["pitch"], R["kind"])
        c = np.mean(np.asarray(P, float), 0); d = np.asarray(P[1], float) - np.asarray(P[0], float); d /= np.linalg.norm(d)
        p = c + d * (rng.random() - 0.5) * 0.5 * np.linalg.norm(np.asarray(P[1]) - np.asarray(P[0]))
        zr = rf.z(*p)
        boxes.append({"box": [p.tolist(), d.tolist(), 0.75, 0.75, zr - 1.0, zr + 1.1], "col": col.get("finish", col["wall"]), "role": "finish"})
    return {"walls": walls, "faces": faces, "boxes": boxes, "tubes": tubes, "z1": z1}


def _match_edge(F, a, b, tol=0.05):
    """The index of the edge of outline F on which segment a–b lies (a jetty, a domus part), or None."""
    a = np.asarray(a, float); b = np.asarray(b, float); m = (a + b) / 2
    for k in range(len(F)):
        p, q = np.asarray(F[k], float), np.asarray(F[(k + 1) % len(F)], float); d = q - p; L = np.linalg.norm(d)
        if L < 1e-6: continue
        dist = abs(d[0] * (m[1] - p[1]) - d[1] * (m[0] - p[0])) / L
        if dist < tol + 0.6 and -0.5 <= float((m - p) @ d) / L <= L + 0.5 and abs(float((b - a) @ d)) > 0.8 * np.linalg.norm(b - a) * L:
            return k
    return None


def _openings(prog, cls, is_door_edge, L, V, zf, W, rng, top):
    """Openings of an edge: (s, z0, width, height, kind, filler role). Neighbor — blank; street — a door, shops, windows at a
    rhythm; yard and rear — fewer; inner (the house's yard) — doors and windows."""
    if cls == "neighbor" or L < 1.2: return []
    out = []; ts = np.array([q[0] for q in top]); tz = np.array([q[1] for q in top])
    street = cls == "street"; pol = W.get("street", "yes")
    bay = W["bay"] * (1.0 if street or cls == "inner" else 1.5)
    for f in V["floors"]:
        z = zf[f]; h_fl = (prog["h0"] if f == 0 else prog["fh"])
        if z < V["lo"] - 1e-6 or z >= V["hi"] - 0.5: continue
        used = []
        if f == 0 and street:
            if is_door_edge:
                s = L / 2 + (rng.random() - 0.5) * max(0.0, L - 2 * prog["door"]["w"] - 2) * 0.6
                dh = min(prog["door"]["h"], h_fl - 0.35); out.append((s, z, prog["door"]["w"], dh, prog["door"]["kind"], "wood")); used.append((s, prog["door"]["w"]))
            if prog.get("shop"):
                S = prog["shop"]
                for s in _spots(L, S["bay"], S["w"], 0.6):
                    if any(abs(s - u) < (S["w"] + w) / 2 + 0.4 for u, w in used): continue
                    out.append((s, z, S["w"], min(S["h"], h_fl - 0.3), S["kind"], "opening")); used.append((s, S["w"]))
                continue
        if f == 0 and cls == "inner" and rng.random() < 0.7:
            s = L * (0.3 + 0.4 * rng.random()); out.append((s, z, 1.2, min(2.3, h_fl - 0.3), "rect", "opening")); used.append((s, 1.2))
        if street and pol == "blind": continue
        small = street and pol == "few"                                           # toward the street — sparse small windows under the ceiling
        b_ = bay * (2.0 if small else 1.0); ww = min(W["w"], 0.6) if small else W["w"]
        for s in _spots(L, b_, ww):
            if any(abs(s - u) < (ww + w) / 2 + 0.3 for u, w in used): continue
            sill = W["sill"] if f > 0 or not street else max(W["sill"], 0.9)
            if small: sill = max(sill, h_fl - 1.1)
            zz = z + sill; hh = min(0.55 if small else W["h"], h_fl - sill - 0.35)
            if hh < 0.4: continue
            if zz + hh > np.interp(s, ts, tz) - 0.3: continue
            out.append((s, zz, ww, hh, W["kind"], "glass"))
    # attic window in the gable
    if V["roof"]:
        zmax = float(np.max(tz)); s_ = float(ts[int(np.argmax(tz))])
        if zmax - V["hi"] > 2.6 and cls != "neighbor" and 1.5 < s_ < L - 1.5:
            hh = min(1.1, (zmax - V["hi"]) * 0.35); out.append((s_, V["hi"] + 0.35, 0.7, hh, "rect", "glass"))
    return out


# ---------------------------------------------------------------- one house on a site
def plan(place, out=None, kind="burgage", seed=1, at=None, wealth=None, trade=None, power=None, **kw):
    """One house on a site: a plot of 12 × 30 m at point at (facade to the south). wealth, trade, power (0–1) — the plot's
    wealth, trade, power: influence adjustments, as in a settlement. → house.json in out."""
    from shapely.geometry import Polygon
    from core.surface.ground import Ground
    G = Ground(place); rng = np.random.default_rng(int(seed))
    if isinstance(at, str): at = [float(v) for v in at.split(",")]
    c = np.asarray(at if at is not None else G.loc.get("center", [0, 0]), float)
    P = Polygon([c + (-6, -15), c + (6, -15), c + (6, 15), c + (-6, 15)])
    plot = {"poly": P, "id": 0, "fronts": [{"p0": list(c + (-6, -15)), "p1": list(c + (6, -15)), "rank": "street", "n": [0, -1], "len": 12.0}]}
    ctx = {"height": G.height, "storeys": [2, 3]}; inf = None
    if any(v is not None for v in (wealth, trade, power)):                         # the plot's three numbers → adjustments of the "influences" core
        from components.buildings.influences import influence as INF
        inf = [float(power or 0.0), float(trade or 0.0), float(wealth if wealth is not None else 0.5)]
        ctx.update(INF.place(*inf)); ctx["corrections"] = INF.cheap(kinds()[kind], inf[2])
    progs = program(kind, plot, ctx, rng)
    from shapely.geometry import box
    edges(progs, box(c[0] - 30, c[1] - 22, c[0] + 30, c[1] - 15))
    out = out or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "out", "house", f"{kind}_{seed}")
    os.makedirs(out, exist_ok=True)
    R = {"type": "house", "place": place, "spec": {"kind": kind, "seed": int(seed), **({"influence": inf} if inf else {})}, "houses": progs, "checks": [],
         "stats": {"n_houses": len(progs)}, "out": out}
    json.dump(R, open(os.path.join(out, "house.json"), "w"), ensure_ascii=False, indent=1)
    import shutil
    for f in ("relief.npz", "location.json"): shutil.copy(os.path.join(place, f), os.path.join(out, f))
    return R
