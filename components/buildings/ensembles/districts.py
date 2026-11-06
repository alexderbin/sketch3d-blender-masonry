"""Districts in the settlement plan: compounds of the power centers and the district map. The fields of the three
influences and the adjustments are part of the "influences" core (../influences/influence.py); here — how power lays
onto the plots: the center's core is a compound, its kind is drawn at random (tower, yard — arena, parade ground,
barracks, chambers), the buildings are of the archetype's kind via the entry (the "house" type)."""
import os
import sys
from components.buildings import arch_parts
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from components.buildings.ensembles import landmarks as LM
from components.buildings.ensembles import plots as PL
from components.buildings.influences.field import COLORS, LABELS


def _parts(g, a=1.0):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= a] if not g.is_empty else []


# ---------------------------------------------------------------- compounds of the power centers
def compounds(plots, IF, deck, touch=0.6):
    """Core of a power center → compound: the nearest dense plot no smaller than the average one, plus those touching it (by
    distance), until the area reaches max(π·(core·r)², `n_plots` average plots of the settlement) (in a hive a plot is a
    block: the compound is a block, not a sliver), and the compound is dense (area to convex hull no less than `density`:
    L-shaped strips around a block do not count).
    → (plots without the merged ones + compounds, compounds)."""
    out = list(plots); made = []
    a_med = float(np.median([q["poly"].area for q in plots])) if plots else 0.0
    for ci, ce in enumerate(IF.centres):
        if ce.get("landmark") or ce.get("deck") != deck: continue
        p = Point(ce["p"]); free = sorted([q for q in out if q.get("compound") is None and not q.get("at_walls")], key=lambda q: q["poly"].distance(p))
        reach = max(ce["r"], 20.0); kd = float(IF.K["power"]["density"])
        dense = lambda g_: g_.area / max(g_.convex_hull.area, 1e-6) >= kd
        close = lambda g_: g_.buffer(touch, join_style="mitre").buffer(-touch, join_style="mitre")
        near = [q for q in free if q["poly"].distance(p) <= reach and dense(q["poly"])]   # the first one is dense (not an L-shaped strip), the nearest
        seed = next((q for q in near if q["poly"].area >= a_med), near[0] if near else None)   # no smaller than the average plot (hive: a block, not a sliver)
        if seed is None: continue
        target = max(math.pi * (IF.K["power"]["core"] * ce["r"]) ** 2, float(IF.K["power"]["n_plots"]) * a_med); grp = [seed]; g = seed["poly"]
        for q in free:
            if q is seed: continue
            if g.area >= target or q["poly"].distance(p) > 1.3 * ce["r"]: break
            if q["poly"].distance(g) < touch:
                g2 = unary_union([g, q["poly"]])
                if dense(close(g2)): grp.append(q); g = g2
        g = close(g).buffer(0)                                                      # close the gaps between plots; make the outline valid (GEOS)
        P = _parts(g, 20.0)
        if not P: continue
        ids = {id(q) for q in grp}; out = [q for q in out if id(q) not in ids]
        cp = {"poly": max(P, key=lambda x: x.area), "compound": ci, "kind": ce["kind"]}; out.append(cp); made.append(cp)
    return out, made


def _ring(A, depth, rng):
    """Buildings along the edge of the compound (a yard in the middle): strips along the outline, the facade faces the yard."""
    C = np.asarray(A.exterior.coords); band = [{"pts": [C[i].tolist(), C[i + 1].tolist()], "w": 0.2, "rank": "passage", "kind": "wall"}
                                               for i in range(len(C) - 1) if np.linalg.norm(C[i + 1] - C[i]) > 4.0]
    got, _ = PL.strips(A, band, [9.0, 14.0], [depth * 0.85, depth], rng, empty=0.0, min_share=0.5)
    for q in got:
        Q = np.asarray(q["poly"].exterior.coords); best = None
        for i in range(len(Q) - 1):
            a, b = Q[i], Q[i + 1]; L = float(np.linalg.norm(b - a))
            if L < 2.0: continue
            d = A.exterior.distance(Point((a + b) / 2))
            if best is None or d > best[0]: best = (d, a, b, L)
        if best is None: q["fronts"] = []; continue
        _, a, b, L = best; t = (b - a) / L; n = np.array([t[1], -t[0]])
        if q["poly"].contains(Point((a + b) / 2 + n * 0.3)): n = -n
        q["fronts"] = [{"p0": [round(float(a[0]), 3), round(float(a[1]), 3)], "p1": [round(float(b[0]), 3), round(float(b[1]), 3)],
                        "rank": "passage", "w": 0.0, "n": [round(float(n[0]), 4), round(float(n[1]), 4)], "len": round(L, 2), "flow": 0.0}]
    return got


def _tower(A, side):
    """Tower square at the back of the compound (pole of inaccessibility), on the compound axis; if it does not fit — smaller
    (not less than 5 m)."""
    from shapely.ops import polylabel
    c = np.asarray(polylabel(A, 0.5).coords[0]); R_ = np.asarray(A.minimum_rotated_rectangle.exterior.coords)
    u = R_[1] - R_[0]; u = u / max(np.linalg.norm(u), 1e-9); v = np.array([-u[1], u[0]]); a = side
    while a >= 5.0:
        h = a / 2; Q = Polygon([c - u * h - v * h, c + u * h - v * h, c + u * h + v * h, c - u * h + v * h])
        if A.buffer(-0.4).contains(Q): return Q
        a -= 0.8
    return None


RANKS = ("artery", "street", "lane", "passage")


def _palace(HM, kind, plot, ctx, Kp, rng):
    """Chambers at the back of the compound, with a ceremonial yard in front of them toward the main street (a palazzo,
    "between court and garden"): a cut along the facade line at a fraction of the depth; the chambers take the whole
    remainder. → (programs, yard) or (None, None)."""
    try: return _palace_(HM, kind, plot, ctx, Kp, rng)
    except shapely.errors.GEOSException: return None, None                       # the outline is at the edge of precision — chambers by another route


def _palace_(HM, kind, plot, ctx, Kp, rng):
    A = plot["poly"].buffer(0); fr = sorted(plot.get("fronts") or [], key=lambda f: (RANKS.index(f["rank"]) if f["rank"] in RANKS else 9, -f["len"]))
    if not fr: return None, None
    f0 = fr[0]; p0 = np.asarray(f0["p0"], float); p1 = np.asarray(f0["p1"], float); n = np.asarray(f0["n"], float)   # n — toward the street
    t = p1 - p0; t /= max(np.linalg.norm(t), 1e-9); C = np.asarray(A.exterior.coords)
    dep = float(((C - p0) @ -n).max()); dc = float(rng.uniform(*Kp["court"])) * dep
    if dep - dc < 8.0: return None, None
    m = p0 - n * dc; big = 1e3
    back = Polygon([m - t * big, m + t * big, m + t * big - n * big, m - t * big - n * big])
    body = _parts(A.intersection(back), 60.0); court = _parts(A.difference(back), 40.0)
    if not body or not court: return None, None
    body = max(body, key=lambda g: g.area); cut = LineString([m - t * big, m + t * big]).intersection(body.buffer(0.01))
    seg = max([g for g in getattr(cut, "geoms", [cut]) if g.geom_type == "LineString"] or [None], key=lambda g: g.length if g is not None else 0)
    if seg is None or seg.length < 6.0: return None, None
    q0, q1 = (np.asarray(v, float) for v in (seg.coords[0], seg.coords[-1]))
    sub = {"poly": body, "id": plot.get("id"), "fronts": [{"p0": [round(float(q0[0]), 3), round(float(q0[1]), 3)], "p1": [round(float(q1[0]), 3), round(float(q1[1]), 3)],
                                                       "rank": "passage", "w": 0.0, "n": [round(float(n[0]), 4), round(float(n[1]), 4)], "len": round(seg.length, 2), "flow": 0.0}]}
    got = HM.program(kind, sub, ctx, rng)
    return (got, max(court, key=lambda g: g.area)) if got else (None, None)


def compound_progs(HM, KH, IF, plot, ce, names, weights, Fl, ctx, rng):
    """Compound of a power center — buildings of the archetype's kind: chambers (a large house of the richest kind, one
    storey taller), a yard (buildings along the edge, the middle is a yard: arena, parade ground, barracks) or a tower
    (a yard + a tower at the back, more storeys); chambers on a compound of 400 m² or more — at the back, with a
    ceremonial yard in front of them toward the main street. → (programs, yard or None)."""
    K = IF.K; vid = ce["kind"]; A = plot["poly"]
    rich = max(names, key=lambda k: (float(KH[k].get("rank", 0.5)), weights[k]))
    main_v = max(names, key=lambda k: weights[k])
    hi = int(Fl["center"][1]) if "center" in Fl else 2
    base = dict(ctx, shops=0.0, tone=abs(float(ctx.get("tone", 0.0))) + 0.04)
    whole = {"outline": "plot", "court": None, "wings": None, "makeup": None, "shed": None, "back": None}   # the building takes its whole plot
    if vid == "chambers" or A.area < 260:
        c2 = dict(base, storeys=[hi, hi + 1]); Kp = K["power"]["chambers"]
        if A.area >= float(Kp["off"]):                                              # chambers at the back, a ceremonial yard in front of them
            got, court = _palace(HM, rich, plot, dict(c2, corrections=whole), Kp, rng)
            if got: return got, court
        if KH[rich]["outline"] == "front": c2["corrections"] = {"depth": [14.0, 20.0]} if A.area >= 400 else whole
        got = HM.program(rich, plot, c2, rng)
        if not got and KH[rich]["outline"] == "front": got = HM.program(rich, plot, dict(c2, corrections=whole), rng)
        return got or _fit(HM, rich, plot, dict(c2, corrections=whole), rng), None
    dep = float(rng.uniform(*K["power"]["block"])); out = []; Tw = None
    if vid == "tower":                                                             # tower — at the back of the compound, buildings — where it isn't
        Tw = _tower(A, float(rng.uniform(*K["power"]["tower"]["side"])))
        if Tw is not None:
            n_t = hi + int(rng.integers(K["power"]["tower"]["above"][0], K["power"]["tower"]["above"][1] + 1))
            c2 = {k: v for k, v in base.items() if k != "limit"}; c2["storeys"] = [n_t, n_t]
            if c2.get("ceiling"): c2["tower"] = float(rng.uniform(2.4, 3.2))
            c2["corrections"] = dict(whole, **({"roof": {"kind": "hip", "slope": [32, 42]}} if KH[main_v]["roof"]["kind"] != "flat" else {}))
            out += (HM.program(main_v, {"poly": Tw, "fronts": [], "id": plot.get("id")}, c2, rng) or [])[:1]
    for q in _ring(A, dep, rng):
        if Tw is not None and q["poly"].distance(Tw) < 1.5: continue
        q["id"] = plot.get("id")
        out += HM.program(main_v, q, dict(base, storeys=[1, max(1, min(2, hi))], corrections=whole), rng) or []
    yard = A.buffer(-dep - 0.5, join_style="mitre")
    if Tw is not None and not yard.is_empty: yard = yard.difference(Tw.buffer(1.0))
    if not out: return _fit(HM, rich, plot, dict(base, storeys=[hi, hi + 1], corrections=whole), rng), None   # the yard didn't fit — chambers
    return out, (yard if not yard.is_empty and yard.area > 40 else None)


def _fit(HM, kind, plot, ctx, rng):
    """Last resort: chambers into the largest rectangle of the compound (axis along the long facade; for an awkwardly shaped
    compound)."""
    A = plot["poly"]; fr = sorted(plot.get("fronts") or [], key=lambda f: -f["len"])
    u = np.asarray(fr[0]["p1"], float) - np.asarray(fr[0]["p0"], float) if fr else np.asarray(A.minimum_rotated_rectangle.exterior.coords[1]) - A.minimum_rotated_rectangle.exterior.coords[0]
    got = LM.fit_rect(A.buffer(-0.3), u, (1.0, 1.5, 2.2), step=1.5)
    if got is None or got[1] * got[2] < 40: return []
    c, L, W = got; u = u / max(np.linalg.norm(u), 1e-9); v = np.array([-u[1], u[0]])
    Q = Polygon([c - u * L / 2 - v * W / 2, c + u * L / 2 - v * W / 2, c + u * L / 2 + v * W / 2, c - u * L / 2 + v * W / 2])
    return HM.program(kind, {"poly": Q, "fronts": [], "id": plot.get("id")}, ctx, rng) or []


# ---------------------------------------------------------------- district map
def draw(path, G, Rj, rib, px_max=1500, decks=None):
    """Map: plots colored by district (reading of P, T, W), streets — thickness by flow, power centers — by sign and kind.
    Hive — tiers in a 2 × 2 grid (tier houses colored by district: poorer at the bottom)."""
    from PIL import Image
    if decks:
        tiles = [_draw(G, Rj, px_max // 2, dk) for dk in range(decks)]; w, h = tiles[0].size; img = Image.new("RGB", (2 * w, 2 * h), (255, 255, 255))
        for k, t in enumerate(tiles): img.paste(t, ((k % 2) * w, (k // 2) * h))
        img.save(path); return
    _draw(G, Rj, px_max, None).save(path)


def _draw(G, Rj, px_max, deck):
    from PIL import Image, ImageDraw, ImageFont
    b = Rj["site"]; c = np.asarray(b["centre"]); half = b["R"] + 30
    x0, y0 = c - half; s = min(px_max / (2 * half), 3.0); W_ = H_ = int(2 * half * s)
    xs = x0 + (np.arange(W_) + 0.5) / s; ys = y0 + (np.arange(H_) + 0.5) / s
    X, Y = np.meshgrid(xs, ys[::-1]); Z = G.height(X, Y)
    gy, gx = np.gradient(Z, 1 / s); shade = np.clip(0.8 + 0.3 * (-gx * 0.7 + gy * 0.7) / np.sqrt(1 + gx * gx + gy * gy), 0.45, 1.1)
    img = Image.fromarray(np.clip(np.stack([214 * shade, 214 * shade, 206 * shade], -1), 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    T = lambda p: ((p[0] - x0) * s, (y0 + 2 * half - p[1]) * s)
    for w_ in Rj.get("water") or []: d.polygon([T(q) for q in w_], fill=(150, 180, 205))
    lab = {p["id"]: p.get("district") for p in Rj["plots"]}
    if deck is None:
        for p in Rj["plots"]:
            if p.get("district"): d.polygon([T(q) for q in p["poly"]], fill=tuple(int(v * 0.55 + 255 * 0.45) for v in COLORS[p["district"]]), outline=(150, 150, 140))
    for h in Rj["houses"]:
        if h.get("deck") != deck: continue
        col = COLORS.get(h.get("district") or lab.get(h.get("plot")), (160, 150, 140)); d.polygon([T(q) for q in h["F"]], fill=col, outline=(50, 45, 40))
    for e in Rj["streets"]["edges"]:
        t = float(e.get("flow", 0.0))
        if e["kind"] in ("edge", "road") or t < 0.12: continue
        d.line([T(q) for q in e["pts"]], fill=(int(120 + 120 * t), int(70 + 40 * t), 30), width=max(1, int(1 + 5 * t)))
    if Rj.get("wall"): d.line([T(q) for q in Rj["wall"]["line"] + Rj["wall"]["line"][:1]], fill=(110, 40, 30), width=3)
    try: f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 22); f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 16)
    except OSError: f = f2 = ImageFont.load_default()
    for y_ in Rj.get("influence", {}).get("yards") or []: d.polygon([T(q) for q in y_], fill=(236, 226, 200), outline=(90, 70, 110))
    for ce in Rj.get("influence", {}).get("centres", []):
        if deck is not None and ce.get("deck") != deck: continue
        x, y = T(ce["p"]); r = 13 if ce["main"] else 9
        d.ellipse([x - r, y - r, x + r, y + r], outline=(80, 30, 110), width=3)
        d.text((x + r + 3, y - 10), ("★ " if ce["main"] else "") + str(ce["kind"]) + (f" · tier {ce['deck']}" if ce.get("deck") is not None else ""), fill=(60, 20, 90), font=f2)
    st = Rj["stats"].get("districts", {}) if deck is None else {k: sum(1 for h in Rj["houses"] if h.get("deck") == deck and h.get("district") == k) for k in LABELS}
    d.rectangle([0, 0, W_, 58], fill=(255, 255, 255))
    d.text((10, 4), f"{Rj['kind']} · districts — reading of the three influences (power P, trade T, wealth W)" + (f" · tier {deck}" if deck is not None else ""), fill=(0, 0, 0), font=f)
    x = 10
    for k in LABELS:
        d.rectangle([x, 34, x + 16, 50], fill=COLORS[k], outline=(40, 40, 40)); t_ = f"{k} {st.get(k, 0)}"
        d.text((x + 22, 33), t_, fill=(30, 30, 30), font=f2); x += 34 + int(d.textlength(t_, font=f2))
    d.text((x + 6, 33), "street: thicker — more passable", fill=(120, 70, 30), font=f2)
    return img
