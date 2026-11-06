"""Reference bench of the "influences" core part: a mechanism, not a picture.
  1. Adjustment formulas: the × of facade and depth grows with wealth, trade narrows the facade, gardens — only for the
     nobility, shops grow with trade, tone is the sign of (W − 0.5), cheap features are more common among the poor, the
     reading yields all six districts, knobs at 0.5 are the identity.
  2. Knobs on settlements (via the buildings.py entry): each one moves its own field in its own direction — wealth (mean W
     of the plots), inequality (spread of W), trade (share of shops and mean T), power (number of centers and area of the
     compounds), wind (the windward side is richer on level ground), center (the main one's compound — at a given point);
     knobs at 0.5 give the same plan as without knobs.
  3. The raster influences.npz: at plot midpoints, the raster's P and W match the plot's (mean difference < 0.03).

.venv/bin/python -m components.buildings.influences.bench.bench [folder] [only=formulas,knobs,raster]
  → OK/FAIL lines, __RESULT__, exit code = number of failures"""
import contextlib, hashlib, io, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PART = os.path.dirname(HERE); COMP = os.path.dirname(PART)
ROOT = COMP
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings import buildings as B                                                          # noqa: E402
from components.buildings.influences import field as FD                                                             # noqa: E402
from components.buildings.influences import influence as INF                                                        # noqa: E402

KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); POS = [a for a in sys.argv[1:] if "=" not in a]
OUT = os.path.abspath(POS[0] if POS else os.path.join(ROOT, "out", "bench_buildings", "influences"))
ONLY = set(KW.get("only", "formulas,knobs,raster").split(","))
CASES = [("town", "cat_river_crossing_2"), ("colony", "cat_river_crossing_1")]   # an organic layout with a wall, and a castrum grid
BAD = []


def check(what, ok, val=""):
    print(("OK   " if ok else "FAIL ") + what, json.dumps(val, ensure_ascii=False, default=float) if val != "" else "", flush=True)
    if not ok: BAD.append(what)


def formulas():
    K = INF.knowledge(); IF = INF.Influence(K, None, [], [], None, 1.0, (0, 1), 0.0, np.array([0, 1.0]), 1, [])
    Ws = np.linspace(0, 1, 11)
    fm = [IF.cut_pw(w, 0.0)[0] for w in Ws]; dm = [IF.cut_pw(w, 0.0)[1] for w in Ws]; em = [IF.cut_pw(w, 0.0)[2] for w in Ws]
    check("× facade grows with wealth (0.6 → 1.7)", all(np.diff(fm) > 0), [round(fm[0], 2), round(fm[-1], 2)])
    check("× depth grows with wealth", all(np.diff(dm) > 0), [round(dm[0], 2), round(dm[-1], 2)])
    check("trade narrows the facade", IF.cut_pw(0.5, 1.0)[0] < IF.cut_pw(0.5, 0.0)[0], [round(IF.cut_pw(0.5, 1.0)[0], 2), round(IF.cut_pw(0.5, 0.0)[0], 2)])
    check("gardens — only for the nobility (W > 0.6), growing with W", max(em[:7]) < 1e-9 and em[-1] > em[8] > 0, [round(v, 2) for v in em[6:]])
    hv = [IF.halves(w, 0.0, 0.4) for w in Ws]; pc = [IF.pieces_exp(w, 0.0) for w in Ws]
    check("blocks: the richer, the less often they are split (castrum: the share halved, hive: the number of parts)", all(np.diff(hv) <= 0) and hv[0] > hv[-1] and all(np.diff(pc) <= 0) and pc[0] > pc[-1],
          {"halves": [round(hv[0], 2), round(hv[-1], 2)], "n_parts": [round(pc[0], 2), round(pc[-1], 2)]})
    sh = [INF.shops(t) for t in Ws]
    check("shops grow with trade", all(np.diff(sh) >= 0) and sh[-1] > 2.5 * sh[0], [round(sh[0], 2), round(sh[-1], 2)])
    check("tone: the rich are lighter, the poor darker", INF.tone(0.9) > 0 > INF.tone(0.1) and INF.tone(0.5) == 0, [INF.tone(0.1), INF.tone(0.9)])
    ch = [INF.cheap({"fachwerk": 0.5, "reach": {"frac": 0.5}, "gallery": {"frac": 0.35}, "cheap": ["fachwerk", "reach", "gallery"]}, w) for w in (0.1, 0.5, 0.9)]
    check("cheap features are more common among the poor (half-timbering, jetties, galleries)", ch[0]["fachwerk"] > ch[1]["fachwerk"] > ch[2]["fachwerk"] and ch[0]["gallery"]["frac"] > ch[2]["gallery"]["frac"],
          [c["fachwerk"] for c in ch])
    pal = [INF.place(0.0, 0.0, w)["palette"]["w"] for w in Ws]
    check("palette: the poor — 0, the nobility — 1, grows with W", pal[0] == 0 and pal[-1] == 1 and all(np.diff(pal) >= 0), pal)
    g = np.linspace(0, 1, 21); P, T, W = (a.ravel() for a in np.meshgrid(g, g, g))
    labs = set(FD.label(P, T, W, K["districts"]))
    check("the reading yields all six districts", labs == set(FD.LABELS), sorted(labs))
    L = INF.levels(); check("knobs default to 0.5", all(v == 0.5 for v in L.values()), L)
    check("knobs — by English names and within 0–1", INF.levels(power=2, wealth=-1)["power"] == 1.0 and INF.levels(wealth=-1)["wealth"] == 0.0)


def plan(kind, place, **kn):
    key = kind + "_" + "_".join(f"{k}{v}" for k, v in sorted(kn.items())); out = os.path.join(OUT, key or kind)
    with contextlib.redirect_stdout(io.StringIO()): R = B.plan("settlement", os.path.join(ROOT, "out", "relief", place), out=out, type=kind, seed=1, **kn)
    return R


def stats(R):
    pl = [p for p in R["plots"] if p.get("influence")]; V = np.array([p["influence"] for p in pl])
    hs = [h for h in R["houses"] if h.get("influence") and h.get("compound") is None]
    shop = np.array([bool(h.get("shop")) or any(w.get("shops") for w in h.get("wings") or []) for h in hs])
    cp = [p for p in R["plots"] if p.get("compound") is not None]
    from shapely.geometry import Polygon
    return {"W": float(V[:, 2].mean()), "Wsd": float(V[:, 2].std()), "T": float(V[:, 1].mean()), "shops": float(shop.mean()) if len(shop) else 0.0,
            "n_centers": len(R["influence"]["centres"]), "compounds_m2": float(sum(Polygon(p["poly"]).area for p in cp)),
            "nobility": sum(p.get("district") == "nobility" for p in pl) / len(pl), "poor": sum(p.get("district") == "poor" for p in pl) / len(pl),
            "X": np.array([np.mean(p["poly"], 0) for p in pl]), "V": V, "fail": [c["what"] for c in R["checks"] if not c["ok"]]}


def knobs():
    for kind, place in CASES:
        t = time.time(); base = plan(kind, place); same = plan(kind, place, power=0.5, trade=0.5, wealth=0.5, inequality=0.5)
        h = lambda R: hashlib.md5(json.dumps({k: R[k] for k in ("plots", "houses", "streets")}, sort_keys=True, default=float).encode()).hexdigest()
        check(f"{kind}: knobs at 0.5 — the same plan", h(base) == h(same))
        S = {k: stats(plan(kind, place, **{k: v})) for k, v in (("wealth", 0.2), ("inequality", 0.2), ("trade", 0.2), ("power", 0.1))}
        S2 = {k: stats(plan(kind, place, **{k: v})) for k, v in (("wealth", 0.8), ("inequality", 0.8), ("trade", 0.8), ("power", 0.9))}
        b = stats(base)
        check(f"{kind}: wealth 0.2 → 0.8 — mean W is higher, more nobility, fewer poor",
              S2["wealth"]["W"] - S["wealth"]["W"] > 0.2 and S2["wealth"]["nobility"] > b["nobility"] > S["wealth"]["nobility"] and S2["wealth"]["poor"] < S["wealth"]["poor"],
              {"W": [round(S["wealth"]["W"], 2), round(b["W"], 2), round(S2["wealth"]["W"], 2)], "nobility": [round(S["wealth"]["nobility"], 2), round(S2["wealth"]["nobility"], 2)]})
        check(f"{kind}: inequality 0.2 → 0.8 — the spread of W is wider", S2["inequality"]["Wsd"] > 1.5 * S["inequality"]["Wsd"],
              [round(S["inequality"]["Wsd"], 3), round(b["Wsd"], 3), round(S2["inequality"]["Wsd"], 3)])
        check(f"{kind}: trade 0.2 → 0.8 — T is higher, more shops", S2["trade"]["T"] > S["trade"]["T"] + 0.1 and S2["trade"]["shops"] > S["trade"]["shops"],
              {"T": [round(S["trade"]["T"], 2), round(S2["trade"]["T"], 2)], "shops": [round(S["trade"]["shops"], 2), round(S2["trade"]["shops"], 2)]})
        check(f"{kind}: power 0.1 → 0.9 — no fewer centers, larger compounds", S2["power"]["n_centers"] >= S["power"]["n_centers"] and S2["power"]["compounds_m2"] > S["power"]["compounds_m2"],
              {"n_centers": [S["power"]["n_centers"], b["n_centers"], S2["power"]["n_centers"]], "m2": [round(S["power"]["compounds_m2"]), round(S2["power"]["compounds_m2"])]})
        bad = sorted({f for s in list(S.values()) + list(S2.values()) for f in s["fail"]})
        check(f"{kind}: the plan checks pass with any knobs", not bad, bad)
        print(f"     {kind}: {round(time.time() - t, 1)} s", flush=True)
    # wind and center — a town on a river (the terrain gives 0.33 of the place term: the wind decides the rest)
    E = stats(plan("town", "cat_river_crossing_2", wind=90)); Wd = stats(plan("town", "cat_river_crossing_2", wind=270))
    cx = lambda s: float(np.corrcoef(s["X"][:, 0], s["V"][:, 2])[0, 1])
    check("wind from the east (90°) — the east is richer, from the west (270°) — the west", cx(E) > cx(Wd), [round(cx(E), 2), round(cx(Wd), 2)])
    R = plan("town", "cat_river_crossing_2", centre="60,40")
    from shapely.geometry import Point, Polygon
    ce = R["influence"]["centres"][0]; cp = [p for p in R["plots"] if p.get("compound") == 0]
    d = float(Polygon(cp[0]["poly"]).distance(Point(60, 40))) if cp else 1e9
    check('center "60,40" — the compound of the main center within its radius', ce["p"] == [60.0, 40.0] and d <= ce["r"], [round(d, 1), ce["r"]])


def raster():
    for kind, place in CASES + [("hive", "cat_river_crossing_3")]:
        R = plan(kind, place); F = FD.load(R["out"])
        if kind == "hive":                                                        # a tier per layer: poorer at the bottom
            Wd = [float(F["W"][k].mean()) for k in range(F["W"].shape[0])]
            check(f"{kind}: raster — a layer per tier, wealth grows from bottom to top", F["P"].shape[0] == R["decks"]["count"] and all(np.diff(Wd) > 0),
                  [round(v, 2) for v in Wd])
            continue
        s = stats(R); P, T, W = FD.sample(F, s["X"])
        dP = float(np.abs(P - s["V"][:, 0]).mean()); dW = float(np.abs(W - s["V"][:, 2]).mean())
        check(f"{kind}: raster at plot midpoints matches the plot (P, W)", dP < 0.03 and dW < 0.03, [round(dP, 3), round(dW, 3)])


def main():
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    if "formulas" in ONLY: formulas()
    if "knobs" in ONLY: knobs()
    if "raster" in ONLY: raster()
    print("__RESULT__", json.dumps({"n_fails": len(BAD), "s": round(time.time() - t0, 1), "folder": OUT, "fails": BAD}, ensure_ascii=False))
    return len(BAD)


if __name__ == "__main__":
    sys.exit(main())
