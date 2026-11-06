"""Reference test of the roofing part of the Constructions component. No Blender, takes seconds.

1. Density, pieces/m²: kinds with manufacturer data — within ±12 %.
2. Coverage without holes: a 4 × 2.6 m slope and a hip triangle — uncovered decking < 1 % (1 cm raster).
3. Bonding: "staggered" — joints of adjacent rows no closer than w/4; "randomly" — no closer than from_side (except at
   the slope edges).
4. Cone: the rings hold a whole number of tiles, tile width 0.6…1.5 w, the count does not grow toward the apex.
5. Roof analysis from faces: a house (gable + hip) — 3 slopes, 1 ridge, 2 hips; a pyramid — 4 slopes, 4 hips, a finial;
   a 24-face tent — a cone; an L-shaped house — 4 slopes, 2 ridges, a hip and a valley (no ridge, tiles no closer than
   6 cm to the valley line: an open trough). Ridge: the links reach the end of the edge, each lies on the previous one;
   for trough tiles the link radius is above the profile (a link on a mortar bed covers the ends of the rolls).
6. Domes (bodies of revolution): onion, cupola — revolve with an apex, apse half-dome — 180°, octagonal tent — slopes
   (not a circle); lemekh rings on the onion — in zones (the count holds, width 0.7…1.35 w), the ring is closed.
7. Diminishing courses (stone slab): the step does not grow from the eaves to the ridge, the first ≈ (L − lap)/2 at the
   eaves.
8. Sheets: the seam on a dome — wedges of one width along the angle, ending where narrower than 12 cm; boards — every
   gap of the lower layer is covered by an upper board.
9. Layout fingerprints (tile count and contour hash) = golden.json; a deliberate change — --save.

.venv/bin/python -m components.constructions.roofing.bench.bench [--save]   → OK/FAIL, __RESULT__, exit code = number of failures
"""
import hashlib
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.roofing import roofing as R  # noqa: E402

GOLD = os.path.join(HERE, "golden.json")
DENSITY = {"plain": 60.0, "beaver": 37.5, "pantile": 15.9, "slate": 19.5, "canal": 33.0}   # [f] manufacturer data
res = []


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


PITCH = math.radians(45)
EV = np.array([0.0, math.cos(PITCH), math.sin(PITCH)])


def rect(W=4.0, Lr=2.6):
    return R.Slope([np.zeros(3), np.array([W, 0, 0]), np.array([W, 0, 0]) + EV * Lr, EV * Lr], np.cross([1, 0, 0], EV))


def tri(W=4.0, Lr=2.6):
    return R.Slope([np.zeros(3), np.array([W, 0, 0]), np.array([W / 2, 0, 0]) + EV * Lr], np.cross([1, 0, 0], EV))


def uncovered(S, tiles, px=0.01):
    """Fraction of the slope outline not covered by any tile (1 cm raster; the slope edge is inset by 1 cm)."""
    W, H = int(S.width / px) + 3, int(S.top / px) + 3
    poly = np.zeros((H, W), np.uint8); cov = np.zeros((H, W), np.uint8)
    cv2.fillPoly(poly, [np.round(np.asarray(S.poly) / px * 8).astype(np.int32)], 1, shift=3)
    poly = cv2.erode(poly, np.ones((3, 3), np.uint8))
    for t in tiles:
        cv2.fillPoly(cov, [np.round(np.asarray(t["poly"]) / px * 8).astype(np.int32)], 1, shift=3)
    return float(((poly > 0) & (cov == 0)).sum()) / max(float(poly.sum()), 1.0)


def main():
    save = "--save" in sys.argv
    for name, ref in DENSITY.items():                                            # 1. density
        k = R.KINDS[name]; S = rect(6.0, 4.0)
        n = sum(1 for t in R.lay(S, k, seed=3) if not t["under"] and t["prof"] not in ("monk", "capping"))
        if name == "canal": n = sum(1 for t in R.lay(S, k, seed=3) if not t["under"])
        dens = n / abs(R.area(S.poly))
        check(f"{name}: density ≈ {ref} pcs/m²", abs(dens / ref - 1) < 0.12, {"pcs_m2": round(dens, 1)})
    for name, k in R.KINDS.items():                                              # 2. coverage, 3. bonding
        for nm, S in (("pitch", rect()), ("hip", tri())):
            tiles = R.lay(S, k, seed=3)
            u = uncovered(S, tiles)
            check(f"{name}, {nm}: decking is covered", u < 0.01, {"exposed_%": round(u * 100, 2), "n_tiles": len(tiles)})
        if k["generator"] == "rows":
            S = rect(); tiles = [t for t in R.lay(S, k, seed=3) if not t["under"]]
            by = {}
            for t in tiles: by.setdefault(t["course"], []).append((t["x0"], t["x1"]))
            lim = R.width(k) / 4 if k["bond"] == "staggered" else float(k["from_side"]) - 0.005
            worst = 9.0
            for c in sorted(by)[1:]:
                if c - 1 not in by: continue
                ja = [b for a, b in by[c - 1] if 0.05 < b < S.width - 0.05]; jb = [b for a, b in by[c] if 0.05 < b < S.width - 0.05]
                for j in jb:
                    if ja: worst = min(worst, min(abs(j - x) for x in ja))
            check(f"{name}: row joints are staggered", worst >= lim, {"min_m": round(worst, 3), "threshold": round(lim, 3)})
    for name in ("plain", "slate", "lemekh", "pantile"):                          # 4. cone
        k = R.KINDS[name]; C = R.Cone((0, 0, 5.0), (0, 0, 0), 2.2)
        tiles = [t for t in R.lay(C, k, seed=3) if not t["under"]]
        by = {}
        for t in tiles: by.setdefault(t["course"], []).append(t)
        ns = [len(by[c]) for c in sorted(by)]
        ws = [(t["x1"] - t["x0"]) for t in tiles]
        w = R.width(k)
        check(f"{name}: cone — rings", all(a >= b for a, b in zip(ns, ns[1:])) and 0.6 * w <= min(ws) and max(ws) <= 1.5 * w + 0.01,
              {"n_rings": len(ns), "n_tiles": len(tiles), "width": [round(min(ws), 3), round(max(ws), 3)]})
    ze, zr, ov = 3.0, 5.25, 0.4                                                  # 5. roof analysis
    e = [(-ov, -ov - 0.0, ze), (7 + ov, -ov, ze), (7 + ov, 4.5 + ov, ze), (-ov, 4.5 + ov, ze)]
    r0, r1 = (-ov, 2.25, zr), (4.75, 2.25, zr)
    house = [([e[0], e[1], r1, r0], "all"), ([e[2], e[3], r0, r1], "all"), ([e[1], e[2], r1], "all")]
    E = [(-2, -2), (2, -2), (2, 2), (-2, 2)]; pyr = [([(*E[i], 0.0), (*E[(i + 1) % 4], 0.0), (0, 0, 4.0)], "towers") for i in range(4)]
    ring = [(5.0 * math.cos(2 * math.pi * i / 24) + 20, 5.0 * math.sin(2 * math.pi * i / 24), 0.0) for i in range(24)]
    cone = [([ring[i], ring[(i + 1) % 24], (20.0, 0.0, 6.0)], "towers") for i in range(24)]
    roofs = R.analyse(house + [([(x + 30, y, z) for x, y, z in f], p) for f, p in pyr] + cone)
    kinds = sorted((r["kind"], len(r["surfaces"]), len(r["ridges"]), len(r["hips"]), r["apex"] is not None) for r in roofs)
    want = sorted([("slopes", 3, 1, 2, False), ("slopes", 4, 0, 4, True), ("cone", 1, 0, 0, True)])
    check("roof analysis: house, pyramid, tent", kinds == want, {"present": kinds})
    Lh = [([(0, 0, 0), (10, 0, 0), (10, 2, 2), (2, 2, 2)], "all"), ([(2, 2, 2), (10, 2, 2), (10, 4, 0), (4, 4, 0)], "all"),   # 5c. valley
          ([(0, 0, 0), (2, 2, 2), (2, 10, 2), (0, 10, 0)], "all"), ([(4, 4, 0), (4, 10, 0), (2, 10, 2), (2, 2, 2)], "all")]
    rf = R.analyse(Lh)[0]; R.open_valleys(rf); R.open_valleys(rf)               # twice — cuts only once
    near = 1e9
    for vl in rf["valleys"]:
        for si in vl["s"]:
            S = rf["surfaces"][si]; A, B = S.uv(vl["a"]), S.uv(vl["b"]); d = (B - A) / np.linalg.norm(B - A); nrm = np.array([-d[1], d[0]])
            if (S.poly.mean(0) - A) @ nrm < 0: nrm = -nrm
            for t in R.lay(S, R.KINDS["plain"], seed=5): near = min(near, min(float((np.asarray(q) - A) @ nrm) for q in t["poly"]))
    check("valley: an L-shaped house — 4 slopes, 2 ridges, a hip, a valley without a ridge, the trough is open",
          (len(rf["surfaces"]), len(rf["ridges"]), len(rf["hips"]), len(rf["valleys"])) == (4, 2, 1, 1) and abs(near - R.VALLEY_GAP) < 0.005,
          {"n_pitches_ridges_hips_valleys": [len(rf["surfaces"]), len(rf["ridges"]), len(rf["hips"]), len(rf["valleys"])], "to_line_m": round(near, 3)})
    ring = [(5 * math.cos(a), 5 * math.sin(a), 0.0) for a in np.linspace(0, math.pi, 11)]   # 5b. apse fan, roof over a walkway
    fan = [([ring[i], ring[i + 1], (0, 0, 4.0)], "all") for i in range(10)]
    P = lambda a, r, z: (r * math.cos(a) + 100, r * math.sin(a), z)
    strip = [([P(math.radians(5 * i), 32, 0.0), P(math.radians(5 * i + 5), 32, 0.0), P(math.radians(5 * i + 5), 30, 1.2), P(math.radians(5 * i), 30, 1.2)], "gates")
             for i in range(14)]
    rr = {r["kind"]: r for r in R.analyse(fan + strip)}
    arc = math.degrees(rr["cone"]["surfaces"][0].th1 - rr["cone"]["surfaces"][0].th0) if "cone" in rr else 0
    check("apse fan — a cone along the arc; a walkway roof on a curved wall — no ridges", abs(arc - 180) < 1 and rr["cone"]["apex"] is None
          and len(rr["slopes"]["surfaces"]) == 14 and not rr["slopes"]["ridges"] and not rr["slopes"]["hips"], {"arc°": round(arc)})
    for Lr in (0.2, 3.0, 7.0):                                                   # 5a. ridge: links to the end of the edge, lap
        cs = R.caps((0, 0, 5), (Lr, 0, 5), (0, 0, 1), R.KINDS["beaver"], 1)
        ends = [c["b"][0] for c in cs]; starts = [c["a"][0] for c in cs]
        ok = abs(ends[-1] - Lr) < 1e-6 and all(b_ > a_ for a_, b_ in zip(starts[1:], ends[:-1])) and len(cs) <= Lr / 0.29 + 2
        check(f"ridge {Lr} m: links reach the end, with a lap", ok, {"n_links": len(cs)})
    low = {n: (round(R.caps((0, 0, 0), (1, 0, 0), (0, 0, 1), k, 1)[0]["r"], 3), round(R.relief(k), 3)) for n, k in R.KINDS.items() if R.relief(k) > 0}
    check("ridge of trough tiles is above the profile", all(r >= h + 0.03 for r, h in low.values()), {"r_plus_profile": low})
    def lathe(prof, n=32, cx=0.0, a1=2 * math.pi):                               # 6. domes: mesh of a body of revolution
        full = a1 > 2 * math.pi - 1e-6; m = n if full else n + 1
        ang = [a1 * i / n for i in range(m)]; rg = [[(cx + r * math.cos(a), r * math.sin(a), z) for a in ang] for r, z in prof]
        F = []
        for j in range(len(prof) - 1):
            for i in range(n):
                i2 = (i + 1) % m if full else i + 1
                F.append(([rg[j][i], rg[j][i2], rg[j + 1][i2]] if prof[j + 1][0] < 1e-6 else [rg[j][i], rg[j][i2], rg[j + 1][i2], rg[j + 1][i]], "all"))
        return F
    rb, hs = 1.6, 3.2
    onion = [(rb * 0.85, 0.0), (rb * 1.15, 0.22 * hs), (rb * 1.05, 0.45 * hs), (rb * 0.6, 0.7 * hs), (rb * 0.18, 0.88 * hs), (0.0, hs)]
    dome = [(2.5 * math.cos(a), 2.5 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 9)]
    half = [(2.0 * math.cos(a), 2.0 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 7)]
    rr = R.analyse(lathe(onion) + lathe(dome, cx=10) + lathe(half, n=16, cx=20, a1=math.pi) + lathe([(2.0, 0.0), (0.0, 3.0)], n=8, cx=30))
    got = sorted((r["kind"], round(math.degrees(r["surfaces"][0].th1 - r["surfaces"][0].th0)) if r["kind"] == "revolve" else len(r["surfaces"]),
                  r["apex"] is not None) for r in rr)
    check("domes: onion, cupola, half-dome — bodies of revolution; octagonal tent — slopes", got == sorted([("revolve", 360, True), ("revolve", 360, True),
          ("revolve", 180, False), ("slopes", 8, True)]), {"present": got})
    on = next(r["surfaces"][0] for r in rr if r["kind"] == "revolve" and r["surfaces"][0].Rmax < 2)
    k = R.KINDS["lemekh"]; wg = R.width(k) + float(k["gap"]); rows = {}
    for t in R.lay(on, k, seed=3):
        if not t["under"]: rows.setdefault(t["course"], []).append(t)
    ok, worst, zones = True, [9, 0], []
    for c_, ts in sorted(rows.items()):
        n = len(ts); rm = float(on.r(ts[0]["v0"] + 0.5 * R.gauge(k))); w_ = 2 * math.pi * rm / n
        worst = [min(worst[0], w_ / wg), max(worst[1], w_ / wg)]
        if not zones or zones[-1] != n: zones.append(n)
        th = sorted((t["x0"] / t["r0"]) % (2 * math.pi) for t in ts); ok &= abs(sum((t["x1"] - t["x0"]) / t["r0"] for t in ts) + n * k["gap"] / rm - 2 * math.pi) < 0.02
    check("domes: lemekh rings on the onion — in zones, width 0.7…1.35 w, the ring is closed", ok and 0.69 <= worst[0] and worst[1] <= 1.36 and len(zones) < len(rows) / 2,
          {"zones": zones, "width_frac": [round(x, 2) for x in worst], "n_rows": len(rows)})
    k = R.KINDS["stone_slate"]; S = rect(4.0, 3.2); cs = sorted({round(t["v0"], 4) for t in R.lay(S, k, seed=3) if not t["under"]})   # 7. diminishing courses
    gs = np.diff(cs); g0 = (k["L"][0] - k["lap"][0]) / 2
    check("diminishing courses: the step does not grow toward the ridge, at the eaves ≈ (L − lap)/2", bool(np.all(np.diff(gs) <= 1e-9)) and abs(gs[0] / g0 - 1) < 0.05 and gs[-1] < 0.7 * gs[0],
          {"step_m": [round(float(gs[0]), 3), round(float(gs[-1]), 3)], "n_rows": len(cs)})
    k = R.KINDS["copper_seam"]; tl = [t for t in R.lay(on, k, seed=3) if t["prof"] == "sheet"]   # 8. sheets
    dth = {round(t["x1"] - t["x0"], 6) for t in tl}; top = max(t["v0"] + t["L"] for t in tl); wt = float(on.r(top)) * next(iter(dth))
    check("sheets: the seam on a dome — wedges of one width, ending where narrower than 12 cm", len(dth) == 1 and 0.11 <= wt <= 0.2,
          {"n_wedges": round(2 * math.pi / next(iter(dth))), "width_at_apex_m": round(wt, 3)})
    k = R.KINDS["boards"]; tl = R.lay(rect(4.0, 2.6), k, seed=3)
    lo = sorted((t["x0"], t["x1"]) for t in tl if t.get("h0", 0) == 0); up = sorted((t["x0"], t["x1"]) for t in tl if t.get("h0", 0) > 0)
    gaps = [(a[1], b[0]) for a, b in zip(lo, lo[1:]) if b[0] - a[1] > 1e-6 and a[1] > 0.0 and b[0] < 4.0]
    covered = all(any(u0 <= g0_ + 1e-6 and u1 >= g1 - 1e-6 for u0, u1 in up) for g0_, g1 in gaps)
    check("sheets: boards — every gap of the lower layer is covered by an upper board", bool(gaps) and covered, {"n_gaps": len(gaps)})
    fp = {}                                                                      # 9. fingerprints
    for name, k in R.KINDS.items():
        for nm, S in (("pitch", rect()), ("hip", tri()), ("cone", R.Cone((0, 0, 5.0), (0, 0, 0), 2.2))):
            tiles = R.lay(S, k, seed=3); h = hashlib.sha1()
            for t in tiles: h.update(json.dumps([[round(x, 4), round(y, 4)] for x, y in t["poly"]]).encode())
            fp[f"{name}:{nm}"] = dict(n_tiles=len(tiles), digest=h.hexdigest()[:16])
    if save or not os.path.exists(GOLD):
        json.dump(dict(version=R.LIBRARY.get("version"), imprints=fp), open(GOLD, "w"), ensure_ascii=False, indent=1); print("reference written:", GOLD)
    gold = json.load(open(GOLD))["imprints"]
    diff = sorted(k_ for k_ in fp if gold.get(k_) != fp[k_])
    check("layout fingerprints = reference", not diff and set(gold) == set(fp), {"n_layouts": len(fp), "otherwise": diff[:6]})
    ok = sum(r["ok"] for r in res)
    print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False))
    sys.exit(len(res) - ok)


main()
