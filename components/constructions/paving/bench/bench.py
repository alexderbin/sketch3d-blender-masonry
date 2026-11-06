"""Reference test of the paving part of the Constructions component. No Blender, takes seconds.

1. Coverage without holes: a courtyard (L-shaped polygon) and a curved road — the stones, grown by the joint, cover the
   area (uncovered < 1.5 %, 1 cm raster; the area edge is inset by 2 cm).
2. Patterns without overlaps: herringbone, basketweave, grid — the stones do not overlap (raster: overlap < 0.5 % of
   the area).
3. A chunk is the same in any order: the stones of a chunk computed separately and in a common pass coincide (number,
   contour).
4. Setts bonding: joints of adjacent rows no closer than from_side (except at the edges).
5. Layout fingerprints (stone count and contour hash) = golden.json; a deliberate change — --save.

.venv/bin/python -m components.constructions.paving.bench.bench [--save]   → OK/FAIL, __RESULT__, exit code = number of failures
"""
import hashlib
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.paving import paving as P  # noqa: E402

GOLD = os.path.join(HERE, "golden.json")
res = []


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


YARD = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
ROAD = [(0, 0), (4, 0), (6.5, 1.2), (8, 3.5)]


def raster(reg, stones, px=0.01, grow=0.0):
    """Raster of the area (xy) and of the stones: (area mask, coverage, overlap count)."""
    out = reg.outline(0.1); lo = out.min(0) - 0.2; hi = out.max(0) + 0.2
    W, H = int((hi[0] - lo[0]) / px) + 1, int((hi[1] - lo[1]) / px) + 1
    def pix(q): return np.round((np.asarray(q) - lo) / px * 8).astype(np.int32)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pix(out)], 1, shift=3)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    cov = np.zeros((H, W), np.uint8); cnt = np.zeros((H, W), np.int16)
    for st in stones:
        uv = P.densify(st["poly"], 0.2, axis=0) if reg.kind == "path" else np.asarray(st["poly"], float)
        q = reg.xy(uv[:, 0], uv[:, 1]); one = np.zeros((H, W), np.uint8); cv2.fillPoly(one, [pix(q)], 1, shift=3)
        if grow > 0: one = cv2.dilate(one, np.ones((2 * int(grow / px) + 1,) * 2, np.uint8))
        cov |= one; cnt += cv2.erode(one, np.ones((3, 3), np.uint8)) if grow == 0 else 0
    return m, cov, cnt


def main():
    save = "--save" in sys.argv
    yard, road = P.Area(YARD, angle=0.0), P.Path(ROAD, 2.5)
    for name, k in P.KINDS.items():                                              # 1. coverage
        for reg, nm in ((yard, "court"), (road, "road")):
            st = P.lay(reg, k, seed=3); m, cov, _ = raster(reg, st, grow=float(k["joint"]) * 0.75 + 0.006)
            u = float(((m > 0) & (cov == 0)).sum()) / max(float(m.sum()), 1.0)
            check(f"{name}, {nm}: covered (with the joint)", u < 0.015, {"exposed_%": round(u * 100, 2), "n_stones": len(st)})
    for name in ("brick_herringbone", "brick_basket", "floor_tiles", "floor_tiles_diag"):   # 2. overlaps
        st = P.lay(yard, P.KINDS[name], seed=3); m, cov, cnt = raster(yard, st)
        ov = float((cnt > 1).sum()) / max(float(m.sum()), 1.0)
        check(f"{name}: stones do not overlap", ov < 0.005, {"overlay_%": round(ov * 100, 3)})
    for name in ("cobble", "setts_large", "planks"):                             # 3. a chunk in any order
        k = P.KINDS[name]; chs = yard.chunks(); P._COURSES.clear()
        a = P.units(yard, k, chs[-1], seed=3); P._COURSES.clear(); P.lay(yard, k, seed=3); b = P.units(yard, k, chs[-1], seed=3)
        same = [(x["sid"], x["poly"]) for x in a] == [(x["sid"], x["poly"]) for x in b]
        check(f"{name}: the chunk is the same in any order", same and len(a) > 0, {"n_stones": len(a)})
    k = P.KINDS["setts_large"]; cs = P._courses(road, k, 3, False); worst = 9.0   # 4. bonding
    for (x0, w0, s0), (x1, w1, s1) in zip(cs, cs[1:]):
        e0 = [b for a, b in s0][:-1]; e1 = [b for a, b in s1][:-1]
        for e in e1:
            if -road.w / 2 + 0.1 < e < road.w / 2 - 0.1 and e0: worst = min(worst, min(abs(e - f) for f in e0))
    check("setts: joints of adjacent rows no closer than from_side", worst >= float(k["from_side"]) - 0.005, {"min_m": round(worst, 3)})
    fp = {}                                                                      # 5. fingerprints
    for name, k in P.KINDS.items():
        for reg, nm in ((yard, "court"), (road, "road")):
            st = P.lay(reg, k, seed=3); h = hashlib.sha1()
            for x in st: h.update(json.dumps([x["sid"], [[round(a, 4), round(b, 4)] for a, b in x["poly"]]]).encode())
            fp[f"{name}:{nm}"] = dict(n_stones=len(st), digest=h.hexdigest()[:16])
    if save or not os.path.exists(GOLD):
        json.dump(dict(version=P.LIBRARY.get("version"), imprints=fp), open(GOLD, "w"), ensure_ascii=False, indent=1); print("reference written:", GOLD)
    gold = json.load(open(GOLD))["imprints"]
    diff = sorted(k_ for k_ in fp if gold.get(k_) != fp[k_])
    check("layout fingerprints = reference", not diff and set(gold) == set(fp), {"n_layouts": len(fp), "otherwise": diff[:6]})
    ok = sum(r["ok"] for r in res)
    print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False))
    sys.exit(len(res) - ok)


main()
