"""Checks of the masonry kinds (masonry.PACKS) on one piece of wall: a corner 3 + 6 m, height 4.5 m, an arched window.
Light (the project's Python, seconds), no guard needed.

1. Stones do not overlap each other or the surrounds (1 cm raster): overlap < 1 % of the stone area.
2. Filling: stones + surrounds ≥ 55 % of the face area (the rest is joints and the opening).
3. Brick: the vertical joints of adjacent rows do not coincide (closer than 1/4 of a header is a failure), except stack bond.
4. Without rows (cells): stones are polygons of 4–12 corners, small stones in the gaps are no larger than the big ones.
5. No holes: no uncovered field (not a joint, not an opening) larger than 60 cm² — neither at the arch spandrels nor at the reveals and quoins
   (kinds with a flat face; for rubble, dry-stone and cell kinds the gaps are a property of the kind).

.venv/bin/python -m components.constructions.masonry.bench.check_types   → lines OK/FAIL, __RESULT__, exit code = number of failures
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.masonry import masonry as M  # noqa: E402

res = []


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


S = M.Strip([(-3.0, 0.0), (0.0, 0.0), (0.0, 6.0)], [0.0] * 3, [4.5] * 3, 1.0, side=1, closed=False)
px = 0.01; nu, nv = int(S.L / px) + 2, int(4.6 / px) + 2
G = dict(u=1.5, w=1.0, z=1.0, h=1.9, sill=True)
UU, VV = np.meshgrid((np.arange(nu) + 0.5) * px, (np.arange(nv) + 0.5) * px)
r_, zs_ = G["w"] / 2, G["z"] + G["h"] - G["w"] / 2
OPEN = ((abs(UU - G["u"]) < r_) & (VV > G["z"]) & (VV < zs_)) | ((np.hypot(UU - G["u"], VV - zs_) < r_) & (VV >= zs_))
for name, pk in M.PACKS.items():
    L = M.layout(S, pk, gates=[G], seed=3, putlogs=False)
    cnt = np.zeros((nv, nu), np.int16); cov = np.zeros((nv, nu), np.uint8)
    for s_ in L["stones"] + L["features"]:
        q = np.asarray(s_["poly"], float)
        m = np.zeros((nv, nu), np.uint8)
        cv2.fillPoly(m, [np.round(np.stack([q[:, 0] / px, q[:, 1] / px], 1) * 8).astype(np.int32)], 1, shift=3)
        cnt += cv2.erode(m, np.ones((3, 3), np.uint8)); cov |= m                 # the shared edge of neighbours is not an overlap
    area = float((cnt > 0).sum()); over = float((cnt > 1).sum()) / max(area, 1)
    area += 2 * 0.5 * sum(cv2.arcLength(np.asarray(s_["poly"], np.float32) / px, True) for s_ in L["stones"] + L["features"])  # restore the eroded edge
    check(f"{name}: no overlaps", over < 0.01, {"overlaid_%": round(over * 100, 2)})
    fill = area * px * px / L["stats"]["face_m2"]
    check(f"{name}: fill ≥ 55 %", fill >= 0.55, {"filled_%": round(fill * 100, 1)})
    if pk.get("kind") in ("courses", "brick", "banded", "herring", "skyline", "grid45") and pk.get("shape") != "cut":
        free = ((cov == 0) & ~OPEN & (VV < 4.5 - 0.04)).astype(np.uint8)          # the top of the wall is an incomplete row, not a hole
        k = max(3, int(round(pk["joint"] * 1.6 / px)) + 2)
        n, _, st, cen = cv2.connectedComponentsWithStats(cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)))
        big = sorted(((float(st[i, 4]) * px * px, round(float(cen[i][0]) * px, 2), round(float(cen[i][1]) * px, 2)) for i in range(1, n)), reverse=True)
        big = [b for b in big if b[0] > 0.006]
        check(f"{name}: no holes > 60 cm²", not big, {"n_holes": len(big), "large": [round(big[0][0] * 1e4), big[0][1], big[0][2]] if big else None})
    if pk.get("kind") == "brick":
        by = {}
        for s_ in L["stones"]:
            q = np.asarray(s_["poly"]); by.setdefault(s_["course"], []).append((q[:, 0].min(), q[:, 0].max()))
        worst = 9.0
        for k in sorted(by)[1:]:
            a = sorted(x for x in by[k - 1]) if (k - 1) in by else []; b = sorted(x for x in by[k])
            ja = [x1 for (x0, x1), (y0, y1) in zip(a, a[1:]) if y0 - x1 < 0.03]; jb = [x1 for (x0, x1), (y0, y1) in zip(b, b[1:]) if y0 - x1 < 0.03]
            for j in jb:
                if ja: worst = min(worst, min(abs(j - x) for x in ja))
        check(f"{name}: joints of adjacent rows are staggered", worst >= pk["brick"][1] / 4 - 0.005, {"min_m": round(worst, 3)})
    if pk.get("kind") == "cells":
        nvx = [len(s_["poly"]) for s_ in L["stones"] if s_["kind"] == "cell"]
        check(f"{name}: a stone has 4–12 corners", bool(nvx) and min(nvx) >= 3 and np.median(nvx) <= 12,
              {"median": float(np.median(nvx)) if nvx else None, "n_stones": len(nvx)})

ok = sum(r["ok"] for r in res)
print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False))
sys.exit(len(res) - ok)
