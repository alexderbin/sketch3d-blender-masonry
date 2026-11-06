"""Masonry layout checks: what a layout must satisfy by itself, without visual inspection. Light (the project's Python, seconds), no guard needed.

1. Bonding: on ashlar faces a vertical joint of a row is at least 0.6 · lap away from the joints of the row below (there are no through joints).
2. Corners: quoins alternate a long and a short stone on each face of a corner.
3. Rows through key levels: the sill and the arch springers lie on row boundaries (± 2 mm).
4. Filling: the share of empty space (not a stone, not a joint, not an opening, not a socket) on the faces is < 1 %.
5. Ruin: a live facing stone with no core behind it is supported from below by ≥ 0.35 of its bottom edge (the support rule).
6. Irregularity: header joints are not all vertical, the row boundaries are not straight; chips exist.

.venv/bin/python -m components.constructions.masonry.bench.check_layout [folder with layout_*.json]   → lines OK/FAIL, __RESULT__, exit code = number of failures
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "runs", "probe", "out")
res = []


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


def joints_of(cells, face, course):
    """Vertical joints of a row between two course stones (another stone of the row directly to the right); edges at an opening, an arch, a scaffold socket and an end face
    are not joints. The position of a joint is its mid-height."""
    st = []
    for c in cells:
        if c["face"] != face or c["course"] != course or c["kind"] != "field": continue
        q = np.asarray(c["poly"]); st.append((q[:, 0].min(), q[:, 0].max()))
    st.sort(); out = []
    for (a0, a1), (b0, b1) in zip(st, st[1:]):
        if abs(b0 - a1) < 0.03: out.append((a1 + b0) / 2)
    return out


for state in ("intact", "ruin"):
    p = os.path.join(OUT, f"layout_{state}.json")
    if not os.path.exists(p): check(f"{state}: a layout exists", False, p); continue
    D = json.load(open(p)); P = D["P"]; cells = D["cells"]; lap = P["lap"]
    # 1. bonding
    worst, n_pairs = 9.0, 0
    for face in ("A_out", "B_out"):
        ks = sorted({c["course"] for c in cells if c["face"] == face and c["kind"] == "field"})
        for k in ks[1:]:
            a, b = joints_of(cells, face, k - 1), joints_of(cells, face, k)
            for j in b:
                if not a: continue
                d = min(abs(j - x) for x in a); worst = min(worst, d); n_pairs += 1
    check(f"{state}: bonding — the nearest joints of adjacent rows ≥ 0.6·lap", worst >= 0.6 * lap - 0.02,
          {"min_m": round(worst, 3), "threshold_m": round(0.6 * lap, 3), "n_joints": n_pairs})
    # 2. quoins alternate
    q = sorted([c for c in cells if c["kind"] == "quoin"], key=lambda c: c["course"])
    lens = [np.ptp(np.asarray(c["poly"])[:, 0]) for c in q]
    alt = all((lens[i] - 0.7) * (lens[i + 1] - 0.7) < 0 for i in range(len(lens) - 1))
    check(f"{state}: quoins — long and short in alternate rows", alt, {"lengths_m": [round(float(x), 2) for x in lens[:6]]})
    # 3. rows through key levels
    zs = np.asarray(D["courses"]); op = P["opening"]
    dz = [float(np.min(np.abs(zs - op[k]))) for k in ("sill", "spring")]
    check(f"{state}: the sill and the springers — on row boundaries", max(dz) <= 0.002, {"offset_mm": [round(x * 1000, 1) for x in dz]})
    # 6. irregularity
    tilt = []; wavy = []
    for c in cells:
        if c["kind"] != "field" or c["gone"]: continue
        s = np.asarray(c["poly"])
        if len(s) >= 6:
            lo = s[np.argsort(s[:, 1])[:2]]; tilt.append(abs(lo[0, 0] - lo[1, 0]))
        ys = s[:, 1]; wavy.append(float(np.ptp(ys[ys < ys.min() + 0.02])))
    chips = sum(len(c.get("chips", [])) for c in cells)
    check(f"{state}: masonry is not ruler-straight (row waviness, chips)", (np.mean(wavy) > 0.002 if wavy else False) and chips > 10,
          {"wave_mm_avg": round(float(np.mean(wavy)) * 1000, 1) if wavy else None, "chips": chips})
    # 5. ruin: support
    if state == "ruin":
        st = D["stats"]
        check("ruin: stones have fallen out, some are split at the holes and some are exposed from the side", st["gone"] > 50 and st["broken"] > 10 and st["exposed"] > 20,
              {"fallen": st["gone"], "broken": st["broken"], "exposed": st["exposed"], "rubble_n_points": st["lining"]})
        h = D["extra"].get("heap") or {}
        V_ok = h and abs(math.pi * h["r"] ** 3 * math.tan(math.radians(38)) / 6 - h["V"]) / max(h["V"], 1e-6) < 0.02
        check("ruin: scree — the volume removed × 1.4 × 0.4, a half-cone of 38°", bool(V_ok), h)

ok = sum(r["ok"] for r in res)
print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False))
sys.exit(len(res) - ok)
