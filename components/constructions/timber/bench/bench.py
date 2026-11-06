"""Reference test of the timber part of the Constructions component. No Blender, takes seconds.

1. Fachwerk: posts at both corners and at each opening jamb, gaps no wider than the "step" (except openings), the infill
   does not enter openings, all elements stay within the wall.
2. Log house: courses cover the wall height, the overhang only at outer corners (≈ "remainder" diameters), a wall of
   the other direction is shifted by half a course, a log is cut at an opening (none crosses a window).
3. Cladding: boards cover the wall except openings (1 cm raster).
4. Palisade — logs tight together; wattle — a rod passes the neighboring stakes from alternate sides.
5. Layout fingerprints = golden.json; a deliberate change — --save.
6. Plan (resolve): the description's fences as a list {path, kind}; a fence without kind takes the kind of the fences
   knob (palisade); a non-fence kind is a failure.

.venv/bin/python -m components.constructions.timber.bench.bench [--save]   → OK/FAIL, __RESULT__, exit code = number of failures
"""
import hashlib
import json
import math
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.timber import timber as T  # noqa: E402

GOLD = os.path.join(HERE, "golden.json")
res = []
OPS = [(1.2, 2.0, 0.9, 2.2), (4.0, 4.8, 0.9, 2.2), (2.6, 3.6, 0.0, 2.1), (1.2, 2.0, 3.7, 5.0), (4.0, 4.8, 3.7, 5.0)]


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


def wall(n=(0, -1)):
    return T.Panel((0, 0), (7, 0), 0.0, 5.6, n=n, storeys=[0.0, 2.8], openings=OPS)


def main():
    save = "--save" in sys.argv
    pn = wall()
    for name in ("fachwerk", "fachwerk_close", "fachwerk_cross"):                # 1. fachwerk
        k = T.KINDS[name]; it = T.panel_items(pn, k, seed=3); sw = k["post"][0]
        posts = sorted({round((e["poly"][0][0] + e["poly"][1][0]) / 2, 3) for e in it if e.get("kind") == "post"})
        need = [sw / 2, pn.L - sw / 2] + [x for o in OPS for x in (o[0] - sw / 2, o[1] + sw / 2)]
        has = all(min(abs(p - x) for p in posts) < 0.36 for x in need)
        gaps = [b - a for a, b in zip(posts, posts[1:]) if not any(o[0] - 0.01 <= (a + b) / 2 <= o[1] + 0.01 for o in OPS)]
        fields = [e for e in it if e["t"] == "field"]
        bad = [f for f in fields for o in OPS if min(f["poly"][1][0], o[1]) - max(f["poly"][0][0], o[0]) > 0.02 and min(f["poly"][2][1], o[3]) - max(f["poly"][0][1], o[2]) > 0.02]
        inside = all(-1e-6 <= p[0] <= pn.L + 1e-6 and pn.z0 - 1e-6 <= p[1] <= pn.z1 + 1e-6 for e in it if "poly" in e for p in e["poly"])
        check(f"{name}: posts at corners and jambs, step ≤ {k['step']}, infill outside openings, everything within the wall",
              has and (not gaps or max(gaps) <= float(k["step"]) + sw + 1e-6) and not bad and inside,
              {"n_posts": len(posts), "largest_step": round(max(gaps), 3) if gaps else None, "fields_within_openings": len(bad)})
    for name in ("log_oblo", "log_lapa"):                                        # 2. log house
        k = T.KINDS[name]; it = T.panel_items(pn, k, seed=3); logs = [e for e in it if e["t"] == "log"]
        top = max(e["zc"] + e["r"] for e in logs); bottom = min(e["zc"] - e["r"] for e in logs)
        ext = max(-min(e["u0"] for e in logs), max(e["u1"] for e in logs) - pn.L)
        exp = float(k["remainder"]) * max(k["d"])
        cross = [e for e in logs for o in OPS if o[2] < e["zc"] < o[3] and e["u0"] < o[1] - 0.01 and e["u1"] > o[0] + 0.01]
        pn2 = wall(n=(1, 0)); z1 = min(e["zc"] for e in T.panel_items(pn2, k, seed=3) if e["t"] == "log")
        z0 = min(e["zc"] for e in logs); hc = float(k["crown"]) * sum(k["d"]) / 2
        check(f"{name}: courses over the full height, overhang ≈ remainder, a half-course shift at the neighboring wall, windows are clear",
              bottom <= pn.z0 + 0.15 and top >= pn.z1 - hc and ext <= exp + 1e-6 and (ext > 0.1 if k["remainder"] > 0 else ext < 1e-6)
              and abs(abs(z1 - z0) - hc / 2) < 0.02 and not cross,
              {"n_courses": len({e["course"] for e in logs}), "extension_m": round(ext, 3), "shift_m": round(abs(z1 - z0), 3), "via_window": len(cross)})
    k = T.KINDS["boards"]; it = T.panel_items(pn, k, seed=3); px = 0.01           # 3. cladding
    W, H = int(pn.L / px), int(pn.z1 / px); m = np.ones((H, W), np.uint8)
    for o in OPS: m[int(o[2] / px):int(o[3] / px), int(o[0] / px):int(o[1] / px)] = 0
    cov = np.zeros((H, W), np.uint8)
    for e in it:
        q = np.asarray(e["poly"]); cv2.fillPoly(cov, [np.round(q / px * 8).astype(np.int32)], 1, shift=3)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)); u = float(((m > 0) & (cov == 0)).sum()) / max(float(m.sum()), 1.0)
    over = float(((cov > 0) & (cv2.dilate(1 - m, np.ones((5, 5), np.uint8)) == 0) & (m == 0)).sum())
    check("cladding: the wall is covered, openings are clear", u < 0.005 and over == 0, {"exposed_%": round(u * 100, 3)})
    path = [(0, 0), (6, 0), (9, 3)]                                              # 4. fences
    ps = T.palisade(path, T.KINDS["palisade"], seed=3)
    gaps = [float(np.hypot(*(np.asarray(b["xy"]) - np.asarray(a["xy"])))) - a["r"] - b["r"] for a, b in zip(ps, ps[1:])]
    st, rods = T.wattle(path, T.KINDS["wattle"], seed=3)
    alt = all(np.sign(np.cross(np.append(np.asarray(r["pts3"][i + 1][:2]) - np.asarray(r["pts3"][i][:2]), 0), [0, 0, 1])[2]) != 0 for r in rods[:3] for i in range(3))
    sides = [[1 if ((np.asarray(r["pts3"][i][:2]) - np.asarray(st[i]["xy"])) @ np.array([0, 1])) > 0 else -1 for i in range(2)] for r in rods[:2]]
    check("palisade tight together, wattle interwoven", max(gaps) < 0.06 and min(gaps) > -0.08 and sides[0][0] != sides[0][1] and sides[0][0] != sides[1][0],
          {"n_stakes": len(st), "n_rods": len(rods), "n_logs": len(ps), "gap_m": [round(min(gaps), 3), round(max(gaps), 3)]})
    fp = {}                                                                      # 5. fingerprints
    for name, k in T.KINDS.items():
        if k["generator"] in ("palisade", "wattle"):
            items = T.palisade(path, k, 3) if k["generator"] == "palisade" else T.wattle(path, k, 3)[0]
            h = hashlib.sha1(json.dumps([[round(x, 4) for x in e["xy"]] + [round(e.get("r", 0), 4)] for e in items]).encode())
        else:
            items = T.panel_items(pn, k, seed=3); h = hashlib.sha1()
            for e in items: h.update(json.dumps([e["t"], e.get("kind"), [[round(a, 4), round(b, 4)] for a, b in e.get("poly", [])], round(e.get("zc", 0), 4), round(e.get("u0", 0), 4)]).encode())
        fp[name] = dict(n_elements=len(items), digest=h.hexdigest()[:16])
    if save or not os.path.exists(GOLD):
        json.dump(dict(version=T.LIBRARY.get("version"), imprints=fp), open(GOLD, "w"), ensure_ascii=False, indent=1); print("reference written:", GOLD)
    gold = json.load(open(GOLD))["imprints"]; diff = sorted(k_ for k_ in fp if gold.get(k_) != fp[k_])
    check("layout fingerprints = reference", not diff and set(gold) == set(fp), {"n_layouts": len(fp), "otherwise": diff})
    with tempfile.TemporaryDirectory() as d:                                     # 6. plan: the description's fences as a list
        open(os.path.join(d, "constructions.yaml"), "w", encoding="utf-8").write(
            "timber:\n  fences: [{path: [[0, 0], [6, 0]], kind: wattle}, {path: [[0, 2], [6, 2]]}]\n")
        p0, p1 = T.resolve(d, {}), T.resolve(d, {"fences": "wattle"})
        k0, k1 = [f["kind"] for f in p0["timber"]["fences"]], [f["kind"] for f in p1["timber"]["fences"]]
        try: T.resolve(d, {"fences": "log_oblo"}); bad = "no failure"
        except SystemExit as e: bad = str(e)
        check("plan: fences as a list, a fence without a kind of its own takes the fences knob", k0 == ["wattle", "palisade"] and k1 == ["wattle", "wattle"]
              and {"palisade", "wattle"} <= set(p0["kinds"]) and "not a fence kind" in bad, {"without knob": k0, "fences=wattle": k1, "fences=log_oblo": bad[:40]})
    ok = sum(r["ok"] for r in res); print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False)); sys.exit(len(res) - ok)


main()
