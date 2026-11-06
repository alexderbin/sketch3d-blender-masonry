"""All checks of the "buildings" component at once: couplings of the parts → fingerprints (the reference) → the
parts' references (castles, temples, edifices, settlements, influences: features, plan checks, knobs, the
Blender build). After editing the core, run this.

.venv/bin/python -m components.buildings.bench.all [--blender]   → one line per check, __RESULT__, exit code = number of failures"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); COMP = os.path.dirname(HERE)
PY = sys.executable
OUT = os.path.join(os.path.dirname(os.path.dirname(COMP)), "out", "bench_buildings")   # not out/bench_castle: masonry takes its reference from there (cx_order_castle)
RUNS = [("links", [PY, "-m", "components.buildings.bench.deps"]),
        ("imprints", [PY, "-m", "components.buildings.bench.golden"] + (["--blender"] if "--blender" in sys.argv else [])),
        ("castles", [PY, "-m", "components.buildings.castles.bench.bench", os.path.join(OUT, "castles")]),
        ("temples", [PY, "-m", "components.buildings.temples.bench.bench", os.path.join(OUT, "temples")]),
        ("edifices", [PY, "-m", "components.buildings.edifices.bench.bench", os.path.join(OUT, "edifices")]),
        ("settlements", [PY, "-m", "components.buildings.ensembles.bench.bench", os.path.join(OUT, "ensembles")]),
        ("influences", [PY, "-m", "components.buildings.influences.bench.bench", os.path.join(OUT, "influences")])]

t0 = time.time(); res = {}
for name, cmd in RUNS:
    t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("__RESULT__")), "")
    bad = [l for l in r.stdout.splitlines() if l.startswith(("FAIL", "DIFF"))]
    res[name] = {"code": r.returncode, "s": round(time.time() - t, 1), "result": line[11:300]}
    print(("OK   " if r.returncode == 0 else "FAIL ") + name, f"{res[name]['s']} s", line[11:200], *bad[:6], sep="\n  " if bad else " ", flush=True)
print("__RESULT__", json.dumps({"n_failures": sum(v["code"] != 0 for v in res.values()), "s": round(time.time() - t0, 1),
                                "parts": {k: v["code"] for k, v in res.items()}}, ensure_ascii=False))
sys.exit(sum(v["code"] != 0 for v in res.values()))
