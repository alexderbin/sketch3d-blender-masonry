"""Reference bench of the "ensembles" part: every settlement archetype at its catalog place, two seeds each.
The plan gets built and the plan checks pass (street connectivity, plot facades, houses neither overlap nor stand on
streets, density within the archetype's measures, height limit); the summary goes into a table. Plan fingerprints
(what moved) — the shared ../../bench/golden.py.

.venv/bin/python -m components.buildings.ensembles.bench.bench [folder] [seeds=1,2] [only=town,hive]
  → one line per run, __RESULT__, exit code = number of runs with failed checks or a crash"""
import contextlib, io, json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__)); PART = os.path.dirname(HERE); COMP = os.path.dirname(PART)
ROOT = COMP
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
from components.buildings import buildings as B                                                          # noqa: E402

# archetype → catalog place (terrain fitted to the design: polis — on a hill, subura — in a lowland,
# hive — on flat ground by a river)
PLACES = {"polis": "cat_fortress_hill_1", "colony": "cat_river_crossing_1", "subura": "cat_jungle_valley_1", "town": "cat_river_crossing_2",
          "posad": "cat_fortress_hill_2", "castle_inside": "cat_fortress_hill_3", "village": "cat_river_crossing_3", "hive": "cat_river_crossing_3"}


def main():
    a = [x for x in sys.argv[1:] if "=" not in x]; kw = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    out = os.path.abspath(a[0] if a else os.path.join(ROOT, "out", "bench_buildings", "ensembles"))
    seeds = [int(s) for s in kw.get("seeds", "1,2").split(",")]; only = kw.get("only", ",".join(PLACES)).split(",")
    rows = []; bad = 0; t0 = time.time()
    for kind in only:
        for seed in seeds:
            t = time.time(); f = os.path.join(out, f"{kind}_{seed}")
            try:
                with contextlib.redirect_stdout(io.StringIO()): R = B.plan("settlement", os.path.join(ROOT, "out", "relief", PLACES[kind]), out=f, type=kind, seed=seed)
                fail = [c["what"] for c in R["checks"] if not c["ok"]]; st = R["stats"]
                row = {"archetype": kind, "seed": seed, "n_houses": st["n_houses"], "GSI": st["GSI"], "FSI": st["FSI"], "streets": st["frac_streets"],
                       "n_stairs": st["n_stairs"], "checks": f"{len(R['checks']) - len(fail)}/{len(R['checks'])}", "s": round(time.time() - t, 1)}
                print(("OK   " if not fail else "FAIL ") + json.dumps(row, ensure_ascii=False), *(["failed:", fail] if fail else []), flush=True)
                bad += bool(fail)
            except Exception as e:
                row = {"archetype": kind, "seed": seed, "error": repr(e)[:200]}; bad += 1
                print("FAIL " + json.dumps(row, ensure_ascii=False)); traceback.print_exc(limit=3)
            rows.append(row)
    json.dump(rows, open(os.path.join(out, "bench.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps({"n_runs": len(rows), "n_failed": bad, "s": round(time.time() - t0, 1), "folder": out}, ensure_ascii=False))
    return bad


if __name__ == "__main__":
    sys.exit(main())
