"""Reference run for edifices (edifices/edifice_plan.py → bl_edifice.py): every type in its own style and in a
foreign one (type × style), on a plain and on a hill, with plan checks; a Blender build of each type. Light
(~1–2 min).
.venv/bin/python -m components.buildings.edifices.bench.bench [output folder]   → OK/FAIL lines, __RESULT__, exit code = number of failures"""
import contextlib, io, json, os, subprocess, sys, time

COMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # root of "buildings" (<part>/bench/this file)
from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "out", "bench_edifice")
from components.buildings.edifices import edifice_plan as EP                                                      # noqa: E402
import core.paths

os.makedirs(OUT, exist_ok=True); res, t0 = [], time.time()
CASES = [("ziggurat", "mesopotamian"), ("ziggurat", "occult"), ("step_pyramid", "mesoamerican"), ("step_pyramid", "dark"), ("temple_mountain", "khmer"),
         ("temple_mountain", "hive_gothic"), ("spiral_tower", "mesopotamian"), ("spiral_tower", "elven"), ("occult_ring", "occult"), ("light_order_temple", "light_order"),
         ("dark_order_temple", "dark_order"), ("dark_citadel", "dark"), ("dark_citadel", "hell"), ("axial_temple", "egyptian"), ("domed", "orthodox"), ("domed", "forge_cult"),
         ("domed", "byzantine")]
PLACES = ["cat_crater_battlefield_2", "cat_fortress_hill_1"]


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


plans = {}
for typ, st in CASES:                                                            # 1. plan: checks on a plain and a hill, with two seeds
    fails = []; n = 0
    for pl in PLACES:
        for seed in (1, 2):
            out = os.path.join(OUT, f"{typ}_{st}_{pl}_{seed}"); n += 1
            with contextlib.redirect_stdout(io.StringIO()):
                R = EP.plan(os.path.join(ROOT, "out/relief", pl), out=out, type=typ, style=st, seed=seed)
            fails += [f"{pl}/{seed}: {c['what']}" for c in R["checks"] if not c["ok"]]
            plans.setdefault((typ, st), (out, R))
    check(f"plan {typ} × {st}: checks at {n} places", not fails, fails[:3] or n)
# 2. features of the types
R = plans[("ziggurat", "mesopotamian")][1]
check("ziggurat: tiers going up and a shrine on top", sum(p["type"] == "tier" for p in R["parts"]) >= 3 and any(p["type"] == "shrine" for p in R["parts"]), R["stats"])
R = plans[("temple_mountain", "khmer")][1]
check("temple-mountain: galleries, a quincunx (the central prang is taller than the corner ones)", any(p["type"] == "gallery" for p in R["parts"])
      and max(p["h"] + p["z0"] for p in R["parts"] if p["type"] == "prang") == max(p["h"] + p["z0"] for p in R["parts"] if p["type"] == "prang" and p["c"] == [0.0, 0.0]),
      R["stats"]["highest_m"])
R = plans[("light_order_temple", "light_order")][1]
check("Light order temple: five minarets, the central one taller", sum(p["type"] == "minaret" for p in R["parts"]) == 5, R["stats"]["highest_m"])
R = plans[("occult_ring", "occult")][1]
check("occult: a sunken circle and a ring of an odd number", any(p["type"] == "court" and p["depth"] > 0 for p in R["parts"])
      and any(p["type"] == "ring" and p["n"] % 2 == 1 and p["n"] > 1 for p in R["parts"]), [p["n"] for p in R["parts"] if p["type"] == "ring"][:2])
R = plans[("axial_temple", "egyptian")][1]
check("axial temple: two pylons, a court, a hypostyle, a sanctuary on the axis", sum(p["type"] == "pylon" for p in R["parts"]) == 2 and any(p["type"] == "hall" for p in R["parts"]), R["stats"])
# 3. build of each type
for (typ, st), (out, R) in plans.items():
    r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "edifices", "bl_edifice.py"), "--", out, "views=aerial,front", "res=480x300"], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None); Bz = json.loads(line.split(" ", 1)[1]) if line else {}
    check(f"build {typ} × {st}", bool(Bz) and Bz.get("n_triangles", 0) > 1000 and os.path.exists(os.path.join(out, "view_front.png")) and Bz.get("n_openings_skipped", 1) == 0,
          Bz.get("n_parts") or r.stderr[-300:])
    V = ((Bz.get("space") or {}).get("check") or {}) if Bz else {}     # model against the space scheme
    check(f"model {typ} × {st} against the scheme", bool(V) and V.get("ok"), {k: v for k, v in V.items() if isinstance(v, list) and v} or "floors, vaults, transitions, walls, flights — ok")
n_fail = sum(not r_["ok"] for r_ in res)
json.dump(res, open(os.path.join(OUT, "bench.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__", json.dumps({"n_checks": len(res), "n_failures": n_fail, "s": round(time.time() - t0, 1)}, ensure_ascii=False))
sys.exit(n_fail)
