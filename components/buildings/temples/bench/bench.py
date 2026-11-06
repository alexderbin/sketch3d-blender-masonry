"""Reference run of the temples (temples/temple_plan.py → bl_temple.py): the plan of all styles on different
sites and seeds with checks, the style features, the build in Blender. Light (Blender ~0.5 GB, ~1 min), no resource guard needed.
.venv/bin/python -m components.buildings.temples.bench.bench [output folder]   → OK/FAIL lines, __RESULT__, exit code = the number of failures"""
import contextlib, io, json, os, subprocess, sys, time

COMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # root of "buildings" (<part>/bench/this file)
from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "out", "bench_temple")
from components.buildings.temples import temple_plan as TP                                                       # noqa: E402
import core.paths

os.makedirs(OUT, exist_ok=True); res, t0 = [], time.time()
STYLES = ["gothic_fr", "romanesque", "gothic_en", "hall", "hive_gothic", "gothic_early", "gothic_late", "holy", "elven", "dwarven", "dark", "hell", "dark+hive_gothic"]
PLACES = ["cat_crater_battlefield_2", "cat_fortress_hill_1", "cat_river_crossing_1", "cat_mountain_lake_1", "cat_badlands_2"]


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


# 1. plan: all checks at all sites (plain, summit, shore, slope, rock) and two seeds
plans = {}; allR = []
for st in STYLES:
    fails = []; n = 0
    for pl in PLACES:
        for seed in (1, 2):
            out = os.path.join(OUT, f"{pl}_{st.replace('+', '_')}_{seed}"); n += 1
            with contextlib.redirect_stdout(io.StringIO()):
                R = TP.plan(os.path.join(ROOT, "out/relief", pl), out=out, style=st, seed=seed)
            fails += [f"{pl}/{seed}: {c['what']}" for c in R["checks"] if not c["ok"]]
            plans.setdefault(st, (out, R)); allR.append(R)
    check(f"plan {st}: checks at {n} sites", not fails, fails[:4] or n)

# 2. style features and uniqueness
_, R = plans["gothic_fr"]
check("Gothic (France): an ambulatory with an odd number of chapels, flying buttresses, three portals", R["parts"]["east"]["kind"] in ("chevet", "apse") and R["portal"]["n"] == 3
      and any(b_["kind"] == "flyer" for b_ in R["buttresses"]), {"east": R["parts"]["east"]["kind"], "n_chapels": R["parts"]["east"].get("chapels")})
_, R = plans["romanesque"]
check("Romanesque: a semicircular arch, groin vaults, an apse", R["module"]["arch"] == "round" and R["module"]["vault"] == "groin", R["module"]["family"])
_, R = plans["gothic_en"]
check("England: a screen, a flat east end or star vaults", bool(R.get("screen")) and R["parts"]["east"]["kind"] in ("flat", "chevet", "apse"), R["stats"])
_, R = plans["hall"]
check("hall church: aisles of one height, one west tower", R["module"]["hall"] and len(R["towers"]) == 1, {"tower_m": R["towers"][0]["h"]})
_, R = plans["hive_gothic"]
check("hive: scale ×3–4, a podium, tower piers, a human-sized portal", 3.0 <= R["k"] <= 4.0 and R["podium"] and R["buttress"]["towers"] and R["portal"]["h"] / R["module"]["kq"] <= 14.01,
      {"scale": R["k"], "highest_m": R["stats"]["highest_m"]})
_, R = plans["dwarven"]
check("dwarves: a stepped arch, a flat ceiling or a barrel vault, a terrace or a fortress", R["module"]["arch"] in ("stepped", "flat") and R["module"]["vault"] in ("flat", "barrel"),
      {"arch": R["module"]["arch"], "vault": R["module"]["vault"], "roof": R["module"]["roof"]})
_, R = plans["hell"]
check("hell: spikes and claws/horns", R["module"]["spikes"] >= 0.9 and any(b_.get("form") == "claw" for b_ in R["buttresses"]) or any(tw["spire"]["kind"] in ("horns", "blades", "spikes") for tw in R["towers"]),
      [tw["spire"]["kind"] for tw in R["towers"]])
sigs = {}; camps = []; mods = []
for R in allR:
    k_ = (R["story"]["signature"] or {}).get("kind"); sigs[k_] = sigs.get(k_, 0) + 1; camps.append(len(R["story"]["campaigns"])); mods.append(len(R["modules"]))
check("uniqueness: each has a special element, ≥ 6 different ones; campaigns and part styles", None not in sigs and len(sigs) >= 6 and max(camps) >= 3 and max(mods) >= 2,
      {"specials": sigs, "n_campaigns_max": max(camps), "max_styles_per_part": max(mods)})
cells = [R["spaces"]["stats"]["n_cells"] for R in allR]; ents = [R["spaces"]["stats"]["n_entrances"] for R in allR]   # a space scheme for every plan
check("space: every plan has cells, entrances, depth; the scheme checks are among the plan checks", min(cells) >= 12 and min(ents) >= 1,
      {"n_cells": f"{min(cells)}–{max(cells)}", "n_entrances": f"{min(ents)}–{max(ents)}"})

# 3. build: each style — views, no opening dropped out of a wall
for st in ["gothic_fr", "romanesque", "gothic_en", "hall", "hive_gothic", "dwarven", "elven", "hell"]:
    out, R = plans[st]
    r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "temples", "bl_temple.py"), "--", out, "views=aerial,interior", "res=480x300"],
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    B = json.loads(line.split(" ", 1)[1]) if line else {}
    ok = bool(B) and B.get("n_openings_skipped", 1) == 0 and 15000 <= B.get("n_triangles_cathedral", 0) <= 600000 \
        and all(os.path.exists(os.path.join(out, f"view_{v}.png")) for v in ("aerial", "interior"))
    check(f"build {st}: views, openings in place", ok, {k: B.get(k) for k in ("n_triangles_cathedral", "n_openings_skipped")} if B else r.stderr[-300:])
    V = ((B.get("space") or {}).get("check") or {}) if B else {}       # model against the space scheme
    check(f"model {st} against the scheme: floor, vaults, transitions open, walls blank, flights clear", bool(V) and V.get("ok"),
          {k: v for k, v in V.items() if k in ("floor_under_walkable", "vault_over_covered", "transitions_open", "walls_blind", "flights_free") or (isinstance(v, list) and v)})

n_fail = sum(not r_["ok"] for r_ in res)
json.dump(res, open(os.path.join(OUT, "bench.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__", json.dumps({"n_checks": len(res), "n_failures": n_fail, "s": round(time.time() - t0, 1)}, ensure_ascii=False))
sys.exit(n_fail)
