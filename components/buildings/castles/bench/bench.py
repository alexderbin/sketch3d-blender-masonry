"""Reference run for castles (siting/castle_site.py → castles/castle_plan.py → castles/bl_castle.py): the site
type at known places, the plan with checks across different places, cultures, eras and purposes, the Blender
build. Light (Blender ~0.5 GB, ~2 min), no resource guard needed.
.venv/bin/python -m components.buildings.castles.bench.bench [output folder]   → OK/FAIL lines, __RESULT__, exit code = number of failures"""
import json, os, subprocess, sys, time

COMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # root of "buildings" (<part>/bench/this file)
from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "out", "bench_castle")
from components.buildings.castles import castle_plan as CP                                                       # noqa: E402
from components.buildings.siting import castle_site as CS                                                       # noqa: E402
import core.paths

os.makedirs(OUT, exist_ok=True); res, t0 = [], time.time()
P = lambda n: os.path.join(ROOT, "out/relief", n)


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


# 1. site type at known places (catalog presets)
for place, want in (("cat_fortress_hill_1", ("summit",)), ("cat_desert_canyon_2", ("spur", "rock", "summit")),
                    ("cat_crater_battlefield_2", ("plain",)), ("cat_river_crossing_1", ("shore",))):
    S = CS.read(P(place), area=3000)
    check(f"place {place}: type {S['type']}", S["type"] in want, {"type": S["type_ru"], "area": S["area_m2"], "edge": S["edge_frac"]})

# 2. plan: all checks at "good" places; special modes have their own features
cases = [("hill, German", "cat_fortress_hill_1", dict(culture="german")),
         ("canyon cape, French", "cat_desert_canyon_2", dict(culture="french")),
         ("plain, English, concentric", "cat_crater_battlefield_2", dict(culture="english")),
         ("plain, bastions", "cat_crater_battlefield_2", dict(culture="french", era="artillery", scale="large")),
         ("mountain pass, blocking fort", "cat_alpine_pass_1", dict(culture="german", purpose="pass_fort")),
         ("rock by a river", "cat_real_river_crossing_1", dict(culture="german", site="rock"))]
plans = []
for name, place, spec in cases:
    out = os.path.join(OUT, place + "_" + "_".join(f"{v}" for v in spec.values()))
    t = time.time(); R = CP.plan(P(place), out=out, **spec); dt = time.time() - t
    fails = [c["what"] for c in R["checks"] if not c["ok"]]
    hard = [f for f in fails if not f.startswith(("approach to the gate: ", "ramp", "main buildings"))]   # road and program are soft
    check(f"plan {name}: mandatory checks", not hard, {"non_passing": fails, "s": round(dt, 1), **R["stats"]})
    if spec.get("era") == "artillery":
        check("bastions: no towers, there are ravelins", R["stats"]["n_towers"] == 0 and len(R["wards"][0].get("ravelins", [])) >= 4, R["stats"])
    if spec.get("purpose") == "pass_fort":
        check("barrier: wall wings and two gateways of the outpost", sum(w["kind"] == "barrier" for w in R["walls"]) == 2
              and sum(g.get("ward") != "barbican" for g in R["gates"]) == 2, R["site"]["type_ru"])
    if spec.get("site") == "rock":
        check("rock: the main building spans the whole rock", any(b["ward"] == "inner" for b in R["buildings"]), [b["kind"] for b in R["buildings"]])
    plans.append(out)

# 3. unique elements: on the canyon cape — elements with a reason, landmarks, uniqueness of the plots, logic
R = json.load(open(os.path.join(plans[1], "castle.json")))
kinds = {f["kind"] for f in R["features"]}
check("elements: ≥ 15, each with a reason", len(R["features"]) >= 15 and all(f.get("why") for f in R["features"]), {"n_elements": len(R["features"]), "n_kinds": len(kinds)})
fc = [c for c in R["checks"] if c["what"].startswith(("unique", "landmark", "garderobe", "apse", "wicket", "machicolation", "hoarding", "palace_windows", "oriel"))]
check("elements: logic and uniqueness checks", fc and all(c["ok"] for c in fc), [c["what"] for c in fc if not c["ok"]] or len(fc))
r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "castles", "bl_castle.py"), "--", plans[1], "views=feat", "res=480x320"], capture_output=True, text=True)
nf = len([f for f in os.listdir(plans[1]) if f.startswith("view_feat_")])
check("elements close-up: ≥ 4 kinds", nf >= 4, nf)

# 3a. complex castles: stairstep wall, rock, terraces, annexes, houses as a wall, two-tier defense, projecting
#     tower, logic markup, the knobs change what they promise
cx = {}
for key, place, spec in (("frame", "cat_desert_canyon_2", dict(goal="frame")), ("hillside", "cat_real_volcanic_plateau_1", dict(goal="detour")),
                         ("plain", "cat_crater_battlefield_2", dict(goal="detour", culture="french", era="late")),
                         ("order_castle", "cat_real_river_crossing_1", dict(purpose="order_castle", goal="history")),
                         ("sparse", "cat_crater_battlefield_2", dict(culture="french", era="late", density="0.05")),
                         ("kremlin", "cat_real_volcanic_plateau_1", dict(goal="frame", culture="kremlin"))):
    out = os.path.join(OUT, "cx_" + key); cx[key] = CP.plan(P(place), out=out, **spec); cx[key]["_out"] = out
R = cx["frame"]; st_ = R["stats"]
check("complex: the cape with goal 'frame' — stairstep walls and rock plinths", st_["n_ledges"] > 0 and st_["n_plinths"] > 0, st_)
steps_ = {L["step"] for L in R["logic"]}
check("complex: logic markup — all steps, each record has a 'why'",
      {"place", "courts", "gate", "main tower", "towers", "walls", "rock", "buildings", "roofs"} <= steps_ and all(L.get("why") for L in R["logic"]),
      {"n_records": len(R["logic"]), "steps": sorted(steps_)})
neat = [(k_, c) for k_ in ("frame", "hillside", "kremlin") for c in cx[k_]["checks"] if "clearances" in c["what"]]
check("complex: tidy stairstep — the visible ledges are equal (for the kremlin — merlon by merlon) and stand in the clearances of the row of merlons",
      len(neat) == 3 and all(c["ok"] for _, c in neat), [(k_, c.get("n_ledges"), c.get("non_conforming")) for k_, c in neat])
R = cx["hillside"]; ret = [w for w in R["walls"] if w["kind"] == "retaining"]
check("complex: hillside with goal 'detour' — terraces with retaining walls and stairs", R["stats"]["n_terraces"] >= 2 and ret and all(w.get("stair") is not None for w in ret)
      and all(c["ok"] for c in R["checks"] if c["what"].startswith("terraces")), {"n_levels": R["stats"]["n_terraces"], "retaining": len(ret)})
R = cx["plain"]; R0 = cx["sparse"]
check("complex: annexes by the density knob (0.65 vs 0.05) and the court is not tighter than the norm",
      R["stats"]["n_annexes"] >= 4 and R0["stats"]["n_annexes"] == 0 and all(c["ok"] for c in R["checks"] if c["what"].startswith("court")),
      {"0,65": R["stats"]["n_annexes"], "0,05": R0["stats"]["n_annexes"]})
check("complex: Pierrefonds — a covered walk over the machicolations (late French)", any(f["kind"] == "covered_walk" and f.get("over_mach") for f in R["features"]),
      sum(1 for f in R["features"] if f.get("over_mach")))
R = cx["order_castle"]; dk = [f for f in R["features"] if f["kind"] == "dansker"]
check("complex: the Order castle — a projecting tower on an arcade, the check passed", dk and all(c["ok"] for c in R["checks"] if c["what"].startswith("projecting")),
      [(f["arches"], f["len"]) for f in dk])
r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "castles", "bl_castle.py"), "--", cx["frame"]["_out"], "views=cx", "res=480x320"], capture_output=True, text=True)
ncx = len([f for f in os.listdir(cx["frame"]["_out"]) if f.startswith("view_cx_")])
check("complex close-up: ≥ 3 kinds (plinth, stairstep, edge, house as a wall)", ncx >= 3, ncx)

# 3b. fantasy: towers by tiers, the sorcerer's ring, ring tiers, a bridge over a chasm, bridges between
#     towers. The vampire is on a rock by a river with a chasm at the gate: cat_real_river_crossing_1 was
#     rebuilt since — it is now a slope without a chasm, so there should be no bridge
fx = {}
for key, place, spec in (("mage", "cat_real_river_crossing_1", dict(purpose="mage_tower")), ("sorcerer", "cat_crater_battlefield_2", dict(purpose="sorcerer")),
                         ("vampire", "cat_real_river_crossing_2", dict(purpose="vampire_castle")), ("citadel", "cat_fortress_hill_3", dict(purpose="citadel", culture="gothic"))):
    out = os.path.join(OUT, "fx_" + key); fx[key] = CP.plan(P(place), out=out, **spec); fx[key]["_out"] = out
def hard_fails(R): return [c["what"] for c in R["checks"] if not c["ok"] and not c["what"].startswith(("approach to the gate: ", "ramp", "main buildings", "unique"))]
R = fx["mage"]; k_ = R["keep"]
check("fantasy: mage tower — 4+ tiers, slender, checks", k_ and k_.get("style") == "mage" and len(k_["stages"]) >= 4 and not hard_fails(R),
      {"n_tiers": len(k_["stages"]) if k_ else 0, "height_m": round(k_["top"] - k_["base"]) if k_ else 0, "no": hard_fails(R)})
R = fx["sorcerer"]; k_ = R["keep"]
check("fantasy: sorcerer — a tower ≥ 100 m in the ring, a top with horns, a crown or spikes", R["wards"][0].get("ring_tower") and k_["top"] - k_["base"] >= 100
      and k_["crown"]["kind"] in ("horns", "crown", "spikes") and not hard_fails(R), {"height_m": round(k_["top"] - k_["base"]), "crown": k_["crown"]["kind"]})
R = fx["vampire"]
check("fantasy: vampire — the main tower in tiers and a bridge over a chasm", R["keep"].get("style") == "vampire" and any(b_.get("kind") == "chasm" for b_ in R["bridges"])
      and not hard_fails(R), [(b_.get("len"), b_.get("depth")) for b_ in R["bridges"] if b_.get("kind") == "chasm"])
R = fx["citadel"]; tw_ = [w for w in R["wards"] if w.get("tier")]
check("fantasy: citadel — ring tiers (walls rise toward the summit, gates on different sides)", len(tw_) >= 2 and not hard_fails(R)
      and all(c["ok"] for c in R["checks"] if "tiers-rings" in c["what"]), {"n_tiers": len(tw_), "no": hard_fails(R)})
for key in ("mage", "sorcerer", "vampire", "citadel"):
    r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "castles", "bl_castle.py"), "--", fx[key]["_out"], "views=aerial", "res=480x320"], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    bl = json.loads(line[len("__RESULT__"):]) if line else {"error": (r.stdout + r.stderr)[-300:]}
    check(f"fantasy Blender: {key}", line is not None and 3000 < bl.get("n_triangles_castle", 0) < 600000, bl)

# 4. Blender build: walls, towers, roofs, buildings are present; the triangle count is reasonable
for out in plans[:3]:
    r = subprocess.run(["blender", "-b", "-P", core.paths.RUN, "--", os.path.join(COMP, "castles", "bl_castle.py"), "--", out, "views=aerial", "res=640x400"],
                       capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    bl = json.loads(line[len("__RESULT__"):]) if line else {"error": (r.stdout + r.stderr)[-300:]}
    check(f"Blender {os.path.basename(out)}", line is not None and 3000 < bl.get("n_triangles_castle", 0) < 600000, bl)    # without ground (0.5 m mesh)

fails = sum(not r["ok"] for r in res)
json.dump(res, open(os.path.join(OUT, "bench.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__", json.dumps({"n_checks": len(res), "n_failures": fails, "s": round(time.time() - t0, 1)}, ensure_ascii=False))
sys.exit(fails)
