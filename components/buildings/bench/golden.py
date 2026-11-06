"""Reference result of the "buildings" component: fingerprints of the plans of castles, temples, edifices,
settlements and classics over a fixed matrix of places, intents and seeds — a hash of the plan (numbers to
1e-5, no output paths) and a hash of the relief edits; a summary of the plan — for reading the difference.
With --blender — also the fingerprints of the models (bl_fingerprint.py: vertices in world space, color Col,
roles and cells of faces) on a small sample. A change that alters the result must be deliberate: --save, the
reason goes into the commit message.

.venv/bin/python -m components.buildings.bench.golden [--save] [--blender] [only=castles,temples,edifices,settlements,classics] [golden=file] [out=folder]
  → OK/DIFF lines, __RESULT__, exit code = number of differences."""
import contextlib, hashlib, io, json, os, subprocess, sys, time

import numpy as np
import core.paths

HERE = os.path.dirname(os.path.abspath(__file__)); COMP = os.path.dirname(HERE)
ROOT = COMP
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
ARGS = sys.argv[1:]; KW = dict(a.split("=", 1) for a in ARGS if "=" in a); FLAGS = {a for a in ARGS if a.startswith("--")}
from components.buildings import arch_parts                                                              # noqa: E402
BL = {"castle": os.path.join(COMP, "castles", "bl_castle.py"), "temple": os.path.join(COMP, "temples", "bl_temple.py"),
      "edifice": os.path.join(COMP, "edifices", "bl_edifice.py")}
GOLDEN = os.path.abspath(KW.get("golden", os.path.join(HERE, "golden.json")))
OUT = os.path.abspath(KW.get("out", os.path.join(ROOT, "out", "bench_buildings_golden")))
ONLY = set(KW.get("only", "castles,temples,edifices,settlements,classics").split(","))
P = lambda n: os.path.join(ROOT, "out/relief", n)

# Matrix: the same places and intents as in the parts' references (bench/ of castles, temples, edifices)
CASTLES = [("cat_fortress_hill_1", dict(culture="german")), ("cat_desert_canyon_2", dict(culture="french")),
           ("cat_crater_battlefield_2", dict(culture="english")), ("cat_crater_battlefield_2", dict(culture="french", era="artillery", scale="large")),
           ("cat_alpine_pass_1", dict(culture="german", purpose="pass_fort")), ("cat_real_river_crossing_1", dict(culture="german", site="rock")),
           ("cat_desert_canyon_2", dict(goal="frame")), ("cat_real_volcanic_plateau_1", dict(goal="detour")),
           ("cat_crater_battlefield_2", dict(goal="detour", culture="french", era="late")), ("cat_real_river_crossing_1", dict(purpose="order_castle", goal="history")),
           ("cat_crater_battlefield_2", dict(culture="french", era="late", density="0.05")), ("cat_real_volcanic_plateau_1", dict(goal="frame", culture="kremlin")),
           ("cat_real_river_crossing_1", dict(purpose="mage_tower")), ("cat_crater_battlefield_2", dict(purpose="sorcerer")),
           ("cat_real_river_crossing_1", dict(purpose="vampire_castle")), ("cat_fortress_hill_3", dict(purpose="citadel", culture="gothic"))]
STYLES = ["gothic_fr", "romanesque", "gothic_en", "hall", "hive_gothic", "gothic_early", "gothic_late", "holy", "elven", "dwarven", "dark", "hell", "dark+hive_gothic"]
T_PLACES = ["cat_crater_battlefield_2", "cat_fortress_hill_1", "cat_river_crossing_1", "cat_mountain_lake_1", "cat_badlands_2"]
EDIFICES = [("ziggurat", "mesopotamian"), ("ziggurat", "occult"), ("step_pyramid", "mesoamerican"), ("step_pyramid", "dark"), ("temple_mountain", "khmer"),
            ("temple_mountain", "hive_gothic"), ("spiral_tower", "mesopotamian"), ("spiral_tower", "elven"), ("occult_ring", "occult"), ("light_order_temple", "light_order"),
            ("dark_order_temple", "dark_order"), ("dark_citadel", "dark"), ("dark_citadel", "hell"), ("axial_temple", "egyptian"), ("domed", "orthodox"), ("domed", "forge_cult"),
            ("domed", "byzantine")]
E_PLACES = ["cat_crater_battlefield_2", "cat_fortress_hill_1"]
SETTLEMENTS = [("polis", "cat_fortress_hill_1"), ("colony", "cat_river_crossing_1"), ("subura", "cat_jungle_valley_1"), ("town", "cat_river_crossing_2"),
               ("posad", "cat_fortress_hill_2"), ("castle_inside", "cat_fortress_hill_3"), ("village", "cat_river_crossing_3"), ("hive", "cat_river_crossing_3")]
MODELS = [("castle", "cat_fortress_hill_1|culture=german"), ("castle", "cat_real_river_crossing_1|purpose=order_castle,goal=history"),
          ("castle", "cat_real_river_crossing_1|purpose=mage_tower"), ("temple", "cat_crater_battlefield_2|gothic_fr|1"),
          ("temple", "cat_fortress_hill_1|hive_gothic|1"), ("temple", "cat_mountain_lake_1|romanesque|2"),
          ("edifice", "ziggurat|mesopotamian|cat_crater_battlefield_2|1"), ("edifice", "temple_mountain|khmer|cat_crater_battlefield_2|1"),
          ("edifice", "domed|byzantine|cat_fortress_hill_1|2")]
DROP = {"out", "_out"}


def canon(x):
    """Canonical form of a plan: keys in order, numbers rounded to 1e-5 (−0 → 0), arrays as lists, output paths
    dropped."""
    if isinstance(x, dict): return {str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0])) if k not in DROP}
    if isinstance(x, (list, tuple)): return [canon(v) for v in x]
    if isinstance(x, np.ndarray): return canon(x.tolist())
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): v = round(float(x), 5); return 0.0 if v == 0 else v
    return x


def h(s): return hashlib.sha1(s if isinstance(s, bytes) else s.encode()).hexdigest()[:16]


def relief_hash(out):
    f = os.path.join(out, "relief.npz")
    if not os.path.exists(f): return None
    Z = np.load(f); m = hashlib.sha1()
    for k in sorted(Z.files): m.update(k.encode()); m.update(np.round(np.asarray(Z[k], np.float64), 4).tobytes())
    return m.hexdigest()[:16]


def stamp(R, out):
    chk = R.get("checks", [])
    return {"plan": h(json.dumps(canon(R), ensure_ascii=False, sort_keys=True)), "relief": relief_hash(out),
            "checks": f"{sum(bool(c['ok']) for c in chk)}/{len(chk)}", "summary": canon(R.get("stats", {}))}


def run_plans():
    got = {}
    if "castles" in ONLY:
        from components.buildings.castles import castle_plan as CP
        for place, spec in CASTLES:
            key = "castle|" + place + "|" + ",".join(f"{k}={v}" for k, v in spec.items()); out = os.path.join(OUT, h(key))
            with contextlib.redirect_stdout(io.StringIO()): R = CP.plan(P(place), out=out, **spec)
            got[key] = stamp(R, out)
    if "temples" in ONLY:
        from components.buildings.temples import temple_plan as TP
        for st in STYLES:
            for pl in T_PLACES:
                for seed in (1, 2):
                    key = f"temple|{pl}|{st}|{seed}"; out = os.path.join(OUT, h(key))
                    with contextlib.redirect_stdout(io.StringIO()): R = TP.plan(P(pl), out=out, style=st, seed=seed)
                    got[key] = stamp(R, out)
    if "edifices" in ONLY:
        from components.buildings.edifices import edifice_plan as EP
        for typ, st in EDIFICES:
            for pl in E_PLACES:
                for seed in (1, 2):
                    key = f"edifice|{typ}|{st}|{pl}|{seed}"; out = os.path.join(OUT, h(key))
                    with contextlib.redirect_stdout(io.StringIO()): R = EP.plan(P(pl), out=out, type=typ, style=st, seed=seed)
                    got[key] = stamp(R, out)
    if "settlements" in ONLY:                                                    # settlements go through the entry (an ensemble calls the types: house, castle)
        from components.buildings import buildings as B
        for typ, pl in SETTLEMENTS:
            key = f"settlement|{typ}|{pl}|1"; out = os.path.join(OUT, h(key))
            with contextlib.redirect_stdout(io.StringIO()): R = B.plan("settlement", P(pl), out=out, type=typ, seed=1)
            got[key] = stamp(R, out)
    if "classics" in ONLY:                                                     # classics — a showcase of all kinds (temples, stoa, palaestra …)
        from components.buildings import buildings as B
        for seed in (1, 2):
            key = f"classics|all|cat_river_crossing_1|{seed}"; out = os.path.join(OUT, h(key))
            with contextlib.redirect_stdout(io.StringIO()): R = B.plan("classics", P("cat_river_crossing_1"), out=out, kind="all", seed=seed)
            got[key] = stamp(R, out)
    return got


def model_key(kind, spec):
    if kind == "castle": place, s = spec.split("|"); return "castle|" + place + "|" + s
    if kind == "temple": pl, st, seed = spec.split("|"); return f"temple|{pl}|{st}|{seed}"
    typ, st, pl, seed = spec.split("|"); return f"edifice|{typ}|{st}|{pl}|{seed}"


def run_models():
    got = {}
    for kind, spec in MODELS:
        key = model_key(kind, spec); out = os.path.join(OUT, h(key))
        if not os.path.exists(os.path.join(out, f"{kind}.json")): got["model|" + key] = {"error": "no plan (run the plans first)"}; continue
        r = subprocess.run(["blender", "-b", "--python-exit-code", "1", "-P", core.paths.RUN, "--", os.path.join(HERE, "bl_fingerprint.py"), "--", BL[kind], out],
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("__FP__")), None)
        got["model|" + key] = json.loads(line[7:]) if line else {"error": (r.stderr or r.stdout)[-400:]}
    return got


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    got = run_plans()
    if "--blender" in FLAGS: got.update(run_models())
    old = json.load(open(GOLDEN)) if os.path.exists(GOLDEN) else {}
    if "--save" in FLAGS:
        keep = {k: v for k, v in old.items() if k not in got}; json.dump(dict(keep, **got), open(GOLDEN, "w"), ensure_ascii=False, indent=1, sort_keys=True)
        print("__RESULT__", json.dumps({"saved": len(got), "file": GOLDEN, "s": round(time.time() - t0, 1)}, ensure_ascii=False)); return 0
    diff = []
    for k, v in got.items():
        o = old.get(k)
        if o is None: print("NEW    ", k); continue
        keys = ("plan", "relief") if not k.startswith("model|") else ("vertices", "topology", "color", "roles", "cells", "n_objects", "n_faces")
        bad = [x for x in keys if o.get(x) != v.get(x)]
        if bad: diff.append(k); print("DIFF  ", k, {x: (o.get(x), v.get(x)) for x in bad}, "summary before/after:", o.get("summary") if o.get("summary") != v.get("summary") else "same", v.get("summary") if o.get("summary") != v.get("summary") else "")
        else: print("OK       ", k)
    print("__RESULT__", json.dumps({"compared": len(got), "diff": len(diff), "missing_in_reference": sum(k not in old for k in got), "s": round(time.time() - t0, 1)}, ensure_ascii=False))
    return len(diff)


if __name__ == "__main__":
    sys.exit(main())
