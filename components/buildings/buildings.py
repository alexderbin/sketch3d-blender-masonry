"""Entry of the "buildings" component (see its README): the type registry and three steps for outside callers —
plan, mass, passport. Outside the component only this entry is called, plus the builders as scripts (blender
-b -P <part>/bl_*.py) and the contract (the plan folder, objects with attributes); the modules of the parts
are not called from outside.

.venv/bin/python -m components.buildings.buildings types
.venv/bin/python -m components.buildings.buildings plan <type> <place> [key=value …] [out=folder]   → plan folder, __RESULT__
.venv/bin/python -m components.buildings.buildings build <plan folder> [views=…] [res=…] [blend=1]     → views, __RESULT__ (Blender)
.venv/bin/python -m components.buildings.buildings make <type> <place> [key=value …] [out=folder] [views=…] [res=…] [blend=1]   → plan and mass in one command (the manager's "building" step)
.venv/bin/python -m components.buildings.buildings passport <plan folder>                             → passport (JSON)
.venv/bin/python -m components.buildings.buildings influence <settlement plan folder> at=x,y[;x,y…] [deck=k|z=…]  → power, trade, wealth, district at points
In code: sys.path ← components/buildings; import buildings as B; R = B.plan("temple", place,
style="gothic_fr", out=…); B.build(folder); B.passport(folder); B.influence(folder, points) — the three
influences of a settlement at points (the raster influences.npz; a temple in a town — wealth=W of the
place); in Blender — B.build_in(folder, sc) (the builder as a function over an open scene); an ensemble —
B.module("house") (a house program for a plot). The type is castle | temple | edifice | house | classic |
settlement."""
import importlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from components.buildings import arch_parts                                                              # noqa: E402
import core.paths

# Type registry: part, plan module (plan(place, out=, **intent) → plan, writes <file> to out), plan file,
# builder, intent
TYPES = {
    "castle": {"en": "castle", "part": "castles", "plan": "castle_plan", "file": "castle.json", "builder": "bl_castle.py",
              "intent": "site= purpose= era= culture= scale= mood= goal= density= seed= at= (castles/castles.yaml)"},
    "temple": {"en": "temple", "part": "temples", "plan": "temple_plan", "file": "temple.json", "builder": "bl_temple.py",
             "intent": "style= seed= at= history= wealth= ruin= irregular= gene=value (style/styles.yaml, temples/temples.yaml)"},
    "edifice": {"en": "edifice", "part": "edifices", "plan": "edifice_plan", "file": "edifice.json", "builder": "bl_edifice.py",
                   "intent": "type= style= seed= at= gene=value (edifices/edifices.yaml)"},
    "house": {"en": "house", "part": "houses", "plan": "house_plan", "file": "house.json", "builder": "bl_house.py",
            "intent": "kind= seed= at= (houses/houses.yaml; in a settlement — by a plot program)"},
    "classics": {"en": "classic", "part": "classics", "plan": "classic_plan", "file": "classic.json", "builder": "bl_classic.py",
                 "intent": "kind= seed= at= (classics/classics.yaml: peripteros, prostyle, amphiprostyle, in_antis, podium, tholos, propylon, stoa, bouleuterion, palaestra, altar, theatre; all — a showcase; in a settlement — by a frame program)"},
    "settlement": {"en": "settlement", "part": "ensembles", "plan": "ensemble_plan", "file": "ensemble.json", "builder": "bl_ensemble.py",
                  "intent": "type= seed= at= size= host= (ensembles/settlements.yaml: polis, colony, subura, town, posad, castle_inside, village, hive)"},
}
# the type's build function over an open Blender scene (step by step, in an ensemble): the builder module, a
# function (folder, sc=…, tag=…) → (objects, plan)
BUILD_IN = {"castle": ("bl_castle", "build_castle"), "temple": ("bl_temple", "build_temple"), "edifice": ("bl_edifice", "build_edifice"),
            "house": ("bl_house", "build_house"), "classics": ("bl_classic", "build_classic"), "settlement": ("bl_ensemble", "build_ensemble")}
ALIAS = dict({v["en"]: k for k, v in TYPES.items()}, **{k: k for k in TYPES})


def kind_of(name):
    if name not in ALIAS: raise KeyError(f"unknown type {name}: {', '.join(TYPES)}")
    return ALIAS[name]


def plan(type_, place, out=None, **intent):
    """Intent and place → a plan folder (<type file>, relief.npz, location.json); → the plan dict (out is its
    folder). The type goes first (not as a key: a house has its own key kind= — the house kind)."""
    T = TYPES[kind_of(type_)]
    if isinstance(intent.get("at"), str) and "," in intent["at"]: intent["at"] = [float(v) for v in intent["at"].split(",")]
    return importlib.import_module(arch_parts.module(T["part"], T["plan"])).plan(place, out=out, **intent)


def module(kind, which="plan"):
    """Module of a type for those who assemble many buildings in one scene (an ensemble): "plan" is the plan module
    (project Python), "builder" is the builder module (Blender). An ensemble is tied to a type only this way
    (bench/deps.py)."""
    T = TYPES[kind_of(kind)]
    return importlib.import_module(arch_parts.module(T["part"], T["plan"] if which == "plan" else T["builder"][:-3]))


def build_in(folder, sc=None, **kw):
    """Plan → mass over an open Blender scene (step by step, an ensemble): the type's builder as a function. →
    (objects, plan)."""
    k = kind_in(folder); mod, fn = BUILD_IN[k]
    M = importlib.import_module(arch_parts.module(TYPES[k]["part"], mod))
    if k == "castle": return M.build_castle(folder, sc=sc, with_ground=False, tag=kw.get("tag", True))
    return getattr(M, fn)(folder, sc=sc, **kw)


def kind_in(folder):
    for k, T in TYPES.items():
        if os.path.exists(os.path.join(folder, T["file"])): return k
    raise FileNotFoundError(f"no building plan in {folder} ({', '.join(T['file'] for T in TYPES.values())})")


def build(folder, blender="blender", **keys):
    """Plan → mass in Blender by the type's builder (a separate process); → the builder's __RESULT__ line as a
    dict."""
    T = TYPES[kind_in(folder)]
    cmd = [blender, "-b", "--python-exit-code", "1", "-P", core.paths.RUN, "--", os.path.join(HERE, T["part"], T["builder"]), "--", os.path.abspath(folder)]
    r = subprocess.run(cmd + [f"{k}={v}" for k, v in keys.items()], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if r.returncode or not line: raise RuntimeError(f"build {folder}: " + (r.stderr or r.stdout)[-800:])
    return json.loads(line[11:])


def influence(folder, pts, deck=None, z=None):
    """The three influences of a settlement (the "influences" part of the core, the raster influences.npz of the
    plan folder) at points: [{P, T, W, district}] — power, trade, wealth 0–1 and a reading. The scene and
    neighbors: a temple or castle in a town — wealth= from W of the place, people and shops — from T, dirt —
    from 1 − W."""
    from components.buildings.influences import field
    F = field.load(folder); X = [[float(v) for v in p.split(",")] for p in pts.split(";")] if isinstance(pts, str) else pts
    P, T, W = field.sample(F, X, deck, z); L = field.label(P, T, W, F["L"])
    return [{"P": round(float(a), 3), "T": round(float(b), 3), "W": round(float(c), 3), "district": l} for a, b, c, l in zip(P, T, W, L)]


def passport(folder):
    """Common passport of a building of any type — what the scene and an ensemble need without knowing the type's
    internals: bounds, height, entrances, anchors, cells, checks (from the plan and its space scheme)."""
    k = kind_in(folder); T = TYPES[k]; R = json.load(open(os.path.join(folder, T["file"])))
    sp = R.get("spaces") or {}; cells = sp.get("cells", [])[1:]; chk = R.get("checks", [])
    pts = [p for c in cells if c.get("poly") for p in c["poly"]]
    zs = [c["z0"] for c in cells] + [c["z1"] for c in cells if c.get("z1") is not None]
    box = None if not pts else {"x": [min(p[0] for p in pts), max(p[0] for p in pts)], "y": [min(p[1] for p in pts), max(p[1] for p in pts)],
                                "z": [min(zs), max(zs)]}
    tr = sp.get("trans", [])
    ents = [{"kind": tr[i]["kind"], "between": [tr[i]["a"], tr[i]["b"]], "p": tr[i]["p"], "z": [tr[i]["za"], tr[i]["zb"]], "width": tr[i]["w"]}
            for i in (sp.get("graph") or {}).get("entrances", [])]
    st = R.get("stats", {}); intent = R.get("spec") or {x: R[x] for x in ("type", "style", "seed") if isinstance(R.get(x), (str, int, float))}
    outline = []; courts = []; busy = []
    try:                                                                       # the building outline and the courtyards come from the scheme's cells (ensemble: setback, courtyard development)
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        polys = [Polygon(c["poly"]).buffer(0) for c in cells if c.get("poly") and len(c["poly"]) >= 3]
        U = unary_union([p for p in polys if not p.is_empty]) if polys else None
        if U is not None and not U.is_empty:
            outline = [[[round(x, 2), round(y, 2)] for x, y in g.exterior.coords] for g in getattr(U, "geoms", [U]) if g.area > 1]
        courts = [{"poly": c["poly"], "z": c["z0"], "id": c["id"]} for c in cells if c.get("cls") == "court" and c.get("poly")]
        occ = unary_union([p for p, c in zip(polys, [c for c in cells if c.get("poly") and len(c["poly"]) >= 3]) if c.get("cls") != "court" and not p.is_empty])
        busy = [[[round(x, 2), round(y, 2)] for x, y in g.exterior.coords] for g in getattr(occ, "geoms", [occ]) if not occ.is_empty and g.area > 0.5]
    except ImportError:
        pass
    return {"type": k, "file": T["file"], "place": R.get("place"), "intent": intent, "frame": sp.get("frame"), "outline": outline, "courts": courts, "occupied": busy,
            "bounds": box, "highest_m": st.get("highest_m"), "entrances": ents, "anchors": sp.get("anchors", []),
            **({"influences": {"field": (R.get("influence") or {}).get("field"), "centers": (R.get("influence") or {}).get("centres"), "knobs": (R.get("influence") or {}).get("knobs")}}
               if R.get("influence") else {}),
            "space": sp.get("stats"), "checks": f"{sum(bool(c['ok']) for c in chk)}/{len(chk)}",
            "non_passing": [c["what"] for c in chk if not c["ok"]], "summary": st}


def main():
    a = sys.argv[1:]
    if not a or a[0] == "types":
        for k, T in TYPES.items(): print(f"{k} ({T['en']}): {T['part']}/{T['plan']}.py → {T['file']} → {T['part']}/{T['builder']}; intent: {T['intent']}")
        return
    kw = dict(x.split("=", 1) for x in a[1:] if "=" in x); pos = [x for x in a[1:] if "=" not in x]
    if a[0] == "plan":
        R = plan(pos[0], pos[1], **kw)
        print("__RESULT__", json.dumps({"out": R["out"], **R["stats"], "non_passing": [c["what"] for c in R["checks"] if not c["ok"]]}, ensure_ascii=False, default=float))
    elif a[0] == "build": print("__RESULT__", json.dumps(build(pos[0], **kw), ensure_ascii=False))
    elif a[0] == "make":                                                       # plan and mass in one command: the build keys are views, res, blend
        bk = {k: kw.pop(k) for k in ("views", "res", "blend") if k in kw}
        R = plan(pos[0], pos[1], **kw)
        print("__RESULT__", json.dumps({"out": R["out"], **R["stats"], "non_passing": [c["what"] for c in R["checks"] if not c["ok"]]}, ensure_ascii=False, default=float))
        print("__RESULT__", json.dumps(build(R["out"], **bk), ensure_ascii=False))
    elif a[0] == "passport": print(json.dumps(passport(pos[0]), ensure_ascii=False, indent=1, default=float))
    elif a[0] == "influence":
        print("__RESULT__", json.dumps(influence(pos[0], kw["at"], int(kw["deck"]) if "deck" in kw else None, float(kw["z"]) if "z" in kw else None), ensure_ascii=False))
    else: raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
