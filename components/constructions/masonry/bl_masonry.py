"""Masonry by steps — the Constructions component (components/constructions/): stones over the scene's surfaces layer (core.layers: surfaces.json — walls by
sides and towers as rings, written by Buildings) — bl_stones.run; representation — all as blocks or by camera (blocks, rows, a slab); textures — ../textures/bl_textures.run
(the engine is core/surface/bl_surface.py), if the description has a "textures:" section; the building mass follows the layer's palette. Masonry does not read the castle plan.
The description is <description folder>/constructions.yaml (the sections "masonry:", "textures:"), the plan is masonry.py resolve (→ masonry_plan.json);
step-line keys override the description.
    run_masonry(sc, layer, desc_dir, kw=None, steps=("masonry", "textures"), cams=(), fpx=1000.0, coll=None, work=None, log=print) → stats"""
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
import bpy                                                                     # noqa: E402

from components.constructions.masonry import bl_stones as BS                                                         # noqa: E402  (the masonry step)
from components.constructions.masonry.bl_stones import ROLES, Acc, inset, stones_batch                          # noqa: E402,F401  (compatibility)

PLAN_KEYS = ("detail", "Tlo", "Thi", "Tfar", "rows", "plate", "outer", "inner", "towers", "seed")


def plan_of(cdir, kw, out=None):
    """Masonry plan: masonry.py resolve (the description <cdir>/constructions.yaml + step-line keys) → masonry_plan.json (in out or cdir)."""
    out = os.path.join(out or cdir, "masonry_plan.json")
    args = [f"{k}={v}" for k, v in kw.items() if k in PLAN_KEYS] + [f"out={out}"]
    r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-m", "components.constructions.masonry.masonry", "resolve", cdir, *args],
                       capture_output=True, text=True)
    if r.returncode: raise SystemExit("masonry.py resolve: " + (r.stderr or r.stdout)[-800:])
    return json.load(open(out))


def items_of(layer, plan, kw):
    """Masonry items from the layer: walls by the walls key ("0", "0,1" or all), towers — if the plan has a tower pack (parts["towers"]; towers=0|no — resolve removes it);
    the pack follows the plan's part, the seed is the plan seed + the side (walls) or 99 + the seed + the tower number."""
    k = plan["masonry"]; z = int(k["seed"]); parts = k["parts"]
    items = layer.get("items") or []
    walls = sorted({it["wall"] for it in items if "wall" in it}) if str(kw.get("walls", "0")) == "all" \
        else [int(x) for x in str(kw.get("walls", "0")).split(",")]
    out = []
    for wi in walls:
        out += [dict(it, pack=parts[it["part"]], seed=z + int(it["side"]) - 1) for it in items if it.get("wall") == wi]
    if parts.get("towers"):                                              # the towers knob has a single meaning: the tower pack (0|no — without masonry)
        out += [dict(it, pack=parts["towers"], seed=99 + z + int(it["tower"])) for it in items if "tower" in it]
    return out


def textures_step(sc, plan, palette):
    """The component's "textures" step (../textures/bl_textures.run) by the description section "textures:"; the building mass (objects
    walls, towers, buildings: vertex colour without roles) — the roles come from the surfaces layer's palette."""
    from components.constructions.textures import bl_textures as TX
    pal = {"stone": (palette or {}).get("stone"), "plaster": (palette or {}).get("plaster")}
    mass = [o for o in (bpy.data.objects.get(n) for n in ("walls", "towers", "buildings")) if o is not None]
    return TX.run(sc, plan["textures"], palette={k: v for k, v in pal.items() if v is not None}, mass=mass)


def run_masonry(sc, layer, desc_dir, kw=None, steps=("masonry", "textures"), cams=(), fpx=1000.0, coll=None, work=None, log=print):
    """Masonry and textures over an open scene from the surfaces layer layer (core.layers.surfaces). kw — step-line keys (walls,
    towers, detail, rows, plate, …). steps — "masonry", "textures". cams — camera points for the level of detail, fpx — pixels per frame focal length (as in bl_stones.run),
    coll — the collection for stone (by default a new "constructions"), work — the folder of maps.
    → stats: the result of bl_stones.run (+ towers) and "textures", "textures_s"."""
    kw = dict(kw or {}); work = work or desc_dir
    plan = plan_of(desc_dir, kw, work); info = {}
    if "masonry" in steps:
        if coll is None:
            coll = bpy.data.collections.new("constructions"); sc.collection.children.link(coll)
        items = items_of(layer, plan, kw)
        stats, _ = BS.run(items, plan, coll, cams=list(cams), fpx=fpx, work=os.path.join(work, "masonry_maps"), log=log)
        stats["towers"] = sum(1 for it in items if it["tag"].startswith("t")); info.update(stats)
    if "textures" in steps and plan.get("textures"):
        ts = time.time(); info["textures"] = textures_step(sc, plan, layer.get("palette")); info["textures_s"] = round(time.time() - ts, 1)
    return info
