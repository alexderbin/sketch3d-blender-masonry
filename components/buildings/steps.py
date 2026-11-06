"""The entry of the "buildings" component for the step-by-step build (core/blender/build.py): the "building" step — a castle,
a temple, an edifice, a settlement or a classic.
The plan (the project's Python): the type and the type's intent (castle: site, purpose, era, culture, scale, mood; temple:
style, history, wealth, ruin, irregular; edifice: kind, style; for all — seed, at) and the site (the "site" section of
the run's scene: relief.npz, location.json) → the plan folder <step>/plan/ (castle.json | temple.json | edifice.json,
the edited relief.npz: terraces, moat, yard, platform; location.json).
The building refines the site: the edited relief.npz and location.json are put into the run's scene.
The build (Blender): the ground is rebuilt from the edited relief (the object terrain, the collection "site"); a castle —
bl_castle.build_castle in the collection "castle" (the masonry surfaces layer — the scene's surfaces.json, core.layers);
a temple — bl_temple.build_temple in "temple", an edifice — bl_edifice.build_edifice in "edifice" (faces are labeled by
the space scheme, the model is checked by rays); a settlement — ensembles/bl_ensemble.build_ensemble via the entry
(pavements, stairs, a wall, tiers, houses of the "house" type, the host castle by its own builder)."""
import json, os, shutil

INTENT = {"castle": ("site", "purpose", "era", "culture", "scale", "mood", "seed", "at", "approach"),                     # passport knobs (in English) by building type
          "temple": ("style", "history", "wealth", "ruin", "irregular", "seed", "at"),
          "edifice": ("kind", "style", "seed", "at"),
          "settlement": ("archetype", "seed", "at", "size", "host", "power", "trade", "wealth", "inequality", "wind", "centre"),
          "classics": ("kind", "seed", "at")}


def plan_building(M, keys, ctx):
    from components.buildings import buildings
    SD = ctx["scene"]; out = os.path.join(ctx["out"], "plan")
    kind = buildings.kind_of(keys.get("type") or "castle")              # castle, temple, edifice, settlement, classic → the type (the internal name)
    intent = {k: v for k, v in keys.items() if k in INTENT[kind] and v is not None}
    if "kind" in intent: intent["type"] = intent.pop("kind")                    # the edifice kind (ziggurat, …) — the plan's type key
    if "archetype" in intent: intent["type"] = intent.pop("archetype")              # settlement: polis, colony, subura, town, posad, …
    if kind == "classics" and intent.get("type") == "all": intent["type"] = "all"   # classic: a showcase of all kinds in rows
    intent.setdefault("seed", M.get("seed") or 1)
    buildings.plan(kind, SD, out=out, **intent)
    for f in ("relief.npz", "location.json"):                      # the refined site goes into the run's scene (the "site" section)
        if os.path.exists(os.path.join(out, f)): shutil.copy(os.path.join(out, f), os.path.join(SD, f))
    from components.site import api as SITE                          # the ground anew from the edited relief — the site component's mechanism (its api):
    SITE.refresh(SD)                                                  # texture, moisture and masks, the surface recipe (terraces and the moat show)
    for f in ("relief_tex.png", "relief_tex.json"):                # the site's ground texture — next to the edited relief (the buildings plans)
        if os.path.exists(os.path.join(SD, f)): shutil.copy(os.path.join(SD, f), os.path.join(out, f))
    from components.buildings import layers as BL                   # layers for the neighbors: occupancy and anchors (core.layers)
    n_occ, n_anc = BL.write(out, SD)
    pf = os.path.join(out, buildings.TYPES[kind]["file"])
    C = json.load(open(pf)) if os.path.exists(pf) else {}
    st = C.get("stats") or {}
    return {"building_plan": out, "type": kind, "intent": intent, "center": (C.get("site") or {}).get("centre"),
            "n_towers": st.get("n_towers"), "n_buildings": st.get("n_buildings") or st.get("n_houses"), "walls_m": st.get("walls_m"), "highest_m": st.get("highest_m"),
            "checks": [c for c in (C.get("checks") or []) if not c.get("ok")][:6], "occupancy": n_occ, "anchors": n_anc}


def build_building(ctx):
    import bpy
    from components.buildings import buildings
    from components.buildings.castles import bl_castle as BC
    from components.site import api as SITE
    from core.surface.ground import Ground
    sc = ctx["sc"]; folder = ctx["plan"]["building_plan"]
    old = bpy.data.objects.get("terrain")                           # the ground anew from the edited relief (terraces, moat)
    if old is not None:
        me = old.data; bpy.data.objects.remove(old)
        if me and me.users == 0: bpy.data.meshes.remove(me)
    ter, _ = SITE.build_terrain(Ground(ctx["scene"]), sc=sc)              # the run scene's ground: the edited relief, texture, surface
    place = bpy.data.collections.get("site")
    if place is not None and ter.name not in place.objects:
        if ter.name in sc.collection.objects: sc.collection.objects.unlink(ter)
        place.objects.link(ter)
    ter["sk_layer"] = "site"
    kind = ctx["plan"].get("type", "castle")
    if kind in ("settlement", "classics"):                                       # a settlement (streets, houses, wall, host castle), a classic — via the entry
        obs, R = buildings.build_in(folder, sc=sc)
        st = R.get("stats") or {}
        hf = (R.get("host") or {}).get("folder")                                # the host castle is built by the same bl_castle (its plan and
        if hf and os.path.exists(os.path.join(hf, "castle.json")) and getattr(BC, "C", None) is not None:   # state are held in the process):
            from components.buildings.castles import surfaces as SF               # the masonry surfaces layer for the neighbors
            SF.write(ctx["scene"], BC.C, BC)
        return {"building_plan": folder, "type": buildings.TYPES[kind]["en"], "objects": len(obs), "centre": (R.get("site") or {}).get("centre"),
                "houses": st.get("n_houses"), "ground_vertices": len(ter.data.vertices)}
    if kind != "castle":                                                         # temple, edifice: the type's builder as a function over an open scene
        mod = {"temple": ("temples", "bl_temple", "build_temple"), "edifice": ("edifices", "bl_edifice", "build_edifice")}[kind]
        import importlib
        BT = importlib.import_module(f"components.buildings.{mod[0]}.{mod[1]}")
        obs, T = getattr(BT, mod[2])(folder, sc=sc, tag=True)
        chk = (BT.SP_LAST or {}).get("check") or {}
        return {"building_plan": folder, "type": buildings.TYPES[kind]["en"], "objects": [o.name for o in obs], "centre": (T.get("site") or {}).get("centre"),
                "ground_vertices": len(ter.data.vertices), "model_vs_scheme": "ok" if chk.get("ok", True) else "fail"}
    objs, C = BC.build_castle(folder, sc=sc, with_ground=False, tag=True)
    from components.buildings.castles import surfaces as SF              # the masonry surfaces layer (walls, towers, loopholes) for the neighbors,
    SF.write(ctx["scene"], BC.C, BC)                                      # while the castle's state (stairs, details) is in this process
    names = {k: ([o.name for o in v] if isinstance(v, (list, tuple)) else getattr(v, "name", str(v))) for k, v in objs.items()}
    return {"building_plan": folder, "type": buildings.TYPES[kind]["en"], "objects": names, "centre": (C.get("site") or {}).get("centre"),
            "ground_vertices": len(ter.data.vertices)}
