"""Entry of the Constructions component for step-by-step assembly: masonry, textures, roofing, paving, timber over an open
scene where the building already stands. They read only the scene layers (core.layers, written by Buildings): the
masonry surfaces (surfaces.json), occupancy (courtyards, the road, buildings and houses with a base, an eaves line,
floors), the palette; they do not read the plans of Buildings. The description is constructions.yaml of the run scene
(the scene model section constructions); if absent, a default masonry description is written from its own keys (packs,
textures).
plan: masonry.resolve → the masonry plan (for the journal and the cache); assembly: masonry/bl_masonry.run_masonry
(masonry — bl_stones.run into the step's collection; textures — bl_textures.run); roofing — roofing/bl_roofing.run by
the face contract (the role "roof" or the palette roof color) → tiles."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT = """# Building description — the build step default; edit by hand if you need something else.
masonry:
  packs: {outer: %(outer)s, inner: %(inner)s, towers: %(towers)s}
textures:
  wall: %(wall)s
  finish: rock_01
  rubble: rock_boulder_dry
  mortar: sand
  mass: {stone: %(wall)s, plaster: grey_plaster_03}
"""
# protocol (English knobs of the passport) → names read by the component's internal resolver (.resolve)
PLAN_KEYS = ('detail', 'Tlo', 'Thi', 'Tfar', 'rows', 'plate', 'outer', 'inner', 'towers', 'seed', 'walls')   # step knobs = resolve names (the same ones, in English)


def _ensure_desc(folder, keys):
    p = os.path.join(folder, "constructions.yaml")
    stale = False
    if os.path.exists(p):                                           # the default description in the old language (Russian section names)
        txt = open(p, encoding="utf-8").read()                      # in the folder of a previous run — rebuild it, otherwise the textures are silently skipped
        stale = txt.startswith(DEFAULT.splitlines()[0][:60]) and "\nmasonry:" not in txt
    if not os.path.exists(p) or stale:
        d = {"outer": keys.get("outer", "ashlar"), "inner": keys.get("inner", "rubble"), "towers": keys.get("towers", "ashlar"),
             "wall": keys.get("wall", "rock_surface")}
        open(p, "w", encoding="utf-8").write(DEFAULT % d)
    return p


def _layer(ctx):
    """The masonry surfaces layer of the scene (core.layers: surfaces.json — written by Buildings); if absent — the step fails
    with a reason."""
    from core import layers as L
    lay = L.surfaces(ctx["scene"])
    if not lay.get("items"):
        raise SystemExit("masonry: the scene has no surfaces layer (surfaces.json) — it is written by the building step with a castle (castle, or\na settlement with an owner)")
    return lay


def plan_masonry(M, keys, ctx):
    from components.constructions.masonry import masonry
    SD = ctx["scene"]; n = len(_layer(ctx)["items"]); _ensure_desc(SD, keys)
    kw = {k: str(v) for k, v in keys.items() if k in PLAN_KEYS}
    plan = masonry.resolve(SD, kw)
    k = plan["masonry"]
    return {"kw": kw, "surfaces": n, "detail": k.get("detail"), "parts": k.get("parts"), "thresholds": k.get("thresholds"),
            "textures": bool(plan.get("textures"))}


def _cams(sc):
    cams = [o for o in sc.objects if o.type == "CAMERA" and not o.name.startswith("far_")]
    pts, seen = [], set()                                                # level of detail by distance: shot cameras per frame (hundreds of points
    for o in cams:                                                       # within a meter of each other) are thinned on a 6 m grid
        p = tuple(o.matrix_world.translation); k = tuple(int(round(v / 6.0)) for v in p)
        if k in seen: continue
        seen.add(k); pts.append(p)
    fpx = 1000.0
    if cams:
        c = cams[0]; fpx = sc.render.resolution_x * c.data.lens / (c.data.sensor_width or 36.0)
    return pts, fpx


def _run(ctx, steps):
    from components.constructions.masonry import bl_masonry as BM
    sc = ctx["sc"]; SD = ctx["scene"]
    kw = {k: str(v) for k, v in ctx["keys"].items() if k in PLAN_KEYS}
    pts, fpx = _cams(sc)
    coll = ctx["collection"] if "masonry" in steps else None
    from core import layers as L                                         # textures need only the layer with the palette, masonry needs the layer's points
    lay = _layer(ctx) if "masonry" in steps else L.surfaces(SD)
    info = BM.run_masonry(sc, lay, SD, kw=kw, steps=steps, cams=pts, fpx=fpx, coll=coll, work=ctx["out"])
    return {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))} | {"cameras": len(pts)}   # journal statistics


def build_masonry(ctx):
    return _run(ctx, ("masonry",))


def build_textures(ctx):
    return _run(ctx, ("textures",))


# ── roofing ─────────────────────────────────────────────────────────────────────────────────────────────────────
ROOF_KEYS = ('covering', 'roofing', 'towers', 'buildings', 'gates', 'tents', 'domes', 'detail', 'Tlo', 'Thi', 'Tfar', 'seed')   # step knobs = resolve names (the same ones, in English)


def plan_roofing(M, keys, ctx):
    """Roofing plan: constructions.yaml of the run scene (the "roofing:" section; if absent — from the keys) + coverings.yaml."""
    from components.constructions.roofing import roofing
    folder = ctx["scene"]                                                       # the description is constructions.yaml of the run scene
    kw = {k: str(v) for k, v in keys.items() if k in ROOF_KEYS and v is not None}
    plan = roofing.resolve(folder, kw)
    c = plan["roofing"]
    return {"description": folder, "kw": kw, "parts": c["parts"], "detail": c["detail"], "thresholds": c["thresholds"],
            "roofing": c, "kinds": plan["kinds"], "substances": plan["substances"], "version": plan["version"]}


def build_roofing(ctx):
    """Roofs of the open scene's buildings → tiles (bl_roofing.run) into the step's collection. Roofs are taken by the face
    contract: the role "roof" (sk_role = 2) or the roof color of the building palette (occupancy layer: palette.roof — the
    castle writes vertex color without roles)."""
    from components.constructions.roofing import bl_roofing as BT
    from core import layers as L
    sc = ctx["sc"]; plan = ctx["plan"]; coll = ctx["collection"]
    roof_rgb = L.palette(ctx["scene"]).get("roof")                              # roof color of the building palette (occupancy layer)
    objs = [o for o in sc.objects if o.type == "MESH" and o.get("sk_layer") == "buildings"]
    if not objs: objs = [o for o in sc.objects if o.type == "MESH" and o.name not in coll.objects and not o.get("sk_layer")]
    pts, fpx = _cams(sc)
    st = BT.run(sc, {"roofing": plan["roofing"], "kinds": plan["kinds"], "substances": plan["substances"]}, coll, objs=objs,
                roof_rgb=roof_rgb, cams=pts, fpx=fpx)
    return {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool))} | {"cameras": len(pts)}   # journal statistics


# ── paving ──────────────────────────────────────────────────────────────────────────────────────────────────────
PAVE_KEYS = ('court', 'road', 'floor', 'areas', 'width', 'detail', 'Tlo', 'Thi', 'Tfar', 'seed', 'skip')   # step knobs = resolve names (the same ones, in English)


def plan_paving(M, keys, ctx):
    """Paving plan: constructions.yaml of the run scene (the "paving:" section; if absent — from the keys) + paving.yaml."""
    from components.constructions.paving import paving
    folder = ctx["scene"]                                                       # the description is constructions.yaml of the run scene
    kw = {k: str(v) for k, v in keys.items() if k in PAVE_KEYS and v is not None}
    plan = paving.resolve(folder, kw)
    return {"description": folder, "kw": kw, "paving": plan["paving"], "kinds": plan["kinds"], "substances": plan["substances"],
            "version": plan["version"]}


def build_paving(ctx):
    """Paving of the open scene (bl_paving.run) into the step's collection: building floors (faces with the role "floor"), the
    courtyard and the road — the scene's occupancy layer (yard, road), areas of the description; height follows the site's
    ground."""
    from components.constructions.paving import bl_paving as BP
    sc = ctx["sc"]; plan = ctx["plan"]; coll = ctx["collection"]
    objs = [o for o in sc.objects if o.type == "MESH" and o.get("sk_layer") == "buildings"]
    pts, fpx = _cams(sc)
    st = BP.run(sc, {"paving": plan["paving"], "kinds": plan["kinds"], "substances": plan["substances"]}, coll, objs=objs,
                scene=ctx["scene"], cams=pts, fpx=fpx)
    return {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool))} | {"cameras": len(pts)}   # journal statistics


# ── timber ──────────────────────────────────────────────────────────────────────────────────────────────────────
TIMBER_KEYS = ('fachwerk', 'buildings', 'fences', 'detail', 'Tfar', 'seed')   # step knobs = resolve names (the same ones, in English)


def plan_timber(M, keys, ctx):
    """Timber plan: constructions.yaml of the run scene (the "timber:" section; if absent — from the keys) + timber.yaml."""
    from components.constructions.timber import timber
    folder = ctx["scene"]                                                       # the description is constructions.yaml of the run scene
    kw = {k: str(v) for k, v in keys.items() if k in TIMBER_KEYS and v is not None}
    plan = timber.resolve(folder, kw)
    return {"description": folder, "kw": kw, "timber": plan["timber"], "kinds": plan["kinds"], "substances": plan["substances"], "version": plan["version"]}


def build_timber(ctx):
    """Timber of the open scene (bl_timber.run) into the step's collection: walls come from the scene's occupancy layer
    (buildings: half-timbering by a mark, "buildings" — all; settlement houses — by timber.yaml houses, an izba is a log
    house), openings follow the dark faces; fences of the description."""
    from components.constructions.timber import bl_timber as BT
    sc = ctx["sc"]; plan = ctx["plan"]; coll = ctx["collection"]
    objs = [o for o in sc.objects if o.type == "MESH" and o.get("sk_layer") == "buildings"]
    pts, fpx = _cams(sc)
    st = BT.run(sc, {"timber": plan["timber"], "kinds": plan["kinds"], "substances": plan["substances"]}, coll, objs=objs,
                scene=ctx["scene"], cams=pts, fpx=fpx)
    return {k: v for k, v in st.items() if isinstance(v, (int, float, str, bool))} | {"cameras": len(pts)}   # journal statistics
