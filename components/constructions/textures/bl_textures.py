"""The "textures" step of the Constructions component: substance on the construction elements according to the building
description (constructions.yaml → "textures:"). The surface engine is core/surface/bl_surface.py (a Poly Haven scan per
block with a shift and rotation by the stone number, Mikkelsen tiling on solid surfaces, mortar by noise); here — what
goes on what: the masonry roles (wall, finish, rubble, mortar), stone v2 (bl_stone: weathering per stone — chance and
place in the wall, edges and chips, the finish's edging — by the bl_masks masks from the masonry geometry), mortar in the
joints and building parts as mass (roles by the style palette). The block tone (Col) is the masonry's decision, the
surface does not change it (the substance pattern is normalized to 1).

  import bl_textures as TX; TX.run(scene, plan["textures"], palette={"stone": rgb, …}, mass=[mass objects])
"""
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
from core.surface import bl_surface as SF  # noqa: E402  (surface engine)
from components.constructions.textures import bl_masks as MK  # noqa: E402
from components.constructions.textures import bl_stone as ST  # noqa: E402

MORTAR_RGB = (0.66, 0.63, 0.57)          # default mortar — masonry/packs.yaml "substances: stone"
SUBST_PATH = os.path.join(HERE, "substances.json")   # substance by masonry pack (substances.py from ../masonry/packs.yaml)


def _substances():
    return json.load(open(SUBST_PATH))["packs"] if os.path.exists(SUBST_PATH) else {}


def by_pack(scene, stones, wall_mat, make, over=None):
    """Substance by masonry pack: for a masonry object (property sk_masonry → "pack") the slot of the role "wall" gets the
    pack's scan(s) (substances.json; over — the description key "packs:" takes priority); several scans — the stones differ:
    its own scan per stone (slots by stone number, the frame is no more expensive — a pixel evaluates a single material).
    make(scan) → a material. → {object: [scans]}."""
    table, out = _substances(), {}
    for ob in stones:
        try:
            pk = json.loads(ob.get("sk_masonry", "{}")).get("pack")
        except (TypeError, ValueError):
            pk = None
        scans = (over or {}).get(pk) or (table.get(pk) or {}).get("scans")
        if not scans:
            continue
        scans = [scans] if isinstance(scans, str) else list(scans)
        iw = next((i for i, sl in enumerate(ob.material_slots) if sl.material and
                   (sl.material == wall_mat or sl.material.name.startswith("sk_wall"))), None)
        if iw is None:
            continue
        me = ob.data
        ob.material_slots[iw].material = make(scans[0])
        if len(scans) > 1:
            idx = [iw]
            for sc_ in scans[1:]:
                me.materials.append(make(sc_)); idx.append(len(me.materials) - 1)
            n = len(me.polygons)
            mi = np.empty(n, np.int32); me.polygons.foreach_get("material_index", mi)
            a = me.attributes.get("sk_rnd")
            if a is not None:
                rv = np.empty(n * 3, np.float32); a.data.foreach_get("vector", rv); rv = rv.reshape(-1, 3)
                r = np.mod(rv[:, 0] * 7.31 + rv[:, 2] * 3.17, 1.0)    # its own fraction per stone (x — piece rotation, z — shift)
                k = np.minimum((r * len(scans)).astype(np.int32), len(scans) - 1)
                sel = mi == iw
                mi[sel] = np.asarray(idx, np.int32)[k[sel]]
                me.polygons.foreach_set("material_index", mi); me.update()
        out[ob.name] = {"pack": pk, "scans": scans}
    return out


def run(scene, desc, palette=None, mass=()):
    """Masonry roles: wall / finish / rubble — a scan per block (id from library/materials/surfaces.yaml; stone v2) or
    ig_<kind> (procedural stone); mortar — "sand" (noise) or a scan with tiling. Description keys:
      strength — the visibility of the substance pattern (×0.7 for finish, ×1.2 for rubble), scale — for ig_;
      edges (1) — wear, rounding and chips of the stone edges; weathering (0.6) — the spread of stone weathering (pattern,
      relief, roughness, chips — each stone's own), place (1) — the bottom of the wall is weathered more (the height of the
      stone above the ground);
      edging (0.03 m) — a smooth dressed band along the edge of a finish face; edges=0 weathering=0 edging=0 joints=0 — same
      as v1;
      mass: {palette role: scan} — building parts as mass (vertex color without roles) by palette → Mikkelsen tiling;
      joints (1) — faces of the mass behind the stones are mortar (tone mortar_tone, by default the masonry substance;
      substance — mortar_scan: a scan with tiling, "sand" — noise), 0 — the mass by palette;
      by_pack (1) — substance by masonry pack (substances.json: brick — clay, tuff, concrete, granite; rubble and cells — its
      own scan per stone), "packs: {pack: scan | [scans]}" takes priority over the table; 0 — "wall" for all;
      covers (1) — covering substance (roofing: bl_cover, covers.yaml), covers_over: {substance: scan | {scan, rough, …}};
      0 — coverings in a flat tone.
    → counts by roles, mass, masks and coverings."""
    d = dict(desc or {}); s = float(d.get("strength", 1.0)); k = float(d.get("scale", 2.5)); mats = {}
    ed, var, kant = float(d.get("edges", 1.0)), float(d.get("weathering", 0.6)), float(d.get("edging", 0.03))
    place = float(d.get("place", 1.0))
    for role, ks, seed in (("wall", 1.0, 3), ("finish", 0.7, 3), ("rubble", 1.2, 5)):
        v = d.get(role)
        if not v: continue
        v = str(v)
        if v.startswith("ig_"):
            mats[role] = SF.ig_stone(v[3:], seed=seed, scale=k * {"finish": 1.6, "rubble": 0.8}.get(role, 1.0), strength=s * ks,
                                     name=f"sk_surf_{role}")
        else:
            mats[role] = ST.stone(v, strength=s * ks, edges=ed, weathering=var, edging=kant if role == "finish" else 0.0,
                                  place=place, name=f"sk_surf_{role}")
    r = d.get("mortar")
    if r: mats["mortar"] = SF.mortar(strength=s) if str(r) in ("sand", "mortar") else SF.scan_plane(str(r), strength=s, name="sk_surf_mortar")
    info = {"roles": SF.apply_masonry(scene, mats, uv=True) if mats else {}}
    stone_mats = {m.name for m in mats.values()}
    stones = [ob for ob in scene.objects if ob.type == "MESH" and any(sl.material and sl.material.name in stone_mats for sl in ob.material_slots)]
    if float(d.get("by_pack", 1)) > 0 and stones:                     # substance by masonry pack
        cache = {}

        def make(sid_):
            if sid_ not in cache:
                cache[sid_] = ST.stone(sid_, strength=s, edges=ed, weathering=var, edging=0.0, place=place, name=f"sk_surf_wall_{sid_}")
            return cache[sid_]
        info["by_pack"] = by_pack(scene, stones, mats.get("wall"), make, d.get("packs"))
        stone_mats |= {m.name for m in cache.values()}
    if ed > 0 or kant > 0 or var > 0:                                   # stone masks: the face frame, chamfers, height above the ground
        ground = bpy.data.objects.get("ground") if place > 0 and var > 0 else None
        info["masks"] = {ob.name: {"n_faces": MK.frame(ob), "n_chamfers": MK.bevels(ob),
                                   "n_stones_over_ground": MK.ground_height(ob, ground) if ground else None} for ob in stones}
    info["textures"] = {role: (m["sk_surface"].to_dict() if hasattr(m["sk_surface"], "to_dict") else dict(m["sk_surface"])) for role, m in mats.items()}
    if d.get("mass") and palette and mass:
        from core.surface import bl_matlib as ML
        pm = {role: SF.scan_plane(str(sid), strength=s, name=f"sk_plane_{role}") for role, sid in d["mass"].items()}
        pal = {role: c for role, c in palette.items() if role in pm}
        joints = MK.mortar(mass, [ob for ob in stones if ob.name.startswith("masonry_")]) if float(d.get("joints", 1)) > 0 and stones else {}
        if any(v_.any() for v_ in joints.values()):
            ms = d.get("mortar_scan", "grey_plaster_03")
            pm["mortar"] = ST.mortar(d.get("mortar_tone") or MORTAR_RGB, strength=s, scan=None if ms in ("sand", "noise") else ms,
                                      name="sk_plane_mortar")
        keep = bpy.data.materials.get("col"); out = {}
        for ob in mass:
            ML.plane_uv(ob)
            roles = ML.roles(ob, pal)
            jm = joints.get(ob.name)
            if jm is not None:
                roles = ["mortar" if j else r_ for r_, j in zip(roles, jm)]
            out[ob.name] = ML.assign(ob, roles, pm, fallback=keep)
        info["mass"] = out
    if float(d.get("covers", 1)) > 0:                                   # covering substance (roofing) — bl_cover, knowledge from covers.yaml
        from components.constructions.textures import bl_cover as CV
        cv = CV.run(scene, d.get("covers_over"), log=lambda *a: None)
        if cv["n_objects"] or cv["far"] or cv["without_substance"]: info["coverings"] = cv
    return info
