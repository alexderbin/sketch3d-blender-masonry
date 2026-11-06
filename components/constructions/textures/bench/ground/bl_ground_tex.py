"""Substance stand for paving and timber (Blender 5.2): the "paving" (paving/bl_paving.run) and "timber"
(timber/bl_timber.run) steps on the occupancy layer of its own scene folder → "before" frames (flat layout tone) →
covering substance (bl_cover.run) → "after" frames. The scene: a courtyard 26 × 18 m, a road 4 m wide to the west gate,
a house 8 × 6 m (two floors, half-timbering from the second — a layer mark), a fence along the south edge of the
courtyard; a 1.8 m figure. The stand writes the occupancy layer itself (<out>/scene/occupancy.json — its own scene, not
someone else's). Paving views: court — the courtyard at eye level, macro — stones from 2 m, road — the road; timber:
facade — the facade from 6 m, macro — beam and infill from 2.4 m; fences: fence, macro. The plans are paving and timber
resolve kinds=all (run.sh).

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/constructions/textures/bench/ground/bl_ground_tex.py
  -- <out> [paving:<kind>,timber:<kind>,…] [res=1280x720] → <out>/<part>_<kind>_<view>_{0,1}.png, <out>/run_ground_tex.json"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

from components.constructions.paving import bl_paving as BP  # noqa: E402
from components.constructions.textures import bl_cover as CV  # noqa: E402
from components.constructions.timber import bl_timber as BT  # noqa: E402

ARGV = sys.argv[sys.argv.index("--") + 1:]
OUT = os.path.abspath(ARGV[0]); KW = dict(a.split("=", 1) for a in ARGV[1:] if "=" in a)
ONLY = [x for a in ARGV[1:] if "=" not in a for x in a.split(",") if x]
RES = tuple(int(v) for v in KW.get("res", "1280x720").split("x"))
SCENE = os.path.join(OUT, "scene")
YARD = [[0, 0], [26, 0], [26, 18], [0, 18]]; HOUSE = [[15, 6], [23, 6], [23, 12], [15, 12]]
FENCE = [[2.0, 2.5], [12.0, 2.5]]
VIEWS = {"paving": {"court": ((3.0, 1.5, 1.7), (16.0, 9.0, 0.4), 26), "macro": ((5.5, 3.5, 1.5), (7.0, 5.2, 0.0), 35),
                    "road": ((-30.0, 7.0, 1.7), (-10.0, 9.0, 0.3), 28)},
         "timber": {"facade": ((19.0, 0.3, 3.0), (19.0, 6.0, 3.6), 26), "macro": ((17.6, 3.7, 3.4), (18.3, 6.0, 3.6), 35)},
         "fence": {"fence": ((6.0, -3.5, 1.6), (7.5, 2.5, 0.8), 30), "macro": ((6.4, 1.0, 1.3), (6.8, 2.5, 0.9), 35)}}


def look(sc):
    sc.render.engine = "BLENDER_EEVEE"; sc.view_settings.view_transform = "AgX"; sc.view_settings.exposure = -1.3
    sc.eevee.taa_render_samples = 16
    try: sc.eevee.use_raytracing = True
    except Exception: pass
    sc.render.resolution_x, sc.render.resolution_y = RES
    w = bpy.data.worlds.new("w"); sc.world = w; nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND"); sky = nt.nodes.new("ShaderNodeTexSky")
    try: sky.sky_type = "MULTIPLE_SCATTERING"
    except Exception: pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.12
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 4.5; sun.data.angle = math.radians(1.0); sun.data.color = (1.0, 0.94, 0.86)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.clip_start = 0.05; cam.data.clip_end = 800
    return cam, sun


def key_sun(sun, loc, tgt, side=40.0, elev=38.0):
    """The sun on the camera's side (off to the side by side°, elevation elev°): the substance is lit and the shadows read."""
    b = np.asarray(loc, float)[:2] - np.asarray(tgt, float)[:2]; b /= max(np.linalg.norm(b), 1e-9); a = math.radians(side)
    h = np.array([b[0] * math.cos(a) - b[1] * math.sin(a), b[0] * math.sin(a) + b[1] * math.cos(a)]); e = math.radians(elev)
    sun.rotation_euler = Vector((h[0] * math.cos(e), h[1] * math.cos(e), math.sin(e))).to_track_quat("Z", "Y").to_euler()


def mesh(name, V, F, cols, roles, layer):
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update()
    me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.asarray(roles, np.int32))
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    cc = np.repeat(np.asarray(cols, np.float32), [len(f) for f in F], axis=0)
    ca.data.foreach_set("color", np.column_stack([cc, np.ones(len(cc), np.float32)]).ravel())
    m = bpy.data.materials.new(name + "_m"); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"; nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.9; me.materials.append(m)
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob); ob["sk_layer"] = layer
    return ob


def static(sc):
    """Ground (the place layer), the house mass (the buildings layer: walls, roof), a 1.8 m figure."""
    s = 200.0
    g = mesh("ground", [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [(0, 1, 2, 3)], [(0.25, 0.23, 0.19)], [0], "site")
    x0, y0, x1, y1, ze, zr = 15.05, 6.05, 22.95, 11.95, 6.6, 9.2
    V = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (x0, y0, ze), (x1, y0, ze), (x1, y1, ze), (x0, y1, ze), (x0, 9.0, zr), (x1, 9.0, zr)]
    F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 9, 8), (6, 7, 8, 9), (5, 6, 9), (7, 4, 8)]
    wall, roof = (0.78, 0.74, 0.64), (0.40, 0.20, 0.14)
    house = mesh("house", V, F, [wall] * 4 + [roof] * 2 + [wall] * 2, [0] * 4 + [2] * 2 + [0] * 2, "buildings")
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=1.5, location=(9.0, 6.0, 0.75)); b = bpy.context.object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(9.0, 6.0, 1.66)); h = bpy.context.object
    mf = bpy.data.materials.new("f"); bf = next(n for n in mf.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); bf.inputs["Base Color"].default_value = (0.12, 0.12, 0.13, 1)
    for o in (b, h): o.data.materials.append(mf)
    return g, house


def layer():
    """The stand's occupancy layer: a courtyard, a road to the gate, a house with half-timbering from the second floor."""
    os.makedirs(SCENE, exist_ok=True)
    items = [{"kind": "yard", "poly": YARD, "level": 0.0, "id": "court"},
             {"kind": "road", "line": [[-40.0, 9.0, 0.0], [-20.0, 9.0, 0.0], [0.0, 9.0, 0.0]], "w": 4.0},
             {"kind": "building", "poly": HOUSE, "type": "house", "base": 0.0, "floor": 0.2, "eave": 6.6, "storeys": 2, "fachwerk": 1}]
    json.dump({"items": items, "palette": {}}, open(os.path.join(SCENE, "occupancy.json"), "w"), ensure_ascii=False)


def shoot(sc, cam, sun, views, tag, stem, info):
    for v, (loc, tgt, lens) in views.items():
        cam.location = loc; cam.data.lens = lens; key_sun(sun, loc, tgt)
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(OUT, f"{stem}_{v}_{tag}.png"); t1 = time.time()
        bpy.ops.render.render(write_still=True); info["render_s"][f"{v}_{tag}"] = round(time.time() - t1, 2)


def main():
    sc = bpy.context.scene
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    cam, sun = look(sc); _g, house = static(sc); layer()
    pp = json.load(open(os.path.join(OUT, "plan_paving.json"))); tp = json.load(open(os.path.join(OUT, "plan_timber.json")))
    jobs = [("paving", k) for k in pp["kinds"]] + [("timber", k) for k in tp["kinds"]]
    if ONLY: jobs = [j for j in jobs if f"{j[0]}:{j[1]}" in ONLY or j[1] in ONLY]
    fpx = RES[0] * 28 / 36.0; res = {}
    for part, name in jobs:
        coll = bpy.data.collections.new(f"constructions_{part}_{name}"); sc.collection.children.link(coll); t0 = time.time()
        if part == "paving":
            views = VIEWS["paving"]; cams = [v[0] for v in views.values()]
            p = {"paving": dict(pp["paving"], parts=dict(pp["paving"]["parts"], court=name, road=name), detail="cam"), "kinds": pp["kinds"], "substances": pp["substances"]}
            st = BP.run(sc, p, coll, objs=[house], scene=SCENE, cams=cams, fpx=fpx, log=lambda *a: None)
            k = pp["kinds"][name]
        else:
            k = tp["kinds"][name]; fence = k["generator"] in ("palisade", "wattle")
            views = VIEWS["fence" if fence else "timber"]; cams = [v[0] for v in views.values()]
            parts = {"fences": name} if fence else ({"fachwerk": name} if k["generator"] == "fachwerk" else {"buildings": name})
            c = dict(tp["timber"], parts=parts, detail="cam", fences=[{"path": FENCE, "kind": name}] if fence else [])
            st = BT.run(sc, {"timber": c, "kinds": tp["kinds"], "substances": tp["substances"]}, coll, objs=[house], scene=SCENE, cams=cams, fpx=fpx, log_=lambda *a: None)
        info = {"part": part, "title": k["title"], "substance": k["substance"], "build_s": round(time.time() - t0, 2), "render_s": {},
                "step": {a: b for a, b in st.items() if isinstance(b, (int, float))}}
        shoot(sc, cam, sun, views, 0, f"{part}_{name}", info)
        t1 = time.time(); cv = CV.run(sc, log=lambda *a: None); info["cover_s"] = round(time.time() - t1, 2)
        info["coverings"] = {a: cv[a] for a in ("n_objects", "far", "n_faces_piece", "n_faces_tile", "substances", "without_substance")}
        shoot(sc, cam, sun, views, 1, f"{part}_{name}", info)
        for ob in list(coll.objects): bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(coll)
        res[f"{part}:{name}"] = info; print("ROW", part, name, json.dumps(info, ensure_ascii=False), flush=True)
    path = os.path.join(OUT, "run_ground_tex.json")
    if ONLY and os.path.exists(path): res = dict(json.load(open(path)), **res)
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps({k: [v["coverings"]["n_objects"], v["coverings"]["without_substance"]] for k, v in res.items()}, ensure_ascii=False))


main()
