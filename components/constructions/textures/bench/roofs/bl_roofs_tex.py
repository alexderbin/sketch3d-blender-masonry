"""Substance stand for roofing (Blender 5.2): the "roofing" step (roofing/bl_roofing.run, camera-dependent mode) →
"before" frames (flat layout tone) → the covering texture step (bl_cover.run) → "after" frames. The scene is the same as
for the roof kinds sheet: a house 7 × 4.5 m (gable, a hip on the east), a square tower with a pyramid, a round one with
a cone, a 1.8 m figure; the same group 90 m to the north (far roof: a shell with a row pattern). Views: macro — tiles
from 2.5 m, close — hip and ridge, all — all three masses, deep — the near roof and the far group in one frame (do the
tone and gloss match). The plan is roofing resolve kinds=all → <out>/plan_roofs.json (run.sh).

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/constructions/textures/bench/roofs/bl_roofs_tex.py
  -- <out> [kinds, comma-separated] [res=960x540] → <out>/<kind>_<view>_{0,1}.png, <out>/run_roofs_tex.json"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

from components.constructions.roofing import bl_roofing as BT  # noqa: E402
from components.constructions.textures import bl_cover as CV  # noqa: E402

ARGV = sys.argv[sys.argv.index("--") + 1:]
OUT = os.path.abspath(ARGV[0]); KW = dict(a.split("=", 1) for a in ARGV[1:] if "=" in a)
ONLY = [x for a in ARGV[1:] if "=" not in a for x in a.split(",") if x]
RES = tuple(int(v) for v in KW.get("res", "960x540").split("x"))
WALL = (0.55, 0.52, 0.47); ROOFC = (0.35, 0.18, 0.12)
FAR_DY = 90.0                                                                     # the far group is 90 m to the north
VIEWS = {"macro": ((2.0, -1.3, 4.9), (3.3, 0.9, 4.0), 35), "close": ((6.2, -4.6, 5.4), (3.0, 1.6, 5.4), 32),
         "all": ((15.5, -13.0, 7.5), (3.2, 2.0, 4.6), 30), "deep": ((1.0, -9.0, 7.2), (3.0, FAR_DY, 4.0), 50)}


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
    d = np.array([0.35, -0.55, 0.75]); d /= np.linalg.norm(d)
    try: sky.sun_elevation = math.asin(d[2]); sky.sun_rotation = math.atan2(d[0], d[1])
    except Exception: pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.12
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 4.5; sun.data.angle = math.radians(1.0); sun.data.color = (1.0, 0.94, 0.86)
    sun.rotation_euler = Vector(d).to_track_quat("Z", "Y").to_euler()
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.clip_start = 0.05; cam.data.clip_end = 800
    return cam


def mesh(name, faces, roles, coll):
    """Mass mesh: faces (world) with a role (FACE INT sk_role) and a color Col — as for buildings."""
    V, F, cols = [], [], []
    for pts, role in zip(faces, roles):
        i0 = len(V); V += [tuple(map(float, p)) for p in pts]; F.append(list(range(i0, i0 + len(pts))))
        cols.append(ROOFC if role == 2 else WALL)
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update()
    me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.asarray(roles, np.int32))
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    cc = np.repeat(np.asarray(cols, np.float32), [len(f) for f in F], axis=0)
    ca.data.foreach_set("color", np.column_stack([cc, np.ones(len(cc), np.float32)]).ravel())
    m = bpy.data.materials.new(name + "_m"); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"; nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.9; me.materials.append(m)
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob); return ob


def box_walls(x0, y0, x1, y1, z0, z1):
    P = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [[(*P[i], z0), (*P[(i + 1) % 4], z0), (*P[(i + 1) % 4], z1), (*P[i], z1)] for i in range(4)]


def masses(coll, dy=0.0, name="masses"):
    """A house (gable + a hip at the east end), a tower with a pyramid, a round tower with a cone; 0.4 m overhang; shift dy
    along y."""
    F, Rr = [], []
    x0, y0, x1, y1, ze, ov = 0.0, 0.0, 7.0, 4.5, 3.0, 0.4
    F += box_walls(x0, y0, x1, y1, 0.0, ze); Rr += [0] * 4
    hw = (y1 - y0) / 2 + ov; zr = ze + (y1 - y0) / 2; yc = (y0 + y1) / 2; ze_ = ze - ov
    xr1 = x1 - (y1 - y0) / 2
    e = [(x0 - ov, yc - hw, ze_), (x1 + ov, yc - hw, ze_), (x1 + ov, yc + hw, ze_), (x0 - ov, yc + hw, ze_)]
    r0, r1 = (x0 - ov, yc, zr), (xr1, yc, zr)
    F += [[e[0], e[1], r1, r0], [e[2], e[3], r0, r1], [e[1], e[2], r1]]; Rr += [2, 2, 2]
    F += [[(x0, y0, ze), (x0, y1, ze), (x0, yc, zr)]]; Rr += [0]
    cx, cy, a, zt, hp = 10.5, 2.0, 1.7, 7.0, 4.0
    F += box_walls(cx - a, cy - a, cx + a, cy + a, 0.0, zt); Rr += [0] * 4
    E = [(cx - a - 0.3, cy - a - 0.3), (cx + a + 0.3, cy - a - 0.3), (cx + a + 0.3, cy + a + 0.3), (cx - a - 0.3, cy + a + 0.3)]
    apex = (cx, cy, zt + hp)
    for i in range(4): F.append([(*E[i], zt - 0.1), (*E[(i + 1) % 4], zt - 0.1), apex]); Rr.append(2)
    cx2, cy2, r, zt2, hc, k = -4.5, 3.0, 2.0, 8.0, 5.0, 24
    ring = [(cx2 + r * math.cos(2 * math.pi * i / k), cy2 + r * math.sin(2 * math.pi * i / k)) for i in range(k)]
    for i in range(k): F.append([(*ring[i], 0.0), (*ring[(i + 1) % k], 0.0), (*ring[(i + 1) % k], zt2), (*ring[i], zt2)]); Rr.append(0)
    E2 = [(cx2 + (p[0] - cx2) * 1.12, cy2 + (p[1] - cy2) * 1.12) for p in ring]
    for i in range(k): F.append([(*E2[i], zt2 - 0.2), (*E2[(i + 1) % k], zt2 - 0.2), (cx2, cy2, zt2 + hc)]); Rr.append(2)
    F = [[(p[0], p[1] + dy, p[2]) for p in f] for f in F]
    return mesh(name, F, Rr, coll)


def static(sc):
    me = bpy.data.meshes.new("ground"); s = 300
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("ground", me); sc.collection.objects.link(ob)
    m = bpy.data.materials.new("g"); bs = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bs.inputs["Base Color"].default_value = (0.25, 0.23, 0.19, 1); me.materials.append(m)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.17, depth=1.5, location=(4.0, -2.0, 0.75)); b = bpy.context.object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(4.0, -2.0, 1.66)); h = bpy.context.object
    mf = bpy.data.materials.new("f"); bf = next(n for n in mf.node_tree.nodes if n.type == "BSDF_PRINCIPLED"); bf.inputs["Base Color"].default_value = (0.12, 0.12, 0.13, 1)
    for o in (b, h): o.data.materials.append(mf)


def shoot(sc, cam, name, tag, info):
    for v, (loc, tgt, lens) in VIEWS.items():
        cam.location = loc; cam.data.lens = lens
        cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(OUT, f"{name}_{v}_{tag}.png"); t1 = time.time()
        bpy.ops.render.render(write_still=True); info["render_s"][f"{v}_{tag}"] = round(time.time() - t1, 2)


def main():
    sc = bpy.context.scene
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    cam = look(sc); static(sc)
    mc = bpy.data.collections.new("buildings"); sc.collection.children.link(mc)
    mass = [masses(mc), masses(mc, FAR_DY, "masses_far")]
    plan = json.load(open(os.path.join(OUT, "plan_roofs.json")))
    cams = [v[0] for v in VIEWS.values()]
    fpx = RES[0] * VIEWS["deep"][2] / 36.0                                        # px per focal length — from the deep view (the far view is computed from it)
    names = ONLY or list(plan["kinds"]); res = {}
    for name in names:
        coll = bpy.data.collections.new(f"constructions_{name}"); sc.collection.children.link(coll)
        p = {"roofing": dict(plan["roofing"], parts={"all": name}, detail="cam"), "kinds": plan["kinds"], "substances": plan["substances"]}
        t0 = time.time(); st = BT.run(sc, p, coll, objs=mass, cams=cams, fpx=fpx, log=lambda *a: None)
        info = {"title": plan["kinds"][name]["title"], "substance": plan["kinds"][name]["substance"], "n_tiles": st["n_tiles"],
                "n_rows": st["n_rows"], "afar": st["afar"], "n_pitches_afar": st["n_pitches_afar"], "build_s": round(time.time() - t0, 2), "render_s": {}}
        shoot(sc, cam, name, 0, info)
        t1 = time.time(); cv = CV.run(sc, log=lambda *a: None); info["cover_s"] = round(time.time() - t1, 2)
        info["coverings"] = {k: cv[k] for k in ("n_objects", "far", "n_faces_piece", "n_faces_tile", "substances", "without_substance")}
        shoot(sc, cam, name, 1, info)
        for ob in list(coll.objects): bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(coll)
        res[name] = info; print("ROW", name, json.dumps(info, ensure_ascii=False), flush=True)
    path = os.path.join(OUT, "run_roofs_tex.json")
    if ONLY and os.path.exists(path): res = dict(json.load(open(path)), **res)
    res = {n: res[n] for n in plan["kinds"] if n in res}
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps({k: [v["n_tiles"], v["coverings"]["n_objects"], v["coverings"]["without_substance"]] for k, v in res.items()}, ensure_ascii=False))


main()
