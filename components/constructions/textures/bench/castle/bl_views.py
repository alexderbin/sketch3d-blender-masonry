"""Close-ups of the construction substance on the castle (a check of the "textures" step, not a pipeline output): the
control .blend of the "light" step of the castle_textures.yaml run (masonry, roofing, paving, timber — all elements,
textures, daylight) → frames: a wall up close, a half-timbered facade, a roof from above, the courtyard and the road at
human eye level. The camera is placed from a face of a construction object (masonry masonry_, roofing roof_, paving
pave_ of the courtyard and the road, timber timber_): along the face normal at its own distance, target visibility by a
ray (the first hit of the ray is at the target; the camera is not inside a body and is above the ground). Cameras placed
from the occupancy layer missed: heights and walls between the camera and the target. The sun is on the camera's side.

blender -b <run>/NN_light/scene.blend --factory-startup --python-exit-code 1 -P core/blender/run.py --
  components/constructions/textures/bench/castle/bl_views.py -- <run> [res=1280x720] [views=wall,fachwerk,roof,yard,road] [seed=1]
  → <run>/views/<view>.png, views/views.json"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

ARGV = sys.argv[sys.argv.index("--") + 1:]
RUN = os.path.abspath(ARGV[0]); KW = dict(a.split("=", 1) for a in ARGV[1:] if "=" in a)
RES = tuple(int(v) for v in KW.get("res", "1280x720").split("x")); OUT = os.path.join(RUN, "views")
# view: (object prefix, paving part, face: wall | up | floor, distance, focal length in mm)
KINDS = {"wall": ("masonry_", None, "wall", 6.0, 28), "fachwerk": ("timber_", None, "wall", 6.5, 28),
         "roof": ("roof_", None, "up", 6.0, 30), "yard": ("pave_", "court", "floor", 7.0, 26), "road": ("pave_", "road", "floor", 9.0, 26)}


def faces(ob, side):
    """Centers and normals of an object's faces (world), selected by side: wall — normal sideways, up — up and sideways (a
    slope), floor — up. → (centers, normals)."""
    me = ob.data; n = len(me.polygons)
    c = np.empty(n * 3, np.float32); me.polygons.foreach_get("center", c); c = c.reshape(-1, 3)
    nr = np.empty(n * 3, np.float32); me.polygons.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
    M = np.asarray(ob.matrix_world, float); c = c @ M[:3, :3].T + M[:3, 3]
    nr = nr @ np.linalg.inv(M[:3, :3]); nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-9)
    sel = {"wall": np.abs(nr[:, 2]) < 0.25, "up": (nr[:, 2] > 0.35) & (nr[:, 2] < 0.95), "floor": nr[:, 2] > 0.9}[side]
    return c[sel], nr[sel]


def hit(sc, a, b):
    """First hit of the ray a → b: (distance, object) or (None, None)."""
    dg = bpy.context.evaluated_depsgraph_get(); d = Vector(b) - Vector(a); L = d.length
    ok, loc, _n, _i, ob, _m = sc.ray_cast(dg, Vector(a), d.normalized(), distance=L + 1.0)
    return ((Vector(loc) - Vector(a)).length, ob) if ok else (None, None)


def place(sc, kind, rng):
    """Camera of a view: a random face of the target objects → a shooting point along the normal (floor — at human eye level,
    1.7 m), visibility checks. → (from, to, focal length, object) or None."""
    pre, part, side, dist, lens = KINDS[kind]
    obs = [o for o in sc.objects if o.type == "MESH" and o.name.startswith(pre) and not o.name.startswith(("roof_far_", "pave_bed_"))
           and (part is None or json.loads(o.get("sk_paving", "{}")).get("part") == part)]
    cand = []
    for ob in obs:
        C, Nn = faces(ob, side)
        if len(C): k = rng.choice(len(C), min(len(C), 60), replace=False); cand += [(ob, C[i], Nn[i]) for i in k]
    rng.shuffle(cand)
    down = Vector((0.0, 0.0, -1.0))
    for ob, c, n in cand[:400]:
        if side == "floor":                                                     # human eye level: the camera is on the floor of the same part
            a = rng.uniform(0, 2 * math.pi); loc = c + np.array([math.cos(a), math.sin(a), 0.0]) * dist + np.array([0, 0, 1.7])
            tgt = c + np.array([0, 0, 0.3])
            d0, _o0 = hit(sc, loc, loc + np.array([0, 0, -3.0]))               # stands on the pavement or on the ground nearby (road — 4 m)
            if d0 is None or not 1.2 < d0 < 2.4: continue
        else:
            h = n.copy()
            if side == "wall": h[2] = 0.0; h /= max(np.linalg.norm(h), 1e-9)
            loc = c + h * dist + np.array([0, 0, 0.8 if side == "wall" else 2.0]); tgt = c
            dd, _o = hit(sc, loc, loc + np.array([0, 0, -200.0]))
            if dd is None or dd < 1.0: continue                                  # the camera is inside a body or right above something
        d1, o1 = hit(sc, loc, tgt)
        full = float(np.linalg.norm(np.asarray(tgt) - np.asarray(loc)))
        if d1 is None or d1 < full - 0.35: continue                              # there is something between the camera and the target
        face = c + n * 0.05                                                      # the target face is open to the camera: for the back face of a block
        d2, _o2 = hit(sc, face, loc)                                             # (inside the wall mass) the ray hits the mass — the wrong face
        if d2 is not None and d2 < float(np.linalg.norm(np.asarray(loc) - face)) - 0.1: continue
        up_ok, _ = hit(sc, loc, loc + np.array([0, 0, 0.6]))
        if up_ok is not None: continue                                           # the camera is under an awning or in a vault
        return loc, tgt, lens, ob.name
    return None


def check_light(sc):
    """Check lighting, as on the substance stands (bench/roofs, bench/ground): the Blender sky and a sun; the scene's light is
    off — the daylight of the "light" step is assembled by the "frames" output (lit/sky, tone), without it the raw frame is
    blue and dark."""
    for o in sc.objects:
        if o.type == "LIGHT": o.hide_render = True
    sc.view_settings.view_transform = "AgX"; sc.view_settings.exposure = -1.3
    w = bpy.data.worlds.new("check"); sc.world = w; nt = w.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND"); sky = nt.nodes.new("ShaderNodeTexSky")
    try: sky.sky_type = "MULTIPLE_SCATTERING"
    except Exception: pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.12
    sun = bpy.data.objects.new("check_sun", bpy.data.lights.new("check_sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 4.5; sun.data.angle = math.radians(1.0); sun.data.color = (1.0, 0.94, 0.86)
    return sun


def key_sun(sun, loc, tgt, side=40.0, elev=38.0):
    """The sun on the camera's side (off to the side by side°, elevation elev°)."""
    b = np.asarray(loc, float)[:2] - np.asarray(tgt, float)[:2]; b /= max(np.linalg.norm(b), 1e-9); a = math.radians(side)
    h = np.array([b[0] * math.cos(a) - b[1] * math.sin(a), b[0] * math.sin(a) + b[1] * math.cos(a)]); e = math.radians(elev)
    sun.rotation_euler = Vector((h[0] * math.cos(e), h[1] * math.cos(e), math.sin(e))).to_track_quat("Z", "Y").to_euler()


def main():
    sc = bpy.context.scene; os.makedirs(OUT, exist_ok=True)
    sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 32
    sc.render.resolution_x, sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
    cam = bpy.data.objects.new("vcam", bpy.data.cameras.new("vcam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.clip_start = 0.05; cam.data.clip_end = 3000; res = {}; sun = check_light(sc)
    rng = np.random.default_rng(int(KW.get("seed", 1)))
    only = KW.get("views"); kinds = [k for k in KINDS if not only or k in only.split(",")]
    for name in kinds:
        p = place(sc, name, rng)
        png = os.path.join(OUT, f"{name}.png")
        if os.path.exists(png): os.remove(png)                                    # a frame from a previous run must not get into the sheet
        if p is None: res[name] = {"no": "no visible face"}; print("ROW", name, "no camera", flush=True); continue
        loc, tgt, lens, obn = p
        cam.location = loc; cam.data.lens = lens; cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        key_sun(sun, loc, tgt)
        sc.render.filepath = os.path.join(OUT, f"{name}.png"); t1 = time.time(); bpy.ops.render.render(write_still=True)
        res[name] = dict(loc=[round(float(x), 2) for x in loc], tgt=[round(float(x), 2) for x in tgt], lens=lens, object=obn,
                         frame_s=round(time.time() - t1, 1))
        print("ROW", name, json.dumps(res[name], ensure_ascii=False), flush=True)
    json.dump(res, open(os.path.join(OUT, "views.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps({k: v.get("frame_s", v.get("no")) for k, v in res.items()}, ensure_ascii=False))


main()
