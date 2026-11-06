"""Clay of the levels in Blender: the folder's model.npz (levels/cases.py) → one mesh with face color by class and tone →
Workbench (a studio in world axes, cavity, outline, no cast shadow) → the model's camera frames into <folder>/clay_<view>.png.

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/levels/bl_clay.py -- <folder>
    [views=front,elev] [res=1600x1000] [mono=1] [explode=<m>] [tag=<prefix>]
mono — a single clay without class colors; explode — exploded levels (each level lifted by explode m)."""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

PAL = {0: (0.56, 0.53, 0.49), 1: (0.68, 0.62, 0.54), 2: (0.74, 0.72, 0.66), 3: (0.87, 0.85, 0.81), 4: (0.80, 0.78, 0.74),
       5: (0.72, 0.64, 0.58), 6: (0.52, 0.49, 0.46), 7: (0.79, 0.78, 0.75), 8: (0.83, 0.83, 0.81), 9: (0.76, 0.68, 0.58),
       10: (0.81, 0.79, 0.75), 11: (0.79, 0.79, 0.77), 12: (0.46, 0.53, 0.60), 13: (0.77, 0.74, 0.69)}
MONO = (0.80, 0.78, 0.74)


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    folder = a[0]; kw = dict(x.split("=", 1) for x in a[1:]); return folder, kw


def mesh(folder, mono=False, explode=0.0):
    z = np.load(os.path.join(folder, "model.npz"))
    V = z["verts"].astype(np.float64); T = z["tris"]; C = z["cls"]; Tn = z["tone"]; L = z["level"]
    if explode:
        lv = np.zeros(len(V), np.int64); lv[T.ravel()] = np.repeat(L, 3); V = V.copy(); V[:, 2] += explode * lv
    me = bpy.data.meshes.new("levels"); me.from_pydata(V.tolist(), [], T.tolist()); me.update()
    rgb = np.array([MONO if mono else PAL.get(int(c), MONO) for c in C], np.float32) * np.clip(Tn, 0.2, 1.3)[:, None]
    if "tint" in z.files and not mono: rgb = rgb * z["tint"]
    rgb = np.clip(rgb, 0, 1); ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    ca.data.foreach_set("color", np.column_stack([np.repeat(rgb, 3, axis=0), np.ones(len(T) * 3, np.float32)]).ravel())
    me.color_attributes.active_color = ca
    ob = bpy.data.objects.new("levels", me); bpy.context.scene.collection.objects.link(ob)
    return ob, json.loads(str(z["meta"]))


def setup(sc, res, key_az=-125.0):
    sc.render.engine = "BLENDER_WORKBENCH"; d = sc.display; sh = d.shading; d.render_aa = "8"
    sh.light, sh.color_type = "STUDIO", "VERTEX"
    sh.show_cavity, sh.cavity_type = True, "BOTH"
    sh.cavity_ridge_factor, sh.cavity_valley_factor = 0.6, 1.2
    sh.curvature_ridge_factor, sh.curvature_valley_factor = 0.5, 0.8
    sh.show_shadows = False; sh.show_specular_highlight = False
    sh.show_object_outline, sh.object_outline_color = True, (0.12, 0.11, 0.10)
    sh.use_world_space_lighting = False                                    # studio light in the camera axes: a model photo, the key light from the top left
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.93, 0.92, 0.90)
    vs = sc.view_settings; vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    im = sc.render.image_settings; im.file_format, im.color_mode = "PNG", "RGB"


def camera(sc, v, name, bounds, aspect):
    """A camera by direction: az (from +x counterclockwise, where it looks from), el (above the horizon), lens or ortho; the
    distance (or the ortho scale) is chosen so that the whole extent fit (or the whole model) fits with the margin margin;
    depth — only depth m behind the near face of the extent is visible (a section without a far background). Or explicit
    loc/tgt."""
    cd = bpy.data.cameras.new(name); cam = bpy.data.objects.new(name, cd); sc.collection.objects.link(cam)
    cd.clip_start, cd.clip_end = 0.1, 20000.0
    if "loc" in v:
        loc = Vector(v["loc"]); tgt = Vector(v["tgt"]); cam.location = loc
        cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
        if v.get("ortho"): cd.type = "ORTHO"; cd.ortho_scale = float(v["ortho"])
        else: cd.lens = float(v.get("lens", 35))
        return cam
    lo, hi = (np.asarray(v["fit"][:3], float), np.asarray(v["fit"][3:], float)) if v.get("fit") else bounds
    tgt = (lo + hi) / 2 + np.asarray(v.get("shift", (0, 0, 0)), float)
    az, el = math.radians(v.get("az", -125)), math.radians(v.get("el", 25))
    dvec = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fwd = -dvec; right = np.cross(fwd, (0, 0, 1.0)); right /= max(np.linalg.norm(right), 1e-9); up = np.cross(right, fwd)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - tgt
    m = float(v.get("margin", 1.06))
    if v.get("ortho"):                                                       # ortho scale — by the longer side of the frame
        cd.type = "ORTHO"; w = 2 * np.abs(C @ right).max(); h = 2 * np.abs(C @ up).max()
        cd.ortho_scale = (max(w, h * aspect) if aspect >= 1 else max(h, w / aspect)) * m; D = 3000.0
    else:                                                                    # 36 mm sensor — along the longer side of the frame
        cd.lens = float(v.get("lens", 35)); t = 18.0 / cd.lens
        tx, ty = (t, t / aspect) if aspect >= 1 else (t * aspect, t)
        D = float(max((C @ dvec)[i] + max(abs(C[i] @ right) / tx, abs(C[i] @ up) / ty) * m for i in range(len(C))))
    loc = tgt + dvec * D; cam.location = Vector(loc.tolist())
    cam.rotation_euler = Vector(fwd.tolist()).to_track_quat("-Z", "Y").to_euler()
    if v.get("depth"): cd.clip_end = float(D + (C @ fwd).min() + float(v["depth"]))   # a section without a far background: depth m behind the face
    return cam


def main():
    folder, kw = args(); sc = bpy.context.scene
    for o in list(sc.objects): bpy.data.objects.remove(o, do_unlink=True)
    mono = kw.get("mono") == "1"; explode = float(kw.get("explode", 0) or 0)
    ob, meta = mesh(folder, mono, explode)
    res = tuple(int(x) for x in kw.get("res", "1600x1000").split("x"))
    setup(sc, res, float(kw.get("key", meta.get("key_az", -150))))
    co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    bounds = (co.min(0), co.max(0))
    views = meta.get("views", {})
    if explode and meta.get("views_explode"): views = meta["views_explode"]
    want = kw.get("views", ",".join(views)).split(",")
    tag = kw.get("tag", "explode_" if explode else ("mono_" if mono else ""))
    for name in want:
        v = views[name]; sx, sy = (res if not v.get("res") else v["res"])
        sc.render.resolution_x, sc.render.resolution_y = sx, sy
        cam = camera(sc, v, name, bounds, sx / sy); sc.camera = cam
        sc.render.filepath = os.path.join(folder, f"clay_{tag}{name}.png"); bpy.ops.render.render(write_still=True)
        print("written", sc.render.filepath)


if __name__ == "__main__":
    main()
