"""Fingerprint of a building model (the reference for golden.py --blender): the type's builder builds the plan
(one cheap 64×48 view), then over all meshes by name — vertices in world space (to 1 mm), topology, color
Col, face roles sk_role, the cell and the side of a face (sk_space, sk_side). If the fingerprint matches,
the model is the same to the millimeter and has the same face contract.

blender -b -P core/blender/run.py -- components/buildings/bench/bl_fingerprint.py -- <builder bl_*.py> <plan folder>   → line __FP__ {…}"""
import hashlib
import json
import os
import runpy
import sys

import bpy
import numpy as np

ARGV = sys.argv[sys.argv.index("--") + 1:]
BUILDER, PLAN = os.path.abspath(ARGV[0]), os.path.abspath(ARGV[1])
KIND = os.path.basename(BUILDER)[3:-3]
VIEW = {"castle": "top", "temple": "plan", "edifice": "top"}.get(KIND, "top")
sys.argv = [sys.argv[0], "--", PLAN, f"views={VIEW}", "res=64x48"]
runpy.run_path(BUILDER, run_name="__main__")


def arr(coll, key, n, dt):
    a = np.empty(n, dt); coll.foreach_get(key, a); return a


H = {k: hashlib.sha1() for k in ("vertices", "topology", "color", "roles", "cells")}; nobj = nf = 0
for o in sorted((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: o.name):
    me = o.data; nv, nl, np_ = len(me.vertices), len(me.loops), len(me.polygons); nobj += 1; nf += np_
    M = np.array(o.matrix_world, np.float64)
    W = arr(me.vertices, "co", nv * 3, np.float32).astype(np.float64).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    H["vertices"].update(o.name.encode()); H["vertices"].update((np.round(W, 3) + 0.0).tobytes())
    H["topology"].update(arr(me.polygons, "loop_total", np_, np.int32).tobytes()); H["topology"].update(arr(me.loops, "vertex_index", nl, np.int32).tobytes())
    ca = me.color_attributes.get("Col")
    if ca is not None:
        n = len(ca.data) * 4; H["color"].update(ca.domain.encode()); H["color"].update((np.round(arr(ca.data, "color", n, np.float32), 3) + 0.0).tobytes())
    for name, key in (("sk_role", "roles"), ("sk_space", "cells"), ("sk_side", "cells")):
        a = me.attributes.get(name)
        if a is not None: H[key].update(name.encode()); H[key].update(arr(a.data, "value", len(a.data), np.int32).tobytes())
print("__FP__", json.dumps(dict({k: m.hexdigest()[:16] for k, m in H.items()}, n_objects=nobj, n_faces=nf), ensure_ascii=False))
