"""Substance masks from the masonry output — the "textures" step computes them itself from the geometry (the Constructions
contract: we read sk_role, sk_stone, Col and the mesh; the masonry code does not change):
  frame   — the frame of a stone face in its plane: sk_lo, sk_hi (FACE FLOAT2 — the limits of u, v of the sk_uv
            unrolling on the face) → in the shader the distance to the face edge (exact for a rectangular face, for
            arch voussoirs by the frame);
  bevels  — chamfers: stone faces at 20–70° to its face (the face is the largest face of the stone) → sk_bevel (FACE
            FLOAT 1);
  mortar  — faces of the building mass behind the stones: a ray along the normal from the face points hits a stone
            closer than max_d → mortar in the joints (the wall mass behind the stones is mortar), the rest by the
            palette.
All of it is numpy and BVH, with no per-face loops in Python (except the mass: hundreds of faces)."""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _arr(coll, prop, n, k=1, dt=np.float32):
    a = np.empty(n * k, dt); coll.foreach_get(prop, a)
    return a.reshape(-1, k) if k > 1 else a


def _attr(me, name, kind, domain, data, prop):
    a = me.attributes.get(name)
    if a is not None and (a.data_type != kind or a.domain != domain):
        me.attributes.remove(a); a = None
    a = a or me.attributes.new(name, kind, domain)
    a.data.foreach_set(prop, np.ascontiguousarray(data, np.float32).ravel())
    return name


def frame(ob, uv="sk_uv"):
    """sk_lo, sk_hi — the limits of the uv unrolling (meters along the face plane) on each face. → number of faces or None."""
    me = ob.data; L = me.uv_layers.get(uv)
    if L is None or not me.polygons:
        return None
    n = len(me.polygons)
    UV = _arr(L.data, "uv", len(me.loops), 2)
    ls = _arr(me.polygons, "loop_start", n, dt=np.int32)
    order = np.argsort(ls); starts = ls[order]                          # the face corners run consecutively from loop_start
    lo = np.empty((n, 2), np.float32); hi = np.empty((n, 2), np.float32)
    lo[order] = np.minimum.reduceat(UV, starts, axis=0); hi[order] = np.maximum.reduceat(UV, starts, axis=0)
    _attr(me, "sk_lo", "FLOAT2", "FACE", lo, "vector"); _attr(me, "sk_hi", "FLOAT2", "FACE", hi, "vector")
    return n


def bevels(ob, attr="sk_stone", lo_deg=20.0, hi_deg=70.0):
    """sk_bevel = 1 on chamfers: a stone face at lo…hi degrees to the stone's face (its largest face). → number of chamfers."""
    me = ob.data; a = me.attributes.get(attr)
    if a is None or a.domain != "FACE":
        return None
    n = len(me.polygons)
    sid = _arr(a.data, "value", n, dt=np.int32)
    N = _arr(me.polygons, "normal", n, 3); A = _arr(me.polygons, "area", n)
    order = np.lexsort((-A, sid))                                       # by stone, within it from the largest face
    s_sorted = sid[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    dom_face = order[first]                                             # the face of each stone
    grp = np.cumsum(first) - 1                                          # the stone number in the sorted order
    dom_n = np.empty((n, 3), np.float32); dom_n[order] = N[dom_face][grp]
    c = (N * dom_n).sum(1)
    bev = (c < math.cos(math.radians(lo_deg))) & (c > math.cos(math.radians(hi_deg))) & (sid > 0)
    _attr(me, "sk_bevel", "FLOAT", "FACE", bev.astype(np.float32), "value")
    return int(bev.sum())


def ground_height(ob, ground, attr="sk_stone", out="sk_h"):
    """sk_h — the height of a stone above the ground (m, at the stone's center: the area-weighted mean of its face centers; a
    ray down to ground). Faces without a stone number (plugs) use their own center. The position in the wall for weathering:
    the bottom is the zone of splashes and damp."""
    me = ob.data; a = me.attributes.get(attr)
    if a is None or ground is None or not me.polygons:
        return None
    n = len(me.polygons)
    sid = _arr(a.data, "value", n, dt=np.int32)
    C = _arr(me.polygons, "center", n, 3); A = _arr(me.polygons, "area", n)
    M4 = np.array(ob.matrix_world); C = C @ M4[:3, :3].T + M4[:3, 3]
    keys = np.where(sid > 0, sid, -1 - np.arange(n))                    # without a number — the face itself
    uniq, inv = np.unique(keys, return_inverse=True)
    w = np.bincount(inv, A) + 1e-9
    cen = np.stack([np.bincount(inv, A * C[:, k]) / w for k in range(3)], 1)
    dg = bpy.context.evaluated_depsgraph_get(); t = BVHTree.FromObject(ground, dg)
    gi = ground.matrix_world.inverted(); down = (ground.matrix_world.to_3x3().inverted() @ Vector((0, 0, -1))).normalized()
    h = np.empty(len(uniq), np.float32)
    for i, c in enumerate(cen):
        p = Vector((float(c[0]), float(c[1]), float(c[2]) + 0.01))
        r = t.ray_cast(gi @ p, down, 200.0)
        h[i] = (p.z - (ground.matrix_world @ r[0]).z) if r[0] is not None else 99.0
    _attr(me, out, "FLOAT", "FACE", h[inv], "value")
    return int(len(uniq))


def mortar(mass, stones, max_d=0.3, probe=0.35):
    """Faces of the mass behind the stones → {object: face mask}. A ray along the face normal from the center and from four
    points toward its corners (fraction probe); mortar — if a stone is closer than max_d for at least three of the five rays.
    The BVH of the stones is in their space."""
    dg = bpy.context.evaluated_depsgraph_get()
    trees = [(BVHTree.FromObject(o, dg), o.matrix_world.inverted(), o.matrix_world.to_3x3().inverted())
             for o in stones if o.type == "MESH" and len(o.data.polygons)]
    out = {}
    for ob in mass:
        me = ob.data; M4 = ob.matrix_world; R = M4.to_3x3()
        mask = np.zeros(len(me.polygons), bool)
        for i, p in enumerate(me.polygons):
            nw = (R @ p.normal).normalized(); cw = M4 @ p.center
            pts = [cw] + [cw.lerp(M4 @ me.vertices[v].co, probe) for v in p.vertices[:4]]
            hits = 0
            for q in pts:
                o_ = q + nw * 0.002
                for t, inv, rinv in trees:
                    if t.ray_cast(inv @ o_, (rinv @ nw).normalized(), max_d)[0] is not None:
                        hits += 1; break
            mask[i] = hits >= 3
        out[ob.name] = mask
    return out
