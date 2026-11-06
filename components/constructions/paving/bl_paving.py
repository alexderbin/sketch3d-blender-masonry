"""The "paving" step in Blender — a part of the Constructions component.

Areas — without building code: floors are faces with the role "floor" (sk_role = 8) of building objects (flat connected
groups → contour); the castle courtyard and road are the scene occupancy layer (core.layers: yard, road; holes are
buildings and towers); areas of the description (constructions.yaml → paving: areas). Height: in the street it is the
ground (a ray down onto the site's terrain), for a floor it is the plane of the face.
Stones (paving.units) → geometry: a flat top with a chamfer, a dome (cobblestone), a half-log (half_logs), a plank; tilt
and sinking depend on the stone number; the joint is the bedding under the stones. Camera-dependent view in 4 m chunks
(as in masonry): px per stone ≥ Tlo — stones with a chamfer and a dome, Tfar…Tlo — simple ones, below Tfar — only the
bedding with a pattern in the shader (a cheap far view).
Output contract: objects pave_<n> (stones) and pave_bed_<n> (bedding and far view) in the step's collection; on faces
sk_role (floor | wood; bedding — mortar), sk_stone — the stone number (7·10⁸ + area·10⁶ + chunk·10⁴ + number), tone — Col;
property sk_paving (kind, part, area).

  import bl_paving as BP; stats = BP.run(sc, plan, coll, objs=…, scene=<scene folder>, cams=…, fpx=…)
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
from components.constructions.paving import paving as P  # noqa: E402
from components.constructions.masonry.bl_stones import Acc  # noqa: E402
from core.contract.contract import ROLES  # noqa: E402  (face roles per the core contract: floor is 8)


# ── ground ──────────────────────────────────────────────────────────────────────────────────────────────────────
OVER_UP = 1.5                                       # building slabs near the ground: searched up to 1.5 m above the reference height


class Ground:
    """Ground height: a ray down onto the site meshes (sk_layer "place" / "relief", names terrain, ground); none — z0.
    over — building objects: their slabs near the ground (courtyard, road, castle ramp) have a top above the terrain; the ray
    goes from the reference height ref (courtyard level, road axis; none — the terrain) + OVER_UP downward, and the first
    horizontal face is taken (the castle road slab is flipped — its normal points down; the gate vault is above the search
    window)."""

    def __init__(self, sc, z0=0.0, over=()):
        dg = bpy.context.evaluated_depsgraph_get(); self.trees = []; self.over = []; self.z0 = z0
        for ob in sc.objects:
            if ob.type != "MESH": continue
            nm = ob.name.split(".")[0].lower()
            if ob.get("sk_layer") in ("site", "relief") or nm.startswith(("terrain", "ground", "relief")):
                if nm.startswith(("landmark", "water")): continue
                try: self.trees.append(BVHTree.FromObject(ob, dg))
                except Exception: pass
        for ob in over:
            if ob.type != "MESH": continue
            try: self.over.append(BVHTree.FromObject(ob, dg))
            except Exception: pass

    def z(self, x, y, ref=None):
        best = None; down = Vector((0.0, 0.0, -1.0))
        for t in self.trees:
            hit = t.ray_cast(Vector((float(x), float(y), 5000.0)), down)
            if hit[0] is not None and (best is None or hit[0].z > best): best = hit[0].z
        zt = float(best) if best is not None else self.z0
        top = max(zt, ref if ref is not None else zt) + OVER_UP
        for t in self.over:
            o = Vector((float(x), float(y), top)); lim = top - zt + 0.3
            for _ in range(6):
                hit = t.ray_cast(o, down, lim)
                if hit[0] is None: break
                if abs(hit[1].z) > 0.9: zt = max(zt, float(hit[0].z)); break  # a slab (may be flipped: normal down); a slope — the ray continues
                lim -= o.z - hit[0].z + 0.001; o = hit[0] + down * 0.001
                if lim <= 0: break
        return zt

    def zs(self, pts, ref=None):
        pts = np.asarray(pts, float)
        r = ref(pts) if callable(ref) else np.full(len(pts), np.nan if ref is None else float(ref))
        return np.array([self.z(x, y, None if np.isnan(q) else q) for (x, y), q in zip(pts, r)], float)


def in_polys(xy, polys):
    """Points (x, y) inside any of the polygons (even-odd rule)."""
    xy = np.atleast_2d(np.asarray(xy, float)); out = np.zeros(len(xy), bool)
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    for Q in polys:
        A, B = Q, np.roll(Q, -1, 0)
        c = ((A[:, 1] > y) != (B[:, 1] > y)) & (x < (B[:, 0] - A[:, 0]) * (y - A[:, 1]) / (B[:, 1] - A[:, 1] + 1e-12) + A[:, 0])
        out |= np.count_nonzero(c, axis=1) % 2 == 1
    return out


# ── areas from the scene and the plan ───────────────────────────────────────────────────────────────────────────
def floor_regions(objs, weld=0.003):
    """Floors: upward-facing faces with the role "floor" — connected flat groups → contour (outer loop) and height."""
    F = []
    for ob in objs:
        if ob.type != "MESH": continue
        me = ob.data; ra = me.attributes.get("sk_role")
        if ra is None or ra.domain != "FACE": continue
        r = np.empty(len(me.polygons), np.int32); ra.data.foreach_get("value", r)
        Mw = np.asarray(ob.matrix_world, float)
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        for p in me.polygons:
            if r[p.index] != ROLES.get("floor", 8): continue
            P_ = co[list(p.vertices)]; n = np.sum(np.cross(P_, np.roll(P_, -1, 0)), axis=0)
            if np.linalg.norm(n) < 1e-9 or n[2] / np.linalg.norm(n) < 0.95: continue
            F.append(P_)
    if not F: return []
    key = lambda p: (int(round(p[0] / weld)), int(round(p[1] / weld)), int(round(p[2] / 0.05)))
    ids, fv = {}, []
    for P_ in F: fv.append([ids.setdefault(key(p), len(ids)) for p in P_])
    parent = list(range(len(F)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = {}
    for fi, vs in enumerate(fv):
        for v in vs:
            if v in owner: parent[find(fi)] = find(owner[v])
            else: owner[v] = fi
    groups = {}
    for fi in range(len(F)): groups.setdefault(find(fi), []).append(fi)
    rev = {v: k for k, v in ids.items()}; out = []
    for fis in groups.values():
        cnt = {}
        for fi in fis:
            vs = fv[fi]
            for a, b in zip(vs, vs[1:] + vs[:1]): e = (min(a, b), max(a, b)); cnt[e] = cnt.get(e, 0) + 1
        bnd = [e for e, c in cnt.items() if c == 1]
        if len(bnd) < 3: continue
        nb = {}
        for a, b in bnd: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
        loops, seen = [], set()
        for s in nb:
            if s in seen: continue
            loop, cur, prev = [s], s, None
            while True:
                seen.add(cur); nxt = [x for x in nb[cur] if x != prev and x not in loop[-2:-1]]
                if not nxt: break
                prev, cur = cur, nxt[0]
                if cur == s or cur in seen: break
                loop.append(cur)
            if len(loop) >= 3: loops.append(loop)
        if not loops: continue
        loop = max(loops, key=lambda L: abs(P.area([(rev[v][0] * weld, rev[v][1] * weld) for v in L])))
        poly = [(rev[v][0] * weld, rev[v][1] * weld) for v in loop]
        z = float(np.mean([F[fi][:, 2].mean() for fi in fis]))
        if abs(P.area(poly)) > 0.5: out.append(dict(region=P.Area(poly), part="floor", z=z, name="floor"))
    return out


def _circle(c, r, n=24):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False); return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def layer_regions(scene, width=4.0):
    """Courtyard and road — the scene occupancy layer (core.layers, written by Buildings): the courtyard is yard (reference —
    the courtyard level), the road is road (a polyline x, y, z; reference — the axis height at the nearest point); holes are
    buildings (building) and towers with a keep (tower: circle r + 0.1)."""
    out = []
    if not scene: return out
    from core import layers as LY
    holes = [np.asarray(b["poly"], float)[:, :2] for b in LY.occupancy(scene, ("building",)) if b.get("poly")]
    for t in LY.occupancy(scene, ("tower",)):
        if t.get("xy") and t.get("r"): holes.append(_circle(t["xy"], float(t["r"]) + 0.1))
    for w in LY.occupancy(scene, ("yard",)):
        out.append(dict(region=P.Area(w["poly"]), part="court", z=None, ref=w.get("level"), holes=holes, name=f"courtyard {w.get('id', '')}"))
    for rd in LY.occupancy(scene, ("road",)):
        if len(rd.get("line") or []) < 2: continue
        R = np.asarray(rd["line"], float)
        ref = (lambda xy, R=R: R[np.argmin(((np.asarray(xy)[:, None, :2] - R[None, :, :2]) ** 2).sum(-1), axis=1), 2]) if R.shape[1] > 2 else None
        out.append(dict(region=P.Path(rd["line"], width), part="road", z=None, ref=ref, holes=holes, name="road"))
    return out


def desc_regions(spec):
    out = []
    for i, r in enumerate(spec or []):
        if r.get("path"): reg = P.Path(r["path"], float(r.get("width", 3.0)))
        elif r.get("outline"): reg = P.Area(r["outline"], math.radians(float(r["angle"])) if r.get("angle") is not None else None)
        else: continue
        out.append(dict(region=reg, part=r.get("part", "areas"), z=r.get("z"), name=r.get("name", f"area {i}"), kind=r.get("kind")))
    return out


# ── stone geometry ──────────────────────────────────────────────────────────────────────────────────────────────
def tone(plan, key, tint):
    v = (plan.get("substances") or {}).get(key) or [0.4, 0.38, 0.35]
    return tuple(float(x) for x in np.clip(np.asarray(v, float) * tint, 0, 1))


def _inset(P2, d):
    q = P.inset(P2, d) if len(P2) >= 3 else []
    return np.asarray(q, float) if len(q) >= 3 else None


def stone(acc, reg, st, k, zf, col, role, full, live):
    """Stone: contour (u, v) → world; the bottom follows the ground (zf: heights of the xy points), the top depends on the kind
    (flat with a chamfer, dome, log, plank); tilt and sinking depend on the number."""
    uv = np.asarray(st["poly"], float)
    if reg.kind == "path" or max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) > 0.6: uv = P.densify(uv, 0.3)
    xy = reg.xy(uv[:, 0], uv[:, 1]); c = xy.mean(0)
    if max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) < 0.6:                           # small stone: the ground plane from three rays near the middle
        zz = zf(np.array([c, c + (0.1, 0.0), c + (0.0, 0.1)])); gx, gy = (zz[1] - zz[0]) / 0.1, (zz[2] - zz[0]) / 0.1
        if max(abs(zz[1] - zz[0]), abs(zz[2] - zz[0])) > 0.04: gx = gy = 0.0      # a ray on a step (slab edge): no slope, level at the middle
        zf = lambda P2, z0=zz[0], gx=gx, gy=gy, c=c: z0 + gx * (np.asarray(P2)[:, 0] - c[0]) + gy * (np.asarray(P2)[:, 1] - c[1])
    r = np.random.default_rng([int(st["sid"]) & 0x7FFFFFFF, 41])
    tilt = math.radians(3.0 if k["top"] == "dome" else 1.2) * live
    ax, ay = r.uniform(-tilt, tilt), r.uniform(-tilt, tilt); dz = r.uniform(-0.004, 0.004) * live
    h = float(k["visible"]); top = k["top"] if full else "simple"
    def lift(P2, hh): return np.column_stack([P2[:, 0], P2[:, 1], zf(P2) + hh + dz + math.tan(ax) * (P2[:, 0] - c[0]) + math.tan(ay) * (P2[:, 1] - c[1])])
    sid = st["sid"]
    if top == "log":                                                          # half-log: an arc across the width (along u)
        u0, u1 = uv[:, 0].min(), uv[:, 0].max(); v0, v1 = uv[:, 1].min(), uv[:, 1].max()
        ts = np.linspace(0.0, 1.0, 7); vs = np.linspace(v0, v1, max(2, int(math.ceil((v1 - v0) / 0.3)) + 1))
        G = []
        for v in vs:
            xy_ = reg.xy(u0 + (u1 - u0) * ts, np.full_like(ts, v)); G.append(lift(xy_, 0.0) + np.outer(0.15 * h + 0.85 * h * np.sin(np.pi * ts), [0, 0, 1]))
        for a in range(len(vs) - 1):
            for i in range(len(ts) - 1): acc.add([G[a][i], G[a][i + 1], G[a + 1][i + 1], G[a + 1][i]], col, role, sid)
        for row in (G[0], G[-1]):
            base = row.copy(); base[:, 2] -= 0.15 * h + 0.85 * h * np.sin(np.pi * ts) + 0.02
            for i in range(len(ts) - 1): acc.add([base[i], base[i + 1], row[i + 1], row[i]], col, role, sid)
        return
    if top == "dome":
        mid = _inset(xy, 0.3 * math.sqrt(abs(P.area(xy)) / math.pi))
        rim = lift(xy, 0.35 * h)
        if mid is None: acc.add(lift(xy, h), col, role, sid)
        else:
            M_ = lift(np.asarray([c + (p - c) * 0.62 for p in xy]), 0.85 * h); cz = lift(c[None], h)[0]
            n = len(xy)
            for i in range(n):
                j = (i + 1) % n; acc.add([rim[i], rim[j], M_[j], M_[i]], col, role, sid); acc.add([M_[i], M_[j], cz], col, role, sid)
        bot = lift(xy, 0.0)
        for i in range(len(xy)): j = (i + 1) % len(xy); acc.add([bot[i], bot[j], rim[j], rim[i]], col, role, sid)
        return
    ch = float(k["chamfer"]) if top in ("flat", "plank") else 0.0
    inner = _inset(xy, ch) if ch > 0 else None
    if inner is not None and len(inner) == len(xy) and top != "simple":
        T = lift(inner, h); E = lift(xy, h - ch); acc.add(T, col, role, sid)
        for i in range(len(xy)): j = (i + 1) % len(xy); acc.add([E[i], E[j], T[j], T[i]], col, role, sid)
    else:
        E = lift(xy, h); acc.add(E, col, role, sid)
    bot = lift(xy, 0.0)
    for i in range(len(xy)): j = (i + 1) % len(xy); acc.add([bot[i], bot[j], E[j], E[i]], col, role, sid)


# ── bedding and far view: a chunk mesh with a pattern in the shader ─────────────────────────────────────────────
_FAR = {}
JOINT_DARK = 0.4                                    # far-view joint is in the shadow of the stones (light bedding far away would be a flat light slab)


def far_material(plan, k):
    """Pattern of a kind in the shader by UV in meters: cells — Voronoi (the joint is the distance to the edge), rows,
    herringbone, basketweave, half-logs — a brick texture (stone × row), grid — a checker with a joint; stone tones vary
    randomly, the joint is bedding."""
    key = k["name"]
    if key in _FAR: return _FAR[key]
    m = bpy.data.materials.new(f"paving_far_{key}"); m.use_nodes = True; nt = m.node_tree; N, Lk = nt.nodes, nt.links
    bs = next(n for n in N if n.type == "BSDF_PRINCIPLED"); bs.inputs["Roughness"].default_value = 0.85
    uv = N.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"
    mp = N.new("ShaderNodeMapping"); mp.inputs["Rotation"].default_value[2] = math.radians(float(k.get("angle", 0))); Lk.new(uv.outputs["UV"], mp.inputs["Vector"])
    base = np.asarray(tone(plan, k["substance"], 1.0)); sand = np.asarray(tone(plan, k["bedding"], 1.0)); sp = float(k["spread"])
    if k["generator"] == "cells":
        vo = N.new("ShaderNodeTexVoronoi"); vo.feature = "DISTANCE_TO_EDGE"; vo.inputs["Scale"].default_value = 1.0 / float(np.mean(k["L"]))
        vc = N.new("ShaderNodeTexVoronoi"); vc.inputs["Scale"].default_value = 1.0 / float(np.mean(k["L"]))
        for v_ in (vo, vc): Lk.new(mp.outputs["Vector"], v_.inputs["Vector"])
        rmp = N.new("ShaderNodeValToRGB"); el = rmp.color_ramp.elements
        jw = min(0.3, max(0.02, 0.5 * float(k["joint"]) / float(np.mean(k["L"]))))       # half a joint as a fraction of the cell (Scale = 1 / L)
        el[0].position = 0.3 * jw; el[0].color = (*sand * JOINT_DARK, 1.0); el[1].position = jw; el[1].color = (*base * 0.78, 1.0)
        e2 = el.new(0.45); e2.color = (*np.clip(base * 1.06, 0, 1), 1.0)            # dome: the middle is lighter than the edge
        Lk.new(vo.outputs["Distance"], rmp.inputs["Fac"])
        tint = N.new("ShaderNodeMath"); tint.operation = "MULTIPLY_ADD"; tint.inputs[1].default_value = 2 * sp; tint.inputs[2].default_value = 1.0 - sp
        Lk.new(vc.outputs["Color"], tint.inputs[0])
        mul = N.new("ShaderNodeVectorMath"); mul.operation = "SCALE"; Lk.new(rmp.outputs["Color"], mul.inputs[0]); Lk.new(tint.outputs[0], mul.inputs["Scale"])
        Lk.new(mul.outputs["Vector"], bs.inputs["Base Color"])
        bump = N.new("ShaderNodeBump"); bump.inputs["Distance"].default_value = float(k["visible"]); Lk.new(vo.outputs["Distance"], bump.inputs["Height"]); Lk.new(bump.outputs["Normal"], bs.inputs["Normal"])
    else:
        br = N.new("ShaderNodeTexBrick"); Lk.new(mp.outputs["Vector"], br.inputs["Vector"])
        if k["generator"] == "grid": bw, rh, off = float(k["L"][0]), float(k["L"][0]), 0.0
        elif k["generator"] in ("herringbone", "basketweave"): bw, rh, off = float(np.mean(k["L"])), float(k["w"] if not isinstance(k["w"], list) else k["w"][0]), 0.5
        elif k["generator"] == "half_logs": bw, rh, off = 50.0, float(np.mean(P.course_w(k))), 0.0
        else: bw, rh, off = float(np.mean(k["L"])), float(np.mean(P.course_w(k))), 0.5 if k["bond"] == "staggered" else 0.37
        if k["direction"] == "across" and k["generator"] in ("rows", "half_logs"): bw, rh = rh, bw     # rows across the path: stones along v
        br.offset = off; br.offset_frequency = 2
        for nm, val in (("Scale", 1.0), ("Mortar Size", max(0.002, float(k["joint"]))), ("Mortar Smooth", 0.2), ("Brick Width", bw + float(k["joint"])), ("Row Height", rh + float(k["joint"]))):
            br.inputs[nm].default_value = val
        ch = k.get("checker")
        c1 = np.asarray(tone(plan, ch[0], 1.0)) if ch else base * (1 - sp); c2 = np.asarray(tone(plan, ch[1], 1.0)) if ch else base * (1 + sp)
        br.inputs["Color1"].default_value = (*np.clip(c1, 0, 1), 1.0); br.inputs["Color2"].default_value = (*np.clip(c2, 0, 1), 1.0)
        br.inputs["Mortar"].default_value = (*sand * JOINT_DARK, 1.0)
        if ch:                                                                   # checker: the color is the cell, the joint is the brick texture
            cb = N.new("ShaderNodeTexChecker"); cb.inputs["Scale"].default_value = 1.0
            Lk.new(mp.outputs["Vector"], cb.inputs["Vector"])
            sc_ = N.new("ShaderNodeVectorMath"); sc_.operation = "SCALE"; sc_.inputs["Scale"].default_value = 0.5 / float(k["L"][0])
            Lk.new(mp.outputs["Vector"], sc_.inputs[0]); Lk.new(sc_.outputs["Vector"], cb.inputs["Vector"])
            cb.inputs["Color1"].default_value = (*c1, 1.0); cb.inputs["Color2"].default_value = (*c2, 1.0)
            mix = N.new("ShaderNodeVectorMath"); mix.operation = "MULTIPLY"
            fac = N.new("ShaderNodeMath"); fac.operation = "SUBTRACT"; fac.inputs[0].default_value = 1.0; Lk.new(br.outputs["Fac"], fac.inputs[1])
            Lk.new(cb.outputs["Color"], mix.inputs[0]); mm = N.new("ShaderNodeCombineXYZ")
            for s_ in ("X", "Y", "Z"): Lk.new(fac.outputs[0], mm.inputs[s_])
            Lk.new(mm.outputs["Vector"], mix.inputs[1]); Lk.new(mix.outputs["Vector"], bs.inputs["Base Color"])
        else:
            Lk.new(br.outputs["Color"], bs.inputs["Base Color"])
        bump = N.new("ShaderNodeBump"); bump.inputs["Distance"].default_value = float(k["visible"]) * 0.5
        inv = N.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0; Lk.new(br.outputs["Fac"], inv.inputs[1])
        Lk.new(inv.outputs[0], bump.inputs["Height"]); Lk.new(bump.outputs["Normal"], bs.inputs["Normal"])
    _FAR[key] = m
    return m


def bed_grid(reg, chunk, zf, h, holes=(), step=0.5):
    """Mesh of an area chunk (u, v) → world, cells inside the area and outside the holes (buildings); → (vertices, faces, uv)."""
    ci, u0, u1, v0, v1 = chunk
    us = np.linspace(u0, u1, max(1, int(math.ceil((u1 - u0) / step))) + 1); vs = np.linspace(v0, v1, max(1, int(math.ceil((v1 - v0) / step))) + 1)
    U, V = np.meshgrid(us, vs, indexing="ij"); xy = reg.xy(U.ravel(), V.ravel()); z = zf(xy) + h
    Vt = np.column_stack([xy, z]); F = []
    cu, cv = np.meshgrid((us[:-1] + us[1:]) / 2, (vs[:-1] + vs[1:]) / 2, indexing="ij")
    ok = np.ones(cu.shape, bool)
    if reg.kind == "area": ok &= reg.inside(np.column_stack([cu.ravel(), cv.ravel()])).reshape(cu.shape)
    if len(holes): ok &= ~in_polys(reg.xy(cu.ravel(), cv.ravel()), holes).reshape(cu.shape)
    for i, j in zip(*np.nonzero(ok)):
        a = i * len(vs) + j; F.append([a, a + len(vs), a + len(vs) + 1, a + 1])
    return Vt, F, np.column_stack([U.ravel(), V.ravel()])


# ── step ────────────────────────────────────────────────────────────────────────────────────────────────────────
def run(sc, plan, coll, objs=None, scene=None, cams=(), fpx=1000.0, log=print):
    """The "paving" step: areas (building floors, courtyard and road from the scene's occupancy layer, areas of the
    description) → stones by the kind of the part (plan: paving.resolve). cams — camera points of the frames (the "camera"
    detail mode; without cameras — all stones). → counts."""
    t0 = time.time(); c = plan["paving"]; kinds = plan["kinds"]; parts = c["parts"]; skip = set(c.get("skip") or [])
    every = c["detail"] == "all" or not cams; Tlo, Thi = c["thresholds"]; Tfar = float(c.get("far", 0.0))
    locs = [np.asarray(p, float) for p in cams]; seed = int(c["seed"]) * 7919
    if objs is None: objs = [o for o in sc.objects if o.type == "MESH" and o.name not in coll.objects]
    G = Ground(sc, over=objs)
    regs = [r for r in floor_regions(objs) + layer_regions(scene, float(c.get("road_width", 4.0))) + desc_regions(c.get("regions")) if r["part"] not in skip]
    st = dict(n_regions=len(regs), n_pieces=0, afar=0, n_stones=0, n_simple=0, tris=0, within_buildings=0)
    for ri, R_ in enumerate(regs):
        reg = R_["region"]; k = kinds.get(R_.get("kind") or parts.get(R_["part"]) or parts.get("areas"))
        if k is None: continue
        role = ROLES.get(k["role"], ROLES["floor"]); zc = R_["z"]; holes = R_.get("holes") or []
        zf = (lambda xy, zc=zc: np.full(len(xy), zc)) if zc is not None else (lambda xy, ref=R_.get("ref"): G.zs(xy, ref))
        acc = Acc(); BV, BF, BUV, bmat = [], [], [], []; size = P.unit_size(k); live = float(k["liveliness"])
        for ch in reg.chunks():
            st["n_pieces"] += 1; full = True
            if not every:
                cu, cv = (ch[1] + ch[2]) / 2, (ch[3] + ch[4]) / 2; p = reg.xy(np.array([cu]), np.array([cv]))[0]
                P3 = np.array([p[0], p[1], zf(np.array([p]))[0]]); L_ = min(locs, key=lambda L_: float(np.linalg.norm(P3 - L_)))
                d = float(np.linalg.norm(P3 - L_)); de = max(d - 2.9, 0.5)              # nearest edge of the chunk (half-diagonal of 4 m)
                sin_e = max(0.05, (L_[2] - P3[2]) / max(d, 1e-6))                  # ground at a grazing angle: the stone is squashed in height
                px = fpx * size / de * math.sqrt(sin_e)
                if px < Tfar:                                                    # far view: bedding with a pattern, no stones
                    V_, F_, UV_ = bed_grid(reg, ch, zf, 0.004, holes); o = len(BV); BV += V_.tolist(); BUV += UV_.tolist()
                    BF += [[o + i for i in f] for f in F_]; bmat += [1] * len(F_); st["afar"] += 1; continue
                full = px >= Tlo
            for u in P.units(reg, k, ch, seed, ri):
                if holes:                                                        # a stone inside a house, keep or tower is not laid
                    q = np.asarray(u["poly"], float); cu = q.mean(0)
                    if in_polys(reg.xy(np.array([cu[0]]), np.array([cu[1]])), holes)[0]: st["within_buildings"] += 1; continue
                col = tone(plan, u.get("sub") or k["substance"], u["tint"])
                stone(acc, reg, u, k, zf, col, role, full, live); st["n_stones"] += 1; st["n_simple"] += 0 if full else 1
            V_, F_, UV_ = bed_grid(reg, ch, zf, 0.002, holes); o = len(BV); BV += V_.tolist(); BUV += UV_.tolist()
            BF += [[o + i for i in f] for f in F_]; bmat += [0] * len(F_)
        info = json.dumps(dict(kind=k["name"], part=R_["part"], region=R_["name"]), ensure_ascii=False)
        if acc.f:
            ob = acc.obj(f"pave_{ri}", coll); ob["sk_paving"] = info; st["tris"] += sum(len(p.vertices) - 2 for p in ob.data.polygons)
        if BF:
            me = bpy.data.meshes.new(f"pave_bed_{ri}"); me.from_pydata(BV, [], BF); me.update()
            loops = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", loops)
            me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.asarray(BUV, np.float32)[loops].ravel())
            me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", np.where(np.asarray(bmat) == 1, role, ROLES["mortar"]).astype(np.int32))
            me.attributes.new("sk_stone", "INT", "FACE").data.foreach_set("value", np.zeros(len(BF), np.int32))
            sand = bpy.data.materials.get("paving_bedding") or bpy.data.materials.new("paving_bedding")
            sand.use_nodes = True; next(n for n in sand.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = (*tone(plan, k["bedding"], 1.0), 1.0)
            me.materials.append(sand); me.materials.append(far_material(plan, k)); me.polygons.foreach_set("material_index", np.asarray(bmat, np.int32))
            bo = bpy.data.objects.new(f"pave_bed_{ri}", me); coll.objects.link(bo); bo["sk_paving"] = info
            st["tris"] += 2 * len(BF)
    st["seconds"] = round(time.time() - t0, 1)
    log("paving:", json.dumps(st, ensure_ascii=False))
    return st
