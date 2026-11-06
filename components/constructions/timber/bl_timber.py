"""The "timber" step in Blender — a part of the Constructions component.

Walls — without building code: the scene's occupancy layer (core.layers: building — castle buildings with a base, a
floor, an eaves line, storeys, a fachwerk mark; house — settlement houses, whose kind → walls by the "houses" knowledge
of timber.yaml: an izba is a log house) → building walls (Panel); openings are the dark faces of windows and doors (the
opening color) at the wall plane;
fences — lines of the description (constructions.yaml → timber: fences). The layout is timber.py; geometry: a beam is a
box with an offset (the fachwerk infill sits 7 cm in front of the mass: the castle's flat strips are under it), a log is
a cylinder with end faces (for round-notch joints — an overhang at the corner), a palisade is a log with a point rising
from the ground, wattle is stakes and rods interwoven. Camera-dependent view: px per element < Tfar — fachwerk is not
built (the building's flat strips show), a log house is one strip per course.
Output contract: objects timber_<n> (building walls), fence_<n> (fences) in the step's collection; on faces sk_role
(wood; infill — finish), sk_stone — the element number (8·10⁸ + …), tone — Col; property sk_timber (kind, building,
walls).

  import bl_timber as BT; stats = BT.run(sc, plan, coll, objs=…, scene=<scene folder>, cams=…, fpx=…)
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
from components.constructions.timber import timber as T  # noqa: E402
from components.constructions.masonry.bl_stones import ROLES, Acc  # noqa: E402

D_INF = 0.07                                                                     # the infill goes in front of the building's fachwerk strips (6 cm)
PROUD = 0.035                                                                    # the beam projects from the infill


def tone(plan, key, tint):
    v = (plan.get("substances") or {}).get(key) or [0.3, 0.22, 0.15]
    return tuple(float(x) for x in np.clip(np.asarray(v, float) * tint, 0, 1))


# ── building walls: occupancy layer, openings from the dark faces ───────────────────────────────────────────────
def hole_faces(objs, dark=0.1):
    """Dark faces (windows, doors): [(vertices k×3)]."""
    out = []
    for ob in objs:
        if ob.type != "MESH": continue
        me = ob.data; ca = me.color_attributes.get("Col")
        if ca is None or ca.domain != "CORNER" or not len(me.polygons): continue
        Mw = np.asarray(ob.matrix_world, float)
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
        c = np.empty(len(me.loops) * 4, np.float32); ca.data.foreach_get("color", c); c = c.reshape(-1, 4)[:, :3]
        ls = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
        fc = np.add.reduceat(c, ls) / lt[:, None]
        for i in np.flatnonzero(fc.max(1) < dark): out.append(co[lv[ls[i]:ls[i] + lt[i]]])
    return out


def openings_of(pn, holes, tol=0.15):
    """Wall openings: dark faces at its plane → frames (u0, u1, z0, z1), merged."""
    out = []
    for H in holes:
        rel = H[:, :2] - pn.p0; dd = rel @ pn.n; u = rel @ pn.t
        if np.abs(dd).max() > tol or u.max() < 0.05 or u.min() > pn.L - 0.05: continue
        z0, z1 = float(H[:, 2].min()), float(H[:, 2].max())
        if z1 < pn.z0 + 0.1 or z0 > pn.z1 - 0.1 or float(np.ptp(u)) < 0.2 or z1 - z0 < 0.3: continue
        out.append((max(0.0, float(u.min())), min(pn.L, float(u.max())), max(pn.z0, z0), min(pn.z1, z1)))
    out.sort(); merged = []
    for o in out:
        if merged and o[0] < merged[-1][1] + 0.05 and not (o[3] < merged[-1][2] or o[2] > merged[-1][3]):
            m = merged[-1]; merged[-1] = (m[0], max(m[1], o[1]), min(m[2], o[2]), max(m[3], o[3]))
        else: merged.append(o)
    return merged


def layer_panels(scene, holes, parts, houses=None):
    """Building walls of the scene occupancy layer: castle buildings (building) — fachwerk from the marked storey up to the
    eaves (the part kind "fachwerk") or "buildings" — all over the full height (log house, cladding); settlement houses
    (house) — by the house kind (houses: izba → log house) over the full height. → [(building, kind, [Panel])]."""
    out = []
    if not scene: return out
    from core import layers as LY
    def walls(b, name, kind, full):
        P = np.asarray(b["poly"], float)[:, :2]; cc = P.mean(0); fl = float(b.get("floor", b.get("base", 0.0))); eave = float(b["eave"])
        ns = max(1, int(b.get("storeys", 1))); sh = (eave - fl) / ns
        z0 = float(b.get("base", fl)) if full else fl + sh * int(b.get("fachwerk") or 0)
        if eave - z0 < 1.5: return
        panels = []
        for i in range(len(P)):
            p0, p1 = P[i], P[(i + 1) % len(P)]; t = p1 - p0; n = np.array([t[1], -t[0]])
            if (0.5 * (p0 + p1) - cc) @ n < 0: n = -n
            pn = T.Panel(p0, p1, z0, eave, n=n, storeys=[fl + sh * s for s in range(ns)], name=f"{name}.{i}")
            pn.openings = openings_of(pn, holes); panels.append(pn)
        out.append((name, kind, panels))
    for bi, b in enumerate(LY.occupancy(scene, ("building",))):
        kind = parts.get("buildings") or (parts.get("fachwerk") if b.get("fachwerk") else None)
        if kind and len(b.get("poly") or []) >= 3 and b.get("eave") is not None:
            walls(b, f"{b.get('type', 'building')} {bi}", kind, bool(parts.get("buildings")))
    for hi, h in enumerate(LY.occupancy(scene, ("house",))):
        kind = (houses or {}).get(h.get("type"))
        if kind and len(h.get("poly") or []) >= 3 and h.get("eave") is not None:
            walls(h, f"{h.get('type')} {hi}", kind, True)
    return out


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────────
def outward(pn, Q):
    """Winding order of (u, z) so that the face normal points outward (by the wall's n): for a wall where u grows right to left
    from outside, the winding is reversed (otherwise the face looks into the mass and is unlit)."""
    Q = np.asarray(Q, float)
    return Q[::-1] if float(np.cross([*pn.t, 0.0], [0.0, 0.0, 1.0])[:2] @ pn.n) < 0 else Q


def box(acc, pn, poly, d0, d1, col, role, sid):
    """Beam: a quadrilateral (u, z) from offset d0 to d1 — the face and the sides (the back is in the mass)."""
    Q = outward(pn, poly); F = pn.world(Q[:, 0], Q[:, 1], np.full(len(Q), d1)); B = pn.world(Q[:, 0], Q[:, 1], np.full(len(Q), d0))
    acc.add(F, col, role, sid)
    for i in range(len(Q)): j = (i + 1) % len(Q); acc.add([B[i], B[j], F[j], F[i]], col, role, sid)


def log(acc, pn, e, col, role, flat=False, n=10, simple=False):
    """A course log: a cylinder along the wall (the axis is 0.35 r in front of the mass), round end faces; for a hewn log the
    face is cut flat."""
    r = e["r"]; ax = r * 0.35; u0, u1 = e["u0"], e["u1"]; zc = e["zc"]
    if simple:
        box(acc, pn, T.rect(u0, u1, zc - r * 0.9, zc + r * 0.9), 0.0, ax + r * 0.9, col, role, e["sid"]); return
    th = np.linspace(0, 2 * math.pi, n, endpoint=False); dz, dd = r * np.sin(th), r * np.cos(th)
    if flat: dd = np.minimum(dd, r * 0.82)
    A = pn.world(np.full(n, u0), zc + dz, ax + dd); B = pn.world(np.full(n, u1), zc + dz, ax + dd)
    for i in range(n): j = (i + 1) % n; acc.add([A[i], A[j], B[j], B[i]], col, role, e["sid"])
    acc.add(A[::-1], tuple(c * 0.8 for c in col), role, e["sid"]); acc.add(B, tuple(c * 0.8 for c in col), role, e["sid"])


class Ground:
    def __init__(self, sc):
        dg = bpy.context.evaluated_depsgraph_get(); self.trees = []
        for ob in sc.objects:
            nm = ob.name.split(".")[0].lower()
            if ob.type == "MESH" and (ob.get("sk_layer") in ("site", "relief") or nm.startswith(("terrain", "ground", "relief"))) and not nm.startswith(("landmark", "water")):
                try: self.trees.append(BVHTree.FromObject(ob, dg))
                except Exception: pass

    def z(self, x, y):
        best = None
        for t in self.trees:
            hit = t.ray_cast(Vector((float(x), float(y), 5000.0)), Vector((0.0, 0.0, -1.0)))
            if hit[0] is not None and (best is None or hit[0].z > best): best = hit[0].z
        return float(best) if best is not None else 0.0


def post(acc, xy, z0, h, r, tip, col, role, sid, n=8):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False); ring = lambda z: [(xy[0] + r * math.cos(a), xy[1] + r * math.sin(a), z) for a in th]
    B, Tt = ring(z0 - 0.3), ring(z0 + h - tip); top = (xy[0], xy[1], z0 + h)
    for i in range(n):
        j = (i + 1) % n; acc.add([B[i], B[j], Tt[j], Tt[i]], col, role, sid); acc.add([Tt[i], Tt[j], top], col, role, sid)


def tube(acc, pts, r, col, role, sid, n=4):
    P = np.asarray(pts, float)
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-6: continue
        d /= L; e1 = np.cross(d, [0, 0, 1.0]); e1 = e1 / max(np.linalg.norm(e1), 1e-9); e2 = np.cross(d, e1)
        ring = [np.cos(t) * e1 * r + np.sin(t) * e2 * r for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        for i in range(n): j = (i + 1) % n; acc.add([a + ring[i], a + ring[j], b + ring[j], b + ring[i]], col, role, sid)


# ── step ────────────────────────────────────────────────────────────────────────────────────────────────────────
def run(sc, plan, coll, objs=None, scene=None, cams=(), fpx=1000.0, panels=None, log_=print):
    """The "timber" step: building walls (the scene occupancy layer scene or panels — [(name, kind, [Panel])]) and the
    description's fences → beams, infill, logs, palisade, wattle. → counts."""
    t0 = time.time(); c = plan["timber"]; kinds = plan["kinds"]; parts = c["parts"]; seed = int(c["seed"]) * 7919
    every = c["detail"] == "all" or not cams; Tfar = float(c.get("far", 2.0)); locs = [np.asarray(p, float) for p in cams]
    if objs is None: objs = [o for o in sc.objects if o.type == "MESH" and o.name not in coll.objects]
    groups = list(panels or []) + layer_panels(scene, hole_faces(objs), parts, c.get("houses"))
    st = dict(n_buildings=len(groups), walls=0, n_elements=0, n_logs=0, afar=0, n_fences=0, tris=0)
    for gi, (name, kn, pns) in enumerate(groups):
        k = kinds.get(kn)
        if k is None: continue
        acc = Acc(); role = ROLES.get(k["role"], ROLES["wood"])
        for pi, pn in enumerate(pns):
            st["walls"] += 1; far = False
            if not every:
                cpt = np.array([*(pn.p0 + pn.t * pn.L / 2), (pn.z0 + pn.z1) / 2]); dist = min(float(np.linalg.norm(cpt - L_)) for L_ in locs)
                far = fpx * 0.25 / max(dist - pn.L / 2, 0.5) < Tfar
            if far and k["generator"] == "fachwerk": st["afar"] += 1; continue  # far view: the building's strips
            for e in T.panel_items(pn, k, seed, gi * 50 + pi):
                st["n_elements"] += 1
                if e["t"] == "field":
                    Q = outward(pn, e["poly"]); acc.add(pn.world(Q[:, 0], Q[:, 1], np.full(len(Q), D_INF)), tone(plan, k["infill"], e["tint"]), ROLES["finish"], e["sid"])
                elif e["t"] == "log":
                    log(acc, pn, e, tone(plan, k["substance"], e["tint"]), role, flat=bool(k.get("boards")), simple=far); st["n_logs"] += 1
                else:
                    front = (D_INF + PROUD + (0.01 if e["kind"] in ("threshold", "plate") else 0.0)) if k["generator"] == "fachwerk" else float(e["d"])
                    box(acc, pn, e["poly"], 0.0, front, tone(plan, k["substance"], e["tint"]), role, e["sid"])
        if acc.f:
            ob = acc.obj(f"timber_{gi}", coll); ob["sk_timber"] = json.dumps(dict(kind=kn, building=name, walls=len(pns)), ensure_ascii=False)
            st["tris"] += sum(len(p.vertices) - 2 for p in ob.data.polygons)
    G = Ground(sc); skipped = []
    for fi, f in enumerate(c.get("fences") or []):                               # fences along a line; the kind is its own or the fences knob (resolve)
        k = kinds.get(f.get("kind") or parts.get("fences", "palisade"))
        if k is None or len(f.get("path") or []) < 2 or k["generator"] not in T.FENCE_GEN: skipped.append(fi); continue
        acc = Acc(); role = ROLES.get(k["role"], ROLES["wood"])
        if k["generator"] == "palisade":
            for p_ in T.palisade(f["path"], k, seed, fi):
                post(acc, p_["xy"], G.z(*p_["xy"]), p_["h"], p_["r"], p_["tip"], tone(plan, k["substance"], p_["tint"]), role, p_["sid"]); st["n_elements"] += 1
        elif k["generator"] == "wattle":
            stakes, rods = T.wattle(f["path"], k, seed, fi)
            for s_ in stakes: post(acc, s_["xy"], G.z(*s_["xy"]), s_["h"], s_["r"], 0.06, tone(plan, k["substance"], s_["tint"]), role, s_["sid"], n=6)
            for r_ in rods:
                P3 = np.asarray(r_["pts3"], float); mids = 0.5 * (P3[:-1] + P3[1:]); mids[:, :2] = 0.5 * (P3[:-1, :2] + P3[1:, :2])
                pts = [P3[0]] + [q for a, b in zip(mids, P3[1:]) for q in (a, b)]
                pts = [np.array([q[0], q[1], q[2] + G.z(q[0], q[1])]) for q in pts]
                tube(acc, pts, r_["r"], tone(plan, k["substance"], r_["tint"] * 0.85), role, r_["sid"])
            st["n_elements"] += len(stakes) + len(rods)
        if acc.f:
            ob = acc.obj(f"fence_{fi}", coll); ob["sk_timber"] = json.dumps(dict(kind=k["name"], fence=fi), ensure_ascii=False)
            st["tris"] += sum(len(p.vertices) - 2 for p in ob.data.polygons); st["n_fences"] += 1
    if skipped: st["_skipped"] = f"fences {skipped}: no fence kind or path (≥ 2 points)"
    st["seconds"] = round(time.time() - t0, 1)
    log_("wood:", json.dumps(st, ensure_ascii=False))
    return st
