"""The "house" type — a mass in Blender: house programs (house_plan.program) → geometry (house_plan.geometry, numpy) →
a mesh built with the base's accumulator (bl_arch.Acc): walls with real openings (wall_panel — no booleans), opening
fillers set deep in the reveal (glass, door, dark shop), a roof of faces with thickness and overhang, a plinth, a storey
jetty and beam ends, half-timbering (posts at the openings, rails along the storeys, braces at the corners), balconies
and galleries with railings, pipes, chimneys. Houses go into objects in batches; the face roles sk_role (wall, finish,
roof, wood, metal, glass, floor, opening) — for masonry, roofing and materials; the house number sk_house and, if the
program carries influences, sk_P, sk_T, sk_W (FLOAT: power, trade, wealth 0–1) and sk_district (INT: district number in
field.LABELS, −1 — none) — for the neighbors (materials, layout): poorer — dirtier, more passable — shops.
build_houses(programs, collection) — in an ensemble; build_house(folder, sc) — one house (house.json) step by step;
CLI — a view of the house."""
import os
import sys
from components.buildings import arch_parts
import json
import math
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils import geometry as MG

from components.buildings.base import bl_arch as BA
from components.buildings.houses import house_plan as HP
from components.buildings.influences.field import LABELS


def C(rgb, role): return BA.rc(tuple(rgb), role)


def _w3(p0, d, nrm, s, z, off):
    q = p0 + d * s + nrm * off; return (float(q[0]), float(q[1]), float(z))


def wall(a, W, cols):
    """Wall of a program: a panel with openings, fillers deep in the reveal, half-timbering on its own storeys."""
    p0 = np.asarray(W["p0"], float); p1 = np.asarray(W["p1"], float); L = float(np.linalg.norm(p1 - p0))
    if L < 0.2: return 0
    d = (p1 - p0) / L; nrm = np.array([d[1], -d[0]]); t = W["t"]
    top = [(float(s), float(z)) for s, z in W["top"]]
    col = C(cols["wall_top"] if W.get("fach") and "wall_top" in cols else W["col"], W["role"])
    holes = [(h[0], h[1], h[2], h[3], h[4]) for h in W["holes"]]
    BA.wall_panel(a, p0, p1, W["z0"], top[0][1], t, holes, col, col_jamb=C(cols.get("finish", W["col"]), "finish"), top=top)
    ts = np.array([q[0] for q in top]); tz = np.array([q[1] for q in top])
    for h in W["holes"]:                                                         # filler — deep in the reveal
        s, z0, w, hh, kind, role = h
        if not (0.05 < s - w / 2 and s + w / 2 < L - 0.05 and z0 + hh < np.interp(s, ts, tz) - 0.05): continue
        pts = BA.hole_outline(s, z0, w, hh, kind)
        cc = {"glass": C(cols.get("glass", (0.10, 0.12, 0.14)), "glass"), "wood": C(cols.get("wood", (0.3, 0.2, 0.12)), "wood"),
              "opening": C((0.05, 0.05, 0.06), "opening")}[role]
        a.face([_w3(p0, d, nrm, x, z, -t * 0.25) for x, z in pts], cc)
        if role == "glass" and w > 0.6:                                           # glazing bars — a mullion and a transom
            a.box((p0 + d * s - nrm * (t * 0.25 - 0.03)).tolist(), d.tolist(), 0.05, 0.06, z0, z0 + hh * 0.97, C(cols.get("wood", (0.3, 0.2, 0.12)), "wood"))
            a.box((p0 + d * s - nrm * (t * 0.25 - 0.03)).tolist(), d.tolist(), 0.05, w, z0 + hh * 0.62, z0 + hh * 0.62 + 0.06, C(cols.get("wood", (0.3, 0.2, 0.12)), "wood"))
        if role == "glass" and z0 > W["z0"] + 0.6:                               # windowsill
            a.box((p0 + d * s + nrm * (t / 2 + 0.05)).tolist(), d.tolist(), 0.16, w + 0.16, z0 - 0.1, z0, C(cols.get("finish", W["col"]), "finish"))
    if W.get("fach"): fachwerk(a, W, p0, d, nrm, L, top, cols)
    return 1


def fachwerk(a, W, p0, d, nrm, L, top, cols):
    """Half-timbering of a storey: posts at the corners and at the openings, rails along the bottom of the storey and the
    windowsills, braces at the corners."""
    wood = C(cols.get("wood", (0.26, 0.17, 0.1)), "wood"); ts = np.array([q[0] for q in top]); tz = np.array([q[1] for q in top])
    out = W["t"] / 2 + 0.035                                                        # from the wall axis — to the face and slightly outward
    for f in W["fach"]:
        z0 = W["zf"][f]; z1 = W["zf"][f + 1] if f + 1 < len(W["zf"]) else float(min(tz))
        if z1 - z0 < 1.0: continue
        posts = {0.12, L - 0.12}
        for h in W["holes"]:
            if h[1] >= z0 - 0.01 and h[1] < z1: posts |= {h[0] - h[2] / 2 - 0.1, h[0] + h[2] / 2 + 0.1}
        P = sorted(posts)
        for i in range(len(P) - 1):                                              # intermediate posts in wide piers
            if P[i + 1] - P[i] > 1.9: posts.add((P[i] + P[i + 1]) / 2)
        for s in sorted(posts):
            if not (0.05 < s < L - 0.05): continue
            a.box(_w3(p0, d, nrm, s, 0, out)[:2], d.tolist(), 0.07, 0.18, z0, min(z1, float(np.interp(s, ts, tz)) - 0.05), wood)
        for zz in (z0, z0 + 0.88):
            a.box((p0 + d * L / 2 + nrm * out).tolist(), d.tolist(), 0.07, L - 0.05, zz, zz + 0.16, wood)
        for s0, s1 in ((0.12, min(1.4, L / 3)), (L - 0.12, max(L - 1.4, 2 * L / 3))):
            BA.bar3(a, _w3(p0, d, nrm, s0, z0 + 0.16, out + 0.01), _w3(p0, d, nrm, s1, min(z1, z0 + 2.2) - 0.1, out + 0.01), 0.12, wood)


def roof_face(a, F, col):
    """A roof slope with thickness: the top, the bottom (reversed), the ends."""
    P = [tuple(map(float, p)) for p in F["pts"]]; b = F.get("band", 0.16)
    if len(P) < 3: return
    a.face(P, col); a.face([(x, y, z - b) for x, y, z in P][::-1], col)
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]; a.face([q, p, (p[0], p[1], p[2] - b), (q[0], q[1], q[2] - b)], col)


def flat_face(a, F, col):
    """A flat face with holes (a terrace roof, a yard floor)."""
    loops = [[Vector((float(x), float(y), 0.0)) for x, y in lp] for lp in F["loops"]]
    pts = [(float(x), float(y), float(F["z"])) for lp in F["loops"] for x, y in lp]
    tris = MG.tessellate_polygon(loops)
    if tris: a.tri(pts, [(i, j, k) for i, j, k in tris], col)


def box_item(a, Bx, cols):
    col = C(Bx["col"], Bx["role"])
    if "seg" in Bx:                                                              # plinth — a strip in front of the wall
        p, q = (np.asarray(v, float) for v in Bx["seg"]); n = np.asarray(Bx["nout"], float)
        A = [(*(p + n * Bx["d"]), Bx["z0"]), (*(q + n * Bx["d"]), Bx["z0"])]; Bt = [(*(p + n * Bx["d"]), Bx["z1"]), (*(q + n * Bx["d"]), Bx["z1"])]
        a.face([A[0], A[1], Bt[1], Bt[0]], col)
        a.face([Bt[0], Bt[1], (*(q), Bx["z1"]), (*(p), Bx["z1"])], col)
    elif "bar" in Bx: BA.bar3(a, Bx["bar"][0], Bx["bar"][1], Bx["w"], col)
    elif "box" in Bx: a.box(*Bx["box"][:5], Bx["box"][5], col)
    elif "slab" in Bx:                                                           # balcony, gallery: a slab, brackets, railings
        p, q = (np.asarray(v, float) for v in Bx["slab"]); n = np.asarray(Bx["nout"], float); L = float(np.linalg.norm(q - p))
        if L < 1.5: return
        d = (q - p) / L; m0 = p + d * 0.4; m1 = q - d * 0.4; L2 = L - 0.8; c = (m0 + m1) / 2 + n * Bx["d"] / 2
        a.box(c.tolist(), d.tolist(), Bx["d"], L2, Bx["z"] - Bx["th"], Bx["z"], col)
        k = max(2, int(L2 / 1.25) + 1)
        for s in np.linspace(0, L2, k):
            e = m0 + d * s + n * (Bx["d"] - 0.06)
            a.box(e.tolist(), d.tolist(), 0.06, 0.06, Bx["z"], Bx["z"] + Bx["rail"], col)
            BA.bar3(a, (*(m0 + d * s + n * 0.02), Bx["z"] - Bx["th"] - 0.5), (*(m0 + d * s + n * (Bx["d"] * 0.8)), Bx["z"] - Bx["th"]), 0.1, col)
        a.box((m0 + d * L2 / 2 + n * (Bx["d"] - 0.06)).tolist(), d.tolist(), 0.08, L2, Bx["z"] + Bx["rail"] - 0.08, Bx["z"] + Bx["rail"], col)


def house(a, prog):
    """One house into the accumulator. → the number of walls."""
    G = HP.geometry(prog); cols = dict(prog["col"]); n = 0
    for W in G["walls"]: n += wall(a, W, cols)
    for F in G["faces"]:
        col = C(F["col"], F["role"])
        if "loops" in F: flat_face(a, F, col)
        else: roof_face(a, F, col)
    for Bx in G["boxes"]: box_item(a, Bx, cols)
    for Tb in G["tubes"]: BA.tube(a, Tb["pts"], Tb["r"], C(Tb["col"], Tb["role"]), n=8)
    for K in G.get("columns", []): BA.column(a, K["c"], K["r"], K["z0"], K["z1"], C(cols.get("finish", cols["wall"]), "finish"))
    return n


def tag_influence(ob, progs):
    """House faces → the three influence numbers and the district of its program (sk_P, sk_T, sk_W, sk_district), by the house
    number sk_house."""
    me = ob.data; n = len(me.polygons)
    if not n or "sk_house" not in me.attributes or not any(p.get("influence") for p in progs): return
    hid = np.empty(n, np.int32); me.attributes["sk_house"].data.foreach_get("value", hid)
    m = max(int(hid.max()), max(int(p.get("id", -1)) for p in progs)) + 2; tab = np.zeros((m, 3), np.float32); dis = np.full(m, -1, np.int32)
    for p in progs:
        if p.get("influence") and p.get("id") is not None: tab[int(p["id"])] = p["influence"]; dis[int(p["id"])] = LABELS.index(p["district"]) if p.get("district") in LABELS else -1
    k = np.where(hid >= 0, hid, m - 1)
    for j, nm in enumerate(("sk_P", "sk_T", "sk_W")): me.attributes.new(nm, "FLOAT", "FACE").data.foreach_set("value", tab[k, j])
    me.attributes.new("sk_district", "INT", "FACE").data.foreach_set("value", dis[k])


def build_houses(progs, coll, name="houses", batch=60, groups=None):
    """Programs → objects in batches (batch houses each; groups — own groups, e.g. per tier). The house number on the faces —
    sk_house (the program id; if absent — its index in the list), influences — tag_influence. → objects."""
    obs = []; t0 = time.time()
    for i, p in enumerate(progs):
        if p.get("id") is None: p["id"] = i
    groups = groups or [progs[i:i + batch] for i in range(0, len(progs), batch)]
    for gi, grp in enumerate(groups):
        if not grp: continue
        a = BA.Acc("wall"); a.item_attr = "sk_house"
        for p in grp:
            a.item = int(p["id"]) if p.get("id") is not None else -1
            try: house(a, p)
            except Exception as e: print("house skipped:", p.get("id"), p.get("kind"), repr(e)[:160])
        if a.f:
            ob = a.obj(f"{name}.{gi:03d}", coll); ob["sk_layer"] = "buildings"; ob["sk_houses"] = len(grp); tag_influence(ob, grp); obs.append(ob)
    print(f"houses: {len(progs)} in {len(obs)} objects, {time.time() - t0:.1f} s")
    return obs


def build_house(folder, sc=None, tag=True, **kw):
    """A house (house.json) over an open scene — step by step. → (objects, plan)."""
    R = json.load(open(os.path.join(folder, "house.json")))
    BA.SCENE[0] = sc
    obs = build_houses(R["houses"], BA.coll("house"), name="house")
    BA.SCENE[0] = None
    return obs, R


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]; folder = argv[0]; KW = dict(x.split("=", 1) for x in argv[1:] if "=" in x)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    R = json.load(open(os.path.join(folder, "house.json")))
    obs = build_houses(R["houses"], BA.coll("house"), name="house")
    c = np.mean(np.asarray(R["houses"][0]["F"], float), 0); z = R["houses"][0]["z0"]
    BA.ground(BA.Relief(os.path.join(folder, "relief.npz")), c, 60, 1.0)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); bpy.context.scene.collection.objects.link(cam); bpy.context.scene.camera = cam
    cam.location = (c[0] + 18, c[1] - 26, z + 14); d = Vector((c[0], c[1], z + 4)) - cam.location; cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; res = tuple(int(v) for v in KW.get("res", "1280x800").split("x"))
    sc.render.resolution_x, sc.render.resolution_y = res; shd = sc.display.shading; shd.light = "STUDIO"; shd.color_type = "VERTEX"
    shd.show_cavity = True; shd.show_shadows = True; sc.display.light_direction = (0.45, -0.35, 0.82)
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    sc.render.filepath = os.path.join(folder, "view_aerial.png"); bpy.ops.render.render(write_still=True)
    if KW.get("blend") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "house.blend"))
    print("__RESULT__", json.dumps({"n_houses": len(R["houses"]), "n_objects": len(obs), "kinds": ["aerial"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
