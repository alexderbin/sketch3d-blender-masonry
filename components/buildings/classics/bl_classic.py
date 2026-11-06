"""Type "classics" — the mass in Blender: programs (classic_plan.program) → primitives (classic_plan.geometry,
numpy) → a mesh via the base accumulator (bl_arch.Acc): crepidoma and podium as prisms, columns by order
(Doric — 20 flutes, echinus and abacus; Ionic — base, cushion with volutes; Corinthian — the capital bell;
Tuscan), walls with real openings (a stub is dark, deep in the reveal), entablature with triglyphs, a roof
of faces with thickness, pediments with a raking cornice and acroteria, the cone of a tholos, theatre rows.
Face roles sk_role: finish (order stone), wall, roof, floor, opening.
build(programs, collection) — in an ensemble; build_classic(folder, sc) — step by step; CLI — showcase views:
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/classics/bl_classic.py -- <plan folder> [res=1600x1000] [blend=1]"""
import os
import sys
from components.buildings import arch_parts
import json
import math
import time

import bpy
import numpy as np
from mathutils import Vector

from components.buildings.base import bl_arch as BA
from components.buildings.classics import classic_plan as CP


def C(rgb, role): return BA.rc(tuple(rgb), role)


def column(a, K):
    """Order column: a tapering shaft (Doric — 20 faces, like flutes), the capital and base by order."""
    c = np.asarray(K["c"], float); r = K["r"]; z0, z1 = K["z0"], K["z1"]; h = z1 - z0; o = K["order"]; col = C(K["col"], "finish")
    ax = np.asarray(K["ax"], float); ax = ax / max(np.linalg.norm(ax), 1e-9); nx = np.array([-ax[1], ax[0]])
    if o == "doric" or o == "tuscan":
        hb = 0.0 if o == "doric" else 0.45 * r; hc = min(0.84 * r, 0.14 * h); hs = h - hb - hc; n = 20 if o == "doric" else 16
        if hb: BA.lathe(a, c, [(1.35 * r, 0.0), (1.35 * r, 0.4 * hb), (r, hb)], z0, col, n=n)
        BA.lathe(a, c, [(r, hb), (r * 0.985, hb + 0.35 * hs), (r * 0.8, hb + hs)], z0, col, n=n)
        BA.lathe(a, c, [(r * 0.8, hb + hs), (r * 0.84, hb + hs + 0.1 * hc), (r * 1.22, hb + hs + 0.62 * hc)], z0, col, n=n)
        a.box(c, ax, 2.5 * r, 2.5 * r, z0 + hb + hs + 0.62 * hc, z1, col); return
    hb = 0.5 * r; hc = (2.2 * r if o == "corinthian" else 0.84 * r); hc = min(hc, 0.2 * h); hs = h - hb - hc
    BA.lathe(a, c, [(1.45 * r, 0.0), (1.45 * r, 0.2 * hb), (1.18 * r, 0.5 * hb), (1.3 * r, 0.72 * hb), (r, hb)], z0, col, n=16)
    BA.lathe(a, c, [(r, hb), (r * 0.97, hb + 0.4 * hs), (r * 0.84, hb + hs)], z0, col, n=16)
    zc = z0 + hb + hs
    if o == "corinthian":
        BA.lathe(a, c, [(0.84 * r, 0.0), (0.95 * r, 0.3 * hc), (1.2 * r, 0.72 * hc), (1.42 * r, 0.9 * hc)], zc, col, n=16)
        a.box(c, ax, 2.9 * r, 2.9 * r, zc + 0.88 * hc, z1, col); return
    BA.lathe(a, c, [(0.84 * r, 0.0), (1.02 * r, 0.35 * hc)], zc, col, n=16)          # Ionic: echinus, cushion, volutes, abacus
    a.box(c, ax, 1.5 * r, 2.7 * r, zc + 0.3 * hc, zc + 0.78 * hc, col)
    for s in (-1, 1):
        p = c + ax * s * 1.25 * r; zv = zc + 0.42 * hc
        BA.tube(a, [(*(p - nx * 0.78 * r), zv), (*(p + nx * 0.78 * r), zv)], 0.4 * r, col, n=10)
    a.box(c, ax, 2.0 * r, 2.5 * r, zc + 0.78 * hc, z1, col)


def wall(a, W):
    """A wall with through openings; an opening stub is a dark face deep in the reveal."""
    p0 = np.asarray(W["p0"], float); p1 = np.asarray(W["p1"], float); L = float(np.linalg.norm(p1 - p0))
    if L < 0.2: return
    d = (p1 - p0) / L; nrm = np.array([d[1], -d[0]]); t = W["t"]; top = [(float(s), float(z)) for s, z in W["top"]]
    col = C(W["col"], W["role"])
    BA.wall_panel(a, p0, p1, W["z0"], top[0][1], t, [tuple(h[:5]) for h in W["holes"]], col, top=top)
    ts = np.array([q[0] for q in top]); tz = np.array([q[1] for q in top]); dark = C((0.05, 0.05, 0.06), "opening")
    for h in W["holes"]:
        s, z0, w, hh = h[:4]
        if not (0.05 < s - w / 2 and s + w / 2 < L - 0.05 and z0 + hh < np.interp(s, ts, tz) - 0.05): continue
        a.face([(*(p0 + d * x - nrm * t * 0.3), z) for x, z in BA.hole_outline(s, z0, w, hh, h[4])], dark)


def face(a, F):
    P = [tuple(map(float, p)) for p in F["pts"]]; col = C(F["col"], F["role"]); b = F.get("band", 0.0)
    if len(P) < 3: return
    a.face(P, col)
    if b > 0:
        a.face([(x, y, z - b) for x, y, z in P][::-1], col)
        for i in range(len(P)):
            p, q = P[i], P[(i + 1) % len(P)]; a.face([q, p, (p[0], p[1], p[2] - b), (q[0], q[1], q[2] - b)], col)


def building(a, prog):
    G = CP.geometry(prog)
    for p in G["prisms"]: a.prism(p["poly"], p["z0"], p["z1"], C(p["col"], p["role"]), bottom=p.get("bot", False))
    for K in G["columns"]: column(a, K)
    for W in G["walls"]: wall(a, W)
    for F in G["faces"]: face(a, F)
    for b in G["bars"]: BA.bar3(a, b["p"], b["q"], b["w"], C(b["col"], b["role"]))
    for b in G["boxes"]: a.box(b["c"], b["d"], b["w"], b["l"], b["z0"], b["z1"], C(b["col"], b["role"]))
    for k in G["cones"]: BA.roof_cone(a, k["ring"], k["z"], k["h"], C(k["col"], "roof"), ov=1.0)
    return len(G["columns"])


def build(progs, coll, name="classics"):
    """Programs → one object per building (the name is the kind). → objects."""
    obs = []; t0 = time.time()
    for i, p in enumerate(progs):
        a = BA.Acc("finish")
        try: building(a, p)
        except Exception as e: print("building skipped:", p.get("kind"), repr(e)[:200]); continue
        if a.f:
            ob = a.obj(f"{name}.{p['kind']}.{i:02d}", coll); ob["sk_layer"] = "buildings"; ob["sk_kind"] = p["kind"]; obs.append(ob)
    print(f"classics: {len(obs)} buildings, {time.time() - t0:.1f} s")
    return obs


def build_classic(folder, sc=None, tag=True, **kw):
    """The plan (classic.json) over an open scene — step by step. → (objects, plan)."""
    R = json.load(open(os.path.join(folder, "classic.json"))); BA.SCENE[0] = sc
    obs = build(R["buildings"], BA.coll("classics"))
    BA.SCENE[0] = None
    return obs, R


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]; folder = os.path.abspath(argv[0]); KW = dict(x.split("=", 1) for x in argv[1:] if "=" in x)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obs, R = build_classic(folder)
    rel = BA.Relief(os.path.join(folder, "relief.npz"))
    P = np.array([b["c"] for b in R["buildings"]], float); c = P.mean(0); half = float(np.max(np.abs(P - c))) + 40
    BA.ground(rel, c, half, 1.0)
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; res = tuple(int(v) for v in KW.get("res", "1600x1000").split("x"))
    sc.render.resolution_x, sc.render.resolution_y = res; sc.view_settings.view_transform = "Standard"; shd = sc.display.shading
    shd.light = "STUDIO"; shd.color_type = "VERTEX"; shd.show_cavity = True; shd.cavity_type = "WORLD"; shd.show_shadows = True; shd.shadow_intensity = 0.55
    sc.display.light_direction = (0.45, -0.35, 0.82); sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    z = float(np.median([b["z0"] for b in R["buildings"]])); out = []
    views = {"aerial": ((c[0] - 40, c[1] - half * 1.25, z + half * 0.75), (c[0], c[1] - 10, z), 30)}
    for i, b in enumerate(R["buildings"]):                                         # each building — a three-quarter view
        q = np.asarray(b["c"], float); s = max(b.get("L", 20), b.get("W", 20))
        views[f"b{i:02d}_{b['kind']}"] = ((q[0] + 0.75 * s, q[1] - 1.25 * s, z + 0.45 * s), (q[0], q[1], z + 0.12 * s), 32)
    for k, (loc, tgt, lens) in views.items():
        if KW.get("views") and k not in KW["views"].split(",") and not any(k.endswith(x) for x in KW["views"].split(",")): continue
        cam.location = loc; cam.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat("-Z", "Y").to_euler(); cam.data.lens = lens; cam.data.clip_end = 3000
        sc.render.filepath = os.path.join(folder, f"view_{k}.png"); bpy.ops.render.render(write_still=True); out.append(k)
    if KW.get("blend") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "classic.blend"))
    print("__RESULT__", json.dumps({"n_buildings": len(obs), "kinds": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
