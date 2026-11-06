"""Settlement (ensemble) plan (knowledge — settlements.yaml).
Archetype + place (+ host castle via the entry) → boundary → streets (grid | organic | growth; hive tiers) → terrain edit
(streets level across, squares as a platform; masks road, pave, pad for the site) → blocks → plots → house programs (the
"house" type via the buildings.py entry) → edge classes (street, neighbor, yard) → stairs on steep streets → checks and
density (Spacematrix: GSI, FSI, share of streets against the archetype's measures) → <out>/ensemble.json, relief.npz,
location.json, plan.png.
.venv/bin/python -m components.buildings.ensembles.ensemble_plan <place> type=town [seed=1] [at=x,y] [size=360] [host=castle folder] [out=folder]"""
import os
import sys
from components.buildings import arch_parts
import copy
import json
import math
import shutil
import time
import zlib

import cv2
import numpy as np
import shapely
import yaml
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from components.buildings import buildings as B
from components.buildings.ensembles import districts as DS
from components.buildings.siting import grade as GR
from components.buildings.influences import influence as INF
from components.buildings.ensembles import landmarks as LM
from components.buildings.ensembles import plots as PL
from components.buildings.ensembles import streets as ST
from core.surface.ground import Ground

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = arch_parts.ROOT; T_START = time.time()
KN = yaml.safe_load(open(os.path.join(HERE, "settlements.yaml"), encoding="utf-8"))
KINDS = tuple(KN["archetypes"])


def r2(v): return round(float(v), 2)


def ring(g):
    return [[r2(x), r2(y)] for x, y in list(g.exterior.coords)[:-1]]


def polys(g):
    return [ring(p) for p in PL.parts(g, 0.5)]


def pick(v, rng):
    return v[int(rng.integers(len(v)))] if isinstance(v, list) and v and isinstance(v[0], list) else v


# ---------------------------------------------------------------- pavement
def tri_paving(path, geom, seg=2.5):
    """Pavement for Blender (it has no shapely): constrained Delaunay triangulation along the outlines of streets, roads and
    squares, edges sampled every seg m. → paving.npz (V — xy, T — triples)."""
    V = []; T = []; key = {}
    for p in PL.parts(geom, 0.5):
        tri = shapely.constrained_delaunay_triangles(p.segmentize(seg))
        for t in getattr(tri, "geoms", []):
            ids = []
            for x, y in list(t.exterior.coords)[:3]:
                k = (round(x, 3), round(y, 3))
                if k not in key: key[k] = len(V); V.append(k)
                ids.append(key[k])
            if len(set(ids)) == 3: T.append(ids)
    np.savez_compressed(path, V=np.array(V, np.float32).reshape(-1, 2), T=np.array(T, np.int32).reshape(-1, 3))
    return len(T)


# ---------------------------------------------------------------- water
def water_geom(G, c, R, step=1.0):
    """Water of the place (depth > 5 cm) as polygons: nothing is built in it, streets over it are bridges."""
    xs = np.arange(c[0] - R - 20, c[0] + R + 20, step); ys = np.arange(c[1] - R - 20, c[1] + R + 20, step); X, Y = np.meshgrid(xs, ys)
    M = (G.water(X, Y) > 0.05).astype(np.uint8)
    if not M.any(): return Polygon()
    cs, hier = cv2.findContours(M, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE); out = []
    for i, cn in enumerate(cs):
        if hier[0][i][3] != -1 or len(cn) < 3: continue
        holes = [cs[j][:, 0, :] for j in range(len(cs)) if hier[0][j][3] == i and len(cs[j]) >= 3]
        T_ = lambda a: [(xs[0] + x * step, ys[0] + y * step) for x, y in a]
        out.append(Polygon(T_(cn[:, 0, :]), [T_(h) for h in holes]).buffer(0))
    return unary_union(out).buffer(0.6) if out else Polygon()


def stairs(E, height, limit, step=1.5):
    """Steep street segments → steps: [(s0, s1)] along the edge, and the maximum slope."""
    for e in E:
        g = LineString(e["pts"]); L = g.length
        if L < 2 * step or limit >= 0.99: e["slope"] = 0.0; e["stairs"] = []; continue
        S = np.arange(0, L + 1e-6, step); P = np.array([g.interpolate(s).coords[0] for s in S]); Z = height(P[:, 0], P[:, 1])
        gr = np.abs(np.gradient(Z, S)); e["slope"] = r2(gr.max())
        st = gr > limit; runs = []; i = 0
        while i < len(S):
            if st[i]:
                j = i
                while j + 1 < len(S) and st[j + 1]: j += 1
                if S[j] - S[i] >= 1.4 or (j == i and False): runs.append([r2(max(0, S[i] - step / 2)), r2(min(L, S[j] + step / 2))])
                i = j + 1
            else: i += 1
        e["stairs"] = runs


# ---------------------------------------------------------------- plan
def plan(place, out=None, type="town", seed=1, at=None, size=None, host=None, power=None, trade=None, wealth=None, inequality=None,
         wind=None, centre=None, **kw):
    """Archetype (type: polis, colony, subura, town, posad, castle_inside, village, hive) + place → settlement plan in out.
    Influence knobs (0–1; absent — taken from the archetype, or 0.5): power (rule), trade (commerce), wealth, inequality;
    wind — the direction it blows from, °; centre — "x,y" of the main power center."""
    t0 = time.time(); kind = kw.pop("kind", None) or type
    if kind not in KN["archetypes"]: raise KeyError(f"no archetype {kind}: {', '.join(KINDS)}")
    A = copy.deepcopy(KN["archetypes"][kind]); seed = int(seed)
    rng = np.random.default_rng(zlib.crc32(f"{kind}:{seed}".encode()))
    out = os.path.abspath(out or os.path.join(ROOT, "out", "ensembles", f"{kind}_{seed}")); os.makedirs(out, exist_ok=True)
    logic = []
    def why(step, what, val, because): logic.append({"step": step, "what": what, "value": val, "why": because})
    if isinstance(at, str): at = [float(v) for v in at.split(",")]
    W = {k: float(v) for k, v in A["streets"]["width"].items()}
    S = A["streets"]; vid = S["kind"]
    # ---- host castle (via the entry: plan and passport)
    base = os.path.abspath(place); HP = None; host_dir = None
    if A.get("owner"):
        host_dir = host
        if host_dir is None:
            host_dir = os.path.join(out, "castle")
            hint = dict(A["owner"].get("intent") or {}, **{k: v for k, v in kw.items() if k in ("site", "purpose", "era", "culture", "scale", "mood")})
            if at is not None: hint["at"] = at
            B.plan("castle", base, out=host_dir, seed=seed, **hint)
        HP = B.passport(host_dir); base = os.path.abspath(host_dir)
        why("owner", "castle", HP["intent"], "the posad and the lanes are laid out from the castle gates and outline (passport)")
    G = Ground(base); nb = G.near_bounds()
    if HP is not None:
        outline = unary_union([Polygon(r) for r in HP["outline"]]).buffer(0)
        c = np.asarray(outline.centroid.coords[0])
    else:
        outline = None
        c = np.asarray(at if at is not None else G.loc.get("center", [0.0, 0.0]), float)
    lim = min(c[0] - nb[0], nb[2] - c[0], c[1] - nb[1], nb[3] - c[1]) - 6.0
    R = min(float(size or A.get("size", 300)) / 2, lim)
    why("place", "center and radius", [r2(c[0]), r2(c[1]), r2(R)], "within the near terrain patch (0.5 m grid), 6 m inset from the edge")
    Gb = A.get("boundary", {"kind": "edge", "shape": "circle", "irregularity": 0.2})
    water = Polygon() if A.get("tiers") else water_geom(G, c, R)
    if not water.is_empty: why("place", "water", r2(water.area), "no building in the water or right at the water; streets cross water by bridges, not along it")
    # ---- acropolis (before the streets: the grid goes around the foot)
    LMK = A.get("landmarks") or []; CL = B.module("classics", "plan") if (LMK or A.get("acropolis") or A.get("sanctuary")) else None
    acro = None; th_site = None; reserve = []; gate_acro = None; path_foot = None
    if A.get("acropolis"):
        acro = LM.acropolis(G, c, R, A["acropolis"])
        if acro is not None:
            reserve.append(acro["foot"])
            why("acropolis", "hill", {"projection_m": r2(acro["prom"]), "platform_m2": r2(acro["top"].area), "elevation": r2(acro["z_top"])},
                "a hill that rises above the place's median by the `projection` is an acropolis: a sanctuary on the platform, the grid goes around the foot, a ring street along the foot (Athens)")
    # ---- streets
    lines = []; gates = []; wall = None; plaza = None; sq_list = []; Bd = None; obstacles = []; decks = None
    ang = 0.0
    if vid == "grid":
        o = S.get("orientation", 0)
        if o == "hillside":
            X, Y = np.meshgrid(np.linspace(-R / 2, R / 2, 9) + c[0], np.linspace(-R / 2, R / 2, 9) + c[1])
            gx, gy = G.grad(X.ravel(), Y.ravel()); g = np.array([np.mean(gx), np.mean(gy)])
            if np.linalg.norm(g) > 0.05:
                ang = math.atan2(g[1], g[0]) + math.pi / 2
                why("streets", "grid along the slope", r2(math.degrees(ang)), f"long streets follow the contour lines (Priene); slope {r2(np.linalg.norm(g) * 100)} %")
            else: o = "light_hillside"
        if o in ("light", "light_hillside"):
            ang = math.radians(rng.normal(0, 4))
            why("streets", "grid by the cardinal directions", r2(math.degrees(ang)), "cardo — north–south, decumanus — west–east" + (" (Miletus, Priene: slope gentler than 5 %)" if o == "light_hillside" else ""))
        elif o != "hillside": ang = math.radians(float(o))
        Bd = ST.boundary(Gb.get("shape", "circle"), c, R, float(Gb.get("irregularity", 0.1)), rng, ang)
        if acro is not None and not Bd.buffer(-6.0).contains(acro["foot"]):         # acropolis — inside the ring of walls
            Bd = unary_union([Bd, acro["foot"].buffer(14.0)]).convex_hull.intersection(Point(c).buffer(R * 1.02, quad_segs=24))
        if acro is not None and "theatre" in [q.get("kind") for q in LMK]:
            th_site = LM.site_theatre(G, CL, acro, c, Bd, CL.knowledge()["kinds"]["theatre"], rng, size=float(A["acropolis"].get("theatre", 1.0)))
            if th_site is not None:
                reserve.append(th_site["zone"])
                why("theatre", "in the acropolis slope", {"R": r2(th_site["R"]), "cost": th_site["score"]}, "the cavea lies into the slope (less fill), does not enter the temple platform, a view of the town")
        blk = pick(S["city_block"], rng)
        lines, axes = ST.grid(c, Bd, ang, blk, W, S.get("axes", {"u": 1}), ring=float(S.get("ring", 0) or 0), jitter=float(S.get("distortion", 0) or 0), rng=rng)
        why("streets", "city_block", blk, "measure from the archetype's reference (settlements.yaml)")
        for ax in axes:
            for q in (ax.coords[0], ax.coords[-1]):
                if Bd.exterior.distance(Point(q)) < 1.0: gates.append([r2(q[0]), r2(q[1])])
        if reserve:                                                                # the grid goes around the acropolis and the theatre; a ring street along the foot
            Zr = unary_union(reserve); kept = []
            for l in lines:
                if l["kind"] == "edge": kept.append(l); continue
                g = l["geom"].difference(Zr)
                for part in getattr(g, "geoms", [g]):
                    if part.geom_type == "LineString" and part.length > 8.0: kept.append(dict(l, geom=part))
            lines = kept
            if acro is not None:
                ring_g = acro["foot"].buffer(W.get("street", 5.0) / 2 + 0.4).exterior.intersection(Bd.buffer(-3.0))
                if th_site is not None: ring_g = ring_g.difference(th_site["zone"].buffer(1.0))
                for part in getattr(ring_g, "geoms", [ring_g]):
                    if part.geom_type == "LineString" and part.length > 8.0: lines.append(ST.line(part, "street", W.get("street", 5.0)))
        if acro is not None:                                                       # the ascent to the sanctuary — a path with a slope cost
            sg = unary_union([l["geom"] for l in lines if l["kind"] != "edge"])
            pu = LM.path_up(G, acro, sg, th_site["zone"] if th_site is not None else None)
            if pu is not None:
                g, p0, uo = pu; gate_acro = (p0, uo); path_foot = list(g.coords[-1])
                lines.append(ST.line(g, "street", W.get("street", 5.0) * 0.9, kind="rise"))
                why("acropolis", "rise", r2(g.length), "shortest path over a 2 m lattice: steeper than 30 % is expensive — the stair runs diagonally, around the theatre")
        if Gb.get("kind") == "wall": wall = Bd
    elif vid == "organic":
        O = ST.organic(c, R, S, W, rng, height=G.height, wall=Gb.get("kind") == "wall", n_gates=int(S.get("gate", 3)),
                       plaza=any(q.get("kind") in ("market", "agora", "forum") for q in A.get("squares", [])))
        lines = O["lines"]; Bd = O["city"]; wall = O["wall"]; gates = [[r2(x), r2(y)] for x, y in O["gates"]]; plaza = O["plaza"]
        why("streets", "organic", {"n_city_blocks": len(O["cells"]), "n_gates": len(gates)}, "Voronoi on a spiral, streets from the gates to the square by the shortest path (slope cost)")
    elif vid == "growth":
        if HP is not None and A["owner"].get("where") == "courts":
            Bd = unary_union([Polygon(q["poly"]).buffer(0) for q in HP["courts"]]).buffer(0)
            occupied = unary_union([Polygon(r).buffer(0) for r in HP["occupied"]])
            obstacles.append(occupied.buffer(0.6))
            R = max(float(np.max(np.linalg.norm(np.asarray(Bd.convex_hull.exterior.coords) - c, axis=1))) + 12.0, 30.0)
            seeds = []
            for an in HP["anchors"]:                                                # castle gates from the inside — where the yard lanes start
                if an["name"].startswith("gate") and "inside" in an["name"]:
                    p = np.asarray(an["p"], float); q = np.asarray(shapely.ops.nearest_points(Bd.buffer(-1.0), Point(p))[0].coords[0])
                    d = np.asarray(Bd.centroid.coords[0]) - q; seeds.append((q, math.atan2(d[1], d[0]), "street"))
            for e in ([] if seeds else HP["entrances"]):
                if e["kind"] not in ("gate", "opening", "arch") or e["width"] < 2.4: continue
                p = np.asarray(e["p"] if not isinstance(e["p"][0], list) else e["p"][-1], float)
                if Bd.distance(Point(p)) < 6:
                    q = np.asarray(shapely.ops.nearest_points(Bd.buffer(-1.0), Point(p))[0].coords[0]); d = q - p
                    seeds.append((q, math.atan2(d[1], d[0]), "street"))
            if not seeds: seeds = [(np.asarray(Bd.representative_point().coords[0]), rng.uniform(0, 2 * math.pi), "street")]
            why("streets", "growth from the castle gates", len(seeds), "yard lanes start from the gate passages (passport entries)")
        else:
            Bd = ST.boundary(Gb.get("shape", "circle"), c, R, float(Gb.get("irregularity", 0.2)), rng)
            seeds = []
            if HP is not None:
                zone = outline.buffer(float(A["owner"].get("offset", 25)))
                obstacles.append(zone); Bd = Bd.difference(outline.buffer(4.0))
                for an in HP["anchors"]:                                            # castle gates from the outside — the start of the posad road
                    if an["name"].startswith("gate") and "outside" in an["name"]:
                        p = np.asarray(an["p"], float); d = p - c; d /= max(np.linalg.norm(d), 1e-9)
                        seeds.append((p + d * 2.0, math.atan2(d[1], d[0]), "artery")); gates.append([r2(p[0]), r2(p[1])])
                for e in ([] if seeds else HP["entrances"]):                         # no anchors — passages (gate, opening, ramp) wider than 2.4 m
                    if e["kind"] not in ("gate", "opening", "arch", "ramp", "bridge") or e["width"] < 2.4: continue
                    p = np.asarray(e["p"] if not isinstance(e["p"][0], list) else e["p"][-1], float); d = p - c; d /= max(np.linalg.norm(d), 1e-9)
                    if any(np.linalg.norm(p - np.asarray(g)) < 15 for g in gates): continue
                    seeds.append((p + d * 2.0, math.atan2(d[1], d[0]), "artery")); gates.append([r2(p[0]), r2(p[1])])
                for k in range(int(S.get("roads", 0))):                          # roads from afar — to the posad
                    a0 = seeds[0][1] if seeds else rng.uniform(0, 2 * math.pi)
                    a = a0 + rng.uniform(-0.9, 0.9) if k == 0 else a0 + math.pi + rng.uniform(-1.0, 1.0)
                    p = c + np.array([math.cos(a), math.sin(a)]) * R * 0.98
                    seeds.append((p, a + math.pi, "street"))
                if seeds:
                    obstacles = [zone.difference(unary_union([LineString([s[0], s[0] + np.array([math.cos(s[1]), math.sin(s[1])]) * 60]).buffer(W.get("artery", 6) / 2 + 0.5) for s in seeds]))]
                why("streets", "growth from the castle gates", len(seeds), f"approach road — across the field of fire {A['owner'].get('offset', 25)} m, the posad — beyond it")
            else:
                for k in range(int(S.get("roads", 2))):
                    a = rng.uniform(0, 2 * math.pi) if k == 0 else seeds[0][1] + math.pi + rng.normal(0, 0.5)
                    p = c + np.array([math.cos(a), math.sin(a)]) * R * 0.98
                    seeds.append((p, a + math.pi, "street" if "artery" not in W else "artery"))
                why("streets", "growth from the edges", len(seeds), "roads enter from the edges of the place and grow along the contour lines (Emilien 2012)")
        blocked = None; wet = None
        if not water.is_empty:                                                    # water: a segment end must not be in the water; across a narrow channel — a bridge
            wp = shapely.prepared.prep(water.buffer(1.0)); wet = lambda q: wp.contains(Point(q))
        if obstacles:
            obs = unary_union(obstacles); prep = shapely.prepared.prep(obs)
            blocked = lambda q: prep.contains(Point(q))
        lines = ST.growth(seeds, Bd, S, W, rng, height=G.height, blocked=blocked, wet=wet)
        n0 = len(lines); lines = ST.connect(lines, W, blocked=blocked)
        if len(lines) > n0: why("streets", "linking the parts", len(lines) - n0, "dead ends of neighboring parts are extended toward each other (Parish–Müller)")
    # roads beyond the wall from the gates
    if wall is not None and gates:
        box_ = box(nb[0] + 4, nb[1] + 4, nb[2] - 4, nb[3] - 4).difference(wall.buffer(-0.3))
        for gp in gates:
            d = np.asarray(gp) - c; d /= max(np.linalg.norm(d), 1e-9)
            rl = ST.growth([(np.asarray(gp) + d * 0.6, math.atan2(d[1], d[0]), "artery")], box_,
                           {"step": 20, "branching": 0.0, "turn": 14, "snap": 0.1, "slope": 2.5, "n_segments": 9}, W, rng, height=G.height)
            for l in rl: l["kind"] = "road"
            lines += rl
    if not lines: raise RuntimeError("no streets")
    P_, E = ST.build_graph(lines)
    if not water.is_empty:                                                        # streets across water — bridges; along the channel — drop them
        keep = []
        for e in E:
            g = LineString(e["pts"]); wl = g.intersection(water).length / max(g.length, 1e-6)
            ends_wet = water.contains(Point(e["pts"][0])) or water.contains(Point(e["pts"][-1]))
            if wl > 0.6 and ends_wet: continue                                     # along the channel; across it with dry ends — a bridge
            if wl > 0.0: e["kind"] = "bridge" if e["kind"] != "road" else "road"; e["over_water"] = True
            keep.append(e)
        why("streets", "bridges over water", sum(1 for e in keep if e.get("over_water")), f"edges removed in the channel: {len(E) - len(keep)}")
        E = keep
    # ---- hive tiers
    Y = A.get("tiers")
    if Y:
        zs = G.height(*np.asarray(Bd.exterior.coords).T); z_base = float(np.median(zs))
        decks = {"count": int(Y["count"]), "height": float(Y["height"]), "slab": float(Y["slab"]), "z0": r2(z_base), "outline": ring(Bd)}
        why("tiers", "hive", decks["count"], f"tier {Y['height']} m, slab {Y['slab']} m; platform — at the ground median {r2(z_base)} m")
    # ---- blocks, squares, shafts
    rib = ST.ribbon([e for e in E if e["kind"] != "road"])
    rib = rib.intersection(Bd.buffer(0.5)) if Bd is not None else rib
    roads = ST.ribbon([e for e in E if e["kind"] == "road"])
    wall_band = wall.exterior.buffer(1.8) if wall is not None else Polygon()
    free = Bd.difference(rib).difference(wall_band)
    if not water.is_empty: free = free.difference(water.buffer(2.0))
    for ob in obstacles: free = free.difference(ob)
    for z_ in reserve: free = free.difference(z_)                                     # the acropolis and the theatre are not blocks
    blocks = PL.parts(free, 30.0)
    used = set(); squares = []
    def nearest_blocks(p, k):
        order = sorted(range(len(blocks)), key=lambda i: blocks[i].centroid.distance(Point(p)))
        chosen = []
        for i in order:
            if i in used: continue
            if not chosen or any(blocks[i].distance(blocks[j]) < max(W.values()) + 0.6 for j in chosen): chosen.append(i)
            if len(chosen) >= k: break
        return chosen
    wmax = max(W.values())
    for q in A.get("squares", []):
        k = int(q.get("n_city_blocks", 1)); v_ = q.get("kind")
        if plaza is not None and v_ in ("market", "agora", "forum"):
            sq = plaza.buffer(0); squares.append({"kind": v_, "poly": sq})
            for i, b in enumerate(blocks):
                if b.intersection(sq).area > 0.3 * b.area: used.add(i)
            why("squares", v_, r2(sq.area), "cell 0 is the square, the arterial streets lead to it")
            continue
        if k <= 0:
            if v_ == "trade" and gates:
                p = np.asarray(gates[0]); d = p - c; d /= max(np.linalg.norm(d), 1e-9)
                zone = obstacles[0] if obstacles else Polygon()
                q0 = p + d * (float(A.get("owner", {}).get("offset", 25)) + 8)
                sq = Point(q0).buffer(11).difference(zone)
                squares.append({"kind": v_, "poly": sq}); why("squares", "trade", r2(sq.area), "by the road from the gates, beyond the field of fire")
            continue
        p = c
        if q.get("where") == "center" and v_ == "forum": p = c - np.array([-math.sin(ang), math.cos(ang)]) * (wmax / 2 + 12)
        if q.get("where") == "foot" and acro is not None:                         # agora — at the foot, where the ascent begins
            p = np.asarray(path_foot if path_foot is not None else shapely.ops.nearest_points(acro["foot"].exterior, Point(c))[0].coords[0], float)
        ch = nearest_blocks(p, k)
        if not ch: continue
        for i in ch: used.add(i)
        sq = unary_union([blocks[i].buffer(wmax / 2 + 0.3) for i in ch]).buffer(-(wmax / 2 + 0.3))
        squares.append({"kind": v_, "poly": sq}); why("squares", v_, {"n_city_blocks": len(ch), "m2": r2(sq.area)}, "blocks at the center — an open square")
    sanct = None
    if A.get("sanctuary") and blocks and acro is None:
        cand = [i for i in range(len(blocks)) if i not in used]
        if cand:
            i = max(cand, key=lambda j: float(G.height(*blocks[j].centroid.coords[0])))
            used.add(i); sanct = blocks[i]
            why("sanctuary", "temenos on the highest point", r2(float(G.height(*sanct.centroid.coords[0]))), "the sanctuary is on the acropolis (the highest point inside the walls)")
    shafts = []
    if A.get("shafts") and blocks:
        n_sh = int(rng.integers(A["shafts"]["count"][0], A["shafts"]["count"][1] + 1))
        deg = {}
        for e in E:
            if e["kind"] in ("edge", "road"): continue
            deg[e["a"]] = deg.get(e["a"], 0) + 1; deg[e["b"]] = deg.get(e["b"], 0) + 1
        nodes = [i for i, k in deg.items() if k >= 4 and np.linalg.norm(np.asarray(P_[i]) - c) < R * 0.62]
        rng.shuffle(nodes)
        for ni in nodes:                                                          # shaft — 4 blocks around a crossing
            if len(shafts) >= n_sh: break
            q = Point(P_[ni]); grp = sorted([i for i in range(len(blocks)) if i not in used], key=lambda i: blocks[i].distance(q))[:4]
            if len(grp) < 4 or max(blocks[i].distance(q) for i in grp) > wmax * 0.8: continue
            if any(Polygon(s_).distance(q) < wmax * 4 for s_ in shafts): continue
            if any(sq_["poly"].distance(q) < wmax * 2 for sq_ in squares): continue
            for j in grp: used.add(j)
            sh = unary_union([blocks[j].buffer(wmax / 2 + 0.3) for j in grp]).buffer(-(wmax / 2 + 0.3))
            shafts.append(ring(PL.parts(sh)[0]))
        for e in E:
            m = LineString(e["pts"]).interpolate(0.5, normalized=True)
            if any(Polygon(s).contains(m) for s in shafts): e["kind"] = "bridge"
        why("shafts", "through the tiers", len(shafts), "2 × 2 blocks without slabs — light and lifts; streets inside — bridges")
    if squares and any(q.get("kind") == "stoa" for q in LMK):                      # a square with stoas — the streets inside it are removed
        inner = unary_union([s_["poly"].buffer(0.5) for s_ in squares if s_["kind"] in ("agora", "forum", "market")])
        n0 = len(E); E = [e for e in E if LineString(e["pts"]).intersection(inner).length < 0.75 * LineString(e["pts"]).length]
        if len(E) < n0:
            rib = ST.ribbon([e for e in E if e["kind"] != "road"]); rib = rib.intersection(Bd.buffer(0.5)) if Bd is not None else rib
            why("squares", "streets inside the square", n0 - len(E), "a square made of two blocks — the street between them goes under the stoas")
    civ = []                                                                       # civic blocks: near the agora, near the edge
    sq_main = next((s_ for s_ in squares if s_["kind"] in ("agora", "forum", "market")), None)
    for q in LMK:
        cand = [i for i in range(len(blocks)) if i not in used and blocks[i].area > 700]
        rect = lambda b: b.area / max(b.minimum_rotated_rectangle.area, 1e-6)
        if q.get("where") == "at_agora" and sq_main is not None and cand:
            near = [i for i in cand if blocks[i].distance(sq_main["poly"]) < wmax + 1.0 and rect(blocks[i]) > 0.85] or cand
            i = min(near, key=lambda j: blocks[j].centroid.distance(sq_main["poly"].centroid)); used.add(i); civ.append((q["kind"], blocks[i]))
        elif q.get("where") == "edge" and cand:
            good = [i for i in cand if rect(blocks[i]) > 0.88 and blocks[i].area > 1100] or cand
            i = max(good, key=lambda j: blocks[j].centroid.distance(Point(c))); used.add(i); civ.append((q["kind"], blocks[i]))
    if civ: why("landmarks", "civic blocks", [k for k, _ in civ], "bouleuterion — near the agora (Priene, Miletus), palaestra — by the wall on level ground")
    if not water.is_empty:
        for s_ in squares: s_["poly"] = s_["poly"].difference(water.buffer(1.0))
    sq_geom = unary_union([s["poly"] for s in squares]) if squares else Polygon()
    # ---- terrain edit (before the plots: houses take their height from the edited ground)
    pave = unary_union([rib, roads, sq_geom]) if A.get("pavement") else sq_geom
    flat = None
    if decks: flat = (Bd, decks["z0"])
    dry = (lambda g: g.difference(water.buffer(1.5))) if not water.is_empty else (lambda g: g)
    GR.grade(base, out, dry(unary_union([rib, roads])), [s["poly"] for s in squares] + ([sanct] if sanct is not None else []) + [b for _, b in civ], dry(pave), sq_geom, flat)
    shape_items = []
    if acro is not None:                                                           # the acropolis platform — exactly at the median of the top
        shape_items.append({"geom": acro["top"].buffer(-1.5), "mode": "set", "z": acro["z_top"], "feather": 3.0})
    if th_site is not None:                                                        # the cavea — by cutting the slope, the orchestra and skene — by a platform
        sector, zf, flat_th = LM.cavea_cut(CL, th_site)
        shape_items += [{"geom": sector, "mode": "cut", "z": zf}, {"geom": flat_th, "mode": "set", "z": th_site["z"] - 0.03, "feather": 3.0}]
    if shape_items: GR.reshape(out, shape_items)
    G2 = Ground(out)
    height = G2.height
    # ---- landmarks: programs of the "classic" type via the entry
    landmarks = []
    if CL is not None:
        if acro is not None: landmarks += LM.acropolis_set(CL, acro, gate_acro, G2, rng, avoid=th_site["zone"] if th_site is not None else None)
        if th_site is not None:
            Pt = CL.program("theatre", th_site, {"height": G2.height, "height0": G.height, "z": th_site["z"]}, rng)
            if Pt: landmarks.append(Pt)
        if sq_main is not None and any(q.get("kind") == "stoa" for q in LMK):
            landmarks += LM.agora_stoas(CL, PL.parts(sq_main["poly"], 1.0)[0], ang, path_foot if path_foot is not None else (acro["p"] if acro else None), G2, rng)
        for k_, b_ in civ: landmarks += LM.civic(CL, k_, b_, ang, sq_main["poly"].centroid.coords[0] if sq_main is not None else c, G2, rng)
        if sanct is not None: landmarks += LM.temenos(CL, sanct, G2, rng)
        for P_l in landmarks: P_l["fp"] = CL.footprint(P_l)
        if landmarks:
            lm_fp = unary_union([Polygon(P_l["fp"]).buffer(0) for P_l in landmarks])
            GR.add_pad(out, lm_fp)
            why("landmarks", "buildings", {k: sum(1 for P_l in landmarks if P_l["kind"] == k) for k in sorted({P_l["kind"] for P_l in landmarks})},
                'the "classic" type via the entry: the order and the number of columns — by the frame (the module is the lower diameter of the column)')
    # ---- influences: power, trade, wealth — three fields on any archetype (adjustments to plots and houses)
    U = A["plots"]; mode = U["kind"]; plots = []; rest = None; passages = []; split_log = []   # blocks: (W, T, parts, expectation, ordered to split)
    sq_inf = next((s_ for v_ in ("agora", "forum", "market", "parade", "trade") for s_ in squares if s_["kind"] == v_), None)
    entry = gates + ([[r2(s_[0][0]), r2(s_[0][1])] for s_ in seeds] if vid == "growth" else [])   # entries: gates; for growth — also the road starts
    knobs = dict(A.get("influences") or {}, **{k: v for k, v in (("power", power), ("trade", trade), ("wealth", wealth), ("inequality", inequality)) if v is not None})
    if isinstance(centre, str): centre = [float(v) for v in centre.split(",")]
    IF = INF.model(None, height, c, R, Bd, E, P_, entry, squares, sq_inf, acro, outline if HP is not None else None, rib, water,
                   decks, reserve, float(np.mean(U.get("depth", [24.0, 24.0]))), rng, why, L=INF.levels(**knobs),
                   wind=float(wind) if wind is not None else (A.get("influences") or {}).get("wind"), centre=centre)
    # ---- plots
    blocks_free = [b for i, b in enumerate(blocks) if i not in used]
    free2 = unary_union(blocks_free) if blocks_free else Polygon()
    def wt(b):                                                                     # wealth and trade of a block (at the middle, along the edge)
        q = b.representative_point().coords[0]; return float(IF.W([q])[0]), IF.T_block(b)
    seat_b = {min(range(len(blocks_free)), key=lambda i: blocks_free[i].distance(Point(ce["p"])))   # the block of a power center is not split:
              for ce in IF.centres if not ce.get("landmark") and blocks_free}        # power gathers the land (the compound is a block, not a sliver)
    def empty_by_r(q):
        r = q.centroid.distance(Point(c)) / max(R, 1); return float(U.get("empty", 0.0)) + 0.55 * max(0.0, r - 0.45) ** 2
    if mode in ("strips", "farmsteads", "built_up"):
        Ew = [e for e in E if e["kind"] != "bridge"]
        if HP is not None and U.get("at_walls"):                                  # lean-tos along the curtain wall: the wall acts as a "street" of zero width
            band = []
            for cpoly in [q["poly"] for q in HP["courts"]]:
                C_ = np.asarray(Polygon(cpoly).exterior.coords)
                for i in range(len(C_) - 1):
                    band.append({"pts": [C_[i].tolist(), C_[i + 1].tolist()], "w": 0.2, "rank": "passage", "kind": "wall"})
            wp, free2 = PL.strips(free2, band, U["facade"], [U["at_walls"] * 0.8, U["at_walls"]], rng, empty=0.0, min_share=0.5)
            for p in wp: p["at_walls"] = True
            plots += wp
            why("plots", "lean-tos by the curtain wall", len(wp), "houses are leaned against the castle wall (Carcassonne, Mont-Saint-Michel)")
        sp, rest = PL.strips(free2, Ew, U["facade"], U.get("depth", [20, 30]), rng, empty=float(U.get("empty", 0.0)),
                             empty_fn=empty_by_r if mode == "farmsteads" else None, mod=IF.cut)
        plots += sp
        why("plots", "strips along the streets", len(sp), f"facade {U['facade']} m, depth {U.get('depth')} m; the senior rank takes the facade first; "
            "× facade and depth multipliers, and gardens — by wealth and trade (influences)")
        lim = IF.K["districts"]["nobility" if mode == "built_up" else "poor"]       # a block's yard fills up with buildings: in a slum — for the not-rich, in a town — for the poor
        yard = mode == "built_up" or (mode == "strips" and vid != "growth")         # for growth, behind the houses is a field, not a block yard
        if yard: rest = unary_union([q for q in PL.parts(rest, 30.0) if wt(q)[0] < lim])
        if yard and not rest.is_empty:
            back, lanes = PL.infill(rest, rib, Ew, area=float(np.mean(U["facade"]) * np.mean(U["depth"])) * (1.0 if mode == "built_up" else 0.45),
                                    pw=float(U.get("passage", 1.3)), rng=rng)
            for q in back: q["court"] = True                                         # yard plots are not a facade-based subdivision (the width check excludes them)
            if lanes:
                cut = unary_union([l.buffer(float(U.get("passage", 1.3)) / 2, cap_style="flat") for l in lanes])
                plots = PL.subtract(plots + back, cut)
                passages = [ST.line(l, "passage", float(U.get("passage", 1.3))) for l in ST.snap_ends(lanes, lines)]
                P_, E = ST.build_graph(lines + passages + [ST.line(LineString(e["pts"]), e["rank"], e["w"], e["kind"]) for e in E if e["kind"] == "bridge"])
                for e in E:
                    m = LineString(e["pts"]).interpolate(0.5, normalized=True)
                    if any(Polygon(s).contains(m) for s in shafts): e["kind"] = "bridge"
                rib = unary_union([rib, cut])
            else: plots += back
            why("plots", "yard infill", {"n_back": len(back), "n_passages": len(lanes)}, "a block's yard fills up with back houses, each with a passage to the street "
                "(Subura, Kowloon; in a town — the yards of the poor, Conzen's burgage cycle)")
    elif mode == "rows":
        for b in blocks_free:
            fm = IF.cut_pw(*wt(b))[0]; plots += PL.rows(b, [U["facade"][0] * fm, U["facade"][1] * fm], float(U.get("passage", 1.4)), rng)
        why("plots", "two rows of houses along a passage", len(plots), "Olynthus: a block of 2 × 5 houses ~17 × 17 m, a passage between the rows; house width — by wealth and trade")
    elif mode == "whole":
        sh = U.get("frac_division", 0.3)
        for ib, b in enumerate(blocks_free):
            if ib in seat_b: plots.append({"poly": b}); continue
            Wb, Tb = wt(b); n0 = len(plots); poor = Wb <= IF.K["districts"]["poor"] and b.area > 300
            told = [poor]
            if poor: plots += PL.divide(b, b.area / 3.2, 0.3, rng)                  # a poor block — into 3–4 houses
            else: plots += PL.whole(b, IF.halves(Wb, Tb, sh), rng, told := [])
            split_log.append((Wb, Tb, len(plots) - n0, 3.2 if poor else 1.0 + (IF.halves(Wb, Tb, sh) if b.area > 260 else 0.0), bool(told and told[-1])))
        why("plots", "block = plot", len(plots), "Timgad: a 70-foot block — a house over the whole block, some halved; richer — whole, poorer and on a busy street — divided")
    elif mode == "blocks":
        gap = float(U.get("gap", 3.0)); cuts = []
        for ib, b in enumerate(blocks_free):
            if ib in seat_b: plots.append({"poly": b}); continue
            Wb, Tb = wt(b)                                                         # richer — the whole block, poorer and on a busy street — 2 and 4
            k = IF.pieces(Wb, Tb, rng)
            if k > 1:
                got, segs = PL.divide_cuts(b, b.area / k * 1.02, 0.15, rng)
                cuts += segs; plots += got
            else: got = [{"poly": b}]; plots += got
            split_log.append((Wb, Tb, len(got), IF.pieces_exp(Wb, Tb), k > 1))
        if cuts:
            cut = unary_union([l.buffer(gap / 2, cap_style="flat") for l in cuts]); plots = PL.subtract(plots, cut)
            passages = [ST.line(l, "passage", gap) for l in ST.snap_ends(cuts, lines)]
            P_, E = ST.build_graph(lines + passages)
            for e in E:
                m = LineString(e["pts"]).interpolate(0.5, normalized=True)
                if any(Polygon(s_).contains(m) for s_ in shafts): e["kind"] = "bridge"
            rib = unary_union([rib, cut])
        plots = [p for p in plots if not p["poly"].is_empty and p["poly"].area > 30]
        why("plots", "hub-blocks", len(plots), "a block split into 1–3 sub-blocks with a gap (service passages)")
    for e in E:                                                                    # edges rebuilt (passages, cuts) — the flow comes from the old ones
        if "flow" not in e: e["flow"] = round(IF.T_line(LineString(e["pts"])), 3)
    n_up = IF.promote(E)                                                          # trade raises the rank: a through lane becomes a street (after subdivision)
    if n_up: why("influences", "rank by trade", n_up, "a lane with flow above the threshold is a street: shops and the house's main facade (subdivision and width unchanged)")
    stairs(E, height, float(A.get("stairs", 0.15)))
    if not decks: plots, _ = DS.compounds(plots, IF, None)                     # power center cores — compounds (for a hive — per tier)
    for i, p in enumerate(plots):
        p["id"] = i; p["fronts"] = PL.fronts(p["poly"], [e for e in E if e["kind"] != "wall"])
    # ---- house programs ("house" type via the entry): kind, storeys, shops, tone — by the plot's three influences
    HM = B.module("house", "plan"); KH = HM.kinds()
    weights = A["houses"]; names = list(weights)
    Fl = A.get("storeys", {}); progs = []; yards = []
    deck_list = list(range(decks["count"])) if decks else [None]
    for dk in deck_list:
        dprogs = []; pl_dk = plots
        if decks:
            pl_dk, made = DS.compounds(plots, IF, dk)
            for k_, cp in enumerate(made): cp["id"] = len(plots) + 1000 * dk + k_; cp["fronts"] = PL.fronts(cp["poly"], [e for e in E if e["kind"] != "wall"])
        for p in pl_dk:
            if p["poly"].area < 12: continue
            X = p["poly"].representative_point().coords[0]
            Pv, Tv, Wv = (float(v[0]) for v in IF.at([X], [max([f.get("flow", 0.0) for f in p["fronts"]] or [0.0])], dk))
            ctx = dict(IF.place(Pv, Tv, Wv, Fl), height=height)                    # place value V = max(P, T, W): storeys, tone, shops
            if A.get("limit_height"): ctx["limit"] = float(A["limit_height"])
            if decks:
                ctx["z_deck"] = decks["z0"] + dk * decks["height"] + (decks["slab"] if dk > 0 else 0.0)
                ctx["ceiling"] = decks["height"] - decks["slab"] - (0.0 if dk < decks["count"] - 1 else -decks["height"] * 0.0) - rng.uniform(0, 6)
                if dk == decks["count"] - 1 and rng.random() < 0.35: ctx["tower"] = rng.uniform(1.6, 3.2)
                if p.get("compound") is None and rng.random() < IF.cut_pw(Wv, Tv)[2]: continue   # gardens and terraces of the rich tiers
            if p.get("compound") is not None:                                     # compound: chambers, yard or tower — houses of the archetype's kind
                got, yard = DS.compound_progs(HM, KH, IF, p, IF.centres[p["compound"]], names, weights, Fl, ctx, rng)
                if yard is not None and dk in (None, 0): yards.append(yard)
                lab = "power"
            else:
                kind_h = IF.variant(weights, KH, Wv, Tv, rng)
                if p.get("at_walls") and "annex" in weights: kind_h = "annex"
                elif "annex" in weights and kind_h == "annex": kind_h = [k for k in names if k != "annex"][0]
                ctx["corrections"] = IF.build(KH[kind_h], Wv)
                got = HM.program(kind_h, p, ctx, rng); lab = IF.label(Pv, Tv, Wv)
            if not got: continue
            if dk in (None, 0):                                                     # a plot without a tier (for a hive — the plan field, not the lowest tier)
                Pp, Tp, Wp = (Pv, Tv, Wv) if dk is None else (float(v[0]) for v in IF.at([X], [max([f.get("flow", 0.0) for f in p["fronts"]] or [0.0])], None))
                p["influence"] = [round(Pp, 3), round(Tp, 3), round(Wp, 3)]; p["district"] = lab if dk is None else ("power" if p.get("compound") is not None else IF.label(Pp, Tp, Wp))
            for g in got:
                g["deck"] = dk; g["id"] = len(progs) + len(dprogs); g["influence"] = [round(Pv, 3), round(Tv, 3), round(Wv, 3)]; g["district"] = lab
                if p.get("compound") is not None: g["compound"] = int(p["compound"])
                dprogs.append(g)
        HM.edges(dprogs, rib)
        progs += dprogs
    if IF.centres: why("influences", "compounds", {ce["kind"]: sum(1 for h in progs if h.get("compound") == i) for i, ce in enumerate(IF.centres) if not ce.get("landmark")},
                       "power center core: plots near the center merge; chambers — a large house of the richest kind, yard — buildings along the edge, tower — a yard and a tower")
    GR.add_pad(out, unary_union([Polygon(h["F"]).buffer(0.5) for h in progs if h.get("deck") in (None, 0)]))
    pav_geom = unary_union([rib, roads, sq_geom] + yards)
    if shafts: pav_geom = pav_geom.difference(unary_union([Polygon(q) for q in shafts]))
    wbridges = []
    if not water.is_empty:
        pav_geom = pav_geom.difference(water.buffer(0.3))
        wb = ST.ribbon([e for e in E if e.get("over_water")])
        if not wb.is_empty: wbridges = polys(wb.intersection(water.buffer(2.5)))
    n_tri = tri_paving(os.path.join(out, "paving.npz"), pav_geom)
    bridges = []
    if shafts:
        bridges = polys(ST.ribbon([e for e in E if e["kind"] == "bridge"]).intersection(unary_union([Polygon(q).buffer(0.5) for q in shafts])))
    if mode == "built_up" and progs:                                             # infill: gaps in the yard — back houses (the building cycle, Conzen)
        foot0 = unary_union([Polygon(h["F"]).buffer(0.25) for h in progs])
        gaps = free2.difference(foot0).difference(rib.buffer(0.3))
        if not water.is_empty: gaps = gaps.difference(water.buffer(2.0))
        extra = []; kk = [k for k in weights if k in ("insula", "shack")] or names
        for g in PL.parts(gaps, 30.0):
            Cg = np.asarray(g.minimum_rotated_rectangle.exterior.coords)
            if min(np.linalg.norm(Cg[1] - Cg[0]), np.linalg.norm(Cg[2] - Cg[1])) < 4.0 or g.area > 600: continue
            gp = PL.parts(g.buffer(-0.3, join_style="mitre"), 20.0)
            if not gp: continue
            pl = {"poly": max(gp, key=lambda q: q.area), "fronts": [], "id": len(plots) + len(extra)}
            Pv, Tv, Wv = (float(v[0]) for v in IF.at([pl["poly"].representative_point().coords[0]]))
            if Wv >= IF.K["districts"]["nobility"]: continue                             # for the rich, a gap in the yard is a garden
            ctx = dict(IF.place(Pv, Tv, Wv), height=height,
                       storeys=[max(1, Fl.get("edge", [2, 3])[0] - 1), Fl.get("edge", [2, 3])[1]] if "edge" in Fl else [2, 3])
            if A.get("limit_height"): ctx["limit"] = float(A["limit_height"])
            got = HM.program(kk[0], pl, ctx, rng)
            for gg in (got or [])[:1]:
                gg["deck"] = None; gg["id"] = len(progs) + len(extra); gg["back"] = True
                gg["influence"] = [round(Pv, 3), round(Tv, 3), round(Wv, 3)]; gg["district"] = IF.label(Pv, Tv, Wv); extra.append(gg)
        if extra:
            HM.edges(progs + extra, rib); progs += extra
            GR.add_pad(out, unary_union([Polygon(h["F"]).buffer(0.5) for h in extra]))
            why("houses", "yard infill", len(extra), "gaps in the block fill up with back houses without a facade (light — through small courts)")
    # ---- checks and density
    foot = [Polygon(h["F"]) for h in progs if h.get("deck") in (None, 0)]
    A_s = Bd.area if Bd is not None else unary_union([p["poly"] for p in plots]).convex_hull.area
    if vid == "growth":                                                             # the settlement fabric is plots and streets (Spacematrix), not the place's circle
        A_s = unary_union([p["poly"] for p in plots] + [rib]).buffer(6).buffer(-6).area
    if decks and shafts: A_s -= sum(Polygon(q).area for q in shafts)              # hive: tier area excludes shafts (a shaft is not tier ground)
    gsi = sum(f.area for f in foot) / max(A_s, 1)
    fsi = sum(Polygon(h["F"]).area * h["floors"] for h in progs) / max(A_s, 1)
    rib_s = rib.difference(unary_union([Polygon(q) for q in shafts])) if decks and shafts else rib   # hive: tier streets exclude the bridges over shafts
    st_share = rib_s.intersection(Bd).area / max(A_s, 1) if Bd is not None else rib_s.area / max(A_s, 1)
    tree = shapely.STRtree(foot); over = 0.0
    for i, f in enumerate(foot):
        for j in tree.query(f):
            if j > i: over += f.intersection(foot[j]).area
    on_street = sum(f.intersection(rib).area for f in foot)
    comps = 0; big = 0.0; totl = 0.0
    try:
        import networkx as nx
        Gx = nx.Graph(); EL = []
        for e in E:
            if e["kind"] in ("edge", "wall"): continue
            L_ = LineString(e["pts"]).length; Gx.add_edge(e["a"], e["b"]); totl += L_; EL.append((e["a"], L_))
        cc = list(nx.connected_components(Gx)); comps = len(cc); cid = {n: i for i, k in enumerate(cc) for n in k}
        lens = np.zeros(len(cc))
        for a_, L_ in EL: lens[cid[a_]] += L_
        big = float(lens.max()) / max(totl, 1)
    except Exception: pass
    pf = [p for p in plots if not p.get("at_walls")]                               # lean-tos by the curtain wall face the castle courtyard
    with_front = sum(1 for p in pf if p["fronts"]) / max(len(pf), 1)
    dens = A.get("density", {})
    def within(v, rg): return rg is None or (rg[0] <= v <= rg[1])
    checks = [{"what": "streets are connected (share of length in the main part)", "ok": big >= 0.9, "value": r2(big), "n_parts": comps},
              {"what": "plots with a facade on a street", "ok": with_front >= (0.75 if mode == "built_up" else 0.85), "value": r2(with_front)},
              {"what": "houses do not overlap each other (m²)", "ok": over < 1.0 + 0.002 * sum(f.area for f in foot), "value": r2(over)},
              {"what": "houses are not on streets (m²)", "ok": on_street < 2.0 + 0.003 * sum(f.area for f in foot), "value": r2(on_street)},
              {"what": "GSI (built-up) within the archetype's measures", "ok": within(gsi, dens.get("GSI")), "value": r2(gsi), "measure": dens.get("GSI")},
              {"what": "FSI (storeys) within the archetype's measures", "ok": within(fsi, dens.get("FSI")), "value": r2(fsi), "measure": dens.get("FSI")},
              {"what": "share of streets within the archetype's measures", "ok": within(st_share, dens.get("streets")), "value": r2(st_share), "measure": dens.get("streets")}]
    if landmarks:
        lmp = [(P_l["kind"], Polygon(P_l["fp"]).buffer(0)) for P_l in landmarks]
        ov_h = sum(g.intersection(f).area for _, g in lmp for f in foot if g.distance(f) < 0.01)
        ov_s = sum(g.intersection(rib).area for k_, g in lmp if k_ != "propylon")
        checks += [{"what": "landmarks do not overlap houses (m²)", "ok": ov_h < 1.0, "value": r2(ov_h)},
                   {"what": "landmarks are not on streets (m²)", "ok": ov_s < 5.0 + 0.01 * sum(g.area for _, g in lmp), "value": r2(ov_s)}]
    if A.get("limit_height"):
        hmax = max((h["z0"] - h["z0"] + h["h0"] + (h["floors"] - 1) * h["fh"] for h in progs if h.get("compound") is None), default=0)   # the limit applies to tenement houses; power compounds are exempt
        checks.append({"what": "height to the cornice ≤ limit", "ok": hmax <= float(A["limit_height"]) + 0.01, "value": r2(hmax), "measure": A["limit_height"]})
    # influences: the calculation reached the plan — richer plots are larger, shops on through streets,
    # each power center has a compound
    by_front = mode in ("strips", "farmsteads", "built_up", "rows")               # what wealth drives: the facade of a strip and of a row, the division of a block
    pl_i = [p for p in plots if p.get("influence") and p.get("compound") is None and not p.get("at_walls") and not p.get("court") and p["fronts"] and p["influence"][1] < 0.4]
    room = not by_front or mode == "rows" or float(np.median([LineString(e["pts"]).length for e in PL._chains(
        [e for e in E if e["w"] > 0 and e["kind"] not in ("bridge", "edge", "road", "wall")])] or [0.0])) >= 2.5 * float(np.mean(U["facade"]))
    if by_front and len(pl_i) >= 12 and room:                                      # without trade (trade narrows the facade); a street shorter than 2.5 facades leaves no room for the width to grow
        Wp = np.array([p["influence"][2] for p in pl_i]); Tp = np.array([p["influence"][1] for p in pl_i]); q1, q2 = np.quantile(Wp, [1 / 3, 2 / 3])
        def width(p):                                                              # width along the street: facades of one side (by the normal)
            g = {}
            for f in p["fronts"]: k_ = int(round(math.degrees(math.atan2(f["n"][1], f["n"][0])) / 30.0)) % 12; g[k_] = g.get(k_, 0.0) + f["len"]
            return max(g.values())
        fm = np.array([IF.cut_pw(w_, t_)[0] for w_, t_ in zip(Wp, Tp)]); top, bot = Wp >= q2, Wp <= q1
        r_exp = float(fm[top].mean() / max(fm[bot].mean(), 1e-6))
        if r_exp >= 1.4:                                                           # measure when the formula expects a difference ≥ 40 % (an even town has nothing to measure)
            Sz = np.array([width(p) for p in pl_i])                                # the measure is the slope of log(width) over log(× of the formula's facade): the fraction that
            sl = float(np.polyfit(np.log(fm), np.log(np.maximum(Sz, 0.5)), 1)[0])  # the subdivision actually realized (1 — all of it, 0 — the mechanism had no effect)
            ratio = float(Sz[top].mean() / max(Sz[bot].mean(), 1e-6))
            checks.append({"what": "wealth → plots are wider along the facade (on non-through streets: width slope per the formula ≥ 0.25; top vs bottom third by W, formula)",
                           "ok": sl >= 0.25, "value": [r2(sl), r2(ratio), r2(r_exp)]})
    told_ = [q for q in split_log if q[4]]                                         # castrum, hive: the formula ordered a split — a cut was made (per block
    if not by_front and len(told_) >= 5:                                           # there are few of them: the difference of thirds drowns in the scatter — measured by the influences reference)
        miss = sum(1 for q in told_ if q[2] < 2) / len(told_)
        checks.append({"what": "wealth → block division: where the formula ordered a split, it was split (failures ≤ 10 %)", "ok": miss <= 0.1,
                       "value": [r2(miss), len(told_)]})
    hs_ = [h for h in progs if h.get("influence") and h.get("compound") is None and h["kind"] in names and (KH[h["kind"]].get("shop") or KH[h["kind"]].get("wings"))]
    shop_ = lambda h: bool(h.get("shop")) or any(w_.get("shops") for w_ in h.get("wings") or [])
    Th = np.array([h["influence"][1] for h in hs_]); Sh = np.array([shop_(h) for h in hs_])
    if (Th >= 0.55).sum() >= 5 and (Th < 0.25).sum() >= 5:
        hiT, loT = float(Sh[Th >= 0.55].mean()), float(Sh[Th < 0.25].mean())
        checks.append({"what": "trade → shops on through streets (share at T ≥ 0.55 and T < 0.25)", "ok": hiT >= 0.45 and hiT >= 1.5 * loT, "value": [r2(hiT), r2(loT)]})
    need = [i for i, ce in enumerate(IF.centres) if not ce.get("landmark")]; have = {h["compound"] for h in progs if h.get("compound") is not None}
    checks.append({"what": "power centers by size, each has a compound", "ok": all(i in have for i in need), "value": [len(have), len(need)], "ha": r2(IF.area_ha)})
    n_st = sum(len(e.get("stairs", [])) for e in E)
    stats = {"archetype": kind, "n_houses": len(progs), "n_plots": len(plots), "n_city_blocks": len(blocks), "n_squares": len(squares),
             "streets_m": {rk: r2(sum(LineString(e["pts"]).length for e in E if e["rank"] == rk and e["kind"] not in ("road",))) for rk in ST.RANKS},
             "roads_m": r2(sum(LineString(e["pts"]).length for e in E if e["kind"] == "road")), "n_stairs": n_st, "n_gates": len(gates),
             "GSI": r2(gsi), "FSI": r2(fsi), "frac_streets": r2(st_share), "area_ha": r2(A_s / 1e4),
             "storeys": {"min": min((h["floors"] for h in progs), default=0), "max": max((h["floors"] for h in progs), default=0)},
             "kinds": {k: sum(1 for h in progs if h["kind"] == k) for k in sorted({h["kind"] for h in progs})},
             **({"landmarks": {k: sum(1 for P_l in landmarks if P_l["kind"] == k) for k in sorted({P_l["kind"] for P_l in landmarks})}} if landmarks else {}),
             "districts": {k: sum(1 for h in progs if h.get("district") == k) for k in INF.LABELS if any(h.get("district") == k for h in progs)},
             "power": [ce["kind"] for ce in IF.centres],
             "highest_m": r2(max((h["z0"] + h["h0"] + (h["floors"] - 1) * h["fh"] for h in progs), default=0) - (decks["z0"] if decks else float(np.median([h["z0"] for h in progs])) if progs else 0))}
    # ---- space scheme (for the passport): houses — inside, squares — outside; scene anchors
    cells = [{"id": "outside", "kind": "world", "poly": None, "z0": -1e4, "z1": None, "cls": "exterior"}]
    for s_ in squares:
        for k_, pp in enumerate(PL.parts(s_["poly"], 1.0)):
            z_ = float(height(*pp.centroid.coords[0])); cells.append({"id": f"area_{s_['kind']}_{k_}", "kind": s_["kind"], "poly": ring(pp), "z0": r2(z_), "z1": None, "cls": "exterior"})
    for h in progs:
        cells.append({"id": f"house_{h['id']}", "kind": h["kind"], "poly": h["F"], "z0": h["z0"], "z1": r2(h["z0"] + h["h0"] + (h["floors"] - 1) * h["fh"]), "cls": "interior"})
    anchors = []
    for s_ in squares:
        p = s_["poly"].representative_point(); anchors.append({"name": f"square {s_['kind']}", "cell": "outside", "p": [r2(p.x), r2(p.y)], "z": r2(float(height(p.x, p.y)))})
    for i, gp in enumerate(gates):
        d = np.asarray(gp) - c; anchors.append({"name": f"gate {i}", "cell": "outside", "p": gp, "z": r2(float(height(*gp))), "facing": [r2(-d[0]), r2(-d[1])]})
    mains = sorted([e for e in E if e["rank"] == "artery" and e["kind"] not in ("road", "bridge")], key=lambda e: -LineString(e["pts"]).length)[:3]
    if not mains:                                                                 # without arterial streets (a village, a castle courtyard) — the longest streets
        mains = sorted([e for e in E if e["rank"] == "street" and e["kind"] not in ("road", "bridge")], key=lambda e: -LineString(e["pts"]).length)[:3]
    for i, e in enumerate(mains):
        g = LineString(e["pts"]); m = g.interpolate(0.5, normalized=True); q = np.asarray(g.coords[-1]) - g.coords[0]
        anchors.append({"name": f"street {i}", "cell": "outside", "p": [r2(m.x), r2(m.y)], "z": r2(float(height(m.x, m.y))), "facing": [r2(q[0]), r2(q[1])], "tags": ["street"]})
    for i, ce in enumerate(IF.centres):                                           # power centers are frame anchors (compound: tower, yard, chambers)
        hs_c = [h for h in progs if h.get("compound") == i and h.get("deck") in (None, ce.get("deck"))]
        if not hs_c: continue
        Fc = unary_union([Polygon(h["F"]).buffer(0) for h in hs_c]); q = Fc.centroid; ztop = max(h["z0"] + h["h0"] + (h["floors"] - 1) * h["fh"] for h in hs_c)
        sz = math.sqrt(Fc.convex_hull.area); best = None                           # a street point near the compound — a camera from the street (hive tiers)
        for e in E:
            if e["kind"] in ("edge", "road", "wall"): continue
            g = LineString(e["pts"]); t_ = g.project(q); pt = g.interpolate(t_); d_ = pt.distance(q)
            if sz / 2 + 8.0 <= d_ <= sz / 2 + 70.0 and (best is None or d_ < best[0]): best = (d_, pt)
        anchors.append({"name": f"power {ce['kind']}", "cell": "outside", "p": [r2(q.x), r2(q.y)], "z": r2(min(h["z0"] for h in hs_c)), "top": r2(ztop),
                        "size": r2(sz), "deck": ce.get("deck"), "main": bool(ce.get("main")), **({"from_street": [r2(best[1].x), r2(best[1].y)]} if best else {}),
                        "tags": ["power", ce["kind"], "main" if ce.get("main") else "small"]})
    lab_xy = [(np.asarray(h["F"], float).mean(0), h["district"]) for h in progs if h.get("district") and h.get("deck") in (None, 0) and h.get("compound") is None]
    if lab_xy:                                                                     # districts are frame anchors: a street whose houses belong to this district
        HX = np.array([x for x, _ in lab_xy]); HL = np.array([l for _, l in lab_xy])
        for lab in ("trade", "craft", "nobility", "middle", "poor"):
            best = None
            for e in E:
                if e["kind"] in ("edge", "road", "bridge", "wall") or e["rank"] == "passage": continue
                g = LineString(e["pts"]); L_ = g.length
                if L_ < 18.0: continue
                m = np.asarray(g.interpolate(0.5, normalized=True).coords[0]); near = np.linalg.norm(HX - m, axis=1) < 16.0 + e["w"] / 2
                if near.sum() < 3: continue
                sc = (HL[near] == lab).mean() * min(1.0, L_ / 40.0)
                if best is None or sc > best[0]: best = (sc, m, np.asarray(g.coords[-1]) - g.coords[0])
            if best is not None and best[0] >= 0.5:
                m, f = best[1], best[2]
                anchors.append({"name": f"district {lab}", "cell": "outside", "p": [r2(m[0]), r2(m[1])], "z": r2(float(height(m[0], m[1]))),
                                "facing": [r2(f[0]), r2(f[1])], "frac": r2(best[0]), "tags": ["district", lab]})
    for P_l in landmarks:                                                         # landmarks are frame anchors (temple, theatre, stoa)
        if P_l["kind"] == "altar": continue
        anchors.append({"name": f"landmark {P_l['kind']}", "cell": "outside", "p": [r2(P_l["c"][0]), r2(P_l["c"][1])], "z": r2(P_l["z0"]),
                        "facing": [r2(P_l["u"][0]), r2(P_l["u"][1])], "size": r2(max(P_l.get("L", 10), P_l.get("W", 10))), "L": r2(P_l.get("L", 10)),
                        "W": r2(P_l.get("W", 10)), "tags": ["landmark", P_l["kind"]]})
        cells.append({"id": f"landmark_{len(cells)}", "kind": P_l["kind"], "poly": P_l["fp"], "z0": r2(P_l["z0"]), "z1": None, "cls": "interior"})
    spaces = {"cells": cells, "trans": [], "anchors": anchors, "graph": {"entrances": []}, "stats": {"n_cells": len(cells)}}
    Rj = {"type": "settlement", "kind": kind, "spec": {"kind": kind, "seed": seed, "size": r2(2 * R), **({"host": os.path.relpath(host_dir, out)} if host_dir else {})},
          "place": place, "site": {"centre": [r2(c[0]), r2(c[1])], "R": r2(R), "boundary": ring(Bd) if Bd is not None and Bd.geom_type == "Polygon" else polys(Bd),
                                   "angle": r2(math.degrees(ang))},
          "wall": ({"line": ring(wall), "t": 2.4, "h": 8.5} if wall is not None else None), "gates": gates,
          "streets": {"nodes": P_, "edges": E}, "squares": [{"kind": s_["kind"], "polys": polys(s_["poly"])} for s_ in squares],
          "sanctuary": ring(sanct) if sanct is not None else None, "shafts": shafts, "decks": decks, "bridges": bridges,
          "paving_file": "paving.npz", "paving_tris": n_tri, "water_bridges": wbridges, "water": polys(water),
          "plots": [{"id": p["id"], "poly": ring(p["poly"]), "fronts": p["fronts"], **({"influence": p["influence"], "district": p["district"]} if p.get("district") else {}),
                     **({"compound": int(p["compound"])} if p.get("compound") is not None else {})} for p in plots],
          "houses": progs, **({"landmarks": landmarks} if landmarks else {}),
          "influence": {"centres": IF.centres, "ha": r2(IF.area_ha), "entries": [[r2(x), r2(y)] for x, y in IF.entries], "yards": [ring(q_) for y_ in yards for q_ in PL.parts(y_, 20.0)],
                        "field": INF.FD.FILE, "knobs": IF.L, **({"wind": float(wind)} if wind is not None else {}), **({"center": centre} if centre is not None else {})},
          **({"acropolis": {"foot": ring(acro["foot"]), "top": ring(acro["top"]), "z": r2(acro["z_top"]), "wall": ring(PL.parts(acro["top"].buffer(-1.2).simplify(1.0), 1.0)[0]),
                            "gate": [r2(gate_acro[0][0]), r2(gate_acro[0][1])] if gate_acro else None}} if acro is not None else {}),
          "host": HP and {"folder": host_dir, "type": HP["type"], "outline": HP["outline"], "occupied": HP["occupied"]},
          "paving": A.get("pavement"), "logic": logic, "checks": checks, "stats": stats, "spaces": spaces, "out": out}
    json.dump(Rj, open(os.path.join(out, "ensemble.json"), "w"), ensure_ascii=False, default=float)
    draw(os.path.join(out, "plan.png"), G2, Rj, rib, roads, sq_geom, wall, f"{kind} · seed {seed}")
    DS.draw(os.path.join(out, "districts.png"), G2, Rj, rib, decks=decks["count"] if decks else None)
    IF.raster(os.path.join(out, INF.FD.FILE), c, R + 30.0, 2.0, z0=decks["z0"] if decks else 0.0, dz=decks["height"] if decks else 0.0)   # the field is for neighbors and for the build
    return Rj


# ---------------------------------------------------------------- plan drawing
HCOL = {"court_greek": (214, 196, 160), "domus": (196, 120, 92), "insula": (178, 98, 74), "burgage": (150, 112, 88), "izba": (128, 96, 66),
        "rural": (150, 120, 84), "shack": (120, 108, 96), "annex": (160, 130, 104), "shed": (110, 86, 60), "hub": (110, 110, 116), "back": (132, 104, 84)}


def draw(path, G, R, rib, roads, sqg, wall, title, px_max=1500):
    from PIL import Image, ImageDraw, ImageFont
    b = R["site"]; c = np.asarray(b["centre"]); half = b["R"] + 30
    x0, y0 = c - half; s = min(px_max / (2 * half), 3.0); W_ = H_ = int(2 * half * s)
    xs = x0 + (np.arange(W_) + 0.5) / s; ys = y0 + (np.arange(H_) + 0.5) / s
    X, Y = np.meshgrid(xs, ys[::-1]); Z = G.height(X, Y)
    gy, gx = np.gradient(Z, 1 / s); shade = np.clip(0.75 + 0.35 * (-gx * 0.7 + gy * 0.7) / np.sqrt(1 + gx * gx + gy * gy), 0.35, 1.1)
    zn = (Z - Z.min()) / max(np.ptp(Z), 1e-6)
    base = np.stack([150 + 50 * zn, 160 + 30 * zn, 120 + 30 * zn], -1) * shade[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img)
    T = lambda p: ((p[0] - x0) * s, (y0 + 2 * half - p[1]) * s)
    def fill(g, col, outline=None, w=1):
        mask = Image.new("L", img.size, 0); md = ImageDraw.Draw(mask)            # holes are cut out (blocks inside a ring of streets)
        for p in PL.parts(g, 0.01):
            md.polygon([T(q) for q in p.exterior.coords], fill=255)
            for h in p.interiors: md.polygon([T(q) for q in h.coords], fill=0)
        img.paste(Image.new("RGB", img.size, col), mask=mask)
        if outline is not None:
            for p in PL.parts(g, 0.01):
                d.line([T(q) for q in p.exterior.coords], fill=outline, width=w)
                for h in p.interiors: d.line([T(q) for q in h.coords], fill=outline, width=w)
    fill(roads, (196, 184, 160)); fill(rib, (222, 212, 190)); fill(sqg, (236, 228, 206), (150, 140, 120))
    if R.get("sanctuary"): d.polygon([T(q) for q in R["sanctuary"]], fill=(240, 236, 220), outline=(120, 100, 80), width=2)
    for sh in R.get("shafts") or []: d.polygon([T(q) for q in sh], fill=(40, 40, 46), outline=(20, 20, 20), width=2)
    for e in R["streets"]["edges"]:
        for s0, s1 in e.get("stairs", []):
            g = LineString(e["pts"]); sub = shapely.ops.substring(g, s0, s1)
            for k in np.arange(0, sub.length, 1.2):
                p = np.asarray(sub.interpolate(k).coords[0]); q = np.asarray(sub.interpolate(min(sub.length, k + 0.1)).coords[0]); t_ = q - p
                t_ = t_ / max(np.linalg.norm(t_), 1e-9); n_ = np.array([-t_[1], t_[0]]) * e["w"] / 2
                d.line([T(p - n_), T(p + n_)], fill=(150, 130, 100), width=1)
    if R.get("host"):
        for rr in R["host"].get("occupied") or R["host"]["outline"]: d.polygon([T(q) for q in rr], fill=(92, 88, 84), outline=(30, 30, 30), width=1)
        for rr in R["host"]["outline"]: d.line([T(q) for q in rr], fill=(30, 30, 30), width=2)
    for p in R["plots"]: d.polygon([T(q) for q in p["poly"]], outline=(150, 150, 120))
    for h in R["houses"]:
        if h.get("deck") not in (None, 0): continue
        col = HCOL.get(h["kind"], (140, 120, 100)); d.polygon([T(q) for q in h["F"]], fill=col, outline=(60, 50, 40))
        for ck in ("court", "court2"):
            if h.get(ck): d.polygon([T(q) for q in h[ck]], fill=(214, 220, 190), outline=(90, 80, 60))
    if R.get("acropolis"):
        Ac = R["acropolis"]; d.line([T(q) for q in Ac["top"] + Ac["top"][:1]], fill=(120, 96, 70), width=2)
    for P_l in R.get("landmarks") or []:                                          # landmarks: outline in marble, the cavea — in arcs of rows
        d.polygon([T(q) for q in P_l["fp"]], fill=(244, 240, 228), outline=(70, 60, 50))
        if P_l["kind"] == "theatre":
            cc_ = np.asarray(P_l["c"]); th_ = math.atan2(P_l["u"][1], P_l["u"][0]); sp_ = math.radians(P_l["T"]["span"])
            for r0_, r1_, _, _ in P_l["rows"][::3]:
                d.line([T(cc_ + r1_ * np.array([math.cos(t), math.sin(t)])) for t in np.linspace(th_ - sp_ / 2, th_ + sp_ / 2, 30)], fill=(150, 140, 120), width=1)
    if wall is not None: d.line([T(q) for q in list(wall.exterior.coords)], fill=(110, 40, 30), width=max(2, int(2.4 * s)))
    for gp in R["gates"]: x, y = T(gp); d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(240, 200, 60), outline=(60, 40, 0), width=2)
    try: f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 22); f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 16)
    except OSError: f = f2 = ImageFont.load_default()
    st = R["stats"]
    d.rectangle([0, 0, W_, 58], fill=(255, 255, 255))
    d.text((10, 4), f"{title} · {st['n_houses']} houses · {st['area_ha']} ha", fill=(0, 0, 0), font=f)
    d.text((10, 32), f"GSI {st['GSI']}  FSI {st['FSI']}  streets {st['frac_streets']}  stairs {st['n_stairs']}  storeys {st['storeys']['min']}–{st['storeys']['max']}  "
                     f"checks {sum(c['ok'] for c in R['checks'])}/{len(R['checks'])}", fill=(40, 40, 40), font=f2)
    L50 = 50 * s; d.line([(W_ - 20 - L50, H_ - 20), (W_ - 20, H_ - 20)], fill=(0, 0, 0), width=3); d.text((W_ - 20 - L50, H_ - 44), "50 m", fill=(0, 0, 0), font=f2)
    img.save(path)


def main():
    a = sys.argv[1:]
    if not a: raise SystemExit(__doc__)
    kw = dict(x.split("=", 1) for x in a[1:] if "=" in x)
    for k in ("seed",):
        if k in kw: kw[k] = int(kw[k])
    if "size" in kw: kw["size"] = float(kw["size"])
    R = plan(a[0], **kw)
    print("__RESULT__", json.dumps({"out": R["out"], "sec": r2(time.time() - T_START), **{k: v for k, v in R["stats"].items() if k not in ("streets_m",)},
                                    "non_passing": [c["what"] for c in R["checks"] if not c["ok"]]}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
