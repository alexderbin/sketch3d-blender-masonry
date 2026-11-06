"""Density, excavation, settling-in: three cases on the same world (strata.World), the relate rule and the walkability
check (walk.py).
  kowloon     — a self-built block: towers of 8–14 storeys wall to wall (shared walls are blank), passages of 1–2 m and
                light wells, bridges between the towers over the passages (a second network of routes), shacks and tanks
                on the roofs (Kowloon);
  underground — an underground town beneath a village: vaulted chambers on levels in the rock, tunnels, sloped passages
                between levels, a ventilation shaft down to the bottom, a well with water, millstone doors at the
                tunnels (Derinkuyu);
  arena       — an amphitheatre that became a town: the cavea is a slope turned inside out (ring lanes at different
                elevations), houses along the arena rings and on the steps, dwellings in the arches of the outer wall,
                towers on the ring, a chapel in the center (Arles)."""
import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box as sbox

from components.buildings.levels import geom as G
from components.buildings.levels import strata as S

OLD = (1.06, 0.90, 0.78); SELF = (1.0, 0.97, 0.92)


# ---------------------------------------------------------------- Kowloon
def kowloon(seed=8, size=(64.0, 44.0), plot=(5.0, 9.0), alleys=3, alley=(1.2, 2.0), wells=3, floors=(8, 14), fh=2.9):
    rng = np.random.default_rng(seed); Lx, Ly = size
    W = S.World((0.0, 0.0, Lx, Ly), [lambda X, Y: 0.0 * np.asarray(X, float)])
    xs = sorted(float(rng.uniform(10, Lx - 10)) for _ in range(alleys)); ys = sorted(float(rng.uniform(8, Ly - 8)) for _ in range(alleys - 1))
    cuts = [sbox(x - w / 2, -1, x + w / 2, Ly + 1) for x, w in zip(xs, rng.uniform(*alley, alleys))]
    cuts += [sbox(-1, y - w / 2, Lx + 1, y + w / 2) for y, w in zip(ys, rng.uniform(*alley, alleys))]
    for _ in range(wells):
        cx, cy = float(rng.uniform(8, Lx - 8)), float(rng.uniform(8, Ly - 8)); cuts.append(sbox(cx - 2.2, cy - 2.2, cx + 2.2, cy + 2.2))
    free = shapely.unary_union(cuts)
    nid = 0; x = 0.0
    while x < Lx - 2:                                                       # plots on a grid; a tower is a plot minus the passages
        w = min(float(rng.uniform(*plot)), Lx - x); y = -6.0
        while y < Ly - 2:
            d = min(float(rng.uniform(*plot)), Ly - y)
            cell = sbox(x, y, x + w, y + d).difference(free)
            for part in G.polys(cell):
                if part.area < 12: continue
                b0 = part.bounds
                if (b0[2] - b0[0]) < 2.5 or (b0[3] - b0[1]) < 2.5: continue
                fl = int(rng.integers(*floors)); fl = min(fl, 14)              # the ceiling is 14 storeys (Kai Tak airport)
                b = S.building(b0[0], b0[2], b0[1], b0[3], 0.15, fl, fh=fh, t=0.3, roof="terrace", tint=SELF,
                               tone=float(rng.uniform(0.92, 1.06)), id=nid, bay=2.6)
                W.buildings.append(b); nid += 1
            y += d
        x += w
    for b in W.buildings: S.relate(W, b)
    # bridges between towers over the passages — on the storeys where there is a tower on both sides
    # (a second network of routes)
    for c in cuts[:2 * alleys - 1]:
        bx = c.bounds; along_y = (bx[3] - bx[1]) > (bx[2] - bx[0])
        for t in np.arange(4.0, (Ly if along_y else Lx) - 4, float(rng.uniform(7, 12))):
            k = int(rng.integers(3, 11)); z = 0.15 + k * fh
            if along_y: P = sbox(bx[0] - 0.4, t, bx[2] + 0.4, t + 3.0); probe = ((bx[0] - 1.0, t + 1.5), (bx[2] + 1.0, t + 1.5))
            else: P = sbox(t, bx[1] - 0.4, t + 3.0, bx[3] + 0.4); probe = ((t + 1.5, bx[1] - 1.0), (t + 1.5, bx[3] + 1.0))   # wider than the span step
            if all(W.solid_at(px, py, z + 0.5, z + 2.0) for px, py in probe):
                W.decks.append({"poly": P, "z": z, "t": 0.35, "name": "bridge", "tint": SELF})
    for b in W.buildings:                                                   # doors to the bridges — the second pass of relate
        S.relate(W, b)
        top = b["z_top"]
        for _ in range(int(rng.integers(1, 4))):                             # shacks and tanks on the terrace roofs
            w = float(rng.uniform(1.6, 3.0)); x0 = float(rng.uniform(b["x0"] + 0.4, max(b["x0"] + 0.5, b["x1"] - w - 0.4)))
            y0 = float(rng.uniform(b["y0"] + 0.4, max(b["y0"] + 0.5, b["y1"] - w - 0.4)))
            tank = rng.random() < 0.4
            W.solids.append({"type": "prism", "poly": Point(x0 + w / 2, y0 + w / 2).buffer(w / 2.6, 8) if tank else sbox(x0, y0, x0 + w, y0 + w),
                             "z0": top, "z1": top + (1.6 if tank else 2.3), "cls": "wall", "tint": SELF, "tone": 0.9})
    party = 0
    for b in W.buildings:                                                   # an inner tower with no exit — a door into a neighbor (Kowloon:
        if b["doors"]: continue                                              # "walk through the town without going outside")
        nb = max((o for o in W.buildings if o is not b), key=lambda o: o["poly"].buffer(0.3).intersection(b["poly"].buffer(0.3)).area)
        k = int(rng.integers(0, min(len(b["floors"]), len(nb["floors"])))); z = b["floors"][k]["z"]
        pa = b["poly"].centroid; pb = nb["poly"].centroid
        b["doors"].append({"floor": k, "side": "party", "u": 0.0, "to": nb["id"]}); party += 1
        W.links.append({"kind": "party", "floors": ((b["id"], k), (nb["id"], k))})
    W.origin = (xs[0], Ly - 0.5)
    from components.buildings.levels import walk as WK
    WK.prune(W)
    W.notes.append(f"inner towers with a door into a neighbor {party}; towers {len(W.buildings)}, bridges over passages {sum(1 for d in W.decks if d.get('name') == 'bridge')}, "
                   f"passages {2 * alleys - 1}, wells {wells}")
    return W


# ---------------------------------------------------------------- an underground town
def underground(seed=12, size=(56.0, 40.0), levels=7, step=6.0, rooms=(5, 8), room=(4.0, 9.0), cover=2.0):
    rng = np.random.default_rng(seed); Lx, Ly = size; yc = 12.0             # the section plane — through the middle of the complex
    W = S.World((0.0, yc, Lx, Ly), [lambda X, Y: 0.4 * np.sin(np.asarray(X, float) / 9.0) + 0.0 * np.asarray(Y, float)])
    W.layer_tints = [(1.0, 0.95, 0.88)]
    nid = 0
    for x0, y0, w, d in ((4, 6, 9, 8), (16, 3, 8, 7), (30, 5, 10, 8), (44, 8, 8, 9), (8, 24, 9, 9), (24, 26, 8, 8), (38, 25, 10, 9)):
        b = S.building(x0, x0 + w, y0, y0 + d, 0.55, int(rng.integers(1, 3)), fh=3.0, t=0.6, roof="terrace", tint=OLD, id=nid, bay=3.0)
        W.buildings.append(b); nid += 1
    placed = []                                                              # chambers: rectangles with a rock cover between them
    for k in range(levels):
        z = -(k + 1) * step; n = int(rng.integers(*rooms)); tries = 0; lev = []
        while len(lev) < n and tries < 400:
            tries += 1; w, d = float(rng.uniform(*room)), float(rng.uniform(*room)) * 0.8
            x = float(rng.uniform(2, Lx - w - 2))
            y = yc - d / 2 + float(rng.uniform(-1.5, 1.5)) if rng.random() < 0.55 else float(rng.uniform(yc - 6.0, Ly - d - 2))   # half of them — across the section
            q = sbox(x, y, x + w, y + d)
            if any(q.buffer(cover).intersects(p) for p, _ in lev): continue
            lev.append((q, z))
        placed.append(lev)
        for q, zz in lev:
            bx = q.bounds; axis = "x" if bx[2] - bx[0] >= bx[3] - bx[1] else "y"
            v0, v1, a0, a1 = (bx[1], bx[3], bx[0], bx[2]) if axis == "x" else (bx[0], bx[2], bx[1], bx[3])
            zs = zz + 2.0
            W.voids.append({"kind": "room", "axis": axis, "a0": a0, "a1": a1, "v0": v0, "v1": v1, "z0": zz, "zs": zs, "arch": "round"})
            W.decks.append({"poly": q, "z": zz, "t": 0.0, "draw": False, "name": f"level {k + 1}"})
        for i in range(1, len(lev)):                                          # level tunnels: to the nearest earlier chamber, L-shaped
            (q, zz) = lev[i]; (p, _) = min(lev[:i], key=lambda t: t[0].centroid.distance(q.centroid))
            _tunnel(W, (q.centroid.x, q.centroid.y), (p.centroid.x, p.centroid.y), zz, rng)
    for k in range(levels):                                                  # sloped passages between levels
        up = placed[k - 1] if k else [(sbox(19.0, 17.0, 21.0, 19.0), 0.4)]   # the entrance is from the village yard
        lo = placed[k]
        (a, za), (b_, zb) = min(((u, l) for u in up for l in lo), key=lambda t: t[0][0].centroid.distance(t[1][0].centroid))
        _ramp(W, (a.centroid.x, a.centroid.y, za), (b_.centroid.x, b_.centroid.y, zb), rng)
    sx, sy = Lx * 0.52, yc + 0.4                                              # the ventilation shaft and the well down to the bottom — in the section
    W.voids.append({"kind": "shaft", "axis": "x", "a0": sx - 0.8, "a1": sx + 0.8, "v0": sy - 0.8, "v1": sy + 0.8,
                    "z0": -levels * step - 3.0, "zs": 1.0, "arch": "flat"})
    wx, wy = Lx * 0.3, yc + 0.6
    W.voids.append({"kind": "well", "axis": "x", "a0": wx - 1.2, "a1": wx + 1.2, "v0": wy - 1.2, "v1": wy + 1.2,
                    "z0": -levels * step - 4.0, "zs": 1.0, "arch": "flat"})
    W.solids.append({"type": "prism", "poly": sbox(wx - 1.15, wy - 1.15, wx + 1.15, wy + 1.15), "z0": -levels * step - 4.0,
                     "z1": -levels * step - 2.0, "cls": "water", "tone": 0.8})
    for b in W.buildings: S.relate(W, b)
    W.origin = (19.5, 17.5)
    W.notes.append(f"levels {levels} down to {levels * step:.0f} m, chambers {sum(len(l) for l in placed)}")
    return W


def _tunnel(W, a, b, z, rng, w=1.3, h=2.0):
    """An L-shaped tunnel: first along x, then along y; the vault is flat and low; a millstone door in a niche at the start."""
    (x0, y0), (x1, y1) = a, b
    if abs(x1 - x0) > 0.5:
        W.voids.append({"kind": "tunnel", "axis": "x", "a0": min(x0, x1), "a1": max(x0, x1), "v0": y0 - w / 2, "v1": y0 + w / 2,
                        "z0": z, "zs": z + h, "arch": "round" if rng.random() < 0.5 else "flat"})
    if abs(y1 - y0) > 0.5:
        W.voids.append({"kind": "tunnel", "axis": "y", "a0": min(y0, y1), "a1": max(y0, y1), "v0": x1 - w / 2, "v1": x1 + w / 2,
                        "z0": z, "zs": z + h, "arch": "flat"})
    W.links.append({"kind": "tunnel", "a": (x0, y0, z), "b": (x1, y1, z)})
    if abs(x1 - x0) > 3.0:                                                   # millstone door: a disc standing in a niche beside the tunnel
        xd = x0 + (1.6 if x1 > x0 else -1.6)
        W.solids.append({"type": "extrude", "prof": Point(y0 + w / 2 + 0.35, z + 0.85).buffer(0.85, 12), "axis": "x",
                         "a0": xd - 0.25, "a1": xd + 0.25, "cls": "stair", "tone": 0.85})


def _ramp(W, a, b, rng, w=1.3, h=2.1):
    """A sloped passage between levels: a void strip sloping along x, then horizontal along y; the walkability link is between
    the chamber centers (the geometry ends may shift so that the slope is not steeper)."""
    W.links.append({"kind": "ramp", "a": tuple(a), "b": tuple(b)})
    (x0, y0, z0), (x1, y1, z1) = a, b
    if abs(x1 - x0) < 4.0: x1 = x0 + (4.0 if x1 >= x0 else -4.0)
    xa, xb = sorted((x0, x1)); za, zb = (z0, z1) if x0 <= x1 else (z1, z0)
    prof = Polygon([(xa, za), (xb, zb), (xb, zb + h), (xa, za + h)])
    W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": y0 - w / 2, "a1": y0 + w / 2, "cls": "void", "void": True})
    W.voids.append({"kind": "landing", "axis": "y", "a0": min(y0, y1), "a1": max(y0, y1) + 0.01, "v0": x1 - w / 2, "v1": x1 + w / 2,
                    "z0": z1, "zs": z1 + h, "arch": "flat"})


# ---------------------------------------------------------------- an amphitheatre that became a town
def arena(seed=9, a=68.0, b=53.5, ia=34.5, ib=20.0, height=21.0, lanes=3, plot=(5.0, 8.0)):
    """The amphitheatre ellipse (semi-axes a, b), the arena (ia, ib) — the dimensions of Arles (136 × 107 m, arena 69 × 40); the
    cavea is a slope from the arena (0) to the wall top (height − 3)."""
    rng = np.random.default_rng(seed)
    R = lambda X, Y: np.sqrt((np.asarray(X, float) / a) ** 2 + (np.asarray(Y, float) / b) ** 2)
    r_in = ia / a                                                             # the arena's normalized radius (along the major axis)
    LW = 0.02                                                                 # half-width of a ring lane (≈ 1.4 m at Arles: a 2–3 m passage)

    def ground(X, Y):
        r = R(X, Y); ri = np.sqrt((np.asarray(X, float) / ia) ** 2 + (np.asarray(Y, float) / ib) ** 2)
        t = np.clip((r - r_in) / (1.0 - r_in), 0, 1)
        z = np.where(ri <= 1.0, 0.0, (height - 3.0) * t)
        for k in range(1, lanes + 1):                                         # ring lanes — level strips in the cavea
            rk = r_in + (1.0 - r_in) * k / (lanes + 1); z = np.where(np.abs(r - rk) < LW, (height - 3.0) * (rk - r_in) / (1.0 - r_in), z)
        return np.where(r > 1.0, 0.0, z)
    W = S.World((-a - 8.0, 0.0, a + 8.0, b + 8.0), [ground])                  # a half: the section along the major axis
    ell = lambda s: Polygon([(a * s * math.cos(t), b * s * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 96, endpoint=False)])
    W.terrain_holes = [ell(1.0).difference(ell(r_in * 0.999 * (ia / a) / r_in))]   # the cavea — in stepped rings, not a mesh
    nr = 26                                                                   # seat rows: a ring between ellipses, the height is the ground of the row
    for k in range(nr):
        r0 = r_in + (1.0 - r_in) * k / nr; r1 = r_in + (1.0 - r_in) * (k + 1) / nr; rm = (r0 + r1) / 2
        zr = float(ground(np.array([a * rm]), np.array([0.0]))[0])
        ring = ell(r1).difference(ell(r0)).difference(Polygon([(-1e3, -1e3), (1e3, -1e3), (1e3, 0.0), (-1e3, 0.0)]))
        W.solids.append({"type": "prism", "poly": ring, "z0": -0.3, "z1": max(zr, 0.05), "cls": "ground", "tone": 1.0, "cavea": True})
    wall = ell(1.06).difference(ell(1.0))                                     # outer wall: a ring with arch openings by tiers
    W.solids.append({"type": "prism", "poly": wall, "z0": -0.5, "z1": height + 2.0, "cls": "wall", "tint": OLD, "cut_ground": True})
    nid = 0
    for ang in (0.35, 1.25, 2.1, 2.85):                                       # towers on the ring
        cx, cy = a * 1.03 * math.cos(ang), b * 1.03 * math.sin(ang)
        W.buildings.append(S.building(cx - 3.2, cx + 3.2, cy - 3.2, cy + 3.2, 0.15, 9, fh=3.4, t=0.8, roof="gable", pitch=50, ridge="x",
                                      tint=OLD, id=nid, bay=3.0, kind="tower")); nid += 1
    zlane = lambda k: 0.0 if k == 0 else (height - 3.0) * k / (lanes + 1)          # elevation of lane k (0 — the arena)
    VOMS = (0.6, 1.57, 2.55)                                                  # angles of the radial passages (vomitoria)
    rlane = lambda k: r_in + (1.0 - r_in) * k / (lanes + 1)
    lane_rings = [ell(rlane(k) + LW).difference(ell(rlane(k) - LW)) for k in range(1, lanes + 1)]   # lane rings — do not build on them
    def ring(ea, eb, s0, s1, zf, k):
        """Houses along the ring between the ellipses (ea·s0, eb·s0) and (ea·s1, eb·s1): tangentially, with the facade toward the
        center; clear of the passages."""
        nonlocal nid
        sm = (s0 + s1) / 2; ang = 0.12
        while ang < math.pi - 0.08:
            w = float(rng.uniform(*plot)); arc = sm * math.hypot(ea * math.sin(ang), eb * math.cos(ang))   # local arc length per radian
            da = (w + 0.35) / max(arc, 1e-6); am = ang + da / 2
            cx, cy = ea * sm * math.cos(am), eb * sm * math.sin(am)
            band = (s1 - s0) * math.hypot(ea * math.cos(am), eb * math.sin(am))   # the strip width here (along the radius)
            d = max(2.8, min(7.0, band * 0.85))
            if rng.random() < 0.9 and cy > d / 2 + 0.6 and not any(abs(am - v) < da / 2 + 0.06 for v in VOMS):
                nrm = np.array([math.cos(am) / ea, math.sin(am) / eb]); nrm /= np.linalg.norm(nrm)   # ellipse normal
                th = math.atan2(nrm[1], nrm[0]) - math.pi / 2
                up = int(math.ceil((zlane(min(k + 1, lanes)) - zf) / 3.0)) if 0 < k < lanes else 0
                bb = S.building(cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2, zf + 0.15, max(2, up + int(rng.integers(1, 3))), fh=3.0,
                                t=0.4, roof="gable", pitch=35, ridge="x", tint=SELF, id=nid, bay=2.6, pad=zf, angle=th)
                if not any(bb["poly"].intersection(lr).area > 1.5 for lr in lane_rings): W.buildings.append(bb); nid += 1
            ang += da
    for s0, s1 in ((0.42, 0.64), (0.68, 0.92)):                               # arena: two rings around the square with the chapel
        ring(ia, ib, s0, s1, 0.0, 0)
    for k in range(1, lanes + 1):                                             # cavea: the strip above lane k
        ring(a, b, rlane(k) + LW + 0.004, (rlane(k + 1) - LW - 0.004) if k < lanes else 0.95, zlane(k), k)
    W.buildings = [b for i, b in enumerate(W.buildings) if not any(o["poly"].intersects(b["poly"].buffer(-0.3)) for o in W.buildings[:i])]
    pads = []
    for bb in W.buildings:                                                    # the house pad at the level of its lane (the house is cut into the cavea)
        if bb.get("pad") is not None:
            W.digs.append((bb["poly"].buffer(1.6), bb["pad"]))
            if bb["pad"] > 0.1: pads.append((bb["poly"].buffer(1.6), bb["pad"]))
    if pads:                                                                  # the stepped rings go around the pads; a pad is a floor
        cut = shapely.unary_union([q for q, _ in pads])
        for sol in W.solids:
            if sol.get("cavea"): sol["poly"] = sol["poly"].difference(cut)
        for q, z in pads:
            W.solids.append({"type": "prism", "poly": q.intersection(ell(1.0)), "z0": -0.3, "z1": z, "cls": "street", "tone": 1.0})
    W.buildings.append(S.building(-5.0, 5.0, 2.0, 12.0, 0.15, 1, fh=7.0, t=0.7, roof="gable", pitch=40, ridge="y", tint=OLD, id=999,
                                  kind="temple", bay=3.0))                    # a chapel in the center of the arena
    for i, bb in enumerate(W.buildings): bb["id"] = i
    for k in range(lanes):                                                    # radial stairs along the old passages (vomitoria):
        for ang in VOMS:                                                      # arena → lane 1 → … → the top lane
            r0 = r_in + (1.0 - r_in) * k / (lanes + 1); r1 = r_in + (1.0 - r_in) * (k + 1) / (lanes + 1)
            p0 = (a * r0 * math.cos(ang), b * r0 * math.sin(ang)); p1 = (a * r1 * math.cos(ang), b * r1 * math.sin(ang))
            z0 = float(ground(np.array([p0[0]]), np.array([p0[1]]))[0]); z1 = float(ground(np.array([p1[0]]), np.array([p1[1]]))[0])
            W.links.append({"kind": "stair", "a": (*p0, z0), "b": (*p1, z1)})
    W.links.append({"kind": "gate", "a": (0.0, b + 3.0, 0.0), "b": (0.0, b * rlane(lanes), zlane(lanes))})   # a gate in the wall — onto the top lane
    for bb in W.buildings: S.relate(W, bb)
    W.origin = (0.0, b + 5.0)
    yl = b * rlane(1); W.ring_cam = {"loc": (0.0, yl, zlane(1) + 1.7), "tgt": (a * 0.55, b * 0.45, zlane(1) + 3.0), "lens": 20}   # along a ring lane
    from components.buildings.levels import walk as WK
    WK.prune(W)
    W.notes.append(f"houses {len(W.buildings)}, ring lanes {lanes}, towers 4")
    return W
