"""Verification cases of the levels: a world from knobs → the relate rule → checks → a clay mesh (model.npz), a section
without Blender (slice.png), a plan (plan.json). The clay render is bl_clay.py over the same folder.

.venv/bin/python -m components.buildings.levels.cases <case> [key=value …] [out=out/levels/<case>]
Cases: slope_house (a house on a level change), slope_row (a row on a slope, stair lanes)."""
import json
import math
import os
import sys

import numpy as np
from shapely.geometry import Polygon, box as sbox

from components.buildings.levels import geom as G
from components.buildings.levels import strata as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _views(kinds=("front", "front_r", "elev"), fit=None, lens=35):
    """Clay cameras by direction (bl_clay.camera fits the distance to the extent): three-quarter views of the section face y0
    from the left and right, the section (ortho), a top view. The section is the face y0, the camera is on the −y side."""
    V = {"front": {"az": -125, "el": 24, "lens": lens}, "front_r": {"az": -58, "el": 20, "lens": lens},
         "elev": {"az": -90, "el": 0, "ortho": True}, "top": {"az": -90, "el": 89, "ortho": True},
         "high": {"az": -120, "el": 42, "lens": lens}, "low": {"az": -100, "el": 6, "lens": 24}}
    out = {k: dict(V[k]) for k in kinds}
    if fit:
        for v in out.values(): v["fit"] = list(fit)
    return out


# ---------------------------------------------------------------- a house on a level change
def slope_house(H=12.8, depth=12.0, width=12.0, gap=2.6, yard=6.4, fh=3.2, cellars=1, caves=1, street_vault=True,
                over=2, roof="gable", seed=1, rock_slope=0.6):
    """One house between the lower (z 0) and upper (z H) streets: storeys from the lower street up to H + over storeys; a back
    yard at elevation yard; the retaining wall of the upper street; a bridge from the upper street to the storey at its level;
    cellars under the house and into the rock; a storeroom under the street."""
    x0, x1 = 0.5, 0.5 + depth; xr = x1 + gap; y0, y1 = -width / 2, width / 2
    ramp = xr + 0.0
    Xb = (-7.0, xr + 13.0); Yb = (-0.6, y1 + 5.0)                              # block: the section face y0 passes through the house

    def nat(X, Y):
        X = np.asarray(X, float)
        return np.where(X < 0, 0.0, np.where(X < xr, H * np.clip(X / xr, 0, 1) ** 1.15, np.where(X < xr + 7.0, H, H + (X - xr - 7.0) * rock_slope)))

    def top(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float); z = nat(X, Y)
        in_yard = (X >= x1 - 0.01) & (X < xr) & (Y > y0) & (Y < y1)
        in_plot = (X >= x0) & (X < x1) & (Y > y0) & (Y < y1)
        z = np.where(in_yard, yard, z); return np.where(in_plot, 0.0, z)

    W = S.World((Xb[0], Yb[0], Xb[1], Yb[1]), [top])
    floors = int(math.ceil((H + over * fh) / fh))
    b = S.building(x0, x1, y0, y1, 0.15, floors, fh=fh, cellars=cellars, ch=3.6, roof=roof, caves=caves, id=0, ridge="y", vault="y")
    W.buildings.append(b)
    W.streets.append({"poly": sbox(Xb[0], Yb[0], 0.0, Yb[1]), "z": 0.0, "kind": "street", "t": 0.3})
    W.streets.append({"poly": sbox(xr + 0.6, Yb[0], xr + 7.0, Yb[1]), "z": H, "kind": "street", "t": 0.3})
    W.streets.append({"poly": sbox(x1, y0, xr, y1), "z": yard, "kind": "yard", "t": 0.2, "draw": True, "cls": "street"})
    # retaining walls: of the upper street (with a parapet) and of the yard on the sides
    rw = 0.7
    door_y = (-1.5, 1.5) if street_vault else (0.0, 0.0); vz0, vzs = yard, yard + 2.0
    for ya, yb in ((Yb[0], door_y[0]), (door_y[1], Yb[1])):                     # the retaining wall of the upper street with the storeroom opening
        if yb - ya > 0.05: W.solids.append({"type": "prism", "poly": sbox(xr, ya, xr + rw, yb), "z0": yard - 0.5, "z1": H + 1.0, "cls": "retain"})
    if street_vault:
        W.solids.append({"type": "prism", "poly": sbox(xr, door_y[0], xr + rw, door_y[1]), "z0": vzs + 1.6, "z1": H + 1.0, "cls": "retain"})
        W.voids.append({"kind": "street_vault", "axis": "x", "a0": xr + 0.01, "a1": xr + 5.5, "v0": door_y[0], "v1": door_y[1],
                        "z0": vz0, "zs": vzs, "arch": "round", "of": 0})
    W.solids.append({"type": "prism", "poly": sbox(x1, y1, xr, y1 + rw), "z0": yard - 0.5, "z1": float(nat(np.array([xr]), 0)[0]) + 0.8, "cls": "retain"})
    S.relate(W, b)
    for br in b["bridges"]: W.solids.append(S.bridge_solid(br))
    S.caves(W, b, caves)
    W.notes.append(f"storeys {floors}, cellars {cellars}, passages into the rock {sum(v['kind'] == 'cave' for v in W.voids)}")
    return W, _views(("front", "front_r", "elev"))


# ---------------------------------------------------------------- a row on a slope
def slope_row(H=16.0, depth=11.0, n=5, width=(6.0, 9.0), close=1.8, gap=2.4, fh=3.2, over=(1, 3), seed=2, roof="gable"):
    """A row of houses between the lower and upper streets (Edinburgh, Genoa): each has its own back yard at its own level, an
    entrance from below and a bridge from above; between some of them — a stair lane from the lower street to the upper one
    (close)."""
    rng = np.random.default_rng(seed); x0, x1 = 0.5, 0.5 + depth; xr = x1 + gap
    ys = []; y = 0.0
    for i in range(n):
        w = float(rng.uniform(*width)); ys.append((y, y + w)); y += w
        if i < n - 1 and rng.random() < 0.5: ys.append(("close", y, y + close)); y += close
    Ytot = y; Xb = (-7.0, xr + 12.0); Yb = ((ys[0][0] + ys[0][1]) / 2, Ytot + 1.0)   # the section — through the middle of the first house
    yards = {}

    def nat(X, Y):
        X = np.asarray(X, float)
        return np.where(X < 0, 0.0, np.where(X < xr, H * np.clip(X / xr, 0, 1) ** 1.15, np.where(X < xr + 7.0, H, H + (X - xr - 7.0) * 0.6)))

    W = S.World((Xb[0], Yb[0], Xb[1], Yb[1]), [])
    houses = [s for s in ys if s[0] != "close"]; closes = [s for s in ys if s[0] == "close"]
    for i, (ya, yb) in enumerate(houses):
        yards[i] = (ya, yb, float(rng.choice([2, 3, 4])) * fh * 0 + float(rng.integers(1, 4)) * fh + 0.0)

    def top(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float); z = nat(X, Y)
        for i, (ya, yb, yz) in yards.items():
            z = np.where((X >= x1 - 0.01) & (X < xr) & (Y > ya) & (Y < yb), yz, z)
            z = np.where((X >= x0) & (X < x1) & (Y > ya) & (Y < yb), 0.0, z)
        for _, ya, yb in closes:                                               # lane: a steady descent down the slope from street to street
            z = np.where((X >= 0) & (X < xr + 0.7) & (Y > ya) & (Y < yb), np.clip(X / (xr + 0.7), 0, 1) * H, z)
        return z

    W.layers = [top]
    for i, (ya, yb) in enumerate(houses):
        fl = int(math.ceil(H / fh)) + int(rng.integers(over[0], over[1] + 1))
        b = S.building(x0, x1, ya + 0.0, yb - 0.0, 0.15, fl, fh=fh, cellars=int(rng.integers(0, 2)), roof=roof, id=i,
                       tone=float(rng.uniform(0.93, 1.03)), ridge="x")
        W.buildings.append(b)
    W.streets.append({"poly": sbox(Xb[0], Yb[0], 0.0, Yb[1]), "z": 0.0, "t": 0.3})
    W.streets.append({"poly": sbox(xr + 0.7, Yb[0], xr + 7.0, Yb[1]), "z": H, "t": 0.3})
    rw = 0.7
    W.solids.append({"type": "prism", "poly": sbox(xr, Yb[0], xr + rw, Yb[1]), "z0": 0.0, "z1": H, "cls": "retain"})
    for i, (ya, yb, yz) in yards.items():
        W.streets.append({"poly": sbox(x1, ya, xr, yb), "z": yz, "kind": "yard", "t": 0.2})
    for _, ya, yb in closes:
        S.stairs(W, 0.0, xr + 0.7, (ya + yb) / 2, 0.0, H, width=yb - ya - 0.2)
        W.streets.append({"poly": sbox(0.0, ya, xr + 0.7, yb), "z": -99.0, "draw": False})
    for b in W.buildings: S.relate(W, b)
    gaps = []
    for b in W.buildings:
        for br in b["bridges"]: W.solids.append(S.bridge_solid(br, width=1.6)); gaps.append(br["a"][1])
    cur = Yb[0]                                                                  # the parapet of the upper street — with openings to the bridges
    for gy in sorted(gaps) + [Yb[1] + 1.0]:
        if gy - 0.9 - cur > 0.05: W.solids.append({"type": "prism", "poly": sbox(xr, cur, xr + rw, min(gy - 0.9, Yb[1])), "z0": H, "z1": H + 1.0, "cls": "retain"})
        cur = gy + 0.9
    V = _views(("front", "high", "low"), lens=40); V["back"] = {"az": 35, "el": 28, "lens": 40}
    return W, V


def strata_town(**kw):
    """A town over a town (epochs.py): three eras, fills, fates, a church on a temple; era snapshots — subfolders e<k>."""
    from components.buildings.levels import epochs as EP
    W = EP.strata_town(**kw)
    return W, _views(("front", "front_r", "elev", "high"), lens=40)


def route_solids(W, pts, y=0.6, w=1.4, tint=(1.6, 0.55, 0.25)):
    """A route as a diagram on the section plane: the segments are horizontal ribbons and vertical posts (a lift), a bright tint."""
    from shapely.geometry import LineString, box as sb
    out = []
    for (x0, _, z0), (x1, _, z1) in zip(pts[:-1], pts[1:]):
        if abs(x1 - x0) < 0.3 and abs(z1 - z0) < 0.3: continue
        if abs(x1 - x0) < 0.5: poly = sb(x0 - w / 2, y - w / 2, x0 + w / 2, y + w / 2)          # ascent — a post
        else: poly = LineString([(x0, y), (x1, y)]).buffer(w / 2, cap_style=2)                 # transition — a ribbon
        out.append({"type": "prism", "poly": poly, "z0": min(z0, z1) - 0.2, "z1": max(z0, z1) + 0.9, "cls": "street", "tint": tint, "tone": 1.25})
    for x, _, z in pts[::2]:                                                     # route nodes
        out.append({"type": "prism", "poly": sb(x - w, y - w, x + w, y + w), "z0": z - 0.2, "z1": z + 2.2, "cls": "street", "tint": tint, "tone": 1.3})
    return out


def hive(route=0, **kw):
    """Hive (hive.py): a section of the whole mountain, a three-quarter view, a view from a tier street down the shaft; route=1 —
    a route from the gate to the top."""
    from components.buildings.levels import hive as HV
    W = HV.hive(**kw); H = HV.K["hive"]; zs = W.zs; sx, sw = H["shafts"][0]; s0, s1 = sx - sw / 2, sx + sw / 2
    if route:
        from components.buildings.levels import walk as WK
        D, info, grid = WK.graph(W, 2.0); B = {b["id"]: b for b in W.buildings}
        best = max((k for k in D.p if k.startswith("f:")), key=lambda k: B[int(k.split(":")[1])]["floors"][int(k.split(":")[2])]["z"])
        pts = WK.route(W, W.origin, best, step=2.0); W.solids += route_solids(W, pts, w=2.4)
        W.notes.append(f"route from the west gate to {best}: {len(pts)} points, {pts[0][2]:.0f} → {pts[-1][2]:.0f} m" if pts else "route not found")
    st1 = H["street"][1]; i = min(4, len(zs) - 2)                                       # cameras — inside a tier
    V = {"elev": {"az": -90, "el": 0, "ortho": True, "res": (1300, 1700), "margin": 1.03},
         "front": {"az": -122, "el": 16, "lens": 35, "res": (1300, 1700)},
         "shaft": {"loc": ((s0 + s1) / 2, st1 + 9.0, zs[i] + 1.7), "tgt": ((s0 + s1) / 2 + 0.5, st1 + 2.0, zs[i] - 120.0), "lens": 20,
                   "res": (1100, 1500)},
         "street": street_view(W, zs[i - 1], (-W.hw[i - 1] + 8.0, W.block[1], W.hw[i - 1] - 8.0, W.block[3]))[0]}   # by the place rule
    for k in (0, 2, 4, len(zs) - 2):                                         # the same rule on the tiers: underhive, workshop, dwellings, nobility
        v = street_view(W, zs[k], (-W.hw[k] + 8.0, W.block[1], W.hw[k] - 8.0, W.block[3]))[0]
        if v: V[f"lvl{k}"] = v
    W.snapshots = [n for n in (1, 3, 5) if n < len(H["levels"])]
    return W, V


def hillside(**kw):
    """A town on a slope (hillside.py): terrace streets, stair lanes, houses with entrances at two levels."""
    from components.buildings.levels import hillside as HS
    W = HS.hillside(**kw)
    V = {"up": {"az": 180, "el": 14, "lens": 35}, "front": {"az": -135, "el": 30, "lens": 35},
         "section": {"az": -90, "el": 0, "ortho": True}, "high": {"az": 155, "el": 48, "lens": 35}}
    if getattr(W, "alley_cam", None): V["alley"] = W.alley_cam
    return W, V


def matera(**kw):
    """Matera slope (hillside.py): houses of 1–2 storeys, a house's roof is the pavement of the street above it, cave rooms into
    the hill."""
    base = dict(seed=11, grade=0.8, tiers=8, floors=(1, 2), street=2.6, length=64.0, plot=(4.0, 6.5), alley_every=(9.0, 16.0),
                alley=1.8, wave=(5.0, 16.0), roof_street=0.0, under=0.65, fill=0.85, over=(0, 1), caves=1, cover=1.2, roofs="flat")
    base.update(kw)
    return hillside(**base)


def free_spot(W, x0, x1, y0, y1, step=1.0, extra=()):
    """A point on the ground within a rectangle with the largest clearance from buildings and supports (extra — more polygons)
    → (x, y, clearance)."""
    import shapely
    from shapely.ops import unary_union
    obs = unary_union([b["poly"] for b in W.buildings if b["state"] not in ("buried", "gone")] +
                      [pl["poly"] for d in W.decks for pl in d.get("pillars", [])] + list(extra))
    X, Y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step)); X, Y = X.ravel(), Y.ravel()
    ok = ~shapely.contains_xy(obs, X, Y); X, Y = X[ok], Y[ok]
    d = shapely.distance(obs, shapely.points(X, Y)); k = int(np.argmax(d))
    return float(X[k]), float(Y[k]), float(d[k])


def street_view(W, z, region, eye=1.7, step=1.0, grid=3.0, dirs=24, reach=160.0, lens=20, res=(1600, 1000)):
    """A "level street" frame by the place rule: a point on the walkable surface at elevation z and the direction with the
    longest free ray at eye height (up to a building, a support or the edge of the walkable surface), better — under a
    ceiling (a deck above) and between buildings (the side rays are short: a street, not a square). A 1 m raster — no Blender.
    → a view {loc, tgt, lens} and the ray length."""
    import shapely
    x0, y0, x1, y1 = region
    X = np.arange(x0 + step / 2, x1, step); Y = np.arange(y0 + step / 2, y1, step); Xg, Yg = np.meshgrid(X, Y)
    inside = lambda poly: shapely.contains_xy(poly, Xg, Yg)
    walk = np.abs(np.asarray(W.layer(-1)(Xg, Yg), float) - z) <= 0.6
    for e in W.streets + W.decks:
        if abs(e["z"] - z) <= 0.6: walk |= inside(e["poly"])
    lo, hi = z + 0.5, z + eye + 0.8; block = np.zeros_like(walk); ceil = np.zeros_like(walk)
    for b in W.buildings:
        top = b["z_cut"] if b["z_cut"] is not None else b["z_top"]
        if b["state"] in ("whole", "raised", "cut") and b["z_found"] < hi and top > lo: block |= inside(b["poly"])
    for d in W.decks:
        for pl in d.get("pillars", []): block |= inside(pl["poly"])
        if z + 3.0 < d["z"] - d["t"] < z + 60.0: ceil |= inside(d["poly"])
    for s_ in W.solids:
        if s_.get("void") or s_.get("cls") in ("stair", "deck"): continue
        if s_["type"] == "prism":
            if s_["z0"] < hi and s_["z1"] > lo: block |= inside(s_["poly"])
            continue
        band = s_["prof"].intersection(sbox(-1e5, lo, 1e5, hi))                # extruded (walls, arcades): the trace of the eye-level band
        for g_ in getattr(band, "geoms", [band]):
            if g_.is_empty or g_.area < 1e-6: continue
            u0, _, u1, _ = g_.bounds
            block |= inside(sbox(u0, s_["a0"], u1, s_["a1"]) if s_["axis"] == "y" else sbox(s_["a0"], u0, s_["a1"], u1))
    free = walk & ~block
    k = max(1, int(round(grid / step))); J, I = np.nonzero(free[::k, ::k]); J, I = J * k, I * k
    if not len(I): return None, 0.0
    th = np.linspace(0, 2 * np.pi, dirs, endpoint=False); t = np.arange(1.0, reach, step)

    def ray(I_, J_, ang):                                                     # free ray length from the cells, by angles
        xs = X[I_][:, None, None] + np.cos(ang)[None, :, None] * t[None, None, :]
        ys = Y[J_][:, None, None] + np.sin(ang)[None, :, None] * t[None, None, :]
        ii = np.round((xs - X[0]) / step).astype(int); jj = np.round((ys - Y[0]) / step).astype(int)
        ok = (ii >= 0) & (ii < len(X)) & (jj >= 0) & (jj < len(Y))
        f = np.zeros(ii.shape, bool); f[ok] = free[jj[ok], ii[ok]]
        stop = np.argmin(f, axis=2); stop[f.all(axis=2)] = len(t)
        c = np.zeros(ii.shape, bool); c[ok] = ceil[jj[ok], ii[ok]]
        frac = np.array([[c[a, b, :max(1, stop[a, b])].mean() for b in range(c.shape[1])] for a in range(c.shape[0])])
        return t[np.clip(stop - 1, 0, len(t) - 1)] * (stop > 0), frac
    L, cf = ray(I, J, th)
    side = np.minimum(ray(I, J, th + np.pi / 2)[0], ray(I, J, th - np.pi / 2)[0])
    score = L * (0.5 + 0.5 * cf) * np.where((side >= 2.0) & (side <= 16.0), 1.0, 0.55)
    a, b = np.unravel_index(int(np.argmax(score)), score.shape)
    cx, cy, ang, Lb = float(X[I[a]]), float(Y[J[a]]), float(th[b]), float(L[a, b])
    P = [(cx + math.cos(ang) * d_, cy + math.sin(ang) * d_) for d_ in (0.0, 3.0, 6.0, 9.0, 12.0)]   # across — the median over
    Ii = np.array([int(round((p_[0] - X[0]) / step)) for p_ in P]).clip(0, len(X) - 1)              # the first 12 m of the ray
    Jj = np.array([int(round((p_[1] - Y[0]) / step)) for p_ in P]).clip(0, len(Y) - 1)              # (a gap between houses is not a street)
    lft, rgt = (float(np.median(ray(Ii, Jj, np.array([ang + q]))[0][:, 0])) for q in (np.pi / 2, -np.pi / 2))
    if max(lft, rgt) < 20.0:                                                  # to the middle of the street across the ray
        sh = (lft - rgt) / 2; cx -= math.sin(ang) * sh; cy += math.cos(ang) * sh
    tgt = (cx + math.cos(ang) * Lb * 0.7, cy + math.sin(ang) * Lb * 0.7, z + eye + min(10.0, 0.1 * Lb))
    return {"loc": (cx, cy, z + eye), "tgt": tgt, "lens": lens, "res": res}, Lb


def raised_street(**kw):
    """A raised street (overbuild.py): a section across the street, a passage under the sidewalk."""
    from components.buildings.levels import overbuild as OB
    W = OB.raised_street(**kw); X1 = W.block[2]
    V = {"front": {"az": -118, "el": 22, "lens": 35}, "elev": {"az": -90, "el": 0, "ortho": True},
         "areaway": {"loc": (X1 / 2 - 7.6, 1.5, 1.6), "tgt": (X1 / 2 - 7.4, 30.0, 2.2), "lens": 18}}
    return W, V


def plate(**kw):
    """A slab over an old town (overbuild.py): the slab edge, a section, under the slab."""
    from components.buildings.levels import overbuild as OB
    W = OB.plate(**kw)
    V = {"edge": {"az": 205, "el": 14, "lens": 30}, "elev": {"az": -90, "el": 0, "ortho": True}, "high": {"az": -128, "el": 30, "lens": 35},
         }
    x, y, _ = free_spot(W, 45.0, 110.0, 4.0, 40.0)                            # under the slab: a free spot, looking toward the light well
    lx, ly = min(W.lights, key=lambda c: math.hypot(c[0] - x, c[1] - y)); zg = W.ground(x, y)
    V["under"] = {"loc": (x, y, zg + 1.7), "tgt": (lx, ly, zg + 30.0), "lens": 16}
    return W, V


def viaduct(**kw):
    """A viaduct over a valley (overbuild.py): a section along the bridge axis — vaults with storeys, houses from the bottom up
    above the deck."""
    from components.buildings.levels import overbuild as OB
    W = OB.viaduct(**kw)
    V = {"elev": {"az": -90, "el": 0, "ortho": True}, "front": {"az": -120, "el": 22, "lens": 35},
         "back": {"az": 62, "el": 12, "lens": 35}, "bridge": W.under_cam}
    return W, V


def living_bridge(**kw):
    """A bridge with houses over a river (overbuild.py): the bridge street between rows of houses, landings under the end spans."""
    from components.buildings.levels import overbuild as OB
    W = OB.living_bridge(**kw)
    V = {"elev": {"az": -90, "el": 0, "ortho": True, "depth": 10.0}, "front": {"az": -125, "el": 24, "lens": 32},
         "high": {"az": -60, "el": 42, "lens": 32}}
    V.update(W.cams)
    return W, V


def kowloon(**kw):
    """A self-built block (dense.py): fused towers, passages and wells, bridges, self-built additions on the roofs."""
    from components.buildings.levels import dense as DN
    W = DN.kowloon(**kw)
    return W, {"high": {"az": -128, "el": 38, "lens": 35}, "front": {"az": -110, "el": 14, "lens": 35},
               "alley": {"loc": (W.origin[0], W.block[3] - 0.6, 1.6), "tgt": (W.origin[0], 4.0, 8.0), "lens": 18}}


def underground(**kw):
    """An underground town (dense.py): a section of a rock with chambers, tunnels, passages and a shaft."""
    from components.buildings.levels import dense as DN
    W = DN.underground(**kw)
    return W, {"front": {"az": -118, "el": 20, "lens": 35}, "elev": {"az": -90, "el": 0, "ortho": True}}


def arena(**kw):
    """An amphitheatre that became a town (dense.py)."""
    from components.buildings.levels import dense as DN
    W = DN.arena(**kw)
    return W, {"high": {"az": -110, "el": 40, "lens": 35}, "front": {"az": -125, "el": 18, "lens": 35}, "elev": {"az": -90, "el": 0, "ortho": True},
               "floor": W.ring_cam}


def old_town(**kw):
    """An old town on a ridge (oldtown.py): stair lanes, tenement "lands" on the slope; era 1 — an Exchange built over a lane,
    era 2 — a bridge across the valley; era snapshots — subfolders e<k>, a section along the bridge axis — bridge/."""
    from components.buildings.levels import oldtown as OT
    W = OT.old_town(**kw)
    V = {"front": {"az": 90, "el": 32, "lens": 32}, "elev": {"az": -90, "el": 0, "ortho": True, "depth": 20.0},   # along the valley from the east
         "high": {"az": -135, "el": 52, "lens": 32}, "west": {"az": -98, "el": 24, "lens": 32}}   # from the west: the section and the Exchange in front
    V.update(W.cams); W.snap_views = ("west", "high")
    return W, V


def route3d_solids(pts, w=0.9, lift=0.3, tint=(1.6, 0.55, 0.25)):
    """A route as a ribbon in space: horizontal pieces — a flat strip above the surface, vertical ones (a lift) — a post; sloped
    ones (a stair, a funicular) — in one-meter steps."""
    from shapely.geometry import LineString
    out = []
    for p, q in zip(pts[:-1], pts[1:]):
        h = math.hypot(q[0] - p[0], q[1] - p[1])
        if h < 0.3:
            if abs(q[2] - p[2]) > 0.5:
                out.append({"type": "prism", "poly": sbox(p[0] - 0.35, p[1] - 0.35, p[0] + 0.35, p[1] + 0.35), "z0": min(p[2], q[2]) + lift,
                            "z1": max(p[2], q[2]) + lift + 0.2, "cls": "wall", "tint": tint})
            continue
        n = max(1, int(math.ceil(h / 1.0)))
        for k in range(n):
            a, b = k / n, (k + 1) / n; z = p[2] + (q[2] - p[2]) * (a + b) / 2
            seg = LineString([(p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a), (p[0] + (q[0] - p[0]) * b, p[1] + (q[1] - p[1]) * b)])
            out.append({"type": "prism", "poly": seg.buffer(w / 2, cap_style="flat"), "z0": z + lift, "z1": z + lift + 0.15, "cls": "wall", "tint": tint})
    return out


def cliff_town(route=0, **kw):
    """A town at a cliff (cliff.py): a lower town, the cliff and the slope up to the plateau; a lift in the rock, a lift tower with
    a bridge, a funicular; route=1 — the fastest path from the embankment to a house in the upper town (red) and the same
    without machines (blue)."""
    from components.buildings.levels import cliff as CL
    W = CL.cliff_town(**kw)
    if route:
        from components.buildings.levels import walk as WK
        up = [b for b in W.buildings if b.get("kind") == "upper"]
        tgt = min(up, key=lambda b: abs((b["y0"] + b["y1"]) / 2 - 0.3 * W.block[3]))
        pts, sec = WK.route(W, W.origin, f"f:{tgt['id']}:0", cost=True)
        keep = W.links; W.links = [L for L in keep if L["kind"] not in ("lift", "tunnel_lift", "funicular")]
        pts2, sec2 = WK.route(W, W.origin, f"f:{tgt['id']}:0", cost=True); W.links = keep
        W.solids += route3d_solids(pts) + route3d_solids(pts2, tint=(0.45, 0.75, 1.6), lift=0.6)
        P = np.asarray(pts + pts2, float); lo, hi = P.min(0) - (6.0, 6.0, 2.0), P.max(0) + (6.0, 6.0, 10.0)
        W.cams["route"] = {"az": -128, "el": 50, "lens": 32, "fit": [*map(float, lo), *map(float, hi)]}   # a frame covering the routes
        W.notes.append(f"route from the embankment to house {tgt['id']}: {sec:.0f} s with machines, {sec2:.0f} s without them" if pts2 else "")
    V = {"front": {"az": -160, "el": 34, "lens": 32}, "elev": {"az": -90, "el": 0, "ortho": True, "depth": 12.0},
         "high": {"az": -125, "el": 48, "lens": 32}, "side": {"az": -100, "el": 20, "lens": 32}, "slope": {"az": 90, "el": 28, "lens": 32}}
    V.update(W.cams)
    return W, V


CASES = {"slope_house": slope_house, "slope_row": slope_row, "strata_town": strata_town, "hive": hive, "hillside": hillside,
         "matera": matera, "raised_street": raised_street, "plate": plate, "viaduct": viaduct, "kowloon": kowloon,
         "underground": underground, "arena": arena, "old_town": old_town, "cliff_town": cliff_town, "living_bridge": living_bridge}


def run(case, out, **kw):
    os.makedirs(out, exist_ok=True)
    W, views = CASES[case](**kw)
    bad = S.check(W)
    M = S.emit(W)
    M.save(os.path.join(out, "model.npz"), case=case, views=views, notes=W.notes, checks=bad)
    S.slice_png(M, os.path.join(out, "slice.png"), side="y0", offset=0.05)
    plan = {"case": case, "knobs": kw, "notes": W.notes, "checks": bad,
            "buildings": [{k: v for k, v in b.items() if k not in ("poly",)} for b in W.buildings],
            "voids": W.voids, "streets": [{"z": s["z"], "kind": s.get("kind", "street"), "bounds": list(s["poly"].bounds)} for s in W.streets]}
    json.dump(plan, open(os.path.join(out, "plan.json"), "w"), ensure_ascii=False, indent=1, default=float)
    V, T, *_ = M.arrays()
    print(f"{case}: triangles {len(T)}, vertices {len(V)}; checks: {bad or 'clean'}; {'; '.join(W.notes)}")
    from components.buildings.levels import walk as WK                       # connectivity of the walkable surface
    rep, D, info, grid = WK.report(W, step=1.0 if (W.block[2] - W.block[0]) * (W.block[3] - W.block[1]) < 4e4 else 2.0)
    WK.draw(W, D, grid, os.path.join(out, "walk.png"))
    json.dump(rep, open(os.path.join(out, "walk.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  walkable: parts {rep['n_parts']} (+ ground islets {rep['islets_ground']}), storeys {rep['n_storeys']}, unreachable "
          f"{rep['unreachable_n_storeys']} {rep['examples'][:4]}, decks {rep['decks']}, unreachable: {rep['unreachable_decks'][:6]}")
    if getattr(W, "snapshots", None):                                          # hive: growth snapshots — n tiers
        from components.buildings.levels import hive as HV
        fit = [*map(float, V.min(0)), *map(float, V.max(0))]                    # a common scale for the snapshots — the extent of the final result
        sv = {k: dict(views[k], fit=fit) for k in ("elev", "front")}
        for n in W.snapshots:
            Wn = HV.hive(seed=kw.get("seed", 7), n=n); d = os.path.join(out, f"n{n}"); os.makedirs(d, exist_ok=True)
            S.emit(Wn).save(os.path.join(d, "model.npz"), case=f"{case}_n{n}", views=sv)
    if getattr(W, "history", None) is not None:                                # era snapshots: the same world at the moment of era k
        from components.buildings.levels import epochs as EP
        sv = {k: views[k] for k in getattr(W, "snap_views", ("high", "front"))}
        if getattr(W, "epochs_n", None):                                       # the world of events — snapshots at the scale of the final result
            fit = [*map(float, V.min(0)), *map(float, V.max(0))]; sv = {k: dict(v, fit=fit) for k, v in sv.items()}
        for k in range(getattr(W, "epochs_n", None) or len(W.ground_k)):
            Wk = EP.snapshot(W, k); d = os.path.join(out, f"e{k}"); os.makedirs(d, exist_ok=True)
            S.emit(Wk).save(os.path.join(d, "model.npz"), case=f"{case}_e{k}", views=sv)
    for name, (blk, vw) in getattr(W, "cuts", {}).items():                    # another section: the same world in a different block of the model
        keep = W.block; W.block = blk; d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        S.emit(W).save(os.path.join(d, "model.npz"), case=f"{case}_{name}", views=vw); W.block = keep
    return out


def _val(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return {"true": True, "false": False}.get(v, v)


if __name__ == "__main__":
    case = sys.argv[1]; kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = kw.pop("out", os.path.join(ROOT, "out", "levels", case))
    run(case, out, **{k: _val(v) for k, v in kw.items()})
