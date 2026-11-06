"""New over a living old: three cases on one world (strata.World) and one relate rule — entrances, windows and blank sides
are resolved by themselves.
  raised_street — a street raised by a storey between retaining walls (Seattle after 1889): the sidewalk is a deck over a
                  void with glass "windows"; the houses remain — the entrance moved to the second storey, the old
                  shopfronts look into the under-sidewalk passage;
  plate         — a slab on supports over an old town (Midgar): the supports demolish the houses they stand on; under the
                  slab — shade, light wells, shacks at the supports; at the edge — a lift and a ramp; on top — towers;
  viaduct       — an arcade bridge across a valley (South Bridge, 1785): the vaults under the deck are built up with
                  storeys, the houses along the bridge stand from the valley floor above the deck — entrances from below
                  and from the bridge (per HES — "one to four storeys below the bridge");
  living_bridge — a bridge with houses over a river (Krämerbrücke, the old London Bridge): the bridge street between rows
                  of houses, landings under the end spans, riverside houses — an entrance both from the landing and from
                  the bank street."""
import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box as sbox

from components.buildings.levels import geom as G
from components.buildings.levels import strata as S

OLD = (1.06, 0.90, 0.78); NEW = (0.93, 0.97, 1.04)


def _row(rng, y0, y1, w):
    out = []; y = y0
    while y1 - y > w[0] * 0.7:
        b = min(y + float(rng.uniform(*w)), y1); out.append((y, b)); y = b
    return out


# ---------------------------------------------------------------- a raised street
def raised_street(seed=3, rise=4.0, depth=11.0, road=10.0, walk=3.2, length=36.0):
    rng = np.random.default_rng(seed)
    xL1 = depth; xr0 = xL1 + walk; xr1 = xr0 + road; xR0 = xr1 + walk; X1 = xR0 + depth
    W = S.World((0.0, 0.0, X1, length), [])
    road_p = sbox(xr0, -1, xr1, length + 1)                                   # the embankment goes up to the curb: the sidewalk touches the roadway
    g0 = lambda X, Y: 0.0 * np.asarray(X, float)
    g1 = lambda X, Y: g0(X, Y) + rise * shapely.contains_xy(road_p, np.asarray(X, float), np.asarray(Y, float))
    W.layers = [g0, g1]; W.layer_cls = ["rock", "fill"]; W.layer_tints = [(1, 1, 1), (1.08, 0.95, 0.80)]
    W.top_cls = lambda x, y: (("street", (1, 1, 1)) if road_p.contains(Point(x, y)) else ("ground", (1, 1, 1)))
    for xa, xb in ((xr0 - 0.6, xr0), (xr1, xr1 + 0.6)):                       # retaining walls of the roadway (under the sidewalk edge)
        W.solids.append({"type": "prism", "poly": sbox(xa, 0, xb, length), "z0": -0.5, "z1": rise, "cls": "retain", "cut_ground": True})
    for xa, xb in ((xL1, xr0), (xr1, xR0)):                                 # the sidewalk deck with "windows" into the void under it
        P = sbox(xa, -1, xb, length + 1)
        for y in np.arange(0.8, length, 1.1):
            for x in np.arange(xa + 0.5, xb - 0.4, 1.0): P = P.difference(sbox(x, y, x + 0.32, y + 0.32))
        W.decks.append({"poly": P, "z": rise, "t": 0.35, "name": "sidewalk"})
        prof = sbox(xa, 0.0, xb, rise - 0.35).difference(G.arch_profile((xa + xb) / 2 - 0.9, (xa + xb) / 2 + 0.9, -1.0, 1.9))
        for y in np.arange(4.0, length, 6.0):                                 # brick supports of the sidewalk with an arched passage
            W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": y, "a1": y + 0.5, "cls": "wall", "tint": OLD})
    nid = 0
    for xa, xb in ((0.0, xL1), (xR0, X1)):
        for ya, yb in _row(rng, -3.0, length, (7.0, 11.0)):
            fl = int(rng.integers(3, 5))
            b = S.building(xa, xb, ya + 0.1, yb - 0.1, 0.15, fl, fh=4.0, t=0.5, roof="flat", id=nid, tint=OLD, cellars=0, bay=3.2,
                           state="raised")
            W.buildings.append(b); nid += 1
    for b in W.buildings: S.relate(W, b)
    up = sum(any(d["floor"] == 1 for d in b["doors"]) for b in W.buildings)
    down = sum(any(d["floor"] == 0 for d in b["doors"]) for b in W.buildings)
    W.notes.append(f"street raised by {rise} m; houses {len(W.buildings)}: an entrance from the sidewalk on the 2nd storey — {up}, a door into the under-sidewalk passage — {down}")
    return W


# ---------------------------------------------------------------- a slab over an old town
def plate(seed=4, z=48.0, t=6.0, edge=26.0, length=130.0, depth=64.0, grid=34.0, pillar=8.0, holes=3):
    rng = np.random.default_rng(seed)
    W = S.World((0.0, 0.0, length, depth), [])
    hills = lambda X, Y: 3.0 * np.sin(np.asarray(X, float) / 23.0) + 2.0 * np.cos(np.asarray(Y, float) / 17.0) + 0.04 * np.asarray(X, float)
    W.layers = [hills]
    P = sbox(edge, -1.0, length + 1, depth + 1); lights = []
    for _ in range(holes):
        c = (float(rng.uniform(edge + 25, length - 15)), float(rng.uniform(14, depth - 14))); lights.append(c)
        P = P.difference(Point(c).buffer(6.5, 12))
    pills = []
    for x in np.arange(edge + grid * 0.5, length, grid):
        for y in np.arange(grid * 0.45, depth, grid * 0.9):
            q = sbox(x - pillar / 2, y - pillar / 2, x + pillar / 2, y + pillar / 2)
            if not any(Point(c).distance(q) < 9 for c in lights): pills.append(q)
    W.decks.append({"poly": P, "z": z, "t": t, "name": "slab", "pillars": [{"poly": q, "z0": -2.0} for q in pills], "tint": NEW})

    def dark(x, y, zz):                                                      # shade under the slab; lighter at the wells and at the edge
        if zz > z - t or x < edge: return 1.0
        dl = min([math.hypot(x - cx, y - cy) for cx, cy in lights] + [99.0]); de = x - edge
        return float(np.clip(0.42 + 0.5 * math.exp(-dl / 9.0) + 0.45 * math.exp(-de / 10.0), 0.42, 1.0))
    W.dark = dark; W.lights = lights
    nid = [0]
    # the old town: blocks with 5 m streets, houses of 2–4 storeys; a support demolishes whatever it stands on
    for x in np.arange(2.0, length - 6, 15.0):
        for ya, yb in _row(rng, 2.0, depth - 2, (5.0, 9.0)):
            xa, xb = x, x + float(rng.uniform(8.5, 10.0))
            q = sbox(xa, ya, xb, yb)
            if any(q.intersects(p.buffer(1.0)) for p in pills) or rng.random() < 0.08: continue
            zc = float(hills(np.array([(xa + xb) / 2]), np.array([(ya + yb) / 2]))[0])
            b = S.building(xa, xb, ya + 0.1, yb - 0.1, zc + 0.15, int(rng.integers(2, 5)), fh=3.1, t=0.45, roof="gable" if rng.random() < 0.6 else "flat",
                           pitch=30, ridge="x", tint=OLD, id=nid[0], bay=3.0, epoch=0)
            W.buildings.append(b); nid[0] += 1
    for q in pills:                                                          # shacks at the supports: shade and self-built housing
        cx, cy = q.centroid.x, q.centroid.y
        for k in range(int(rng.integers(3, 7))):
            a = rng.uniform(0, 2 * math.pi); r = pillar / 2 + 1.6; w = float(rng.uniform(2.2, 3.4))
            x0, y0 = cx + r * math.cos(a) - w / 2, cy + r * math.sin(a) - w / 2
            zc = float(hills(np.array([x0]), np.array([y0]))[0])
            for j in range(int(rng.integers(1, 3))):
                W.solids.append({"type": "prism", "poly": sbox(x0, y0, x0 + w, y0 + w), "z0": zc + 2.5 * j, "z1": zc + 2.5 * (j + 1) - 0.1,
                                 "cls": "wall", "tint": OLD, "tone": dark(x0, y0, zc)})
    # towers on the slab; at the edge — a lift tower and a ramp
    for x in np.arange(edge + 6, length - 8, 22.0):
        for y in (8.0, depth - 22.0):
            if rng.random() < 0.25: continue
            w = float(rng.uniform(10, 15)); q = sbox(x, y, x + w, y + w)
            if not P.contains(q): continue
            b = S.building(x, x + w, y, y + w, z + 0.15, int(rng.integers(6, 18)), fh=3.4, t=0.4, roof="flat", tint=NEW, id=nid[0],
                           bay=3.2, epoch=1, z_found=z - 0.3)
            W.buildings.append(b); nid[0] += 1
    W.solids.append({"type": "prism", "poly": sbox(edge - 9.0, 30.0, edge - 1.0, 38.0), "z0": -1.0, "z1": z + 10.0, "cls": "pillar", "tint": NEW})
    zl = float(hills(np.array([edge - 10.5]), np.array([34.0]))[0])            # a lift at the tower: ground ↔ slab edge; the ramp is a link too
    W.links.append({"kind": "elevator", "a": (edge - 10.5, 34.0, zl), "b": (edge + 1.0, 34.0, z)})
    zr = float(hills(np.array([1.0]), np.array([depth - 7.0]))[0])
    W.links.append({"kind": "ramp", "a": (1.0, depth - 7.0, zr), "b": (edge + 1.0, depth - 7.0, z)})
    ramp = Polygon([(edge - 1.0, depth - 10.0), (edge - 1.0, depth - 4.0), (2.0, depth - 4.0), (2.0, depth - 10.0)])
    W.solids.append({"type": "extrude", "prof": Polygon([(2.0, 0.0), (edge - 1.0, z - t), (edge - 1.0, z), (2.0, 1.0)]), "axis": "y",
                     "a0": depth - 10.0, "a1": depth - 4.0, "cls": "deck", "tint": NEW})
    for b in W.buildings: S.relate(W, b)
    W.notes.append(f"slab {z:.0f} m above the old town, supports {len(pills)}, light wells {len(lights)}, "
                   f"old houses {sum(b['epoch'] == 0 for b in W.buildings)}, towers {sum(b['epoch'] == 1 for b in W.buildings)}")
    return W


# ---------------------------------------------------------------- a viaduct over a valley
def viaduct(seed=6, deck=24.0, width=12.0, length=104.0, depth=44.0, span=7.5, pier=2.4, fh=3.3):
    rng = np.random.default_rng(seed)
    yb0, yb1 = (depth - width) / 2, (depth + width) / 2
    vx0, vx1 = 30.0, length - 30.0                                            # the valley floor; the slopes — up to the ridges at deck height

    def valley(X, Y):                                                        # the floor at 0, slopes of 22 m up to the ridges at deck height
        X = np.asarray(X, float)
        return deck - deck * np.clip(np.minimum((X - (vx0 - 22.0)) / 22.0, ((vx1 + 22.0) - X) / 22.0), 0, 1) + 0.0 * np.asarray(Y, float)
    W = S.World((0.0, (yb0 + yb1) / 2, length, depth), [valley])              # the section is along the bridge axis: the vaults with storeys in cross-section
    a0, a1 = vx0 - 24.0, vx1 + 24.0
    W.decks.append({"poly": sbox(a0, yb0, a1, yb1), "z": deck, "t": 1.2, "name": "bridge"})
    prof = G.arcade_profile(a0, a1, -2.0, deck - 1.2, span, pier, deck * 0.62, "round")
    W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": yb0, "a1": yb1, "cls": "wall", "tint": OLD, "cut_ground": True})
    # vaults under the bridge: built up with storeys (end walls with openings and floors across the arch);
    # one is open (the valley-floor street)
    n = max(1, int((a1 - a0 - pier) // (span + pier))); s_ = (a1 - a0 - pier * (n + 1)) / n
    open_k = n // 2
    for k in range(n):
        xa = a0 + pier + k * (s_ + pier); xb = xa + s_; xm = (xa + xb) / 2
        zg = float(valley(np.array([xm]), np.array([0.0]))[0])
        if k == open_k or deck * 0.62 - zg < 4.0: continue
        arch = G.arch_profile(xa, xb, -5.0, deck * 0.62, "round")
        for zz in np.arange(zg + fh, deck * 0.62 + s_ / 2 - 1.0, fh):          # the dwelling floors inside a vault
            sl = arch.intersection(sbox(xa, zz - 0.25, xb, zz))
            if not sl.is_empty: W.solids.append({"type": "extrude", "prof": sl, "axis": "y", "a0": yb0 + 0.4, "a1": yb1 - 0.4, "cls": "slab", "tint": OLD})
        face = arch.intersection(sbox(xa, zg, xb, 1e3))
        for zz in np.arange(zg, deck * 0.62 + s_ / 2 - 1.0, fh):              # windows on each storey and a door at the bottom in the end wall
            face = face.difference(sbox(xm - (0.7 if zz == zg else 0.55), zz + (0.0 if zz == zg else 0.9), xm + (0.7 if zz == zg else 0.55),
                                        zz + (2.2 if zz == zg else 2.3)))
        for ya in (yb0, yb1 - 0.4):
            W.solids.append({"type": "extrude", "prof": face, "axis": "y", "a0": ya, "a1": ya + 0.4, "cls": "wall", "tint": OLD})
    W.streets.append({"poly": sbox(vx0, -1, vx1, depth + 1), "z": 0.0, "t": 0.25, "draw": False})   # the valley-floor street (the Cowgate)
    W.under_cam = {"loc": (a0 + 6.0, (yb0 + yb1) / 2 + 1.5, deck + 1.7), "tgt": (a1 - 10.0, yb1 + 2.0, deck + 3.0), "lens": 22}   # on the bridge, along the street
    nid = 0
    for ya, yb in ((yb1 + 0.6, depth - 1.0),):                                # houses along the bridge (the far row) — from the floor up above the deck
        x = a0 + 2.0
        while x < a1 - 4:
            w = min(float(rng.uniform(7.0, 10.0)), a1 - 2 - x)
            if w < 5.0: break
            zmin = float(np.min(valley(np.linspace(x, x + w, 9), np.zeros(9))))
            z0, k, fhb = S.floor_fit(zmin, deck)                             # the storey height is fitted: a floor both at the bottom and on the bridge; the house does not hang
            over = int(rng.integers(2, 4)); fl = k + over
            b = S.building(x, x + w, ya, yb, z0, fl, fh=fhb, t=0.55, roof="gable", pitch=40, ridge="x" if rng.random() < 0.5 else "y",
                           tint=OLD, id=nid, bay=3.0, z_found=min(z0 - 0.6, zmin - 0.3), below=k)
            W.buildings.append(b); nid += 1; x += w + 0.2
    for b in W.buildings: S.relate(W, b)
    both = sum(len({round(b["floors"][d["floor"]]["z"]) for d in b["doors"]}) >= 2 for b in W.buildings)
    below = [b["below"] for b in W.buildings]
    W.notes.append(f"bridge {deck:.0f} m, vaults {n}, houses {len(W.buildings)}, "
                   f"with entrances both from below and from the bridge — {both}; storeys below the bridge: {min(below)}–{max(below)}")
    return W


# ---------------------------------------------------------------- an inhabited bridge
def living_bridge(seed=3, bank=7.0, quay_z=1.0, river=50.0, street=6.0, house=7.5, over=2.0, floors=(3, 6), plot=(5.0, 8.0),
                  span=9.0, pier=4.0, length=64.0, fh=3.0, gap=5.0):
    """A bridge with houses (Krämerbrücke: 120–125 m, 32 houses, there were 62; the old London Bridge: houses of up to seven
    storeys overhung the river and the street, a roadway of ~6 m, demolished in 1757–62): along the bridge — a street between
    two rows of houses on a deck, the houses project over the river by over (on brackets); in the middle — a gap with a view.
    Under the bridge — the river and landings at the water (a passage under the end spans); on the banks — streets at bank
    level; the riverside houses stand on a landing — an entrance both from the landing and from the bank street (relate);
    stairs from the bank down to the landing at the bridge. Axes: x — across the river, y — along the current; the section
    is the face y0 through the south row of the bridge's houses."""
    rng = np.random.default_rng(seed)
    xs0, xq0 = 16.0, 28.0; xw0 = xq0 + 10.0; xw1 = xw0 + river; xq1 = xw1 + 10.0; xs1 = xq1 + 12.0; X1 = xs1 + 16.0
    yc = street / 2 + house / 2 + over / 2                                    # the section y0 — through the middle of the south row of the bridge's houses
    half = street / 2 + house                                                 # deck: the street and two rows of houses
    deck = bank + 0.3

    def nat(X, Y):
        X = np.asarray(X, float)
        z = np.where((X < xq0) | (X >= xq1), bank, quay_z)
        return np.where((X >= xw0) & (X < xw1), -2.0, z) + 0.0 * np.asarray(Y, float)
    W = S.World((0.0, 0.0, X1, length), [nat]); W.origin = ((xs0 + xq0) / 2, length - 4.0)
    for a, b in ((xs0, xq0 - 0.4), (xq1 + 0.4, xs1)):                        # bank streets
        W.streets.append({"poly": sbox(a, -1.0, b, length + 1.0), "z": bank, "t": 0.25})
    for a, b in ((xw0 - 3.6, xw0), (xw1, xw1 + 3.6)):                        # landings at the water — a passage under the end spans
        W.streets.append({"poly": sbox(a, -1.0, b, length + 1.0), "z": quay_z, "t": 0.25})
    W.solids.append({"type": "prism", "poly": sbox(xw0, -1.0, xw1, length + 1.0), "z0": -2.0, "z1": 0.0, "cls": "water"})
    band = (yc - half - over - 0.6, yc + half + over + 0.6)
    st = [band[1] + 1.6]                                                      # stairs from the bank to the landing — by the bridge
    gaps = [band] + [(v - 1.4, v + 1.4) for v in st]
    for x0, x1, z0, z1 in ((xq0 - 0.4, xq0, quay_z - 0.5, bank), (xq1, xq1 + 0.4, quay_z - 0.5, bank),
                           (xw0 - 0.4, xw0, -2.5, quay_z), (xw1, xw1 + 0.4, -2.5, quay_z)):   # bank and landing walls
        for a, b in [(-1.0, gaps[0][0])] + [(gaps[i][1], gaps[i + 1][0]) for i in range(len(gaps) - 1)] + [(gaps[-1][1], length + 1.0)]:
            if b - a > 0.3:
                W.solids.append({"type": "prism", "poly": sbox(x0, a, x1, b), "z0": z0, "z1": z1, "cls": "retain"})
    for v in st:
        S.stairs(W, xq0 - 2.0, xw0 - 3.8, v, bank, quay_z, width=2.4, axis="x")
        S.stairs(W, xq1 + 2.0, xw1 + 3.8, v, bank, quay_z, width=2.4, axis="x")
    # bridge: an arcade (the end spans are over the landings), a deck, parapets over the landings and at the gap
    step = span + pier; c0 = xw0 - 1.8
    cs = [c for c in (c0 + step * k for k in range(0, 20)) if c + span / 2 <= xq1 - pier]
    prof = sbox(xq0, -2.5, xq1, deck - 1.2)
    for c in cs: prof = prof.difference(G.arch_profile(c - span / 2, c + span / 2, -3.0, deck - 1.2 - span / 2 - 1.0))
    W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": yc - half, "a1": yc + half, "cls": "wall", "tint": OLD, "cut_ground": True})
    W.decks.append({"poly": sbox(xs0 + 6.0, yc - half, xs1 - 6.0, yc + half), "z": deck, "t": 1.2, "name": "bridge", "tint": OLD})
    xm = (xw0 + xw1) / 2; holes = [(xq0 - 6.0, xw0 + 0.5), (xm - gap / 2, xm + gap / 2), (xw1 - 0.5, xq1 + 6.0)]
    for a, b in holes:
        for ya in (yc - half, yc + half - 0.4):
            W.solids.append({"type": "prism", "poly": sbox(a, ya, b, ya + 0.4), "z0": deck - 0.1, "z1": deck + 1.1, "cls": "wall", "tint": OLD})
    nid = 0
    for a, b in ((xw0 + 0.5, xm - gap / 2), (xm + gap / 2, xw1 - 0.5)):     # the bridge's houses: two rows, an entrance from the bridge street
        for ya, yb in ((yc - half - over, yc - street / 2), (yc + street / 2, yc + half + over)):
            for xa, xb in _row(rng, a, b, plot):
                W.buildings.append(S.building(xa + 0.05, xb - 0.05, ya, yb, deck + 0.15, int(rng.integers(floors[0], floors[1] + 1)), fh=fh,
                                              t=0.45, roof="gable", pitch=50, ridge="y", tint=OLD, id=nid, bay=2.6, z_found=deck - 1.4,
                                              tone=float(rng.uniform(0.94, 1.05)), kind="bridge_house"))
                nid += 1
    for x0, x1 in ((xq0 + 0.3, xw0 - 3.8), (xw1 + 3.8, xq1 - 0.3)):          # houses on a landing: a floor both at the landing and at the bank street
        for y0, y1 in _plots(rng, -float(rng.uniform(1.0, 4.0)), length, plot, gaps):
            z0, k, f_ = S.floor_fit(quay_z, bank)
            W.buildings.append(S.building(x0, x1, y0 + 0.05, y1 - 0.05, z0, k + int(rng.integers(1, 3)), fh=f_, t=0.5, roof="gable", pitch=45,
                                          ridge="x", tint=OLD, id=nid, bay=2.8, z_found=z0 - 0.6, kind="quay"))
            nid += 1
    for x0, x1 in ((0.3, xs0 - 0.5), (xs1 + 0.5, X1 - 0.3)):                 # rows behind the bank streets
        for y0, y1 in _plots(rng, -float(rng.uniform(1.0, 4.0)), length, plot, []):
            W.buildings.append(S.building(x0, x1, y0 + 0.05, y1 - 0.05, bank + 0.15, int(rng.integers(3, 5)), fh=fh, t=0.5, roof="gable",
                                          pitch=45, ridge="x", tint=OLD, id=nid, bay=2.8, kind="bank"))
            nid += 1
    for b in W.buildings: S.relate(W, b)
    bh = [b for b in W.buildings if b["kind"] == "bridge_house"]; qh = [b for b in W.buildings if b["kind"] == "quay"]
    W.notes.append(f"bridge {xq1 - xq0:.0f} m, spans {len(cs)}; houses on the bridge {len(bh)} (with an entrance from the bridge street: "
                   f"{sum(bool(b['doors']) for b in bh)}), at the landing {len(qh)} (entrances both from the landing and from the bank: "
                   f"{sum(len({round(b['floors'][d['floor']]['z']) for d in b['doors']}) >= 2 for b in qh)})")
    W.cams = {"river": {"loc": (xw0 - 1.8, length - 18.0, quay_z + 1.7), "tgt": (xm, yc, deck + 2.0), "lens": 24},
              "street": {"loc": (xs0 + 8.0, yc - 0.6, deck + 1.7), "tgt": (xs1 - 4.0, yc + 0.6, deck + 3.0), "lens": 22},
              "quay": {"loc": (xw0 - 1.8, length - 6.0, quay_z + 1.7), "tgt": (xw0 - 1.8, 0.0, quay_z + 2.5), "lens": 22}}
    return W


def _plots(rng, y, y_end, w, gaps):
    out = []
    while y_end - y >= w[0] * 0.6:
        e = y + float(rng.uniform(*w))
        if y_end - e < w[0] * 0.6: e = y_end
        hit = [q for q in gaps if q[0] - 0.2 < e and q[1] + 0.2 > y]
        if hit:
            q = min(hit, key=lambda q: q[0])
            if q[0] - 0.2 - y >= w[0] * 0.6: out.append((y, q[0] - 0.2))
            y = q[1] + 0.2; continue
        out.append((y, e)); y = e
    return out
