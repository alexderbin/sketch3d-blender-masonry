"""Hive — levels over time, pushed to the limit (knowledge — levels.yaml, section hive).

The same world (strata.World) as for a house on a slope and a town over a town, but the era's ground is a deck: the domes
of era 0 stand on a plain → rubble inside and ash outside bury their bottoms (layers) → the first deck is above the domes
→ tiers upward, each narrower than the previous by a step (a stepped shell) → a spire. A tier's zone depends on age and
depth (from the bottom: underhive — ruins, abandoned workshops, workshops, lower dwellings, dwellings, upper dwellings,
nobility), each zone has its own grain (building kind, plot width, density).
Shafts are holes in the decks with bridges along the tier street; the central shaft goes down to a flooded cave. The clay
tone follows depth."""
import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box as sbox

from components.buildings.levels import geom as G
from components.buildings.levels import strata as S
from components.buildings.levels.epochs import K


def _circle(cx, cy, r, n=40):
    return Polygon([(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)])


def zone(i, n):
    """The zone of tier i with n tiers built — by the rank from the top (levels.yaml: rank)."""
    R = K["hive"]["rank"]; r = n - 1 - i; return R[min(r, len(R) - 1)]


def hive(seed=7, n=None):
    """A hive with n tiers (all by default): a growth snapshot — the same mechanism, fewer tiers, less ash."""
    H = K["hive"]
    W0, D, step, dk, sh = H["width"], H["depth"], H["step"], H["deck"], H["shell"]
    hw0 = W0 / 2; n = len(H["levels"]) if n is None else int(n); LV = [{"h": h, "zone": zone(i, n)} for i, h in enumerate(H["levels"][:n])]
    z_deck1 = max(r for _, r in H["domes"]) + H["drum"] + 6.0
    zs = [z_deck1]
    for L in LV: zs.append(zs[-1] + L["h"])                                     # zs[i] is the top of the deck of tier i (0 — the first one above the domes)
    z_top = zs[-1]; zb = H["sump"]["floor"] - 6.0
    rs = np.random.default_rng(seed + 1)                                       # the steps are uneven: its own on each side and tier
    hl = list(hw0 - np.cumsum(step * rs.uniform(0.55, 1.45, len(H["levels"]) + 1)))[:len(LV) + 1]
    hr = list(hw0 - np.cumsum(step * rs.uniform(0.55, 1.45, len(H["levels"]) + 1)))[:len(LV) + 1]
    hw = hl
    arc = {i for i, L in enumerate(LV) if L["zone"] in ("underhive", "ruined_factory", "factory")}   # the lower tiers carry arcades
    sump = H["sump"]

    def g0(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        basin = sump["floor"] * np.clip(1.0 - (np.abs(X) - sump["half"] * 0.6) / (sump["half"] * 1.2), 0, 1)
        return np.where(np.abs(X) < hw0, basin, 0.0) + 0.0 * Y

    def rubble(X, Y):
        X = np.asarray(X, float); g = g0(X, Y)
        return np.where(np.abs(X) < hw0 - sh, np.maximum(g, H["rubble"] + 1.5 * np.sin(X / 13.0)), g)

    n_ash = int(round(len(H["ash"]) * n / len(H["levels"])))
    layers = [g0] + ([rubble] if n >= 2 else [])
    for k, (a, L) in enumerate(H["ash"][:n_ash]):
        layers.append(lambda X, Y, prev=layers[-1], a=a, L=L, k=k: prev(X, Y) + np.where(
            np.abs(np.asarray(X, float)) >= hw0, a * np.clip(1.0 - (np.abs(np.asarray(X, float)) - hw0) / L, 0, 1) ** 1.3, 0.0))
    W = S.World((-hw0 - 85.0, 0.0, hw0 + 85.0, D), layers)
    W.layer_cls = ["rock"] + ["fill"] * (len(layers) - 1)
    W.layer_tints = ([(0.85, 0.82, 0.80)] + ([(1.02, 0.88, 0.76)] if n >= 2 else []) + [(0.92, 0.95, 1.0), (0.98, 1.0, 1.02), (1.04, 1.04, 1.04)][:n_ash])
    W.step = 2.0
    zt_all = z_deck1 + sum(H["levels"])
    dark = lambda x, y, z: float(np.clip(0.30 + 0.78 * ((z - zb) / (zt_all + 120 - zb)) ** 0.9, 0.30, 1.08))
    W.dark = dark
    st0, st1 = H["street"]; cy = D / 2
    sink = H["heat_sink"] / 2
    shafts = [(x - w / 2, x + w / 2) for x, w in H["shafts"]]
    nid = [0]

    def add(b):
        b["id"] = nid[0]; nid[0] += 1; W.buildings.append(b); return b

    # ---- era 0: domes on a plain (stepped rings), one of them collapsed; rubble and ash in layers above
    for i, (dx, R) in enumerate(H["domes"]):                                # the dome center is at the section plane: an arc in the section
        broken = i == 0 and n >= 2
        _dome(W, dx, 2.0, R, H["drum"], th=2.6, tint=tuple(H["zones"]["underhive" if n >= 2 else "noble"]["tint"]), broken=broken, dark=dark)
    W.voids.append({"kind": "sump", "axis": "y", "a0": 0.0, "a1": D, "v0": -sump["half"], "v1": sump["half"],
                    "z0": sump["floor"], "zs": sump["floor"] + 6.0, "arch": "round"})
    W.solids.append({"type": "prism", "poly": sbox(-sump["half"] + 0.3, 0.0, sump["half"] - 0.3, D), "z0": sump["floor"],
                     "z1": sump["water"], "cls": "water", "tone": 0.7, "cut_ground": True})
    # ---- shell: stepped walls from the ground to the spire
    edges = [(-hw0, hw0, 0.0, zs[0] - dk, None)] + [(-hl[i], hr[i], zs[i], zs[i + 1] - dk, LV[i]["zone"]) for i in range(len(LV))]
    for xa, xb, za, zb_, zn in edges:                                          # the deck rests on the shell; dwellings and nobility have arcades
        tone = dark(0, 0, (za + zb_) / 2)                                      # in the shell (light and air inside, Lexicanum)
        for x0, x1 in ((xa, xa + sh), (xb - sh, xb)):
            if zn in ("hab_low", "hab", "hab_high", "noble") and zb_ - za > 14:
                prof = G.arcade_profile(0.0, D, za - 2.0, zb_, span=9.0, pier=4.0, z_spring=za + (zb_ - za) * 0.45, kind="pointed")
                W.solids.append({"type": "extrude", "prof": prof, "axis": "x", "a0": x0, "a1": x1, "cls": "retain", "tone": tone})
            else:
                W.solids.append({"type": "prism", "poly": sbox(x0, 0.0, x1, D), "z0": za - (2.0 if za > 0 else 0.0), "z1": zb_,
                                 "cls": "retain", "tone": tone, "cut_ground": za <= 0.0})
    # ---- tier decks: holes of the shafts (tiers 1–6) and of the central shaft; bridges and supports along the street
    for i, L in enumerate(LV + [None]):                                        # deck i spans the full width of the tier under it (the roof
                                                                                  # of the lower one), the shell of tier i stands on it with a step
        z = zs[i]; xa, xb = (-hw0 + 0.5, hw0 - 0.5) if i == 0 else (-hl[i - 1], hr[i - 1])
        poly = sbox(xa, 0.0, xb, D).difference(sbox(-sink, -1.0, sink, D + 1.0))
        has_sh = i < len(LV) - 1
        if i == len(LV):                                                          # the roof of the top tier is the base of the spire (the needle —
            W.decks.append({"poly": sbox(xa, 0.0, xb, D), "z": z, "t": dk, "tone": dark(0, 0, z), "name": "roof", "prunable": False}); break   # the top of the central shaft)
        sh_in = [(s0, s1) for s0, s1 in shafts if s0 > xa + 4 and s1 < xb - 4] if has_sh else []   # the shaft is inside the tier
        for s0, s1 in sh_in: poly = poly.difference(sbox(s0, -1.0, s1, D + 1.0))
        W.decks.append({"poly": poly, "z": z, "t": dk, "tone": dark(0, 0, z), "name": f"tier {i}", "prunable": False,
                        "pillars": [] if (i - 1) in arc else _pillars(xa, xb, z - (zs[i - 1] if i else 0.0), z, st0, st1, shafts, sink, i, zs,
                                                                         every=15.0 if i <= 2 else 30.0)})
        for s0, s1 in ([(-sink, sink)] + sh_in):                               # bridges over the shafts along the tier street
            W.decks.append({"poly": sbox(s0 - 0.5, st0 + 1.0, s1 + 0.5, st1 - 1.0), "z": z, "t": 1.6, "tone": dark(0, 0, z) * 1.05,
                            "name": "shaft bridge", "prunable": False})
        if i == 0: continue
        W.decks[-1]["level"] = 0
    # ---- lifts (stops on the decks where the shaft is open, and at the bottom)
    # and a gate on the tier at the ash height
    zr = H["rubble"]
    for s0, s1 in shafts + [(-sink, sink)]:
        stops = [zr] + [zs[i] for i in range(len(LV)) if i < len(LV) - 1 or (s0, s1) == (-sink, sink)]
        if (s0, s1) == (-sink, sink): stops.append(zs[len(LV)])
        x, y = s0 - 1.2, st0 - 1.5
        W.links.append({"kind": "lift", "x": x, "y": y, "z": stops})
        W.solids.append({"type": "prism", "poly": sbox(s0 + 0.3, y - 1.4, s0 + 2.9, y + 1.0), "z0": zr, "z1": max(stops) + 4.0,
                         "cls": "pillar", "tone": 0.9})
    for sgn in (-1, 1):
        xo = sgn * (hw0 + 4.0); z_out = float(np.asarray(W.layer(-1)(np.array([xo]), np.array([cy])))[0])
        lv = [i for i in range(len(LV)) if zs[i] <= z_out + 1.5]
        if not lv: continue
        i = lv[-1]; zi = zs[i]; xin = sgn * ((hr[i] if sgn > 0 else hl[i]) - sh - 6.0)
        W.links.append({"kind": "gate", "a": (xo, cy, z_out), "b": (xin, cy, zi)})
        for yy in (cy - 9.0, cy + 4.0):                                       # gate towers outside the shell
            x0g = sgn * (hw0 + 0.5) - (5.0 if sgn < 0 else 0.0)
            W.solids.append({"type": "prism", "poly": sbox(x0g, yy, x0g + 5.0, yy + 5.0), "z0": z_out - 2.0, "z1": max(z_out, zi) + 22.0,
                             "cls": "retain", "tone": 1.0})
        W.notes.append(f"gate {'west' if sgn < 0 else 'east'}: ash {z_out:.0f} m → tier {i} ({zi:.0f} m)")
        if sgn < 0: W.origin = (xo, cy)                                       # start of the walk-through — at the west gate (the ash slope toward it
                                                                              # is steeper than walkable: the approach is the road's business)
    # ---- zones: what stands on the deck
    rng = np.random.default_rng(seed)
    for i, L in enumerate(LV):
        z = zs[i]; h = LV[i]["h"]; Z = H["zones"][L["zone"]]; xa, xb = -hl[i] + sh + 1.0, hr[i] - sh - 1.0
        rng = np.random.default_rng(seed * 100 + i * 7 + len(L["zone"]))         # each tier and its zone has its own seed (the snapshots are consistent)
        free = [(xa, xb)]
        for s0, s1 in [(-sink - 1.0, sink + 1.0)] + ([q for q in shafts if q[0] > xa + 3 and q[1] < xb - 3] if i < len(LV) - 1 else []):
            free = _cut(free, s0 - 1.5, s1 + 1.5)
        for row in ((-6.0, st0), (st1, D + 6.0)):
            for a, c in free:
                x = a
                while c - x > Z["plot"][0] * 0.7:
                    w = min(float(rng.uniform(*Z["plot"])), c - x)
                    if rng.random() < Z["fill"]: _zone_building(W, add, rng, Z, x, x + w, row, z, h, dk, tint=tuple(Z["tint"]))
                    elif Z["kind"] in ("ruin", "ruin_hall"): _rubble(W, x, x + w, row, z, dark, hgt=h * float(rng.uniform(0.15, 0.35)))
                    x += w
        if Z["kind"] == "hab": _skybridges(W, xa, xb, z, h, dk, shafts, sink, st0, st1, rng, dark)
        if L["zone"] == "underhive":                                           # the hive bottom: almost solid rubble with cavities near the ruins
            hole = shapely.unary_union([b["poly"].buffer(2.5) for b in W.buildings if b.get("ruin") and abs(b["z0"] - z - 0.15) < 0.5] +
                                       [sbox(s0 - 2, -1, s1 + 2, D + 1) for s0, s1 in [(-sink, sink)] + shafts] +
                                       [sbox(xa - 1, st0 - 1.0, xb + 1, st1 + 1.0)])
            mass = sbox(xa - 1.0, 0.0, xb + 1.0, D).difference(hole)
            hr_ = h * float(rng.uniform(0.45, 0.6))
            for q in G.polys(mass):
                W.solids.append({"type": "prism", "poly": q, "z0": z, "z1": z + hr_, "cls": "fill", "tone": dark(0, 0, z) * 1.1,
                                 "tint": (1.04, 0.92, 0.80)})
        if i in arc:                                                           # an arcade along the street axis: holds the deck above
            for a, c in free:
                if c - a < 20: continue
                prof = G.arcade_profile(a, c, z, z + h - dk, span=float(rng.uniform(10, 16)), pier=3.0, z_spring=z + (h - dk) * 0.55)
                W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": (st0 + st1) / 2 - 1.2, "a1": (st0 + st1) / 2 + 1.2,
                                 "cls": "wall", "tone": dark(0, 0, z + h / 2), "tint": tuple(Z["tint"])})
        if i >= len(LV) // 2 and i > 0:                                        # towers on the terraces of the upper tiers (the spires grow downward)
            for x0t, x1t in ((-hl[i - 1] + 2.0, -hl[i] - 1.0), (hr[i] + 1.0, hr[i - 1] - 2.0)):
                if x1t - x0t < 6.0: continue
                for _ in range(int(rng.integers(2, 5))):                       # the higher the tier, the taller the terrace towers
                    w = float(min(rng.uniform(4.5, 8), x1t - x0t - 0.5)); x = float(rng.uniform(x0t, x1t - w))
                    hmax = 18 + 40 * i / max(1, len(LV) - 1)
                    _tower(add, x + w / 2, float(rng.uniform(4, D - w - 4)), w, z, float(rng.uniform(12, hmax)), tuple(H["zones"]["noble"]["tint"]))
    # ---- spire
    _spire(W, add, np.random.default_rng(seed), min(hl[len(LV) - 1], hr[len(LV) - 1]) - sh - 4.0, z_top, H["spire"], cy, dark)
    _outer_lifts(W, hl, hr, hw0, zs, len(LV), D)
    for b in W.buildings:
        if b["state"] in ("whole", "cut"):
            b["lod"] = 2 if (b["y0"] < 1.0 or _near_shaft(b, shafts, sink)) else 0
            if b["lod"]: S.relate(W, b)
    from components.buildings.levels import walk as WK
    WK.prune(W, step=2.0)                                                       # no bridge to nowhere is built (the tiers cannot be pruned)
    W.notes.append(f"tiers {len(LV)}, height to the spire {z_top:.0f} m, buildings {len(W.buildings)}")
    W.zs = zs; W.hw = hw
    return W


def _outer_lifts(W, hl, hr, hw0, zs, n, D, which=((1, 2), (-1, 4))):
    """External lifts (transport.lift_bridge): a tower on the ash at the foot and a bridge at the elevation of tier k to the
    edge of its terrace — above the lower terraces; the bridge strip — where there are no terrace towers at its elevation."""
    from shapely.geometry import LineString
    from components.buildings.levels import transport as T
    for sgn, k in which:
        if not 0 < k < n: continue
        xt = sgn * (hw0 + 24.0); xe = sgn * ((hr if sgn > 0 else hl)[k - 1] - 1.5)
        for y in (D * 0.75, D * 0.6, D * 0.88, D * 0.45):
            band = LineString([(xt, y), (xe, y)]).buffer(6.0)
            if not any(b["state"] != "gone" and b["poly"].intersects(band) and b["z_top"] > zs[k] - 3.0 and b["z0"] < zs[k] + 6.0
                       for b in W.buildings): break
        else: continue
        zg = float(np.asarray(W.layer(-1)(np.array([xt]), np.array([y])))[0])
        T.lift_bridge(W, xt, y, zg, zs[k], (xe, y), width=8.0, tower=12.0, t=2.5, name=f"external lift to tier {k}", tint=(0.92, 0.94, 1.0))


def _skybridges(W, xa, xb, z, h, dk, shafts, sink, st0, st1, rng, dark, every=24.0):
    """Bridges across the street of a dwelling tier at two or three heights (Kowloon): a second network of routes inside the
    tier; the walkable ones give entrances. A bridge only between two blocks (there is a house on this level on both sides
    of the street), otherwise it would lead into emptiness."""
    row = lambda x, zz, front: any(b["x0"] + 1.5 < x < b["x1"] - 1.5 and b["z0"] < zz < b["z_top"] and
                                    (abs(b["y1"] - st0) < 0.6 if front else abs(b["y0"] - st1) < 0.6) for b in W.buildings if b.get("kind") == "hab")
    x = xa + every * 0.6
    while x < xb - 6:
        if not any(s0 - 4 < x < s1 + 4 for s0, s1 in shafts + [(-sink, sink)]) and row(x, z + 6.0, True) and row(x, z + 6.0, False):
            for k in sorted(rng.choice([3, 5, 7, 9], size=2, replace=False)):
                zz = z + 0.15 + 3.4 * int(k)
                if zz < z + h - dk - 3.0 and row(x, zz + 1.0, True) and row(x, zz + 1.0, False):
                    W.decks.append({"poly": sbox(x - 1.7, st0 - 0.2, x + 1.7, st1 + 0.2), "z": zz, "t": 0.6, "tone": dark(x, st0, zz) * 1.05,
                                    "name": "street bridge"})                      # wider than the span step: the door lands on the bridge
        x += every * float(rng.uniform(0.8, 1.3))


def _cut(free, a, b):
    out = []
    for x0, x1 in free:
        if b <= x0 or a >= x1: out.append((x0, x1)); continue
        if a > x0: out.append((x0, a))
        if b < x1: out.append((b, x1))
    return out


def _near_shaft(b, shafts, sink, r=12.0):
    return any(b["x1"] > s0 - r and b["x0"] < s1 + r for s0, s1 in shafts + [(-sink, sink)]) and b["y0"] < 34.0


def _pillars(xa, xb, hgt, z, st0, st1, shafts, sink, i, zs, every=30.0, w=3.0):
    """Deck supports — along the street of the tier below (on its axis), clear of the shafts; the height — from the deck below."""
    z0 = zs[i - 1] if i else 0.0; y = (st0 + st1) / 2; out = []
    x = xa + every / 2
    while x < xb - every / 3:
        if not any(s0 - 3 < x < s1 + 3 for s0, s1 in shafts + [(-sink, sink)]):
            out.append({"poly": sbox(x - w / 2, y - w / 2, x + w / 2, y + w / 2), "z0": z0})
        x += every
    return out


def _zone_building(W, add, rng, Z, x0, x1, row, z, h, dk, tint):
    y0, y1 = row; kind = Z["kind"]; ceil = z + h - dk
    if kind == "hab":                                                          # a dwelling block up to the tier ceiling
        top = ceil - float(rng.uniform(0.3, 1.5)); fl = max(2, int((top - z - 0.15) / 3.4))
        b = S.building(x0 + 0.3, x1 - 0.3, y0, y1, z + 0.15, fl, fh=3.4, t=0.5, roof="flat", kind="hab", tint=tint, bay=3.2)
        add(b)
    elif kind == "palace":                                                     # nobility: stepped, with squares
        fl = int(rng.integers(4, 8)); b = add(S.building(x0 + 1.5, x1 - 1.5, y0 + 2.0, y1 - 2.0, z + 0.15, fl, fh=4.2, t=0.6,
                                                        roof="terrace", kind="palace", tint=tint, bay=4.0))
        if rng.random() < 0.6 and b["z_top"] + 10 < ceil:
            add(S.building(x0 + 5.0, x1 - 5.0, y0 + 6.0, y1 - 6.0, b["z_top"] + 0.15, int(rng.integers(2, 4)), fh=4.2, t=0.5,
                           roof="gable", pitch=35, ridge="x", kind="palace", tint=tint, bay=4.0, z_found=b["z_top"] - 0.3))
    elif kind == "hall":                                                       # workshop: a tall hall, pipes up to the deck above
        hh = min(ceil - z - 3.0, float(rng.uniform(26, 40)))
        b = add(S.building(x0 + 1.0, x1 - 1.0, y0 + 1.0, y1 - 1.0, z + 0.15, 1, fh=hh, t=1.0, roof="flat", kind="hall", tint=tint, bay=6.0))
        for _ in range(int(rng.integers(1, 4))):
            cx = float(rng.uniform(b["x0"] + 4, b["x1"] - 4)); cyy = float(rng.uniform(b["y0"] + 4, b["y1"] - 4))
            W.solids.append({"type": "prism", "poly": Point(cx, cyy).buffer(1.8, 6), "z0": b["z_top"], "z1": ceil, "cls": "pillar",
                             "tone": W.dark(cx, cyy, b["z_top"]), "cut_ground": False})
    elif kind in ("ruin", "ruin_hall"):                                        # ruin: cut at a random height, without a roof
        if kind == "ruin":
            top = ceil - float(rng.uniform(1.0, 8.0)); fl = max(2, int((top - z) / 3.4))
            b = S.building(x0 + 0.3, x1 - 0.3, y0, y1, z + 0.15, fl, fh=3.4, t=0.5, roof="flat", kind="ruin", tint=tint, bay=3.2)
        else:
            hh = min(ceil - z - 3.0, float(rng.uniform(24, 36)))
            b = S.building(x0 + 1.0, x1 - 1.0, y0 + 1.0, y1 - 1.0, z + 0.15, 1, fh=hh, t=1.0, roof="flat", kind="ruin", tint=tint, bay=6.0)
        b["state"] = "cut"; b["z_cut"] = float(z + (b["z_top"] - z) * rng.uniform(0.45, 0.95)); b["ruin"] = True
        add(b)
        _shanties(W, rng, b, z, ceil, tint)


def _rubble(W, x0, x1, row, z, dark, hgt=6.0):
    """A rubble heap on an empty ruin plot: four steps, narrowing upward, height hgt."""
    y0, y1 = row; cx = (x0 + x1) / 2; cy = (y0 + y1) / 2; w = (x1 - x0) * 0.48; d = (y1 - y0) * 0.45
    for k, s in enumerate((1.0, 0.75, 0.5, 0.25)):
        W.solids.append({"type": "prism", "poly": sbox(cx - w * s, cy - d * s, cx + w * s, cy + d * s), "z0": z + hgt / 4 * k,
                         "z1": z + hgt / 4 * (k + 1), "cls": "ruin", "tone": dark(cx, cy, z)})


def _shanties(W, rng, b, z, ceil, tint):
    """Underhive self-built housing: box shacks on the cut of a ruin and by its walls (Kowloon from below) — a finer grain than
    the ruin."""
    for _ in range(int(rng.integers(3, 9))):
        w, d, hh = float(rng.uniform(2.5, 4.5)), float(rng.uniform(2.5, 4.0)), float(rng.uniform(2.2, 3.0))
        on_top = rng.random() < 0.6
        x = float(rng.uniform(b["x0"], b["x1"] - w)); y = float(rng.uniform(b["y0"], b["y1"] - d))
        zz = b["z_cut"] if on_top else z
        k = int(rng.integers(1, 4 if on_top else 3))
        for j in range(k):
            if zz + hh * (j + 1) > ceil - 1.0: break
            W.solids.append({"type": "prism", "poly": sbox(x + 0.2 * j, y, x + w - 0.2 * j, y + d), "z0": zz + hh * j, "z1": zz + hh * (j + 1) - 0.15,
                             "cls": "wall", "tone": W.dark(x, y, zz) * float(rng.uniform(0.85, 1.05)), "tint": tint})


def _dome(W, cx, cy, R, drum, th, tint, broken, dark, n=12):
    """A dome on a drum made of stepped rings (the section is an arc); a collapsed one — without a sector and with a breach in
    the vault."""
    tone = dark(cx, cy, R * 0.5)
    ring = _circle(cx, cy, R).difference(_circle(cx, cy, R - th))
    W.solids.append({"type": "prism", "poly": ring, "z0": 0.0, "z1": drum, "cls": "wall", "tone": tone, "tint": tint, "cut_ground": True})
    hole = Polygon([(cx, cy), (cx - R * 1.5, cy + R * 1.4), (cx - R * 0.2, cy + R * 1.8)]) if broken else None
    for i in range(n):
        a0, a1 = math.pi / 2 * i / n, math.pi / 2 * (i + 1) / n
        r = R * math.cos((a0 + a1) / 2); z0 = drum + R * math.sin(a0); z1 = drum + R * math.sin(a1)
        P = _circle(cx, cy, r + 0.01).difference(_circle(cx, cy, max(r - th / max(math.cos(a1), 0.25), 0.3))) if i < n - 1 else _circle(cx, cy, r)
        if broken and i >= n // 3: P = P.difference(hole)
        if broken and i >= n * 2 // 3: continue                              # the vault has collapsed
        if not P.is_empty: W.solids.append({"type": "prism", "poly": P, "z0": z0, "z1": z1, "cls": "wall", "tone": tone, "tint": tint, "cut_ground": True})


def _tower(add, x, y0, w, z, hgt, tint):
    """A stepped tower (4 steps, each narrower by 28 %), a gable top; the entrance to the upper ones is from inside the lower
    one (via)."""
    zz = z; ww = w; below = None
    for t in range(4):
        hh = hgt * (0.4, 0.3, 0.2, 0.1)[t]; nf = max(1, int(hh / 4.0))
        below = add(S.building(x - ww / 2, x + ww / 2, y0 + (w - ww) / 2, y0 + (w + ww) / 2, zz + 0.15, nf, fh=hh / nf,
                               t=0.5, roof="gable" if t == 3 else "flat", pitch=60, ridge="x", kind="spire", tint=tint, bay=3.0,
                               z_found=zz - 0.3, **({"via": below["id"]} if below is not None else {})))
        zz += hh; ww *= 0.72


def _spire(W, add, rng, half, z, sp, cy, dark):
    """Spire: stepped towers on the top deck, a needle in the middle."""
    tint = tuple(K["hive"]["zones"]["noble"]["tint"])
    for k in range(sp["towers"]):
        x = float(rng.uniform(-half, half)); w = float(rng.uniform(7, 14)); hgt = float(rng.uniform(*sp["h"])) * (1.0 - abs(x) / half * 0.6)
        y0 = float(rng.uniform(4, cy - 8)) if rng.random() < 0.5 else float(rng.uniform(cy + 2, K["hive"]["depth"] - w - 4))
        _tower(add, x, y0, w, z, hgt, tint)
    nd = sp["needle"]; W.solids.append({"type": "prism", "poly": sbox(-4.0, cy - 4.0, 4.0, cy + 4.0), "z0": z, "z1": z + nd * 0.7,
                                        "cls": "pillar", "tone": 1.05, "tint": tint})
    W.solids.append({"type": "prism", "poly": sbox(-1.6, cy - 1.6, 1.6, cy + 1.6), "z0": z + nd * 0.7, "z1": z + nd, "cls": "pillar", "tone": 1.05, "tint": tint})
