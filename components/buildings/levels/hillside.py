"""A town on a slope: streets along the contour lines at different elevations — terraces with retaining walls, stair lanes
along the fall line, houses in the strip between two streets — an entrance from the lower one, a bridge or a door from
the upper one (the relate rule), some houses with a flat roof level with the upper street — the roof serves as the
street's terrace (Matera, Positano). The slope is a plane with a wave of contour lines (a bay), the incline grade.

The world is the same strata.World; the section is the block face y0, passing through the houses of the first row of
lanes."""
import math

import numpy as np
from shapely.geometry import box as sbox

from components.buildings.levels import strata as S


def hillside(seed=5, grade=0.62, tiers=4, floors=(3, 5), street=5.0, length=96.0, plot=(6.0, 10.5), alley_every=(16.0, 30.0),
             alley=2.6, wave=(7.0, 30.0), roof_street=0.35, fh=3.2, over=(0, 3), fill=0.88, under=0.0, caves=0, cover=1.4,
             roofs="gable", turn=True):
    """floors — the level change between neighboring streets in storeys (random per tier); fill — the share of built-up plots
    (the rest are gardens); under — the share of houses under the upper street (their flat roof is its pavement, Matera);
    caves — cave rooms per house (into the hill, a cover cover above the vault); roofs — gable | flat."""
    rng = np.random.default_rng(seed)
    A, Lw = wave
    sx = lambda Y: A * np.sin(np.asarray(Y, float) / Lw) + 0.35 * A * np.sin(np.asarray(Y, float) / (Lw * 0.43) + 1.3)   # the bend of the contour lines
    dzs = [fh * int(rng.integers(floors[0], floors[1] + 1)) for _ in range(tiers)]
    zst = [0.0] + list(np.cumsum(dzs))                                        # elevation of street j
    s_st = [z / grade for z in zst]                                           # the start of street j along the slope (s = x − bend)
    Xb = (-12.0, s_st[-1] + street + 14.0); Yb = (0.0, length)

    # stair lanes along the fall line
    al = []; y = float(rng.uniform(8, 16))
    while y < length - 8:
        al.append((y, y + alley)); y += alley + float(rng.uniform(*alley_every))

    def nat(X, Y):
        return grade * (np.asarray(X, float) - sx(Y))

    blds = []                                                                # (x0, x1, y0, y1, tier) — pads for houses
    W = S.World((Xb[0], Yb[0], Xb[1], Yb[1]), [])
    for j in range(tiers):
        y = 0.0 + float(rng.uniform(-4, 0))
        while y < length - 3:
            w = float(rng.uniform(*plot)); y0, y1 = y, min(y + w, length + 4)
            if any(a - 0.2 < y1 and b + 0.2 > y0 for a, b in al):                # a lane is not a plot
                nxt = min(b for a, b in al if b + 0.2 > y0); y = nxt + 0.3; continue
            blds.append((j, y0, y1)); y = y1
    pads = {}

    def top(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float); s = X - sx(Y)
        z = np.where(s < 0, 0.0, grade * s)                                  # the slope; below the lower street — flat
        for j in range(tiers + 1):                                           # street terraces; between them — an even slope without a drop
            z = np.where((s >= s_st[j]) & (s < s_st[j] + street), zst[j], z)
            if j < tiers:
                a = s_st[j] + street; k = (zst[j + 1] - zst[j]) / max(s_st[j + 1] - a, 1e-6)
                z = np.where((s >= a) & (s < s_st[j + 1]), zst[j] + (s - a) * k, z)
        for (j, y0, y1), (xa, xb, zy) in pads.items():                       # house pads and the yards behind them
            m = (Y > y0) & (Y < y1)
            if xa == "turn": continue                                         # a rotated house — via an excavation (W.digs)
            if xa is None:                                                   # a garden: two terraces — the lower and the upper thirds of the level change;
                ga, gb = xb                                                   # straight boundaries (even walls)
                mid = (ga + gb) / 2; zl = zst[j] + dzs[j] / 3; zu = zst[j] + dzs[j] * 2 / 3
                z = np.where(m & (s >= s_st[j] + street - 0.01) & (X < ga), zst[j], z)
                z = np.where(m & (X >= ga) & (X < mid), zl, z)
                z = np.where(m & (X >= mid) & (X < gb), zu, z); continue
            z = np.where(m & (s >= s_st[j] + street - 0.01) & (X < xb), zst[j], z)   # the pad starts at the street edge (a wedge is a small yard)
            if zy is not None: z = np.where(m & (X >= xb) & (X < xb + zy[1]), zy[0], z)   # the yard behind the house is straight
        return z

    W.layers = [top]
    gardens = []
    for k, (j, y0, y1) in enumerate(blds):
        yy = np.linspace(y0, y1, 8); xw0, xw1 = float(sx(yy).max()), float(sx(yy).min())
        if rng.random() > fill:                                               # a garden: two terraces with straight walls
            pads[(j, y0, y1)] = (None, (s_st[j] + street + xw0 + 0.05, s_st[j + 1] + xw1 - 0.3), None); gardens.append((j, y0, y1)); continue
        ym = (y0 + y1) / 2; xw = float(sx(np.array([ym]))[0])
        dsx = float((sx(np.array([ym + 0.5])) - sx(np.array([ym - 0.5])))[0])   # the slope of the contour line: the house follows the fall line
        th = -math.atan(dsx) if turn else 0.0
        gap = float(rng.choice([0.0, 0.0, 2.2, 3.0])) if under == 0 else 0.0
        und = j < tiers - 1 and rng.random() < under                         # a house under the upper street: its roof is that street's pavement
        fa = s_st[j] + street + xw + 0.05                                     # facade — the edge of the lower street (along the house axis)
        fb = (s_st[j + 1] + street + xw - 0.4) if und else (s_st[j + 1] + xw - gap - 0.3)   # rear — at the upper street or under it
        if not turn: fa = s_st[j] + street + xw0 + 0.05; fb = (s_st[j + 1] + street + xw1 - 0.4) if und else (s_st[j + 1] + xw1 - gap - 0.3)
        if fb - fa < (3.5 if under else 6.0): continue
        flat = und or (rng.random() < roof_street and gap == 0.0)            # a roof level with the upper street — a terrace
        fl = int(round(dzs[j] / fh)) if flat else int(math.ceil(dzs[j] / fh)) + int(rng.integers(over[0], over[1] + 1))
        zy = (zst[j] + float(rng.integers(1, max(2, int(dzs[j] / fh)))) * fh, gap) if gap else None
        w_ = (y1 - y0) - 0.3; cx = (fa + fb) / 2; d_ = fb - fa
        b = S.building(cx - d_ / 2, cx + d_ / 2, ym - w_ / 2, ym + w_ / 2, zst[j] + 0.15, fl, fh=fh, cellars=int(rng.random() < 0.3), ch=3.4,
                       roof="street" if und else ("terrace" if flat or roofs == "flat" else "gable"), pitch=float(rng.uniform(28, 38)),
                       ridge="x" if rng.random() < 0.5 else "y", vault="y", id=len(W.buildings), tone=float(rng.uniform(0.93, 1.05)), tier=j,
                       under_street=und, angle=th)
        if turn:                                                              # the pad — a rotated excavation from the street to the rear; the yard behind the house
            from shapely import affinity
            pad = affinity.rotate(sbox(s_st[j] + street + xw - 1.5, ym - w_ / 2, fb, ym + w_ / 2), th, origin=(cx, ym), use_radians=True)
            W.digs.append((pad, zst[j]))
            if zy is not None:
                W.digs.append((affinity.rotate(sbox(fb, ym - w_ / 2, fb + gap, ym + w_ / 2), th, origin=(cx, ym), use_radians=True), zy[0]))
            pads[(j, y0, y1)] = ("turn", None, None)
        else:
            pads[(j, y0, y1)] = (fa - 0.01, fb, zy)
        W.buildings.append(b)
    for j in range(tiers + 1):                                               # streets (pavement) — except the roofs of the houses under them
        poly = _band(s_st[j], s_st[j] + street, sx, Yb)
        for b in W.buildings:
            if b.get("under_street") and b["tier"] == j - 1: poly = poly.difference(b["poly"])
        W.streets.append({"poly": poly, "z": zst[j], "t": 0.25})
    W.auto_retain = True                                                     # walls — where the ground drops off at plots, yards, gardens
    from shapely.ops import unary_union
    W.retain_zone = unary_union([sbox(-1e3, y0 - 0.6, 1e4, y1 + 0.6).intersection(sbox(s_st[j] + street - 2.0, -1e3, s_st[j + 1] + 4.0 + A * 1.4, 1e4))
                                 for (j, y0, y1) in pads])
    for a, b in al:                                                          # lane stairs: flights from street to street
        yc = (a + b) / 2; xw = float(sx(np.array([yc]))[0])
        for j in range(tiers):
            S.stairs(W, s_st[j] + street + xw, s_st[j + 1] + xw, yc, zst[j], zst[j + 1], width=alley - 0.3)
    for b in W.buildings: S.relate(W, b)
    if caves:
        for b in W.buildings: S.caves(W, b, caves, length=(4.0, 10.0), cover=cover)
    for b in W.buildings:
        for br in b["bridges"]: W.solids.append(S.bridge_solid(br, width=1.6))
    if al:                                                                    # a view from the foot of the lane up the stairs
        ya, yb = al[len(al) // 2]; yc = (ya + yb) / 2; xw = float(sx(np.array([yc]))[0])
        W.alley_cam = {"loc": (s_st[0] + street * 0.45 + xw, yc, zst[0] + 1.7), "tgt": (s_st[1] + xw + 6.0, yc, zst[1] + 4.0), "lens": 18}
    W.notes.append(f"street tiers {tiers + 1} (level changes {', '.join(f'{d:.1f}' for d in dzs)} m), houses {len(W.buildings)}, gardens {len(gardens)}, lanes {len(al)}, "
                   f"entrances at two or more levels {sum(len({d['floor'] for d in b['doors']}) >= 2 for b in W.buildings)}")
    return W


def _band(s0, s1, sx, Yb, n=48):
    """A strip between the contour lines s0 and s1 (with the wave) → a polygon."""
    from shapely.geometry import Polygon
    Y = np.linspace(Yb[0] - 2, Yb[1] + 2, n); X = sx(Y)
    return Polygon(list(zip(X + s0, Y)) + list(zip((X + s1)[::-1], Y[::-1])))


def _retain(s, sx, Yb, z0, z1, al, fronts, t=0.6, step=1.0):
    """A retaining wall at the upper edge of a street — in short links along the wave; not where there is a lane (a stair) or a
    house facade (the house holds the slope itself). The height — up to the natural ground behind the edge, plus a parapet."""
    out = []; Y = np.arange(Yb[0], Yb[1], step)
    for a in Y:
        b = a + step
        if any(a < q1 and b > q0 for q0, q1 in al) or any(a < q1 - 0.1 and b > q0 + 0.1 for q0, q1 in fronts): continue
        x = float(sx(np.array([(a + b) / 2]))[0]) + s
        out.append({"type": "prism", "poly": sbox(x, a, x + t, b), "z0": z0, "z1": z1 + 2.2, "cls": "retain"})
    return out
