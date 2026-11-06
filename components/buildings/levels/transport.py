"""Lifts between levels: where a stair would be too long, the town places a machine. A lift tower with a bridge over the roofs
(Lisbon, Santa Justa, 1902: a 45 m tower, a 25 m bridge over the roofs of the Baixa), a lift in the rock — a tunnel from
the lower street and a shaft going up (Valparaíso, Polanco: ~60 m), a funicular along the fall line (Valparaíso). Each is
both a body and a walkability link (W.links), like a stair: walk.py knows where it leads."""
import numpy as np
from shapely.geometry import LineString, Polygon, box as sbox

CAB = (1.25, 1.05, 0.55)                                                       # the cab — in a noticeable tint


def lift_bridge(W, x, y, z_low, z_up, to, width=3.0, tower=4.0, t=1.0, name="lift with a bridge", tint=(1.0, 1.0, 1.0)):
    """A lift tower at the lower street (x, y, z_low) and a bridge from it at elevation z_up to the point to (x, y) of the upper
    walkable surface — over the roofs (the roofs under the bridge must be lower than z_up − t − 1.5: the world's check
    verifies this via W.notes). → the bridge (a deck)."""
    h = tower / 2; c = max(0.6, tower * 0.08)
    for px, py in ((x - h, y - h), (x + h - c, y - h), (x - h, y + h - c), (x + h - c, y + h - c)):   # tower: corner posts,
        W.solids.append({"type": "prism", "poly": sbox(px, py, px + c, py + c), "z0": z_low - 0.5, "z1": z_up, "cls": "wall", "tint": tint})
    for zz in np.arange(z_low + 6.0, z_up - 2.0, 6.0):                          # belts, the machine room at the top,
        for q in (sbox(x - h, y - h, x + h, y - h + c), sbox(x - h, y + h - c, x + h, y + h), sbox(x - h, y - h, x - h + c, y + h),
                  sbox(x + h - c, y - h, x + h, y + h)):
            W.solids.append({"type": "prism", "poly": q, "z0": zz, "z1": zz + 0.5, "cls": "wall", "tint": tint})
    W.solids.append({"type": "prism", "poly": sbox(x - h, y - h, x + h, y + h), "z0": z_up - 0.3, "z1": z_up + 4.5, "cls": "wall", "tint": tint})
    zc = z_low + (z_up - z_low) * 0.4; r = max(1.2, tower * 0.3)              # the cab on its way
    W.solids.append({"type": "prism", "poly": sbox(x - r, y - r, x + r, y + r), "z0": zc, "z1": zc + 2.3 * r, "cls": "wall", "tint": CAB})
    line = LineString([(x, y), to]); deck = line.buffer(width / 2, cap_style="flat")
    W.decks.append({"poly": deck, "z": z_up, "t": t, "name": name, "tint": tint})
    for side in (1, -1):                                                       # bridge railings
        rail = line.offset_curve(side * (width / 2 - 0.15)).buffer(0.15, cap_style="flat")
        W.solids.append({"type": "prism", "poly": rail, "z0": z_up, "z1": z_up + 1.1, "cls": "wall", "tint": tint})
    W.links.append({"kind": "lift", "x": float(x), "y": float(y), "z": [float(z_low), float(z_up)], "name": name})
    low = [b for b in W.buildings if b["state"] in ("whole", "raised") and b["poly"].intersects(deck) and b["z_top"] + 4.0 > z_up - t - 1.5]
    W.notes.append(f"{name}: tower {z_up + 4.5 - z_low:.0f} m, bridge {line.length:.0f} m" + (f"; houses under the bridge are too tall {[b['id'] for b in low]}" if low else ""))
    return W.decks[-1]


def rock_lift(W, mouth, shaft, z_low, z_up, width=3.0, house=(5.0, 5.0, 6.0), name="lift in the rock", tint=(1.0, 1.0, 1.0)):
    """A lift in the rock (Polanco): a tunnel from the mouth mouth (x, y) on the lower street to the shaft shaft (x, y) — a tunnel
    along x, the shaft going up to z_up, with a cabin above it at the upper elevation. The walkability link — from the mouth to
    the platform by the cabin."""
    (mx, my), (sx, sy) = mouth, shaft
    a0, a1 = sorted((mx + (0.5 if sx > mx else -0.5), sx))                   # the tunnel — from the foot (the mouth is on the street)
    W.voids.append({"kind": "tunnel", "axis": "x", "a0": float(a0), "a1": float(a1), "v0": my - width / 2, "v1": my + width / 2,
                    "z0": float(z_low), "zs": float(z_low + 2.4), "arch": "round"})
    W.solids.append({"type": "prism", "poly": sbox(sx - width / 2, sy - width / 2, sx + width / 2, sy + width / 2), "z0": z_low,
                     "z1": z_up - 0.3, "cls": "void", "void": True})               # the shaft (up to the cabin)
    hw, hd, hh = house
    W.solids.append({"type": "prism", "poly": sbox(sx - hw / 2, sy - hd / 2, sx + hw / 2, sy + hd / 2), "z0": z_up - 0.3, "z1": z_up + hh,
                     "cls": "wall", "tint": tint})                                # a cabin with the machinery
    W.links.append({"kind": "tunnel_lift", "name": name, "a": (float(mx), float(my), float(z_low)), "b": (float(sx + hw / 2 + 1.0), float(sy), float(z_up))})
    W.notes.append(f"{name}: tunnel {abs(sx - mx):.0f} m, shaft {z_up - z_low:.0f} m")


def funicular(W, a, b, width=3.4, cars=(0.3, 0.7), car=7.0, name="funicular", tint=(1.0, 1.0, 1.0)):
    """A funicular along the fall line along x: a track from a (x, y, z) to b (x, y, z) — a sloped slab on an embankment and
    supports (where the ground is more than 1.5 m below the track), a cut where it is higher; two counterweight cars (cars —
    fractions of the track). The link — from a to b."""
    (xa, y, za), (xb, _, zb) = a, b
    k = (zb - za) / (xb - xa); z = lambda x: za + k * (x - xa)
    prof = Polygon([(xa, za - 0.7), (xb, zb - 0.7), (xb, zb + 0.25), (xa, za + 0.25)])
    W.solids.append({"type": "extrude", "prof": prof, "axis": "y", "a0": y - width / 2, "a1": y + width / 2, "cls": "deck", "tint": tint})
    for x in np.arange(xa + 2.0, xb - 1.0, 4.0):                                # supports and a cut following the ground under the track
        g = float(W.layer(-1)(np.array([x]), np.array([y]))[0]); zt = z(x)
        if g < zt - 2.2: W.solids.append({"type": "prism", "poly": sbox(x - 0.4, y - width / 2 + 0.3, x + 0.4, y + width / 2 - 0.3), "z0": g - 0.5,
                                          "z1": zt - 0.7, "cls": "pillar", "tint": tint})
        elif g > zt - 0.5: W.digs.append((sbox(x - 2.0, y - width / 2 - 0.3, x + 2.0, y + width / 2 + 0.3), zt - 0.7))
    for f in cars:                                                              # cars: a sloped box stepping along the track
        x1 = xa + (xb - xa) * f - car / 2; x2 = x1 + car
        cab = Polygon([(x1, z(x1) + 0.3), (x2, z(x2) + 0.3), (x2, z(x2) + 3.2), (x1, z(x1) + 3.2)])
        W.solids.append({"type": "extrude", "prof": cab, "axis": "y", "a0": y - 1.2, "a1": y + 1.2, "cls": "wall", "tint": CAB})
    W.links.append({"kind": "funicular", "a": (float(xa - 1.0), float(y), float(za)), "b": (float(xb + 1.0), float(y), float(zb)), "name": name})
    W.notes.append(f"{name}: {np.hypot(xb - xa, zb - za):.0f} m of track, rise {zb - za:.0f} m, slope {np.degrees(np.arctan(k)):.0f}°")
