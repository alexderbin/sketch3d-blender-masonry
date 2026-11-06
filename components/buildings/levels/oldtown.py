"""An old town on a ridge and new buildings over it (modeled on Edinburgh). The High Street runs along the ridge, the
Cowgate on the valley floor, stair lanes (closes) along the fall line, tenement "lands" of 6–10 storeys with entrances at
several levels; the Exchange over Mary King's Close (1753–61) — the lower storeys of the old houses became its basements,
the lane between them a street under the building; South Bridge (1785–88) — a strip was demolished, the span over the
Cowgate is open, the others are closed by the front houses and became vaulted rooms, the valley is not visible from the
bridge. Three eras of one world:
  0 — the ridge, the valley, the lanes, the lands; 1 — new over old (epochs.overbuild); 2 — a bridge over the valley
  (epochs.bridge_over).
Axes: x — across the valley (south → north), y — along the ridge; the section is the face y0 (across the valley, through
the Exchange); the second section — along the bridge axis (W.cuts)."""
import numpy as np
from shapely.geometry import box as sbox

from components.buildings.levels import epochs as EP
from components.buildings.levels import strata as S


def old_town(seed=8, length=84.0, ridge=14.6, south=14.2, plateau=12.0, cow=(26.0, 32.0), high=(60.0, 68.0), fall=0.42,
             close=2.2, close_every=(9.0, 15.0), plot=(6.0, 9.5), back=1.6, over=(3, 6), deck=14.4, bridge_at=0.64,
             exchange=True, bridge=True):
    """ridge/south — the elevations of the ridge and of the plateau beyond the valley; cow/high — the strips of the Cowgate and
    the High Street along x; close — the lane width; back — the gap between the lower and upper rows of lands; over — storeys
    above the High Street; deck — the bridge elevation; bridge_at — the bridge axis (a fraction of the length); exchange,
    bridge — the events of eras 1 and 2."""
    rng = np.random.default_rng(seed); E = EP.K["eras"]
    OLD, NEW = tuple(E["medieval"]["tint"]), tuple(E["georgian"]["tint"])
    c0, c1 = cow; h0, h1 = high; Xb = (0.0, 82.0); xm = (c1 + h0) / 2

    def nat(X, Y):
        X = np.asarray(X, float)
        z = np.where(X < plateau, south, south * (c0 - X) / (c0 - plateau))   # the plateau and the south slope
        z = np.where(X >= c0, 0.0, z)                                         # the floor — the Cowgate
        z = np.where(X >= c1, ridge * (X - c1) / (h0 - c1), z)                # the north slope up to the ridge — even, with stairs on it
        z = np.where(X >= h0, ridge, z)
        z = np.where(X >= h1, ridge - fall * (X - h1), z)                     # beyond the ridge — downward
        return z + 0.0 * np.asarray(Y, float)
    g = lambda x: float(nat(np.array([x]), np.array([0.0]))[0])
    W = S.World((Xb[0], 0.0, Xb[1], length), [nat]); W.history = []; W.epochs_n = 1 + int(exchange) + int(bridge)
    W.streets += [{"poly": sbox(-1.0, -1.0, plateau - 0.5, length + 1.0), "z": south, "t": 0.25},
                  {"poly": sbox(c0, -1.0, c1, length + 1.0), "z": 0.0, "t": 0.25},
                  {"poly": sbox(h0, -1.0, h1, length + 1.0), "z": ridge, "t": 0.25}]
    W.origin = ((h0 + h1) / 2, length / 2)

    cl = []; y = float(rng.uniform(4.0, 8.0))                                 # stair lanes from the Cowgate to the High Street
    while y < length - 6.0:
        cl.append((y, y + close)); y += close + float(rng.uniform(*close_every))
    for a, b in cl:
        S.stairs(W, c1, h0, (a + b) / 2, 0.0, ridge, width=close - 0.3, axis="x")

    def plots(gaps, y, y_end):
        """Plots along the ridge from y to the block edge y_end (the last one up to the edge); the lanes gaps are not a plot."""
        out = []
        while y_end - y >= plot[0] * 0.6:
            y1 = y + float(rng.uniform(*plot))
            if y_end - y1 < plot[0] * 0.6: y1 = y_end
            hit = [q for q in gaps if q[0] - 0.1 < y1 and q[1] + 0.1 > y]
            if hit:
                q = min(hit, key=lambda q: q[0])
                if q[0] - 0.1 - y >= plot[0] * 0.6: out.append((y, q[0] - 0.1))
                y = q[1] + 0.1; continue
            out.append((y, y1)); y = y1
        return out
    nid = [0]

    def land(x0, x1, y0, y1, z_up, more, row):
        """A land on a slope: the floor at the lower ground; with a street at the top (z_up) — also at its elevation (floor_fit)."""
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, 6), np.linspace(y0, y1, 4)); zz = nat(Xg, Yg); zmin, zmax = float(zz.min()), float(zz.max())
        if z_up is not None: z0, k, fh = S.floor_fit(zmin, z_up)
        else: z0, k, fh = zmin + 0.15, int(np.ceil((zmax + 0.5 - zmin - 0.15) / 3.0)), 3.0
        b = S.building(x0, x1, y0 + 0.05, y1 - 0.05, z0, k + int(rng.integers(more[0], more[1] + 1)), fh=fh, t=0.6, roof="gable",
                       pitch=float(rng.uniform(45, 55)), ridge="x" if rng.random() < 0.6 else "y", epoch=0, kind="land", id=nid[0], tint=OLD,
                       tone=float(rng.uniform(0.94, 1.05)), z_found=min(z0 - 0.6, zmin - 0.3), bay=2.8, row=row)
        nid[0] += 1; W.buildings.append(b); EP._record(b, 0); return b
    for y0, y1 in plots(cl, -float(rng.uniform(2.0, 5.0)), length):   # the lower row: toward the Cowgate
        land(c1 + 0.3, xm - back / 2, y0, y1, None, (1, 2), "lower")
    upper = [land(xm + back / 2, h0 - 0.6, y0, y1, ridge, over, "upper")     # the upper row: an entrance from the High Street, going down the slope
             for y0, y1 in plots(cl, -float(rng.uniform(2.0, 5.0)), length)]
    ncl = []; y = float(rng.uniform(10.0, 20.0))                              # the north row: beyond the High Street, passages going down
    while y < length - 6.0:
        ncl.append((y, y + close)); y += close + float(rng.uniform(18.0, 30.0))
    for y0, y1 in plots(ncl, -float(rng.uniform(2.0, 5.0)), length):
        land(h1 + 0.6, Xb[1] - 1.0, y0, y1, ridge, (2, 4), "north")
    for y0, y1 in plots([], -float(rng.uniform(2.0, 5.0)), length):    # the south row: an entrance from the plateau and from the Cowgate
        land(plateau + 0.4, c0 - 0.3, y0, y1, south, (1, 3), "south")

    hall = None
    if exchange and cl:                                                       # era 1: the Exchange over the first lane
        a, b = cl[0]
        A = max((u for u in upper if u["y1"] <= a + 0.1), key=lambda u: u["y1"], default=None)
        B = min((u for u in upper if u["y0"] >= b - 0.1), key=lambda u: u["y0"], default=None)
        if A and B:
            hall = EP.overbuild(W, [A, B], 1, floors=4, fh=4.4, z_floor=ridge, kind="hall", roof="gable", pitch=24, ridge="y", tint=NEW,
                                tone=1.04, t=0.7, bay=3.6)
    yc = bridge_at * length; front = []
    if bridge:                                                                # era 2: a bridge across the valley
        front = EP.bridge_over(W, W.epochs_n - 1, plateau - 0.5, h0 + 0.3, yc, deck, keep_open=[cow], tint=NEW, rng=rng, over=(2, 4))
    for b in W.buildings:
        if b["state"] in ("whole", "raised", "cut"): S.relate(W, b)
    multi = sum(len({round(b["floors"][d["floor"]]["z"]) for d in b["doors"]}) >= 2 for b in W.buildings if b["kind"] == "land" and b["state"] == "whole")
    W.notes.append(f"lanes {len(cl)}, lands {sum(b['kind'] == 'land' for b in W.buildings)} (standing {sum(b['kind'] == 'land' and b['state'] == 'whole' for b in W.buildings)}), "
                   f"with entrances at two or more levels {multi}")
    if hall:
        x_end = c1 + (hall["z_found"] - 2.2) / ridge * (h0 - c1)
        W.notes.append(f"street under the Exchange: {max(0.0, x_end - hall['x0']):.1f} m of the lane under the building (height ≥ 2.2 m), entered from below")
    if front:
        both = sum(len({round(b["floors"][d["floor"]]["z"]) for d in b["doors"]}) >= 2 for b in front)
        W.notes.append(f"bridge front: storeys below the deck {min(b['below'] for b in front)}–{max(b['below'] for b in front)}, "
                       f"entrances both from the bridge and from below — at {both} of {len(front)}")
    ci = min(range(len(cl)), key=lambda i: abs(sum(cl[i]) / 2 - 0.36 * length)) if cl else None
    W.cams = {"cowgate": {"loc": ((c0 + c1) / 2, yc - 30.0, 1.7), "tgt": ((c0 + c1) / 2, yc + 2.0, 9.0), "lens": 20},
              "bridge": {"loc": (plateau + 1.0, yc + 2.0, deck + 1.7), "tgt": (h0 + 2.0, yc - 1.0, deck + 3.5), "lens": 24}}
    if ci is not None:
        yq = sum(cl[ci]) / 2
        W.cams["close"] = {"loc": (c1 + 1.0, yq, 1.7), "tgt": (h0, yq, ridge + 3.0), "lens": 18}
    if hall:
        yq = sum(cl[0]) / 2
        W.cams["under"] = {"loc": (xm - 7.0, yq, g(xm - 7.0) + 1.7), "tgt": (hall["x0"] + 8.0, yq, g(hall["x0"] + 8.0) + 0.5), "lens": 20}
    if bridge:                                                                # a section along the bridge axis: vaults, the storeys in them, the front
        W.cuts = {"bridge": ((Xb[0], yc, Xb[1], length), {"elev": {"az": -90, "el": 0, "ortho": True, "depth": 8.0},
                                                            "front": {"az": -125, "el": 22, "lens": 35}})}
    return W
