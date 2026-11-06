"""A town at a cliff (the cause is a "level change"): a lower town on a grid of blocks (Baixa), the cliff and the slope up
to the upper town on the plateau; the links are machines, not only stairs (transport.py): a lift tower with a bridge over a
street (Santa Justa), a lift in the rock — a tunnel and a shaft (Polanco), a funicular along the slope (Valparaíso);
houses of 9–10 storeys at the foot of the cliff — an entrance from below and a bridge from the top storey onto the plateau
(Kuixinglou — it comes out of the relate rule by itself).
Axes: x — from the embankment to the plateau, y — along the cliff; zone A (y < split) — the cliff, zone B — the slope; the
section is the face y0 (through the lift in the rock)."""
import numpy as np
from shapely.geometry import Polygon, box as sbox

from components.buildings.levels import strata as S
from components.buildings.levels import transport as T


def cliff_town(seed=4, H=28.0, foot=44.0, run=(12.0, 36.0), split=44.0, blend=8.0, length=96.0, depth=102.0, plot=(7.0, 10.0),
               cross=(24.0, 32.0), lower=(4, 5), fh=3.2):
    """H — the plateau height above the lower town; foot — the foot (x); run — the horizontal run of the cliff (zone A) and of the
    slope (zone B); split — the zone boundary along y; cross — the spacing of the lower town's cross streets; lower — the
    storey count of the blocks."""
    rng = np.random.default_rng(seed)
    run_y = lambda Y: run[0] + (run[1] - run[0]) * np.clip((np.asarray(Y, float) - split) / blend, 0, 1)

    def nat(X, Y):
        return H * np.clip((np.asarray(X, float) - foot) / run_y(Y), 0, 1)
    W = S.World((0.0, 0.0, depth, length), [nat]); W.origin = (3.5, length / 2)
    top = lambda y: foot + float(run_y(np.array([y]))[0])                     # plateau edge
    yb = split + blend                                                        # start of zone B
    # streets of the lower town: the embankment, a middle one, a lane at the foot;
    # cross streets — from the embankment to the foot
    W.streets += [{"poly": sbox(-1.0, -1.0, 7.0, length + 1.0), "z": 0.0, "t": 0.25}, {"poly": sbox(21.0, -1.0, 27.0, length + 1.0), "z": 0.0, "t": 0.25},
                  {"poly": sbox(41.0, -1.0, foot, length + 1.0), "z": 0.0, "t": 0.25}]
    cs = []; y = float(rng.uniform(10.0, 16.0))
    while y < length - 8.0:
        cs.append((y - 3.0, y + 3.0)); y += float(rng.uniform(*cross))
    for a, b in cs: W.streets.append({"poly": sbox(-1.0, a, foot, b), "z": 0.0, "t": 0.25})
    ys = np.linspace(-1.0, length + 1.0, 80)                                  # the street along the plateau edge
    W.streets.append({"poly": Polygon([(top(v) + 0.5, v) for v in ys] + [(top(v) + 7.5, v) for v in ys[::-1]]), "z": H, "t": 0.25})
    nid = [0]

    def add(b): b["id"] = nid[0]; nid[0] += 1; W.buildings.append(b); return b

    def plots(y0, y1, gaps):
        out = []; y = y0
        while y1 - y >= plot[0] * 0.6:
            e = y + float(rng.uniform(*plot))
            if y1 - e < plot[0] * 0.6: e = y1
            hit = [q for q in gaps if q[0] - 0.2 < e and q[1] + 0.2 > y]
            if hit:
                q = min(hit, key=lambda q: q[0])
                if q[0] - 0.2 - y >= plot[0] * 0.6: out.append((y, q[0] - 0.2))
                y = q[1] + 0.2; continue
            out.append((y, e)); y = e
        return out
    for x0, x1 in ((7.5, 20.5), (27.5, 40.5)):                                # blocks of the lower town
        for y0, y1 in plots(-float(rng.uniform(1.0, 4.0)), length, cs):
            add(S.building(x0, x1, y0 + 0.05, y1 - 0.05, 0.15, int(rng.integers(lower[0], lower[1] + 1)), fh=fh, roof="gable", pitch=32,
                           ridge="y", tone=float(rng.uniform(0.95, 1.05))))
    # zone A: a lift in the rock at the section, a lift tower with a bridge over a cross street,
    # houses at the foot with a bridge to the plateau
    T.rock_lift(W, (foot - 0.5, 0.0), (top(0.0) + 4.0, 0.0), 0.0, H, width=3.6)   # the axis lies on the section plane
    ca = [q for q in cs if q[1] < split - 2.0]
    yc = (ca[len(ca) // 2][0] + ca[len(ca) // 2][1]) / 2 if ca else split / 2
    T.lift_bridge(W, 24.0, yc, 0.0, H, (top(yc) + 1.0, yc))
    for y0, y1 in plots(6.0, split - 1.0, [(yc - 3.5, yc + 3.5)]):
        Xg, Yg = np.meshgrid(np.linspace(foot + 0.3, top(y0) - 5.0, 5), np.linspace(y0, y1, 3)); zmin = float(nat(Xg, Yg).min())
        z0, k, f_ = S.floor_fit(zmin, H)
        add(S.building(foot + 0.3, top(y0) - 5.0, y0 + 0.05, y1 - 0.05, z0, k + int(rng.integers(1, 3)), fh=f_, roof="gable", pitch=30,
                       ridge="y", z_found=min(z0 - 0.6, zmin - 0.3), kind="cliff"))
    # zone B: a funicular and stairs along the slope, two rows of houses
    # (the lower one from the foot, the upper one with its floor at the plateau street)
    yf = yb + 0.3 * (length - yb); st = [yb + 0.06 * (length - yb), yb + 0.72 * (length - yb)]
    T.funicular(W, (foot - 1.0, yf, 0.0), (top(yf) + 0.5, yf, H))
    for v in st: S.stairs(W, foot, top(v), v, 0.0, H, width=2.4, axis="x")
    gaps = [(yf - 3.0, yf + 3.0)] + [(v - 1.4, v + 1.4) for v in st]
    xm = foot + 0.42 * run[1]
    for (x0, x1, up) in ((foot + 0.3, xm - 0.6, None), (xm + 0.6, foot + run[1] - 0.6, H)):
        for y0, y1 in plots(yb + 0.5, length, gaps):
            Xg, Yg = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 3)); zz = nat(Xg, Yg); zmin, zmax = float(zz.min()), float(zz.max())
            if up is None: z0, k, f_ = zmin + 0.15, int(np.ceil((zmax + 0.5 - zmin - 0.15) / 3.0)), 3.0
            else: z0, k, f_ = S.floor_fit(zmin, up)
            add(S.building(x0, x1, y0 + 0.05, y1 - 0.05, z0, k + int(rng.integers(1, 3)), fh=f_, roof="gable", pitch=34,
                           ridge="y" if rng.random() < 0.5 else "x", z_found=min(z0 - 0.6, zmin - 0.3), kind="slope"))
    for y0, y1 in plots(-float(rng.uniform(1.0, 4.0)), length, []):          # the upper town behind the plateau street
        x0 = top((y0 + y1) / 2) + 8.0
        if x0 + 8.0 > depth - 0.5: continue
        add(S.building(x0, min(x0 + 12.0, depth - 0.5), y0 + 0.05, y1 - 0.05, H + 0.15, int(rng.integers(3, 5)), fh=fh, roof="gable",
                       pitch=32, ridge="y", tone=float(rng.uniform(0.95, 1.05)), kind="upper"))
    for b in W.buildings: S.relate(W, b)
    for b in W.buildings:
        for br in b["bridges"]: W.solids.append(S.bridge_solid(br, width=2.0))
    cl = [b for b in W.buildings if b["kind"] == "cliff"]
    W.notes.append(f"houses {len(W.buildings)}; at the foot of the cliff {len(cl)}, of them with a bridge to the plateau "
                   f"{sum(bool(b['bridges']) for b in cl)}; with entrances at two or more levels: "
                   f"{sum(len({round(b['floors'][d['floor']]['z']) for d in b['doors']}) >= 2 for b in W.buildings)}")
    W.cams = {"lift": {"loc": (8.0, yc - 1.4, 1.7), "tgt": (40.0, yc, 26.0), "lens": 18},
              "drone": {"loc": (6.0, yc - 26.0, 36.0), "tgt": (42.0, yc + 2.0, 18.0), "lens": 28},
              "funic": {"loc": (foot - 2.0, yf, 2.2), "tgt": (top(yf) - 6.0, yf, H + 4.0), "lens": 22},
              "cut": {"az": -90, "el": 0, "ortho": True, "depth": 5.0, "fit": [foot - 14.0, 0.0, -3.0, top(1.0) + 20.0, 4.0, H + 9.0]}}
    return W
