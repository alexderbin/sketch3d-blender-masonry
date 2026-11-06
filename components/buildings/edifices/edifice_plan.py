"""Plan of an edifice: the type is a composition of parts (ziggurat, stepped pyramid, temple-mountain, spiral
tower, occult sanctuary, light and dark order temples, dark citadel, axial temple, centric domed;
edifices/edifices.yaml), the form language of the parts is the style genes (style/style_genes.py,
styles.yaml), the place is the platform, the axis, the leveling (as for temples) → parts with reasons,
markup and checks. The build is edifices/bl_edifice.py.

.venv/bin/python -m components.buildings.edifices.edifice_plan <place> [type=…] [style=family|a+b] [seed=1] [at=x,y] [out=folder] [gene=value …]
  → <out>/edifice.json (+ relief.npz, location.json), __RESULT__
Coordinates are local: u runs along the axis (the approach from the west, from the −u side), v across (+
toward the north), z above the base (the platform + the paving frame.floor)."""
import json, math, os, shutil, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
from components.buildings.base import arch_plan as AL                                                         # noqa: E402  (shared with the temples: a per-name seed, a tower by tiers)
from components.buildings.siting import castle_site as CS                                                       # noqa: E402
from components.buildings.edifices import edifice_space as ESP                                                    # noqa: E402
from components.buildings.style import style_genes as TS                                                       # noqa: E402
from core.surface.ground import Ground                                                      # noqa: E402

EK = yaml.safe_load(open(os.path.join(HERE, "edifices.yaml"))); TYPES = EK["types"]; CANON = EK["canon"]
ED_DEF = {"type": "ziggurat", "batter": 0.1, "edge": "none", "column": "square", "shrine_roof": "flat", "ring_obj": "obelisk", "pilaster": 0.0,
          "tablero": 0.0, "cornice": "plain"}
TYPE_RU = {"ziggurat": "ziggurat", "step_pyramid": "stepped pyramid", "temple_mountain": "temple-mountain", "spiral_tower": "spiral tower",
           "occult_ring": "occult sanctuary", "light_order_temple": "Light order temple", "dark_order_temple": "Dark order temple", "dark_citadel": "dark citadel",
           "axial_temple": "temple-axis", "domed": "domed centralized"}
uni = AL.uni


def rect(cu, cv, L, W): return [[cu - L / 2, cv - W / 2], [cu + L / 2, cv - W / 2], [cu + L / 2, cv + W / 2], [cu - L / 2, cv + W / 2]]


def wall_poly(P, gates, half):
    """Outline of an enclosure with vertices at the gate edges (the build skips only the gate segment, not the
    whole side)."""
    out = []
    for i in range(len(P)):
        a = np.asarray(P[i], float); b = np.asarray(P[(i + 1) % len(P)], float); d = b - a; L = float(np.linalg.norm(d)); out.append([round(a[0], 2), round(a[1], 2)])
        for g in sorted(gates, key=lambda g: float((np.asarray(g) - a) @ d)):
            s_ = float((np.asarray(g, float) - a) @ d) / max(L, 1e-9)
            q_ = np.asarray(g, float) - a; off = abs(d[0] * q_[1] - d[1] * q_[0]) / max(L, 1e-9)
            if L > 2 * half + 1.0 and off < 0.5 and half < s_ < L - half:
                for x in (s_ - half, s_ + half): q = a + d / L * x; out.append([round(q[0], 2), round(q[1], 2)])
    return out


def circ(cu, cv, r, n=32, ph=0.0):
    return [[cu + r * math.cos(ph + 2 * math.pi * i / n), cv + r * math.sin(ph + 2 * math.pi * i / n)] for i in range(n)]


class Ctx:
    """Parts builder: seed, scale, style and edifice genes, markup."""

    def __init__(self, Sp, ed, typ, rng, k, note, seed):
        self.Sp, self.ed, self.typ, self.rng, self.k, self.kq, self.note, self.seed = Sp, ed, typ, rng, k, k ** 0.5, note, seed
        self.parts = []; self.flights = []; self.summit = 0.0

    def add(self, kind_, why="", **kw):
        p = {"id": f"{kind_}{len(self.parts)}", "type": kind_, **kw}
        if why: p["why"] = why
        self.parts.append(p); return p

    def flight(self, p0, p1, z0, z1, w, cheek=None, why=""):
        """Flight: bottom p0 → top p1 (u, v), from z0 to z1; the slope is checked separately."""
        L = math.dist(p0, p1); sl = math.degrees(math.atan2(z1 - z0, max(L, 1e-6)))
        f = self.add("flight", why, p0=[round(p0[0], 2), round(p0[1], 2)], p1=[round(p1[0], 2), round(p1[1], 2)], z0=round(z0, 2), z1=round(z1, 2),
                     w=round(w, 2), cheek=round(1.0 * self.kq if cheek is None else cheek, 2), slope=round(sl, 1))
        self.flights.append(f); return f

    def tower(self, c, side, z0, h, crown=None, octagon=None, why="", TSg=None):
        """Tower by tiers (temple_plan.tower_plan) at mark z0; the top comes from the style genes or is given."""
        TSg = TSg or self.Sp["tower"]; zf = min(0.35 * h, max(0.25 * h, 6.0 * self.kq))
        tp = AL.tower_plan(h, side, 0.0, zf, TSg, self.k, self.rng, crown=crown, octagon=octagon)
        return self.add("tower", why or tp["why"], c=[round(c[0], 2), round(c[1], 2)], z0=round(z0, 2), side=round(side, 2), stages=tp["stages"],
                        spire=tp["spire"], h=round(z0 + tp["h"], 1), pinnacles=tp["pinnacles"], buttresses=tp["buttresses"])


def tiers_up(C, L0, W0, n, Ht, first, top, batter, why, poly="rect", tablero=False, terr=None, slope=None):
    """Tiers going up: the bottom one is taller by a factor of first, the top platform is ~top·L0; the setback of a
    tier is at least the batter + a terrace."""
    kq = C.kq; w = np.array([first] + [1.0] * (n - 1)); h = Ht * w / w.sum(); terr = 1.5 * kq if terr is None else terr
    sL = (L0 - top * L0) / 2; out = []; z = 0.0; L, W = L0, W0
    for i in range(n):
        P = rect(0.0, 0.0, L, W) if poly == "rect" else circ(0.0, 0.0, L / 2, 32)
        edge = C.ed["edge"] if i == n - 1 else ("merlons" if C.ed["edge"] == "merlons" else ("spikes" if C.ed["edge"] == "spikes" else "none"))
        t = C.add("tier", why if i == 0 else "", poly=[[round(a, 2), round(b, 2)] for a, b in P], z0=round(z, 2), z1=round(z + h[i], 2), batter=batter,
                  parapet=round((0.9 if edge in ("merlons", "spikes", "pinnacles") else 0.5) * kq, 2), edge=edge,
                  pilasters=round(4.5 * kq, 2) if C.ed.get("pilaster") else 0.0, tablero=bool(tablero), L=round(L, 2), W=round(W, 2))
        out.append(t); s = max(sL / n, batter * h[i] + terr); L, W = max(L - 2 * s, 0.12 * L0), max(W - 2 * s, 0.12 * W0); z += h[i]
    return out, z


def axial_flights(C, tiers, w0, start=0, sides=((-1, 0),), shrink=0.85, slope=None, why="", avail0=None):
    """Flights along the axis to each tier (from the bottom of the tier to the top of its face) from the given
    sides; the bottom of a flight is on the terrace of the tier below: if a gentle one does not fit — a steeper
    one, up to 52°; if that does not fit either — an entrance in the tier's face and a stair inside its body
    (tall ziggurats: the terrace is narrower than the flight)."""
    slope = slope or CANON["stair_slope"]
    for i, t in enumerate(tiers[start:], start):
        h = t["z1"] - t["z0"]; w = w0 * shrink ** (i - start); run = h / math.tan(math.radians(slope))
        if i > 0 or avail0 is not None:                                          # the terrace of the tier below (for the first one — avail0): from its edge to the base of this one
            if i > 0:
                b = tiers[i - 1]; avail = min((b["L"] - 2 * b["batter"] * (b["z1"] - b["z0"]) - t["L"]) / 2, (b["W"] - 2 * b["batter"] * (b["z1"] - b["z0"]) - t["W"]) / 2)
            else: avail = avail0
            avail = avail + t["batter"] * h - 0.6 * C.kq
            if run > avail:
                if avail >= h / math.tan(math.radians(52.0)): run = avail
                else:                                                            # the flight does not fit: a door in the face and a stair inside the body
                    for du, dv in sides:
                        ext = (t["L"] if du else t["W"]) / 2
                        C.add("stair_in", "the terrace is narrower than the flight — an entrance in the tier's face, a stair inside its body" if (du, dv) == sides[0] else "", c=[round(du * ext, 2), round(dv * ext, 2)],
                              dir=[du, dv], z0=t["z0"], z1=t["z1"], w=round(min(w, 2.4 * C.kq), 2), h=round(min(0.6 * h, 3.6 * C.kq), 2), batter=t["batter"])
                    continue
        for du, dv in sides:
            ext = (t["L"] if du else t["W"]) / 2; e1 = ext - t["batter"] * h
            p1 = (du * e1, dv * e1); p0 = (du * (e1 + run), dv * (e1 + run))
            C.flight(p0, p1, t["z0"], t["z1"], w, why=why if (i == start and (du, dv) == sides[0]) else "")


# ---------------------------------------------------------------- types
def ziggurat(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); n = int(rng.integers(T["tiers"][0], T["tiers"][1] + 1)); Ht = L0 * uni(T["height"], rng)
    tiers, z = tiers_up(C, L0, W0, n, Ht, T["first"], uni(T["top"], rng), C.ed["batter"], "tiers with a batter, the bottom one taller: Ur — base 64 × 46 m, 3 tiers; "
                        "Etemenanki — 7 tiers", slope=CANON["stair_slope"])
    st = TS.choose(T["stairs"], rng); h0 = tiers[0]["z1"]; wc = 0.06 * L0 + 2.0 * kq
    if st == "triple" and n >= 2:                                                # Ur: three flights converge at the gate on the first terrace
        dl = 0.05 * L0 + 3.0 * kq; wl = wc * 1.6; run = h0 / math.tan(math.radians(CANON["stair_slope"])); u_f = -L0 / 2
        ext_ = C.ed["batter"] * h0 + 0.3                                         # up to the tier's edge: no gap above the batter
        C.add("tier", "the gate platform in front of the facade at the height of the first tier", poly=rect(u_f - dl + (dl + ext_) / 2, 0.0, dl + ext_, wl), z0=0.0, z1=round(h0, 2), batter=0.0,
              parapet=round(0.6 * kq, 2), edge="none", pilasters=0.0, tablero=False, L=dl, W=wl)
        C.flight((u_f - dl - run, 0.0), (u_f - dl, 0.0), 0.0, h0, wc, why="Ur: the middle flight along the axis")
        ws = 0.6 * wc; ul = u_f - dl + ws / 2 + 0.5 * kq; m_ = 2.0 * kq                 # the side ones go onto the western part of the platform, the gate behind them
        for sgn in (1, -1): C.flight((ul, sgn * (W0 / 2 - m_)), (ul, sgn * (wl / 2 + 0.3)), 0.0, h0, ws, why="Ur: side flights along the facade" if sgn > 0 else "")
        ug0 = u_f - dl + ws + 1.0 * kq; ug1 = u_f + ext_ - 0.3
        C.add("gate", "a gate where the flights converge: the platform in front of it — three flights meet", c=[round((ug0 + ug1) / 2, 2), 0.0], dir=[-1.0, 0.0], w=round(wl * 0.8, 2),
              h=round(min(0.5 * h0, 9.0 * kq) + 3.0 * kq, 2), d=round(max(1.5 * kq, ug1 - ug0), 2), z0=round(h0, 2), kind="arch")
        axial_flights(C, tiers, wc * 0.8, start=1, why="further — along the axis to each tier")
    else:
        axial_flights(C, tiers, wc, start=0, why="Etemenanki: a stair along the axis to each tier")
    top_t = tiers[-1]; hT = top_t["z1"] - top_t["z0"]; Ls = min(uni(T["shrine"], rng) * L0, 0.8 * (top_t["L"] - 2 * top_t["batter"] * hT))   # the shrine is on the platform
    C.add("shrine", "the shrine on top is the 'house of the god' (Ur, Etemenanki — blue glaze)", c=[0.0, 0.0], L=round(Ls, 2), W=round(min(0.75 * Ls, (top_t["W"] - 2 * top_t["batter"] * hT) * 0.8), 2),
          H=round(0.5 * Ls, 2), z0=round(z, 2), roof=C.ed["shrine_roof"], dir=[-1.0, 0.0], comb=0.0)
    C.summit = z + 0.5 * Ls
    C.note("makeup", f"{n} tiers {L0:.0f} × {W0:.0f} m, height {z:.0f} m; stair {st}", "type — ziggurat: tiers going up, a stair, a shrine on top", "type=, seed=")


def step_pyramid(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); n = int(rng.integers(T["tiers"][0], T["tiers"][1] + 1)); Ht = L0 * uni(T["height"], rng)
    top = uni(T["top"], rng); s_ = (L0 - top * L0) / (2 * n); bat = min(0.85 * s_ / (Ht / n), C.ed["batter"] * 2.2)
    tiers, z = tiers_up(C, L0, W0, n, Ht, 1.0, top, bat, "steep talud-tablero tiers: Temple I at Tikal — 9 tiers, 47 m (the nine levels of the underworld)",
                        tablero=bool(C.ed.get("tablero")), terr=0.3 * kq)
    st = TS.choose(T["stairs"], rng); sl = uni(T["slope"], rng); run = z / math.tan(math.radians(sl)); hl = tiers[-1]["z1"] - tiers[-1]["z0"]
    inset = (bat * 0.6 * hl - 0.04 * hl) if tiers[-1]["tablero"] else bat * hl              # the flight goes up to the edge of the top platform
    Lt, Wt = tiers[-1]["L"] - 2 * inset, tiers[-1]["W"] - 2 * inset
    ws = 0.28 * Wt + 1.5 * kq
    if st == "four":                                                             # Chichen Itza: a flight on each face
        for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            e = (Lt if du else Wt) / 2; C.flight((du * (e + run), dv * (e + run)), (du * e, dv * e), 0.0, z, ws * 0.8, why="Chichen Itza: flights on four sides" if du == -1 else "")
    elif st == "twin":                                                           # Templo Mayor: two flights and two shrines
        for sgn in (1, -1): C.flight((-Lt / 2 - run, sgn * (ws / 2 + 0.6 * kq)), (-Lt / 2, sgn * (ws / 2 + 0.6 * kq)), 0.0, z, ws, why="Templo Mayor: a double flight to two shrines" if sgn > 0 else "")
    else: C.flight((-Lt / 2 - run, 0.0), (-Lt / 2, 0.0), 0.0, z, ws, why=f"one steep flight ({sl:.0f}°, Tikal ~70°)")
    Ls = uni(T["shrine"], rng) * L0; Hs = 0.55 * Ls; comb = uni(T["comb"], rng) * Hs if C.ed["shrine_roof"] == "comb" else 0.0
    if st == "twin":
        for sgn in (1, -1): C.add("shrine", "two shrines on one pyramid (Huitzilopochtli and Tlaloc)" if sgn > 0 else "", c=[0.0, round(sgn * Wt / 4, 2)], L=round(min(Ls, Lt * 0.8), 2),
                                  W=round(Wt * 0.42, 2), H=round(Hs, 2), z0=round(z, 2), roof=C.ed["shrine_roof"], dir=[-1.0, 0.0], comb=round(comb, 2))
    else: C.add("shrine", "a shrine of three rooms under a corbel vault, with a roof comb on top (Tikal)", c=[0.0, 0.0], L=round(min(Ls, Lt * 0.85), 2),
                W=round(min(0.7 * Ls, Wt * 0.85), 2), H=round(Hs, 2), z0=round(z, 2), roof=C.ed["shrine_roof"], dir=[-1.0, 0.0], comb=round(comb, 2))
    C.summit = z + Hs + comb
    C.note("makeup", f"{n} tiers {L0:.0f} m, height {z:.0f} m; stair {st} {sl:.0f}°", "stepped pyramid: steep tiers, a steep flight, a shrine with a roof comb", "type=, seed=")


def temple_mountain(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); m = int(rng.integers(T["levels"][0], T["levels"][1] + 1)); f = uni(T["shrink"], rng)
    wg = min(uni(T["gallery_w"], rng) * L0, 6.0 * kq); z = 0.0; lv = []          # a gallery at human scale: for a giant no wider than 6 m × √scale
    for i in range(m):
        Li, Wi = L0 * f ** i, W0 * f ** i; hi = L0 * uni(T["level_h"], rng) * (1 + 0.45 * i)
        t = C.add("tier", "terrace tiers, each higher than the previous (Angkor: three galleries one above another)" if i == 0 else "", poly=rect(0.0, 0.0, Li, Wi), z0=round(z, 2),
                  z1=round(z + hi, 2), batter=0.05, parapet=round(0.5 * kq, 2), edge="none", pilasters=0.0, tablero=False, L=round(Li, 2), W=round(Wi, 2))
        z += hi; lv.append(t)
        gp = rect(0.0, 0.0, Li - 2 * 0.05 * hi - 0.6, Wi - 2 * 0.05 * hi - 0.6); hg = 0.35 * hi + 4.0 * kq
        C.add("gallery", "a covered gallery along the tier edge on posts" if i == 0 else "", poly=gp, z0=round(z, 2), w=round(wg, 2), h=round(hg, 2), post=round(3.6 * kq, 2),
              column=C.ed["column"])
        last = i == m - 1; hc = uni(T["central"], rng) * L0
        for (a_, b_) in gp:                                                      # corner towers: a quincunx on top, pavilions below
            hh = hc * T["corner"] if last else 2.4 * wg; half = (0.9 if last else 0.6) * wg
            C.add("prang", "", c=[round(a_ * 0.97, 2), round(b_ * 0.97, 2)], half=round(half, 2), z0=round(z + (hg if last else hg * 0.6), 2), h=round(hh, 2))
        for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):                        # gopuras over the middles of the galleries
            e = ((Li if du else Wi) / 2 - 0.05 * hi) * 0.99
            C.add("prang", "gopura gates on the axes" if (i == 0 and du == -1) else "", c=[round(du * e, 2), round(dv * e, 2)], half=round(0.55 * wg, 2), z0=round(z + hg * 0.5, 2),
                  h=round(1.8 * wg, 2))
    Lt = lv[-1]["L"] * 0.32; hcore = 0.6 * (lv[-1]["z1"] - lv[-1]["z0"]); hc = uni(T["central"], rng) * L0
    C.add("tier", "the central shrine on its own podium", poly=rect(0.0, 0.0, Lt, Lt), z0=round(z, 2), z1=round(z + hcore, 2), batter=0.05, parapet=round(0.4 * kq, 2),
          edge="none", pilasters=0.0, tablero=False, L=round(Lt, 2), W=round(Lt, 2))
    C.add("prang", "a quincunx of towers: the central one is taller (Angkor ~65 m), four at the corners of the top tier — Mount Meru", c=[0.0, 0.0], half=round(0.42 * Lt, 2),
          z0=round(z + hcore, 2), h=round(hc, 2))
    for i, t in enumerate(lv):                                                    # flights on all four sides, the top one is steep
        h = t["z1"] - t["z0"]; sl = 45.0 if i < m - 1 else 62.0; run = h / math.tan(math.radians(sl))
        for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            e = (t["L"] if du else t["W"]) / 2 - 0.05 * h; C.flight((du * (e + run), dv * (e + run)), (du * e, dv * e), t["z0"], t["z1"], 0.06 * L0 * (0.8 ** i) + 1.5 * kq,
                                                                   why="flights along the axes on all sides; the top one is steep (~60–70°)" if (i == 0 and du == -1) else "")
    C.add("flight_core", "", z0=round(z, 2), z1=round(z + hcore, 2))
    Lo, Wo = L0 * 1.35, W0 * 1.35; mo = uni(T["moat"], rng) * L0
    gw_ = 0.06 * L0 + 3 * kq                                                     # the gate under a gopura — the width of the causeway
    C.add("wall", "an enclosure with a gopura gate on the west", poly=wall_poly(rect(0.0, 0.0, Lo, Wo), [(-Lo / 2, 0.0)], gw_ / 2 + 1.0 * kq), z0=0.0,
          h=round(0.03 * L0 + 3.0 * kq, 2), t=round(1.2 * kq, 2), edge="none", gates=[[round(-Lo / 2, 2), 0.0]])
    C.add("prang", "", c=[round(-Lo / 2, 2), 0.0], half=round(0.9 * wg, 2), z0=round(0.03 * L0 + 3.0 * kq, 2), h=round(3.0 * wg, 2))
    if mo > 6.0 * kq:
        C.add("water", "a moat — the 'world ocean' around Mount Meru (Angkor: ~190 m)", outer=rect(0.0, 0.0, Lo + 2 * mo + 4 * kq, Wo + 2 * mo + 4 * kq), inner=rect(0.0, 0.0, Lo + 4 * kq, Wo + 4 * kq), z=-1.6)
        C.add("bridge", "a causeway across the moat to the gate", p0=[round(-Lo / 2 - mo - 6 * kq, 2), 0.0], p1=[round(-Lo / 2, 2), 0.0], z=0.6, w=round(0.06 * L0 + 3 * kq, 2), rail=1.0)
    C.summit = z + hcore + hc
    C.note("makeup", f"{m} tiers {L0:.0f} m, summit {C.summit:.0f} m; galleries, a quincunx of prangs, gopuras" + (f", moat {mo:.0f} m" if mo > 6 * kq else ""),
           "temple-mountain: tiers with galleries, five towers (the center is taller), gates on the axes, an enclosure and a moat", "type=, seed=")


def spiral_tower(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    D0 = uni(T["base"], rng) * k; H = D0 * uni(T["height"], rng); turns = uni(T["turns"], rng); kind = TS.choose(T["kind"], rng); rt = uni(T["top"], rng) * D0 / 2
    w = 0.1 * D0 + 1.2 * kq
    C.add("tier", "square base", poly=rect(0.0, 0.0, D0 * 1.3, D0 * 1.3), z0=0.0, z1=round(2.5 * kq, 2), batter=0.0, parapet=0.0, edge="none", pilasters=0.0,
          tablero=False, L=D0 * 1.3, W=D0 * 1.3)
    z0 = 2.5 * kq
    if kind == "cone":                                                           # Samarra: a smooth cone, a ramp on the outside
        C.add("cone", "a cone with a spiral ramp (the Malwiya, Samarra: 52 m with a 33 m base)", c=[0.0, 0.0], r0=round(D0 / 2, 2), r1=round(rt, 2), z0=round(z0, 2), z1=round(z0 + H, 2))
        C.add("ramp", "", c=[0.0, 0.0], r_in=[round(D0 / 2 - 0.15 * w, 2), round(rt - 0.15 * w, 2)], r_out=[round(D0 / 2 + 0.85 * w, 2), round(rt + 0.85 * w, 2)],
              z0=round(z0, 2), z1=round(z0 + H, 2), a0=math.pi, turns=round(turns, 2))
        C.add("drum", "a pavilion on top (a platform around it where the ramp comes out)", c=[0.0, 0.0], r=round(rt * 0.55, 2), z0=round(z0 + H, 2), h=round(0.07 * H, 2), n=8, roof="dome")
        C.summit = z0 + H * 1.12
    else:                                                                         # "Tower of Babel": arcade tiers and a ramp on the outside
        nt = int(np.clip(round(turns * 1.6), 5, 9)); r = D0 / 2; z = z0
        dr = max((r - rt) / nt, 1.05 * w); nt = max(3, min(nt, int((r - 0.6 * rt) / dr))); hz = H / nt   # the ramp lies on top of the tier below
        for i in range(nt):
            r1 = r - dr; broken = i == nt - 1 and rng.random() < 0.6
            C.add("drum", "tiers with arcades, each narrower (Bruegel)" if i == 0 else ("the top is unfinished — a 'Babylonian' break" if broken else ""), c=[0.0, 0.0],
                  r=round(r1, 2), z0=round(z, 2), h=round(hz * (0.6 if broken else 1.0), 2), n=24, roof="broken" if broken else "flat", arcade=True)
            C.add("ramp", "", c=[0.0, 0.0], r_in=[round(r1 - 0.1 * w, 2)] * 2, r_out=[round(r1 + 0.9 * w, 2)] * 2, z0=round(z, 2), z1=round(z + hz, 2), a0=math.pi + i * 0.5, turns=1.0)
            r = r1; z += hz
        C.summit = z
    C.flight((-D0 * 0.65 - 6.0 * kq, 0.0), (-D0 * 0.65, 0.0), 0.0, 2.5 * kq, w, why="entrance to the base")
    C.note("makeup", f"{kind}: base {D0:.0f} m, height {H:.0f} m, ramp turns {turns:.1f}", "spiral tower: a ramp around the body — a path upward in circles", "type=, seed=")


def occult_ring(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    R = uni(T["r"], rng) * k; dep = R * uni(T["depth"], rng); nt = int(rng.integers(T["terraces"][0], T["terraces"][1] + 1)); nr = int(rng.choice(T["ring"]))
    obj = C.ed["ring_obj"] if isinstance(C.ed["ring_obj"], str) else TS.choose(C.ed["ring_obj"], rng); ho = R * uni(T["obj_h"], rng); cen = TS.choose(T["center"], rng)
    C.add("court", f"sunken circle: {nt} terrace steps down {dep:.0f} m — an amphitheatre for a rite", c=[0.0, 0.0], r=round(R, 2), r_floor=round(0.38 * R, 2),
          depth=round(dep, 2), terraces=nt, center=cen, z0=0.0)
    C.flight((-R - 1.0 * kq, 0.0), (-0.38 * R, 0.0), 0.0, -dep, 0.07 * R + 1.5 * kq, why="descent along the axis into the circle")
    if cen == "spire":
        C.tower((0.0, 0.0), 0.22 * R, -dep, uni([1.0, 1.6], rng) * R, why="a dark spire from a well at the center of the circle")
    if cen == "ziggurat":                                                        # a dark ziggurat in the circle: tiers, a flight, an altar on top
        Lz = 0.62 * R; tz, zt = tiers_up(C, Lz, Lz, int(rng.integers(3, 5)), Lz * uni([0.45, 0.7], rng), 1.0, 0.3, max(C.ed["batter"], 0.12),
                                         "a dark ziggurat at the bottom of the circle — an altar above the crowd", terr=0.6 * kq, slope=42.0)
        for t in tz:
            t["z0"] = round(t["z0"] - dep, 2); t["z1"] = round(t["z1"] - dep, 2)
        axial_flights(C, tz, 0.07 * Lz + 1.0 * kq, start=0, shrink=0.9, slope=42.0, why="flight to the altar", avail0=0.38 * R - Lz / 2)   # the bottom is at the floor of the circle
        C.add("ring", "", c=[0.0, 0.0], r=0.0, n=1, obj="altar", h=round(2.0 * kq, 2), w=round(1.6 * kq, 2), z0=round(zt - dep, 2))
    C.add("ring", f"a ring of {nr} — an odd number (an 'ill omen'); the descent axis is between the stones", c=[0.0, 0.0], r=round(R + 2.5 * kq, 2), n=nr, obj=obj, h=round(ho, 2),
          w=round(0.08 * R + 1.0 * kq, 2), z0=0.0, a0=round(math.pi + math.pi / nr, 4))
    C.add("ring", "", c=[0.0, 0.0], r=round(R * 1.3 + 4 * kq, 2), n=2 * nr, obj="brazier", h=round(2.4 * kq, 2), w=round(1.4 * kq, 2), z0=0.0, a0=round(math.pi + math.pi / (2 * nr), 4))
    Rw = R * 1.65 + 6.0 * kq
    if rng.random() < T["wall"]:
        ng = 4 if rng.random() < 0.6 else 3
        C.add("wall", "an outer wall all around with a gate", poly=circ(0.0, 0.0, Rw, 48), z0=0.0, h=round(uni(T["wall_h"], rng) * R + 4.0 * kq, 2), t=round(2.0 * kq, 2), edge=C.ed["edge"],
              gates=[[round(Rw * math.cos(math.pi + 2 * math.pi * i / ng), 2), round(Rw * math.sin(math.pi + 2 * math.pi * i / ng), 2)] for i in range(ng)])
        for i in range(ng):
            a_ = math.pi + 2 * math.pi * i / ng; C.add("gate", "", c=[round(Rw * math.cos(a_), 2), round(Rw * math.sin(a_), 2)], dir=[round(math.cos(a_), 4), round(math.sin(a_), 4)],
                                                       w=round(0.22 * R + 4 * kq, 2), h=round(0.3 * R + 6 * kq, 2), d=round(0.1 * R + 3 * kq, 2), z0=0.0, kind="pylon")
        nt_ = int(rng.integers(T["towers"][0], T["towers"][1] + 1))
        for i in range(nt_):
            a_ = math.pi / 4 + 2 * math.pi * i / max(nt_, 1); C.tower((Rw * math.cos(a_), Rw * math.sin(a_)), 0.14 * R + 3 * kq, 0.0, uni([1.0, 1.6], rng) * R,
                                                                       why="towers on the wall" if i == 0 else "")
    L_av = uni(T["avenue"], rng) * R
    for j in range(max(2, int(L_av / (7.0 * kq)))):                               # an avenue to the western gate
        for sgn in (1, -1): C.add("ring", "an avenue of obelisks to the gate" if (j == 0 and sgn > 0) else "", c=[round(-Rw - 4 * kq - j * 7 * kq, 2), round(sgn * (0.12 * R + 3 * kq), 2)],
                                  r=0.0, n=1, obj=obj if obj != "menhir" else "obelisk", h=round(ho * 0.45, 2), w=round(0.04 * R + 0.6 * kq, 2), z0=0.0)
    C.summit = max([ho] + [p["h"] for p in C.parts if p["type"] == "tower"])
    C.note("makeup", f"circle R {R:.0f} m, depth {dep:.0f} m, {cen} at the center; ring {nr} × {obj}", "occult sanctuary: a sunken circle, a ring, a wall with a gate, an avenue", "type=, seed=")


def light_order_temple(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); n = int(rng.integers(T["tiers"][0], T["tiers"][1] + 1)); Ht = L0 * uni(T["height"], rng)
    tiers, z = tiers_up(C, L0, W0, n, Ht, T["first"], uni(T["top"], rng), C.ed["batter"], "ziggurat — a climb to enlightenment by steps (an order temple: the 'typology of Maya temples')",
                        slope=CANON["stair_slope"])
    axial_flights(C, tiers, 0.08 * L0 + 2 * kq, start=0, shrink=0.92, why="a wide stair along the axis")
    Lt, Wt = tiers[-1]["L"], tiers[-1]["W"]
    C.add("shrine", "a hall on top", c=[0.0, 0.0], L=round(0.55 * Lt, 2), W=round(0.55 * Wt, 2), H=round(0.1 * Ht + 3 * kq, 2), z0=round(z, 2), roof="flat", dir=[-1.0, 0.0], comb=0.0)
    hc = uni(T["spire"], rng) * Ht; rc = 0.03 * L0 + 1.0 * kq
    C.add("minaret", "five minaret spires: the central one is taller (the Spire of Serenity), like the minarets of Hagia Sophia", c=[0.0, 0.0], r=round(rc, 2), z0=round(z + 0.1 * Ht, 2), h=round(hc, 2))
    for a_, b_ in rect(0.0, 0.0, Lt * 0.78, Wt * 0.78):
        C.add("minaret", "", c=[round(a_, 2), round(b_, 2)], r=round(rc * 0.7, 2), z0=round(z, 2), h=round(hc * uni(T["corner"], rng), 2))
    C.summit = z + 0.1 * Ht + hc
    C.note("makeup", f"{n} tiers {L0:.0f} m, spire {hc:.0f} m", "Light order temple: a ziggurat and a quincunx of spires (like the temple-mountain)", "type=, seed=")


def dark_order_temple(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); n = int(rng.integers(T["tiers"][0], T["tiers"][1] + 1)); Ht = L0 * uni(T["height"], rng)
    tiers, z = tiers_up(C, L0, W0, n, Ht, T["first"], uni(T["top"], rng), max(C.ed["batter"], 0.2), "a steep dark pyramid — a threat by silhouette", terr=0.4 * kq, slope=42.0)
    nf = int(rng.integers(T["fins"][0], T["fins"][1] + 1)); t0 = tiers[0]
    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):                            # blade fins along the faces of the bottom tier
        e = (t0["L"] if du else t0["W"]) / 2; s_ = (t0["W"] if du else t0["L"])
        for j in range(nf):
            x = -s_ / 2 + (j + 0.5) * s_ / nf
            if du == -1 and abs(x) < 0.08 * L0: continue
            p = (du * e, x) if du else (x, dv * e); C.add("fin", "blade fins along the faces" if (du == -1 and j == 0) else "", p=[round(p[0], 2), round(p[1], 2)],
                                                           n=[du, dv], z0=0.0, z1=round(t0["z1"] * 1.6, 2), d0=round(0.05 * L0, 2), d1=round(0.01 * L0, 2), w=round(0.8 * kq, 2))
    axial_flights(C, tiers, 0.045 * L0 + 1.2 * kq, start=0, shrink=1.0, slope=42.0, why="a narrow stair along the axis")
    ho = uni(T["obelisk"], rng) * z
    C.add("obelisk", "an obelisk on top", c=[0.0, 0.0], base=round(0.07 * L0, 2), z0=round(z, 2), h=round(ho, 2))
    C.summit = z + ho
    C.note("makeup", f"{n} tiers {L0:.0f} m, obelisk {ho:.0f} m, fins {nf} per face", "Dark order temple: a steep pyramid, fins, an obelisk", "type=, seed=")


def dark_citadel(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    L0 = uni(T["base"], rng) * k; W0 = L0 * uni(T["ratio"], rng); n = int(rng.integers(T["tiers"][0], T["tiers"][1] + 1)); Ht = L0 * uni(T["height"], rng)
    tiers, z = tiers_up(C, L0, W0, n, Ht, T["first"], uni(T["top"], rng), max(C.ed["batter"], 0.12), "bastions in tiers — a wall on a wall (Barad-dur)", slope=CANON["stair_slope"])
    for t in tiers:                                                              # buttress fins along the tier walls
        for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            e = (t["L"] if du else t["W"]) / 2; s_ = (t["W"] if du else t["L"]); nf = max(2, int(s_ / (12.0 * kq)))
            for j in range(nf):
                x = -s_ / 2 + (j + 0.5) * s_ / nf
                if du == -1 and abs(x) < 0.06 * L0 + 3 * kq: continue
                p = (du * e, x) if du else (x, dv * e); C.add("fin", "", p=[round(p[0], 2), round(p[1], 2)], n=[du, dv], z0=t["z0"], z1=round(t["z1"] + 2.0 * kq, 2),
                                                              d0=round(0.05 * (t["z1"] - t["z0"]) + 2 * kq, 2), d1=round(0.8 * kq, 2), w=round(1.4 * kq, 2))
    axial_flights(C, tiers, 0.06 * L0 + 2 * kq, start=0, shrink=0.85, why="the ascent along the axis through the gate of each tier")
    C.add("gate", "a pylon gate in the bottom tier", c=[round(-tiers[0]["L"] / 2, 2), 0.0], dir=[-1.0, 0.0], w=round(0.3 * W0, 2), h=round(tiers[0]["z1"] * 1.3, 2),
          d=round(0.08 * L0, 2), z0=0.0, kind="pylon")
    hm = uni(T["main"], rng) * z; Lt, Wt = tiers[-1]["L"], tiers[-1]["W"]; side = 0.45 * min(Lt, Wt)
    main = C.tower((0.0, 0.0), side, z, hm, crown=TS.choose({"horns": 0.4, "blades": 0.35, "spikes": 0.25}, rng), why="the main tower — an overwhelming vertical, horns or a crown")
    nt = int(rng.integers(T["towers"][0], T["towers"][1] + 1)); sat = []
    for i in range(nt):
        ti = tiers[max(0, n - 2 - (i % 2))]; a_ = math.pi / 4 + 2 * math.pi * i / nt; e = 0.42 * min(ti["L"], ti["W"])
        c = (e * math.cos(a_) * ti["L"] / min(ti["L"], ti["W"]), e * math.sin(a_) * ti["W"] / min(ti["L"], ti["W"]))
        sat.append(C.tower(c, 0.35 * side, ti["z1"], uni([0.45, 0.8], rng) * hm, why="a cluster of towers on the lower tiers" if i == 0 else ""))
    nbr = min(int(rng.integers(T["bridges"][0], T["bridges"][1] + 1)), len(sat))
    for i in range(nbr):                                                          # bridges between tower tops (Ravenloft)
        s_ = sat[i]; zb = min(s_["h"], main["h"]) - 0.25 * (min(s_["h"], main["h"]) - s_["z0"])
        C.add("bridge", "bridges between towers" if i == 0 else "", p0=list(main["c"]), p1=list(s_["c"]), z=round(zb, 2), w=round(2.5 * kq, 2), rail=1.0, high=True)
    C.summit = main["h"]
    C.note("makeup", f"{n} tiers {L0:.0f} m, main tower up to {main['h']:.0f} m, towers {nt}, bridges {nbr}", "dark citadel: bastions in tiers, a cluster of towers, bridges", "type=, seed=")


def axial_temple(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq
    Wd = uni(T["width"], rng) * k; ph = uni(T["pylon_h"], rng) * Wd; pd = 0.17 * Wd; Lc = uni(T["court"], rng) * Wd; Lh = uni(T["hall"], rng) * Wd
    Hh = uni(T["hall_h"], rng) * Wd; Ls = uni(T["sanctuary"], rng) * Wd; La = uni(T["avenue"], rng) * Wd; u = 0.0; col = C.ed["column"]
    for j in range(max(3, int(La / (5.5 * kq)))):                                # an avenue of sphinxes
        for sgn in (1, -1): C.add("ring", "an avenue of sphinxes to the pylon" if (j == 0 and sgn > 0) else "", c=[round(-pd - 8 * kq - j * 5.5 * kq, 2), round(sgn * (0.14 * Wd + 2 * kq), 2)],
                                  r=0.0, n=1, obj="statue", h=round(2.4 * kq, 2), w=round(1.6 * kq, 2), z0=0.0)
    if rng.random() < T["obelisks"]:
        for sgn in (1, -1): C.add("obelisk", "a pair of obelisks in front of the pylon" if sgn > 0 else "", c=[round(-pd / 2 - 4 * kq, 2), round(sgn * 0.2 * Wd, 2)], base=round(0.04 * Wd, 2), z0=0.0,
                                  h=round(0.55 * ph, 2))
    C.add("pylon", "pylon — the gate into the world of the temple: two towers with a batter and an entrance between them", c=[round(u, 2), 0.0], w=round(Wd, 2), h=round(ph, 2), d=round(pd, 2), z0=0.0)
    u += pd / 2
    C.add("court", "an open court with colonnades on the sides", c=[round(u + Lc / 2, 2), 0.0], L=round(Lc, 2), W=round(0.92 * Wd, 2), shape="rect", depth=0.0, terraces=0,
          center="altar" if rng.random() < 0.4 else "none", z0=0.0, colonnade=True, column=col, h=round(0.4 * ph, 2))
    u += Lc
    C.add("pylon", "the second pylon: wider than the court — it closes its end, the entrance is only through the gate", c=[round(u + pd * 0.4, 2), 0.0], w=round(max(0.85 * Wd, 0.92 * Wd + 3.0 * kq), 2),
          h=round(0.85 * ph, 2), d=round(pd * 0.8, 2), z0=0.0)
    u += pd * 0.8
    nx = max(4, int(Lh / (3.6 * kq * 1.2))); ny = max(4, int(0.9 * Wd / (3.6 * kq * 1.2))); ny += ny % 2   # an even number of rows: the procession axis runs between the columns
    C.add("hall", f"hypostyle hall: a forest of columns {nx} × {ny}, the middle nave is taller — lattice windows (Karnak: 134 columns, 102 × 53 m)", c=[round(u + Lh / 2, 2), 0.0],
          L=round(Lh, 2), W=round(0.9 * Wd, 2), H=round(Hh, 2), Hs=round(0.62 * Hh, 2), nx=nx, ny=ny, column=col, z0=0.0)
    u += Lh
    C.add("shrine", "the sanctuary deep inside — darker and lower", c=[round(u + Ls / 2, 2), 0.0], L=round(Ls, 2), W=round(0.55 * Wd, 2), H=round(0.5 * Hh, 2), z0=0.0, roof="flat",
          dir=[-1.0, 0.0], comb=0.0)
    u += Ls
    C.add("wall", "an enclosure around the temple: the entrance is only through the first pylon", poly=wall_poly(rect((u + 0.15 * Wd) / 2, 0.0, u + 0.15 * Wd, 1.35 * Wd), [(0.0, 0.0)], Wd / 2),
          z0=0.0, h=round(0.25 * ph, 2), t=round(1.5 * kq, 2), edge="none", gates=[[0.0, 0.0]])
    C.summit = ph
    C.note("makeup", f"width {Wd:.0f} m, length along the axis {u:.0f} m; pylons {ph:.0f} m, hall {nx}×{ny} columns, vault {Hh:.0f} m",
           "axial temple: avenue → pylon → court → pylon → hypostyle → sanctuary, ever darker and lower toward the sanctuary", "type=, seed=")


def domed(C):
    T = C.typ; rng = C.rng; k, kq = C.k, C.kq; Sp = C.Sp; Cc = Sp.get("central", {})
    kind = Cc.get("kind") if isinstance(Cc.get("kind"), str) else TS.choose(T["kinds"], rng)
    W = uni(Cc.get("W", [10.0, 16.0]), rng) * k; H = W * uni(Cc.get("HW", [1.1, 1.4]), rng); arm = uni(Cc.get("arm", [0.5, 0.8]), rng) * W
    drum = uni(Cc.get("drum_k", [0.3, 0.5]), rng) * W; dk = Cc.get("dome", "hemi") if isinstance(Cc.get("dome", "hemi"), str) else TS.choose(Cc["dome"], rng)
    dh = W / 2 * uni(Cc.get("dome_k", [0.9, 1.0]), rng); corner = int(Cc.get("corner_domes", 0) if not isinstance(Cc.get("corner_domes"), dict) else TS.choose(Cc["corner_domes"], rng))
    t = Sp.get("t", 1.6) * (k if k > 1 else 1.0)
    C.add("domed", f"centric: {kind} — a dome on pendentives over the middle square (Hagia Sophia); arms under barrel vaults", kind=kind, c=[0.0, 0.0], W=round(W, 2),
          arm=round(arm if kind != "rotunda" else 0.0, 2), H=round(H, 2), t=round(t, 2), drum=round(drum, 2), dome=dk, dh=round(dh, 2), corner=corner if kind == "cross_in_square" else 0,
          apses=int(Cc.get("apses", 3)) if kind != "rotunda" else 0, narthex=round(0.8 * arm if kind != "rotunda" else 0.3 * W, 2),
          windows=int(rng.integers(*Cc.get("drum_windows", [12, 16]))), fins=int(rng.integers(*Cc.get("fins", [0, 1]))) if Cc.get("fins") else 0,
          stacks=int(rng.integers(*Cc.get("stacks", [0, 1]))) if Cc.get("stacks") else 0, pitch=float(Sp.get("roof_pitch", 24)))
    C.summit = H + drum + dh
    C.note("makeup", f"{kind}: dome {W:.0f} m, summit {C.summit:.0f} m" + (f", domes {1 + corner}" if corner else ""), "centric: a dome on a drum, arms, apses, a narthex",
           "type=, central.kind=")


BUILD = {"ziggurat": ziggurat, "step_pyramid": step_pyramid, "temple_mountain": temple_mountain, "spiral_tower": spiral_tower, "occult_ring": occult_ring,
         "light_order_temple": light_order_temple, "dark_order_temple": dark_order_temple, "dark_citadel": dark_citadel, "axial_temple": axial_temple, "domed": domed}


def extent(p):
    """Extent of a part in plan [u0, u1, v0, v1]."""
    t = p["type"]
    if "poly" in p: P = np.asarray(p["poly"], float); return [P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()]
    if t == "water": P = np.asarray(p["outer"], float); return [P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()]
    if t in ("flight", "bridge"): P = np.asarray([p["p0"], p["p1"]], float); w = p["w"] / 2 + 1; return [P[:, 0].min() - w, P[:, 0].max() + w, P[:, 1].min() - w, P[:, 1].max() + w]
    if t == "fin": return [p["p"][0] - p["d0"] - 1, p["p"][0] + p["d0"] + 1, p["p"][1] - p["d0"] - 1, p["p"][1] + p["d0"] + 1]
    c = p.get("c", [0.0, 0.0])
    r = {"tower": p.get("side", 0) * 0.75, "prang": p.get("half", 0) * 1.3, "minaret": p.get("r", 0) * 1.6, "obelisk": p.get("base", 0) * 1.5,
         "ring": p.get("r", 0) + p.get("w", 0), "court": max(p.get("r", 0), p.get("L", 0) / 2, p.get("W", 0) / 2), "cone": p.get("r0", 0) + 3, "ramp": max(p.get("r_out", [0])) + 1,
         "drum": p.get("r", 0) + 1, "gate": max(p.get("w", 0), p.get("d", 0)) / 2 + 1, "pylon": max(p.get("w", 0), p.get("d", 0)) / 2 + 1,
         "shrine": max(p.get("L", 0), p.get("W", 0)) / 2 + 1, "hall": max(p.get("L", 0), p.get("W", 0)) / 2 + 1,
         "domed": (p.get("W", 0) / 2 + p.get("arm", 0) + p.get("narthex", 0) + p.get("W", 0) / 2 + 2) if p.get("kind") else 0}.get(t, 0.0)
    return [c[0] - r, c[0] + r, c[1] - r, c[1] + r]


def plan(folder, out=None, type=None, style="mesopotamian", seed=1, at=None, **kw):
    rng = np.random.default_rng(int(seed)); logic = []

    def note(step, what, value, why, alt=None):
        logic.append({"step": step, "what": what, "value": value, "why": why, **({"otherwise": alt} if alt else {})})
    over = TS.parse_over(kw); S = TS.resolve(style, over); Sp = TS.pick(S, AL.irng(seed, S["_name"]))
    ed = dict(ED_DEF); ed.update({k_: v for k_, v in (Sp.get("ed") or {}).items()})
    typ = type or ed.get("type") or "ziggurat"
    if typ not in BUILD: raise SystemExit(f"unknown type {typ}: {', '.join(BUILD)}")
    k = uni(Sp.get("scale", 1.0), rng); kq = k ** 0.5
    note("intent", f"{TYPE_RU[typ]} in the style {S['_name']}", f"edge {ed['edge']}, columns {ed['column']}, shrine roof {ed['shrine_roof']}, batter {ed['batter']}, "
         f"tower tops {Sp['tower']['spire']}" + (f"; scale ×{k:.1f}" if k > 1 else ""),
         "type — which parts and how they stand; style — what they are made of (arch, tower tops, palette, spikes); type × style gives something new (a ziggurat in the 'dark' style — "
         "an occult ziggurat)", "type=, style=")
    C = Ctx(Sp, ed, TYPES[typ], rng, k, note, seed)
    BUILD[typ](C)
    ex = np.array([extent(p) for p in C.parts if p["type"] != "flight_core"]); u_lo, u_hi, v_lo, v_hi = ex[:, 0].min(), ex[:, 1].max(), ex[:, 2].min(), ex[:, 3].max()
    # ---- place: platform, axis, leveling (as for temples)
    G = Ground(folder); area = (u_hi - u_lo + 12) * (v_hi - v_lo + 12)
    Ssite = CS.read(folder, area=min(area * 0.8, 60000.0), at=at, g=G)
    c = np.asarray(Ssite["centre"], float); ax = np.array([1.0, 0.0]); best = -1.0
    ax_site = np.asarray(Ssite.get("axis", [1.0, 0.0]), float); ax_site /= max(np.linalg.norm(ax_site), 1e-9)
    for cand in (ax_site, -ax_site):
        if cand[0] >= math.cos(math.radians(35)) and cand[0] > best: ax, best = cand.copy(), cand[0]
    pv = np.array([-ax[1], ax[0]]); origin = c - ax * (u_lo + u_hi) / 2 - pv * (v_lo + v_hi) / 2
    note("place", "platform and axis", f"{Ssite['type_ru']}, axis at {math.degrees(math.atan2(ax[1], ax[0])):+.0f}° from east", "approach from the west, the axis toward the long side of the platform", "at=")
    pad = 8.0 * kq; Hn = G.Hn.copy(); b = G.near_bounds(); ny, nx = Hn.shape
    X, Y = np.meshgrid(b[0] + np.arange(nx) * G.nres, b[1] + np.arange(ny) * G.nres)
    def to_local(X_, Y_): return (X_ - origin[0]) * ax[0] + (Y_ - origin[1]) * ax[1], (X_ - origin[0]) * pv[0] + (Y_ - origin[1]) * pv[1]
    def dout(U_, V_): return np.maximum.reduce([np.zeros_like(U_), (u_lo - pad) - U_, U_ - (u_hi + pad), (v_lo - pad) - V_, V_ - (v_hi + pad)])
    U, V = to_local(X, Y); m = dout(U, V) <= 0; mb = (U > u_lo) & (U < u_hi) & (V > v_lo) & (V < v_hi)
    zmed = float(np.median(Hn[m])) if m.any() else float(G.height(*c)); zhi = float(np.percentile(Hn[mb], 99.5)) if mb.any() else zmed
    z0 = max(zmed, zhi - 8.0)
    big = max(u_hi - u_lo, v_hi - v_lo); clip_ = max(8.0, 0.06 * big); fe_ = max(14.0, 0.06 * big)   # for a giant — cut and fill scaled to its size
    def level(Hg, d_):
        f = np.clip(1 - d_ / fe_, 0, 1); f = f * f * (3 - 2 * f)
        return Hg - np.clip(Hg - z0, -clip_, clip_) * f
    Hn = level(Hn, dout(U, V)); above = float(np.percentile(Hn[mb], 99.5) - z0) if mb.any() else 0.0; under = float(z0 - np.min(Hn[mb])) if mb.any() else 0.0
    def carve(Hg, U_, V_):                                                       # moat: the bottom 3 m below the platform, batters 2 m; a sunken circle — down to its own floor
        for p in C.parts:
            if p["type"] == "court" and p.get("depth", 0) > 0 and p.get("shape") != "rect":
                rr = np.hypot(U_ - p["c"][0], V_ - p["c"][1]); f = np.clip((p["r"] + 0.5 - rr) / 1.0, 0, 1)
                Hg = np.where(rr < p["r"] + 0.5, Hg - (Hg - (z0 - p["depth"] - 1.0)) * f, Hg); continue
            if p["type"] != "water": continue
            Po, Pi = np.asarray(p["outer"]), np.asarray(p["inner"])
            do = np.maximum.reduce([Po[:, 0].min() - U_, U_ - Po[:, 0].max(), Po[:, 1].min() - V_, V_ - Po[:, 1].max()])   # >0 — outside the moat
            di = np.maximum.reduce([Pi[:, 0].min() - U_, U_ - Pi[:, 0].max(), Pi[:, 1].min() - V_, V_ - Pi[:, 1].max()])   # >0 — outside the island
            inside = np.minimum(-do, di); f = np.clip(inside / 2.0, 0, 1)
            Hg = np.where(inside > 0, Hg - (Hg - (z0 - 3.0)) * f, Hg)
        return Hg
    Hn = carve(Hn, U, V)
    def gz_(u_, v_):                                                             # ground after leveling (local u, v)
        xy = origin + ax * u_ + pv * v_; i_ = int(round((xy[1] - b[1]) / G.nres)); j_ = int(round((xy[0] - b[0]) / G.nres))
        return float(Hn[min(max(i_, 0), ny - 1), min(max(j_, 0), nx - 1)])
    fl_rel = CANON["plinth"]; feet = 0                                           # feet down to the ground: on a slope the flight is longer, at the gate — steps
    for p in [p for p in C.parts if p["type"] == "flight" and min(p["z0"], p["z1"]) <= 0.05 and max(p["z0"], p["z1"]) > 0.5]:
        lo_, hi_ = ("p0", "p1") if p["z0"] <= p["z1"] else ("p1", "p0"); zl = min(p["z0"], p["z1"]); d_ = np.subtract(p[lo_], p[hi_]); d_ = d_ / max(np.linalg.norm(d_), 1e-9)
        g_ = min(gz_(*p[lo_]), gz_(*(np.asarray(p[lo_]) + d_ * 2.0))) - z0 - fl_rel
        if g_ < zl - 0.3:
            run_ = (zl - g_) / math.tan(math.radians(max(p["slope"], 30.0))); p[lo_] = [round(p[lo_][0] + d_[0] * run_, 2), round(p[lo_][1] + d_[1] * run_, 2)]
            p["z0" if lo_ == "p0" else "z1"] = round(g_, 2); feet += 1
    for p in [p for p in C.parts if p["type"] == "bridge" and not p.get("high")]:   # a causeway over a lowland — steps from the bank
        d_ = np.subtract(p["p0"], p["p1"]); d_ = d_ / max(np.linalg.norm(d_), 1e-9); q_ = np.asarray(p["p0"], float); g_ = min(gz_(*q_), gz_(*(q_ + d_ * 2.0))) - z0 - fl_rel
        if g_ < p["z"] - 0.6:
            run_ = (p["z"] - g_) / math.tan(math.radians(33.0)); C.flight(tuple(q_ + d_ * run_), tuple(q_), g_, p["z"], p["w"], why="steps from the bank to the causeway"); feet += 1
    for p in [p for p in C.parts if p["type"] == "domed"]:                      # centric on a slope: steps to the west door
        h_ = p["W"] / 2 + p["arm"] if p["kind"] != "rotunda" else p["W"] / 2 + p["t"] / 2; q_ = np.array([p["c"][0] - h_ - p["narthex"], p["c"][1]]); d_ = np.array([-1.0, 0.0])
        g_ = min(gz_(*q_), gz_(*(q_ + d_ * 2.0))) - z0 - fl_rel
        if g_ < -0.6:
            run_ = -g_ / math.tan(math.radians(33.0)); C.flight(tuple(q_ + d_ * run_), tuple(q_), g_, 0.0, min(0.3 * p["W"], 3.5 * kq) + 1.0, why="steps to the door on a slope"); feet += 1
    gates_ = [g for p in C.parts if p["type"] == "wall" for g in p.get("gates", [])]
    for p in [p for p in C.parts if p["type"] == "gate" and p.get("kind") == "pylon" and any(math.dist(p["c"], g) < 1.0 for g in gates_)]:
        d_ = np.asarray(p["dir"], float); q_ = np.asarray(p["c"], float) + d_ * p["d"] / 2; g_ = min(gz_(*q_), gz_(*(q_ + d_ * 2.0))) - z0 - fl_rel
        if g_ < -0.3:
            run_ = -g_ / math.tan(math.radians(33.0)); C.flight(tuple(q_ + d_ * run_), tuple(q_), g_, 0.0, 0.18 * p["w"], why="steps to the gate on a slope"); feet += 1
    cut = 0                                                                      # the approach hit a slope — a cut: the bottom at foot level, 45° batters
    ent = [(np.asarray(p["p0"], float), np.asarray(p["p0"], float) - np.asarray(p["p1"], float), p["z"], p["w"]) for p in C.parts if p["type"] == "bridge" and not p.get("high")]
    ent += [(np.asarray(p["p0"] if p["z0"] <= p["z1"] else p["p1"], float), np.subtract(p["p0"], p["p1"]) if p["z0"] <= p["z1"] else np.subtract(p["p1"], p["p0"]),
             min(p["z0"], p["z1"]), p["w"]) for p in C.parts if p["type"] == "flight" and min(p["z0"], p["z1"]) <= 0.05]
    for q_, d_, zf, w_ in ent:
        d_ = d_ / max(np.linalg.norm(d_), 1e-9); zf_w = z0 + fl_rel + zf
        if gz_(*(q_ + d_ * 2.0)) <= zf_w + 0.3: continue
        Uq = (U - q_[0]) * d_[0] + (V - q_[1]) * d_[1]; Xq = np.abs(-(U - q_[0]) * d_[1] + (V - q_[1]) * d_[0])
        m_ = (Uq > -1.0) & (Uq < 80.0); cap = zf_w + np.maximum(0.0, Xq - w_ / 2 - 1.0)
        Hn = np.where(m_ & (Hn > cap), cap, Hn); cut += 1
    if feet or cut: note("place", "feet and approaches", f"extended to the ground: {feet}; cuts: {cut}", "on a slope the flight starts below the platform — longer, at the gate — "
                                                                                    "steps; an uphill approach — a cut at foot level", "at=")
    out = out or os.path.join(ROOT, "out/edifice", os.path.basename(folder.rstrip("/")) + f"_{typ}_{S['_name'].replace(':', '_').replace(',', '_')}_{seed}")
    os.makedirs(out, exist_ok=True)
    zz = dict(np.load(os.path.join(folder, "relief.npz"))); zz["near"] = Hn.astype(zz["near"].dtype)
    fb = G.far_bounds(); Fy, Fx = G.Hf.shape; FX, FY = np.meshgrid(fb[0] + np.arange(Fx) * G.fres, fb[1] + np.arange(Fy) * G.fres); FU, FV = to_local(FX, FY)
    zz["far"] = carve(level(G.Hf, np.maximum(dout(FU, FV) - G.fres, 0.0)), FU, FV).astype(zz["far"].dtype)
    np.savez_compressed(os.path.join(out, "relief.npz"), **zz); shutil.copy(os.path.join(folder, "location.json"), os.path.join(out, "location.json"))
    note("place", "alignment", f"platform at mark {z0:.1f} m" + (f"; a substructure up to {under:.0f} m under the base" if under > 2 else ""),
         "the base on a level platform; on a slope — at the upper mark, with a substructure below; a 14 m batter beyond the edge", "at=")
    # ---- checks
    checks = []
    def chk(what, ok, **kw_): checks.append(dict({"what": what, "ok": bool(ok)}, **kw_))
    steep = 72.0 if typ in ("step_pyramid", "temple_mountain") else 52.0
    sl = [f["slope"] for f in C.flights if f["z1"] > f["z0"]]
    chk(f"flights are passable (slope ≤ {steep:.0f}°)", not sl or max(sl) <= steep + 1e-6, largest=max(sl) if sl else 0)
    tops = sorted({round(p["z1"], 1) for p in C.parts if p["type"] == "tier"})
    if C.flights and tops and typ not in ("occult_ring", "spiral_tower", "axial_temple", "domed"):
        reach = max([f["z1"] for f in C.flights] + [p["z1"] for p in C.parts if p["type"] == "stair_in"]) + max([p["z1"] - p["z0"] for p in C.parts if p["type"] == "flight_core"] + [0.0])
        chk("the stairs reach the top platform", reach >= max(tops) - 0.6, top_m=max(tops), stairs_m=round(reach, 1))
    tw = [p for p in C.parts if p["type"] in ("tower", "minaret", "prang", "obelisk")]
    ov = 0
    for i in range(len(tw)):
        for j in range(i + 1, len(tw)):
            a_, b_ = tw[i], tw[j]
            if abs(a_["z0"] - b_["z0"]) > 1.0: continue
            ra = a_.get("side", 0) / 2 or a_.get("half", 0) or a_.get("r", 0) or a_.get("base", 0) / 2
            rb = b_.get("side", 0) / 2 or b_.get("half", 0) or b_.get("r", 0) or b_.get("base", 0) / 2
            if math.dist(a_["c"], b_["c"]) < 0.9 * (ra + rb): ov += 1
    chk("towers do not cut into each other", ov == 0, n_intersections=ov)
    chk("the ground is not above the base, substructure ≤ 12 m", above <= 0.5 and under <= 12.0 * kq, above_m=round(above, 1), undercroft_m=round(under, 1))
    R = {"type": typ, "type_ru": TYPE_RU[typ], "style": S["_name"], "seed": int(seed), "k": round(k, 2), "place": folder, "out": out,
         "frame": {"origin": origin.round(3).tolist(), "ax": ax.round(5).tolist(), "pv": pv.round(5).tolist(), "z0": round(z0, 2), "floor": CANON["plinth"],
                   "u_lo": round(float(u_lo), 2), "u_hi": round(float(u_hi), 2), "v_lo": round(float(v_lo), 2), "v_hi": round(float(v_hi), 2), "under": round(under, 2)},
         "colors": Sp["colors"], "ed": ed, "spikes": float(Sp.get("spikes", 0.0)), "kq": round(kq, 3), "parts": C.parts, "logic": logic, "checks": checks,
         "tower_open": Sp["tower"].get("open", "pointed"), "arch": Sp.get("arch", "round") if isinstance(Sp.get("arch"), str) else "round",
         "site": {k_: Ssite[k_] for k_ in ("type", "type_ru", "centre", "area_m2")}}
    R["spaces"] = ESP.spaces(R, Ground(out).height).to_dict()                     # space: terraces, passages, checks — before the model
    checks += [dict(c_, what="space: " + c_["what"]) for c_ in R["spaces"]["checks"]]
    R["stats"] = {"type": TYPE_RU[typ], "style": S["_name"], "length_m": round(float(u_hi - u_lo), 1), "width_m": round(float(v_hi - v_lo), 1),
                  "highest_m": round(C.summit, 1), "n_parts": len(C.parts), "n_flights": len(C.flights), "n_rooms": R["spaces"]["stats"]["n_cells"],
                  "n_entrances": R["spaces"]["stats"]["n_entrances"], "checks": f"{sum(c_['ok'] for c_ in checks)}/{len(checks)}"}
    json.dump(R, open(os.path.join(out, "edifice.json"), "w"), ensure_ascii=False, default=float)
    return R


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:])
    at = kw.pop("at", None)
    if at and "," in at: at = [float(v) for v in at.split(",")]
    R = plan(a[0], out=kw.pop("out", None), at=at, **kw)
    print("__RESULT__", json.dumps({"out": R["out"], **R["stats"], "non_passing": [c["what"] for c in R["checks"] if not c["ok"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
