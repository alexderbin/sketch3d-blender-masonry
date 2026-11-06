"""Edifice in Blender from the plan edifice.json (edifices/edifice_plan.py): a builder for each kind of part —
tiers (batter, talud-tablero, pilasters, merlons, spikes, pinnacles along the edge), stair flights with
cheek walls, gates (an arch, a pylon), shrines (a flat roof, a Maya roof comb, a gable roof, a dome, a
pyramid, steps), towers by tiers (bl_arch.staged), prangs, minarets, obelisks, galleries on posts, a sunken
circle with steps (a well, an altar), rings of objects (obelisks, menhirs, pillars, sphinxes, braziers),
walls with gates, a moat, a causeway and bridges between towers, a cone and a spiral ramp, arcade tiers,
fins, a hypostyle hall, a centric domed temple (cross-in-square, Greek cross, a forge rotunda). Colors are a
vertex attribute by the style palette. The shared geometry is in base/bl_arch.py.

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/edifices/bl_edifice.py -- <plan folder> [views=aerial,front,side,top,inside,plan] [res=1280x800] [blend=1]
→ <folder>/view_<view>.png, edifice.blend (if blend=1), the __RESULT__ line

As a function over an open scene (a step-by-step build; the import reads and builds nothing):
    from components.buildings.edifices import bl_edifice as BE
    objs, T = BE.build_edifice(folder, sc=None, tag=True)       # load + build (without ground and the human-scale figure) + face markup
or in parts: BE.load(folder, kw=None) → BE.build(sc=None, with_ground=True, fresh=True, marker=True) → (part count, objects)."""
import json, math, os, sys, time

import bpy
import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.base import bl_arch as BA                                                             # noqa: E402
from components.buildings.space import bl_space as BS                                                            # noqa: E402
from components.buildings.space.space import Space                                                          # noqa: E402

# Plan state — filled by load(folder); on import the module reads and builds nothing (as a function over an
# open scene — build_edifice)
OUT = None; KW = {}; T = None; T0 = time.time(); ZB = -3.0; SP_LAST = None
WATER = BA.rc((0.12, 0.22, 0.30), "water"); MAN = (0.55, 0.12, 0.10)


def cols_of(c):
    """Palette with face roles (sk_role): stone — wall, bands — finish, floor, roof, metal, opening."""
    s = BA.rc(c["stone"], "wall")
    return {"stone": s, "dark": BA.rc(BA.shade(s, 0.82), "finish"), "vault": BA.shade(s, 1.07), "rib": BA.rc(BA.shade(s, 0.74), "finish"),
            "floor": BA.rc(BA.shade(s, 0.7), "floor"), "roof": BA.rc(c["roof"], "roof"), "accent": BA.rc(c.get("accent", c["roof"]), "metal"),
            "hole": BA.rc(c.get("hole", (0.05, 0.05, 0.06)), "opening")}


def load(folder, kw=None):
    """Read the edifice plan (edifice.json, relief.npz — in the plan folder) and fill the module state: the frame,
    the palette, fresh mesh accumulators, the enclosure gates. kw — the keys of the line (views, res, cam, hide,
    blend …), as after "--". Does not touch the scene."""
    global OUT, KW, T, T0, FR, ED, SPK, kq, ARCH, O, AXV, PVV, Z0, FL, ZB, C, AW, AR, AT, AD, AF, AWT, AM, GATES, SKIP, AX_ANG
    OUT = os.path.abspath(folder); KW = dict(kw or {})
    T = json.load(open(os.path.join(OUT, "edifice.json"))); T0 = time.time()
    FR = T["frame"]; ED = T["ed"]; SPK = float(T.get("spikes", 0.0)); kq = float(T["kq"]); ARCH = T.get("arch", "round")
    O = np.asarray(FR["origin"], float); AXV = np.asarray(FR["ax"], float); PVV = np.asarray(FR["pv"], float)
    Z0 = float(FR["z0"]); FL = Z0 + float(FR["floor"]); ZB = -3.0
    C = cols_of(T["colors"])
    AW, AR, AT, AD, AF, AWT = (BA.Acc(r) for r in ("wall", "roof", "wall", "finish", "floor", "water"))   # walls and tiers, roofs, towers, small things, floors and flights, water
    AM = BA.Acc()                                                                # human-scale figure (not a building)
    GATES = [g for p in T["parts"] if p["type"] == "wall" for g in p.get("gates", [])]
    SKIP = [0]; AX_ANG = math.atan2(AXV[1], AXV[0])



def gaps_at(z, w_pad=1.0):
    """Passages at mark z (above the base): [(world point, width)] — breaks of the parapet and the edge where the
    space scheme has a transition (the top of a flight, a terrace joint, a gate passage); without a scheme — at
    the tops of the flights."""
    out = []
    for t in (T.get("spaces") or {}).get("trans", []):
        if t["kind"] not in ("stair", "ramp", "opening", "gate", "arch") or t.get("hidden"): continue
        P = t["p"] if isinstance(t["p"][0], list) else [t["p"], t["p"]]
        for q, zq in ((P[0], t["za"]), (P[-1], t["zb"])):
            if abs(zq - Z(z)) < 0.05: out.append((np.asarray(q, float), t["w"] + 1.2 * kq + w_pad))
    if out or T.get("spaces"): return out
    for p in T["parts"]:
        if p["type"] != "flight": continue
        top, zt = (p["p1"], p["z1"]) if p["z1"] >= p["z0"] else (p["p0"], p["z0"])
        if abs(zt - z) < 0.05: out.append((L(*top), p["w"] + 2 * p.get("cheek", 0.0) + w_pad))
    return out


def L(u, v): return O + AXV * u + PVV * v
SKIP = [0]
def WP(*a, **k): SKIP[0] += BA.wall_panel(*a, **k)                             # a wall with openings; a dropped opening is counted
def D(du, dv): d = AXV * du + PVV * dv; return d / max(np.linalg.norm(d), 1e-9)
def Z(z): return FL + z
def L3(u, v, z): q = L(u, v); return (q[0], q[1], FL + z)
AX_ANG = 0.0                                                                     # load(): the axis angle


def skirt(P, z_top):
    """Foundation down to the ground under a tier/tower that stands on the base."""
    AW.quad_strip([(*p, Z(ZB)) for p in P + [P[0]]], [(*p, Z(z_top)) for p in P + [P[0]]], C["stone"])


def edge_deco(Pt, z, kind, closed=True):
    """Tier edge: merlons, spikes, pinnacles (closed=False — an open polyline between the passages)."""
    if kind == "merlons": BA.merlon_row(AW, Pt, z, 0.9 * kq, C["stone"], closed=closed)
    elif kind == "spikes": BA.spike_row(AD, Pt, z, 4.5 * kq, 0.55 * kq, (2.2 + 2.0 * SPK) * kq, C["dark"], closed=closed)
    elif kind == "pinnacles":
        for p, q in (zip(Pt, Pt[1:] + Pt[:1]) if closed else zip(Pt[:-1], Pt[1:])):
            Lq = float(np.linalg.norm(np.asarray(q) - np.asarray(p))); n = max(1, int(Lq / (10.0 * kq)))
            for i in range(n): BA.pinnacle(AD, np.asarray(p) + (np.asarray(q) - np.asarray(p)) * i / n, z, 0.7 * kq, 1.6 * kq, C["stone"], C["roof"])


def b_tier(p):
    P = [L(*q) for q in p["poly"]]
    if p["z0"] < 0.5: skirt(P, p["z0"])
    gp = gaps_at(p["z1"])
    Pt = BA.tier(AW, P, Z(p["z0"]), Z(p["z1"]), p["batter"], C["stone"], col_top=C["floor"], parapet=p["parapet"], tablero=p.get("tablero"),
                 col_band=BA.shade(C["accent"], 1.0) if p.get("tablero") else None, gaps=gp)
    if p.get("pilasters"):                                                       # pilasters along the batter (Ur)
        h = p["z1"] - p["z0"]
        for (a_, b_), (at_, bt_) in zip(zip(P, P[1:] + P[:1]), zip(BA.offset_poly(P, p["batter"] * h), BA.offset_poly(P, p["batter"] * h)[1:] + BA.offset_poly(P, p["batter"] * h)[:1])):
            d = b_ - a_; Lq = float(np.linalg.norm(d)); n = int(Lq // p["pilasters"]); nr = np.array([d[1], -d[0]]) / max(Lq, 1e-9)
            for i in range(1, n):
                f = i / n; q0 = a_ + d * f; q1 = at_ + (bt_ - at_) * f; w = 0.45 * p["pilasters"] * 0.5
                tg = d / max(Lq, 1e-9); R0 = [q0 - tg * w, q0 + tg * w, q0 + tg * w + nr * 0.6 * kq, q0 - tg * w + nr * 0.6 * kq]
                R1 = [q1 - tg * w, q1 + tg * w, q1 + tg * w + nr * 0.6 * kq, q1 - tg * w + nr * 0.6 * kq]
                AW.quad_strip([(*q, Z(p["z0"])) for q in R0 + [R0[0]]], [(*q, Z(p["z1"]) - 0.2) for q in R1 + [R1[0]]], C["dark"])
    for run in BA.ring_runs(BA.offset_poly(Pt, 0.25 * p["parapet"]) if p["parapet"] else Pt, gp):   # edge — with passages at the flights
        edge_deco(run, Z(p["z1"] + p["parapet"]), p.get("edge", "none"), closed=False)


def b_flight(p):
    p0, p1, z0, z1 = p["p0"], p["p1"], p["z0"], p["z1"]
    if z1 < z0: p0, p1, z0, z1 = p1, p0, z1, z0
    BA.flight(AF, L(*p0), L(*p1), Z(z0), Z(z1), p["w"], BA.shade(C["stone"], 0.95), cheek=p["cheek"], col_cheek=C["dark"], ground=Z(min(ZB, z0 - 1.0)))


def b_gate(p):
    c = L(*p["c"]); d = D(*p["dir"]); h, w, dd = p["h"], p["w"], p["d"]
    if p.get("kind") == "pylon": BA.pylon(AW, AR, c, d, w, h, dd, Z(p["z0"]), C, blind=not any(math.dist(p["c"], g) < 1.0 for g in GATES)); return   # in an enclosure — a through passage
    tg = np.array([-d[1], d[0]])
    for s_ in (-1, 1):                                                           # arch: two pylons (along the passage — the full depth), a lintel with an arch
        AW.box(c + tg * s_ * w * 0.38, d, w * 0.24, dd, Z(p["z0"] - (0 if p["z0"] > 0.5 else 3)), Z(p["z0"] + h), C["stone"])
    WP(AW, c - tg * w * 0.26, c + tg * w * 0.26, Z(p["z0"] - 0.3), Z(p["z0"] + h), dd, [(w * 0.26, Z(p["z0"]), w * 0.36, 0.72 * h, ARCH if ARCH != "flat" else "round")], C["stone"])   # a wall below the threshold — an opening inside
    AW.face([(*(c + tg * x * w / 2 + d * y * dd / 2), Z(p["z0"] + h)) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], C["dark"])
    edge_deco([c + tg * x * w / 2 + d * y * dd / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], Z(p["z0"] + h), ED.get("edge", "none") if ED.get("edge") != "gallery" else "none")


def b_shrine(p):
    c = np.asarray(p["c"], float); Ls, Ws, H, z0 = p["L"], p["W"], p["H"], p["z0"]; d = np.asarray(p.get("dir", [-1, 0]), float)
    P = [(c[0] - Ls / 2, c[1] - Ws / 2), (c[0] + Ls / 2, c[1] - Ws / 2), (c[0] + Ls / 2, c[1] + Ws / 2), (c[0] - Ls / 2, c[1] + Ws / 2)]
    dh = min(0.62 * H, (4.5 + 1.0 * kq) * kq); dw = min(0.32 * Ws, 2.4 * kq); kind = ARCH if ARCH not in ("flat",) else "rect"
    for i in range(4):
        a_, b_ = P[i], P[(i + 1) % 4]; mid = ((a_[0] + b_[0]) / 2 - c[0], (a_[1] + b_[1]) / 2 - c[1]); Lq = math.dist(a_, b_)
        door = (mid[0] * d[0] + mid[1] * d[1]) > 0.4 * min(Ls, Ws)
        holes = [(Lq / 2, Z(z0), dw, dh, kind)] if door else ([(Lq * f, Z(z0 + 0.45 * H), min(0.12 * Lq, 1.2 * kq), 0.28 * H, kind) for f in (0.3, 0.7)] if Lq > 6 * kq else [])
        WP(AW, L(*a_), L(*b_), Z(z0 - 0.3), Z(z0 + H), 0.9 * kq, holes, C["stone"])
    Pw = [L(*q) for q in P]; zr = Z(z0 + H); roof = p.get("roof", "flat"); cw = L(*c)
    if roof in ("flat", "comb"):
        AR.face([(*q, zr) for q in BA.offset_poly(Pw, -0.4 * kq)], C["roof"]); AW.quad_strip([(*q, zr) for q in Pw + [Pw[0]]], [(*q, zr + 0.8 * kq) for q in Pw + [Pw[0]]], C["stone"])
        if roof == "comb" and p.get("comb"):                                     # Maya roof comb along the roof axis
            hc = p["comb"]; a_ = L(c[0] - 0.05 * Ls, c[1] - Ws * 0.42); b_ = L(c[0] - 0.05 * Ls, c[1] + Ws * 0.42)
            WP(AW, a_, b_, zr, zr + hc, 0.8 * kq, [(Ws * 0.84 * f, zr + 0.25 * hc, 0.08 * Ws, 0.45 * hc, "rect") for f in (0.2, 0.4, 0.6, 0.8)], C["accent"],
                          top=[(0.0, zr + hc * 0.75), (Ws * 0.42, zr + hc), (Ws * 0.84, zr + hc * 0.75)])
    elif roof == "gable": BA.roof_gable(AR, L(c[0] - Ls / 2, c[1]), L(c[0] + Ls / 2, c[1]), Ws / 2 + 0.4 * kq, zr, 35, C["roof"], ov=0.5 * kq)
    elif roof == "dome": BA.dome(AW, AR, cw, 0.42 * min(Ls, Ws), zr, 0.42 * min(Ls, Ws), "hemi", C)
    elif roof == "pyramid": BA.roof_fan(AR, (*cw, zr + 0.6 * min(Ls, Ws)), Pw + [Pw[0]], zr, C["roof"])
    elif roof == "stepped":
        for i in range(3):
            f = 0.92 - 0.24 * i; Q = [cw + (q - cw) * f for q in Pw]; AW.prism(Q, zr + i * 0.14 * H, zr + (i + 1) * 0.14 * H, C["stone"])
    if SPK >= 0.5: BA.spike_row(AD, Pw, zr + 0.8 * kq, 3.0 * kq, 0.45 * kq, (2.0 + 2.0 * SPK) * kq, C["dark"], closed=True)


def b_tower(p):
    c = L(*p["c"]); st = p["stages"]
    if p["z0"] < 0.5:
        P0 = [c + (AXV * x + PVV * y) * st[0]["half"] for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]; skirt(P0, p["z0"])
    BA.staged(AT, AR, c, AXV, st, p["spire"], C, z_off=Z(p["z0"]), s=kq, pinnacles=p.get("pinnacles", False), buttresses=p.get("buttresses", False),
              fort=ED.get("edge") == "merlons" and p["spire"]["kind"] == "flat", seed=float(p["c"][0] * 0.13 + p["c"][1] * 0.07), open_k=T.get("tower_open", "pointed"))


def b_ring(p):
    c = np.asarray(p["c"], float); n = int(p["n"]); obj = p["obj"]; h, w, z0 = p["h"], p["w"], p["z0"]
    a0 = p.get("a0", math.pi)
    pts = [c] if p["r"] <= 0 else [c + p["r"] * np.array([math.cos(a0 + 2 * math.pi * i / n), math.sin(a0 + 2 * math.pi * i / n)]) for i in range(n)]
    for i, q in enumerate(pts):
        qw = L(*q); seed = q[0] * 0.37 + q[1] * 0.11
        if obj == "obelisk": BA.obelisk(AD, AR, qw, w, Z(z0), h, C["dark"] if SPK > 0.3 else C["stone"], C["accent"], ax=AXV)
        elif obj == "menhir": BA.menhir(AD, qw, w * 1.4, h, Z(z0), BA.shade(C["stone"], 0.9), seed=seed)
        elif obj == "statue":                                                    # sphinx/guardian: pedestal, body, head
            AD.box(qw, AXV, w * 1.1, w * 2.4, Z(z0), Z(z0 + 0.35 * h), C["dark"]); AD.box(qw, AXV, w * 0.9, w * 2.0, Z(z0 + 0.35 * h), Z(z0 + 0.7 * h), C["stone"])
            AD.box(qw + AXV * (np.sign(-(q[1])) * 0 + 0.7 * w), AXV, w * 0.7, w * 0.7, Z(z0 + 0.7 * h), Z(z0 + h), C["stone"])
        elif obj == "altar": BA.altar(AD, qw, w, Z(z0), C["dark"], C["accent"])
        elif obj == "brazier":
            BA.column(AD, qw, 0.18 * w + 0.1, Z(z0), Z(z0 + 0.8 * h), C["dark"], "round"); BA.lathe(AD, qw, [(0.2 * w, 0.0), (0.5 * w, 0.2 * h * 0.3)], Z(z0 + 0.8 * h), C["dark"], n=12)
            AD.face([(*(qw + 0.45 * w * np.array([math.cos(2 * math.pi * j / 12), math.sin(2 * math.pi * j / 12)])), Z(z0 + 0.86 * h)) for j in range(12)], C["hole"])
        else: BA.column(AD, qw, 0.3 * w, Z(z0), Z(z0 + h), C["stone"], ED.get("column", "round") if ED.get("column") != "papyrus" else "round")


def b_court(p):
    c = np.asarray(p["c"], float)
    if p.get("shape") == "rect":                                                 # court of the axial temple: paving, a colonnade along the long sides, walls
        Ls, Ws, h = p["L"], p["W"], p.get("h", 8.0)
        AF.face([L3(c[0] + x * Ls / 2, c[1] + y * Ws / 2, 0.02) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], C["floor"])
        for sgn in (1, -1):
            WP(AW, L(c[0] - Ls / 2, c[1] + sgn * Ws / 2), L(c[0] + Ls / 2, c[1] + sgn * Ws / 2), Z(ZB), Z(h), 1.4 * kq, [], C["stone"])
            n = max(4, int(Ls / (4.0 * kq))); vc = c[1] + sgn * (Ws / 2 - 0.16 * Ws)
            for j in range(n + 1): BA.column(AD, L(c[0] - Ls / 2 + Ls * j / n, vc), 0.55 * kq, Z(0.0), Z(0.82 * h), C["stone"], ED.get("column", "round"))
            AR.face([L3(c[0] - Ls / 2, c[1] + sgn * Ws / 2, 0.82 * h + 0.6 * kq), L3(c[0] + Ls / 2, c[1] + sgn * Ws / 2, 0.82 * h + 0.6 * kq),
                     L3(c[0] + Ls / 2, vc - sgn * 0.6 * kq, 0.82 * h + 0.6 * kq), L3(c[0] - Ls / 2, vc - sgn * 0.6 * kq, 0.82 * h + 0.6 * kq)], C["roof"])
        if p.get("center") == "altar": BA.altar(AD, L(*c), 1.6 * kq, Z(0.0), C["dark"], C["accent"])
        return
    R, rf, dep, nt = p["r"], p["r_floor"], p["depth"], max(1, int(p["terraces"])); cw = L(*c); n = 64
    def ring(r, z): return [(*(cw + r * np.array([math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)])), z) for i in range(n)]
    dsc = [q for q in T["parts"] if q["type"] == "flight" and q["z1"] < q["z0"]]   # descent into the circle: the angle and half-width of the passage
    gap = [(math.atan2(*(L(*q["p0"]) - cw)[::-1]), (q["w"] / 2 + 0.8) / max(R, 1.0)) for q in dsc]
    def strip_gap(A_, B_, col, acc):
        for i in range(n):
            am = 2 * math.pi * (i + 0.5) / n
            if any(abs((am - g0 + math.pi) % (2 * math.pi) - math.pi) < hw for g0, hw in gap): continue
            j = (i + 1) % n; acc.face([A_[i], A_[j], B_[j], B_[i]], col)
    for i in range(nt):                                                          # terrace steps down to the floor of the circle
        r0 = R - (R - rf) * i / nt; r1 = R - (R - rf) * (i + 1) / nt; z_a = -dep * i / nt; z_b = -dep * (i + 1) / nt
        AF.quad_strip(ring(r0, Z(z_a)), ring(r1, Z(z_a)), BA.shade(C["stone"], 0.92 - 0.04 * (i % 2)), closed=True)
        strip_gap(ring(r1, Z(z_a)), ring(r1, Z(z_b)), C["dark"], AF)              # risers — with a passage at the descent
    AF.face(ring(rf, Z(-dep))[::-1], C["floor"])
    strip_gap(ring(R + 0.8 * kq, Z(ZB)), ring(R + 0.8 * kq, Z(0.6 * kq)), C["stone"], AW)   # the circle's curb — with a passage at the descent
    strip_gap(ring(R + 0.8 * kq, Z(0.6 * kq)), ring(R, Z(0.6 * kq)), C["stone"], AW); strip_gap(ring(R, Z(0.6 * kq)), ring(R, Z(0.0)), C["stone"], AW)
    if p["center"] == "pit":                                                     # a well into darkness
        rp = 0.3 * rf; AW.quad_strip(ring(rp, Z(-dep)), ring(rp, Z(-dep + 1.0 * kq)), C["stone"], closed=True)
        AW.quad_strip(ring(rp * 0.92, Z(-dep + 1.0 * kq)), ring(rp * 0.92, Z(-dep - 20.0)), (0.02, 0.02, 0.03), closed=True)
        AWT.face(ring(rp * 0.92, Z(-dep - 6.0)), C["hole"])
    elif p["center"] == "altar": BA.altar(AD, cw, 0.12 * rf + 1.2 * kq, Z(-dep), C["dark"], C["accent"])


def b_wall(p):
    P = [np.asarray(q, float) for q in p["poly"]]; n = len(P); gates = [np.asarray(g, float) for g in p.get("gates", [])]; h, t = p["h"], p["t"]
    gw = max([0.0] + [g_["w"] for g_ in T["parts"] if g_["type"] == "gate"]) * 0.55 or 6.0 * kq
    for i in range(n):
        a_, b_ = P[i], P[(i + 1) % n]; mid = (a_ + b_) / 2
        if any(np.linalg.norm(mid - g) < gw for g in gates): continue
        WP(AW, L(*a_), L(*b_), Z(ZB), Z(h), t, [], C["stone"])
        e = ED.get("edge", "none")
        if e == "spikes": BA.spike_row(AD, [L(*a_), L(*b_)], Z(h), 3.0 * kq, 0.45 * kq, (1.8 + 1.6 * SPK) * kq, C["dark"])
        elif e == "merlons": BA.merlon_row(AW, [L(*a_), L(*b_)], Z(h), 0.8 * kq, C["stone"], closed=False)


def b_water(p):
    Po = [L(*q) for q in p["outer"]]; Pi = [L(*q) for q in p["inner"]]
    for i in range(4): AWT.face([(*Po[i], Z(p["z"])), (*Po[(i + 1) % 4], Z(p["z"])), (*Pi[(i + 1) % 4], Z(p["z"])), (*Pi[i], Z(p["z"]))], WATER)


def b_bridge(p):
    a_, b_ = L(*p["p0"]), L(*p["p1"]); d = b_ - a_; Lq = float(np.linalg.norm(d)); d /= max(Lq, 1e-9); mid = (a_ + b_) / 2; w = p["w"]; z = Z(p["z"]); nn = np.array([-d[1], d[0]])
    if p.get("high"):                                                            # bridge between towers: deck, railings, an arch below
        AT.box(mid, d, w, Lq, z - 0.8 * kq, z, C["stone"])
        for s_ in (-1, 1): AT.box(mid + nn * s_ * w / 2, d, 0.3 * kq, Lq, z, z + 1.1 * kq, C["dark"])
        pts = [(*(a_ + d * Lq * f), z - 0.8 * kq - 0.18 * Lq * math.sin(math.pi * f)) for f in np.linspace(0, 1, 13)]
        BA.tube(AT, pts, 0.35 * kq + 0.02 * Lq, C["stone"], n=6)
    else:                                                                        # causeway: an embankment with balustrades (nagas)
        AF.box(mid, d, w, Lq, Z(ZB), z, BA.shade(C["stone"], 0.9))
        for s_ in (-1, 1): AF.box(mid + nn * s_ * (w / 2 - 0.3), d, 0.5 * kq, Lq, z, z + 0.9 * kq, C["dark"])


def b_cone(p):
    BA.lathe(AW, L(*p["c"]), [(p["r0"], 0.0), (p["r1"], p["z1"] - p["z0"])], Z(p["z0"]), C["stone"], n=40)
    AW.face([(*(L(*p["c"]) + p["r1"] * np.array([math.cos(2 * math.pi * i / 40), math.sin(2 * math.pi * i / 40)])), Z(p["z1"])) for i in range(40)], C["floor"])


def b_ramp(p):
    BA.spiral(AW, L(*p["c"]), p["r_in"], p["r_out"], Z(p["z0"]), Z(p["z1"]), p["a0"] + AX_ANG, p["turns"], BA.shade(C["stone"], 0.95), parapet=1.0 * kq)


def b_drum(p):
    """Arcade tier (Babylon) or pavilion (Samarra): a ring of faces with through arches around a solid core."""
    c = L(*p["c"]); r, z0, h, n = p["r"], p["z0"], p["h"], int(p["n"]); P = BA.octagon(c, r, phase=0.0, n=n); roof = p.get("roof", "flat")
    for i in range(n):
        a_, b_ = P[i], P[(i + 1) % n]; Lq = float(np.linalg.norm(b_ - a_))
        ha = 0.7 * h; top = None
        if roof == "broken":                                                     # collapsed top: the arch is below the edge of the break
            f_ = (math.sin(i * 12.9898 + r) * 43758.5453) % 1.0; f2 = (f_ * 7.3) % 1.0; top = [(0.0, Z(z0 + h * (0.55 + 0.45 * f_))), (Lq, Z(z0 + h * (0.55 + 0.45 * f2)))]
            ha = min(ha, h * (0.55 + 0.45 * min(f_, f2)) - 0.12 * h - 0.6 * kq)
        holes = [(Lq / 2, Z(z0 + 0.12 * h), 0.55 * Lq, ha, ARCH if ARCH not in ("flat", "stepped") else "round")] if p.get("arcade", True) and Lq > 1.2 and ha > 1.0 else []
        WP(AW, a_, b_, Z(z0 - (0.0 if z0 > 0.5 else 3.0)), Z(z0 + h), 0.9 * kq, holes, C["stone"], top=top)
    Pc = BA.octagon(c, r * 0.72, phase=0.0, n=n); AW.prism(Pc, Z(z0), Z(z0 + h * (0.9 if roof == "broken" else 1.0)), C["dark"])
    if roof == "flat": AW.face([(*q, Z(z0 + h)) for q in P], C["floor"])
    elif roof == "dome": BA.dome(AW, AR, c, r * 0.95, Z(z0 + h), r * 0.9, "hemi", C, lantern=False)


def b_fin(p):
    BA.fin(AD, L(*p["p"]), D(*p["n"]), Z(p["z0"] - (0 if p["z0"] > 0.5 else 3.0)), Z(p["z1"]), p["d0"], p["d1"], p["w"], C["dark"])


def b_pylon(p):
    BA.pylon(AW, AR, L(*p["c"]), AXV, p["w"], p["h"], p["d"], Z(p["z0"]), C, blind=False)   # pylon of the axial temple — a through passage
    AW.prism([L(p["c"][0] + x * p["d"] / 2, p["c"][1] + y * p["w"] / 2) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], Z(ZB), Z(p["z0"]), C["stone"], top=False)


def b_hall(p):
    """Hypostyle hall (Karnak): walls, a forest of columns — the middle nave is taller, roofs at two levels,
    lattice windows between them."""
    c = np.asarray(p["c"], float); Ls, Ws, H, Hs, nx, ny = p["L"], p["W"], p["H"], p["Hs"], int(p["nx"]), int(p["ny"]); col = ED.get("column", "papyrus")
    P = [(c[0] - Ls / 2, c[1] - Ws / 2), (c[0] + Ls / 2, c[1] - Ws / 2), (c[0] + Ls / 2, c[1] + Ws / 2), (c[0] - Ls / 2, c[1] + Ws / 2)]
    for i in range(4):
        a_, b_ = P[i], P[(i + 1) % 4]; Lq = math.dist(a_, b_); door = i == 3
        WP(AW, L(*a_), L(*b_), Z(ZB), Z(Hs + 0.8 * kq), 1.6 * kq, [(Lq / 2, Z(0.0), 0.16 * Ws, 0.5 * Hs, "rect")] if (door or i == 1) else [], C["stone"])
    AF.face([L3(*q, 0.02) for q in P], C["floor"]); wn = Ws * 0.24
    for i in range(nx):
        for j in range(ny):
            u_ = c[0] - Ls / 2 + Ls * (i + 0.5) / nx; v_ = c[1] - Ws / 2 + Ws * (j + 0.5) / ny; nave = abs(v_ - c[1]) < wn / 2 + 0.6 * Ws / ny
            BA.column(AD, L(u_, v_), (0.9 if nave else 0.62) * Ws / ny * 0.32, Z(0.0), Z(H if nave else Hs), C["stone"], col)
    AR.face([L3(P[0][0], P[0][1], Hs + 0.8 * kq), L3(P[1][0], P[1][1], Hs + 0.8 * kq), L3(P[1][0], c[1] - wn / 2 - 0.6 * kq, Hs + 0.8 * kq), L3(P[0][0], c[1] - wn / 2 - 0.6 * kq, Hs + 0.8 * kq)], C["roof"])
    AR.face([L3(P[3][0], c[1] + wn / 2 + 0.6 * kq, Hs + 0.8 * kq), L3(P[2][0], c[1] + wn / 2 + 0.6 * kq, Hs + 0.8 * kq), L3(P[2][0], P[2][1], Hs + 0.8 * kq), L3(P[3][0], P[3][1], Hs + 0.8 * kq)], C["roof"])
    for s_ in (-1, 1):                                                           # lattice windows between the roof levels
        n_ = max(4, int(Ls / (2.8 * kq)))
        WP(AW, L(P[0][0], c[1] + s_ * (wn / 2 + 0.6 * kq)), L(P[1][0], c[1] + s_ * (wn / 2 + 0.6 * kq)), Z(Hs), Z(H + 0.8 * kq), 0.9 * kq,
                      [(Ls * (k_ + 0.5) / n_, Z(Hs + 0.25 * (H - Hs)), 0.5 * Ls / n_, 0.55 * (H - Hs), "rect") for k_ in range(n_)], C["stone"])
    AR.face([L3(P[0][0], c[1] - wn / 2 - 0.6 * kq, H + 0.8 * kq), L3(P[1][0], c[1] - wn / 2 - 0.6 * kq, H + 0.8 * kq), L3(P[1][0], c[1] + wn / 2 + 0.6 * kq, H + 0.8 * kq),
             L3(P[0][0], c[1] + wn / 2 + 0.6 * kq, H + 0.8 * kq)], C["roof"])


def b_domed(p):
    """Centric temple: cross-in-square (five domes), Greek cross, a forge rotunda — a dome on pendentives and a
    drum."""
    kind = p["kind"]; W, arm, H, t, drum, dh = p["W"], p["arm"], p["H"], p["t"], p["drum"], p["dh"]; c = np.asarray(p["c"], float); cw = L(*c)
    He = H + 0.06 * W + 0.8 * kq; spring = H - W / 2; dk = p["dome"]; win_k = ARCH if ARCH not in ("flat", "stepped") else "round"
    if kind == "rotunda":                                                        # forge: a round wall, dome ribs, fin teeth, chimneys
        R = W / 2; ph = AX_ANG + math.pi - math.pi / 32; P = BA.octagon(cw, R + t / 2, phase=ph, n=32)   # a face exactly to the west: the door in one piece
        for i in range(32):
            a_, b_ = P[i], P[(i + 1) % 32]; Lq = float(np.linalg.norm(b_ - a_)); mid = (a_ + b_) / 2 - cw
            door = (mid @ (-AXV)) > 0.995 * np.linalg.norm(mid)
            holes = [(Lq / 2, Z(0.0), 0.8 * Lq, 0.4 * H, "rect")] if door else ([(Lq / 2, Z(0.45 * H), 0.4 * Lq, 0.3 * H, win_k)] if i % 2 else [])
            WP(AW, a_, b_, Z(ZB), Z(He), t, holes, C["stone"])
        AF.face([(*q, Z(0.02)) for q in BA.octagon(cw, R, phase=ph, n=32)], C["floor"])
        BA.drum(AW, cw, R * 0.9, Z(He), drum, p["windows"], C, t=0.8 * kq, n=24)
        BA.dome(AW, AR, cw, R * 0.92, Z(He + drum), dh, dk if dk != "ribbed" else "ribbed", C, ribs=20 if dk == "ribbed" else 0)
        AW.face([(*q, Z(He)) for q in BA.octagon(cw, R + t / 2 + 0.2, phase=ph, n=32)], C["roof"])
        for i in range(p.get("fins", 0)):                                        # gear fin teeth
            a_ = AX_ANG + math.pi + 2 * math.pi * (i + 0.5) / max(p["fins"], 1); u = np.array([math.cos(a_), math.sin(a_)])   # the door axis is between the teeth
            BA.fin(AD, cw + u * (R + t / 2), u, Z(ZB), Z(0.85 * He), 0.16 * W, 0.03 * W, 1.0 * kq, C["dark"])
        for i in range(p.get("stacks", 0)):                                      # forge chimneys around
            a_ = AX_ANG + math.pi + 2 * math.pi * (i + 0.5) / max(p["stacks"], 1); u = np.array([math.cos(a_), math.sin(a_)]); hs = (He + drum + dh) * (1.2 + 0.5 * ((i * 7) % 3) / 2)
            BA.stack(AD, cw + u * (R + 0.3 * W), 0.05 * W + 0.5 * kq, Z(ZB), Z(hs), C["dark"], C["accent"])
        ring = [(*(cw + (R + t + 1.0 * kq) * np.array([math.cos(2 * math.pi * i / 32), math.sin(2 * math.pi * i / 32)])), Z(0.7 * He)) for i in range(33)]
        BA.tube(AD, ring, 0.5 * kq, C["accent"], n=8, cap=False)
        nd = p["narthex"]; u0 = c[0] - R - t / 2 - nd
        for s_ in (-1, 1): WP(AW, L(u0, c[1] + s_ * 0.28 * W), L(c[0] - R * 0.8, c[1] + s_ * 0.28 * W), Z(ZB), Z(0.55 * H), t, [], C["stone"])
        WP(AW, L(u0, c[1] - 0.28 * W), L(u0, c[1] + 0.28 * W), Z(ZB), Z(0.55 * H), t, [(0.28 * W, Z(0.0), 0.3 * W, 0.4 * H, "rect")], C["stone"])
        AR.face([L3(u0 - 0.5, c[1] - 0.28 * W - 0.5, 0.55 * H), L3(c[0] - R * 0.8, c[1] - 0.28 * W - 0.5, 0.55 * H), L3(c[0] - R * 0.8, c[1] + 0.28 * W + 0.5, 0.55 * H),
                 L3(u0 - 0.5, c[1] + 0.28 * W + 0.5, 0.55 * H)], C["roof"])
        return
    S = W + 2 * arm; h_ = S / 2
    if kind == "cross_in_square":                                                # square body
        outline = [(c[0] - h_, c[1] - h_), (c[0] + h_, c[1] - h_), (c[0] + h_, c[1] + h_), (c[0] - h_, c[1] + h_)]
    else:                                                                        # Greek cross: arms outward
        a, w2 = h_, W / 2
        outline = [(c[0] - a, c[1] - w2), (c[0] - w2, c[1] - w2), (c[0] - w2, c[1] - a), (c[0] + w2, c[1] - a), (c[0] + w2, c[1] - w2), (c[0] + a, c[1] - w2),
                   (c[0] + a, c[1] + w2), (c[0] + w2, c[1] + w2), (c[0] + w2, c[1] + a), (c[0] - w2, c[1] + a), (c[0] - w2, c[1] + w2), (c[0] - a, c[1] + w2)]
    n = len(outline)
    for i in range(n):
        a_, b_ = outline[i], outline[(i + 1) % n]; Lq = math.dist(a_, b_); mid = ((a_[0] + b_[0]) / 2 - c[0], (a_[1] + b_[1]) / 2 - c[1])
        west = mid[0] < 0 and abs(mid[1]) < 0.1 * W; east = mid[0] > 0 and abs(mid[1]) < 0.1 * W   # the ends on the axis, not the sides of the arms
        holes = []
        if west: holes.append((Lq / 2, Z(0.0), min(0.4 * W, 4.0 * kq), min(0.45 * H, 7.0 * kq), win_k))
        elif east and p.get("apses"): holes.append((Lq / 2, Z(0.0), 2 * 0.42 * W * 0.9, 0.62 * H * 0.85, win_k))   # an arch into the apse (scheme)
        elif not east and Lq > 3 * kq: holes += [(Lq * f, Z(0.4 * H), min(0.14 * Lq, 1.6 * kq), 0.3 * H, win_k) for f in ((0.5,) if Lq < W * 0.9 else (0.3, 0.7))]
        WP(AW, L(*a_), L(*b_), Z(ZB), Z(He), t, holes, C["stone"])
        if east and p.get("apses"):                                              # apse with a half-dome
            ra = 0.42 * W if abs(mid[1]) < 1.0 else 0.32 * arm; ca = (c[0] + (h_ if kind == "cross_in_square" else W / 2 + arm), c[1] + mid[1] * 0)
            P = [(ca[0] + ra * math.cos(-math.pi / 2 + math.pi * j / 10), ca[1] + ra * math.sin(-math.pi / 2 + math.pi * j / 10)) for j in range(11)]
            for j in range(10): WP(AW, L(*P[j]), L(*P[j + 1]), Z(ZB), Z(0.62 * H), t * 0.8, [], C["stone"])
            BA.radial_vault(AW, L(*ca), [L(*q) for q in P], Z(0.62 * H - ra * 0.9), ra * 0.9, C["vault"], ribs=False)
            BA.roof_fan(AR, (*L(*ca), Z(0.62 * H + ra * 0.6)), [L(ca[0] + (ra + t) * math.cos(-math.pi / 2 + math.pi * j / 10), ca[1] + (ra + t) * math.sin(-math.pi / 2 + math.pi * j / 10)) for j in range(11)],
                        Z(0.62 * H), C["roof"])
    AF.face([L3(*q, 0.02) for q in outline], C["floor"])
    for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):                            # four piers under the dome
        AW.box(L(c[0] + x * W / 2, c[1] + y * W / 2), AXV, 0.14 * W, 0.14 * W, Z(ZB), Z(spring), C["stone"])
    for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1)):                            # arms under barrel vaults
        if du: Q = [L(c[0] + du * W / 2, c[1] - W / 2), L(c[0] + du * W / 2, c[1] + W / 2), L(c[0] + du * h_, c[1] + W / 2), L(c[0] + du * h_, c[1] - W / 2)]
        else: Q = [L(c[0] - W / 2, c[1] + dv * W / 2), L(c[0] + W / 2, c[1] + dv * W / 2), L(c[0] + W / 2, c[1] + dv * h_), L(c[0] - W / 2, c[1] + dv * h_)]
        BA.rib_vault(AW, Q, Z(spring), W / 2, C["vault"], C["rib"], kind="barrel", rib_w=0.3 * kq, band_w=0.5 * kq)
        a_ = L(c[0] + du * W / 2, c[1] + dv * W / 2); b_ = L(c[0] + du * (h_ + 0.5 * kq), c[1] + dv * (h_ + 0.5 * kq))
        BA.roof_gable(AR, a_, b_, W / 2 + t / 2, Z(He), p["pitch"], C["roof"], ov=0.4 * kq)
    if kind == "cross_in_square":
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):                        # corner bays: a groin vault, a flat roof, domes
            cc = (c[0] + x * (W / 2 + arm / 2), c[1] + y * (W / 2 + arm / 2))
            Q = [L(cc[0] - arm / 2, cc[1] - arm / 2), L(cc[0] + arm / 2, cc[1] - arm / 2), L(cc[0] + arm / 2, cc[1] + arm / 2), L(cc[0] - arm / 2, cc[1] + arm / 2)]
            BA.rib_vault(AW, Q, Z(0.75 * H - arm / 2), arm / 2, C["vault"], C["rib"], kind="groin")
            AR.face([(*q, Z(He)) for q in Q], C["roof"])
            if p.get("corner"):
                BA.drum(AW, L(*cc), arm * 0.3, Z(He), 0.55 * arm, 8, C, t=0.5 * kq, n=12)
                BA.dome(AW, AR, L(*cc), arm * 0.31, Z(He + 0.55 * arm), arm * 0.31 * (1.4 if dk == "onion" else 1.0), dk if dk in ("onion", "hemi", "pointed") else "hemi", C)
    BA.pendentives(AW, cw, W / 2, Z(spring), C["vault"])
    BA.drum(AW, cw, W / 2, Z(He - 1.0 * kq), drum + 1.0 * kq, p["windows"], C, t=t * 0.6, n=16, win_kind=win_k)
    BA.dome(AW, AR, cw, W / 2 * 1.02, Z(He + drum), dh, dk if dk in ("onion", "hemi", "pointed") else "hemi", C)
    if p.get("narthex"):
        nd = p["narthex"]; u0 = c[0] - h_ - nd; w2 = h_ if kind == "cross_in_square" else W / 2 + 0.2 * arm; Hn_ = 0.62 * H
        for s_ in (-1, 1): WP(AW, L(u0, c[1] + s_ * w2), L(c[0] - h_, c[1] + s_ * w2), Z(ZB), Z(Hn_), t, [], C["stone"])
        WP(AW, L(u0, c[1] - w2), L(u0, c[1] + w2), Z(ZB), Z(Hn_), t, [(w2, Z(0.0), min(0.3 * W, 3.5 * kq), min(0.5 * Hn_, 6.0 * kq), win_k)], C["stone"])
        BA.roof_lean(AR, L(c[0] - h_, c[1] - w2), L(c[0] - h_, c[1] + w2), D(-1, 0), nd, Z(Hn_), Z(Hn_ + 0.3 * nd), C["roof"], ov=0.4 * kq)
        AF.face([L3(u0, c[1] - w2, 0.02), L3(c[0] - h_, c[1] - w2, 0.02), L3(c[0] - h_, c[1] + w2, 0.02), L3(u0, c[1] + w2, 0.02)], C["floor"])


B = {"tier": b_tier, "flight": b_flight, "gate": b_gate, "shrine": b_shrine, "tower": b_tower, "ring": b_ring, "court": b_court, "wall": b_wall,
     "water": b_water, "bridge": b_bridge, "cone": b_cone, "ramp": b_ramp, "drum": b_drum, "fin": b_fin, "pylon": b_pylon, "hall": b_hall, "domed": b_domed,
     "prang": lambda p: BA.prang(AT, AR, L(*p["c"]), p["half"], Z(p["z0"]), p["h"], C, ax=AXV),
     "minaret": lambda p: BA.minaret(AT, AR, L(*p["c"]), p["r"], Z(p["z0"]), p["h"], C),
     "obelisk": lambda p: BA.obelisk(AT, AR, L(*p["c"]), p["base"], Z(p["z0"]), p["h"], C["dark"] if SPK > 0.3 else C["stone"], C["accent"], ax=AXV),
     "gallery": lambda p: BA.gallery(AW, AR, [L(*q) for q in p["poly"]], Z(p["z0"]), p["w"], p["h"], C, post=p["post"], kind=p.get("column", "square"),
                                     arch=ARCH if ARCH != "flat" else "rect", doors=gaps_at(p["z0"], 0.6)),   # passages under the gopuras — where the flights come out
     "flight_core": lambda p: None, "stair_in": lambda p: b_stair_in(p)}


def b_stair_in(p):
    """Entrance to a stair inside a tier: a portal in the face at the terrace (a dark opening with a frame)."""
    d = D(*p["dir"]); tg = np.array([-d[1], d[0]]); q = L(*p["c"]) + d * 0.05; w, h, z0 = p["w"], p["h"], p["z0"]
    AW.face([(*(q + tg * x), z) for x, z in BA.hole_outline(0.0, Z(z0), w, h, ARCH if ARCH not in ("flat",) else "round", 6)], C["hole"])
    for s_ in (-1, 1): AW.box(q + tg * s_ * (w / 2 + 0.3 * kq) + d * 0.2 * kq, d, 0.6 * kq, 0.5 * kq, Z(z0), Z(z0 + h + 0.4 * kq), C["dark"])
    AW.box(q + d * 0.2 * kq, d, w + 1.2 * kq, 0.5 * kq, Z(z0 + h), Z(z0 + h + 0.7 * kq), C["dark"])


def build(sc=None, with_ground=True, fresh=True, marker=True):
    """Build the edifice from the loaded plan (load). The command path starts from a clean scene, with ground and a
    human-scale figure; the step path builds into the open scene sc without them (the ground belongs to the
    place step, on the edited relief). → (part count, objects)."""
    global ZB
    if fresh: bpy.ops.wm.read_factory_settings(use_empty=True)
    BA.SCENE[0] = sc
    R = BA.Relief(os.path.join(OUT, "relief.npz"))
    ul, uh, vl, vh = FR["u_lo"], FR["u_hi"], FR["v_lo"], FR["v_hi"]
    UU, VV = np.meshgrid(np.arange(ul - 2, uh + 2, 3.0), np.arange(vl - 2, vh + 2, 3.0))
    zg = R.height(O[0] + AXV[0] * UU + PVV[0] * VV, O[1] + AXV[1] * UU + PVV[1] * VV); ZB = min(-3.0, float(zg.min()) - FL - 1.0)
    c = L((ul + uh) / 2, (vl + vh) / 2); half = max(uh - ul, vh - vl) / 2 + 60 * kq; step = 1.0 if half < 170 else (2.0 if half < 340 else 4.0)
    if with_ground: BA.ground(R, c, half, step); BA.ground(R, c, half * 3.2, 8.0 * max(1.0, step / 2), hole=half - 4, name="ground_far")
    counts = {}
    for p in T["parts"]:
        B[p["type"]](p); counts[p["type"]] = counts.get(p["type"], 0) + 1
    fl = [p for p in T["parts"] if p["type"] == "flight"]
    if fl and marker:                                                            # a 1.8 m person at the start of the main flight
        p0 = np.asarray(fl[0]["p0"], float); BA.lathe(AM, L(*(p0 + np.array([-2.5 * kq, 0.0]))), [(0.24, 0.0), (0.24, 1.5), (0.13, 1.5), (0.13, 1.8)], Z(0.0), MAN, n=8)
    obs = [a.obj(name, BA.coll("edifice")) for a, name in ((AW, "walls"), (AR, "roofs"), (AT, "towers"), (AD, "details"), (AF, "floor"), (AWT, "water")) if a.f]
    if AM.f and marker: AM.obj("man", BA.coll("marker"))                         # the human-scale figure only in the command path (views, sheets)
    BA.SCENE[0] = None
    return counts, obs


def build_edifice(folder, sc=None, tag=True, kw=None):
    """The edifice as a function over an open scene (a step-by-step build): load(folder) + build(sc) without ground
    and the human-scale figure + face markup by the space scheme (bl_space.tag) and a ray check of the model
    (bl_space.verify).
    → (objects, plan); the result is SP_LAST."""
    global SP_LAST
    load(folder, kw)
    counts, obs = build(sc, with_ground=False, fresh=False, marker=False); SP_LAST = None
    if tag and T.get("spaces"):
        S = Space.from_dict(T["spaces"]); SP_LAST = {"faces": BS.tag(S, obs), "check": BS.verify(S)}
    return obs, T


def cameras():
    sc = bpy.context.scene; cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)
    con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    R = BA.Relief(os.path.join(OUT, "relief.npz"))
    def gz(p): return float(R.height(p[0], p[1]))
    ul, uh, vl, vh = FR["u_lo"], FR["u_hi"], FR["v_lo"], FR["v_hi"]; um, vm = (ul + uh) / 2, (vl + vh) / 2; Lt = max(uh - ul, vh - vl)
    hmax = max([p.get("h", 0.0) + (p.get("z0", 0.0) if p["type"] in ("prang", "minaret", "obelisk") else 0.0) for p in T["parts"]
                if p["type"] in ("tower", "prang", "minaret", "obelisk")] + [p.get("z1", 0.0) for p in T["parts"] if p["type"] in ("tier", "cone", "ramp")]
               + [p.get("z0", 0.0) + p.get("H", 0.0) for p in T["parts"] if p["type"] in ("shrine", "hall")] + [p.get("h", 0.0) for p in T["parts"] if p["type"] in ("pylon", "wall")]
               + [p.get("H", 0.0) + p.get("drum", 0.0) + p.get("dh", 0.0) for p in T["parts"] if p["type"] == "domed"] + [10.0])
    Dd = 1.45 * max(Lt, 1.3 * hmax); views = {}
    p = L(um - 0.62 * Dd, vm - 0.8 * Dd); views["aerial"] = dict(loc=(*p, Z0 + 0.48 * Dd), tgt=(*L(um, vm), FL + 0.25 * hmax), lens=35)
    dist = max(0.95 * hmax, 0.75 * (vh - vl)) + 0.2 * (uh - ul); p = L(ul - dist, vm - 0.12 * dist); zc = gz(p) + 1.7
    q = L(ul, vm); zs = max(gz(p + (q - p) * f) for f in np.linspace(0.05, 0.95, 40))   # a slope between the camera and the edifice — rise above it
    if zs + 2.0 > zc: zc = zs + 2.0
    views["front"] = dict(loc=(*p, zc), tgt=(*q, zc), lens=28, shift=0.28)
    ds = 1.15 * max(uh - ul, 1.5 * hmax); p = L(um, vl - ds); views["side"] = dict(loc=(*p, gz(p) + 0.12 * ds), tgt=(*L(um, vm), FL + 0.33 * hmax), lens=35)
    tops = [p for p in T["parts"] if p["type"] == "shrine"] or [p for p in T["parts"] if p["type"] in ("tier",)][-1:]
    if tops:
        tp = tops[-1]; zt = tp.get("z0", tp.get("z1", 0.0)); cu = (tp.get("c") or [0.0, 0.0])
        views["top"] = dict(loc=L3(cu[0] - 0.9 * max(tp.get("L", 20.0), 20.0) - 6 * kq, cu[1] - 0.35 * max(tp.get("W", 20.0), 12.0), zt + 1.7 + 6 * kq),
                            tgt=L3(cu[0], cu[1], zt + 0.35 * max(tp.get("H", 8.0), 6.0)), lens=22)
    ins = [p for p in T["parts"] if p["type"] in ("hall", "court", "domed")]
    if ins:
        q = ins[0]; cu = q["c"]
        if q["type"] == "hall": views["inside"] = dict(loc=L3(cu[0] - q["L"] / 2 + 3.0 * kq, cu[1] + 0.05 * q["W"], 1.7), tgt=L3(cu[0] + q["L"] / 2, cu[1], 0.45 * q["H"]), lens=16, inside=True)
        elif q["type"] == "domed": views["inside"] = dict(loc=L3(cu[0] - q["W"] / 2 - q.get("arm", 0) * 0.8, cu[1], 1.7), tgt=L3(cu[0] + q["W"] * 0.2, cu[1], q["H"] * 0.9), lens=14, inside=True)
        elif q.get("shape") != "rect": views["inside"] = dict(loc=L3(cu[0] - q["r"] * 1.05, cu[1] - q["r"] * 0.35, 2.5 + 0.2 * q["r"]), tgt=L3(cu[0] + 0.2 * q["r"], cu[1], -q["depth"]), lens=20)
        else: views["inside"] = dict(loc=L3(cu[0] - q["L"] / 2 + 2 * kq, cu[1] + 0.25 * q["W"], 1.7), tgt=L3(cu[0] + q["L"] / 2, cu[1], 0.3 * q.get("h", 8.0)), lens=18)
    views["plan"] = dict(loc=(*L(um, vm), FL + hmax + 300), tgt=(*L(um, vm), FL), ortho=Lt * 1.1)
    zc_ = sorted({c["z0"] for c in (T.get("spaces") or {}).get("cells", []) if c["walk"] and c["cls"] in ("interior", "covered") and c["z0"] > -1e3})
    Dk = 1.0 * Lt; p = L(um - 0.3 * Dk, vm - 0.62 * Dk)                              # section: inside — warm, outside — blue
    if KW.get("cam"):                                                            # custom view: cam=x,y,z;x,y,z (world) [lens=]
        a_, b_ = KW["cam"].split(";"); views["cam"] = dict(loc=tuple(map(float, a_.split(","))), tgt=tuple(map(float, b_.split(","))), lens=float(KW.get("lens", 24)))
    views["cutaway"] = dict(loc=(*p, FL + 0.8 * Dk + hmax * 0.3), tgt=(*L(um, vm), FL), lens=30, side=True, cut=(zc_[0] if zc_ else FL) + 2.2 * kq)
    return cam, tgt, views


def render(cam, tgt, views, which, res, obs=()):
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"; shd = sc.display.shading
    shd.light = "STUDIO"; shd.color_type = "VERTEX"; shd.show_cavity = True; shd.cavity_type = "WORLD"
    shd.shadow_intensity = 0.55; sc.display.light_direction = (0.45, -0.35, 0.82)
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    out = {}
    for k in which:
        if k not in views: continue
        v = views[k]; cam.location = v["loc"]; tgt.location = v["tgt"]
        cam.data.type = "ORTHO" if "ortho" in v else "PERSP"
        if "ortho" in v: cam.data.ortho_scale = v["ortho"]
        else: cam.data.lens = v["lens"]
        cam.data.shift_y = v.get("shift", 0.0); cam.data.clip_start = 0.3; cam.data.clip_end = 9000; shd.show_shadows = not (v.get("inside") or v.get("side"))
        BS.show("Side" if v.get("side") else "Col"); back = None
        if v.get("cut") and obs: _, back = BS.cutaway(obs, v["cut"]); BS.show("Side")
        sc.render.filepath = os.path.join(OUT, f"view_{k}.png"); bpy.ops.render.render(write_still=True); out[k] = sc.render.filepath
        if back: back()
    BS.show("Col")
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    load(argv[0], dict(a.split("=", 1) for a in argv[1:] if "=" in a))
    counts, obs = build(); t_build = time.time() - T0; SP = None
    if T.get("spaces"):                                                          # faces — cell and side by the scheme; model against the scheme
        S = Space.from_dict(T["spaces"]); SP = {"faces": BS.tag(S, obs)}; SP["check"] = BS.verify(S)
    cam, tgt, views = cameras(); res = tuple(int(v) for v in KW.get("res", "1280x800").split("x"))
    out = render(cam, tgt, views, KW.get("views", "aerial,front,side,top,inside,plan,cutaway").split(","), res, obs if SP else ())
    dg = bpy.context.evaluated_depsgraph_get()
    tris = sum(len(o.evaluated_get(dg).to_mesh().loop_triangles) for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground"))
    if KW.get("blend", "0") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "edifice.blend"))
    print("__RESULT__", json.dumps({"build_s": round(t_build, 2), "total_s": round(time.time() - T0, 1), "n_triangles": tris, "n_parts": counts, "kinds": list(out),
                                    "n_openings_skipped": SKIP[0], "space": SP}, ensure_ascii=False))


if __name__ == "__main__":
    main()
