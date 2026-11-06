"""A cathedral in Blender from the temple.json v2 plan (temples/temple_plan.py): a module and a palette for each part
(the style of its campaign) — walls with real openings (arcade, triforium, clerestory, windows, portals, a rose; arches of
all kinds, tracery by style: bar, flamboyant, net, plate, "elven"), piers, vaults (groin, ribbed, star, barrel, flat
with beams), the crossing (a flèche, a tower, an octagonal lantern, a dome on a drum with pendentives), the transept, the
east end (an apse with a gable, an ambulatory with chapels, a polygon, a flat one with a Lady chapel), the west end (two
towers, one, a screen), tiered towers with tops (bl_arch.crown), supports (pilaster strips, buttresses, flying buttresses,
claws, branches, hive tower piers, the round towers of a fortress), roofs (gabled, flat with a parapet, spikes along the
ridge), special elements and additions (chapels, a chapter house, a cloister, a porch, a sacristy, turrets, a campanile, a
baptistery, a westwork, transept towers), an unfinished building with a crane, a ruin, a turn of the choir axis. Colors
are a vertex attribute. The shared geometry is base/bl_arch.py.

blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/buildings/temples/bl_temple.py -- <plan folder> [views=aerial,facade,side,east,interior,cross,detail,plan] [res=1280x800] [blend=1]
→ <folder>/view_<view>.png, temple.blend (if blend=1), the __RESULT__ line

As a function over an open scene (a step-by-step build; importing reads and builds nothing):
    from components.buildings.temples import bl_temple as BT
    objs, T = BT.build_temple(folder, sc=None, tag=True)        # load + build (without the ground and the human scale figure) + face labeling
or in parts: BT.load(folder, kw=None) → BT.build(sc=None, with_ground=True, fresh=True, marker=True) → (uw, ue, objects)."""
import json, math, os, sys, time

import bpy
import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.base import bl_arch as BA                                                             # noqa: E402
from components.buildings.space import bl_space as BS                                                            # noqa: E402
from components.buildings.space.space import Space                                                          # noqa: E402

# The plan state — filled in by load(folder); importing the module reads and builds nothing
# (as a function over an open scene — build_temple)
OUT = None; KW = {}; T = None; T0 = time.time(); ZB = -3.0; SKIP = [0]; SP_LAST = None
MAN = (0.55, 0.12, 0.10)


def cols_of(c):
    """A part's palette with face roles (sk_role): stone — wall, belts and ribs — finish, floor, roof, metal, opening."""
    s = BA.rc(c["stone"], "wall")
    return {"stone": s, "dark": BA.rc(BA.shade(s, 0.84), "finish"), "vault": BA.shade(s, 1.07), "rib": BA.rc(BA.shade(s, 0.74), "finish"),
            "trac": BA.rc(BA.shade(s, 0.9), "finish"), "floor": BA.rc(BA.shade(s, 0.66), "floor"), "roof": BA.rc(c["roof"], "roof"),
            "accent": BA.rc(c.get("accent", c["roof"]), "metal"), "hole": BA.rc(c.get("hole", (0.05, 0.05, 0.06)), "opening")}


class Mo:
    """A part's module (temple.json → modules): numbers and palette."""

    def __init__(self, d): self.__dict__.update(d); self.C = cols_of(d["colors"]); self.KIND = d["arch"]


def mod(part): return MO[((PT.get(part) or {}).get("mod")) or PT["nave"]["mod"]]


def load(folder, kw=None):
    """Read a temple plan (temple.json, relief.npz — in the plan folder) and fill in the module state: the frame, the parts, the
    modules with palettes, fresh mesh accumulators. kw — the command-line keys (views, res, cam, hide, blend …), as after
    "--". It does not touch the scene."""
    global OUT, KW, T, T0, FR, PT, STY, STATE, SIG, EX, FORT, O, AXV, PVV, Z0, FL, ZB, SKIP, MO, Mn, Mt, Me, Mw, W, A, kq
    global u_n1, X_, u_x1, u_c1, CH_AIS, TR, v1, E, BEND, u_mid, AW, AP, AV, AR, AT, AB, AXT, AF, AM
    OUT = os.path.abspath(folder); KW = dict(kw or {})
    T = json.load(open(os.path.join(OUT, "temple.json"))); T0 = time.time()
    FR = T["frame"]; PT = T["parts"]; STY = T.get("story", {}); STATE = STY.get("state", {}); SIG = (STY.get("signature") or {}).get("kind")
    EX = T.get("extras", {}); FORT = bool(T.get("fort"))
    O = np.asarray(FR["origin"], float); AXV = np.asarray(FR["ax"], float); PVV = np.asarray(FR["pv"], float)
    Z0 = float(FR["z0"]); FL = Z0 + float(FR["floor"]); ZB = -3.0; SKIP = [0]
    MO = {k: Mo(d) for k, d in T["modules"].items()}
    Mn = mod("nave"); Mt = mod("transept") if PT.get("transept") else (mod("crossing") if PT.get("crossing") else Mn); Me = mod("choir"); Mw = mod("west")
    W, A, kq = Mn.W, Mn.A, Mn.kq
    u_n1 = PT["nave"]["u1"]; X_ = PT.get("crossing"); u_x1 = X_["u1"] if X_ else u_n1; u_c1 = PT["choir"]["u1"]; CH_AIS = PT["choir"]["aisles"]
    TR = PT.get("transept"); v1 = TR["v1"] if TR else 0.0; E = PT["east"]; BEND = float(PT["choir"].get("bend", 0.0))
    u_mid = (u_n1 + u_x1) / 2
    AW, AP, AV, AR, AT, AB, AXT, AF = (BA.Acc(r) for r in ("wall", "wall", "wall", "roof", "wall", "wall", "finish", "floor"))   # walls, piers and ribs, vaults, roofs, towers, supports, tracery, floor
    AM = BA.Acc()                                                                    # human-scale figure (not a building)


def L(u, v): return O + AXV * u + PVV * v


def Lc(u, v):
    """Choir: the axis turns by BEND around (u_x1, 0), growing over the first bay of the choir (Quimper)."""
    if not BEND or u <= u_x1 + 1e-9: return L(u, v)
    f = min(1.0, (u - u_x1) / max(Me.B, 1.0)); a_ = math.radians(BEND) * f; du = u - u_x1
    return L(u_x1 + du * math.cos(a_) - v * math.sin(a_), du * math.sin(a_) + v * math.cos(a_))


def D(du, dv): d = AXV * du + PVV * dv; return d / max(np.linalg.norm(d), 1e-9)


def Dc(du, dv):
    a_ = math.radians(BEND); return D(du * math.cos(a_) - dv * math.sin(a_), du * math.sin(a_) + dv * math.cos(a_))


def Z(z): return FL + z
def L3(u, v, z, T_=L): q = T_(u, v); return (q[0], q[1], FL + z)


def panel(a, p0, p1, z0, z1, th, holes, col, M, top=None, z1_end=None, trac=None, T_=L):
    """A wall in local (u, v) via the frame T_; heights above the floor; openings (s, z0, w, h, kind) or {"pts"}; trac — tracery by the module's style."""
    k_ = float(np.linalg.norm(T_(*p1) - T_(*p0))) / max(math.dist(p0, p1), 1e-9)   # a wall along a chord (a rotated choir): the openings are given as fractions of its length
    hs = [{"pts": [(s_ * k_, Z(z_)) for s_, z_ in h_["pts"]]} if isinstance(h_, dict) else (h_[0] * k_, Z(h_[1]), h_[2], h_[3], h_[4]) for h_ in holes]
    SKIP[0] += BA.wall_panel(a, T_(*p0), T_(*p1), Z(z0), Z(z1), th, hs, col, top=[(s_, Z(z_)) for s_, z_ in top] if top else None,
                             z1_end=Z(z1_end) if z1_end is not None else None)
    if trac and M.win["tracery"] not in ("none", "plate"):
        p0w, p1w = T_(*p0), T_(*p1); d = (p1w - p0w) / max(np.linalg.norm(p1w - p0w), 1e-9); n = np.array([d[1], -d[0]])
        for s_, z0_, w_, h_ in trac:
            if w_ >= 1.1 * kq: BA.tracery(AXT, p0w + d * s_, n, Z(z0_), w_, h_, M.C["trac"], th=0.05 * w_ + 0.06 * kq, kind=M.KIND, style=M.win["tracery"])


def win(M, sc, z0, w, h, holes, trac):
    """A window by the module's style: plate tracery — two lancets and a circle in a solid wall; otherwise a single opening (+ bar tracery)."""
    if M.win["tracery"] == "plate" and w >= 1.2 * kq:
        lw, lh = 0.36 * w, 0.7 * h; holes += [(sc - 0.24 * w, z0, lw, lh, M.KIND), (sc + 0.24 * w, z0, lw, lh, M.KIND)]
        d = min(0.4 * w, 0.22 * h); holes.append((sc, z0 + lh + 0.05 * h, d, d, "circle"))
    else: holes.append((sc, z0, w, h, M.KIND)); trac.append((sc, z0, w, h))


def cyl(a, c, r, z0, z1, col, n=12, top=True):
    a.prism([np.asarray(c) + r * np.array([math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)]) for i in range(n)], z0, z1, col, top=top)


def dark_open(a, c, n, z0, w, h, col, kind="pointed", eps=0.05):
    """A dark opening on a face (of a tower): c — a point on the face (world), n — outward."""
    n = np.asarray(n, float); n = n / max(np.linalg.norm(n), 1e-9); tg = np.array([-n[1], n[0]]); c = np.asarray(c, float) + n * eps
    a.face([(*(c + tg * x), z) for x, z in BA.hole_outline(0.0, z0, w, h, kind, 6)], col)


# ---------------------------------------------------------------- piers
def arc_w(M, span):
    """The width of an arcade opening: for a round pier the wall face (tc) hides within its circle."""
    if M.pier_kind == "square": return span - M.pier
    rp = M.pier / 2; return span - 2 * math.sqrt(max(rp * rp - (M.tc / 2) ** 2, 0.04)) * 0.95


def pier_at(M, u, v, z_cap, main, inward, z_shaft=None, big=1.0, T_=L, DT=D):
    """A pier: square (Romanesque, dwarves; on the main ones — a half-column up to the springing of the vault), compound (round
    with shafts, a vaulting shaft along the wall up to the springing of the vault), clustered (hall church). inward — (du, dv)
    toward the space axis."""
    c = T_(u, v); r = M.pier / 2 * big; dn = DT(*inward); C = M.C
    if M.pier_kind == "square":
        AP.box(c, AXV, M.pier * big, M.pier * big, Z(ZB), Z(z_cap), C["stone"]); AP.box(c, AXV, M.pier * big * 1.18, M.pier * big * 1.18, Z(z_cap), Z(z_cap + 0.4 * kq), C["dark"])
        if main: cyl(AP, c + dn * M.pier * big / 2, 0.24 * M.pier, Z(ZB), Z(z_shaft or M.spring), C["stone"])
        return
    n8 = 8 if M.pier_kind == "fasciculate" else 4
    cyl(AP, c, r, Z(ZB), Z(z_cap), C["stone"], n=16)
    for i in range(n8):
        a_ = 2 * math.pi * i / n8; cyl(AP, c + r * np.array([math.cos(a_), math.sin(a_)]), 0.11 * M.pier * big, Z(ZB), Z(z_cap), C["stone"], n=8)
    cyl(AP, c, r + 0.16 * M.pier, Z(z_cap - 0.45 * kq), Z(z_cap), C["dark"], n=16)
    if main and z_shaft: cyl(AP, c + dn * (M.tc / 2 + 0.06 * M.pier), 0.09 * M.pier * big, Z(z_cap), Z(z_shaft), C["stone"], n=8)


# ---------------------------------------------------------------- nave, aisles, choir
def tower_over(u, v):
    return any(abs(u - tw["u"]) < tw["side"] / 2 and abs(v - tw["v"]) < tw["side"] / 2 for tw in T["towers"])


def jagged_top(Lw, z_hi, z_lo, seed):
    """The crumbled top of a ruin wall."""
    pts = [(0.0, z_hi)]
    for i, s_ in enumerate(np.linspace(0, Lw, max(3, int(Lw / 1.8)) + 1)[1:-1]):
        f = (math.sin(seed * 3.1 + i * 12.9898) * 43758.5453) % 1.0; pts.append((float(s_), z_hi - (z_hi - z_lo) * f))
    return pts + [(Lw, z_hi)]


def arcade_wall(M, ua, ub, sgn, T_, ruin=False):
    """A nave wall over a bay: an arcade across the side-aisle spans, a triforium, a clerestory (under a tower — blank above the
    arcade)."""
    holes = []; trac = []; v = sgn * W / 2; blind = tower_over((ua + ub) / 2, sgn * (W / 2 + A / 2)); ab = M.ab; bay_a = M.B / ab
    for m in range(ab):
        sc = (m + 0.5) * bay_a; holes.append((sc, 0.0, arc_w(M, bay_a), M.z_arc, M.KIND))
        if M.tri and not blind:
            for i in range(M.tri["n"]): holes.append((sc + (i - (M.tri["n"] - 1) / 2) * (bay_a - M.pier) / M.tri["n"], M.tri["z0"], M.tri["w"], M.tri["h"], M.KIND))
        if M.win["cl"] and not blind: win(M, sc, M.win["cl"]["z0"], M.win["cl"]["w"], M.win["cl"]["h"], holes, trac)
    top = jagged_top(ub - ua, M.He, M.z_tri + 0.2 * (M.He - M.z_tri), ua * 0.37 + sgn) if ruin else None
    panel(AW, (ua, v), (ub, v), ZB, M.He, M.tc, holes, M.C["stone"], M, top=top, trac=None if ruin else trac, T_=T_)


def hall_arcade(M, ua, ub, sgn, T_):
    """Hall church: between the aisles — only piers and arches along the wall arcs of the vaults (above — the vaults themselves)."""
    ps = M.pier * 0.8; sa, sb = ps / 2, M.B - ps / 2
    def top(s_):
        x = s_ - M.B / 2
        return min(M.spring + BA._arch_z(M.B, M.rise, x), M.spring_a + BA._arch_z(M.B, M.ra, x)) - 0.15 * kq if M.rise > 1e-3 else M.H - 0.2 * kq
    pts = [(sa, 0.0), (sb, 0.0)] + [(s_, top(s_)) for s_ in np.linspace(sb, sa, 17)]
    panel(AW, (ua, sgn * W / 2), (ub, sgn * W / 2), ZB, M.H, 0.7 * M.pier, [{"pts": pts}], M.C["stone"], M, T_=T_)


def sacristy_door(part, ua, ub, sgn):
    """The sacristy door in the choir wall (the scheme: a transition into sacristy) and the wall segment covered by the sacristy
    (no windows are cut there)."""
    sa = EX.get("sacristy")
    if not sa or part != "choir" or sgn != sa["side"]: return None, None
    um = (sa["u0"] + sa["u1"]) / 2; door = (um - ua, 0.0, 1.4 * kq, 2.6 * kq, Me.KIND if Me.KIND != "flat" else "round") if ua <= um < ub else None
    return door, (sa["u0"] - ua, sa["u1"] - ua)


def aisle_wall(M, ua, ub, sgn, T_, chapel=None, part=None):
    holes = []; trac = []; v = sgn * (W / 2 + A); bay_a = M.B / M.ab; door, span = sacristy_door(part, ua, ub, sgn)
    if door: holes.append(door)
    for m in range(M.ab):
        sc = (m + 0.5) * bay_a
        if tower_over(ua + sc, v) or (span and span[0] - 0.5 * kq <= sc <= span[1] + 0.5 * kq): continue
        if chapel: holes.append((sc, 0.0, min(bay_a - M.buttress["w"] - 0.6 * kq, bay_a * 0.8), chapel * 0.82, M.KIND)); continue
        win(M, sc, M.win["ai"]["z0"], M.win["ai"]["w"], M.win["ai"]["h"], holes, trac)
    panel(AW, (ua, v), (ub, v), ZB, M.Ea, M.t, holes, M.C["stone"], M, trac=trac, T_=T_)
    if FORT or M.fortified >= 0.6: BA.merlon_row(AW, [T_(ua, v), T_(ub, v)], Z(M.Ea), 0.9 * kq, M.C["stone"], closed=False)


def choir_wall(M, ua, ub, sgn, T_, part="choir"):
    """A choir without aisles: a tall wall with windows (Romanesque — two tiers, hall church — one tall one); by the sacristy — a
    door, windows above its roof."""
    holes = []; trac = []; door, span = sacristy_door(part, ua, ub, sgn)          # the choir has no aisles per the plan, but the campaign module is basilican: windows of two tiers
    if door: holes.append(door)
    for w_ in (M.win["choir"] or [w__ for w__ in (M.win["ai"], M.win["cl"]) if w__]):
        if span and span[0] <= M.B / 2 <= span[1] and w_["z0"] < EX["sacristy"]["h"] + 1.5 * kq: continue
        win(M, M.B / 2, w_["z0"], w_["w"], w_["h"], holes, trac)
    panel(AW, (ua, sgn * W / 2), (ub, sgn * W / 2), ZB, M.He, M.t, holes, M.C["stone"], M, trac=trac, T_=T_)


def nave_vault(M, ua, ub, T_):
    BA.rib_vault(AV, [T_(ua, W / 2), T_(ua, -W / 2), T_(ub, -W / 2), T_(ub, W / 2)], Z(M.spring), M.rise, M.C["vault"], M.C["rib"], kind=M.vault,
                 rib_w=0.3 * kq, band_w=0.75 * M.pier if M.vault in ("groin", "barrel") else None, a_rib=AP)


def aisle_vault(M, ua, ub, sgn, T_):
    vo, vi = sgn * (W / 2 + A), sgn * W / 2
    BA.rib_vault(AV, [T_(ua, vo), T_(ua, vi), T_(ub, vi), T_(ub, vo)], Z(M.spring_a), M.ra, M.C["vault"], M.C["rib"], kind=M.vault, rib_w=0.24 * kq,
                 band_w=0.5 * M.pier if M.vault in ("groin", "barrel") else None, a_rib=AP)


def body(M, u0, nb, aisles, part):
    """The bays of the nave or the choir from u0: walls, vaults, piers (the choir in its own frame, the ruin of the nave without
    vaults and roof)."""
    T_ = Lc if part == "choir" else L; DT = Dc if part == "choir" else D
    ruin = part == "nave" and STATE.get("nave_roofless"); bay_a = M.B / M.ab
    z_cap = M.z_arc - min(BA.arch_rise(M.KIND, arc_w(M, bay_a)), M.z_arc * 0.95)
    ch = EX.get("chapels") if part == "nave" else None
    for j in range(nb):
        ua, ub = u0 + j * M.B, u0 + (j + 1) * M.B
        for sgn in (1, -1):
            if not aisles: choir_wall(M, ua, ub, sgn, T_, part); continue
            if M.hall: hall_arcade(M, ua, ub, sgn, T_)
            else: arcade_wall(M, ua, ub, sgn, T_, ruin=ruin)
            aisle_wall(M, ua, ub, sgn, T_, chapel=ch["h"] if ch and j in ch["bays"] else None, part=part)
            for m in range(M.ab): aisle_vault(M, ua + m * bay_a, ua + (m + 1) * bay_a, sgn, T_)
        if not ruin: nave_vault(M, ua, ub, T_)
    if not aisles: return
    for j in range(nb * M.ab + 1):
        u = u0 + j * bay_a
        if j == 0 and u0 < 1e-6: continue                                        # at the west wall — no pier
        for sgn in (1, -1):
            if M.hall: pier_at(M, u, sgn * W / 2, min(M.spring, M.spring_a), False, (0, -sgn), T_=T_, DT=DT)
            else: pier_at(M, u, sgn * W / 2, z_cap, j % M.ab == 0, (0, -sgn), z_shaft=M.spring, T_=T_, DT=DT)


# ---------------------------------------------------------------- crossing and transept
def gable(M, Lw, sc, hw, z_lo=None, z_peak=None):
    """The top of a gable wall: from the eave He at the edges to the ridge above the middle sc (the roof half-width hw); with a
    flat roof — a parapet."""
    zl = M.He if z_lo is None else z_lo
    if M.roof != "gable" and z_peak is None: return [(0.0, max(zl, M.He + 1.2 * kq)), (Lw, max(zl, M.He + 1.2 * kq))]
    pitch = M.roof_pitch if M.roof_pitch > 1 else 30.0
    zp = (M.He + (hw + M.ov) * math.tan(math.radians(pitch))) if z_peak is None else z_peak
    pts = [(0.0, zl)]
    for s_ in np.linspace(0, Lw, 41): pts.append((float(s_), float(max(zl, zp - abs(s_ - sc) * math.tan(math.radians(pitch))))))
    pts.append((Lw, zl)); out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6: out.append(p)
    return out


def crossing_and_transept():
    if not X_: return
    M = Mt; cr = T.get("crossing"); ck = cr["kind"] if cr else None
    for uu in (u_n1, u_x1):
        for sgn in (1, -1): pier_at(M, uu, sgn * W / 2, M.spring, False, (0, -sgn), big=1.45)
    if ck == "dome": BA.pendentives(AV, L(u_mid, 0.0), W / 2, Z(M.spring), M.C["vault"])   # pendentives under the dome (Florence)
    elif ck != "lantern":
        BA.rib_vault(AV, [L(u_n1, W / 2), L(u_n1, -W / 2), L(u_x1, -W / 2), L(u_x1, W / 2)], Z(M.spring), M.rise, M.C["vault"], M.C["rib"], kind=M.vault,
                     rib_w=0.3 * kq, band_w=0.75 * M.pier if M.vault in ("groin", "barrel") else None, a_rib=AP)
    if not TR:                                                                   # without arms: the sides of the crossing and the ends of the aisles are walls
        for sgn in (1, -1):
            holes = []; trac = []
            if M.win["cl"]: win(M, (u_x1 - u_n1) / 2, M.win["cl"]["z0"], min(M.win["cl"]["w"], 0.5 * (u_x1 - u_n1)), M.win["cl"]["h"], holes, trac)
            panel(AW, (u_n1, sgn * W / 2), (u_x1, sgn * W / 2), ZB, M.He, M.tc, holes, M.C["stone"], M, trac=trac)
            for uu, ais, Ma in ((u_n1, True, Mn), (u_x1, CH_AIS, Me)):
                if ais and not Ma.hall: panel(AW, (uu, sgn * W / 2), (uu, sgn * (W / 2 + A + Ma.t / 2)), ZB, Ma.z_lean, Ma.t, [], Ma.C["stone"], Ma, z1_end=Ma.Ea)
                elif ais: panel(AW, (uu, sgn * W / 2), (uu, sgn * (W / 2 + A + Ma.t / 2)), ZB, Ma.He, Ma.t, [], Ma.C["stone"], Ma)
        return
    kb = TR["bays"]; cl_w = min(M.win["cl"]["w"], 0.62 * M.B) if M.win["cl"] else 0.0
    for sgn in (1, -1):
        clo = EX.get("cloister") or {}
        for uu, ais in ((u_n1, True), (u_x1, CH_AIS)):                          # the west and east walls of the arm
            holes = []; trac = []; dsc = None
            if uu == u_n1 and clo.get("door_v") is not None and np.sign(clo["door_v"]) == sgn:   # a door into the cloister (the scheme)
                dsc = abs(clo["door_v"]) - W / 2; holes.append((dsc, 0.0, 1.6 * kq, 3.0 * kq, M.KIND if M.KIND != "flat" else "round"))
            for j in range(kb):
                sc = (j + 0.5) * M.B
                if j == 0 and ais and not M.hall: holes.append((A / 2, 0.0, arc_w(M, A), M.z_arc, M.KIND))
                elif dsc is None or abs(sc - dsc) > (M.win["ai"]["w"] + 1.6 * kq) / 2 + 0.4 * kq: win(M, sc, M.win["ai"]["z0"], M.win["ai"]["w"], M.win["ai"]["h"], holes, trac)
                if M.win["cl"]: win(M, sc, M.win["cl"]["z0"], cl_w, M.win["cl"]["h"], holes, trac)
            panel(AW, (uu, sgn * W / 2), (uu, sgn * (v1 + M.t / 2)), ZB, M.He, M.t, holes, M.C["stone"], M, trac=trac)
        for j in range(kb):
            va, vb = sgn * (W / 2 + j * M.B), sgn * (W / 2 + (j + 1) * M.B)
            BA.rib_vault(AV, [L(u_n1, va), L(u_x1, va), L(u_x1, vb), L(u_n1, vb)], Z(M.spring), M.rise, M.C["vault"], M.C["rib"], kind=M.vault, rib_w=0.3 * kq,
                         band_w=0.75 * M.pier if M.vault in ("groin", "barrel") else None, a_rib=AP)
        Lw = u_x1 - u_n1 + M.t; mid = Lw / 2; gl = gable(M, Lw, Lw / 2, W / 2 + M.t / 2)        # the end of the arm: a portal, a rose or windows, a gable
        holes = []; trac = []; pw, ph = T["portal"]["w"] * 0.8, T["portal"]["h"] * 0.85
        holes.append((mid, 0.0, pw, ph, M.KIND)); zr = r_ = None
        if T.get("rose"):
            r_ = T["rose"]["r"] * 0.85; zr = min(T["rose"]["z"], M.He - r_ - 1.0 * kq); holes.append((mid, zr - r_, 2 * r_, 2 * r_, "circle"))
        else:
            for x in (-1, 1): holes.append((mid + x * W / 4, M.win["ai"]["z0"] + ph * 0.6, min(M.win["ai"]["w"], 0.2 * W), M.win["ai"]["h"] * 0.7, M.KIND))
            if M.win["cl"]: win(M, mid, M.win["cl"]["z0"], cl_w, M.win["cl"]["h"], holes, trac)
        p0 = (u_n1 - M.t / 2, sgn * v1); p1 = (u_x1 + M.t / 2, sgn * v1)
        if sgn < 0: p0, p1 = p1, p0
        panel(AW, p0, p1, ZB, M.He, M.t, holes, M.C["stone"], M, top=gl, trac=trac)
        pc = L(u_mid, sgn * (v1 + M.t / 2))
        BA.porch(AXT, pc, D(0, sgn), Z(0.0), pw, ph, M.KIND, 0.25 * pw + 0.3 * kq, M.C["stone"], gable=M.KIND not in ("round", "stepped", "flat"))
        if not (EX.get("chapter_house") and np.sign(EX["chapter_house"]["v"]) == sgn):
            BA.steps(AF, pc + D(0, sgn) * (0.25 * pw + 0.3 * kq), D(0, -sgn), pw * 1.5, Z0, FL, M.C["dark"])
        if r_: BA.rose(AXT, L(u_mid, sgn * v1), D(0, sgn), Z(zr), r_, M.C["trac"], th=0.22 * kq)
    roof_part(M, L(u_mid, -v1), L(u_mid, v1), W / 2 + M.t / 2)


def roof_part(M, p0, p1, half, hip1=False, ruin=False):
    """The roof of a part: gabled (hipped on request) or a flat terrace with a parapet; spikes along the ridge for "spiked" styles."""
    if ruin: return
    if M.roof == "gable":
        zr = BA.roof_gable(AR, p0, p1, half, Z(M.He), M.roof_pitch, M.C["roof"], ov=M.ov, hip1=hip1)
        if M.spikes >= 0.5 or SIG == "pinnacle_forest":                         # spikes or pinnacles along the ridge
            L_ = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))); n = max(2, int(L_ / (6.0 * kq)))
            for i in range(1, n):
                q = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * i / n
                if M.spikes >= 0.5: BA.spike(AR, q, zr - 0.2, 0.5 * kq, (2.5 + 2.0 * M.spikes) * kq, M.C["dark"])
                else: BA.pinnacle(AR, q, zr - 0.3, 0.45 * kq, 1.0 * kq, M.C["stone"], M.C["roof"])
    else:                                                                        # terrace: a slab and a parapet (dwarves)
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9); n_ = np.array([-d[1], d[0]])
        P = [p0 - n_ * half, p1 - n_ * half, p1 + n_ * half, p0 + n_ * half]
        AR.face([(*p, Z(M.He)) for p in P], BA.shade(M.C["roof"], 0.9))
        AW.quad_strip([(*p, Z(M.He)) for p in P + [P[0]]], [(*p, Z(M.He + 1.2 * kq)) for p in P + [P[0]]], M.C["stone"])
        if M.fortified >= 0.5: BA.merlon_row(AW, P, Z(M.He + 1.2 * kq), 0.9 * kq, M.C["stone"])


# ---------------------------------------------------------------- east end
def hemi(r, n, uc=None):
    uc = u_c1 if uc is None else uc
    return [(uc + r * math.cos(-math.pi / 2 + math.pi * i / n), r * math.sin(-math.pi / 2 + math.pi * i / n)) for i in range(n + 1)]


def east_end():
    M = Me; kind = E["kind"]; uc = u_c1; C = M.C
    if kind == "chevet":
        n = E["n"]; r2 = E["r2"]; P1 = hemi(W / 2, n); P2 = hemi(r2, n); nch = E["chapels"]; rc = E["rc"]; s_ = E["straight"]; z_cap_h = M.z_arc
        crown_c = M.Ea * 0.92 - 0.06 * rc - 0.6 * kq
        for i in range(n):                                                       # the semicircle: an arcade, a triforium, a clerestory
            p, q = P1[i], P1[i + 1]; ch = math.dist(p, q); w_ar = max(1.0, arc_w(M, ch)); holes = [(ch / 2, 0.0, w_ar, M.z_arc, M.KIND)]; trac = []
            z_cap_h = M.z_arc - min(BA.arch_rise(M.KIND, w_ar), M.z_arc * 0.95)
            if M.tri:
                for j in (-1, 1): holes.append((ch / 2 + j * ch / 4, M.tri["z0"], min(M.tri["w"], 0.3 * ch), M.tri["h"], M.KIND))
            if M.win["cl"]: win(M, ch / 2, M.win["cl"]["z0"], min(M.win["cl"]["w"], 0.62 * ch), M.win["cl"]["h"] * 0.9, holes, trac)
            panel(AW, p, q, ZB, M.He, M.tc, holes, C["stone"], M, trac=trac, T_=Lc)
            BA.rib_vault(AV, [Lc(*P1[i]), Lc(*P2[i]), Lc(*P2[i + 1]), Lc(*P1[i + 1])], Z(M.spring_a), M.ra, C["vault"], C["rib"],
                         kind=M.vault if M.vault != "star" else "rib4", rib_w=0.24 * kq, a_rib=AP)
            p2, q2 = P2[i], P2[i + 1]; ch2 = math.dist(p2, q2); holes = []; trac = []
            if nch: holes.append((ch2 / 2, 0.0, min(2 * rc - 0.3 * kq, ch2 - 0.6 * kq), min(M.z_arc * 0.92, crown_c - 0.3 * kq), M.KIND))
            else: win(M, ch2 / 2, M.win["ai"]["z0"], min(M.win["ai"]["w"], 0.6 * ch2), M.win["ai"]["h"], holes, trac)
            panel(AW, p2, q2, ZB, M.Ea, M.t, holes, C["stone"], M, trac=trac, T_=Lc)
            a0_, a1_ = -math.pi / 2 + math.pi * i / n, -math.pi / 2 + math.pi * (i + 1) / n; ri, ro = W / 2 + M.tc / 2, r2 + M.t / 2 + M.ov
            AR.face([L3(uc + ri * math.cos(a0_), ri * math.sin(a0_), M.z_lean, Lc), L3(uc + ri * math.cos(a1_), ri * math.sin(a1_), M.z_lean, Lc),
                     L3(uc + ro * math.cos(a1_), ro * math.sin(a1_), M.Ea - 0.3 * kq, Lc), L3(uc + ro * math.cos(a0_), ro * math.sin(a0_), M.Ea - 0.3 * kq, Lc)], C["roof"])
            if nch: chapel(M, -math.pi / 2 + math.pi * (i + 0.5) / n, r2, rc, s_)
        for i in range(1, n): pier_at(M, P1[i][0], P1[i][1], z_cap_h, True, (uc - P1[i][0], -P1[i][1]), z_shaft=M.spring, T_=Lc, DT=Dc)
        BA.radial_vault(AV, Lc(uc, 0), [Lc(*p) for p in P1], Z(M.spring), M.rise, C["vault"], C["rib"], rib_w=0.3 * kq, a_rib=AP)
        hw = W / 2 + M.tc / 2 + M.ov
        if M.roof == "gable":
            BA.roof_fan(AR, (*Lc(uc, 0), Z(M.ridge)), [Lc(uc + hw * math.cos(a_), hw * math.sin(a_)) for a_ in np.linspace(-math.pi / 2, math.pi / 2, n + 1)], Z(M.He), C["roof"])
        else: AR.face([(*Lc(uc, 0), Z(M.He))] + [(*Lc(uc + hw * math.cos(a_), hw * math.sin(a_)), Z(M.He)) for a_ in np.linspace(-math.pi / 2, math.pi / 2, n + 1)], C["roof"])
    elif kind == "apse":                                                         # the apse lower than the choir, with the east gable above it
        n = E["n"]; P = hemi(W / 2, n); top = E["top"]
        for i in range(n):
            p, q = P[i], P[i + 1]; ch = math.dist(p, q); holes = []; trac = []
            if i % 2 == 1: win(M, ch / 2, 0.42 * top, min(0.62 * ch, max(M.win["cl"]["w"] if M.win["cl"] else 1.5, 1.4 * kq)), 0.3 * top, holes, trac)
            panel(AW, p, q, ZB, top, M.t, holes, C["stone"], M, trac=trac, T_=Lc)
        if M.rise > 1e-3: BA.radial_vault(AV, Lc(uc, 0), [Lc(*p) for p in P], Z(M.spring), W / 2, C["vault"], C["rib"], rf=math.dist(P[0], P[1]) / 2, ribs=False)
        else: AV.face([(*Lc(uc, 0), Z(M.H))] + [(*Lc(*p), Z(M.H)) for p in P], C["vault"])
        hr = W / 2 + M.t / 2 + M.ov
        BA.roof_fan(AR, (*Lc(uc, 0), Z(top + hr * math.tan(math.radians(max(M.roof_pitch, 25))) * 0.9)),
                    [Lc(uc + hr * math.cos(a_), hr * math.sin(a_)) for a_ in np.linspace(-math.pi / 2, math.pi / 2, n + 1)], Z(top), C["roof"])
        Lw = W + M.t; panel(AW, (uc, -(W / 2 + M.t / 2)), (uc, W / 2 + M.t / 2), ZB, M.He, M.t, [(Lw / 2, 0.0, W - M.t, M.H - 0.4 * kq, M.KIND)], C["stone"], M,
                            top=gable(M, Lw, Lw / 2, W / 2 + M.tc / 2), T_=Lc)
    elif kind == "polygon":                                                      # the choir's polygonal apse
        n = E["n"]; P = hemi(W / 2, n); cw = M.win["choir"][0] if M.win["choir"] else M.win["ai"]
        for i in range(n):
            p, q = P[i], P[i + 1]; ch = math.dist(p, q); holes = []; trac = []
            win(M, ch / 2, cw["z0"], min(cw["w"], 0.55 * ch), cw["h"] * 0.92, holes, trac)
            panel(AW, p, q, ZB, M.He, M.t, holes, C["stone"], M, trac=trac, T_=Lc)
        BA.radial_vault(AV, Lc(uc, 0), [Lc(*p) for p in P], Z(M.spring), M.rise, C["vault"], C["rib"], rib_w=0.3 * kq, a_rib=AP)
        hw = W / 2 + M.t / 2 + M.ov
        BA.roof_fan(AR, (*Lc(uc, 0), Z(M.He + hw * math.tan(math.radians(max(M.roof_pitch, 25))))), [Lc(uc + hw * math.cos(a_), hw * math.sin(a_))
                                                                                               for a_ in np.linspace(-math.pi / 2, math.pi / 2, n + 1)], Z(M.He), C["roof"])
    else:                                                                        # a flat east end (+ a Lady chapel)
        vo = W / 2 + A + M.t / 2 if CH_AIS else W / 2 + M.t / 2; Lw = 2 * vo; s1 = vo - W / 2 - M.tc / 2
        top = [(0.0, M.Ea), (s1 - 0.02, M.z_lean)] + [(s1 + s_, z_) for s_, z_ in gable(M, W + M.tc, (W + M.tc) / 2, W / 2 + M.tc / 2)] + [(Lw - s1 + 0.02, M.z_lean), (Lw, M.Ea)]
        lady = E.get("lady", 0.0); lw = E.get("lady_w", 0.7 * W); lh = E.get("lady_h", M.Ha)
        holes = []
        if lady: holes.append((vo, 0.0, lw - M.t, lh - 2.0 * kq, M.KIND))
        zl0 = (lh + 2.0 * kq) if lady else M.win["ai"]["z0"]
        for x, hk in ((-1, 0.8), (0, 1.0), (1, 0.8)):
            hl = (M.He - 1.5 * kq - zl0) * hk
            if hl > 3 * kq: holes.append((vo + x * 0.21 * W, zl0, 0.15 * W, hl, "lancet" if M.KIND == "pointed" else M.KIND))
        if CH_AIS:
            for x in (-1, 1): holes.append((vo + x * (W / 2 + A / 2), M.win["ai"]["z0"], min(M.win["ai"]["w"], 0.55 * A), M.win["ai"]["h"] * 0.9, M.KIND))
        panel(AW, (uc, -vo), (uc, vo), ZB, M.He, M.t, holes, C["stone"], M, top=top, T_=Lc)
        if lady:
            ul = uc + lady; rl = 0.62 * lw; spl = lh - 0.06 * lw - 0.6 * kq - rl
            for sgn in (1, -1):
                for j in range(2):
                    ua, ub = uc + j * lady / 2, uc + (j + 1) * lady / 2; hl_ = []; tr_ = []
                    win(M, lady / 4, 2.4 * kq, min(M.win["ai"]["w"], 0.45 * lady / 2), spl - 2.4 * kq + rl * 0.6, hl_, tr_)
                    panel(AW, (ua, sgn * lw / 2), (ub, sgn * lw / 2), ZB, lh, M.t, hl_, C["stone"], M, trac=tr_, T_=Lc)
            for j in range(2):
                ua, ub = uc + j * lady / 2, uc + (j + 1) * lady / 2
                BA.rib_vault(AV, [Lc(ua, lw / 2), Lc(ua, -lw / 2), Lc(ub, -lw / 2), Lc(ub, lw / 2)], Z(spl), rl, C["vault"], C["rib"], rib_w=0.22 * kq, a_rib=AP)
            Lw2 = lw + M.t; pl_ = max(M.roof_pitch, 30)
            panel(AW, (ul, -(lw / 2 + M.t / 2)), (ul, lw / 2 + M.t / 2), ZB, lh, M.t,
                  [(Lw2 / 2 + x * 0.22 * lw, 2.6 * kq, 0.14 * lw, (spl + 0.4 * rl - 2.6 * kq) * hk, "lancet" if M.KIND == "pointed" else M.KIND) for x, hk in ((-1, 0.8), (0, 1.0), (1, 0.8))],
                  C["stone"], M, top=gable(M, Lw2, Lw2 / 2, lw / 2 + M.t / 2, z_lo=lh, z_peak=lh + (lw / 2 + M.t / 2 + M.ov) * math.tan(math.radians(pl_))), T_=Lc)
            BA.roof_gable(AR, Lc(uc - 0.5, 0), Lc(ul, 0), lw / 2 + M.t / 2, Z(lh), pl_, C["roof"], ov=M.ov)

    if kind in ("apse", "polygon") and CH_AIS:                                   # the ends of the choir aisles at the apse — walls with a window
        for sgn in (1, -1):
            hl = []; tr_ = []; win(M, A / 2, M.win["ai"]["z0"], min(M.win["ai"]["w"], 0.5 * A), M.win["ai"]["h"] * 0.9, hl, tr_)
            panel(AW, (uc, sgn * (W / 2 + M.tc / 2)), (uc, sgn * (W / 2 + A + M.t / 2)), ZB, M.Ea, M.t, hl, C["stone"], M, trac=tr_, T_=Lc)

def chapel(M, a_m, r2, rc, s_):
    """A radial chapel: straight walls from the ambulatory, a 5/8 polygon around the center, windows, a fan vault, a pyramidal
    roof."""
    dv = np.array([math.cos(a_m), math.sin(a_m)]); tg = np.array([-dv[1], dv[0]]); co = np.array([u_c1, 0.0]) + dv * r2; cc = co + dv * (s_ + M.t / 2)
    Q = [co - tg * rc, cc - tg * rc] + [cc + rc * np.array([math.cos(a_m + f), math.sin(a_m + f)]) for f in (-math.pi / 4, 0.0, math.pi / 4)] + [cc + tg * rc, co + tg * rc]
    Ec = M.Ea * 0.92; crown = Ec - 0.06 * rc - 0.6 * kq; rise_c = min(rc * 1.05, crown * 0.4); spr_c = crown - rise_c
    for i in range(len(Q) - 1):
        p, q = Q[i], Q[i + 1]; L_ = math.dist(p, q); holes = []; trac = []
        if 1 <= i <= 4 and L_ > 1.2 * kq: win(M, L_ / 2, 2.4 * kq, 0.42 * L_, spr_c + 0.55 * rise_c - 2.4 * kq, holes, trac)
        panel(AW, tuple(p), tuple(q), ZB, Ec, M.t * 0.8, holes, M.C["stone"], M, trac=trac, T_=Lc)
    BA.radial_vault(AV, Lc(*cc), [Lc(*q) for q in Q[1:-1]], Z(spr_c), rise_c, M.C["vault"], M.C["rib"], rib_w=0.2 * kq, a_rib=AP)
    if s_ > 0.3 * kq: BA.rib_vault(AV, [Lc(*Q[-1]), Lc(*Q[0]), Lc(*Q[1]), Lc(*Q[-2])], Z(spr_c), rise_c, M.C["vault"], M.C["rib"], rib_w=0.2 * kq, a_rib=AP)
    AF.prism([Lc(*q) for q in Q], Z(ZB), Z(0.0), M.C["floor"])
    ring = [Lc(*(cc + (q - cc) * (1 + (M.t * 0.4 + M.ov) / max(np.linalg.norm(q - cc), 1e-6)))) for q in Q]
    BA.roof_fan(AR, (*Lc(*cc), Z(Ec + rc * 1.1)), ring, Z(Ec), M.C["roof"])


# ---------------------------------------------------------------- towers
def foot(c, half, plan):
    if plan == "square": return [c + AXV * x * half + PVV * y * half for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    rr = half / math.cos(math.pi / 8); a0 = math.atan2(AXV[1], AXV[0]) + math.pi / 8
    return [c + rr * np.array([math.cos(a0 + math.pi / 4 * i), math.sin(a0 + math.pi / 4 * i)]) for i in range(8)]


def church_tower(tw, low_skip=None, T_=L, doors=None):
    """A tiered tower (temple_plan.tower_plan): a base with corner buttresses, bell-tower tiers with openings by style, belts,
    pinnacles at the transition to the octagon, the top — bl_arch.crown; an unfinished one — a temporary roof and a crane.
    doors — real openings in the base faces: [{"side": w|e|in|out, "at": (u, v), "w", "h", "kind"}] (the space scheme)."""
    c = T_(tw["u"], tw["v"]); st = tw["stages"]; sk = tw["spire"]; low_skip = low_skip or {}; C = cols_of(tw.get("colors") or T["colors"])
    open_k = st[-1].get("open", "pointed") if len(st) > 1 else "pointed"
    open_k = {"round": "round", "stepped": "stepped"}.get(open_k, "pointed")
    for k, sg in enumerate(st):
        P = foot(c, sg["half"], sg["plan"]); n = len(P); z0 = Z(sg["z0"] if (k or sg["z0"] > 0.5) else ZB); z1 = Z(sg["z1"])
        for i in range(n):
            p, q = P[i], P[(i + 1) % n]; mid = (p + q) / 2; nn = (mid - c) / max(np.linalg.norm(mid - c), 1e-9)
            side = "w" if nn @ AXV < -0.7 else ("e" if nn @ AXV > 0.7 else ("in" if (nn @ PVV) * tw["v"] < 0 else "out"))
            zz0 = max(z0, Z(low_skip[side])) if (k == 0 and side in low_skip) else z0
            dr = [d_ for d_ in (doors or []) if k == 0 and d_["side"] == side]
            if dr:                                                               # the base door: a wall with an opening (an entrance per the scheme)
                dd = (q - p) / max(float(np.linalg.norm(q - p)), 1e-9)
                SKIP[0] += BA.wall_panel(AT, p, q, zz0, z1, 0.5 * kq, [(float((T_(*d_["at"]) - p) @ dd), Z(0.0), d_["w"], d_["h"], d_.get("kind", "pointed")) for d_ in dr], C["stone"])
            else: AT.face([(*p, zz0), (*q, zz0), (*q, z1), (*p, z1)], C["stone"])
            if sg["kind"] in ("belfry", "lantern"):
                hz = sg["z1"] - sg["z0"]; ln = float(np.linalg.norm(q - p))
                cnt = 1 if sg["plan"] == "poly8" else (2 if open_k == "pointed" else min(3, 1 + k))
                for j in range(cnt):
                    x = (j + 0.5) / cnt; wv = min(0.7 * ln / cnt, (0.22 if open_k == "pointed" else 0.16) * 2 * sg["half"])
                    dark_open(AT, p + (q - p) * x, nn, Z(sg["z0"] + 0.18 * hz), wv, 0.62 * hz if open_k == "pointed" else 0.4 * hz, C["hole"], open_k)
        Pb = [c + (p - c) * (1 + 0.5 * kq / max(np.linalg.norm(p - c), 1e-6)) for p in P]   # a belt cornice with a projection
        AT.quad_strip([(*p, z1 - 0.45 * kq) for p in Pb] + [(*Pb[0], z1 - 0.45 * kq)], [(*p, z1) for p in Pb] + [(*Pb[0], z1)], C["dark"])
        AT.face([(*p, z1) for p in Pb], C["dark"])
        if tw.get("buttresses") and sg["plan"] == "square" and k <= 1:           # corner buttresses: along the faces, flush with the corner
            bw_ = 0.12 * 2 * sg["half"]; bd_ = 0.16 * 2 * sg["half"]; zt = z1 - 2.5 * kq
            for p in P:
                n1 = AXV * (1.0 if (p - c) @ AXV > 0 else -1.0); n2 = PVV * (1.0 if (p - c) @ PVV > 0 else -1.0)
                for dd, other in ((n1, n2), (n2, n1)):
                    if k == 0 and (("in" in low_skip and (dd @ PVV) * tw["v"] < -0.5) or ("e" in low_skip and dd @ AXV > 0.5)): continue
                    AT.box(p + dd * bd_ / 2 - other * bw_ / 2, dd, bw_, bd_, z0, z0 + (zt - z0) * 0.55, C["stone"])
                    AT.box(p + dd * bd_ * 0.35 - other * bw_ / 2, dd, bw_, bd_ * 0.7, z0 + (zt - z0) * 0.55, zt, C["stone"])
                    if tw.get("pinnacles"): BA.pinnacle(AT, p + dd * bd_ * 0.35 - other * bw_ / 2, zt, bw_ * 0.8, 1.2 * bw_, C["stone"], C["roof"])
        if k < len(st) - 1:
            nx = st[k + 1]
            if nx["plan"] == "poly8" and sg["plan"] == "square" and tw.get("pinnacles"):
                wp = 0.16 * 2 * sg["half"]; hp = 0.45 * (nx["z1"] - nx["z0"])
                for p in P: BA.pinnacle(AT, c + (p - c) * 0.86, z1, wp, hp, C["stone"], C["roof"])
            elif sg["half"] - nx["half"] > 0.3 * kq:
                AT.quad_strip([(*p, z1) for p in P] + [(*P[0], z1)], [(*p, z1 + 1.1 * kq) for p in P] + [(*P[0], z1 + 1.1 * kq)], C["stone"])
    top = st[-1]; P = foot(c, top["half"], top["plan"]); zt = Z(top["z1"])
    BA.crown(AT, AR, c, P, top["half"], top["plan"], zt, sk["kind"], sk["h"], C, pinnacles=tw.get("pinnacles", False),
             fort=FORT and sk["kind"] == "flat", seed=float(tw["u"] * 0.13 + tw["v"] * 0.07), s=kq)
    if tw.get("crane"): BA.crane(AT, c + (P[0] - c) * 0.5, zt + sk["h"] * 0.2, 0.9 * top["half"] + 3.0 * kq, BA.rc((0.34, 0.25, 0.17), "wood"), out=tuple(-AXV))


# ---------------------------------------------------------------- west end
def west():
    towers = T["towers"]; po = T["portal"]; pw, ph = po["w"], po["h"]; M = Mw; C = M.C
    west_tw = [tw for tw in towers if tw["role"] == "west"]
    z_lo = Z(-float(FR["floor"]) + (T["podium"]["levels"] * T["podium"]["h"] if T.get("podium") else 0.0))
    kd = M.KIND if M.KIND != "flat" else "round"
    for tw in towers:
        dr = None; sg_ = 1.0 if tw["v"] > 0 else -1.0
        if tw["role"] == "west" and len(west_tw) == 2 and tw.get("portal") and (STY.get("unfinished") or {}).get("kind") != "nave":
            dr = [{"side": "w", "at": (tw["u"] - tw["side"] / 2, tw["portal"]["v"]), "w": tw["portal"]["w"], "h": tw["portal"]["h"], "kind": kd}]
        elif tw["role"] == "west" and len(west_tw) == 1:
            dr = [{"side": "w", "at": (tw["u"] - tw["side"] / 2, 0.0), "w": pw, "h": ph, "kind": kd}, {"side": "e", "at": (tw["u"] + tw["side"] / 2, 0.0), "w": pw * 0.8, "h": ph * 0.9, "kind": kd}]
        church_tower(tw, low_skip={"in": M.z_lean - 1.0 * kq, "e": M.Ea - 0.5 * kq} if (tw["role"] == "west" and len(west_tw) == 2) else None, doors=dr)
    UNF = STY.get("unfinished") or {}
    if UNF.get("kind") == "nave":                                                # Beauvais: a temporary west wall, there is no nave
        vo = W / 2 + A + M.t / 2; Lw = 2 * vo
        panel(AW, (0.0, -vo), (0.0, vo), ZB, M.He, M.t * 0.8, [(vo, 0.0, 2.6 * kq, 4.0 * kq, "flat")], BA.shade(C["stone"], 0.92), M, top=gable(M, Lw, vo, W / 2 + M.tc / 2))
        portal_main = (L(-M.t / 2, 0.0), 0.2)
    elif T.get("screen"):                                                        # England: a screen facade with turrets at the corners
        sc = T["screen"]; hw = sc["half"]; Lw = 2 * hw; zs = sc["z"]
        top = [(0.0, zs), (hw - W / 2 - M.tc / 2, zs), (hw, M.ridge + 2.0 * kq), (hw + W / 2 + M.tc / 2, zs), (Lw, zs)]
        holes = [(hw, 0.0, pw, ph, M.KIND)]
        for x in (-1, 1): holes.append((hw + x * (W / 2 + A / 2), 0.0, pw * 0.6, ph * 0.75, M.KIND))
        zl0 = ph + 4.0 * kq
        for x, hk in ((-1, 0.82), (0, 1.0), (1, 0.82)):
            hl = (zs - 2.5 * kq - zl0) * hk
            if hl > 3 * kq: holes.append((hw + x * 0.2 * W, zl0, 0.14 * W, hl, "lancet" if M.KIND == "pointed" else M.KIND))
        panel(AW, (0.0, -hw), (0.0, hw), ZB, zs, M.t * 1.3, holes, C["stone"], M, top=top)
        for x in (-1, 1):
            BA.porch(AXT, L(-M.t * 0.65, x * (W / 2 + A / 2)), D(-1, 0), Z(0.0), pw * 0.6, ph * 0.75, M.KIND, 0.2 * pw, C["stone"])
            if sc["turrets"]:
                cpt = L(0.0, x * (hw - sc["turret_r"] * 0.6)); P8 = foot(cpt, sc["turret_r"], "poly8")
                AT.prism(P8, Z(ZB), Z(sc["turret_h"]), C["stone"]); BA.roof_fan(AR, (*cpt, Z(sc["turret_h"] + 3.0 * sc["turret_r"])), P8 + [P8[0]], Z(sc["turret_h"]), C["roof"])
        portal_main = (L(-M.t * 0.65, 0.0), 0.3 * pw)
    elif len(west_tw) == 2:                                                      # two towers: the middle between them, a gallery (Notre-Dame, Amiens)
        hw = W / 2 + M.tc / 2 + 0.35 * kq; Lw = 2 * hw; z_f = west_tw[0]["stages"][0]["z1"] if not west_tw[0].get("unfinished") else west_tw[1]["stages"][0]["z1"]
        gal = M.KIND not in ("round", "stepped", "flat"); z_g = z_f + (0.07 * M.H if gal else 0.0)
        top = [(0.0, z_g), (Lw, z_g)] if gal else gable(M, Lw, hw, W / 2 + M.tc / 2, z_lo=z_f)
        holes = [(hw, 0.0, pw, ph, M.KIND)]
        if gal:
            ng = max(5, int(Lw / (1.7 * kq))) | 1; zg0 = z_f - 0.05 * M.H
            for i in range(ng): holes.append(((i + 0.5) * Lw / ng, zg0, 0.55 * Lw / ng, z_g - 0.6 * kq - zg0, "lancet" if M.KIND == "pointed" else M.KIND))
            ug = M.t / 2 + M.ov + 0.15 * kq
            panel(AW, (ug, -(W / 2 + M.tc / 2)), (ug, W / 2 + M.tc / 2), M.He - 1.0 * kq, M.He, M.tc, [], C["stone"], M, top=gable(M, W + M.tc, W / 2 + M.tc / 2, W / 2 + M.tc / 2))
        if T.get("rose"): r_ = T["rose"]["r"]; zr = T["rose"]["z"]; holes.append((hw, zr - r_, 2 * r_, 2 * r_, "circle"))
        else:
            zw = max(ph + 3.0 * kq, M.z_tri)
            for x in (-1, 0, 1): holes.append((hw + x * 0.24 * W, zw, 0.13 * W, min(0.22 * M.H, z_f - zw - 2.0 * kq), M.KIND))
        panel(AW, (0.0, -hw), (0.0, hw), ZB, z_f, M.t, holes, C["stone"], M, top=top)
        if T.get("rose"): BA.rose(AXT, L(0.0, 0.0), D(-1, 0), Z(zr), r_, C["trac"], th=0.22 * kq)
        for tw in west_tw:                                                       # the side portals are in the tower bases (the plan: within the face and the aisle)
            if tw.get("portal"):
                fc = L(tw["u"] - tw["side"] / 2, tw["portal"]["v"])
                BA.porch(AXT, fc, D(-1, 0), Z(0.0), pw * 0.75, ph * 0.8, M.KIND, 0.25 * pw, C["stone"], gable=M.KIND not in ("round", "stepped", "flat"))
                BA.steps(AF, fc + D(-1, 0) * 0.25 * pw, D(1, 0), pw * 1.3, z_lo, FL, C["dark"])
        if EX.get("westwork"):                                                   # westwork: the west block with a central tower (Corvey)
            ww = EX["westwork"]; u0w, u1w = -M.t / 2, 0.9 * Mn.B; vw = W / 2 + M.tc / 2 + 0.3 * kq   # above the first bay, higher than the nave roof
            cpt = L((u0w + u1w) / 2, 0.0); P = [L(u0w, -vw), L(u1w, -vw), L(u1w, vw), L(u0w, vw)]
            AT.prism(P, Z(ww["z0"]), Z(ww["z1"]), C["stone"], top=False)
            for i in range(4):
                p, q = P[i], P[(i + 1) % 4]; mid = (p + q) / 2; nn = (mid - cpt) / max(np.linalg.norm(mid - cpt), 1e-9)
                for j in range(3): dark_open(AT, p + (q - p) * (j + 1) / 4, nn, Z(ww["z1"] - 0.35 * (ww["z1"] - ww["z0"])), 0.12 * ww["side"], 0.22 * (ww["z1"] - ww["z0"]), C["hole"], "round")
            BA.crown(AT, AR, cpt, P, min(vw, (u1w - u0w) / 2), "rect", Z(ww["z1"]), "pyramid" if M.KIND != "stepped" else "stepped", 0.35 * ww["side"], C, s=kq)
        portal_main = (L(-M.t / 2, 0.0), 0.4 * kq + 0.3 * pw)
    else:                                                                        # one tower in front of the facade (Ulm) or no towers
        vo = W / 2 + A + M.t / 2; Lw = 2 * vo
        top = gable(M, Lw, vo, vo) if M.hall else gable(M, Lw, vo, W / 2 + M.tc / 2)
        holes = [(vo, 0.0, pw * 0.8, ph * 0.9, M.KIND)]; trac = []
        for x in (-1, 1): win(M, vo + x * (W / 2 + A / 2), M.win["ai"]["z0"], M.win["ai"]["w"], M.win["ai"]["h"], holes, trac)
        panel(AW, (0.0, -vo), (0.0, vo), ZB, M.He, M.t, holes, C["stone"], M, top=top, trac=trac)
        tw = west_tw[0] if west_tw else None
        if tw:
            fc = L(tw["u"] - tw["side"] / 2, 0.0); portal_main = (fc, 0.3 * pw); hs_ = tw["side"] / 2      # the tower base is a vestibule: a floor
            AF.prism([L(tw["u"] - hs_, -hs_), L(0.0, -hs_), L(0.0, hs_), L(tw["u"] - hs_, hs_)], Z(ZB), Z(0.0), C["floor"])
        else: portal_main = (L(-M.t / 2, 0.0), 0.3 * pw)
    pc, dp = portal_main
    if EX.get("porch"):                                                          # a galilee porch in front of the portal (Durham, Ely)
        pr = EX["porch"]; d_, w_, h_ = pr["d"], pr["w"], pr["h"]; u1 = pr.get("u1", -M.t / 2); u0 = pr.get("u0", u1 - d_); pl_ = max(M.roof_pitch, 30)
        panel(AW, (u0, -w_ / 2), (u0, w_ / 2), ZB, h_, M.t * 0.7, [(w_ / 2, 0.0, pw * 1.15, ph * 1.05, M.KIND)], C["stone"], M,
              top=gable(M, w_, w_ / 2, w_ / 2, z_lo=h_, z_peak=h_ + w_ / 2 * math.tan(math.radians(pl_))))
        for sgn in (1, -1): panel(AW, (u0, sgn * w_ / 2), (u1, sgn * w_ / 2), ZB, h_, M.t * 0.7, [(d_ / 2, 1.0 * kq, 0.5 * d_, 0.6 * h_, M.KIND)], C["stone"], M)
        BA.rib_vault(AV, [L(u0, w_ / 2), L(u0, -w_ / 2), L(u1, -w_ / 2), L(u1, w_ / 2)], Z(h_ * 0.55), h_ * 0.35, C["vault"], C["rib"], rib_w=0.2 * kq, a_rib=AP)
        BA.roof_gable(AR, L(u0, 0.0), L(u1, 0.0), w_ / 2 + 0.3, Z(h_), pl_, C["roof"], ov=0.4)
        AF.prism([L(u0, -w_ / 2), L(u1, -w_ / 2), L(u1, w_ / 2), L(u0, w_ / 2)], Z(ZB), Z(0.0), C["floor"])
        pc = L(u0 - M.t * 0.35, 0.0); dp = 0.1
    else: BA.porch(AXT, pc, D(-1, 0), Z(0.0), pw, ph, M.KIND, dp, C["stone"], gable=M.KIND not in ("round", "stepped", "flat"))
    BA.steps(AF, pc + D(-1, 0) * dp, D(1, 0), pw * 1.8, z_lo, FL, C["dark"])
    man(pc + D(-1, 0) * (dp + 3.5 * kq) + D(0, 1) * pw * 0.4, z_lo)


def man(p, z):
    """A 1.8 m human — the scale reference (the canon)."""
    cyl(AM, p, 0.24, z, z + 1.5, MAN, n=8); cyl(AM, p, 0.13, z + 1.5, z + 1.8, MAN, n=8)


# ---------------------------------------------------------------- supports
def buttresses():
    for b in T["buttresses"]:
        M = MO[b["mod"]]; T_ = Lc if b["part"] == "choir" else L; DT = Dc if b["part"] == "choir" else D; C = M.C
        q = T_(b["u"], b["v"]); n = DT(*b["n"]); w_, d_, zt = b["w"], b["d"], b["z_top"]; form = b.get("form", "buttress"); tg = np.array([-n[1], n[0]])
        spiky = M.spikes >= 0.5
        if form == "lesene":
            AB.box(q + n * d_ / 2, n, w_, d_, Z(ZB), Z(zt), C["stone"]); continue
        if form == "turret":                                                     # fortress (Albi): a semicircular tower buttress
            rr = max(0.55 * w_, 0.35 * d_); cyl(AB, q + n * rr * 0.6, rr, Z(ZB), Z(zt + 2.5 * kq), C["stone"], n=12)
            BA.merlon_row(AB, BA.octagon(q + n * rr * 0.6, rr * 0.95, n=12), Z(zt + 2.5 * kq), 0.7 * kq, C["stone"]); continue
        if form == "branch":                                                     # elves: a trunk-like support widening toward the ground
            BA.tube(AB, [(*(q + n * d_ * 0.5), Z(ZB)), (*(q + n * d_ * 0.42), Z(zt * 0.45)), (*(q + n * d_ * 0.3), Z(zt))], [w_ * 0.95, w_ * 0.6, w_ * 0.45], C["stone"], n=8)
        else:
            zm = ZB + (zt - ZB) * 0.55
            AB.box(q + n * d_ / 2, n, w_, d_, Z(ZB), Z(zm), C["stone"]); AB.box(q + n * d_ * 0.35, n, w_, d_ * 0.7, Z(zm), Z(zt), C["stone"])
            AB.face([(*(q + n * d_ - tg * w_ / 2), Z(zm)), (*(q + n * d_ + tg * w_ / 2), Z(zm)), (*(q + n * d_ * 0.7 + tg * w_ / 2), Z(zm + 0.3 * d_)),
                     (*(q + n * d_ * 0.7 - tg * w_ / 2), Z(zm + 0.3 * d_))], C["dark"])
        if b.get("pinnacle"):
            if spiky: BA.spike(AB, q + n * d_ * 0.4, Z(zt), 0.5 * w_, b["pinnacle"] * 1.1, C["dark"])
            elif form != "branch": BA.pinnacle(AB, q + n * d_ * 0.4, Z(zt), 0.6 * w_, b["pinnacle"] * 0.45, C["stone"], C["roof"])
            else: BA.lathe(AB, q + n * d_ * 0.3, BA.dome_profile("leaf", 0.5 * w_, b["pinnacle"] * 0.9), Z(zt), C["accent"], n=8)
        if b["kind"] == "flyer":
            pw_ = q - n * b["span"]
            for tr in b["tiers"]:
                if form == "claw": BA.claw(AB, q + n * 0.15 * d_, pw_, Z(tr["z_pier"]), Z(tr["z_wall"]), 0.55 * w_, b["th"], C["stone"], C["dark"])
                elif form == "branch": BA.branch(AB, q + n * 0.25 * d_, pw_, Z(tr["z_pier"]), Z(tr["z_wall"]), 0.7 * w_ + 0.2 * b["th"], C["stone"])
                else: BA.flyer(AB, q + n * 0.15 * d_, pw_, Z(tr["z_pier"]), Z(tr["z_wall"]), 0.55 * w_, b["th"], C["stone"])
            AB.box(pw_ + n * 0.2 * M.tc, n, 0.6 * w_, 0.4 * M.tc, Z(M.z_lean - 0.5 * kq), Z(M.He - 0.3 * kq), C["stone"])
        if b.get("tower"):                                                       # hive: a tower pier with a spire
            tw_ = b["tower"]; s_ = tw_["side"]; cb = q + n * (d_ - s_ / 2); P = [cb + (AXV * x + PVV * y) * s_ / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            AB.prism(P, Z(ZB), Z(tw_["z1"]), C["stone"])
            for k_ in range(3):
                for p, qq in zip(P, P[1:] + P[:1]):
                    mid = (p + qq) / 2; nn = (mid - cb) / max(np.linalg.norm(mid - cb), 1e-9)
                    dark_open(AB, mid, nn, Z(zt + (tw_["z1"] - zt) * (0.15 + 0.28 * k_)), 0.25 * s_, 0.22 * (tw_["z1"] - zt), C["hole"], "pointed")
            for p in P: BA.pinnacle(AB, cb + (p - cb) * 0.9, Z(tw_["z1"]), 0.14 * s_, 0.3 * s_, C["stone"], C["roof"])
            P8 = foot(cb, s_ * 0.38, "poly8"); BA.roof_fan(AR, (*cb, Z(tw_["z1"] + tw_["spire"])), P8 + [P8[0]], Z(tw_["z1"]), C["roof"])
            if M.industrial >= 0.4 and int(abs(b["u"]) * 7) % 3 == 0:            # factory hive: pipes on some of the supports
                BA.stack(AB, cb + (P[0] - cb) * 0.55, 0.12 * s_, Z(tw_["z1"]), Z(tw_["z1"] + 0.6 * tw_["spire"]), C["dark"], C["accent"])


# ---------------------------------------------------------------- podium, floor, roofs
def podium():
    pd = T.get("podium")
    if not pd: return
    nlev, h, mg, run, sw = pd["levels"], pd["h"], pd["margin"], pd["run"], pd["stair_w"]; C = Mn.C
    uw = min([-Mn.t / 2 - 0.4 * kq] + [tw["u"] - tw["side"] / 2 for tw in T["towers"]]) - 2.0 * kq; ue = FR["u_end"] + 2.0 * kq; vh = FR["v_half"] + 2.0 * kq
    for i in range(nlev):
        e = mg * (nlev - i); zb_, zt_ = (Z0 - 3.0 if i == 0 else Z0 + i * h - 0.5), Z0 + (i + 1) * h
        P = [L(uw - e, -vh - e), L(ue + e, -vh - e), L(ue + e, vh + e), L(uw - e, vh + e)]
        AF.prism(P, zb_, zt_, BA.shade(C["stone"], 0.9), scale_top=1.0)
        for rn in BA.ring_runs(P, [(L(uw - e, 0.0), sw + 1.0 * kq)]):           # the tier parapet — with a gap at the flight
            AF.quad_strip([(*p, zt_) for p in rn], [(*p, zt_ + 1.2 * kq) for p in rn], C["dark"])
        for x, y in ((uw - e, -vh - e), (ue + e, -vh - e), (ue + e, vh + e), (uw - e, vh + e)):   # corner bastions
            s_ = 0.35 * mg; cpt = L(x, y); Pb = [cpt + (AXV * a_ + PVV * b_) * s_ / 2 for a_, b_ in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            AF.prism(Pb, zb_, zt_ + 0.4 * h, C["stone"]); BA.roof_fan(AR, (*cpt, zt_ + 0.4 * h + 0.9 * s_), Pb + [Pb[0]], zt_ + 0.4 * h, C["roof"])
        z_from = Z0 if i == 0 else Z0 + i * h; run_ = run                         # a flight up to a tier: on the axis from the west; the first one — down to the ground
        if i == 0: pf_ = L(uw - e - run, 0.0) - AXV * 1.0; z_from = min(Z0, float(RLF.height(pf_[0], pf_[1]))); run_ = run * (zt_ - z_from) / h
        nst = max(1, round((zt_ - z_from) / 0.17))
        BA.steps(AF, L(uw - e, 0.0), D(1, 0), sw, z_from, zt_, BA.shade(C["stone"], 0.95), rise=(zt_ - z_from) / nst, tread=run_ / nst)
        for x in (-1, 1):
            a0 = L(uw - e - run, x * (sw / 2 + 0.6 * kq)); a1 = L(uw - e, x * (sw / 2 + 0.6 * kq))
            AF.box((a0 + a1) / 2, D(1, 0), 1.2 * kq, run, z_from - 0.5, z_from + 0.3, C["dark"])
            AF.face([(*a0, z_from + 0.3), (*a1, zt_ + 1.2 * kq), (*a1, zt_), (*a0, z_from)], C["dark"])


def floor():
    """The floor on the plinth: along the wall axes (the edges are inside the walls)."""
    def slab(poly, T_=L): AF.prism([T_(*p) for p in poly], Z(ZB), Z(0.0), Mn.C["floor"])
    slab([(0.0, -(W / 2 + A)), (u_n1, -(W / 2 + A)), (u_n1, W / 2 + A), (0.0, W / 2 + A)])
    if X_: slab([(u_n1, -max(v1, W / 2)), (u_x1, -max(v1, W / 2)), (u_x1, max(v1, W / 2)), (u_n1, max(v1, W / 2))])
    vc = W / 2 + A if CH_AIS else W / 2
    slab([(u_x1, -vc), (u_c1, -vc), (u_c1, vc), (u_x1, vc)], Lc)
    if E["kind"] == "chevet": slab(hemi(E["r2"], E["n"] * 2), Lc)
    elif E["kind"] in ("apse", "polygon"): slab(hemi(W / 2, E["n"]), Lc)
    elif E.get("lady"): slab([(u_c1, -E["lady_w"] / 2), (u_c1 + E["lady"], -E["lady_w"] / 2), (u_c1 + E["lady"], E["lady_w"] / 2), (u_c1, E["lady_w"] / 2)], Lc)


def roofs():
    ruin = STATE.get("nave_roofless")
    if Mn.hall:                                                                  # hall church: one roof over three aisles, the choir lower
        roof_part(Mn, L(0.0, 0.0), L(u_n1, 0.0), W / 2 + A + Mn.t / 2, hip1=True, ruin=ruin)
        roof_part(Me, Lc(u_n1, 0.0), Lc(u_c1, 0.0), W / 2 + Me.t / 2)
        return
    u_r0 = (Mn.t / 2 + Mn.ov + 0.15 * kq + Mn.ov) if (len([tw for tw in T["towers"] if tw["role"] == "west"]) == 2 and Mw.KIND not in ("round", "stepped", "flat")) else 0.0
    u_j = u_mid if X_ else u_n1
    roof_part(Mn, L(u_r0, 0.0), L(u_j, 0.0), W / 2 + Mn.tc / 2, ruin=ruin)
    roof_part(Me, Lc(u_j, 0.0), Lc(u_c1, 0.0), W / 2 + Me.tc / 2)
    if not ruin and abs(Mn.ridge - Me.ridge) > 0.6 * kq and Mn.roof == Me.roof == "gable":   # a gable at the junction of parts of different heights
        Mh = Mn if Mn.ridge > Me.ridge else Me; Ml = Me if Mh is Mn else Mn
        panel(AW, (u_j, -(W / 2 + Mh.tc / 2)), (u_j, W / 2 + Mh.tc / 2), Ml.He - 0.5 * kq, Mh.He, Mh.tc, [], Mh.C["stone"], Mh,
              top=gable(Mh, W + Mh.tc, W / 2 + Mh.tc / 2, W / 2 + Mh.tc / 2, z_lo=Ml.He), T_=Lc if Mh is Me else L)
    runs = [(Mn, 0.0, u_n1, L)] + ([(Me, u_x1, u_c1, Lc)] if CH_AIS else [])
    for M, ua, ub, T_ in runs:
        for sgn in (1, -1):
            u0 = ua
            for tw in T["towers"]:
                if tw["v"] * sgn > 0 and tw["u"] - tw["side"] / 2 <= u0 + 0.1 <= tw["u"] + tw["side"] / 2: u0 = tw["u"] + tw["side"] / 2 - 0.3 * kq
            u1 = ub
            for tw in T["towers"]:
                if tw["v"] * sgn > 0 and tw["u"] - tw["side"] / 2 <= u1 - 0.1 <= tw["u"] + tw["side"] / 2: u1 = tw["u"] - tw["side"] / 2 + 0.3 * kq
            if u1 - u0 < 0.5: continue
            if M.roof == "gable":
                BA.roof_lean(AR, T_(u0, sgn * (W / 2 + M.tc / 2)), T_(u1, sgn * (W / 2 + M.tc / 2)), (Dc if T_ is Lc else D)(0, sgn), A + M.t / 2 - M.tc / 2,
                             Z(M.Ea), Z(M.z_lean), M.C["roof"], ov=M.ov)
            else:
                AR.face([L3(u0, sgn * W / 2, M.Ea, T_), L3(u1, sgn * W / 2, M.Ea, T_), L3(u1, sgn * (W / 2 + A), M.Ea, T_), L3(u0, sgn * (W / 2 + A), M.Ea, T_)],
                        BA.shade(M.C["roof"], 0.9))


def crossing_top():
    cr = T.get("crossing")
    if not cr: return
    M = Mt; C = M.C
    if cr["kind"] == "spire":                                                    # flèche: an octagonal lantern and a needle
        c = L(u_mid, 0.0); P8 = foot(c, cr["r"], "poly8")
        AT.prism(P8, Z(cr["z0"]), Z(cr["z1"]), BA.shade(C["roof"], 1.1))
        for p, q in zip(P8, P8[1:] + P8[:1]):
            mid = (p + q) / 2; nn = (mid - c) / max(np.linalg.norm(mid - c), 1e-9)
            dark_open(AT, mid, nn, Z(cr["z0"] + 0.25 * (cr["z1"] - cr["z0"])), 0.5 * np.linalg.norm(q - p), 0.55 * (cr["z1"] - cr["z0"]), C["hole"], "pointed")
        BA.roof_fan(AR, (*c, Z(cr["h"])), P8 + [P8[0]], Z(cr["z1"]), BA.shade(C["roof"], 1.1))
    elif cr["kind"] == "lantern":                                                # Ely: a stone octagon with through windows and a wooden lantern
        c = L(u_mid, 0.0); BA.drum(AT, c, cr["r"], Z(cr["z0"]), cr["z1"] - cr["z0"], 8, C, t=M.tc * 0.8, n=8, win_kind=M.KIND if M.KIND != "flat" else "round")
        BA.drum(AT, c, cr["r2"], Z(cr["z1"]), cr["z2"] - cr["z1"], 8, {**C, "stone": BA.rc((0.42, 0.32, 0.22), "wood")}, t=0.4 * kq, n=8)
        P8 = BA.octagon(c, cr["r"] * 1.04, phase=0.0); AR.face([(*p, Z(cr["z1"])) for p in P8], C["roof"])
        P8b = BA.octagon(c, cr["r2"] * 1.05, phase=0.0); BA.roof_fan(AR, (*c, Z(cr["h"])), P8b + [P8b[0]], Z(cr["z2"]), C["roof"])
    elif cr["kind"] == "dome":                                                   # Florence: a drum with windows, a dome, a lantern
        c = L(u_mid, 0.0); BA.drum(AT, c, cr["r"], Z(cr["z0"]), cr["zd"] - cr["z0"], cr["windows"], C, t=M.tc, n=16, win_kind="round")
        BA.dome(AT, AR, c, cr["r"] * 1.02, Z(cr["zd"]), cr["dh"], cr["dome"], C, ribs=8 if cr["dome"] == "pointed" else 0)
    else: church_tower(cr)


# ---------------------------------------------------------------- special elements and additions
def extras():
    M = Mn; C = M.C
    ch = EX.get("chapels")
    if ch:                                                                       # chapels between the nave buttresses
        for j in ch["bays"]:
            ua, ub = j * M.B + M.buttress["w"] / 2, (j + 1) * M.B - M.buttress["w"] / 2
            for sgn in (1, -1):
                vi, vo = sgn * (W / 2 + A + M.t / 2), sgn * (W / 2 + A + M.t / 2 + ch["depth"]); hc = ch["h"]; holes = []; trac = []
                win(M, (ub - ua) / 2, 2.2 * kq, min(0.55 * (ub - ua), M.win["ai"]["w"]), hc * 0.62, holes, trac)
                panel(AW, (ua, vo), (ub, vo), ZB, hc, M.t * 0.7, holes, C["stone"], M, trac=trac)
                for uu in (ua, ub): panel(AW, (uu, vi), (uu, vo), ZB, hc, M.t * 0.6, [], C["stone"], M)   # the side walls (the neighboring support is sometimes removed by a tower)
                BA.rib_vault(AV, [L(ua, vo), L(ua, vi), L(ub, vi), L(ub, vo)], Z(hc * 0.62), hc * 0.26, C["vault"], C["rib"], kind="rib4" if M.vault != "flat" else "flat",
                             rib_w=0.18 * kq, a_rib=AP)
                AR.face([L3(ua - 0.3, vi, M.Ea - 0.3 * kq), L3(ub + 0.3, vi, M.Ea - 0.3 * kq), L3(ub + 0.3, vo + sgn * 0.5, hc + 0.2), L3(ua - 0.3, vo + sgn * 0.5, hc + 0.2)], C["roof"])
                AF.prism([L(ua, vi), L(ub, vi), L(ub, vo), L(ua, vo)], Z(ZB), Z(0.0), C["floor"])
    chh = EX.get("chapter_house")
    if chh:                                                                      # chapter house: a polygon with a column, a fan vault, a pyramidal roof
        c = L(chh["u"], chh["v"]); n = chh["n"]; r = chh["r"]; h = chh["h"]; sg = float(np.sign(chh["v"])); tw_ = -sg * PVV
        ph_ = math.atan2(tw_[1], tw_[0]) - math.pi / n; P = BA.octagon(c, r, phase=ph_, n=n)          # the face toward the cathedral — a door from the vestibule
        for i in range(n):
            p, q = P[i], P[(i + 1) % n]; Lw = float(np.linalg.norm(q - p)); mid = (p + q) / 2; nn = (mid - c) / max(np.linalg.norm(mid - c), 1e-9)
            holes = [(Lw / 2, 0.22 * h, 0.5 * Lw, 0.6 * h, "lancet" if M.KIND == "pointed" else M.KIND)] if nn @ (PVV * -sg) < 0.92 else [(Lw / 2, 0.0, 0.42 * Lw, 0.5 * h, M.KIND)]
            BA.wall_panel(AW, p, q, Z(ZB), Z(h), M.t * 0.8, [(s_, Z(z_), w_, hh, k_) for s_, z_, w_, hh, k_ in holes], C["stone"])
        cyl(AP, c, 0.08 * r + 0.3 * kq, Z(0.0), Z(0.62 * h), C["stone"], n=12)
        BA.radial_vault(AV, c, list(P) + [P[0]], Z(0.62 * h), 0.3 * h, C["vault"], C["rib"], rib_w=0.18 * kq, a_rib=AP)
        P2 = BA.octagon(c, r * 1.08 + M.t * 0.4, phase=ph_, n=n); BA.roof_fan(AR, (*c, Z(h + r * 1.1)), P2 + [P2[0]], Z(h), C["roof"])
        AF.prism(list(P), Z(ZB), Z(0.0), C["floor"])
        vw = chh["vest_w"]; va = sg * (v1 + Mt.t / 2); vb = chh["v"] - sg * r * 0.95; ux = chh["u"]
        for x in (-1, 1): panel(AW, (ux + x * vw / 2, va), (ux + x * vw / 2, vb), ZB, 0.5 * h, M.t * 0.6, [], C["stone"], M)
        AF.prism([L(ux - vw / 2, va), L(ux + vw / 2, va), L(ux + vw / 2, vb), L(ux - vw / 2, vb)], Z(ZB), Z(0.0), C["floor"])   # vestibule floor
        AR.face([L3(ux - vw / 2 - 0.4, va, 0.5 * h), L3(ux + vw / 2 + 0.4, va, 0.5 * h), L3(ux + vw / 2 + 0.4, vb, 0.5 * h), L3(ux - vw / 2 - 0.4, vb, 0.5 * h)], C["roof"])
    cl_ = EX.get("cloister")
    if cl_:                                                                      # cloister: four galleries with arcades around the court
        u0, u1, va, vb = cl_["u0"], cl_["u1"], cl_["v0"], cl_["v1"]; wk = cl_["walk"]; h = cl_["h"]; vlo, vhi = min(va, vb), max(va, vb)
        cen = np.array([(u0 + u1) / 2, (vlo + vhi) / 2])
        kd = M.KIND if M.KIND != "flat" else "round"
        for k_, (p, q) in enumerate((((u0, vlo), (u1, vlo)), ((u1, vlo), (u1, vhi)), ((u1, vhi), (u0, vhi)), ((u0, vhi), (u0, vlo)))):
            p = np.array(p, float); q = np.array(q, float); d = (q - p) / max(np.linalg.norm(q - p), 1e-9); nr = np.array([-d[1], d[0]])
            nin = nr if (cen - (p + q) / 2) @ nr > 0 else -nr
            pi, qi = p + nin * wk + d * wk, q + nin * wk - d * wk; Lw = float(np.linalg.norm(qi - pi)); nb_ = max(3, int(Lw / (2.6 * kq)))
            ho = []                                                              # doors per the scheme: from the west and from the transept
            if k_ == 3: ho.append((float(np.linalg.norm(q - p)) / 2, 0.0, 1.6 * kq, 3.0 * kq, kd))
            if k_ == 1 and cl_.get("door_v") is not None and TR: ho.append((cl_["door_v"] - vlo, 0.0, 1.6 * kq, 3.0 * kq, kd))
            panel(AW, tuple(p), tuple(q), ZB, h + 0.6 * kq, M.t * 0.6, ho, C["stone"], M)
            holes = [((i + 0.5) * Lw / nb_, 0.0 if (k_ == 0 and i == nb_ // 2) else 0.6 * kq, 0.62 * Lw / nb_, 0.62 * h + (0.6 * kq if (k_ == 0 and i == nb_ // 2) else 0.0), kd)
                     for i in range(nb_)]                                        # the middle arch of the first gallery goes down to the floor: a passage to the garden
            panel(AW, tuple(pi), tuple(qi), ZB, h * 0.8, M.t * 0.5, holes, C["stone"], M)
            AR.face([L3(*p, h + 0.6 * kq), L3(*q, h + 0.6 * kq), L3(*qi, h * 0.8 - 0.2), L3(*pi, h * 0.8 - 0.2)], C["roof"])
            AF.prism([L(*p), L(*q), L(*qi), L(*pi)], Z(ZB), Z(0.0), C["floor"])
    sa = EX.get("sacristy")
    if sa:                                                                       # a sacristy by the side wall of the choir
        sg = sa["side"]; vi = sg * (W / 2 + (A if CH_AIS else 0.0) + Me.t / 2); vo = vi + sg * sa["d"]; h = sa["h"]
        panel(AW, (sa["u0"], vo), (sa["u1"], vo), ZB, h, Me.t * 0.7, [((sa["u1"] - sa["u0"]) / 2, 0.45 * h, 1.0 * kq, 0.35 * h, Me.KIND)], Me.C["stone"], Me, T_=Lc)
        for uu in (sa["u0"], sa["u1"]): panel(AW, (uu, vi), (uu, vo), ZB, h, Me.t * 0.7, [], Me.C["stone"], Me, T_=Lc)
        AF.prism([Lc(sa["u0"], vi), Lc(sa["u1"], vi), Lc(sa["u1"], vo), Lc(sa["u0"], vo)], Z(ZB), Z(0.0), Me.C["floor"])   # sacristy floor
        AR.face([L3(sa["u0"] - 0.3, vi, h + 1.5 * kq, Lc), L3(sa["u1"] + 0.3, vi, h + 1.5 * kq, Lc), L3(sa["u1"] + 0.3, vo + sg * 0.4, h, Lc),
                 L3(sa["u0"] - 0.3, vo + sg * 0.4, h, Lc)], Me.C["roof"])
    for u_, v_, r_, zt in EX.get("turrets", []):                                 # stair turrets at the corners of the transept
        c = L(u_, v_); P8 = BA.octagon(c, r_); AT.prism(P8, Z(ZB), Z(zt), Mt.C["stone"])
        for zz in np.arange(3.0 * kq, zt - 2.0 * kq, 4.0 * kq):
            p = P8[int(zz * 7) % 8]; nn = (p - c) / max(np.linalg.norm(p - c), 1e-9); dark_open(AT, c + nn * r_ * 0.98, nn, Z(zz), 0.25 * kq, 1.2 * kq, Mt.C["hole"], "flat")
        BA.crown(AT, AR, c, P8, r_, "poly8", Z(zt), "flat" if FORT else ("spikes" if Mt.spikes >= 0.5 else "needle"), 2.6 * r_, Mt.C, fort=FORT, s=kq)
    cp = EX.get("campanile")
    if cp:
        church_tower(cp, doors=[{"side": "in", "at": (cp["u"], cp["v"] - math.copysign(cp["side"] / 2, cp["v"])), "w": 1.6 * kq, "h": 3.0 * kq,
                                 "kind": Mw.KIND if Mw.KIND != "flat" else "round"}])
        hs_ = cp["side"] / 2; AF.prism([L(cp["u"] - hs_, cp["v"] - hs_), L(cp["u"] + hs_, cp["v"] - hs_), L(cp["u"] + hs_, cp["v"] + hs_), L(cp["u"] - hs_, cp["v"] + hs_)],
                                        Z(ZB), Z(0.0), Mw.C["floor"])                     # campanile floor
    for tw in EX.get("transept_towers", []): church_tower(tw)
    bp = EX.get("baptistery")
    if bp:                                                                       # baptistery: an octagon with a dome on the axis in front of the facade (Pisa)
        c = L(bp["u"], bp["v"]); P = BA.octagon(c, bp["r"], phase=0.0); h = bp["h"]
        for i in range(8):
            p, q = P[i], P[(i + 1) % 8]; Lw = float(np.linalg.norm(q - p)); mid = (p + q) / 2; nn = (mid - c) / max(np.linalg.norm(mid - c), 1e-9)
            holes = [(Lw / 2, 0.0, 0.45 * Lw, 0.42 * h, Mw.KIND)] if nn @ AXV > 0.9 else [(Lw / 2, 0.45 * h, 0.3 * Lw, 0.35 * h, Mw.KIND)]
            BA.wall_panel(AW, p, q, Z(ZB), Z(h), Mw.t, [(s_, Z(z_), w_, hh, k_) for s_, z_, w_, hh, k_ in holes], Mw.C["stone"])
        BA.drum(AT, c, bp["r"] * 0.7, Z(h), 0.25 * h, 8, Mw.C, t=0.6 * kq, n=16)
        BA.dome(AT, AR, c, bp["r"] * 0.72, Z(h * 1.25), bp["dh"] * 0.75, "hemi", Mw.C)
        AR.face([(*p, Z(h)) for p in BA.octagon(c, bp["r"] * 1.03, phase=0.0)], Mw.C["roof"]); AF.prism(list(P), Z(ZB), Z(0.0), Mw.C["floor"])
    if Mn.industrial >= 0.4 or Me.industrial >= 0.4:                             # factory hive: pipes by the choir
        for i, sgn in enumerate((1, -1, 1, -1)):
            u_ = u_x1 + (0.3 + 0.2 * i) * (u_c1 - u_x1); c = Lc(u_, sgn * (W / 2 + A + Me.t / 2 + Me.buttress["depth"] + 3.0 * kq))
            BA.stack(AB, c, 0.04 * W + 0.5 * kq, Z(ZB), Z(Me.He * (1.1 + 0.15 * i)), Me.C["dark"], Me.C["accent"])


# ---------------------------------------------------------------- build and views
def build(sc=None, with_ground=True, fresh=True, marker=True):
    """Build a temple from the loaded plan (load). The command-line path — from a clean scene, with the ground and the human
    scale figure; step by step — into an open scene sc without them (the ground belongs to the site step, from the edited
    relief). → (uw, ue, the temple's objects)."""
    global ZB
    if fresh: bpy.ops.wm.read_factory_settings(use_empty=True)
    BA.SCENE[0] = sc
    global RLF
    R = RLF = BA.Relief(os.path.join(OUT, "relief.npz"))
    uw = float(FR["u_west"]); ue = FR["u_end"]; vlo, vhi = float(FR.get("v_lo", -FR["v_half"])), float(FR.get("v_hi", FR["v_half"]))
    UU, VV = np.meshgrid(np.arange(uw - 2, ue + 2, 2.0), np.arange(vlo - 2, vhi + 2, 2.0))
    zg = R.height(O[0] + AXV[0] * UU + PVV[0] * VV, O[1] + AXV[1] * UU + PVV[1] * VV); zmin = float(zg.min())
    ZB = min(-3.0, zmin - FL - 1.0)                                               # the foundation goes down to the ground (on a slope — an undercroft)
    ter = T.get("terrace") if "terrace" in T else ({"e": 3.0 * kq, "z": Z0 + 0.3, "zmin": zmin} if (not T.get("podium") and Z0 - zmin > 1.5) else None)
    if ter:                                                                      # an undercroft terrace under the temple on a slope (from the plan)
        e_ = ter["e"]; P = [L(uw - e_, vlo - e_), L(ue + e_, vlo - e_), L(ue + e_, vhi + e_), L(uw - e_, vhi + e_)]
        AF.prism(P, min(zmin, ter["zmin"]) - 1.0, ter["z"], BA.rc(BA.shade(Mn.C["stone"], 0.88), "wall"))
        for t in (T.get("spaces") or {}).get("trans", []):                        # stairs from the platform to the ground — per the scheme
            if t.get("build") != "steps": continue
            foot, top = np.asarray(t["p"][0], float), np.asarray(t["p"][-1], float); d = top - foot; Ls = float(np.linalg.norm(d)); n_ = max(1, round((t["zb"] - t["za"]) / 0.17))
            BA.steps(AF, top, d / max(Ls, 1e-9), t["w"], t["za"], t["zb"], Mn.C["dark"], rise=(t["zb"] - t["za"]) / n_, tread=Ls / n_)
    um = (uw + ue) / 2; c = L(um, (vlo + vhi) / 2); half = max(ue - uw, vhi - vlo) / 2 + 60 * kq
    if T.get("podium"): half += T["podium"]["margin"] * T["podium"]["levels"] + T["podium"]["run"]
    step = 1.0 if half < 170 else (2.0 if half < 340 else 4.0)
    if with_ground: BA.ground(R, c, half, step); BA.ground(R, c, half * 3.2, 8.0 * max(1.0, step / 2), hole=half - 4, name="ground_far")
    podium(); floor()
    body(Mn, 0.0, PT["nave"]["bays"], True, "nave"); body(Me, u_x1, PT["choir"]["bays"], CH_AIS, "choir")
    if not X_ and not CH_AIS:                                                    # hall church without a transept: the ends of the aisles at the choir
        for sgn in (1, -1):
            hl = []; tr_ = []; win(Mn, A / 2 + Mn.t / 4, Mn.win["ai"]["z0"], min(Mn.win["ai"]["w"], 0.5 * A), Mn.win["ai"]["h"] * 0.9, hl, tr_)
            panel(AW, (u_n1, sgn * W / 2), (u_n1, sgn * (W / 2 + A + Mn.t / 2)), ZB, Mn.He, Mn.t, hl, Mn.C["stone"], Mn, trac=tr_)
    crossing_and_transept(); east_end(); west(); buttresses(); roofs(); crossing_top(); extras()
    if marker: man(L(u_n1 * 0.55, W * 0.12), FL)
    obs = [a.obj(name, BA.coll("temple"), smooth=name == "vaults") for a, name in ((AW, "walls"), (AP, "piers"), (AV, "vaults"), (AR, "roofs"), (AT, "towers"),
                                                                                (AB, "buttresses"), (AXT, "tracery"), (AF, "floor")) if a.f]
    if AM.f and marker: AM.obj("man", BA.coll("marker"))                         # the human-scale figure only in the command path (views, sheets)
    BA.SCENE[0] = None
    return uw, ue, obs


def build_temple(folder, sc=None, tag=True, kw=None):
    """A temple as a function over an open scene (step-by-step build): load(folder) + build(sc) without the ground and the human
    scale figure + face labeling by the space scheme (bl_space.tag) and a ray check of the model (bl_space.verify).
    → (objects, plan); the result is SP_LAST."""
    global SP_LAST
    load(folder, kw)
    uw, ue, obs = build(sc, with_ground=False, fresh=False, marker=False); SP_LAST = None
    if tag and T.get("spaces"):
        S = Space.from_dict(T["spaces"]); SP_LAST = {"faces": BS.tag(S, obs), "check": BS.verify(S)}
    return obs, T


def cameras(uw, ue):
    sc = bpy.context.scene; cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)
    con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis, con.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    R = BA.Relief(os.path.join(OUT, "relief.npz"))
    def gz(p): return float(R.height(p[0], p[1]))
    hmax = max([tw["h"] for tw in T["towers"]] + [T["crossing"]["h"] if T.get("crossing") else 0.0] + [M.ridge for M in MO.values()]
               + [tw["h"] for tw in EX.get("transept_towers", [])] + ([EX["campanile"]["h"]] if EX.get("campanile") else []))
    vlo, vhi = FR.get("v_lo", -FR["v_half"]), FR.get("v_hi", FR["v_half"]); vm = (vlo + vhi) / 2
    um = (uw + ue) / 2; Lt = ue - uw; vh = max(-vlo, vhi); Dd = 1.55 * max(Lt, 2 * vh, 0.9 * hmax) + (FL - Z0)
    views = {}
    p = L(um - 0.62 * Dd, -0.8 * Dd); views["aerial"] = dict(loc=(*p, Z0 + 0.5 * Dd), tgt=(*L(um, vm), FL + 0.22 * hmax), lens=35)
    hw = max([tw["h"] for tw in T["towers"] if tw["role"] == "west"] + [Mw.ridge, (T["screen"]["turret_h"] if T.get("screen") else 0.0)]) + (FL - Z0)
    u_f = uw - (T["podium"]["margin"] * T["podium"]["levels"] + T["podium"]["run"] if T.get("podium") else 0.0)
    dist = (1.15 if not T.get("podium") else 0.8) * hw; p = L(u_f - dist, -0.1 * dist); zc = gz(p) + 1.7
    views["facade"] = dict(loc=(*p, zc), tgt=(*L(u_f, 0.0), zc), lens=24, shift=0.27)
    ds = 1.1 * max(Lt, 1.55 * (hmax + FL - Z0)); p = L(um, vlo - ds); views["side"] = dict(loc=(*p, gz(p) + 0.08 * ds), tgt=(*L(um, vm), FL + 0.36 * hmax), lens=35)
    p = L(ue + 0.55 * Dd, 0.5 * Dd); views["east"] = dict(loc=(*p, Z0 + 0.3 * Dd), tgt=(*L(um + 0.2 * (ue - um), vm), FL + 0.22 * hmax), lens=35)
    ui = max(0.45 * Mn.B, 2.5 * kq) + (Mn.t / 2)
    views["interior"] = dict(loc=L3(ui, 0.0, 1.7), tgt=L3(ue, 0.0, 1.7, Lc), lens=14 if kq < 1.5 else 12, shift=0.24, inside=True)
    if X_: views["cross"] = dict(loc=L3(u_n1 - 1.6 * Mn.B, -W / 2 + 1.6 * kq, 1.7), tgt=L3(u_x1 + 0.4 * Me.B, 0.15 * W, Mt.spring * 0.95), lens=16, inside=True)
    if any(b["kind"] == "flyer" for b in T["buttresses"]):
        views["detail"] = dict(loc=L3(u_n1 - 0.3 * Mn.B, -(W / 2 + 0.45 * A), Mn.z_lean + 0.35 * (Mn.spring - Mn.z_lean)),
                               tgt=L3(0.5 * Mn.B, -(W / 2 + 0.65 * A), Mn.spring + 0.2 * Mn.rise), lens=18)
    views["plan"] = dict(loc=(*L(um, vm), FL + hmax + 300), tgt=(*L(um, vm), FL), ortho=max(Lt, vhi - vlo) * 1.12)
    if KW.get("cam"):                                                            # custom view: cam=x,y,z;x,y,z (world) [lens=]
        a_, b_ = KW["cam"].split(";"); views["cam"] = dict(loc=tuple(map(float, a_.split(","))), tgt=tuple(map(float, b_.split(","))), lens=float(KW.get("lens", 24)))
    Dk = 1.0 * max(Lt, 2 * vh); p = L(um - 0.3 * Dk, vm - 0.62 * Dk); views["cutaway"] = dict(loc=(*p, FL + 0.8 * Dk), tgt=(*L(um, vm), FL), lens=30, side=True,
                                                                  cut=FL + min(0.45 * Mn.Ha, 7.0 * kq))   # section: inside — warm, outside — blue
    return cam, tgt, views


def render(cam, tgt, views, which, res, obs=()):
    sc = bpy.context.scene; sc.render.engine = "BLENDER_WORKBENCH"; sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"; shd = sc.display.shading
    shd.light = "STUDIO"; shd.color_type = "VERTEX"; shd.show_cavity = True; shd.cavity_type = "WORLD"
    shd.shadow_intensity = 0.55; sc.display.light_direction = (0.45, -0.35, 0.82)
    sc.world = sc.world or bpy.data.worlds.new("w"); sc.world.color = (0.62, 0.70, 0.80)
    for o in bpy.data.objects:
        if o.name in KW.get("hide", "").split(","): o.hide_render = True
    out = {}
    for k in which:
        if k not in views: continue
        v = views[k]; cam.location = v["loc"]; tgt.location = v["tgt"]
        cam.data.type = "ORTHO" if "ortho" in v else "PERSP"
        if "ortho" in v: cam.data.ortho_scale = v["ortho"]
        else: cam.data.lens = v["lens"]
        cam.data.shift_y = v.get("shift", 0.0); cam.data.clip_start = v.get("clip", 0.3); cam.data.clip_end = 9000
        shd.show_shadows = not (v.get("inside") or v.get("side")); BS.show("Side" if v.get("side") else "Col"); back = None
        if v.get("cut") and obs: _, back = BS.cutaway(obs, v["cut"]); BS.show("Side")
        sc.render.filepath = os.path.join(OUT, f"view_{k}.png"); bpy.ops.render.render(write_still=True); out[k] = sc.render.filepath
        if back: back()
    BS.show("Col")
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    load(argv[0], dict(a.split("=", 1) for a in argv[1:] if "=" in a))
    uw, ue, obs = build(); t_build = time.time() - T0
    SP = None
    if T.get("spaces"):                                                          # faces — cell and side by the scheme; model against the scheme
        S = Space.from_dict(T["spaces"]); SP = {"faces": BS.tag(S, obs)}; SP["check"] = BS.verify(S)
    cam, tgt, views = cameras(uw, ue)
    res = tuple(int(v) for v in KW.get("res", "1280x800").split("x"))
    out = render(cam, tgt, views, KW.get("views", "aerial,facade,side,east,interior,cross,detail,plan,cutaway").split(","), res, obs if SP else ())
    dg = bpy.context.evaluated_depsgraph_get()
    tris = sum(len(o.evaluated_get(dg).to_mesh().loop_triangles) for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("ground"))
    if KW.get("blend", "0") == "1": bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "temple.blend"))
    print("__RESULT__", json.dumps({"build_s": round(t_build, 2), "total_s": round(time.time() - T0, 1), "n_triangles_cathedral": tris, "kinds": list(out),
                                    "n_openings_skipped": SKIP[0], "space": SP}, ensure_ascii=False))


if __name__ == "__main__":
    main()
