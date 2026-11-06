"""The space of a cathedral from its plan (the scheme is space/space.py): cells — the nave, the side aisles, the crossing,
the transept arms, the choir, the choir aisles, the ambulatory, chapels (radiating, side, Lady), a porch, the base and
tiers of the towers, tower platforms, attics, additions (a chapter house with a vestibule, a cloister with a court, a
sacristy, a baptistery, a campanile), podium terraces; transitions — portals, arcades, openings, stairs in the thickness
of tower walls; anchors for the scene. The geometry is the same as the build's (temples/bl_temple.py: the plan frame, the
choir in its own rotated frame, cells up to the wall axes); the elevations are world ones.
Called by temples/temple_plan.py (temple.json → spaces); standalone: .venv/bin/python -m components.buildings.temples.temple_space <plan folder> [draw=1]"""
import json, math, os, sys

import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.space.space import Space                                                         # noqa: E402

STAIR = math.tan(math.radians(38.0))                                             # a stair in the wall thickness: a step of 0.17 × 0.22


def hemi(uc, r, n):
    """A semicircle around (uc, 0) from −90° to +90°, n segments (like bl_temple.hemi)."""
    return [(uc + r * math.cos(-math.pi / 2 + math.pi * i / n), r * math.sin(-math.pi / 2 + math.pi * i / n)) for i in range(n + 1)]


def arc_w(M, span):
    """The width of an arcade opening (like bl_temple.arc_w)."""
    if M["pier_kind"] == "square": return span - M["pier"]
    rp = M["pier"] / 2; return span - 2 * math.sqrt(max(rp * rp - (M["tc"] / 2) ** 2, 0.04)) * 0.95


def octagon(c, r, phase=0.0, n=8):
    """A polygon with apothem r (like bl_arch.octagon), world."""
    rr = r / math.cos(math.pi / n); return [np.asarray(c, float) + rr * np.array([math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n)]) for i in range(n)]


def spaces(R, gz=None):
    FR = R["frame"]; O = np.asarray(FR["origin"], float); AX = np.asarray(FR["ax"], float); PV = np.asarray(FR["pv"], float)
    Z0 = float(FR["z0"]); FL = Z0 + float(FR["floor"]); PT = R["parts"]; MO = R["modules"]; EX = R.get("extras") or {}
    STY = R.get("story") or {}; STATE = STY.get("state") or {}; UNF = STY.get("unfinished") or {}; FORT = bool(R.get("fort"))
    def mod(part): return MO[(PT.get(part) or {}).get("mod") or PT["nave"]["mod"]]
    Mn = mod("nave"); Mt = mod("transept") if PT.get("transept") else (mod("crossing") if PT.get("crossing") else Mn); Me = mod("choir")
    W, A, kq, t = Mn["W"], Mn["A"], Mn["kq"], Mn["t"]
    u_n1 = PT["nave"]["u1"]; X_ = PT.get("crossing"); u_x1 = X_["u1"] if X_ else u_n1; u_c1 = PT["choir"]["u1"]; CH = PT["choir"]["aisles"]
    TR = PT.get("transept"); v1 = TR["v1"] if TR else 0.0; E = PT["east"]; ek = E["kind"]; BEND = float(PT["choir"].get("bend", 0.0))
    u_mid = (u_n1 + u_x1) / 2; pod = R.get("podium"); ruin = bool(STATE.get("nave_roofless"))
    S = Space(frame={"origin": O.round(3).tolist(), "ax": AX.round(5).tolist(), "floor_z": round(FL, 2), "kind": "temple"})

    def L(u, v): return O + AX * u + PV * v
    def Lc(u, v):
        if not BEND or u <= u_x1 + 1e-9: return L(u, v)
        f = min(1.0, (u - u_x1) / max(Me["B"], 1.0)); a_ = math.radians(BEND) * f; du = u - u_x1
        return L(u_x1 + du * math.cos(a_) - v * math.sin(a_), du * math.sin(a_) + v * math.cos(a_))
    def rect(u0, u1, v0, v1, T_=L, n=1):
        v0, v1 = min(v0, v1), max(v0, v1); us = np.linspace(u0, u1, n + 1); return [T_(u, v0) for u in us] + [T_(u, v1) for u in us[::-1]]
    def nrm(T_, u, v, du, dv):
        d = T_(u + du * 0.5, v + dv * 0.5) - T_(u - du * 0.5, v - dv * 0.5); return d / max(np.linalg.norm(d), 1e-9)
    OUT, z_out = "outside", Z0; ter = R.get("terrace"); feet = []
    def descend(p, n_out, w):
        """From the platform to the ground in front of an entrance: at the platform edge — a stair down (a drop above 0.6 m) or a
        threshold."""
        q = np.asarray(p, float) - O; pu, pv_ = q @ AX, q @ PV; nu, nv = float(np.asarray(n_out) @ AX), float(np.asarray(n_out) @ PV); e = ter["e"]
        ts = [((FR["u_west"] - e) - pu) / nu if nu < -1e-6 else np.inf, ((FR["u_end"] + e) - pu) / nu if nu > 1e-6 else np.inf,
              ((FR["v_lo"] - e) - pv_) / nv if nv < -1e-6 else np.inf, ((FR["v_hi"] + e) - pv_) / nv if nv > 1e-6 else np.inf]
        pe = np.asarray(p, float) + np.asarray(n_out) * max(0.0, min(ts))
        if any(np.linalg.norm(pe - f) < max(8.0, w) for f in feet): return
        feet.append(pe); zt = ter["z"]; g = float(gz(*(pe + np.asarray(n_out) * 1.0)))
        if zt - g <= 0.6: S.link("outside", "terrace", "opening", pe, g, zt, w, n=-np.asarray(n_out), ground=g, why="platform edge — a threshold"); return
        for _ in range(2): Ls = max(zt - g, 0.2) / math.tan(math.radians(33.0)); foot = pe + np.asarray(n_out) * Ls; g = float(gz(*foot))
        S.link("outside", "terrace", "stair", [foot, pe], g, zt, w, ground=float(gz(*(foot + np.asarray(n_out) * 0.5))), n=-np.asarray(n_out),
               why="stair from the platform down to the ground in front of the entrance", build="steps")
    def door_out(cell, p, n_out, w, h, kind="door", why=""):
        """Entrance from outside (from the ground, from a platform on a slope, or from the upper terrace of a podium): p is on the wall axis, n_out points outward."""
        g = None
        if gz is not None and OUT == "outside": q = np.asarray(p) + np.asarray(n_out) * 3.0; g = float(gz(q[0], q[1]))
        S.link(OUT, cell, kind, p, z_out, FL, w, h, n=-np.asarray(n_out), ground=g, why=why)
        if OUT == "terrace" and gz is not None: descend(p, n_out, max(w * 1.4, 3.0 * kq))
    def chord(u, v, u0, B, T_):
        """Wall point of a bay: the build puts the wall straight between the ends of the bay (the choir, in the rotated frame)."""
        j = int((u - u0) // B); ua = u0 + j * B; a_, b_ = T_(ua, v), T_(ua + B, v); return a_ + (b_ - a_) * (u - ua) / B
    def arcade(a, b, u0, nb_, B, ab, v, M, T_=L):
        """Arcade between a and b along u on line v: one opening per span of the side aisle (for a hall church, one per bay)."""
        S.relate(a, b, "arcade"); sg = 1.0 if v > 0 else -1.0
        if M["hall"]:
            for j in range(nb_): S.link(a, b, "arch", chord(u0 + (j + 0.5) * B, v, u0, B, T_), FL, FL, B - 0.8 * M["pier"], min(M["spring"], M["spring_a"]), n=PV * sg)
        else:
            for j in range(nb_ * ab): S.link(a, b, "arch", chord(u0 + (j + 0.5) * B / ab, v, u0, B, T_), FL, FL, arc_w(M, B / ab), M["z_arc"], n=PV * sg)
    def tower(name, tw, base, p0, z_base, base_void, T_=L):
        """Tower by tiers: a shaft above the vault (a void) or its own base; the belfry tiers are walkable (the floor is the belt course of the
        tier below), a platform if the top is flat; a stair in the wall thickness from base (point p0)."""
        c = T_(tw["u"], tw["v"]); st = tw["stages"]; tops = []
        def foot(half, plan):
            if plan == "square": return [c + AX * x * half + PV * y * half for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            return octagon(c, half, phase=math.atan2(AX[1], AX[0]) + math.pi / 8)
        if base_void: S.cell(f"{name}_shaft", "tower shaft", foot(st[0]["half"], "square"), z_base, FL + st[0]["z1"], "void", walk=False, why="above the vault — masonry and shaft")
        prev, zp, pp = base, FL, p0
        for k in range(1, len(st)):
            sg_ = st[k]; cid = f"{name}_b{k}"; z0_ = FL + sg_["z0"]
            S.cell(cid, "belfry tier", foot(sg_["half"], sg_["plan"]), z0_, FL + sg_["z1"], "covered", use="bells", access="private")
            S.link(prev, cid, "spiral", [pp, c], zp, z0_, 0.8 * kq, length=(z0_ - zp) / STAIR, hidden=True, why="spiral stair in the wall thickness")
            prev, zp, pp = cid, z0_, c
        if tw["spire"]["kind"] == "flat" and not tw.get("unfinished"):
            cid = f"{name}_top"; z0_ = FL + st[-1]["z1"]
            S.cell(cid, "tower platform", foot(st[-1]["half"], st[-1]["plan"]), z0_, None, "exterior", edge="parapet", use="overview", access="defense" if FORT else "private")
            S.link(prev, cid, "spiral", [pp, c], zp, z0_, 0.8 * kq, length=(z0_ - zp) / STAIR, hidden=True, why="a crawlway to the platform")
            tops.append((cid, c, z0_))
        return tops

    # ---- podium (hive): terraces in tiers, a flight to each on the axis from the west
    if pod:
        nlev, h, mg, run, sw = pod["levels"], pod["h"], pod["margin"], pod["run"], pod["stair_w"]
        uw = min([-t / 2 - 0.4 * kq] + [tw["u"] - tw["side"] / 2 for tw in R["towers"]]) - 2.0 * kq; ue = FR["u_end"] + 2.0 * kq; vh = FR["v_half"] + 2.0 * kq
        prev, zp = "outside", Z0
        for i in range(nlev):
            e = mg * (nlev - i); zt = Z0 + (i + 1) * h; cid = f"podium{i}"
            S.cell(cid, "podium terrace", rect(uw - e, ue + e, -vh - e, vh + e), zt, None, "exterior", edge="parapet", use="terrace",
                   holes=[rect(uw - e + mg, ue + e - mg, -vh - e + mg, vh + e - mg)] if i < nlev - 1 else None)   # the tier above is a mass on the terrace
            a0 = L(uw - e - run, 0.0); g = None
            if gz is not None and i == 0:                                        # the first flight goes down to the ground (below the platform — longer)
                zp = min(Z0, float(gz(*(a0 - AX * 1.0)))); a0 = L(uw - e - run * (zt - zp) / h, 0.0); g = float(gz(*(a0 - AX * 0.5)))
            S.link(prev, cid, "stair", [a0, L(uw - e, 0.0)], zp, zt, sw, ground=g, why="podium flight on the axis from the west")
            prev, zp = cid, zt
        OUT, z_out = prev, zp
    elif ter:                                                                    # a platform on a slope: a terrace with a drop, with stairs to the entrances
        e = ter["e"]; OUT, z_out = "terrace", ter["z"]
        S.cell("terrace", "cathedral platform", rect(FR["u_west"] - e, FR["u_end"] + e, FR["v_lo"] - e, FR["v_hi"] + e), ter["z"], None, "exterior", edge="edge", use="parvis",
               why=f"undercroft on a slope: a drop of up to {ter['z'] - ter['zmin']:.0f} m")
    # ---- nave and aisles
    wt = [tw for tw in R["towers"] if tw["role"] == "west"]
    S.cell("nave", "nave", rect(0.0, u_n1, -W / 2, W / 2), FL, None if ruin else FL + Mn["H"], "court" if ruin else "interior", use="assembly",
           why="ruin: no vault and no roof — under the open sky" if ruin else "")
    for sg, nm in ((1, "n"), (-1, "s")):
        S.cell(f"aisle_{nm}", "side aisle", rect(0.0, u_n1, sg * W / 2, sg * (W / 2 + A)), FL, FL + Mn["Ha"], use="passage",
               why="the first bay is under the facade tower (the entrance is through the tower face)" if len(wt) == 2 else "")
        arcade("nave", f"aisle_{nm}", 0.0, PT["nave"]["bays"], Mn["B"], Mn["ab"], sg * W / 2, Mn)
        if not Mn["hall"]: S.cell(f"attic_aisle_{nm}", "side-aisle attic", rect(0.0, u_n1, sg * W / 2, sg * (W / 2 + A)), FL + Mn["Ha"], FL + Mn["z_lean"], "void", walk=False)
    if not ruin:
        hw = W / 2 + A if Mn["hall"] else W / 2
        S.cell("attic_nave", "attic", rect(0.0, u_n1, -hw, hw), FL + Mn["H"], FL + Mn["ridge"], "void", walk=False)
    ch = EX.get("chapels")
    if ch:                                                                       # chapels between the nave buttresses
        bw = Mn["buttress"]["w"]; bay_a = Mn["B"] / Mn["ab"]
        for j in ch["bays"]:
            ua, ub = j * Mn["B"] + bw / 2, (j + 1) * Mn["B"] - bw / 2
            for sg, nm in ((1, "n"), (-1, "s")):
                cid = f"chapel_{nm}{j}"; vi = sg * (W / 2 + A)
                S.cell(cid, "chapel", rect(ua, ub, vi, sg * (W / 2 + A + t / 2 + ch["depth"])), FL, FL + ch["h"] * 0.88, use="chapel", access="sacred")
                for m in range(Mn["ab"]):
                    S.link(f"aisle_{nm}", cid, "arch", L(j * Mn["B"] + (m + 0.5) * bay_a, vi), FL, FL, min(bay_a - bw - 0.6 * kq, bay_a * 0.8), ch["h"] * 0.82, n=PV * sg)
    # ---- crossing and transept
    cr = R.get("crossing") or {}; ck = cr.get("kind")
    if X_:
        z1x = FL + (cr["z2"] if ck == "lantern" else (cr["zd"] + cr["dh"] if ck == "dome" else Mt["H"]))
        S.cell("crossing", "crossing", rect(u_n1, u_x1, -W / 2, W / 2), FL, z1x, use="assembly",
               why={"lantern": "open upward into the lantern", "dome": "under the dome"}.get(ck, ""))
        S.relate("nave", "crossing", "open"); S.link("nave", "crossing", "opening", L(u_n1, 0.0), FL, FL, W - 1.45 * Mt["pier"], Mt["H"], n=AX)
        if ck not in ("lantern", "dome"): S.cell("attic_crossing", "attic", rect(u_n1, u_x1, -W / 2, W / 2), FL + Mt["H"], FL + Mt["ridge"], "void", walk=False)
        if cr.get("stages"):
            st = cr["stages"]; c = L(u_mid, 0.0); hs = cr["side"] / 2
            S.cell("crossing_tower", "crossing tower", [c + AX * x * hs + PV * y * hs for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
                   FL + st[0]["z0"], FL + st[-1]["z1"], "void", walk=False)
    if TR:
        for sg, nm in ((1, "n"), (-1, "s")):
            cid = f"transept_{nm}"
            S.cell(cid, "transept arm", rect(u_n1, u_x1, sg * W / 2, sg * v1), FL, FL + Mt["H"], use="assembly")
            S.cell(f"attic_transept_{nm}", "attic", rect(u_n1, u_x1, sg * W / 2, sg * v1), FL + Mt["H"], FL + Mt["ridge"], "void", walk=False)
            S.relate("crossing", cid, "open"); S.link("crossing", cid, "opening", L(u_mid, sg * W / 2), FL, FL, (u_x1 - u_n1) - 1.45 * Mt["pier"], Mt["H"], n=PV * sg)
            if not Mt["hall"]:
                S.link(f"aisle_{nm}", cid, "arch", L(u_n1, sg * (W / 2 + A / 2)), FL, FL, arc_w(Mt, A), Mt["z_arc"], n=AX)
                if CH: S.link(cid, f"choir_aisle_{nm}", "arch", L(u_x1, sg * (W / 2 + A / 2)), FL, FL, arc_w(Mt, A), Mt["z_arc"], n=AX)
            chh = EX.get("chapter_house"); pe = L(u_mid, sg * v1); pw_, ph_ = R["portal"]["w"] * 0.8, R["portal"]["h"] * 0.85
            if chh and np.sign(chh["v"]) == sg: S.link(cid, "chapter_vest", "door", pe, FL, FL, pw_, ph_, n=PV * sg, why="the end portal — into the chapter-house vestibule")
            else: door_out(cid, pe, PV * sg, pw_, ph_, why="transept end portal")
    # ---- choir and east end
    if ek in ("chevet", "apse", "polygon"):
        H1 = hemi(u_c1, W / 2, E["n"])
        poly = [Lc(u, -W / 2) for u in np.linspace(u_x1, u_c1, 5)] + [Lc(*p) for p in H1[1:-1]] + [Lc(u, W / 2) for u in np.linspace(u_c1, u_x1, 5)]
    else: poly = rect(u_x1, u_c1, -W / 2, W / 2, Lc, 4)
    S.cell("choir", "choir", poly, FL, FL + Me["H"], use="altar", access="sacred", why={"apse": "with an apse", "polygon": "with a polygonal apse",
                                                                                     "chevet": "with a semicircular arcade"}.get(ek, "flat east end"))
    S.cell("attic_choir", "attic", poly, FL + Me["H"], FL + Me["ridge"], "void", walk=False)
    if X_: S.relate("crossing", "choir", "open"); S.link("crossing", "choir", "opening", L(u_x1, 0.0), FL, FL, W - 1.45 * Mt["pier"], Me["H"], n=AX)
    else: S.relate("nave", "choir", "open"); S.link("nave", "choir", "opening", L(u_n1, 0.0), FL, FL, W - Mn["pier"], min(Mn["H"], Me["H"]), n=AX)
    if CH:
        for sg, nm in ((1, "n"), (-1, "s")):
            cid = f"choir_aisle_{nm}"
            S.cell(cid, "choir aisle", rect(u_x1, u_c1, sg * W / 2, sg * (W / 2 + A), Lc, 4), FL, FL + Me["Ha"], use="detour")
            arcade("choir", cid, u_x1, PT["choir"]["bays"], Me["B"], Me["ab"], sg * W / 2, Me, Lc)
            if not Me["hall"]: S.cell(f"attic_choir_aisle_{nm}", "side-aisle attic", rect(u_x1, u_c1, sg * W / 2, sg * (W / 2 + A), Lc, 4), FL + Me["Ha"], FL + Me["z_lean"], "void", walk=False)
            if not X_: S.relate(f"aisle_{nm}", cid, "open"); S.link(f"aisle_{nm}", cid, "opening", L(u_n1, sg * (W / 2 + A / 2)), FL, FL, A - Mn["pier"], Mn["Ha"], n=AX)
    if ek == "chevet":                                                           # ambulatory, chapels in a fan
        n_e = E["n"]; r2_ = E["r2"]; H1 = hemi(u_c1, W / 2, n_e); H2 = hemi(u_c1, r2_, n_e)
        amb = [Lc(*p) for p in H2] + [Lc(*p) for p in H1[::-1]]
        S.cell("ambulatory", "detour", amb, FL, FL + Me["Ha"], use="detour")
        S.cell("attic_ambulatory", "ambulatory attic", amb, FL + Me["Ha"], FL + Me["z_lean"], "void", walk=False)
        S.relate("choir", "ambulatory", "arcade")
        for i in range(n_e):
            ch_ = math.dist(H1[i], H1[i + 1]); m_ = ((H1[i][0] + H1[i + 1][0]) / 2, (H1[i][1] + H1[i + 1][1]) / 2)
            S.link("choir", "ambulatory", "arch", Lc(*m_), FL, FL, max(1.0, arc_w(Me, ch_)), Me["z_arc"], n=Lc(*m_) - Lc(u_c1, 0.0))
        if CH:
            for sg, nm in ((1, "n"), (-1, "s")):
                S.relate(f"choir_aisle_{nm}", "ambulatory", "open")
                S.link(f"choir_aisle_{nm}", "ambulatory", "opening", Lc(u_c1, sg * (W / 2 + A / 2)), FL, FL, A - 0.5 * Me["pier"], Me["Ha"], n=nrm(Lc, u_c1, 0.0, 1.0, 0.0))
        nch = E.get("chapels", 0); rc = E.get("rc", 0.0); s_ = E.get("straight", 0.0)
        crown_c = Me["Ea"] * 0.92 - 0.06 * rc - 0.6 * kq
        for i in range(n_e if nch else 0):
            a_m = -math.pi / 2 + math.pi * (i + 0.5) / n_e; dv = np.array([math.cos(a_m), math.sin(a_m)]); tg = np.array([-dv[1], dv[0]])
            ce = np.array([u_c1, 0.0]); co = ce + dv * r2_ * math.cos(math.pi / (2 * n_e)); cc = ce + dv * (r2_ + s_ + Me["t"] / 2)
            Q = [co - tg * rc, cc - tg * rc] + [cc + rc * np.array([math.cos(a_m + f), math.sin(a_m + f)]) for f in (-math.pi / 4, 0.0, math.pi / 4)] + [cc + tg * rc, co + tg * rc]
            cid = f"chapel_e{i}"; ch2 = math.dist(H2[i], H2[i + 1])
            S.cell(cid, "radiating chapel", [Lc(*q) for q in Q], FL, FL + crown_c, use="chapel", access="sacred")
            S.link("ambulatory", cid, "arch", Lc(*co), FL, FL, min(2 * rc - 0.3 * kq, ch2 - 0.6 * kq), min(Me["z_arc"] * 0.92, crown_c - 0.3 * kq), n=Lc(*co) - Lc(u_c1, 0.0))
    elif ek == "flat" and E.get("lady"):
        lw, lh = E["lady_w"], E["lady_h"]
        S.cell("lady_chapel", "Lady chapel", rect(u_c1, u_c1 + E["lady"], -lw / 2, lw / 2, Lc, 2), FL, FL + lh - 0.06 * lw - 0.6 * kq, use="chapel", access="sacred")
        S.relate("choir", "lady_chapel", "wall"); S.link("choir", "lady_chapel", "arch", Lc(u_c1, 0.0), FL, FL, lw - Me["t"], lh - 2.0 * kq, n=nrm(Lc, u_c1, 0.0, 1.0, 0.0))
    # ---- west end: portal, porch, tower base, towers
    po = R["portal"]; pw, ph = po["w"], po["h"]; tw1 = wt[0] if len(wt) == 1 else None; west_out = -AX
    pr = EX.get("porch"); main_in = "nave"
    if tw1:                                                                      # one tower in front of the facade: the base is a vestibule
        hs = tw1["side"] / 2; uf = tw1["u"] - hs
        S.cell("tower_base", "tower base", rect(uf, 0.0, -hs, hs), FL, FL + tw1["stages"][0]["z1"], use="vestibule",
               why="a tower in front of the facade (Ulm): the entrance goes through it")
        S.link("tower_base", "nave", "door", L(0.0, 0.0), FL, FL, pw * 0.8, ph * 0.9, n=AX, why="west wall portal")
        main_in, p_main = "tower_base", L(uf, 0.0)
    else: p_main = L(0.0, 0.0)
    if UNF.get("kind") == "nave": mw, mh = 2.6 * kq, 4.0 * kq                     # Beauvais: a temporary wall with a door
    elif R.get("screen") or len(wt) == 2: mw, mh = pw, ph
    else: mw, mh = (pw, ph) if tw1 else (pw * 0.8, ph * 0.9)
    if pr:
        u1p = pr.get("u1", -t / 2); u0p = pr.get("u0", u1p - pr["d"]); ua_ = uf if tw1 else 0.0
        S.cell("porch", "narthex", rect(u0p, ua_, -pr["w"] / 2, pr["w"] / 2), FL, FL + pr["h"] * 0.9, "covered", use="vestibule", why="galilee in front of the portal")
        S.link("porch", main_in, "door", p_main, FL, FL, mw, mh, n=AX, why="main portal")
        door_out("porch", L(u0p, 0.0), west_out, pw * 1.15, ph * 1.05, kind="arch", why="porch arch")
        for sg in (1, -1): S.window("porch", OUT, L((u0p + u1p) / 2, sg * pr["w"] / 2), FL + 1.0 * kq, 0.5 * pr["d"], 0.6 * pr["h"])
        p_front = L(u0p, 0.0)
    else:
        door_out(main_in, p_main, west_out, mw, mh, why="main portal"); p_front = p_main
    if R.get("screen") and UNF.get("kind") != "nave":                            # screen: side portals into the side aisles
        for sg, nm in ((1, "n"), (-1, "s")): door_out(f"aisle_{nm}", L(0.0, sg * (W / 2 + A / 2)), west_out, pw * 0.6, ph * 0.75, why="screen side portal")
    tops = []
    for i, tw in enumerate(R["towers"]):
        sg = 1.0 if tw["v"] > 0 else -1.0; nm = "n" if sg > 0 else "s"
        if tw["role"] == "west" and len(wt) == 2:                                # above the first side-aisle bay
            if tw.get("portal") and UNF.get("kind") != "nave":
                door_out(f"aisle_{nm}", L(tw["u"] - tw["side"] / 2, tw["portal"]["v"]), west_out, tw["portal"]["w"], tw["portal"]["h"], why="side portal in the tower base")
            tops += tower(f"tower_{nm}", tw, f"aisle_{nm}", L(tw["u"], sg * (W / 2 + A / 2)), FL + Mn["Ha"], True)
        elif tw["role"] == "west":
            tops += tower("tower_w", tw, "tower_base", L(tw["u"], 0.0), FL, False)
        else:                                                                    # the second pair — at the choir: a shaft, a passage from the choir aisle
            host = f"choir_aisle_{nm}" if CH else "choir"; vv = sg * ((W / 2 + A - 0.6) if CH else (W / 2 - 0.6))
            S.cell(f"tower_{nm}e_base", "tower base", rect(tw["u"] - tw["side"] / 2, tw["u"] + tw["side"] / 2, tw["v"] - tw["side"] / 2, tw["v"] + tw["side"] / 2),
                   FL, FL + tw["stages"][0]["z1"], "void", walk=False, why="blank base")
            tops += tower(f"tower_{nm}e", tw, host, L(tw["u"], vv), FL, False)
    for i, tw in enumerate(EX.get("transept_towers", [])):
        sg = 1.0 if tw["v"] > 0 else -1.0; nm = "n" if sg > 0 else "s"
        tops += tower(f"ttower_{nm}", tw, f"transept_{nm}", L(u_mid, sg * (v1 - 1.0)), FL + tw["stages"][0]["z0"], True)
    if EX.get("westwork"):
        ww = EX["westwork"]; vw = W / 2 + Mw_tc(R) / 2 + 0.3 * kq
        S.cell("westwork", "westwork", rect(-t / 2, 0.9 * Mn["B"], -vw, vw), FL + ww["z0"], FL + ww["z1"], "void", walk=False, why="block above the first bay")
    # ---- additions
    chh = EX.get("chapter_house")
    if chh:
        sg = float(np.sign(chh["v"])); c = L(chh["u"], chh["v"]); tow = -sg * PV                  # the face toward the cathedral — a door from the vestibule
        P = octagon(c, chh["r"], math.atan2(tow[1], tow[0]) - math.pi / chh["n"], chh["n"]); ux = chh["u"]; vw = chh["vest_w"]
        S.cell("chapter_house", "chapter house hall", P, FL, FL + 0.92 * chh["h"], use="assembly", access="private")
        S.cell("chapter_vest", "chapter-house vestibule", rect(ux - vw / 2, ux + vw / 2, sg * v1, chh["v"] - sg * chh["r"]), FL, FL + 0.5 * chh["h"], use="passage")
        best = max(range(len(P)), key=lambda i: float(((P[i] + P[(i + 1) % len(P)]) / 2 - c) @ tow))
        pm = (P[best] + P[(best + 1) % len(P)]) / 2; Lw = math.dist(P[best], P[(best + 1) % len(P)])
        S.link("chapter_vest", "chapter_house", "door", pm, FL, FL, 0.42 * Lw, 0.5 * chh["h"], n=-tow)
    cl = EX.get("cloister")
    if cl:
        u0c, u1c = cl["u0"], (u_n1 if cl.get("door_v") is not None and TR else cl["u1"])     # by the door from the transept — up to the axis of its wall
        vlo, vhi = sorted((cl["v0"], cl["v1"])); wk = cl["walk"]; hc = cl["h"]; cen = np.array([(u0c + u1c) / 2, (vlo + vhi) / 2])
        sides = [((u0c, vlo), (u1c, vlo)), ((u1c, vlo), (u1c, vhi)), ((u1c, vhi), (u0c, vhi)), ((u0c, vhi), (u0c, vlo))]; inner = []; cen_w = []
        for i, (p, q) in enumerate(sides):
            p = np.array(p, float); q = np.array(q, float); d = (q - p) / max(np.linalg.norm(q - p), 1e-9); nr = np.array([-d[1], d[0]])
            nin = nr if (cen - (p + q) / 2) @ nr > 0 else -nr; pi_, qi_ = p + nin * wk + d * wk, q + nin * wk - d * wk; inner.append((pi_, qi_, q))
            cen_w.append((p + q + qi_ + pi_) / 4)
            S.cell(f"cloister_{i}", "cloister gallery", [L(*p), L(*q), L(*qi_), L(*pi_)], FL, FL + 0.8 * hc - 0.2, "covered", use="detour", access="private")
        zg = ter["z"] if ter else (float(gz(*L(*cen))) if gz is not None else Z0)
        S.cell("garth", "cloister court", rect(u0c + wk, u1c - wk, vlo + wk, vhi - wk), zg, None, "court", use="garden", access="private")
        for i in range(4):
            S.relate(f"cloister_{i}", f"cloister_{(i + 1) % 4}", "open"); S.relate(f"cloister_{i}", "garth", "arcade")
            pi_, qi_, q = inner[i]; p_, q_ = (np.array(x, float) for x in sides[i]); nn = (q_ - p_) / max(np.linalg.norm(q_ - p_), 1e-9)   # along the gallery — to the corner
            S.link(f"cloister_{i}", f"cloister_{(i + 1) % 4}", "opening", L(*((q + qi_) / 2)), FL, FL, wk, 0.8 * hc - 0.2, n=L(*nn) - L(0.0, 0.0))
        pi_, qi_, _ = inner[0]; nb_ = max(3, int(float(np.linalg.norm(qi_ - pi_)) / (2.6 * kq))); qa = pi_ + (qi_ - pi_) * (nb_ // 2 + 0.5) / nb_   # the middle arch — as in the build
        S.link("cloister_0", "garth", "arch", L(*qa), FL, zg, 1.4 * kq, 0.62 * hc, n=L(*(cen - qa)) - L(0.0, 0.0), why="passage into the garden")
        door_out("cloister_3", L(u0c, (vlo + vhi) / 2), -AX, 1.6 * kq, 3.0 * kq, why="entrance to the cloister from the west")
        if cl.get("door_v") is not None and TR:
            S.link(f"transept_{'n' if cl['side'] > 0 else 's'}", "cloister_1", "door", L(u_n1, cl["door_v"]), FL, FL, 1.6 * kq, 3.0 * kq, n=-AX, why="door from the transept into the cloister")
    sa = EX.get("sacristy")
    if sa:
        sg = sa["side"]; nm = "n" if sg > 0 else "s"; vw0 = W / 2 + (A if CH else 0.0)
        S.cell("sacristy", "sacristy", rect(sa["u0"], sa["u1"], sg * vw0, sg * (vw0 + Me["t"] / 2 + sa["d"]), Lc, 2), FL, FL + sa["h"], use="sacristy", access="private")
        S.link(f"choir_aisle_{nm}" if CH else "choir", "sacristy", "door", chord((sa["u0"] + sa["u1"]) / 2, sg * vw0, u_x1, Me["B"], Lc), FL, FL, 1.4 * kq, 2.6 * kq,
               n=PV * sg, why="sacristy door")
    bp = EX.get("baptistery")
    if bp:
        c = L(bp["u"], bp["v"]); P = octagon(c, bp["r"], 0.0, 8)
        S.cell("baptistery", "baptistery", P, FL, FL + bp["h"], use="baptism", access="sacred")
        best = max(range(8), key=lambda i: float(((P[i] + P[(i + 1) % 8]) / 2 - c) @ AX)); pm = (P[best] + P[(best + 1) % 8]) / 2; Lw = math.dist(P[best], P[(best + 1) % 8])
        door_out("baptistery", pm, (pm - c) / max(np.linalg.norm(pm - c), 1e-9), 0.45 * Lw, 0.42 * bp["h"], why="door toward the cathedral facade")
    cp = EX.get("campanile")
    if cp:
        hs = cp["side"] / 2; sg = 1.0 if cp["v"] > 0 else -1.0
        S.cell("campanile", "campanile base", rect(cp["u"] - hs, cp["u"] + hs, cp["v"] - hs, cp["v"] + hs), FL, FL + cp["stages"][0]["z1"], use="vestibule")
        door_out("campanile", L(cp["u"], cp["v"] - sg * hs), -sg * PV, 1.6 * kq, 3.0 * kq, why="campanile door toward the cathedral")
        tops += tower("campanile", cp, "campanile", L(cp["u"], cp["v"]), FL, False)
    # ---- anchors
    S.anchor("portal outside", OUT, p_front + west_out * 6.0 * kq, z_out if OUT != "outside" or gz is None else float(gz(*(p_front + west_out * 6.0 * kq))), facing=AX, tags=["entrance"])
    S.anchor("portal inside", "nave", L(2.5 * kq + t / 2, 0.0), FL, facing=AX, tags=["entrance"])
    S.anchor("middle of the nave", "nave", L(u_n1 / 2, 0.0), FL, facing=AX)
    if X_: S.anchor("crossing", "crossing", L(u_mid, 0.0), FL, facing=AX)
    ua_ = u_c1 - (0.12 * W if ek in ("apse", "chevet", "polygon") else 0.3 * Me["B"])
    S.anchor("altar", "choir", Lc(ua_, 0.0), FL, facing=-nrm(Lc, ua_, 0.0, 1.0, 0.0), tags=["sacred"])
    for cid, c, z in tops: S.anchor(f"top {cid}", cid, c, z, tags=["overview"])
    if cl: S.anchor("cloister garden", "garth", L(*cen), zg)
    if chh: S.anchor("chapter house hall", "chapter_house", L(chh["u"], chh["v"]), FL)
    S.auto_bounds()
    return S


def Mw_tc(R):
    return R["modules"][R["parts"]["west"]["mod"]]["tc"]


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:] if "=" in x); out = os.path.abspath(a[0])
    R = json.load(open(os.path.join(out, "temple.json")))
    from core.surface.ground import Ground
    S = spaces(R, Ground(out).height); D = S.to_dict(); R["spaces"] = D
    json.dump(R, open(os.path.join(out, "temple.json"), "w"), ensure_ascii=False, default=float)
    if kw.get("draw"): S.draw(os.path.join(out, "space_plan.png"), title=os.path.basename(out), route=S.path("outside", "choir"))
    print("__RESULT__", json.dumps({"stats": D["stats"], "non_passing": [c["what"] for c in D["checks"] if not c["ok"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
