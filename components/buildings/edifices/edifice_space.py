"""Space of an edifice from the plan (the scheme is space/space.py): cells — tier terraces (the top with a
batter), shrines, a hypostyle hall, a centric temple with a narthex and an apse, a court with colonnades, a
sunken circle with steps, pylon passages, an enclosure with gates, a causeway, the top of a cone and of drum
tiers, galleries of a temple-mountain; transitions — flights and ramps (the ends are found by geometry: what
is under the bottom, what is beyond the top), doors, gates, openings, a stair inside the body of a podium;
anchors for the scene. The geometry is the same as in the build (edifices/bl_edifice.py); the marks are in
world coordinates. Called by edifices/edifice_plan.py (edifice.json → spaces); standalone: .venv/bin/python
-m components.buildings.edifices.edifice_space <plan folder> [draw=1]"""
import json, math, os, sys

import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.space.space import Space, area, inside, offset_convex, signed_area               # noqa: E402

STAIR = math.tan(math.radians(38.0))


def octagon(c, r, phase=0.0, n=8):
    rr = r / math.cos(math.pi / n); return [np.asarray(c, float) + rr * np.array([math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n)]) for i in range(n)]


def wall_gates(R):
    """Enclosure gates: points (local) — from the wall and the gate parts on its line."""
    out = []
    for p in R["parts"]:
        if p["type"] == "wall": out += [tuple(g) for g in p.get("gates", [])]
    return out


def spaces(R, gz=None):
    FR = R["frame"]; O = np.asarray(FR["origin"], float); AX = np.asarray(FR["ax"], float); PV = np.asarray(FR["pv"], float)
    Z0 = float(FR["z0"]); FL = Z0 + float(FR["floor"]); kq = float(R["kq"]); parts = R["parts"]; typ = R["type"]
    steep = 72.0 if typ in ("step_pyramid", "temple_mountain") else 52.0
    S = Space(frame={"origin": O.round(3).tolist(), "ax": AX.round(5).tolist(), "floor_z": round(FL, 2), "kind": "edifice", "steep": steep})
    def L(u, v): return O + AX * u + PV * v
    def Lp(P): return [L(*q) for q in P]
    def D(du, dv): d = AX * du + PV * dv; return d / max(np.linalg.norm(d), 1e-9)
    def rect(c, Lx, Wy): return [L(c[0] + x * Lx / 2, c[1] + y * Wy / 2) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    def at(pw, z, skip=None, walk_only=False): return S.cells[S.locate(np.array([[pw[0], pw[1], z]]), skip=skip, walk_only=walk_only)[0]]["id"]
    seen = []; pend = []
    def door(*a, **k): pend.append((a, k))                                      # doors go after all the cells (the neighbor is found by geometry)
    def door_now(cell, pw, n_out, z, w, h, kind="door", why=""):
        """An opening of the cell cell at the point pw (world) outward n_out: the neighbor is found by geometry behind
        the opening; a repeat of the same pair nearby is skipped."""
        other = at(pw + n_out * 1.0, z + 0.6, skip=S.idx[cell])
        if other == cell or any({other, cell} == pr and np.linalg.norm(pw - q) < 3.0 for pr, q in seen): return
        if other == "outside" and any(np.linalg.norm(pw - f) < 1.5 for f in feet_): return     # steps lead to the door — a flight transition
        seen.append(({other, cell}, pw)); g = None
        if other == "outside" and gz is not None: g = float(gz(*(pw + n_out * 2.0)))
        S.link(other, cell, kind, pw, Z0 if other == "outside" else S.cells[S.idx[other]]["z0"], z, w, h, n=-n_out, ground=g, why=why)
    ED = R.get("ed", {}); cnt = {}
    def nid(base): cnt[base] = cnt.get(base, 0) + 1; return f"{base}{cnt[base] - 1}"
    by = {}
    for p in parts: by.setdefault(p["type"], []).append(p)
    feet_ = [L(*(p["p1"] if p["z1"] >= p["z0"] else p["p0"])) for p in by.get("flight", [])]
    # ---- tier terraces: a top with a batter, a parapet or a drop
    def top_of(p):
        h = p["z1"] - p["z0"]; return offset_convex(Lp(p["poly"]), (p["batter"] * 0.6 * h - 0.04 * h) if p.get("tablero") else p["batter"] * h)
    def ring_w(Pt, Q):
        """Ledge width: from the base of the tier above to the edge of the top of this one (the outlines are convex and
        nested)."""
        best = np.inf
        for x in Q:
            for a, b in zip(Pt, Pt[1:] + Pt[:1]):
                e = b - a; nr = np.array([e[1], -e[0]]) / max(np.linalg.norm(e), 1e-9); best = min(best, abs(float((x - a) @ nr)))
        return best
    for p in by.get("tier", []):                                                 # a ledge narrower than 1.2 m (the tier above takes almost the whole width) is not walked on
        Pt = top_of(p); ledge = None
        for q in by["tier"]:
            if q is p or abs(q["z0"] - p["z1"]) > 0.05: continue
            Q = np.asarray(Lp(q["poly"]))
            if not bool(inside(np.asarray(Pt), *Q.mean(0))): continue                # the tier above does not stand on this one (a gate platform is next to it)
            ledge = ring_w(Pt, Q) if bool(inside(np.asarray(Pt), Q[:, 0], Q[:, 1]).all()) else 0.0
        S.cell(nid("tier"), "ledge" if ledge is not None and ledge < 1.2 else "tier terrace", Pt, FL + p["z1"], None, "exterior",
               walk=not (ledge is not None and ledge < 1.2), edge="parapet" if p.get("parapet", 0) >= 0.4 else "edge", use="terrace", why=p.get("why", ""))
    for p in by.get("cone", []):
        c = L(*p["c"]); S.cell(nid("cone_top"), "cone top", octagon(c, p["r1"], 0.0, 40), FL + p["z1"], None, "exterior", edge="edge", use="summit")
    for p in by.get("drum", []):                                                 # drum tier: a gallery behind the sill, the core, the top
        c = L(*p["c"]); n = int(p["n"]); r, z0, h = p["r"], p["z0"], p["h"]; k_ = nid("drum")
        S.cell(f"{k_}_core", "tier core", octagon(c, r * 0.72, 0.0, n), FL + z0, FL + z0 + h, "void", walk=False)
        S.cell(f"{k_}_arcade", "tier gallery", octagon(c, r, 0.0, n), FL + z0, FL + z0 + h, "covered", walk=False, holes=[octagon(c, r * 0.72, 0.0, n)],
               why="arches with a sill — no entering from outside")
        if p.get("roof", "flat") == "flat": S.cell(f"{k_}_top", "tier top", octagon(c, r, 0.0, n), FL + z0 + h, None, "exterior", edge="edge", use="terrace")
    gal = []
    for p in by.get("gallery", []):                                              # temple-mountain gallery: a ring along the tier edge
        P = Lp(p["poly"]); k_ = nid("gallery"); gal.append((p, k_))
        S.cell(k_, "gallery", P, FL + p["z0"], FL + p["z0"] + 0.82 * p["h"], "covered", holes=[offset_convex(P, p["w"])], use="detour")
    # ---- halls and sanctuaries
    for p in by.get("shrine", []):
        c = np.asarray(p["c"], float); d = np.asarray(p.get("dir", [-1, 0]), float); k_ = nid("shrine")
        S.cell(k_, "sanctuary", rect(c, p["L"], p["W"]), FL + p["z0"], FL + p["z0"] + p["H"], use="sanctuary", access="sacred", why=p.get("why", ""))
        half = p["L"] / 2 if abs(d[0]) > 0.5 else p["W"] / 2
        dh = min(0.62 * p["H"], (4.5 + 1.0 * kq) * kq); dw = min(0.32 * p["W"], 2.4 * kq)
        door(k_, L(*(c + d * half)), D(*d), FL + p["z0"], dw, dh, why="sanctuary entrance")
    for p in by.get("hall", []):
        c = np.asarray(p["c"], float); k_ = nid("hall")
        S.cell(k_, "hypostyle hall", rect(c, p["L"], p["W"]), FL, FL + p["H"], use="assembly", why=p.get("why", ""))
        for sg in (-1, 1): door(k_, L(c[0] + sg * p["L"] / 2, c[1]), D(sg, 0), FL, 0.16 * p["W"], 0.5 * p["Hs"], why="hall gate on the axis")
    for p in by.get("domed", []):
        c = np.asarray(p["c"], float); W, arm, H, t = p["W"], p["arm"], p["H"], p["t"]; He = H + 0.06 * W + 0.8 * kq; ztop = He + p["drum"] + p["dh"]
        if p["kind"] == "rotunda":
            R_ = W / 2; body = octagon(L(*c), R_ + t / 2, math.atan2(AX[1], AX[0]) + math.pi - math.pi / 32, 32); S.cell("domed", "rotunda", body, FL, FL + ztop, use="assembly")
            nd = p["narthex"]; u0 = c[0] - R_ - t / 2 - nd; w2 = 0.28 * W; Rr = R_ + t / 2; al = math.asin(min(0.99, w2 / Rr))
            arc = [L(c[0] + Rr * math.cos(math.pi + a_), c[1] + Rr * math.sin(math.pi + a_)) for a_ in np.linspace(al, -al, 9)]
            S.cell("narthex", "narthex", [L(u0, -w2), arc[0]] + arc[1:-1] + [arc[-1], L(u0, w2)], FL, FL + 0.55 * H, use="vestibule")
            door("domed", L(c[0] - Rr, c[1]), D(-1, 0), FL, 0.8 * 2 * math.pi * Rr / 32, 0.4 * H, why="rotunda door")
            door("narthex", L(u0, c[1]), D(-1, 0), FL, 0.3 * W, 0.4 * H, why="entrance to the narthex")
            continue
        h_ = W / 2 + arm
        if p["kind"] == "cross_in_square": body = rect(c, 2 * h_, 2 * h_)
        else:
            a, w2 = h_, W / 2
            body = Lp([(c[0] - a, c[1] - w2), (c[0] - w2, c[1] - w2), (c[0] - w2, c[1] - a), (c[0] + w2, c[1] - a), (c[0] + w2, c[1] - w2), (c[0] + a, c[1] - w2),
                       (c[0] + a, c[1] + w2), (c[0] + w2, c[1] + w2), (c[0] + w2, c[1] + a), (c[0] - w2, c[1] + a), (c[0] - w2, c[1] + w2), (c[0] - a, c[1] + w2)])
        S.cell("domed", "a domed temple", body, FL, FL + ztop, use="assembly", why=p.get("why", ""))
        if p.get("apses"):                                                       # apse in the east: an arch in the east wall
            ra = 0.42 * W; ca = (c[0] + h_, c[1])
            S.cell("apse", "apse", [L(ca[0] + ra * math.cos(-math.pi / 2 + math.pi * j / 10), ca[1] + ra * math.sin(-math.pi / 2 + math.pi * j / 10)) for j in range(11)],
                   FL, FL + 0.62 * H, use="altar", access="sacred")
            S.relate("domed", "apse", "wall"); S.link("domed", "apse", "arch", L(*ca), FL, FL, 2 * ra * 0.9, 0.62 * H * 0.85, n=D(1, 0))
        if p.get("narthex"):
            nd = p["narthex"]; u0 = c[0] - h_ - nd; w2 = h_ if p["kind"] == "cross_in_square" else W / 2 + 0.2 * arm; Hn_ = 0.62 * H
            S.cell("narthex", "narthex", Lp([(u0, c[1] - w2), (c[0] - h_, c[1] - w2), (c[0] - h_, c[1] + w2), (u0, c[1] + w2)]), FL, FL + Hn_, use="vestibule")
            door("narthex", L(u0, c[1]), D(-1, 0), FL, min(0.3 * W, 3.5 * kq), min(0.5 * Hn_, 6.0 * kq), why="entrance to the narthex")
        door("domed", L(c[0] - h_, c[1]), D(-1, 0), FL, min(0.4 * W, 4.0 * kq), min(0.45 * H, 7.0 * kq), why="west door")
    # ---- courts
    for p in by.get("court", []):
        c = np.asarray(p["c"], float)
        if p.get("shape") == "rect":                                             # court of the axial temple: open, colonnades along the long sides
            k_ = nid("court"); h = p.get("h", 8.0)
            S.cell(k_, "court", rect(c, p["L"], p["W"]), FL, None, "court", use="procession", why=p.get("why", ""))
            for sg in (1, -1):
                vc = c[1] + sg * (p["W"] / 2 - 0.16 * p["W"]); cid = f"{k_}_colonnade{'n' if sg > 0 else 's'}"
                S.cell(cid, "colonnade", Lp([(c[0] - p["L"] / 2, vc), (c[0] + p["L"] / 2, vc), (c[0] + p["L"] / 2, c[1] + sg * p["W"] / 2), (c[0] - p["L"] / 2, c[1] + sg * p["W"] / 2)]),
                       FL, FL + 0.82 * h, "covered", use="detour")
                S.relate(k_, cid, "open"); S.link(k_, cid, "opening", L(c[0], vc), FL, FL, 3.0 * kq, 0.82 * h)
            if p.get("center") == "altar": S.anchor("court altar", k_, L(*c), FL, tags=["sacred"])
            continue
        R_, rf, dep, nt = p["r"], p["r_floor"], p["depth"], max(1, int(p["terraces"])); cw = L(*c)   # sunken circle with steps
        hole = [octagon(cw, 0.3 * rf * 0.92, 0.0, 32)] if p.get("center") == "pit" else None
        S.cell("pit_floor", "circle floor", octagon(cw, rf, 0.0, 64), FL - dep, None, "court", use="rite", holes=hole, why=p.get("why", ""))
        for i in range(1, nt): S.cell(f"pit_step{i}", "circle step", octagon(cw, R_ - (R_ - rf) * i / nt, 0.0, 64), FL - dep * i / nt, None, "court", edge="edge", use="spectators",
                                      holes=[octagon(cw, R_ - (R_ - rf) * (i + 1) / nt, 0.0, 64)])           # ring: inside — the step below
        S.anchor("circle floor", "pit_floor", cw + D(-1, 0) * 0.6 * rf, FL - dep, facing=D(1, 0), tags=["sacred"])
        if p.get("center") == "altar": S.anchor("altar", "pit_floor", cw, FL - dep, tags=["sacred"])
    # ---- pylon passages: the gates into the world of the temple
    gates = wall_gates(R)
    def passage(c, d, w, h, depth, why):
        k_ = nid("pylon_pass"); tg = np.array([-d[1], d[0]]); cw = np.asarray(c, float)
        S.cell(k_, "pylon passage", [cw - d * depth / 2 - tg * w / 2, cw + d * depth / 2 - tg * w / 2, cw + d * depth / 2 + tg * w / 2, cw - d * depth / 2 + tg * w / 2],
               FL, FL + h, "covered", use="gate", access="gate", why=why)
        return k_
    pas = []
    for p in by.get("pylon", []):
        pas.append((passage(L(*p["c"]), AX, 0.18 * p["w"], 0.465 * p["h"], p["d"], p.get("why", "")), L(*p["c"]), AX, p["d"], 0.18 * p["w"], 0.465 * p["h"]))
    for p in by.get("gate", []):
        if p.get("kind") != "pylon":                                             # a gatehouse tower on a platform: a passage under an arch, its own place is the mass
            d = D(*p["dir"]); pas.append((passage(L(*p["c"]), d, p["w"] * 0.36, 0.72 * p["h"], p["d"], "passage under the gatehouse tower"), L(*p["c"]), d, p["d"], p["w"] * 0.36,
                                           0.72 * p["h"]))
            S.cells[-1]["z0"] = round(FL + p["z0"], 2); S.cells[-1]["z1"] = round(FL + p["z0"] + 0.72 * p["h"], 2); continue
        if p.get("kind") == "pylon" and any(math.dist(p["c"], g) < 1.0 for g in gates):
            d = D(*p["dir"]); pas.append((passage(L(*p["c"]), d, 0.18 * p["w"], 0.465 * p["h"], p["d"], "gate in the enclosure"), L(*p["c"]), d, p["d"], 0.18 * p["w"], 0.465 * p["h"]))
    # ---- causeways and bridges
    cws = []
    for p in by.get("bridge", []):
        a_, b_ = L(*p["p0"]), L(*p["p1"]); d = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9); tg = np.array([-d[1], d[0]]); w = p["w"]
        P = [a_ - tg * w / 2, b_ - tg * w / 2, b_ + tg * w / 2, a_ + tg * w / 2]
        if p.get("high"): S.cell(nid("bridge"), "bridge between towers", P, FL + p["z"], None, "exterior", walk=False, edge="rail", why="there is no entrance in the towers — the bridge is unreachable")
        else: cws.append((p, S.cell(nid("causeway"), "dam", P, FL + p["z"], None, "exterior", edge="rail", use="approach", why=p.get("why", ""))))
    # ---- enclosures: a court inside, if the gate does not lead into another cell
    for p in by.get("wall", []):
        if not p.get("gates"): continue
        P = Lp(p["poly"]); cen = np.mean(P, axis=0); need = []
        for g in p["gates"]:
            gw = L(*g); inn = (cen - gw) / max(np.linalg.norm(cen - gw), 1e-9)                # behind the gate (pylon) passage — what is inside
            dd = max([q["d"] / 2 for q in by.get("gate", []) + by.get("pylon", []) if math.dist(q["c"], g) < 1.0] + [0.0])
            need.append((gw, inn, at(gw + inn * (dd + 1.5), FL + 0.6)))
        if any(x[2] == "outside" for x in need):
            k_ = nid("yard"); S.cell(k_, "court behind the enclosure", P, Z0, None, "court", use="approach", why=p.get("why", ""))
            for gw, inn, cell in need:
                if any(np.linalg.norm(gw - q[1]) < 2.0 for q in pas) or any(np.linalg.norm(gw - L(*b["p1"])) < 2.0 for b, _ in cws): continue   # the pylon and the causeway have their own
                outer = at(gw - inn * 2.0, FL + 0.9)
                S.link(outer, k_, "gate", gw, S.cells[S.idx[outer]]["z0"] if outer != "outside" else Z0, Z0, 6.0 * kq, 6.0 * kq, n=inn,
                       ground=float(gz(*(gw - inn * 2.0))) if (gz is not None and outer == "outside") else None, why="enclosure gate")
    def gate_foot(q):
        d = D(*q["dir"]); tg = np.array([-d[1], d[0]]); c = L(*q["c"])
        return [c + d * x * q["d"] / 2 + tg * y * q["w"] / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # ---- masses standing on a terrace or in a court (a higher tier, a cone, a drum, a gatehouse tower) are
    # holes: their own place is around
    masses = [(FL + q["z0"], Lp(q["poly"])) for q in by.get("tier", [])] + [(FL + q["z0"], octagon(L(*q["c"]), q["r0"], 0.0, 40)) for q in by.get("cone", [])] \
        + [(FL + q["z0"], octagon(L(*q["c"]), q["r"] + 0.5 * kq, 0.0, int(q["n"]))) for q in by.get("drum", [])] \
        + [(FL + q["z0"], gate_foot(q)) for q in by.get("gate", []) if q.get("kind") != "pylon"] \
        + [(FL + q["z0"], [L(*q["c"]) + (AX * x + PV * y) * q["stages"][0]["half"] for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) for q in by.get("tower", [])] \
        + [(FL + q["z0"], [L(*q["c"]) + (AX * x + PV * y) * q["half"] for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) for q in by.get("prang", [])] \
        + [(FL + q["z0"], octagon(L(*q["c"]), q["r"] * 1.3, 0.0, 12)) for q in by.get("minaret", [])] \
        + [(FL + q["z0"], [L(*q["c"]) + (AX * x + PV * y) * q["base"] * 0.75 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) for q in by.get("obelisk", [])] \
        + [(Z0, octagon(L(*q["c"]), q["r"] + 0.8 * kq, 0.0, 48)) for q in by.get("court", []) if q.get("depth", 0) > 0 and q.get("shape") != "rect"]   # a circle is a hole in the court
    for i, c in enumerate(S.cells):
        if c["poly"] is None or c["cls"] not in ("exterior", "court"): continue
        for zm, Mp in masses:
            Mq = np.asarray(Mp, float)
            if abs(zm - c["z0"]) > 0.6 or not bool(S.inside_cell(i, *Mq.mean(0))) or not bool(inside(S.poly(i), Mq[:, 0], Mq[:, 1]).all()): continue
            if area(Mq) > 0.97 * area(S.poly(i)): continue
            c.setdefault("holes", []).append([[round(float(x), 2), round(float(y), 2)] for x, y in (Mq if signed_area(Mq) < 0 else Mq[::-1])])
    for k_, cw, d, depth, w, h in pas:                                           # pylon passage: outside → inside (unless by steps)
        for sg in (-1, 1):
            q = cw + d * sg * (depth / 2); zk = S.cells[S.idx[k_]]["z0"]
            if any(np.linalg.norm(q - f) < 1.0 for f in feet_): continue
            other = at(q + d * sg * 1.0, zk + 0.6, skip=S.idx[k_])
            g = float(gz(*(q + d * sg * 2.0))) if (gz is not None and other == "outside") else None
            S.relate(k_, other, "open"); S.link(other, k_, "opening", q, Z0 if other == "outside" else S.cells[S.idx[other]]["z0"], zk, w, h, n=-d * sg, ground=g)
    for p, cid in cws:                                                           # causeway ends (from the bank — a threshold or steps)
        a_, b_ = L(*p["p0"]), L(*p["p1"]); d = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9)
        if not any(np.linalg.norm(a_ - f) < 1.0 for f in feet_):
            S.link("outside", cid, "opening", a_, Z0, FL + p["z"], p["w"], n=d, ground=float(gz(*(a_ - d * 1.5))) if gz is not None else None, why="causeway from the bank")
        inner = at(b_ + d * 1.5, FL + 0.6)
        if inner != "outside": S.link(cid, inner, "gate", b_, FL + p["z"], S.cells[S.idx[inner]]["z0"], p["w"], 6.0 * kq, n=d, why="causeway — to the enclosure gate")
    for a_, k_ in pend: door_now(*a_, **k_)
    # ---- flights, ramps, sunken circle: the ends — by geometry
    pit = next((p for p in by.get("court", []) if p.get("depth", 0) > 0 and p.get("shape") != "rect"), None)
    for p in by.get("flight", []):
        pb, pt, zb, zt = (p["p0"], p["p1"], p["z0"], p["z1"]) if p["z1"] >= p["z0"] else (p["p1"], p["p0"], p["z1"], p["z0"])
        if pit is not None and p["z1"] < p["z0"] and math.dist(p["p1"], pit["c"]) <= pit["r"]:   # descent into the circle: a chain along the steps
            R_, rf, dep, nt = pit["r"], pit["r_floor"], pit["depth"], max(1, int(pit["terraces"])); u0, u1 = np.asarray(p["p0"], float), np.asarray(p["p1"], float)
            prev = at(L(*u0) - D(*(u1 - u0)) * 0.5, FL + 0.6); zp = FL if prev != "outside" else Z0
            for i in range(1, nt + 1):
                q0, q1 = u0 + (u1 - u0) * (i - 1) / nt, u0 + (u1 - u0) * i / nt; cid = f"pit_step{i}" if i < nt else "pit_floor"
                S.link(prev, cid, "stair", [L(*q0), L(*q1)], zp, FL - dep * i / nt, p["w"], why="descent along the axis into the circle" if i == 1 else "")
                prev, zp = cid, FL - dep * i / nt
            continue
        a_, b_ = L(*pb), L(*pt); d = (b_ - a_) / max(np.linalg.norm(b_ - a_), 1e-9); tg = np.array([-d[1], d[0]])
        lo = at(a_ - d * 0.4, FL + zb + 0.6); hi = at(b_ + d * 0.5, FL + zt + 0.6)
        if lo == "outside" and FL + zb < Z0 and (pit is None or math.dist(pb, pit["c"]) > pit["r"] + 1.0): lo = at(a_ - d * 0.4, Z0 + 0.6)   # the foot is below the court — in the court
        if lo == hi: continue
        g = float(gz(*(a_ - d * 0.6))) if (gz is not None and lo == "outside") else None
        mids = []                                                                # the flight passes intermediate terraces — step off onto each
        for i, c in enumerate(S.cells):
            if not c["walk"] or c["poly"] is None or c["cls"] not in ("exterior", "court") or c["id"] in (lo, hi) or not (FL + zb + 0.3 < c["z0"] < FL + zt - 0.3): continue
            f = (c["z0"] - FL - zb) / max(zt - zb, 1e-9); q = a_ + (b_ - a_) * f
            if bool(S.inside_cell(i, *np.array([q + tg * (p["w"] / 2 + 0.6), q - tg * (p["w"] / 2 + 0.6), q]).T).any()): mids.append((f, c["id"], q, c["z0"]))
        chain = [(0.0, lo, a_, FL + zb)] + sorted(mids) + [(1.0, hi, b_, FL + zt)]
        for (f0, c0, q0, z0_), (f1, c1, q1, z1_) in zip(chain[:-1], chain[1:]):
            S.link(c0, c1, "stair", [q0, q1], z0_, z1_, p["w"], ground=g if f0 == 0.0 else None, why=p.get("why", "") if f0 == 0.0 else "stepping off onto a terrace")
    for p in by.get("ramp", []):
        c = L(*p["c"]); ri, ro = p["r_in"], p["r_out"]; a0 = p["a0"] + math.atan2(AX[1], AX[0]); N = max(8, int(24 * p["turns"]))
        pts = []; length = 0.0
        for i in range(N + 1):
            f = i / N; ang = a0 + 2 * math.pi * p["turns"] * f; rm = (ri[0] + (ri[1] - ri[0]) * f + ro[0] + (ro[1] - ro[0]) * f) / 2
            pts.append(c + rm * np.array([math.cos(ang), math.sin(ang)]))
            if i: length += float(np.linalg.norm(pts[-1] - pts[-2]))
        u_e = (pts[-1] - c) / max(np.linalg.norm(pts[-1] - c), 1e-9); wr = (ro[1] - ri[1])
        lo = at(pts[0], FL + p["z0"] + 0.6, walk_only=True); hi = at(pts[-1] - u_e * (wr / 2 + 0.6), FL + p["z1"] + 0.6, walk_only=True)
        if lo == hi or abs(S.cells[S.idx[hi]]["z0"] - (FL + p["z1"])) > 1.0: continue   # the top has collapsed — a ramp to the ruin
        S.link(lo, hi, "ramp", pts[::max(1, N // 24)] + [pts[-1]], FL + p["z0"], FL + p["z1"], wr, length=length, why="spiral ramp around the body")
    for p in by.get("stair_in", []):                                             # a door in the tier's face, a stair inside the body up to its top
        d = D(*p["dir"]); q = L(*p["c"]); inn = p["batter"] * (p["z1"] - p["z0"]) + 1.5
        lo = at(q + d * 1.0, FL + p["z0"] + 0.6); hi = at(q - d * inn, FL + p["z1"] + 0.6)
        if lo != hi: S.link(lo, hi, "stair", [q + d * 0.5, q - d * inn], FL + p["z0"], FL + p["z1"], p["w"], length=(p["z1"] - p["z0"]) / STAIR, hidden=True,
                            why="a stair inside the tier body (the terrace is narrower than the flight)")
    for p in by.get("flight_core", []):                                          # a stair inside the podium body (temple-mountain)
        tops = [c for c in S.cells if c["id"].startswith("tier") and abs(c["z0"] - (FL + p["z1"])) < 0.3]
        lows = [c for c in S.cells if c["id"].startswith("tier") and abs(c["z0"] - (FL + p["z0"])) < 0.3]
        if tops and lows:
            ctop = np.mean(np.asarray(tops[0]["poly"]), axis=0); half = float(np.max(np.abs((np.asarray(tops[0]["poly"]) - ctop) @ AX)))
            S.link(lows[0]["id"], tops[0]["id"], "stair", [ctop - AX * (half + 1.0), ctop], FL + p["z0"], FL + p["z1"], 1.2 * kq,
                   length=(p["z1"] - p["z0"]) / STAIR, hidden=True, why="a stair inside the podium body")
    # ---- terraces at the same mark next to each other (gate platform ↔ tier), gallery ↔ terrace
    ter = [i for i, c in enumerate(S.cells) if c["cls"] == "exterior" and c["walk"] and c["poly"] is not None]
    for x in range(len(ter)):
        for y in range(x + 1, len(ter)):
            A_, B_ = S.cells[ter[x]], S.cells[ter[y]]
            if abs(A_["z0"] - B_["z0"]) > 0.3: continue
            PA, PB = S.poly(ter[x]), S.poly(ter[y]); best = (1e9, None)
            for P1, P2 in ((PA, PB), (PB, PA)):                                  # the vertices of one — to the edges of the other, both ways
                for q in P1:
                    for j in range(len(P2)):
                        a_, b_ = P2[j], P2[(j + 1) % len(P2)]; e = b_ - a_; t_ = float(np.clip((q - a_) @ e / max(e @ e, 1e-9), 0, 1)); dd = float(np.linalg.norm(q - a_ - e * t_))
                        if dd < best[0]: best = (dd, (q + a_ + e * t_) / 2)
            if best[0] < 1.5:                                                    # the passage — opposite the middle of the smaller one, across the joint
                small, big_ = (ter[x], ter[y]) if S.area_cell(ter[x]) <= S.area_cell(ter[y]) else (ter[y], ter[x]); cs = S.poly(small).mean(0); PB = S.poly(big_)
                qq = min(((a_ + (b_ - a_) * float(np.clip((cs - a_) @ (b_ - a_) / max((b_ - a_) @ (b_ - a_), 1e-9), 0, 1))) for a_, b_ in zip(PB, np.roll(PB, -1, 0))),
                         key=lambda q: float(np.linalg.norm(q - cs)))
                S.link(S.cells[small]["id"], S.cells[big_]["id"], "opening", qq, A_["z0"], B_["z0"], 3.0 * kq, n=qq - cs, why="terrace to terrace")
    for gp_, c in [(p, S.cells[S.idx[k_]]) for p, k_ in gal]:                   # gallery ↔ terrace: between the posts, across the row
        Pi = offset_convex(Lp(gp_["poly"]), gp_["w"]); cen = np.mean(Pi, axis=0)
        for j in range(len(Pi)):
            a_, b_ = Pi[j], Pi[(j + 1) % len(Pi)]; m = max(2, int(float(np.linalg.norm(b_ - a_)) / gp_["post"])); q = a_ + (b_ - a_) * (m // 4 + 0.5) / m   # to the side of the axis (the flight is there)
            nq = (cen - q) / max(np.linalg.norm(cen - q), 1e-9); host = at(q + nq * 1.0, c["z0"] + 0.6, skip=S.idx[c["id"]])   # the terrace — right behind the row of posts
            if host == "outside": continue
            S.relate(c["id"], host, "arcade"); S.link(c["id"], host, "opening", q, c["z0"], S.cells[S.idx[host]]["z0"], 2.0 * kq, n=nq)
    linked = {t["a"] for t in S.trans} | {t["b"] for t in S.trans}               # a terrace that no transition leads to is a ledge
    for c in S.cells:
        if c["cls"] == "exterior" and c["walk"] and c["poly"] is not None and c["id"] not in linked and c["id"].startswith("tier"):
            c["walk"] = False; c["kind"] = "ledge without access"; c["why"] = (c.get("why", "") + "; " if c.get("why") else "") + "the stair passes by — a decorative ledge"
    # ---- anchors
    fl = [p for p in by.get("flight", [])]
    if fl:
        p = fl[0]; pb = p["p0"] if p["z1"] >= p["z0"] else p["p1"]; pt = p["p1"] if p["z1"] >= p["z0"] else p["p0"]; d = D(*(np.asarray(pt) - np.asarray(pb)))
        S.anchor("stair foot", at(L(*pb) - d * 3.0, FL + 0.6), L(*pb) - d * 3.0, Z0 if at(L(*pb) - d * 3.0, FL + 0.6) == "outside" else FL, facing=d, tags=["entrance"])
    walk = [c for c in S.cells if c["walk"] and c["poly"] is not None and c["cls"] in ("exterior", "court")]
    if walk:
        top = max(walk, key=lambda c: c["z0"]); S.anchor("summit", top["id"], np.mean(np.asarray(top["poly"]), axis=0), top["z0"], tags=["overview"])
    for c in S.cells:
        if c["kind"] in ("sanctuary", "a domed temple", "rotunda", "hypostyle hall"): S.anchor(c["kind"], c["id"], np.mean(np.asarray(c["poly"]), axis=0), c["z0"])
    for p in by.get("ring", []):
        if p["obj"] == "altar": S.anchor("altar", at(L(*p["c"]), FL + p["z0"] + 0.6), L(*p["c"]), FL + p["z0"], tags=["sacred"])
    S.auto_bounds()
    return S


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:] if "=" in x); out = os.path.abspath(a[0])
    R = json.load(open(os.path.join(out, "edifice.json")))
    from core.surface.ground import Ground
    S = spaces(R, Ground(out).height); D = S.to_dict(); R["spaces"] = D
    json.dump(R, open(os.path.join(out, "edifice.json"), "w"), ensure_ascii=False, default=float)
    if kw.get("draw"): S.draw(os.path.join(out, "space_plan.png"), title=os.path.basename(out))
    print("__RESULT__", json.dumps({"stats": D["stats"], "non_passing": [c["what"] for c in D["checks"] if not c["ok"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
