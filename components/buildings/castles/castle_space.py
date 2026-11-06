"""Space of a castle from the plan castle.json (the scheme is space/space.py): cells — courts (a ring has a
hole, terraces are strips along the retaining-wall lines), gate passages, bridges and causeways, wall
walkways as segments between towers and gates, tower bodies and platforms, the main tower, buildings;
transitions — gates (a road with a ramp, a bridge), stairs to the wall (the "stair" element), terrace
stairs, stairs in towers and building doors. The castle build (castles/bl_castle.py) builds towers and
buildings as solid bodies with dark "openings" on the faces: such doors are decoys (model: patch), the
stairs in towers are not built (hidden); the model check does not see through them, but the summary counts
them. World coordinates.
Called by castles/castle_plan.py (castle.json → spaces); standalone: .venv/bin/python -m
components.buildings.castles.castle_space <plan folder> [draw=1]"""
import json, math, os, sys

import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.space.space import Space, area, inside                                            # noqa: E402

STAIR = math.tan(math.radians(38.0))
RU = {"palas": "palas", "palas_big": "palas", "chapel": "chapel", "kitchen": "kitchen", "stable": "stable", "store": "warehouse", "governor": "castellan's house",
      "hall": "hall", "wing": "wing", "tower_house": "house-tower", "barracks": "barracks", "smithy": "forge", "well": "well"}


def footprint(plan, c, r, d, n_round=24):
    """Tower outline (like bl_castle._footprint)."""
    c = np.asarray(c, float); d = np.asarray(d if d is not None else (1, 0), float); d /= max(np.linalg.norm(d), 1e-9); o = np.array([d[1], -d[0]])
    if plan == "round": return np.array([c + r * np.array([math.cos(2 * math.pi * i / n_round), math.sin(2 * math.pi * i / n_round)]) for i in range(n_round)])
    if plan == "D": return np.array([c + d * r * math.cos(t) + o * r * math.sin(t) for t in np.linspace(0, math.pi, 13)] + [c - d * r - o * r, c + d * r - o * r])
    k = {"square": 4, "poly6": 6, "poly8": 8}.get(plan, 8); a0 = math.atan2(d[1], d[0]) + (math.pi / 4 if k == 4 else math.pi / k); rr = r / math.cos(math.pi / k)
    return np.array([c + rr * np.array([math.cos(a0 + 2 * math.pi * i / k), math.sin(a0 + 2 * math.pi * i / k)]) for i in range(k)])


def wall_frame(w):
    """Points, outward normals, walkway, bottom, wall thickness (like bl_castle.wall_frame)."""
    P = np.asarray(w["pts"], float); closed = bool(w["closed"])
    tn = (np.roll(P, -1, 0) - P) if closed else np.vstack([P[1:] - P[:-1], P[-1:] - P[-2:-1]])
    tp = (P - np.roll(P, 1, 0)) if closed else np.vstack([P[1:2] - P[:1], P[1:] - P[:-1]])
    for D in (tn, tp):                                                           # ledge: two stations at one point — the direction of the neighboring one
        bad = np.linalg.norm(D, axis=1) < 1e-6
        for j in np.flatnonzero(bad): D[j] = D[(j + 1) % len(P)] if not bad[(j + 1) % len(P)] else D[j - 1]
    tn = tn / np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9); tp = tp / np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-9)
    N = np.column_stack([tn[:, 1], -tn[:, 0]]) + np.column_stack([tp[:, 1], -tp[:, 0]]); N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    T = np.asarray(w["t"], float) if isinstance(w["t"], list) else np.full(len(P), float(w["t"]))
    return P, N, np.asarray(w["walk"], float), np.asarray(w["base"], float), T


def clip_band(P, g, s0, s1):
    """Polygon P ∩ {s0 ≤ p·g ≤ s1} (Sutherland–Hodgman, two half-planes)."""
    def half(Q, f):
        out = []
        for i in range(len(Q)):
            a, b = Q[i], Q[(i + 1) % len(Q)]; fa, fb = f(a), f(b)
            if fa >= 0: out.append(a)
            if (fa >= 0) != (fb >= 0): out.append(a + (b - a) * (fa / (fa - fb)))
        return out
    Q = [np.asarray(p, float) for p in P]
    if np.isfinite(s0): Q = half(Q, lambda p: p @ g - s0)
    if len(Q) >= 3 and np.isfinite(s1): Q = half(Q, lambda p: s1 - p @ g)
    return Q


def spaces(R, gz=None):
    st = R["style"]; pt = float((st.get("crenel") or {}).get("pt", 0.6)); k_ = float(st.get("k", 1.0)); wall_t = float(st.get("wall_t", 2.6))
    c0 = np.asarray(R["site"]["centre"], float)
    S = Space(frame={"origin": c0.round(3).tolist(), "ax": [1.0, 0.0], "kind": "castle", "steep": 45.0})
    def at(p, z, skip=None, walk_only=False): return S.cells[S.locate(np.array([[p[0], p[1], z]]), skip=skip, walk_only=walk_only)[0]]["id"]
    def ward_at(p):
        """The court (terrace) whose outline contains the point; for nested ones — the smaller; none — None. The court
        and threshold marks differ — we search by plan."""
        hit = [(S.area_cell(S.idx[cid]), cid) for cids in wards.values() for cid in cids if bool(S.inside_cell(S.idx[cid], p[0], p[1]))]
        return min(hit)[1] if hit else None
    def ground(p, fall): return float(gz(p[0], p[1])) if gz is not None else fall
    # ---- courts: a ring has a hole; terraces are strips between the retaining-wall lines
    wards = {}
    for w in R["wards"]:
        P = np.asarray(w["poly"], float); holes = [w["hole"]] if w.get("hole") else None; tr = w.get("terraces")
        if tr and tr.get("levels") and tr.get("cuts"):
            g = np.asarray(tr["dir"], float); g = g / max(np.linalg.norm(g), 1e-9); b_ = [-np.inf] + list(tr["cuts"]) + [np.inf]
            for k in range(len(tr["levels"])):
                Q = clip_band(P, g, b_[k], b_[k + 1])
                if len(Q) < 3 or area(Q) < 6.0: continue
                cid = f"ward_{w['id']}_{k}"; S.cell(cid, "court terrace", Q, tr["levels"][k], None, "court", use="court", why=f"terrace {k + 1} of {len(tr['levels'])}")
                wards.setdefault(w["id"], []).append(cid)
        else:
            cid = f"ward_{w['id']}"; S.cell(cid, "court" if not holes else "court-ring", P, w["level"], None, "court", use="court", holes=holes); wards[w["id"]] = [cid]
    # ---- gates: a passage through the wall (at a gatehouse tower — through the tower)
    gaps = []; gp_cells = []
    for i, g in enumerate(R["gates"]):
        p = np.asarray(g["xy"], float); o = np.asarray(g["out"], float); d = np.asarray(g["dir"], float); depth = wall_t * 1.25
        if g.get("kind") == "tower" and g.get("r"): depth = max(depth, 2 * float(g["r"]) * 0.95)
        half_ = float(g["r"]) * 1.05 if g.get("kind") == "tower" else float(g["w"]) / 2 + 1.4; gaps.append((p, half_))
        wd = float(g["w"]); cid = f"gate{i}"
        S.cell(cid, "gate passage", [p + o * depth / 2 - d * wd / 2, p + o * depth / 2 + d * wd / 2, p - o * depth / 2 + d * wd / 2, p - o * depth / 2 - d * wd / 2],
               float(g["z"]), float(g["z"]) + float(g["h"]), "covered", use="gate", access="gate", why=f"gate {g.get('kind', '')}")
        gp_cells.append((i, cid, p, o, depth, wd, float(g["z"]), float(g["h"])))
    # ---- bridges (over a moat, a chasm) — the deck
    br_cells = []
    for i, b in enumerate(R.get("bridges") or []):
        a_, b2 = np.asarray(b["pts"][0], float), np.asarray(b["pts"][-1], float); dd = (b2 - a_) / max(np.linalg.norm(b2 - a_), 1e-9); tg = np.array([-dd[1], dd[0]]); w_ = float(b["w"])
        cid = f"bridge{i}"; S.cell(cid, "bridge", [a_ - tg * w_ / 2, b2 - tg * w_ / 2, b2 + tg * w_ / 2, a_ + tg * w_ / 2], float(b["deck"]), None, "exterior", edge="rail", use="approach",
                                   why="bridge over a chasm" if b.get("kind") == "chasm" else "bridge over a moat"); br_cells.append((cid, a_, b2, w_, float(b["deck"])))
    # ---- wall walkways: segments between towers, gates and "house-walls"
    TW = R["towers"]; walks = []
    for wi, w in enumerate(R["walls"]):
        if w["kind"] == "retaining": continue
        P, N, wk, bs, T = wall_frame(w); n = len(P); keep = np.ones(n, bool)
        for gp, hh in gaps: keep &= np.hypot(P[:, 0] - gp[0], P[:, 1] - gp[1]) > hh
        for t in TW: keep &= np.hypot(P[:, 0] - t["xy"][0], P[:, 1] - t["xy"][1]) > float(t["r"]) + 0.3
        if w.get("absorbed"): keep[np.asarray(w["absorbed"], int)] = False
        runs = []; cur = []
        for i in range(n):
            if keep[i]: cur.append(i)
            elif cur: runs.append(cur); cur = []
        if cur: runs.append(cur)
        if w["closed"] and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1: runs[0] = runs[-1] + runs[0]; runs.pop()
        if not w.get("steps"):                                                    # stepped walkway on a steep stretch (like the build): a 1.1 m step
            stp = np.zeros(n, bool)
            for f in R.get("features", []):
                if f.get("wall") == wi and f["kind"] == "stepped_walk" and "ii" in f: stp[f["ii"]] = True
            wk = np.where(stp, np.floor(wk / 1.1) * 1.1 + 0.55, wk)
        cov_ii = {j for f in R.get("features", []) if f.get("wall") == wi and f["kind"] == "covered_walk" for j in f.get("ii", [])}   # covered walk
        k = 0
        for r_ in runs:                                                           # a piece of walkway is almost level (±0.8 m); between the pieces — steps
            if len(r_) < 2: continue
            chunks = [[r_[0]]]
            for j in r_[1:]:
                cur_ = chunks[-1]
                if max(wk[cur_ + [j]]) - min(wk[cur_ + [j]]) > 0.8 and len(cur_) >= 2: chunks.append([cur_[-1], j])
                else: cur_.append(j)
            prev = None
            for ch in chunks:
                Q = P[ch]; Nn = N[ch]; tt = T[ch]; inner = Q - Nn * (tt / 2)[:, None]; outer = Q + Nn * (tt / 2 - pt)[:, None]
                if len(ch) < 2 or float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))) < 0.6: continue
                poly = list(inner) + list(outer[::-1]); cid = f"walk{wi}_{k}"; k += 1; z0_ = float(np.min(wk[ch])); cov = np.mean([j in cov_ii for j in ch]) > 0.5
                S.cell(cid, "covered walk" if cov else "wall walkway", poly, z0_, z0_ + 2.4 if cov else None, "covered" if cov else "exterior", edge="parapet", use="defense",
                       access="defense", why=f"walkway {z0_:.1f}–{float(np.max(wk[ch])):.1f} m" + (" under a roof" if cov else ""))
                walks.append((cid, wi, ch, Q, Nn, tt, wk[ch]))
                if prev is not None:                                              # steps between the pieces of walkway
                    q_ = P[ch[0]]; dz_ = z0_ - S.cells[S.idx[prev]]["z0"]
                    S.link(prev, cid, "stair", [q_ - (P[ch[1]] - q_) * 0.5, q_ + (P[ch[1]] - q_) * 0.5], S.cells[S.idx[prev]]["z0"], z0_, float(tt[0]) - pt,
                           length=max(abs(dz_) / STAIR, 0.5), hidden=abs(dz_) > 0.4, why="stepped walkway")
                prev = cid
    # ---- towers: the body (mass), the platform (if the roof is flat); the stair inside runs from the court to
    # the wall walkways and to the platform
    tops = []
    for i, t in enumerate(TW):
        F_ = footprint(t["plan"], t["xy"], max(float(t["r"]), 1.5), t.get("dir")); cid = f"tower{i}"
        S.cell(cid, "tower", F_, float(t["base"]), float(t["top"]), "void", walk=False, why="tower body (solid in the model)")
        if t.get("roof", "flat") == "flat" or t.get("butterfass"):
            S.cell(f"{cid}_top", "tower platform", F_, float(t["top"]), None, "exterior", edge="parapet", use="defense", access="defense"); tops.append((f"{cid}_top", t))
    for cids in wards.values():                                                  # towers and the main tower in a court are holes (the court's own place is around them)
        for cid in cids:
            c_ = S.cells[S.idx[cid]]; Pw = np.asarray(c_["poly"], float)
            for t in TW + ([R["keep"]] if R.get("keep") else []):
                F_ = footprint(t["plan"], t["xy"], max(float(t["r"]), 1.5), t.get("dir") or R.get("threat"))
                if bool(inside(Pw, F_[:, 0], F_[:, 1]).all()): c_.setdefault("holes", []).append([[round(float(x), 2), round(float(y), 2)] for x, y in F_[::-1]])
    kp = R.get("keep")
    if kp:
        F_ = footprint(kp["plan"], kp["xy"], max(float(kp["r"]), 1.5), R.get("threat")); S.cell("keep", "main tower", F_, float(kp["base"]), float(kp["top"]), "void", walk=False,
                                                                                             why="entrance high up, by a ladder (not in the model)")
        if kp.get("roof", "flat") == "flat": S.cell("keep_top", "main tower platform", F_, float(kp["top"]), None, "exterior", walk=False, edge="parapet")
    # ---- buildings: solid bodies in the model; a door is a dark face on the facade toward the court
    doors = []
    for i, b in enumerate(R["buildings"]):
        Rb = np.asarray(b["poly"], float); cid = f"bld{i}"
        S.cell(cid, RU.get(b["kind"], b["kind"]), Rb, float(b["floor"]), float(b["eave"]), "interior", use=b["kind"], access="private",
               why="a solid body in the model, doors and windows are dark faces"); S.cells[-1]["solid"] = True; S.cells[-1]["zb"] = round(float(b["base"]), 2)
        d = np.asarray(b["axis"], float); n_in = np.array([-d[1], d[0]]); doors.append((cid, (Rb[2] + Rb[3]) / 2, n_in, float(b["floor"])))
    # ---- transitions: gates (the "inward" side is where the gate's own court is; a narrow outer court — the near
    # points)
    def side_of(q, o_):
        for dist in (0.4, 1.0, 2.0, 3.0):
            w_ = ward_at(q + o_ * dist)
            if w_: return w_
        return None
    for i, cid, p, o, depth, wd, zg, hg in gp_cells:
        g = R["gates"][i]; ea, eb = p - o * depth / 2, p + o * depth / 2; wa, wb = side_of(ea, -o), side_of(eb, o)
        own = lambda w_: w_ is not None and w_.startswith(f"ward_{g['ward']}")
        if own(wb) and not own(wa): ea, eb, wa, wb, o = eb, ea, wb, wa, -o           # the plan's "outward" faces the court — by geometry
        qi, qo = ea, eb; inn = wa or at(qi - o * 1.2, zg + 0.6, skip=S.idx[cid]); S.relate(cid, inn, "open")
        S.link(cid, inn, "opening", qi, zg, S.cells[S.idx[inn]]["z0"] if inn != "outside" else zg, wd, hg, n=-o, why="gate — into the court")
        brg = [b_[0] for b_ in br_cells if min(np.linalg.norm(qo - b_[1]), np.linalg.norm(qo - b_[2])) < 3.0]
        outc = brg[0] if brg else (wb or "outside"); S.relate(cid, outc, "open")
        if outc == "outside" and g.get("foot") is not None:                      # outside: a ramp from the foot to the threshold
            foot = np.asarray(g["foot"], float); zf = ground(foot, zg - float(g.get("ramp", 0.0)) * float(g.get("ramp_m", 0.0)))
            if np.linalg.norm(foot - qo) > 0.8:
                S.link("outside", cid, "ramp", [foot, qo], zf, zg, wd, ground=ground(foot - o * 0.5, zf), n=-o, why="approach to the gate — a ramp")
            else: S.link("outside", cid, "opening", qo, zf, zg, wd, hg, n=-o, ground=zf, why="gate threshold")
        else:
            S.link(outc, cid, "opening", qo, S.cells[S.idx[outc]]["z0"] if outc != "outside" else zg, zg, wd, hg, n=-o, why="gate — from the bridge or from the outer court")
    for cid, a_, b2, w_, z in br_cells:                                          # bridge: the far end — onto the ground, the near end — to the gate
        ends = sorted([(min(np.linalg.norm(q - np.asarray(g["xy"])) for g in R["gates"]), q) for q in (a_, b2)], key=lambda e: e[0])
        far = ends[1][1]; near = ends[0][1]; d_ = (near - far) / max(np.linalg.norm(near - far), 1e-9)
        g0 = max(ground(far - d_ * 1.5, z), ground(far - d_ * 3.0, z)); S.link("outside", cid, "opening", far, g0, z, w_, n=d_, ground=g0, why="bridge — from the bank")
    # ---- stairs to the wall (the "stair" element) and terrace stairs
    for f in R.get("features", []):
        if f["kind"] != "stairs": continue
        w = R["walls"][f["wall"]]; P, N, wk, bs, T = wall_frame(w); i = int(f["i"]); p, n_, t = P[i], N[i], float(T[i])
        zi = ground(p - n_ * (t / 2 + 1.5), float(wk[i]) - 3.0); dz = float(wk[i]) - zi
        if not 1.5 <= dz <= 10.5: continue
        hit = [wc for wc in walks if wc[1] == f["wall"] and i in wc[2]]
        if not hit: continue
        tg = np.array([-n_[1], n_[0]]) * (1 if f.get("left") else -1); L_ = math.ceil(dz / 0.2) * 0.3; top = p - n_ * (t / 2 + 0.65); foot = top - tg * L_
        lo = ward_at(foot) or at(foot, zi + 0.6, walk_only=True)
        if lo != "outside": S.link(lo, hit[0][0], "stair", [foot, top, p - n_ * (t / 2 - 0.6)], zi, float(wk[i]), 1.3, ground=None, why="stair from the court to the wall walkway")
    for w in R["walls"]:
        if w["kind"] != "retaining" or w.get("stair") is None: continue
        P, N, wk, bs, T = wall_frame(w); j = int(w["stair"]); n_ = N[j]; tg = np.array([-n_[1], n_[0]]); t = float(T[j])
        zl = ground(P[j] + n_ * (t / 2 + 1.5), float(wk[j]) - 3.0); dz = float(wk[j]) - zl
        top = P[j] + tg * 2.0 + n_ * (t / 2 + 0.8); foot = top - tg * math.ceil(max(dz, 0.2) / 0.2) * 0.3
        lo = ward_at(P[j] + n_ * (t / 2 + 1.5)) or "outside"; hi = ward_at(P[j] - n_ * (t / 2 + 1.5)) or "outside"
        if lo != hi and "outside" not in (lo, hi): S.link(lo, hi, "stair", [foot, top, P[j] + tg * 2.0 - n_ * (t / 2 + 0.6)], zl, float(wk[j]), 1.6, why="stair from terrace to terrace")
    # ---- stairs in towers: from the court — to the walkways that abut the tower, and to the platform (not built
    # in the model)
    for i, t in enumerate(TW):
        c = np.asarray(t["xy"], float); r = float(t["r"]); wid = [wc for wc in walks if min(np.linalg.norm(wc[3][0] - c), np.linalg.norm(wc[3][-1] - c)) <= r + 3.0]
        yard = None
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):              # court at the foot of the tower
            q = c + (r + 1.5) * np.array([math.cos(ang), math.sin(ang)]); cand = ward_at(q)
            if cand: yard = (cand, q); break
        if yard is None: continue
        zy = S.cells[S.idx[yard[0]]]["z0"]
        for wc in wid:
            end = wc[3][0] if np.linalg.norm(wc[3][0] - c) <= np.linalg.norm(wc[3][-1] - c) else wc[3][-1]; zw = S.cells[S.idx[wc[0]]]["z0"]
            S.link(yard[0], wc[0], "spiral", [yard[1], c, end], zy, zw, 0.9, length=abs(zw - zy) / STAIR + r, hidden=True, why="the tower's stair and doors are not built in the model")
        if f"tower{i}_top" in S.idx:
            S.link(yard[0], f"tower{i}_top", "spiral", [yard[1], c], zy, float(t["top"]), 0.9, length=abs(float(t["top"]) - zy) / STAIR, hidden=True, why="crawl hole to the tower platform")
    for i, cid, p, o, depth, wd, zg, hg in gp_cells:                             # gatehouse tower: a stair to the walkways that abut the gate
        yard = ward_at(p - o * (depth / 2 + 1.5))
        if yard is None: continue
        for wc in walks:
            for end in (wc[3][0], wc[3][-1]):
                if np.linalg.norm(end - p) <= gaps[i][1] + 3.0:
                    zw = S.cells[S.idx[wc[0]]]["z0"]; zy = S.cells[S.idx[yard]]["z0"]
                    S.link(yard, wc[0], "spiral", [p - o * (depth / 2 + 1.5), p, end], zy, zw, 0.9, length=abs(zw - zy) / STAIR + 2.0, hidden=True,
                           why="the gatehouse tower's stair is not built in the model"); break
    # ---- building doors (decoys) — from the court
    for cid, pm, n_in, fl in doors:
        yard = ward_at(pm + n_in * 1.5) or "outside"
        if yard != "outside": S.link(yard, cid, "door", pm, S.cells[S.idx[yard]]["z0"], fl, 1.4, 2.4, n=-n_in, model="patch", why="door — a dark face (the passage is not cut)")
    # ---- anchors
    if R.get("road"): r0 = R["road"][0]; S.anchor("start of the approach", "outside", r0[:2], r0[2], tags=["approach"])
    for i, cid, p, o, depth, wd, zg, hg in gp_cells:
        S.anchor(f"gate {i} outside", "outside", p + o * (depth / 2 + 4.0), zg, facing=-o, tags=["entrance"]); S.anchor(f"gate {i} inside", cid, p, zg, facing=-o)
    for wid_, cids in wards.items():
        for cid in cids:
            c_ = S.cells[S.idx[cid]]; from components.buildings.space.space import inner_point
            S.anchor(f"court {wid_}", cid, inner_point(c_["poly"], holes=c_.get("holes")), c_["z0"])
    for cid, t in tops[:4]: S.anchor(f"top {cid}", cid, t["xy"], float(t["top"]), tags=["overview"])
    S.auto_bounds()
    return S


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:] if "=" in x); out = os.path.abspath(a[0])
    R = json.load(open(os.path.join(out, "castle.json")))
    from core.surface.ground import Ground
    S = spaces(R, Ground(out).height); D = S.to_dict(); R["spaces"] = D
    json.dump(R, open(os.path.join(out, "castle.json"), "w"), ensure_ascii=False)
    if kw.get("draw"): S.draw(os.path.join(out, "space_plan.png"), title=os.path.basename(out))
    print("__RESULT__", json.dumps({"stats": D["stats"], "non_passing": [c["what"] for c in D["checks"] if not c["ok"]],
                                    "unreachable": next((c.get("unreachable") for c in D["checks"] if c["what"].startswith("each")), None)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
