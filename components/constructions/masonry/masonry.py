"""Masonry by a structure's plan: the layout of stones on the unfolded faces.

The basis is Legakis, Dorsey, Gortler 2001 (Feature-Based Cellular Texturing): the features first (corners, openings, arches), then the faces in rows;
rows pass through the key levels; a row's joint is at least lap away from the joints of the row below; mortar is not geometry (a stone = a cell shrunk by
half a joint). Here it is a vector layout for large structures (a castle wall hundreds of metres long): rows × segments, with no raster over the whole face;
the raster is only for mask maps, by pieces and with a texel for the camera.

A wall side is a strip along the path (`Strip`): u is the arc length along the wall axis, v is the absolute height. The bottom of the strip is the ground or
rock (base by stations), the top is the wall walk or the top of the parapet (a stair: two stations at one point make a step). Rows are horizontal by absolute
height (on a slope the lower rows are cut by the ground in steps, like a foundation); at a path kink steeper than 20° there is a joint (a stone does not bend
around a corner); a gate is an opening with "long-short" reveals and an arch of voussoirs.

Masonry packs (PACKS) are a face type and sizes; the outside and the inside have their own (ashlar in rows / rubble in rows).

    import masonry as M
    lay = json.load(open("surfaces.json")); S = M.strip_from(lay["items"][0]["strip"])   # a wall side from the surfaces layer
    L = M.layout(S, M.PACKS["ashlar"], gaps=lay["items"][0].get("gaps", []), seed=1)   # → dict(courses, stones, stats)
    .venv/bin/python -m components.constructions.masonry.masonry <surfaces.json> [wall=0] [out=folder]      # unfold the outer side → PNG + summary
"""
import json
import math
import os
import sys

import numpy as np

from core.surface.strip import BEND, CORNER, FOLD, Strip, strip_from   # noqa: F401 — side unrolling: shared geometry of the surfaces layer

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "packs.yaml")                                          # knowledge lives next to the code (the Constructions component)


def load_library(path=LIB):
    """Masonry knowledge (library/masonry/packs.yaml): packs (base + own; name, title), substances (tone), parts (a pack per
    building part). Needs PyYAML — the project's Python; in Blender the packs arrive in the request JSON (resolve / build)."""
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    base, packs = d["base"], {}
    for name, kw in d["packs"].items():
        pk = dict(base); pk.update({k: v for k, v in kw.items() if k != "title"}); pk["name"] = name; pk["title"] = kw.get("title", name)
        packs[name] = pk
    return packs, d


try:
    PACKS, LIBRARY = load_library()
except ImportError:                                                              # Blender's Python: no YAML, the packs come from the request
    PACKS, LIBRARY = {}, {}
MARGIN = 1.5                                       # margin of the segment map at the edges, m
SAMPLE = 0.02                                      # profile sampling step along u, m


# ── wall side: a strip along the path ───────────────────────────────────────────────────────────────────────────
def course_levels(z0, z1, keys, lo, hi, rng, first=None):
    """Courses from z0 to z1 through the key levels keys: the number of courses by the mean height, heights random and
    normalized so that a course boundary lies exactly on a key level; the first course is the plinth of height first."""
    levels = sorted({z0, z1, *[k for k in keys if z0 + lo * 0.6 < k < z1 - lo * 0.6]})
    zs = [levels[0]]
    if first: zs.append(levels[0] + first)
    for a, b in zip(levels[:-1], levels[1:]):
        a = max(a, zs[-1])
        if b - a < lo * 0.6: continue
        n = max(1, int(round((b - a) / ((lo + hi) / 2))))
        w = rng.uniform(lo, hi, n); w *= (b - a) / w.sum()
        zs += list(a + np.cumsum(w))
    return np.array(zs)


def fill_row(u0, u1, lens, lap, prev, rng, forced=()):
    """Stones of a course on [u0, u1]: lengths from lens, a joint no closer than lap to the joints of the course below;
    forced joints (path kinks) are mandatory; a tail shorter than 0.6 of the minimum length is added to the last stone.
    → [(a, b)]."""
    stops = sorted(f for f in forced if u0 + 0.05 < f < u1 - 0.05) + [u1]
    out, u = [], u0
    for stop in stops:
        while u < stop - 1e-6:
            j = u + rng.uniform(*lens)
            for _ in range(8):
                near = [p for p in prev if abs(p - j) < lap]
                if not near: break
                j = max(near) + lap + rng.uniform(0.0, 0.05)
            if stop - j < lens[0] * 0.6: j = stop
            j = min(j, stop); out.append((u, j)); u = j
    return out


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def layout(S, pack, gates=(), seed=1, putlogs=True, gaps=(), openings=()):
    """Layout of side S: framings (gate with an arch and jambs, quoins at the corners, arrow-slit openings) by framing
    courses → filling the free area according to the masonry kind pack["kind"]:
      courses — courses at absolute height through the threshold and the gate springers, stones with bonding (ashlar,
                rustication, rubble in courses, dry-stone, megablocks); the bottom steps along the ground, the top follows
                the run of the wall; putlog holes are gaps in the courses;
      brick   — brick with the bond pack["bond"] (stretcher, English, Flemish, monk, garden-wall, Sussex);
      banded  — stone with brick lacing belts; herring — "herringbone" with flat lacing courses;
      skyline — random ashlar (stones 1–5 small courses high); cells — cells: rubble without courses, polygonal,
                cyclopean (large blocks + small stone in the gaps), Japanese; grid45 — Roman lattice between brick belts.
    Stones are polygons in (u, v) with a kind, a material and a depth."""
    rng = np.random.default_rng(seed)
    us = np.arange(0.0, S.L + SAMPLE / 2, SAMPLE)
    b_u, t_u = S.base_at(us), S.top_at(us)
    zmin, zmax = float(b_u.min()), float(t_u.max())
    kind = pack.get("kind", "courses")
    keys = []
    for g in gates:                                                               # threshold, springers and the course above the arch keystone
        keys += [g["z"], g["z"] + g["h"] - g["w"] / 2, g["z"] + g["h"] + pack.get("ring", 0.45) + max(0.12, 0.3 * pack["course"][0])]
    if kind == "brick":                                                           # brick course: height + joint, exactly
        hc = pack["brick"][2] + pack["joint"]; zs = np.arange(zmin, zmax + hc * 0.5, hc)
    elif kind == "banded":
        zs, band = banded_levels(zmin, zmax, pack, rng); zs = _snap_levels(zs, band, keys, pack["course"][0])
    else:
        zs = course_levels(zmin, zmax, keys, *pack["course"], rng, first=pack.get("plinth"))
    if kind != "banded": band = ["stone"] * (len(zs) - 1)
    waves = {k: (rng.uniform(*pack.get("wave", (0.0, 0.0))), rng.uniform(1.6, 3.2), rng.uniform(0, 6.3)) for k in range(len(zs))}

    def wave(k, u):
        if k <= 0 or k >= len(zs) - 1 or pack.get("wave", (0, 0))[1] <= 0: return 0.0
        a, lam, ph = waves[k]; return a * math.sin(2 * math.pi * u / lam + ph)

    stones, feats = [], []
    occ = [(a, b, -1e9, 1e9) for a, b in gaps]                                  # occupied: gaps and features (u0, u1, z0, z1)
    dz_ = pack.get("dress_course", None)                                          # framing courses for kinds without courses
    zd = course_levels(zmin, zmax, keys, *dz_, rng) if dz_ else zs
    # gate: the opening, "long-and-short" jambs by courses, arch of voussoirs (odd count, keystone in the middle)
    for g in gates:
        r = g["w"] / 2; zs_ = g["z"] + g["h"] - r; ring = pack.get("ring", 0.45); re = r + ring; uc = g["u"]
        occ.append((uc - r, uc + r, g["z"] - 0.01, zs_ + r))                      # gate passage
        k0 = int(np.searchsorted(zd, g["z"] - 1e-6)); k1 = int(np.searchsorted(zd, zs_ - 1e-6))
        for k in range(k0, k1):
            lj, sj = pack["jamb"]; aL, aR = (lj, sj) if (k - k0) % 2 == 0 else (sj, lj)
            for u0_, u1_ in ((uc - r - aL, uc - r), (uc + r, uc + r + aR)):
                feats.append(dict(kind="jamb", poly=[(u0_, zd[k]), (u1_, zd[k]), (u1_, zd[k + 1]), (u0_, zd[k + 1])], course=k))
                occ.append((u0_, u1_, zd[k], zd[k + 1]))
        n = int(round(math.pi * (r + re) / 2 / pack.get("voussoir", 0.32))); n += (n + 1) % 2
        for i in range(n):
            a0, a1 = math.pi - i * math.pi / n, math.pi - (i + 1) * math.pi / n
            arc = [(uc + r * math.cos(a), zs_ + r * math.sin(a)) for a in np.linspace(a0, a1, 4)]
            ext = [(uc + re * math.cos(a), zs_ + re * math.sin(a)) for a in np.linspace(a1, a0, 4)]
            feats.append(dict(kind="key" if i == n // 2 else "voussoir", poly=arc + ext, course=k1))
        occ.append(("arch", uc, zs_, re))
        if g.get("sill") and k0 > 0:                                              # window: the sill is a through-stone under the opening
            ext = min(pack["jamb"][1], 0.15); s0 = zd[k0 - 1]
            if kind in ("cells", "skyline", "grid45"): s0 = max(s0, zd[k0] - 0.3)   # no courses: not higher than 0.3 m
            feats.append(dict(kind="sill", poly=[(uc - r - ext, s0), (uc + r + ext, s0), (uc + r + ext, zd[k0]), (uc - r - ext, zd[k0])], course=k0 - 1))
            occ.append((uc - r - ext, uc + r + ext, s0, zd[k0]))
    # contour corners: the quoin wraps the corner, long-and-short on alternate courses (by framing courses)
    for uc in S.corners:
        for k in range(len(zd) - 1):
            lq, sq = pack.get("quoin", (0.9, 0.5)); a_, b_ = (lq, sq) if k % 2 == 0 else (sq, lq)
            a_ += rng.uniform(-0.05, 0.05) * (lq / 0.9)
            z0, z1 = zd[k], zd[k + 1]
            if float(S.top_at(uc)) < z0 + 0.06 or float(S.base_at(uc)) > z1 - 0.06: continue
            lo_, hi_ = max(z0, float(S.base_at(uc))), min(z1, float(S.top_at(uc)))
            lim_a = lim_b = 1e9; arc_a = arc_b = 1e9
            for o in occ:                                                          # the quoin does not overlap neighboring framings
                if o[0] == "arch":
                    _, ua_, zc_, re_ = o; ob = (ua_ - re_, ua_ + re_, zc_, zc_ + re_)
                else:
                    ob = o
                if ob[2] < z1 - 0.01 and ob[3] > z0 + 0.01:
                    if ob[1] <= uc + 1e-6:
                        if o[0] == "arch": arc_a = min(arc_a, uc - ob[1] - pack["joint"])
                        else: lim_a = min(lim_a, uc - ob[1] - pack["joint"])
                    elif ob[0] >= uc - 1e-6:
                        if o[0] == "arch": arc_b = min(arc_b, ob[0] - uc - pack["joint"])
                        else: lim_b = min(lim_b, ob[0] - uc - pack["joint"])
            a_ = lim_a if lim_a - a_ < 0.18 else a_                                # gap to the framing under 18 cm: the quoin
            b_ = lim_b if lim_b - b_ < 0.18 else b_                                # reaches it (otherwise a sliver is left in the gap);
            a_, b_ = min(a_, arc_a), min(b_, arc_b)                                # not up to the arch: there is a spandrel at the springers
            if a_ < 0.08 or b_ < 0.08: continue
            poly = [(uc - a_, lo_), (uc, lo_), (uc + b_, lo_), (uc + b_, hi_), (uc, hi_), (uc - a_, hi_)]
            feats.append(dict(kind="quoin", poly=poly, course=k, depth=pack.get("quoin_d", 0.45))); occ.append((uc - a_, uc + b_, z0, z1))
    # arrow slits and windows: opening, 0.22 m jambs, lintel over the opening
    for o in openings:
        u0_, u1_, z0_, z1_ = o["u"] - o["w"] / 2, o["u"] + o["w"] / 2, o["z"] - o["h"] / 2, o["z"] + o["h"] / 2
        occ.append((u0_ - 0.22, u1_ + 0.22, z0_ - 0.12, z1_ + 0.3))
        for a_, b_ in ((u0_ - 0.22, u0_), (u1_, u1_ + 0.22)):
            feats.append(dict(kind="jamb", poly=[(a_, z0_ - 0.12), (b_, z0_ - 0.12), (b_, z1_), (a_, z1_)], course=-1, depth=0.4))
        feats.append(dict(kind="sill", poly=[(u0_, z0_ - 0.12), (u1_, z0_ - 0.12), (u1_, z0_), (u0_, z0_)], course=-1, depth=0.4))
        feats.append(dict(kind="lintel", poly=[(u0_ - 0.22, z1_), (u1_ + 0.22, z1_), (u1_ + 0.22, z1_ + 0.3), (u0_ - 0.22, z1_ + 0.3)],
                          course=-1, depth=0.4))

    if kind == "skyline":                                                         # arch spandrels: two or three stones dressed to the arc:
        un = pack.get("unit", 0.11); j = pack["joint"]                            # otherwise the horizon above the arc becomes a staircase of slivers
        for o in [o for o in occ if o[0] == "arch"]:
            _, uc, zc, re = o; zt = zc + re + un; zm = zc + max(2, int(round(0.5 * re / un))) * un
            for sg in (-1, 1):
                e = 0.25
                for o2 in occ:                                                    # the spandrel does not overlap neighboring framings
                    if o2[0] == "arch" or not (o2[2] < zt - 0.01 and o2[3] > zc + 0.01): continue
                    if sg < 0 and o2[1] <= uc - re + 1e-6: e = min(e, uc - re - o2[1] - j)
                    if sg > 0 and o2[0] >= uc + re - 1e-6: e = min(e, o2[0] - uc - re - j)
                e = max(e, 0.0); ua, ub = sorted((uc, uc + sg * (re + e)))
                for za, zb in (((zc, zm), (zm, zt)) if e >= 0.08 else ((zc, zt),)):
                    q = _arch_cut([(ua, za), (ub, za), (ub, zb), (ua, zb)], [(uc, zc, re)])
                    if q: stones.append(dict(kind="field", poly=q, course=-1, d=[0.0, float(rng.uniform(*pack["face_d"]))],
                                             tint=round(float(rng.normal(1.0, pack.get("tint", 0.07))), 3), mat=pack.get("mat", "stone")))
                occ.append((ua, ub, zc, zt))

    def occ_mask(U, V):                                                           # occupied by features (vectorized)
        m = np.zeros(np.broadcast(U, V).shape, bool)
        for o in occ:
            if o[0] == "arch":
                _, uc, zc, re = o; m |= (np.hypot(U - uc, V - zc) < re) & (V >= zc)
            else:
                m |= (U > o[0]) & (U < o[1]) & (V > o[2]) & (V < o[3])
        return m

    holes = []
    if kind in ("courses", "brick", "banded", "herring"):
        holes = _fill_courses(S, pack, zs, band, us, b_u, t_u, occ, wave, stones, rng, putlogs)
    elif kind == "skyline":
        _fill_skyline(S, pack, us, b_u, t_u, occ_mask, stones, rng)
    elif kind == "cells":
        _fill_cells(S, pack, b_u, t_u, occ_mask, stones, rng)
    elif kind == "grid45":
        _fill_grid45(S, pack, us, b_u, t_u, occ, stones, rng)
    for f in feats:                                                              # through features (gate): −1 means to the back; quoins and jambs have their own depth
        f.update(d=[0.0, f.pop("depth", -1.0)], tint=round(float(rng.normal(1.05, 0.04)), 3), mat=pack.get("dress_mat", pack.get("mat", "stone")))
    area = sum(abs(_area(np.asarray(s["poly"]))) for s in stones)
    face_area = float(np.trapezoid(np.maximum(t_u - b_u, 0), us))
    return dict(courses=np.round(zs, 4).tolist(), stones=stones, features=feats, holes=holes, gates=list(gates), gaps=list(gaps),
                openings=list(openings), pack=pack.get("name", ""),
                stats=dict(stones=len(stones), features=len(feats), putlogs=len(holes), courses=len(zs) - 1,
                           L=round(S.L, 2), face_m2=round(face_area, 1), stone_m2=round(area, 1),
                           bends=len(S.bends), corners=len(S.corners), openings=len(openings)))


def banded_levels(zmin, zmax, pack, rng):
    """Bands: brick belts of nb courses (brick height + joint) every band_every meters of stone; stone between the belts
    in courses taken from pack["course"]. → (course levels, course kind: stone | brick)."""
    zs, kinds = [zmin], []
    hb = pack["brick"][2] + pack["joint"]; z = zmin
    while z < zmax - 0.05:
        span = min(rng.uniform(*pack["band_every"]), zmax - z)
        n = max(1, int(round(span / np.mean(pack["course"])))); w = rng.uniform(*pack["course"], n); w *= span / w.sum()
        for x in w: z += x; zs.append(z); kinds.append("stone")
        for _ in range(pack["band_n"]):
            if z + hb > zmax + 1e-6: break
            z += hb; zs.append(z); kinds.append("brick")
    return np.array(zs), kinds


def _snap_levels(zs, kinds, keys, lo):
    """Bands through key levels (threshold, springers and top of the arch): the nearest boundary between two stone courses
    is moved onto the key level if the neighboring courses stay at least 8 cm thick; brick belts are left alone."""
    zs = np.array(zs, float)
    for key in keys:
        cand = [i for i in range(1, len(zs) - 1) if kinds[i - 1] == "stone" and kinds[i] == "stone"]
        if not cand: continue
        i = min(cand, key=lambda i: abs(zs[i] - key))
        if abs(zs[i] - key) < 0.6 * lo and zs[i - 1] + 0.08 < key < zs[i + 1] - 0.08: zs[i] = key
    return zs


def bond_units(bond, k, L_, W_, j):
    """Units of course k of a brick bond: the template [(kind, length)] and the course phase shift (m). S — stretcher, H — header."""
    S_, H_ = ("S", L_), ("H", W_)
    if bond == "stretcher": return [S_], (0.0 if k % 2 == 0 else (L_ + j) / 2)
    if bond == "header": return [H_], (0.0 if k % 2 == 0 else (W_ + j) / 2)
    if bond == "english": return ([S_], 0.0) if k % 2 == 0 else ([H_], -(L_ / 2 - W_ / 2))
    if bond == "garden": return ([H_], 0.0) if k % 4 == 3 else ([S_], ((L_ + j) / 2) * (k % 2) + (W_ + j) / 2)
    if bond == "flemish": return [H_, S_], (0.0 if k % 2 == 0 else j / 2 + (L_ + W_) / 2 + j / 2)
    if bond == "monk": return [H_, S_, S_], (0.0 if k % 2 == 0 else (W_ + j) / 2 + (L_ + j) + (W_ + j) / 2 - (W_ + j) / 2)
    if bond == "sussex": return [H_, S_, S_, S_], (0.0 if k % 2 == 0 else (W_ + j) + (L_ + j) + (L_ + j) / 2 - (W_ + j) / 2)
    return [S_], 0.0


def _shape_cut(a, b, z0, z1, rng, k=(0.08, 0.32), top=0.22):
    """Rubble: a rectangle with cut corners and less than full height (an octagon)."""
    hh = z1 - z0; tp = z1 - rng.uniform(0, top) * hh; ww = b - a
    c = lambda: rng.uniform(*k) * min(hh, ww)
    c1, c2, c3, c4 = c(), c(), c(), c()
    return [(a + c1, z0), (b - c2, z0), (b, z0 + c2), (b, tp - c3), (b - c3, tp), (a + c4, tp), (a, tp - c4), (a, z0 + c1)]


def _clip(poly, u0, u1, v0=-1e9, v1=1e9):
    """Polygon ∩ rectangle [u0, u1] × [v0, v1] (Sutherland–Hodgman)."""
    P = [tuple(p) for p in poly]
    for ax, val, keep in ((0, u0, 1), (0, u1, -1), (1, v0, 1), (1, v1, -1)):
        if not P: break
        Q = []
        for i in range(len(P)):
            a, b = P[i - 1], P[i]
            ia, ib = (a[ax] - val) * keep >= 0, (b[ax] - val) * keep >= 0
            if ib:
                if not ia:
                    t = (val - a[ax]) / (b[ax] - a[ax]); Q.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                Q.append(b)
            elif ia:
                t = (val - a[ax]) / (b[ax] - a[ax]); Q.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        P = Q
    return P


def _simplify(Q, tol=0.0015):
    """Remove contour points lying on the straight line between their neighbors (deviation < tol)."""
    while len(Q) > 3:
        a, b = np.roll(Q, -1, 0) - np.roll(Q, 1, 0), Q - np.roll(Q, 1, 0)
        dist = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(np.hypot(a[:, 0], a[:, 1]), 1e-9)
        idx = np.flatnonzero(dist < tol)
        if not len(idx): break
        Q = np.delete(Q, idx[::2], 0)
    return Q


def _despike(Q, ang=40.0):
    """Cut off sharp spikes of the contour (counter-clockwise): a vertex with an interior angle < ang° is removed while any
    remain — a wedge where the straight edge of a stone meets the arc almost tangentially (otherwise the face chamfer -
    moving the vertex along the bisector by up to 3 chamfers — turns the thin wedge inside out: a dark line on the stone
    face)."""
    c = math.cos(math.radians(ang))
    while len(Q) > 3:
        a, b = np.roll(Q, 1, 0) - Q, np.roll(Q, -1, 0) - Q
        la, lb = np.hypot(a[:, 0], a[:, 1]), np.hypot(b[:, 0], b[:, 1])
        cosang = (a * b).sum(1) / np.maximum(la * lb, 1e-12)
        convex = (-a[:, 0]) * b[:, 1] - (-a[:, 1]) * b[:, 0] > 0
        bad = np.flatnonzero(convex & (cosang > c))
        if not len(bad): break
        Q = np.delete(Q, bad[:1], 0)
    return Q


def _arch_square(a, b, lo, hi, arches, pad=0.0):
    """Brick at an arch is not dressed (plinfa: the joint is as thick as the brick): the end is straight — at the widest point
    of the arc within the brick height, and the wedge up to the arc is mortar (along the arc a thin brick becomes a long
    wedge that _despike eats up). A brick across the arch axis (above the keystone) follows the arc. → (a, b) or None
    (not touched, or across the axis — decided by _arch_cut)."""
    hit = False
    for uc, zc, re in arches:
        R = re + pad
        if hi <= zc or lo >= zc + R: continue
        h = math.sqrt(max(R * R - (max(lo, zc) - zc) ** 2, 0.0))
        if b <= uc - h or a >= uc + h: continue
        if a < uc < b: return None
        if (a + b) / 2 < uc: b = min(b, uc - h)
        else: a = max(a, uc + h)
        hit = True
    return (a, b) if hit else None


def _arch_cut(poly, arches, pad=0.0, step=0.02, min_area=0.0025, min_thick=0.025):
    """A stone at an arch is dressed to its outer arc (this is how spandrel stones are cut): polygon minus the half-circle of
    the arch above the springers. The contour is densified; each entry into the circle is replaced by an arc from the angle
    of the entry's first point to the angle of its last — clockwise (the contour runs counter-clockwise: the circle is on
    the right). Radius + pad — for stones already shrunk by half a joint. arches — [(uc, zc, re)]. Sharp wedges are cut off
    (_despike); a remainder thinner than min_thick (2·area / perimeter) or smaller than min_area is a sliver, no stone.
    → polygon (list) or None."""
    P = np.asarray(poly, float)
    if _area(P) < 0: P = P[::-1]
    hit = [(uc, zc, re + pad) for uc, zc, re in arches
           if P[:, 0].min() < uc + re + pad and P[:, 0].max() > uc - re - pad and P[:, 1].max() > zc and P[:, 1].min() < zc + re + pad]
    if not hit: return poly
    Q = []
    for i in range(len(P)):                                                      # denser contour: entries into the circle are detected by points
        a, b = P[i], P[(i + 1) % len(P)]; n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / step)))
        Q.extend(a + (b - a) * t for t in np.arange(n) / n)
    Q = np.asarray(Q); moved = False
    for uc, zc, re in hit:
        d = Q - (uc, zc); m = (np.hypot(d[:, 0], d[:, 1]) < re - 1e-6) & (Q[:, 1] > zc - 1e-6)
        if not m.any(): continue
        if m.all(): return None
        sh = int(np.flatnonzero(~m)[0]); Q, m = np.roll(Q, -sh, 0), np.roll(m, -sh)  # start outside the circle
        out, i = [], 0
        while i < len(Q):
            if not m[i]: out.append(Q[i]); i += 1; continue
            j = i
            while j < len(Q) and m[j]: j += 1
            ang = lambda p: min(max(math.atan2(p[1] - zc, p[0] - uc), 0.0), math.pi)
            t1, t2 = ang(Q[i]), ang(Q[j - 1])
            if t1 < t2: t1, t2 = (t1, t1) if abs(t1 - t2) < 1e-6 else (t1, t2)
            k = max(1, int(math.ceil((t1 - t2) * re / step))) if t1 >= t2 else 0
            if t1 >= t2:
                out.extend(np.array([uc + re * math.cos(t), zc + re * math.sin(t)]) for t in np.linspace(t1, t2, k + 1))
            else:                                                                 # degenerate entry: points projected radially onto the arc
                dd = Q[i:j] - (uc, zc); rr = np.maximum(np.hypot(dd[:, 0], dd[:, 1]), 1e-9)[:, None]
                out.extend(np.array([uc, zc]) + dd / rr * re)
            i = j
        Q = np.asarray(out); moved = True
    if not moved: return poly
    keep = [Q[0]]
    for p in Q[1:]:
        if np.hypot(*(p - keep[-1])) > 0.002: keep.append(p)
    if len(keep) > 3 and np.hypot(*(keep[0] - keep[-1])) < 0.002: keep.pop()
    Q = _despike(_simplify(np.asarray(keep)))
    if len(Q) < 3: return None
    A_ = abs(_area(Q)); per = float(np.hypot(*np.diff(np.vstack([Q, Q[:1]]), axis=0).T).sum())
    if A_ < min_area or 2 * A_ / per < min_thick: return None
    return np.round(Q, 4).tolist()


def _cut_occ(poly, occ, pad=0.0, min_area=0.0015):
    """A stone next to occupied space (framings, openings, gaps): the rectangle cuts the stone with whichever of its sides
    leaves more stone; an arch cuts with its arc (_arch_cut). For kinds without courses (grid, belts in it).
    → polygon or None."""
    P = [tuple(p) for p in poly]
    for o in occ:
        if o[0] == "arch": continue
        q = np.asarray(P)
        if q[:, 0].max() <= o[0] + 1e-6 or q[:, 0].min() >= o[1] - 1e-6 or q[:, 1].max() <= o[2] + 1e-6 or q[:, 1].min() >= o[3] - 1e-6:
            continue
        cand = [c for c in (_clip(P, -1e9, o[0] - pad), _clip(P, o[1] + pad, 1e9), _clip(P, -1e9, 1e9, -1e9, o[2] - pad),
                            _clip(P, -1e9, 1e9, o[3] + pad, 1e9)) if len(c) >= 3]
        if not cand: return None
        P = max(cand, key=lambda c: abs(_area(np.asarray(c))))                   # the cut that leaves more stone
    q = _arch_cut(P, [(o[1], o[2], o[3]) for o in occ if o[0] == "arch"], pad=pad)
    if q is None or abs(_area(np.asarray(q, float))) < min_area: return None
    return np.round(np.asarray(q, float), 4).tolist()


def _fill_courses(S, pack, zs, band, us, b_u, t_u, occ, wave, stones, rng, putlogs):
    """Courses: free segments of a course (above the ground, below the top, not occupied) → stones by kind (ashlar and
    rubble — lengths with bonding, brick — the bond template, herringbone — inclined slabs); scaffold putlog holes are
    gaps. → putlog holes."""
    kind = pack.get("kind", "courses")
    putlog_rows = set()
    if putlogs and pack.get("putlog"):
        k = int(rng.integers(*pack["putlog"]["every"]))
        while k < len(zs) - 2:
            putlog_rows.add(k); k += int(rng.integers(pack["putlog"]["every"][0], pack["putlog"]["every"][1] + 1))
    prev, holes = [], []
    arches = [(o[1], o[2], o[3]) for o in occ if o[0] == "arch"]
    for k in range(len(zs) - 1):
        z0, z1 = zs[k], zs[k + 1]
        inside = (b_u < z1 - 0.03) & (t_u > z0 + 0.03)
        for o in occ:
            if o[0] == "arch":                                                    # the course goes around the arch at the narrowest place
                _, uc, zc, re = o                                                 # (at the top of the course), the rest — stones dressed to the arc
                if z1 > zc + 1e-6 and zc - 1e-6 <= z0 < zc + re - 1e-6:          # a course across the springers is not forbidden: jambs below the springers
                    zt = min(z1 - min(0.06, 0.35 * (z1 - z0)), zc + re)             # next to the arc a stone is at least 6 cm thick (no slivers)
                    half = math.sqrt(max(re * re - (zt - zc) ** 2, 0.0)) if zt > zc else re
                    if half > 1e-3: inside &= ~((us > uc - half) & (us < uc + half))
            elif o[2] < z1 - 0.01 and o[3] > z0 + 0.01:
                inside &= ~((us > o[0] + 1e-6) & (us < o[1] - 1e-6))
        joints = []
        bk = band[k] if k < len(band) else "stone"
        for c0, c1 in runs(inside):
            u0, u1 = us[c0], us[min(c1, len(us) - 1)]
            if u1 - u0 < 0.04: continue
            if kind == "brick" or bk == "brick":                                   # brick: the bond template with the course phase
                L_, W_, _h = pack["brick"]; j = pack["joint"]
                units, ph = bond_units(pack.get("bond", "stretcher") if kind == "brick" else "stretcher", k, L_, W_, j)
                per = sum(l + j for _, l in units); x = -ph % per - per; row = []; i = 0; m_of = {}
                while x < u1:
                    nm, l = units[i % len(units)]; a, b = x, x + l
                    if b > u0 + 1e-6:
                        a2, b2 = max(a, u0), min(b, u1)
                        if b2 - a2 > 0.035: row.append((a2, b2, nm)); m_of[a2] = int(round(x / (per / 2)))   # node of the header lattice
                    x = b + j; i += 1
                for a, b, nm in row:
                    glazed = pack.get("diaper") and nm == "H" and _diaper(m_of.get(a), k, pack["diaper"])
                    poly = [(a, z0 + j / 2), (b, z0 + j / 2), (b, z1 - j / 2), (a, z1 - j / 2)]
                    lo = np.maximum(np.array([z0 + j / 2] * 2), S.base_at(np.array([a, b]))); hi = np.minimum(np.array([z1 - j / 2] * 2), S.top_at(np.array([a, b])))
                    if float((hi - lo).min()) < 0.03: continue
                    ab = _arch_square(a, b, float(lo.min()), float(hi.max()), arches, pad=j / 2) if pack.get("arch_end") == "square" else None
                    if ab is not None:                                            # straight end, the wedge up to the arc is mortar
                        a, b = ab
                        if b - a < 0.035: continue
                        lo = np.maximum(np.array([z0 + j / 2] * 2), S.base_at(np.array([a, b]))); hi = np.minimum(np.array([z1 - j / 2] * 2), S.top_at(np.array([a, b])))
                    poly = _arch_cut([(a, lo[0]), (b, lo[1]), (b, hi[1]), (a, hi[0])], arches, pad=j / 2)
                    if poly is None: continue
                    stones.append(dict(kind="brick_" + nm.lower(), poly=np.round(poly, 4).tolist(), course=k,
                                       d=[0.0, pack["brick"][0] if nm == "H" else pack["brick"][1]],
                                       tint=round(float(rng.normal(1.0, pack.get("tint", 0.08))), 3),
                                       mat="glazed" if glazed else pack.get("mat", "brick"), pre_inset=True))
                continue
            if kind == "herring" and k % pack.get("lace_every", 3) != pack.get("lace_every", 3) - 1:   # herringbone: slabs at 45°
                h = z1 - z0; w = pack["herring_w"]; sg = 1 if k % 2 == 0 else -1; x = u0 - h
                while x < u1 + h:
                    q = _clip([(x, z0), (x + w, z0), (x + w + sg * h, z1), (x + sg * h, z1)], u0, u1)
                    if len(q) >= 3: q = _arch_cut(q, arches, min_area=0.002)
                    if q is not None and len(q) >= 3 and abs(_area(np.asarray(q))) > 0.002:
                        stones.append(dict(kind="herring", poly=np.round(q, 4).tolist(), course=k, d=[0.0, float(rng.uniform(0.15, 0.25))],
                                           tint=round(float(rng.normal(1.0, 0.09)), 3), mat=pack.get("mat", "stone")))
                    x += w + pack["joint"] * 1.4
                continue
            lens = pack["plinth_stone"] if (k == 0 and pack.get("plinth_stone")) else (pack.get("lace_stone") if kind == "herring" else pack["stone"])
            row = fill_row(u0, u1, lens, pack["lap"], prev, rng, forced=S.bends)
            if k in putlog_rows:                                                  # putlog holes are gaps in the course
                pl = pack["putlog"]; u = u0 + rng.uniform(0.6, 1.4)
                cut = []
                while u < u1 - 0.6:                                              # scaffold lifts start at ~1 m above the ground, not near the top
                    if rng.random() > 0.15 and z0 > float(S.base_at(u)) + 1.0 and z1 < float(S.top_at(u)) - 0.5:
                        cut.append((u, u + rng.uniform(*pl["w"]), bool(rng.random() < pl["plug"])))
                    u += rng.uniform(*pl["step"])
                jnts = sorted({x for ab in row for x in ab})
                snapped = []
                for h0, h1, plug in cut:                                          # the edge of a hole near a joint is snapped to the joint
                    j0 = [j for j in jnts if abs(j - h0) < 0.12]; j1 = [j for j in jnts if abs(j - h1) < 0.12]
                    h0 = min(j0, key=lambda j: abs(j - h0)) if j0 else h0; h1 = min(j1, key=lambda j: abs(j - h1)) if j1 else h1
                    if h1 - h0 > 0.08: snapped.append((h0, h1, plug))
                cut = snapped
                new = []
                for a, b in row:
                    pieces = [(a, b)]
                    for h0, h1, plug in cut:
                        nxt = []
                        for pa, pb in pieces:
                            if h1 <= pa or h0 >= pb: nxt.append((pa, pb)); continue
                            if h0 - pa > 0.12: nxt.append((pa, h0))
                            if pb - h1 > 0.12: nxt.append((h1, pb))
                        pieces = nxt
                    new += pieces
                row = new
                for h0, h1, plug in cut:
                    holes.append(dict(u0=h0, u1=h1, z0=z0, z1=z1, plug=plug))
            tilt = {}; tl = pack.get("tilt", 0.0)
            for a, b in row:
                ta = tilt.setdefault(round(a, 4), (rng.uniform(-tl, tl), rng.uniform(-tl, tl)))
                tb = tilt.setdefault(round(b, 4), (rng.uniform(-tl, tl), rng.uniform(-tl, tl)))
                if abs(a - u0) < 1e-6 or a in S.bends: ta = (0.0, 0.0)
                if abs(b - u1) < 1e-6 or b in S.bends: tb = (0.0, 0.0)
                if pack.get("shape") == "cut":                                    # rubble and dry-stone: cut corners, less than full height
                    poly = _shape_cut(a, b, max(z0, float(S.base_at((a + b) / 2))), min(z1, float(S.top_at((a + b) / 2))), rng)
                else:
                    m = max(2, int(math.ceil((b - a) / 0.1)) + 1)
                    ua = np.linspace(a, b, m)
                    lo_ = np.maximum(z0 + np.array([wave(k, u) for u in ua]), S.base_at(ua))
                    hi_ = np.minimum(z1 + np.array([wave(k + 1, u) for u in ua]), S.top_at(ua))
                    if np.all(hi_ - lo_ < 0.06): continue
                    ok = hi_ - lo_ >= 0.06                                       # trimming: where there is no stone — thinner than 6 cm
                    if not ok.all():
                        idx = np.flatnonzero(ok); ua, lo_, hi_ = ua[idx[0]:idx[-1] + 1], lo_[idx[0]:idx[-1] + 1], hi_[idx[0]:idx[-1] + 1]
                    ob = np.linspace(ta[0], tb[0], len(ua)); ot = np.linspace(ta[1], tb[1], len(ua))
                    poly = [(u + o, z) for u, o, z in zip(ua, ob, lo_)] + [(u + o, z) for u, o, z in zip(ua[::-1], ot[::-1], hi_[::-1])]
                if arches:
                    poly = _arch_cut(poly, arches)
                    if poly is None: continue
                bond = pack.get("bond_every") and rng.random() < 0.35
                mat = "brick" if bk == "brick" else pack.get("mat", "stone")
                stones.append(dict(kind="field" if pack.get("face") == "ashlar" else "rubble", poly=np.round(poly, 4).tolist(),
                                   course=k, d=[0.0, 0.62 if bond else float(rng.uniform(*pack["face_d"]))],
                                   tint=round(float(rng.normal(1.0, pack.get("tint", 0.07))), 3), mat=mat, through=bool(bond)))
                joints.append(b)
            joints.append(u0)
        prev = joints
    return holes


def _diaper(m, k, D):
    """Rhombic lattice of glazed headers (brick Gothic): the headers of the bond form a lattice (m, k), m is the half-period
    number along the course (in monk bond the parity of m equals the parity of the course), k is the course; the lines of
    the rhombus are diagonals of the lattice m ∓ k = 2D·n (a header on every course of the line): a rhombus of 2D
    half-periods × 2D courses."""
    return m is not None and ((m - k) % (2 * D) == 0 or (m + k) % (2 * D) == 0)


def _fill_skyline(S, pack, us, b_u, t_u, occ_mask, stones, rng):
    """Random ashlar (Legakis, Random Ashlar): a "horizon" along u; at its lowest step a stone 1–5 small courses (unit) high
    with a length taken from stone; a narrow step is extended to its neighbor; occupied space is an obstacle (the horizon
    jumps over it)."""
    du = 0.02; uu = np.arange(0.0, S.L, du); unit = pack["unit"]
    sky = S.base_at(uu + du / 2).copy(); top = S.top_at(uu + du / 2)
    def lift(i0, i1):                                                              # horizon inside occupied space — raised to its top
        for _ in range(40):
            m = occ_mask(uu[i0:i1] + du / 2, sky[i0:i1] + 0.01)
            if not m.any(): break
            sky[i0:i1][m] += 0.02
    lift(0, len(uu))
    for _ in range(200000):
        live = sky < top - 0.04
        if not live.any(): break
        z = float(sky[live].min()); i = int(np.flatnonzero(live & (np.abs(sky - z) < 1e-6))[0]); j = i
        while j < len(uu) and abs(sky[j] - z) < 1e-6 and sky[j] < top[j] - 0.04: j += 1
        seg = (j - i) * du
        wmin, wmax = pack["stone"]
        if seg < wmin:                                                             # narrow step: extend to the neighbor
            nb = [sky[i - 1]] if i > 0 else []
            if j < len(uu): nb.append(sky[j])
            nb = [x for x in nb if x > z + 0.02]
            h = (min(nb) - z) if nb else min(unit * rng.integers(*pack["heights"]), float(top[i:j].min()) - z)
            w = seg
        else:
            w = rng.uniform(wmin, min(wmax, seg));
            if seg - w < wmin: w = seg
            h = unit * int(rng.integers(*pack["heights"]))
            h = min(h, float(top[i:i + int(round(w / du))].min()) - z)
        k = max(1, int(round(w / du)))
        cols_ = uu[i:i + k:max(1, k // 8)] + du / 2                                # occupied space above the stone limits the height
        for zz in np.arange(z + 0.01, z + h, 0.01):
            if occ_mask(cols_, np.full(len(cols_), zz)).any(): h = zz - z - 0.005; break
        if h < 0.03: sky[i:i + k] = sky[i:i + k] + 0.02; lift(i, i + k); continue
        a, b = uu[i], uu[i] + k * du
        stones.append(dict(kind="field", poly=[(a, z), (b, z), (b, z + h), (a, z + h)], course=-1,
                           d=[0.0, float(rng.uniform(*pack["face_d"]))], tint=round(float(rng.normal(1.0, pack.get("tint", 0.07))), 3),
                           mat=pack.get("mat", "stone")))
        sky[i:i + k] = z + h; lift(i, i + k)


def _fill_cells(S, pack, b_u, t_u, occ_mask, stones, rng):
    """Cells (rubble without courses, polygonal, cyclopean, Japanese): seeds over the free field spaced by size (larger at
    the bottom by bottom_k), Voronoi cells on a raster (OpenCV, no SciPy), Lloyd smoothing, relax iterations; a stone is the cell
    contour simplified to 5–9 corners. Cyclopean and Japanese: the blocks are shrunk by 2–6 cm, the gaps between them are
    filled with small stone (fill)."""
    import cv2
    res = pack.get("res", 0.01)
    u0, u1 = 0.0, S.L; v0, v1 = float(b_u.min()), float(t_u.max())
    nu, nv = int(math.ceil((u1 - u0) / res)), int(math.ceil((v1 - v0) / res))
    U = u0 + (np.arange(nu) + 0.5) * res; V = v0 + (np.arange(nv) + 0.5) * res
    UU, VV = np.meshgrid(U, V)
    region = (VV >= S.base_at(U)[None, :]) & (VV <= S.top_at(U)[None, :]) & ~occ_mask(UU, VV)

    def seeds(mask, size, bottom_k=1.0, tries=60):
        ys, xs = np.nonzero(mask)
        if not len(ys): return np.zeros((0, 2))
        order = rng.permutation(len(ys))[: int(mask.sum() * res * res / (size[0] ** 2) * tries / 10) + 50]
        pts, rad = [], []
        cell = size[0] / res; grid = {}
        for idx in order:
            y, x = ys[idx], xs[idx]
            hf = 1.0 + (bottom_k - 1.0) * (1.0 - y / max(nv - 1, 1))                  # larger at the bottom
            r = rng.uniform(*size) * hf / res * 0.5
            gx, gy = int(x // cell), int(y // cell); ok = True
            for ddx in range(-3, 4):
                for ddy in range(-3, 4):
                    for (px, py, pr) in grid.get((gx + ddx, gy + ddy), ()):
                        if (px - x) ** 2 + (py - y) ** 2 < (max(r, pr) * 0.95) ** 2: ok = False; break
                    if not ok: break
                if not ok: break
            if ok:
                grid.setdefault((gx, gy), []).append((x, y, r)); pts.append((x, y)); rad.append(r)
        return np.asarray(pts, float)

    def voronoi(P, mask):
        lab = np.zeros((nv, nu), np.int32)
        if not len(P): return lab
        sm = np.ones((nv, nu), np.uint8); Pi = np.clip(np.round(P).astype(int), [0, 0], [nu - 1, nv - 1])
        sm[Pi[:, 1], Pi[:, 0]] = 0
        _, lb = cv2.distanceTransformWithLabels(sm, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        idmap = np.zeros(lb.max() + 1, np.int32); idmap[lb[Pi[:, 1], Pi[:, 0]]] = np.arange(1, len(Pi) + 1)
        lab = idmap[lb]; lab[~mask] = 0
        return lab

    def relax(P, mask, it):
        for _ in range(it):
            lab = voronoi(P, mask)
            if not lab.any(): break
            cnt = np.bincount(lab.ravel(), minlength=len(P) + 1)
            sx = np.bincount(lab.ravel(), weights=np.broadcast_to(np.arange(nu)[None, :], lab.shape).ravel(), minlength=len(P) + 1)
            sy = np.bincount(lab.ravel(), weights=np.broadcast_to(np.arange(nv)[:, None], lab.shape).ravel(), minlength=len(P) + 1)
            ok = cnt[1:] > 0; P = P.copy(); P[ok, 0] = sx[1:][ok] / cnt[1:][ok]; P[ok, 1] = sy[1:][ok] / cnt[1:][ok]
        return P

    def polys(lab, n, eps, min_area, shrink=None):
        out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(ys) < min_area / (res * res): continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
            if shrink: m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(shrink(i)))
            cs, _ = cv2.findContours(np.pad(m, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs: continue
            c = max(cs, key=cv2.contourArea)
            if cv2.contourArea(c) * res * res < min_area: continue
            c = cv2.approxPolyDP(c, eps / res, True)[:, 0, :].astype(float)
            if len(c) < 3: continue
            out.append(np.stack([u0 + (c[:, 0] + x0 - 1 + 0.5) * res, v0 + (c[:, 1] + y0 - 1 + 0.5) * res], 1))
        return out

    P = seeds(region, pack["size"], pack.get("bottom_k", 1.0)); P = relax(P, region, pack.get("relax", 1))
    lab = voronoi(P, region)
    big = polys(lab, len(P), pack.get("eps", 0.02), pack.get("min_area", 0.01),
                shrink=(lambda i: rng.integers(*pack["shrink"])) if pack.get("shrink") else None)
    for q in big:
        for _ in range(int(pack.get("round", 0))):                                 # rounded stone: corners cut off (Chaikin)
            q = np.concatenate([[0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b] for a, b in zip(q, np.roll(q, -1, 0))])
        sz = math.sqrt(abs(_area(q)))
        stones.append(dict(kind="cell", poly=np.round(q, 4).tolist(), course=-1, d=[0.0, float(np.clip(sz * pack.get("depth_k", 0.6), 0.18, 1.2))],
                           tint=round(float(rng.normal(1.0, pack.get("tint", 0.09))), 3), mat=pack.get("mat", "stone")))
    if pack.get("fill"):                                                           # gaps between the blocks — small stone
        cov = np.zeros((nv, nu), np.uint8)
        for q in big:
            pts = np.round(np.stack([(q[:, 0] - u0) / res - 0.5, (q[:, 1] - v0) / res - 0.5], 1) * 8).astype(np.int32)
            cv2.fillPoly(cov, [pts], 1, shift=3)
        gap = region & (cov == 0)
        gap = cv2.erode(gap.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Q = seeds(gap, pack["fill"]); lab2 = voronoi(Q, gap)
        for q in polys(lab2, len(Q), 0.012, 0.004):
            stones.append(dict(kind="spall", poly=np.round(q, 4).tolist(), course=-1, d=[0.0, 0.15],
                               tint=round(float(rng.normal(0.95, 0.1)), 3), mat=pack.get("mat", "stone")))


def _fill_grid45(S, pack, us, b_u, t_u, occ, stones, rng):
    """Roman lattice (opus reticulatum, with brick belts — opus mixtum): belts of band_n brick courses every band_every
    meters, between them — square stones (side length side) rotated by 45°. The height of the lattice field is a whole number of
    half-steps: at a belt the stones are cut diagonally (triangles), no teeth. Framings and the arch trim the stones in
    place (_cut_occ)."""
    zmin, zmax = float(b_u.min()), float(t_u.max())
    h = pack["side"] / math.sqrt(2); j = pack["joint"]; dx = 2 * h + j * math.sqrt(2)
    hb = pack["brick"][2] + pack["joint"]; z = zmin; zones = []
    while z < zmax - 0.02:
        n = max(2, int(round(rng.uniform(*pack["band_every"]) / (dx / 2))))
        zb = min(z + n * dx / 2, zmax); zones.append(("net", z, zb)); z = zb
        for _ in range(pack["band_n"]):
            if z + hb > zmax + 1e-6: break
            zones.append(("brick", z, z + hb)); z += hb
    for k, (zk, z0, z1) in enumerate(zones):
        inside = (b_u < z1 - 0.02) & (t_u > z0 + 0.02)
        for c0, c1 in runs(inside):
            u0, u1 = us[c0], us[min(c1, len(us) - 1)]
            if u1 - u0 < 0.05: continue
            if zk == "brick":
                L_ = pack["brick"][0]; ph = (L_ + j) / 2 * (k % 2); x = u0 - ph
                while x < u1:
                    a, b = max(x, u0), min(x + L_, u1)
                    q = _cut_occ([(a, z0 + j / 2), (b, z0 + j / 2), (b, z1 - j / 2), (a, z1 - j / 2)], occ, pad=j / 2) if b - a > 0.03 else None
                    if q:
                        stones.append(dict(kind="brick_s", poly=q, course=k, d=[0.0, pack["brick"][1]],
                                           tint=round(float(rng.normal(1.0, 0.08)), 3), mat="brick", pre_inset=True))
                    x += L_ + j
                continue
            r = 0
            y = z0
            while y - h < z1:
                off = (dx / 2) * (r % 2); x = u0 - dx + off
                while x - h < u1:
                    q = _clip([(x, y - h), (x + h, y), (x, y + h), (x - h, y)], u0, u1, z0, z1)
                    if len(q) >= 3 and abs(_area(np.asarray(q))) > h * h * 0.2: q = _cut_occ(q, occ, min_area=h * h * 0.2)
                    else: q = None
                    if q:
                        stones.append(dict(kind="net", poly=q, course=k, d=[0.0, 0.2],
                                           tint=round(float(rng.normal(1.0, 0.07)), 3), mat=pack.get("mat", "tufa")))
                    x += dx
                y += dx / 2; r += 1


def _area(p):
    return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])


# ── map of the side segment mask (for the slab under the blocks and for the far view) ───────────────────────────
JOINT_SHADE = dict(bed=0.62, perp=0.86)            # joint in shadow: the bed joint is darker than the perpend (calibrated by a probe on blocks)
FAR_SHADE = dict(bed=0.40, perp=0.60, band=0.42, band_w=0.05, tint=1.6)   # far view without courses: the bed-joint shadow is given by the map


def face_maps(L, u0, u1, v0, v1, tex, pack, stone_rgb=(0.52, 0.47, 0.40), mortar=(0.66, 0.63, 0.57), far=False):
    """Color and height of the segment [u0, u1] × [v0, v1] of a side with step tex — from the same stones of layout L:
    computed with subsampling ≤ 5 mm and averaged over the area (a narrow joint darkens a texel instead of vanishing);
    the bed joint is darker than the perpend, chamfer, stone tint; row 0 is the bottom (like Blender pixels).
    → (alb[nv, nu, 3], hgt[nv, nu])"""
    import cv2
    sub = 1 if tex <= 0.008 else max(1, min(8, int(math.ceil(tex / 0.005)))); fine = tex / sub   # up to 8 mm: a joint spans ≥ 2 texels
    nu, nv = max(4, int(math.ceil((u1 - u0) / tex))), max(4, int(math.ceil((v1 - v0) / tex)))
    NU, NV = nu * sub, nv * sub
    lab = np.zeros((NV, NU), np.int32); tint = [1.0]; trim = [False]
    for st in list(L["stones"]) + list(L["features"]):
        q = np.asarray(st["poly"], float)
        if q[:, 0].max() < u0 or q[:, 0].min() > u1: continue
        tint.append(st["tint"]); trim.append(st["kind"] in ("jamb", "voussoir", "key", "quoin", "sill"))
        pts = np.round(np.stack([(q[:, 0] - u0) / fine, (q[:, 1] - v0) / fine], 1) * 8).astype(np.int32)
        cv2.fillPoly(lab, [pts], len(tint) - 1, shift=3)
    tint = np.asarray(tint, np.float32); trim = np.asarray(trim)
    dx = lab[:, 1:] != lab[:, :-1]; dy = lab[1:, :] != lab[:-1, :]
    ex = np.zeros(lab.shape, bool); ey = np.zeros(lab.shape, bool)
    ex[:, 1:] |= dx; ex[:, :-1] |= dx; ey[1:, :] |= dy; ey[:-1, :] |= dy
    dxs = cv2.distanceTransform((~ex).astype(np.uint8), cv2.DIST_L2, 3) * fine
    dys = cv2.distanceTransform((~ey).astype(np.uint8), cv2.DIST_L2, 3) * fine
    dist = np.minimum(dxs, dys); jh = pack["joint"] / 2
    st_ = (lab > 0) & (dist > jh)
    prof = np.clip((dist - jh) / 0.014, 0, 1) ** 0.6
    alb = np.empty((NV, NU, 3), np.float32); alb[:] = mortar
    base = np.asarray(stone_rgb, np.float32)
    tn = (1.0 + (tint - 1.0) * FAR_SHADE["tint"]) if far else tint                # far view: stone tones are made more distinct
    k = tn[lab] * np.where(trim[lab], 1.05, 1.0)
    alb[st_] = base * k[st_][:, None] * (0.93 + 0.07 * prof[st_][:, None])
    jm = (lab > 0) & ~st_
    js = FAR_SHADE if far else JOINT_SHADE
    alb[jm] *= np.where(dys <= dxs, js["bed"], js["perp"])[jm][:, None]
    if far:                                                                      # far view without courses: the bed-joint shadow is a band under the top of the stone
        dtop = np.zeros((NV, NU), np.float32)
        for r in range(NV - 2, -1, -1): dtop[r] = np.where(lab[r] == lab[r + 1], dtop[r + 1] + fine, 0.0)
        alb[st_] *= (1.0 - FAR_SHADE["band"] * np.exp(-dtop[st_] / FAR_SHADE["band_w"]))[:, None]
    hgt = np.where(st_, 0.55 + 0.45 * prof, 0.25).astype(np.float32)
    for o in L.get("openings", []):                                               # arrow slits and windows: the darkness of the opening
        c0, c1 = int((o["u"] - o["w"] / 2 - u0) / fine), int((o["u"] + o["w"] / 2 - u0) / fine)
        r0, r1 = int((o["z"] - o["h"] / 2 - v0) / fine), int((o["z"] + o["h"] / 2 - v0) / fine)
        c0, c1, r0, r1 = max(c0, 0), min(c1, NU), max(r0, 0), min(r1, NV)
        if c1 > c0 and r1 > r0: alb[r0:r1, c0:c1] = 0.08; hgt[r0:r1, c0:c1] = 0.0
    for h in L.get("holes", []):                                                  # putlog holes: an open one is dark, a plugged one is a darker stone
        if h["u1"] < u0 or h["u0"] > u1: continue
        c0, c1 = int((h["u0"] - u0) / fine), int((h["u1"] - u0) / fine); r0, r1 = int((h["z0"] - v0) / fine), int((h["z1"] - v0) / fine)
        c0, c1, r0, r1 = max(c0, 0), min(c1, NU), max(r0, 0), min(r1, NV)
        if c1 <= c0 or r1 <= r0: continue
        if h["plug"]:
            g_ = int(round(0.01 / fine)); alb[r0 + g_:r1 - g_, c0 + g_:c1 - g_] = base * 0.8; hgt[r0:r1, c0:c1] = 0.3
        else:
            alb[r0:r1, c0:c1] = 0.14; hgt[r0:r1, c0:c1] = 0.0
    if sub > 1:
        alb = alb.reshape(nv, sub, nu, sub, 3).mean((1, 3)); hgt = hgt.reshape(nv, sub, nu, sub).mean((1, 3))
    return alb, hgt


# ── unrolled view for visual inspection ─────────────────────────────────────────────────────────────────────────
def preview(S, L, path, tex=0.02, stone_rgb=(0.52, 0.47, 0.40), mortar=(0.66, 0.63, 0.57)):
    """Unrolled side: stones (tint), joints, bottom and top of the strip, features; PNG (PIL from the project's Python)."""
    import cv2
    us = np.arange(0.0, S.L, tex); zmin, zmax = float(S.base.min()) - 0.5, float(S.top.max()) + 0.5
    W, H = len(us), int(math.ceil((zmax - zmin) / tex))
    img = np.full((H, W, 3), 0.9, np.float32)
    def pix(poly):
        q = np.asarray(poly, float); return np.round(np.stack([q[:, 0] / tex, (zmax - q[:, 1]) / tex], 1) * 8).astype(np.int32)
    # the strip is mortar
    band = [(u, S.top_at(u)) for u in us[::10]] + [(u, S.base_at(u)) for u in us[::-10]]
    cv2.fillPoly(img, [pix(band)], tuple(float(c) for c in mortar), shift=3)
    base = np.array(stone_rgb, np.float32)
    for s in L["stones"]:
        cv2.fillPoly(img, [pix(s["poly"])], tuple(float(c) for c in base * s["tint"]), shift=3)
        cv2.polylines(img, [pix(s["poly"])], True, tuple(float(c) for c in np.array(mortar) * 0.7), 1, shift=3)
    for f in L["features"]:
        cv2.fillPoly(img, [pix(f["poly"])], tuple(float(c) for c in base * 1.12), shift=3)
        cv2.polylines(img, [pix(f["poly"])], True, (0.3, 0.3, 0.3), 1, shift=3)
    for h in L["holes"]:
        cv2.fillPoly(img, [pix([(h["u0"], h["z0"]), (h["u1"], h["z0"]), (h["u1"], h["z1"]), (h["u0"], h["z1"])])],
                     (0.45, 0.42, 0.38) if h["plug"] else (0.12, 0.12, 0.12), shift=3)
    for u in S.bends:
        x = int(u / tex); cv2.line(img, (x, 0), (x, 12), (0.8, 0.2, 0.2), 2)
    cv2.imwrite(path, (np.clip(img, 0, 1)[:, :, ::-1] ** (1 / 2.2) * 255).astype(np.uint8))
    return W, H


def seg_vrange(S, s0, s1):
    """Heights for the segment map [s0, s1] (with a MARGIN margin along u): from the bottom to the top of the side over this
    piece ± 0.1 m."""
    uu = np.linspace(max(0.0, s0 - MARGIN), min(S.L, s1 + MARGIN), 40)
    return float(S.base_at(uu).min()) - 0.1, float(S.top_at(uu).max()) + 0.1


def build_request(req):
    """Layout and maps for assembly in Blender (Blender's Python has no OpenCV): req = {out, items: [{tag, strip — the side's
    passport (Strip.spec), pack, seed, gates, openings, gaps, putlogs, seg, texels — texel size per segment}], packs:
    {name: pack} — otherwise from library} → <out>/<tag>_layout.json and the segment maps <tag>_seg<i>_{alb,hgt}.png
    (PNG row 0 is the top). One source of layout for blocks, courses and the slab."""
    import cv2
    packs = dict(PACKS); packs.update(req.get("packs") or {})
    os.makedirs(req["out"], exist_ok=True); done = []
    for it in req["items"]:
        pack = packs[it["pack"]]
        S = strip_from(it["strip"]); gaps = it.get("gaps", []); tag = it["tag"]
        L = layout(S, pack, gates=it.get("gates", []), seed=it["seed"], putlogs=it.get("putlogs", pack.get("putlog") is not None),
                   gaps=gaps, openings=it.get("openings", []))
        json.dump(L, open(os.path.join(req["out"], tag + "_layout.json"), "w"))
        seg = it.get("seg", 10.0)
        for i, tex in enumerate(it.get("texels", [])):
            v0, v1 = seg_vrange(S, i * seg, min(S.L, (i + 1) * seg))                     # height of its own piece
            s0, s1 = i * seg - MARGIN, min(S.L, (i + 1) * seg) + MARGIN                # margin: stones and courses at the boundary
            far = req.get("far", 0) > 0 and tex >= float(np.mean(pack["course"])) / (2.0 * req["far"]) - 1e-9   # a far segment (no courses)
            alb, hgt = face_maps(L, s0, s1, v0, v1, tex, pack, far=far)
            a8 = (np.clip(alb, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)[::-1, :, ::-1]
            cv2.imwrite(os.path.join(req["out"], f"{tag}_seg{i}_alb.png"), a8)
            cv2.imwrite(os.path.join(req["out"], f"{tag}_seg{i}_hgt.png"), (np.clip(hgt, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1])
        done.append(dict(tag=tag, **L["stats"]))
    return done




DESC_FILES = ("constructions.yaml", "masonry.yaml")                            # building description (masonry.yaml is the former name)
ALIASES = {"surface": "textures"}                                          # former section names
OFF = ("0", "no")                                                             # towers=0|no — towers without masonry: the plan has no tower pack


def resolve(folder, kw=None):
    """Masonry plan of a building for Blender (like mat_plan.py resolve): <folder>/constructions.yaml — the "masonry:" section
    (packs per part, detail, thresholds, rows, plate, seed); other sections (textures: …) are passed on to the
    following steps as they are; command-line keys override the description (outer=, inner=, towers= (0|no — towers
    without masonry), detail=cam|all, Tlo=, Thi=, Tfar=, rows=0|1, plate=0|1, seed=).
    Packs — only the needed ones and whole (there is no YAML in Blender); packs=all — all of them."""
    import yaml
    kw = dict(kw or {}); path = next((os.path.join(folder, f) for f in DESC_FILES if os.path.exists(os.path.join(folder, f))), "")
    d = (yaml.safe_load(open(path, encoding="utf-8")) or {}) if path else {}
    d = {ALIASES.get(key, key): val for key, val in d.items()}
    k = dict(d.get("masonry") or {})
    parts = dict(LIBRARY["parts"]); parts.update(k.get("packs") or {})
    parts.update({p_: kw[p_] for p_ in list(parts) if p_ in kw})
    if str(parts.get("towers")).strip().lower() in OFF: del parts["towers"]       # towers without masonry: items_of does not take them
    bad = sorted({str(v) for v in parts.values() if str(v) not in PACKS})
    if bad: raise SystemExit(f"masonry: no pack {bad}; available: {', '.join(PACKS)} (components/constructions/masonry/packs.yaml; "
                             f"towers=0|no — towers without masonry)")
    view = kw.get("detail") or k.get("detail", "cam")
    if view not in ("cam", "all"): raise SystemExit(f"detail level '{view}': cam | all")
    lo, hi = (k.get("thresholds") or [6, 14])
    yes = lambda x: str(x).lower() in ("1", "true", "yes")
    plan = dict(masonry=dict(parts=parts, detail=view, thresholds=[float(kw.get("Tlo", lo)), float(kw.get("Thi", hi))],
                            far=float(kw.get("Tfar", k.get("far", 3.0))),     # px per course: below it only a slab with a map (no courses)
                            rows=yes(kw.get("rows", k.get("rows", "yes"))),
                            plate=yes(kw.get("plate", k.get("plate", "yes"))),
                            seed=int(kw.get("seed", k.get("seed", 1)))),
                packs={n: PACKS[n] for n in sorted(set(parts.values()))}, substances=LIBRARY.get("substances", {}),
                version=LIBRARY.get("version"), description=path or None)
    if kw.get("packs") == "all": plan["packs"] = dict(PACKS)                    # kinds sheet, checks
    for key, val in d.items():                                                   # sections of the following steps are passed on as they are
        if key != "masonry": plan[key] = val
    return plan


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "build":
    print("__RESULT__", json.dumps(build_request(json.load(open(sys.argv[2]))), ensure_ascii=False))
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "resolve":
    KW = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    plan = resolve(sys.argv[2], KW)
    if KW.get("out"): json.dump(plan, open(KW["out"], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False))
elif __name__ == "__main__":                                                     # wall layout from the scene's surfaces layer
    import time
    KW = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    lay = json.load(open(sys.argv[1])); wi = int(KW.get("wall", 0))           # surfaces.json (core.layers; written by the buildings component)
    out = KW.get("out", os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "masonry"))
    os.makedirs(out, exist_ok=True)
    side = {it["side"]: it for it in lay["items"] if it.get("wall") == wi}
    S_out, S_in = strip_from(side[1]["strip"]), strip_from(side[2]["strip"])
    t0 = time.time()
    Lo = layout(S_out, PACKS["ashlar"], gaps=side[1].get("gaps", []), seed=1)
    Li = layout(S_in, PACKS["rubble"], gaps=side[2].get("gaps", []), seed=2, putlogs=False)
    dt = time.time() - t0
    preview(S_out, Lo, os.path.join(out, f"wall{wi}_out.png"))
    json.dump(dict(outer=Lo, inner=Li), open(os.path.join(out, f"wall{wi}_layout.json"), "w"))
    print("__RESULT__", json.dumps(dict(outer=Lo["stats"], inner=Li["stats"], seconds=round(dt, 2)), ensure_ascii=False))
