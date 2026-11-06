"""Masonry surfaces layer (core.layers: surfaces.json) — an adapter from the Buildings castle plan: walls with
two sides (the outer one from the bottom, the rock, up to the top of the parapet, the inner one up to the
walkway; breaks under gates and "houses as a wall"), towers as a ring (outline bl_castle.footprint, plinth
with a batter, loopholes — openings by the bl_castle.win_blocked check). The "building" step writes it in
Blender right after the castle is built (the state of stairs and loophole details lives in the same
process); the masonry of Constructions lays stones by the layer and does not read the castle plan. Block
numbers (sid0) — as in the earlier castle_items of the masonry: wall — 10⁷·(wall+1) + side·10⁶, tower —
5·10⁷ + tower·10⁶; seeds belong to the reader (its plan seed)."""
import math

import numpy as np

from core.surface.strip import Strip, strip_from


def wall_strips(C, wi=0, parapet=None):
    """Sides of wall wi from castle.json: the outer one from the bottom (the rock, if any) to the top of the
    parapet, the inner one to the walkway. Returns [outer, inner]. Breaks (gates, "houses as a wall") are not
    cut out yet — the masonry makes the gates itself."""
    w = C["walls"][wi]; st = C.get("style", {})
    par = parapet if parapet is not None else (st.get("crenel") or {}).get("parapet", 1.1)
    base = np.asarray(w["base"], float); walk = np.asarray(w["walk"], float)
    if w.get("rock_top"): base = np.array([max(b, v) if v is not None else b for b, v in zip(base, w["rock_top"])])
    t = w["t"]; bat = float(st.get("batter", 0.0))
    outer = Strip(w["pts"], base, walk + par, t, side=1, closed=w["closed"], batter=bat, name=f"wall{wi}_out", walk=walk)
    inner = Strip(w["pts"], base, walk, t, side=-1, closed=w["closed"], name=f"wall{wi}_in", walk=walk)
    return [outer, inner]


def gaps_on(C, S, wi=0):
    """Breaks of a side along the path length, as in bl_castle.wall: at a gate — half-width w/2 + 1.4 m (at a gate
    tower — r · 1.05), at a "house as a wall" — the absorbed stations; → [(u0, u1)]."""
    out = []
    for g in C.get("gates", []):
        half = float(g["r"]) * 1.05 if g.get("kind") == "tower" else float(g.get("w", 3.0)) / 2 + 1.4
        d = np.linalg.norm(S.P - np.asarray(g["xy"], float), axis=1); idx = np.flatnonzero(d <= half)
        if len(idx): out.append((float(S.U[idx].min()) - 0.3, float(S.U[idx].max()) + 0.3))
    w = C["walls"][wi]
    for j in w.get("absorbed") or []:
        if j < len(S.U) - 1: out.append((float(S.U[j]) - 0.5, float(S.U[min(j + 1, len(S.U) - 1)]) + 0.5))
    return out


def gates_on(C, S, wi=0):
    """The gates of this wall on the unrolled view: u is the projection of the gate center on the axis, the width,
    the threshold, the passage height."""
    out = []
    for g in C.get("gates", []):
        xy = np.asarray(g["xy"], float)
        d = np.linalg.norm(S.P - xy, axis=1); i = int(np.argmin(d))
        if d[i] > max(4.0, g.get("w", 3) * 1.5): continue
        out.append(dict(u=float(S.U[i]), w=float(g.get("w", 3.0)), z=float(g.get("z", S.base[i])), h=float(g.get("h", 4.0)),
                        kind=g.get("kind", "gate")))
    return out


# ── layout ────────────────────────────────────────────────────────────────────────────────────────────────────────────


def tower_spec(t, BC):
    """Tower ring as in bl_castle.tower: the footprint outline, a plinth with a batter (offset batter·0.8·r up to
    the plinth knee), the top is the parapet of a flat roof, under a roof — a cornice, at machicolations — up to
    their corbels; loopholes — the same places and the win_blocked check. The ring seam is on the rear face
    (away from the threat).
    → (spec, openings [(xy, z, w, h)]) or None."""
    if t.get("stages") or abs(float(t.get("lean", 0.0))) > 1e-6: return None
    plan = t["plan"]; r = max(float(t["r"]), 1.5); c = np.asarray(t["xy"], float); d = t.get("dir")
    base, top = float(t["base"]), float(t["top"])
    if t.get("rock"): base = max(base, float(t["rock"]["z1"]))
    P = np.asarray(BC.footprint(plan, c, r, d, prow=bool(t.get("prow"))), float)
    h = top - base; plinth = base + min(h * 0.3, 2.5 + h * 0.15)
    roofk = "flat" if t.get("butterfass") else t.get("roof", "flat")
    gal = getattr(BC, "TWO_LEVEL", False) and roofk != "flat"
    mach = BC.ST.get("machicolation") and (roofk == "flat" or gal)
    ztop = top - 1.6 if mach else (top + BC.MERLON["parapet"] if roofk == "flat" else top)
    dd = np.asarray(d, float) if d else np.array([0.0, 1.0]); oo = np.array([dd[1], -dd[0]]) if d else np.array([1.0, 0.0])
    mids = (P + np.roll(P, -1, 0)) / 2; start = int(np.argmin((mids - c) @ dd))
    ops = []
    for zz in np.arange(plinth + 2.0, top - 2.2, 4.2):
        for ang in (-0.9, 0.0, 0.9):
            v = oo * math.cos(ang) + dd * math.sin(ang); v /= np.linalg.norm(v)
            q = c + v * (r * (1.0 if plan in ("round", "D") else 1.0 / math.cos(math.pi / 8) * 0.92))
            if zz + 0.75 > ztop or BC.win_blocked(q, v, zz): continue
            ops.append((q, float(zz), 0.22, 1.5))
    return dict(pts=P.tolist(), base=base, top=ztop, bat=float(BC.ST["batter"]) * 0.8 * r, knee=plinth, start=start), ops


def project_u(S, xy, z):
    """The length u of the unrolled station nearest to the point xy (sampled every 2 cm)."""
    us = np.arange(0.0, S.L, 0.02); W = S.world(us, np.full(len(us), z), 0.0)
    return float(us[int(np.argmin(np.linalg.norm(W[:, :2] - np.asarray(xy)[None], axis=1)))])


def items(C, BC):
    """Layer items: each side of each wall (wall, side), towers (tower) — the strip passport, breaks, openings,
    block numbers."""
    out = []
    for wi in range(len(C.get("walls") or [])):
        for side, (S, part) in enumerate(zip(wall_strips(C, wi), ("outer", "inner")), start=1):
            out.append(dict(tag=f"w{wi}_{'out' if side == 1 else 'in'}", wall=wi, side=side, strip=S.spec(), part=part,
                            sid0=10_000_000 * (wi + 1) + side * 1_000_000, gaps=[list(g) for g in gaps_on(C, S, wi)]))
    for ti, t in enumerate(C.get("towers") or []):
        sp = tower_spec(t, BC)
        if not sp: continue
        spec, ops = sp; S = strip_from(spec)
        out.append(dict(tag=f"t{ti}", tower=ti, strip=spec, part="towers", sid0=50_000_000 + ti * 1_000_000,
                        openings=[dict(u=project_u(S, q, zz), z=zz, w=w, h=h) for q, zz, w, h in ops]))
    return out


def write(scene, C, BC):
    """Surfaces layer of a castle (plan C, the module bl_castle with the castle built) → the scene's surfaces.json.
    → the number of items."""
    from core import layers as L
    it = items(C, BC)
    L.write(scene, "surfaces", {"items": it, "palette": C.get("colors") or {}})
    return len(it)
