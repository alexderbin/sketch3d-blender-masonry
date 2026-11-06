"""Type "classics": order buildings of a polis and a forum — a temple (peripteros, prostyle, amphiprostyle, in
antis, podium, tholos), stoa, bouleuterion, palaestra, propylon, altar, a theatre on a slope. The knowledge
is classics.yaml.
  program(kind, frame, setting, rng) — a building in a frame {c, u — the facade axis, L — along the axis,
            W — across} (theatre: c — the center of the orchestra, u — toward the cavea, R — the cavea
            radius): the order and the column count are fitted to the frame, the module is the column's
            lower diameter D → a dict (JSON) or None (does not fit);
  geometry(program) — numpy only (Blender and project Python): prisms (crepidoma, podium, cella, theatre
            rows), columns, walls with openings, faces (a roof with thickness, pediments), beams (the raking
            cornice), a cone (tholos);
  footprint(program) — the outline on the ground (ensemble: what is occupied).
Separately: plan(place, out, kind=…|all, at=…) — a building or a showcase of all kinds at the place →
classic.json."""
import os
import sys
from components.buildings import arch_parts
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_K = None
TEMPLES = ("peripteros", "prostyle", "amphiprostyle", "in_antis", "podium", "tholos", "propylon")
KINDS = TEMPLES + ("stoa", "bouleuterion", "palaestra", "altar", "theatre")


def knowledge():
    """The type's knowledge (classics.yaml); only in project Python."""
    global _K
    if _K is None:
        import yaml
        _K = yaml.safe_load(open(os.path.join(HERE, "classics.yaml"), encoding="utf-8"))
    return _K


def _u(r, rng): return float(r[0] + (r[1] - r[0]) * rng.random()) if isinstance(r, (list, tuple)) else float(r)


def _pick(d, rng):
    if not isinstance(d, dict): return d
    k = list(d); w = np.array([float(d[x]) for x in k]); return k[int(rng.choice(len(k), p=w / w.sum()))]


def _col(v, rng, k=0.04):
    if isinstance(v[0], (list, tuple)): v = v[int(rng.integers(len(v)))]
    m = 1 + (rng.random() - 0.5) * 2 * k
    return [round(float(min(1, max(0, c * m))), 4) for c in v]


class Fr:
    """Building frame: center c, axis u (where the facade faces), v to the left of u; (x along u, y along v) →
    world."""

    def __init__(self, c, u):
        self.c = np.asarray(c, float); u = np.asarray(u, float); self.u = u / max(np.linalg.norm(u), 1e-9); self.v = np.array([-self.u[1], self.u[0]])

    def w(self, x, y): return self.c + self.u * x + self.v * y

    def rect(self, x0, x1, y0, y1):
        return [[float(q[0]), float(q[1])] for q in (self.w(x0, y0), self.w(x1, y0), self.w(x1, y1), self.w(x0, y1))]

    def p3(self, x, y, z):
        q = self.w(x, y); return (float(q[0]), float(q[1]), float(z))


def _ground(height, pts, z=None):
    """Base level and bottom: z (the platform) or the median ground along the outline; the bottom is below the
    lowest point."""
    P = np.asarray(pts, float); zs = np.asarray(height(P[:, 0], P[:, 1]), float) if height is not None else np.zeros(len(P))
    z0 = float(z) if z is not None else float(np.median(zs)) + 0.05
    return z0, float(min(zs.min(), z0)) - 0.4


def _sample(F, hx, hy, n=5):
    xs = np.linspace(-hx, hx, n); ys = np.linspace(-hy, hy, n); X, Y = np.meshgrid(xs, ys)
    return [F.w(x, y) for x, y in zip(X.ravel(), Y.ravel())]


# ---------------------------------------------------------------- program
def program(kind, fr, ctx, rng):
    """Program of a building of kind kind in the frame fr; ctx: height(x, y) — the edited ground, z — the platform
    level (if given), height0 — the ground before the edit (theatre: the wall behind the cavea), order (own
    choice). → a dict or None."""
    K = knowledge(); S = K["kinds"][kind]; F = Fr(fr["c"], fr["u"]); H = ctx.get("height")
    C = K["color"]; col = {k: _col(C[k], rng) for k in ("stone", "wall", "roof", "floor", "sand", "door")}
    col["painting"] = list(C["painting"]); col["border"] = list(C["border"])
    if kind != "theatre" and rng.random() < 0.7: col["roof"] = _col([[0.68, 0.38, 0.25], [0.62, 0.34, 0.22]], rng)
    P = {"kind": kind, "c": [round(float(v), 3) for v in F.c], "u": [round(float(v), 5) for v in F.u], "col": col, "seed": int(rng.integers(1 << 30))}
    ordn = ctx.get("order") or _pick(S.get("order", "doric"), rng)
    O = dict(K["orders"][ordn]); O["column"] = round(_u(O["column"], rng), 3); O["clearance"] = round(_u(O["clearance"], rng), 3)
    P["order"] = ordn; P["ord"] = O
    if kind in TEMPLES: return _temple(kind, S, F, fr, ctx, rng, P, O, H)
    return {"stoa": _stoa, "bouleuterion": _bouleuterion, "palaestra": _palaestra, "altar": _altar, "theatre": _theatre}[kind](S, F, fr, ctx, rng, P, O, H)


def _temple(kind, S, F, fr, ctx, rng, P, O, H):
    a = 1.0 + O["clearance"]; steps = int(S.get("stair_steps", 0)); tr, ri = 0.5, 0.27      # crepidoma: tread and riser — in D
    Dlo, Dhi = S["D"]
    if kind == "tholos":
        n = int(rng.integers(S["n_columns"][0], S["n_columns"][1] + 1)); R1 = 0.5 * min(fr["L"], fr["W"])
        # radius of the column ring: the pitch along the circumference is a·D, with steps and a cornice around
        D = min(Dhi, R1 / (n * a / (2 * math.pi) + 0.85 + steps * tr))
        if D < Dlo * 0.8: return None
        Rc = n * a * D / (2 * math.pi)
        P.update(D=round(D, 3), n=n, Rc=round(Rc, 3), steps=steps)
        Rt = Rc + 0.85 * D + steps * tr * D
        z0, zb = _ground(H, [F.w(Rt * math.cos(t), Rt * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12)], ctx.get("z"))
        P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(2 * Rt, 2), W=round(2 * Rt, 2)); return P
    fw = sorted(int(k) for k in S["front"]); nf = int(_pick({int(k): v for k, v in S["front"].items()}, rng))
    order = [nf] + [x for x in fw[::-1] if x < nf]                                  # does not fit — fewer columns
    for nf in order:
        wd = (nf - 1) * a + 1.4 + 2 * steps * tr                                      # crepidoma width in D
        if kind == "peripteros":
            Dw = fr["W"] / wd
            ns_max = int((fr["L"] / Dw - 1.4 - 2 * steps * tr) / a) + 1
            ns = int(np.clip(ns_max, nf + 3, 2 * nf + 1))
            ld = (ns - 1) * a + 1.4 + 2 * steps * tr
        else:
            ns = 0; ratio = _u(S.get("length", [1.8, 1.8]), rng); ld = wd * ratio
        if kind == "podium": wd += 1.0; ld += 1.0
        D = min(fr["W"] / wd, fr["L"] / ld, Dhi)
        if D >= Dlo: break
    else:
        return None
    if kind == "peripteros": ns = int(np.clip(int((fr["L"] / D - 1.4 - 2 * steps * tr) / a) + 1, nf + 3, 2 * nf + 1))
    hw = (nf - 1) * a * D / 2                                                        # half-width along the column axes
    if kind == "peripteros": hl = (ns - 1) * a * D / 2
    else: hl = (ld - 2 * steps * tr - 1.4 - (1.0 if kind == "podium" else 0.0)) * D / 2
    P.update(D=round(D, 3), a=round(a * D, 3), nf=nf, ns=ns, hw=round(hw, 3), hl=round(hl, 3), steps=steps)
    if kind == "podium": P["podium"] = round(_u(S["podium"], rng), 2)
    ext = hl + 0.7 * D + steps * tr * D; exw = hw + 0.7 * D + steps * tr * D
    z0, zb = _ground(H, _sample(F, ext, exw), ctx.get("z"))
    P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(2 * ext, 2), W=round(2 * exw, 2))
    if kind == "propylon": P["doors"] = 1 if nf <= 4 else 3
    return P


def _stoa(S, F, fr, ctx, rng, P, O, H):
    Ls = float(fr["W"]); Ds = float(min(fr["L"], _u(S["depth"], rng) if fr["L"] > S["depth"][1] else fr["L"]))
    if Ls < 12 or Ds < 5: return None
    D = float(np.clip(Ds / 13.5, *S["D"])); a = (1.0 + O["clearance"]) * D
    steps = int(S.get("stair_steps", 2))
    n = max(3, int(round((Ls - 1.4 * D) / a)) + 1)
    F2 = Fr(F.w(-fr["L"] / 2 + Ds / 2, 0.0), F.u)                                   # stoa — the back wall toward the frame edge (the street), the front toward the square
    P.update(c=[round(float(v), 3) for v in F2.c], D=round(D, 3), a=round((Ls - 1.4 * D) / (n - 1), 3), n=n, Ls=round(Ls, 2), Ds=round(Ds, 2), steps=steps,
             two=bool(Ds >= float(S["two_naves"])), shops=bool(rng.random() < float(S.get("shops", 0.0))))
    z0, zb = _ground(H, _sample(F2, Ds / 2, Ls / 2), ctx.get("z")); P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(Ds, 2), W=round(Ls, 2))
    return P


def _bouleuterion(S, F, fr, ctx, rng, P, O, H):
    W = min(fr["W"], 26.0); L = min(fr["L"], W * _u([0.9, 1.15], rng), 28.0)
    if W < 10 or L < 10: return None
    porch = rng.random() < float(S.get("portico", 0.5)) and fr["L"] > L + 4.5
    D = _u(S["D"], rng); hgt = _u(S["height"], rng)
    pd = 1.6 * (1 + O["clearance"]) * D if porch else 0.0
    if porch: L = min(L, fr["L"] - pd - 1.0)
    P.update(D=round(D, 3), Lh=round(L, 2), Wh=round(W, 2), h=round(hgt, 2), porch=bool(porch), pd=round(pd, 2),
             c=[round(float(v), 3) for v in F.w(-pd / 2, 0.0)])
    z0, zb = _ground(H, _sample(Fr(P["c"], F.u), L / 2 + pd, W / 2), ctx.get("z")); P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(L + pd, 2), W=round(W, 2))
    return P


def _palaestra(S, F, fr, ctx, rng, P, O, H):
    L = float(min(fr["L"], 70.0)); W = float(min(fr["W"], 70.0)); dr = _u(S["rooms"], rng)
    if min(L, W) < 2 * dr + 10: return None
    D = _u(S["D"], rng); hgt = _u(S["height"], rng)
    P.update(D=round(D, 3), Lp=round(L, 2), Wp=round(W, 2), dr=round(dr, 2), h=round(hgt, 2), a=round((1 + O["clearance"]) * D, 3))
    z0, zb = _ground(H, _sample(F, L / 2, W / 2), ctx.get("z")); P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(L, 2), W=round(W, 2))
    return P


def _altar(S, F, fr, ctx, rng, P, O, H):
    L = float(np.clip(fr["L"], 3.0, 9.0)); W = float(np.clip(fr["W"], 4.0, 14.0)); h = _u(S["height"], rng)
    P.update(La=round(L, 2), Wa=round(W, 2), h=round(h, 2))
    z0, zb = _ground(H, _sample(F, L / 2 + 2.0, W / 2), ctx.get("z")); P.update(z0=round(z0, 3), zb=round(zb, 3), L=round(L + 2.5, 2), W=round(W, 2))
    return P


def theatre_shape(S, R, rng):
    """Theatre dimensions from the cavea radius: orchestra, rows, diazoma, stairs, opening angle."""
    ro = float(np.clip(_u(S["orchestra"], rng) * R / 30.0, 6.0, 12.0)); rise, dep = S["row"]["rise"], S["row"]["depth"]
    rows = max(8, int((R - ro - 1.0) / dep)); dia = int(S["diazoma"]) if rows > S["diazoma"] + 6 else 0
    if dia: rows = max(8, int((R - ro - 1.0 - 1.4) / dep))
    return {"ro": round(ro, 2), "rise": rise, "dep": dep, "rows": rows, "dia": dia, "span": round(_u(S["mortar"], rng), 1),
            "stairs": int(rng.integers(S["n_stairs"][0], S["n_stairs"][1] + 1))}


def cavea_z(T, r):
    """Top of a cavea row at distance r from the orchestra center (the orchestra level is 0)."""
    k = np.floor((np.asarray(r, float) - T["ro"] - 1.0 - (np.asarray(r, float) > T["ro"] + 1.0 + T["dia"] * T["dep"]) * (1.4 if T["dia"] else 0.0)) / T["dep"])
    return np.where(np.asarray(r) < T["ro"] + 1.0, 0.0, 0.4 + np.clip(k, 0, None) * T["rise"])


def _theatre(S, F, fr, ctx, rng, P, O, H):
    T = fr.get("shape") or theatre_shape(S, float(fr["R"]), rng); zo = float(fr.get("z", ctx.get("z") or 0.0))
    sk = S["skene"]; ds = _u(sk["depth"], rng); hs = _u(sk["height"], rng); pk = _u(sk["proskenion"], rng)
    P.update(T=T, zo=round(zo, 3), R=round(float(fr["R"]), 2), ds=round(ds, 2), hs=round(hs, 2), pk=round(pk, 2), D=0.5)
    span = math.radians(T["span"]); th = math.atan2(F.u[1], F.u[0]); rows = []
    H0 = ctx.get("height0") or H
    for k in range(T["rows"]):                                                       # the bottom of a row follows the ground under it
        r0 = T["ro"] + 1.0 + k * T["dep"] + (1.4 if T["dia"] and k >= T["dia"] else 0.0); r1 = r0 + T["dep"] + (1.4 if T["dia"] and k == T["dia"] - 1 else 0.0)
        ang = np.linspace(th - span / 2, th + span / 2, 9); rm = (r0 + r1) / 2
        X = F.c[0] + rm * np.cos(ang); Y = F.c[1] + rm * np.sin(ang)
        zg = float(np.min(H(X, Y))) if H is not None else zo
        zt = zo + 0.4 + k * T["rise"]; rows.append([round(r0, 3), round(r1, 3), round(min(zg - 0.3, zt - T["rise"]), 3), round(zt, 3)])
    P["rows"] = rows
    back = []
    if H0 is not None:                                                               # retaining wall behind the cavea — down to the slope ground (smoothed)
        r = rows[-1][1] + 1.2; ang = np.linspace(th - span / 2, th + span / 2, 33)
        zt = np.asarray(H0(F.c[0] + r * np.cos(ang), F.c[1] + r * np.sin(ang)), float)
        zt = np.array([zt[max(0, i - 2):i + 3].max() for i in range(len(zt))]); zt = np.convolve(np.pad(zt, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        back = [[round(float(t), 4), round(float(z), 3)] for t, z in zip(ang, zt)]
    P["back"] = back
    P.update(z0=round(zo, 3), zb=round(zo - 0.5, 3), L=round(2 * float(fr["R"]), 2), W=round(2 * float(fr["R"]), 2))
    return P


def footprint(P):
    """Building outline on the ground (world): the frame rectangle; a theatre — the cavea, orchestra and skene; a
    tholos — a circle."""
    F = Fr(P["c"], P["u"])
    if P["kind"] == "theatre":
        T = P["T"]; th = math.atan2(F.u[1], F.u[0]); sp = math.radians(T["span"]); R = P["rows"][-1][1] + 0.8
        arc = [F.c + R * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(th - sp / 2, th + sp / 2, 40)]
        back = F.w(-(T["ro"] + 2.0 + P["pk"] + P["ds"]), 0.0); ws = T["ro"] + 5.0
        sk = [F.w(-(T["ro"] + 2.0 + P["pk"] + P["ds"]), -ws), F.w(-(T["ro"] + 2.0 + P["pk"] + P["ds"]), ws)]
        pts = arc + [sk[1], sk[0]]
        return [[round(float(q[0]), 2), round(float(q[1]), 2)] for q in pts]
    if P["kind"] == "tholos":
        R = P["Rc"] + 0.85 * P["D"] + P["steps"] * 0.5 * P["D"]
        return [[round(float(q[0]), 2), round(float(q[1]), 2)] for q in (F.w(R * math.cos(t), R * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False))]
    return F.rect(-P["L"] / 2, P["L"] / 2, -P["W"] / 2, P["W"] / 2)


# ---------------------------------------------------------------- geometry (numpy)
class _Out:
    def __init__(self): self.d = {"prisms": [], "columns": [], "walls": [], "faces": [], "bars": [], "cones": [], "boxes": []}

    def prism(self, poly, z0, z1, col, role="finish", bot=False):
        self.d["prisms"].append({"poly": [[float(x), float(y)] for x, y in poly], "z0": float(z0), "z1": float(z1), "col": col, "role": role, "bot": bot})

    def column(self, c, r, z0, z1, order, ax, col, cap=None):
        self.d["columns"].append({"c": [float(c[0]), float(c[1])], "r": float(r), "z0": float(z0), "z1": float(z1), "order": order,
                                  "ax": [float(ax[0]), float(ax[1])], "cap": cap, "col": col})

    def wall(self, p0, p1, z0, top, t, holes, col, role="wall"):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); L = float(np.linalg.norm(p1 - p0))
        if L < 0.2: return
        top = [(0.0, float(top)), (L, float(top))] if np.isscalar(top) else [(float(s), float(z)) for s, z in top]
        self.d["walls"].append({"p0": p0.tolist(), "p1": p1.tolist(), "z0": float(z0), "top": top, "t": float(t), "holes": holes, "col": col, "role": role})

    def face(self, pts, col, role="roof", band=0.0):
        self.d["faces"].append({"pts": [tuple(map(float, p)) for p in pts], "col": col, "role": role, "band": float(band)})

    def bar(self, p, q, w, col, role="finish"):
        self.d["bars"].append({"p": list(map(float, p)), "q": list(map(float, q)), "w": float(w), "col": col, "role": role})

    def box(self, c, d, w, l, z0, z1, col, role="finish"):
        self.d["boxes"].append({"c": [float(c[0]), float(c[1])], "d": [float(d[0]), float(d[1])], "w": float(w), "l": float(l), "z0": float(z0), "z1": float(z1), "col": col, "role": role})


def _gable(o, F, x0, x1, yr, z_e, pitch, col, ov=0.25, band=0.22):
    """Gable roof along the frame axis over [x0, x1] × [−yr, yr]: eaves on the long sides at z_e, the ridge over
    the axis."""
    zr = z_e + yr * math.tan(math.radians(pitch))
    for s in (-1, 1):
        o.face([F.p3(x0 - ov, s * yr, z_e), F.p3(x1 + ov, s * yr, z_e), F.p3(x1 + ov, 0.0, zr), F.p3(x0 - ov, 0.0, zr)][:: (1 if s > 0 else -1)], col, "roof", band)
    return zr


def _pediment(o, F, x, xin, yr, z_e, pitch, cs, cr, D):
    """Pediment at the end x (xin points inward): a recessed tympanum, a raking cornice of beams, acroteria."""
    zr = z_e + yr * math.tan(math.radians(pitch)); s = 1 if xin < x else -1
    xt = x + (0.12 * D if s > 0 else -0.12 * D)
    tri = [F.p3(xt, -yr + 0.25 * D, z_e), F.p3(xt, yr - 0.25 * D, z_e), F.p3(xt, 0.0, zr - 0.18 * D)]
    o.face(tri if s < 0 else tri[::-1], cs, "finish")
    tri2 = [F.p3(xin, -yr + 0.25 * D, z_e), F.p3(xin, yr - 0.25 * D, z_e), F.p3(xin, 0.0, zr - 0.18 * D)]
    o.face(tri2[::-1] if s < 0 else tri2, cs, "finish")
    o.bar(F.p3(x, -yr, z_e + 0.12 * D), F.p3(x, 0.0, zr + 0.05 * D), 0.32 * D, cs)
    o.bar(F.p3(x, yr, z_e + 0.12 * D), F.p3(x, 0.0, zr + 0.05 * D), 0.32 * D, cs)
    for y, z in ((-yr, z_e), (yr, z_e), (0.0, zr)):                                 # acroteria — blocks at the pediment corners
        q = F.w(x - s * 0.2 * D, y); o.box(q, F.u, 0.45 * D, 0.45 * D, z + 0.1 * D, z + 0.75 * D, cs)


def _entablature(o, F, x0, x1, yr, z_cap, O, D, col, sides="all"):
    """Architrave, frieze (for Doric — triglyphs at the half-span pitch, corner ones at the corner), a painted
    band, a cornice → the top of the cornice."""
    ar, fz, cr = O["architrave"] * D, O["frieze"] * D, O["cornice"] * D; cs = col["stone"]
    o.prism(F.rect(x0, x1, -yr, yr), z_cap, z_cap + ar, cs, bot=True)
    o.prism(F.rect(x0 + 0.04 * D, x1 - 0.04 * D, -yr + 0.04 * D, yr - 0.04 * D), z_cap + ar, z_cap + ar + fz, cs)
    if O.get("triglyphs"):
        tw = 0.44 * D; zt0, zt1 = z_cap + ar, z_cap + ar + fz * 0.98
        edges = [((x0, -yr), (x1, -yr), (0, -1)), ((x1, -yr), (x1, yr), (1, 0)), ((x1, yr), (x0, yr), (0, 1)), ((x0, yr), (x0, -yr), (-1, 0))]
        for (ax_, ay_), (bx_, by_), (nx_, ny_) in edges:
            L = math.hypot(bx_ - ax_, by_ - ay_); k = max(1, int(round(L / (0.75 * (1 + O["clearance"]) * D))))
            for i in range(k + 1):
                f = (tw / 2 + (L - tw) * i / k) / L; x = ax_ + (bx_ - ax_) * f; y = ay_ + (by_ - ay_) * f
                d = F.u * (bx_ - ax_) + F.v * (by_ - ay_); q = F.w(x + nx_ * 0.04 * D, y + ny_ * 0.04 * D)
                o.box(q, d, 0.1 * D, tw, zt0, zt1, col["painting"])
    o.prism(F.rect(x0 - 0.02 * D, x1 + 0.02 * D, -yr - 0.02 * D, yr + 0.02 * D), z_cap + ar + fz - 0.07 * D, z_cap + ar + fz, col["border"])
    o.prism(F.rect(x0 - 0.3 * D, x1 + 0.3 * D, -yr - 0.3 * D, yr + 0.3 * D), z_cap + ar + fz, z_cap + ar + fz + cr, cs, bot=True)
    return z_cap + ar + fz + cr


def _krepis(o, F, hx0, hx1, hy, z0, zb, steps, rise, tread, col):
    """Crepidoma: steps around the stylobate [hx0, hx1] × [−hy, hy]; the lowest goes down to below the ground."""
    for k in range(steps):
        e = (steps - 1 - k) * tread
        o.prism(F.rect(hx0 - e, hx1 + e, -hy - e, hy + e), zb if k == 0 else z0 + k * rise, z0 + (k + 1) * rise, col)
    return z0 + steps * rise


def _door_wall(o, F, x, y0, y1, z0, z1, t, doors, col):
    """Transverse wall at x from y0 to y1 with openings doors [(y, width, height)] (a stub is dark, set deep)."""
    p0 = F.w(x, y0); p1 = F.w(x, y1); L = abs(y1 - y0)
    holes = [((y - y0) if y1 > y0 else (y0 - y), z0 + 0.02, w, h, "rect", "opening") for y, w, h in doors]
    o.wall(p0, p1, z0, z1, t, holes, col)


def _temple_geo(P, o):
    F = Fr(P["c"], P["u"]); O = P["ord"]; D = P["D"]; col = P["col"]; cs = col["stone"]; kind = P["kind"]; ordn = P["order"]
    steps = P["steps"]; rise = 0.27 * D; tread = 0.5 * D; z0, zb = P["z0"], P["zb"]
    H_ = O["column"] * D; r = D / 2
    if kind == "tholos":
        Rc = P["Rc"]; n = P["n"]; ring = lambda R_, m=32: [F.w(R_ * math.cos(t), R_ * math.sin(t)) for t in np.linspace(0, 2 * math.pi, m, endpoint=False)]
        for k in range(steps):
            o.prism(ring(Rc + 0.7 * D + (steps - 1 - k) * tread), zb if k == 0 else z0 + k * rise, z0 + (k + 1) * rise, cs)
        zs = z0 + steps * rise; zc = zs + H_
        for i in range(n):
            t = 2 * math.pi * (i + 0.5) / n; c = F.w(Rc * math.cos(t), Rc * math.sin(t)); o.column(c, r, zs, zc, ordn, (-math.sin(t), math.cos(t)), cs)
        o.prism(ring(Rc - 1.0 * (1 + O["clearance"]) * D, 24), zs, zc + O["architrave"] * D, col["wall"], "wall")
        ze = zc + (O["architrave"] + O["frieze"]) * D
        o.prism(ring(Rc + 0.55 * D), zc, ze, cs, bot=True); o.prism(ring(Rc + 0.85 * D), ze, ze + O["cornice"] * D, cs, bot=True)
        ze += O["cornice"] * D
        o.d["cones"].append({"ring": [list(map(float, q)) for q in ring(Rc + 0.9 * D)], "z": float(ze), "h": float((Rc + 0.9 * D) * math.tan(math.radians(22))), "col": col["roof"]})
        return
    nf, ns, a, hw, hl = P["nf"], P["ns"], P["a"], P["hw"], P["hl"]
    podium = kind == "podium"; prost = kind in ("prostyle", "amphiprostyle", "in_antis", "podium", "propylon")
    t = 0.55 * D
    # platform: peripteros — around the columns; prostyle — a portico in front, the cella behind (the whole length
    # hl)
    sx0, sx1, sy = -hl - 0.7 * D, hl + 0.7 * D, hw + 0.7 * D
    if podium:
        hp = P["podium"]; zs = z0 + hp
        o.prism(F.rect(sx0 - 0.4 * D, sx1, -sy - 0.4 * D, sy + 0.4 * D), zb, zs - 0.35 * D, cs)
        o.prism(F.rect(sx0 - 0.55 * D, sx1, -sy - 0.55 * D, sy + 0.55 * D), zs - 0.35 * D, zs, cs)       # podium cornice
        n_st = max(3, int(round(hp / 0.18))); tr_ = 0.32; x_end = sx1 + n_st * tr_
        for i in range(n_st):                                                         # stair along the facade between the cheek walls
            xa = sx1 + (n_st - 1 - i) * tr_; o.prism(F.rect(xa, xa + tr_ + 0.02, -hw, hw), zb, z0 + (i + 1) * hp / n_st, cs)
        for s in (-1, 1): o.prism(F.rect(sx1, x_end, s * hw + (0 if s > 0 else -0.9 * D), s * hw + (0.9 * D if s > 0 else 0)), zb, zs - 0.2, cs)
    else:
        zs = _krepis(o, F, sx0, sx1, sy, z0, zb, steps, rise, tread, cs)
    zc = zs + H_
    cols = []
    if kind == "peripteros":
        for i in range(nf):
            y = -hw + 2 * hw * i / (nf - 1); cols += [(hl, y), (-hl, y)]
        for j in range(1, ns - 1):
            x = -hl + 2 * hl * j / (ns - 1); cols += [(x, hw), (x, -hw)]
    else:
        ys = [-hw + 2 * hw * i / max(nf - 1, 1) for i in range(nf)] if nf > 1 else [0.0]
        if kind == "in_antis": ys = [-hw / 3, hw / 3]
        cols += [(hl, y) for y in ys]
        if kind in ("amphiprostyle", "propylon"): cols += [(-hl, y) for y in ys]
    for x, y in cols: o.column(F.w(x, y), r, zs, zc, ordn, F.v, cs)
    # cella
    ar = O["architrave"] * D; zw = zc + ar
    if kind == "peripteros":
        cy = hw - a + 0.15 * D; cx0, cx1 = -hl + 1.25 * a, hl - 1.25 * a; pro = 0.9 * a
        for s in (-1, 1): o.wall(F.w(cx0, s * cy), F.w(cx1, s * cy), zs, zw, t, [], col["wall"])
        _door_wall(o, F, cx1 - pro, -cy - t / 2, cy + t / 2, zs, zw, t, [(0.0, min(0.36 * 2 * cy, 2.2 * D), min(0.62 * H_, 7.5))], col["wall"])
        _door_wall(o, F, cx0 + pro, cy + t / 2, -cy - t / 2, zs, zw, t, [], col["wall"])
        for x in (cx1, cx0):                                                          # two columns in antis — pronaos and opisthodomos
            for y in (-cy / 3, cy / 3): o.column(F.w(x, y), r * 0.9, zs, zc, ordn, F.v, cs)
        x_e0, x_e1, ye = -hl - 0.55 * D, hl + 0.55 * D, hw + 0.55 * D
    elif kind == "propylon":
        cy = hw + 0.15 * D
        for s in (-1, 1): o.wall(F.w(-hl + 0.6 * a, s * cy), F.w(hl - 0.6 * a, s * cy), zs, zw, t, [], col["wall"])
        nd = P["doors"]; w0 = min(0.55 * a, 3.6); ds_ = [(0.0, w0 * 1.25, min(0.7 * H_, 6.5))] + [(s * a * k, w0, min(0.55 * H_, 5.0)) for k in range(1, nd // 2 + 1) for s in (-1, 1)]
        _door_wall(o, F, 0.0, -cy - t / 2, cy + t / 2, zs, zw, 1.2 * t, ds_, col["wall"])
        x_e0, x_e1, ye = -hl - 0.55 * D, hl + 0.55 * D, hw + 0.55 * D
    else:
        cy = hw + (0.25 * D if kind != "in_antis" else 0.0) if not podium else hw + 0.1 * D
        porch = (1.7 if kind != "podium" else 3.2) * a * (0.6 if kind == "in_antis" else 1.0)
        cx0 = -hl + (porch if kind == "amphiprostyle" else 0.3 * D); cx1 = hl - porch if kind != "in_antis" else hl + 0.3 * D
        xa1 = hl if kind == "in_antis" else cx1
        for s in (-1, 1): o.wall(F.w(cx0, s * cy), F.w(xa1, s * cy), zs, zw, t, [], col["wall"])
        dx = cx1 - (porch * 0.6 if kind == "in_antis" else 0.0)
        _door_wall(o, F, dx, -cy - t / 2, cy + t / 2, zs, zw, t, [(0.0, min(0.4 * 2 * cy, 2.4 * D), min(0.62 * H_, 6.5))], col["wall"])
        _door_wall(o, F, cx0, cy + t / 2, -cy - t / 2, zs, zw, t, [], col["wall"])
        if podium:                                                                    # half-columns along the cella walls and the rear
            k = max(2, int(round((cx1 - cx0) / a)))
            for i in range(k + 1):
                x = cx0 + (cx1 - cx0) * i / k
                for s in (-1, 1): o.column(F.w(x, s * (cy + t / 2)), r * 0.85, zs, zc, ordn, F.v, cs)
            m = max(2, int(round(2 * cy / a)))
            for i in range(1, m): o.column(F.w(cx0 - t / 2, -cy + 2 * cy * i / m), r * 0.85, zs, zc, ordn, F.u, cs)
        x_e0, x_e1, ye = cx0 - t / 2 - 0.05 * D, hl + 0.55 * D, cy + t / 2 + 0.05 * D
    ze = _entablature(o, F, x_e0, x_e1, ye, zc, O, D, col)
    yr = ye + 0.3 * D; pitch = O["pediment"]
    _gable(o, F, x_e0 - 0.3 * D, x_e1 + 0.3 * D, yr + 0.1 * D, ze, pitch, col["roof"], ov=0.12 * D, band=0.18 * D)
    _pediment(o, F, x_e1 + 0.3 * D, x_e1 - 0.4 * D, yr, ze, pitch, cs, col["roof"], D)
    _pediment(o, F, x_e0 - 0.3 * D, x_e0 + 0.4 * D, yr, ze, pitch, cs, col["roof"], D)


def _stoa_geo(P, o):
    F = Fr(P["c"], P["u"]); O = P["ord"]; D = P["D"]; col = P["col"]; cs = col["stone"]; Ls, Ds = P["Ls"], P["Ds"]
    z0, zb = P["z0"], P["zb"]; rise, tread = 0.27 * D, 0.5 * D; steps = P["steps"]; hy = Ls / 2
    for k in range(steps):                                                          # steps — along the facade
        e = (steps - 1 - k) * tread
        o.prism(F.rect(-Ds / 2, Ds / 2 - (steps - 1) * tread + e, -hy, hy), zb if k == 0 else z0 + k * rise, z0 + (k + 1) * rise, cs)
    zs = z0 + steps * rise; H_ = O["column"] * D; zc = zs + H_; t = 0.6
    xc = Ds / 2 - (steps - 1) * tread - 0.7 * D; xb = -Ds / 2 + t / 2
    for j in range(P["n"]):
        y = -(hy - 0.7 * D) + P["a"] * j; o.column(F.w(xc, y), D / 2, zs, zc, P["order"], F.v, cs)
    ze = _entablature(o, F, xc - 0.55 * D, xc + 0.55 * D, hy - 0.1, zc, O, D, col)
    pitch = 16.0; xe_f = xc + 0.85 * D; xe_b = -Ds / 2 - 0.35; xm = (xe_f + xe_b) / 2; half = (xe_f - xe_b) / 2; zr = ze + half * math.tan(math.radians(pitch))
    for s, xe in ((1, xe_f), (-1, xe_b)):
        pts = [F.p3(xe, -hy - 0.3, ze), F.p3(xe, hy + 0.3, ze), F.p3(xm, hy + 0.3, zr), F.p3(xm, -hy - 0.3, zr)]
        o.face(pts if s > 0 else pts[::-1], col["roof"], "roof", 0.18)
    roof_z = lambda x: ze + (half - abs(x - xm)) * math.tan(math.radians(pitch))
    holes = []
    if P["shops"]:                                                                  # shop doors in the back wall
        k = max(1, int(Ls / 4.6))
        holes = [((i + 0.5) * Ls / k, zs + 0.02, 1.3, 2.5, "rect", "opening") for i in range(k)]
    o.wall(F.w(xb, hy), F.w(xb, -hy), zs - 0.02, ze, t, holes, col["wall"])          # outside is the rear (−u)
    for s in (-1, 1):                                                               # ends: a wall with a gable up to the ridge
        xa, xb2 = xb - t / 2, xc + 0.3 * D; L = xb2 - xa; top = [(0.0, roof_z(xa)), (xm - xa, zr - 0.05), (L, roof_z(xb2))] if xa < xm < xb2 else [(0.0, roof_z(xa)), (L, roof_z(xb2))]
        p0, p1 = (F.w(xa, s * (hy - t / 2)), F.w(xb2, s * (hy - t / 2)))
        if s > 0: o.wall(p0, p1, zs - 0.02, top, t, [], col["wall"])
        else: o.wall(p1, p0, zs - 0.02, [(L - q[0], q[1]) for q in top[::-1]], t, [], col["wall"])
    if P["two"]:                                                                    # inner row — Ionic, every other one, up to the ridge
        xi = (xc + xb) / 2
        for j in range(1, P["n"] - 1, 2):
            y = -(hy - 0.7 * D) + P["a"] * j; o.column(F.w(xi, y), D * 0.42, zs, roof_z(xi) - 0.25, "ionic", F.v, cs)
    o.prism(F.rect(xb, xc + 0.6 * D, -hy, hy), zs - 0.05, zs + 0.01, col["floor"], "floor")


def _bouleuterion_geo(P, o):
    F = Fr(P["c"], P["u"]); O = P["ord"]; D = P["D"]; col = P["col"]; cs = col["stone"]; L, W = P["Lh"], P["Wh"]; z0, zb = P["z0"], P["zb"]
    zs = _krepis(o, F, -L / 2, L / 2, W / 2, z0, zb, 2, 0.22, 0.4, cs); ze = zs + P["h"]; t = 0.8
    holes_side = lambda Lw: [(s, ze - 2.6, 1.2, 1.5, "rect", "opening") for s in np.arange(2.2, Lw - 1.5, 3.2)]
    corners = [F.w(-L / 2 + t / 2, -W / 2 + t / 2), F.w(L / 2 - t / 2, -W / 2 + t / 2), F.w(L / 2 - t / 2, W / 2 - t / 2), F.w(-L / 2 + t / 2, W / 2 - t / 2)]
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]; Lw = float(np.linalg.norm(q - p))
        hs = holes_side(Lw)
        if i == 1: hs = [h for h in hs if abs(h[0] - Lw / 2) > 3.5] + [(Lw / 2 - 1.6, zs + 0.02, 1.6, 3.2, "rect", "opening"), (Lw / 2 + 1.6, zs + 0.02, 1.6, 3.2, "rect", "opening")]
        o.wall(p, q, zs - 0.02, ze, t, hs, col["wall"])
    o.prism(F.rect(-L / 2 - 0.1, L / 2 + 0.1, -W / 2 - 0.1, W / 2 + 0.1), ze, ze + 0.5, cs)
    ze += 0.5; p_ = 14.0; hz = min(L, W) / 2 * math.tan(math.radians(p_)); ov = 0.5
    a0 = F.p3(-L / 2 - ov, -W / 2 - ov, ze); a1 = F.p3(L / 2 + ov, -W / 2 - ov, ze); a2 = F.p3(L / 2 + ov, W / 2 + ov, ze); a3 = F.p3(-L / 2 - ov, W / 2 + ov, ze)
    if L >= W:
        r0 = F.p3(-L / 2 + W / 2, 0.0, ze + hz); r1 = F.p3(L / 2 - W / 2, 0.0, ze + hz)
        for f in ([a0, a1, r1, r0], [a1, a2, r1], [a2, a3, r0, r1], [a3, a0, r0]): o.face(f, col["roof"], "roof", 0.2)
    else:
        r0 = F.p3(0.0, -W / 2 + L / 2, ze + hz); r1 = F.p3(0.0, W / 2 - L / 2, ze + hz)
        for f in ([a0, a1, r0], [a1, a2, r1, r0], [a2, a3, r1], [a3, a0, r0, r1]): o.face(f, col["roof"], "roof", 0.2)
    if P["porch"]:                                                                  # portico: four columns and a pediment
        pd = P["pd"]; hw = min(W * 0.32, 7.0); a = 2 * hw / 3; H_ = P["h"] - 0.9; zc = zs + H_
        F2 = Fr(F.w(L / 2 + pd / 2, 0.0), F.u)
        o.prism(F2.rect(-pd / 2, pd / 2, -hw - 0.7 * D, hw + 0.7 * D), zb, zs, cs)
        for i in range(4): o.column(F2.w(pd / 2 - 0.7 * D, -hw + a * i), D / 2, zs, zc, P["order"], F.v, cs)
        ze2 = _entablature(o, F2, -pd / 2 - 0.3, pd / 2 - 0.15 * D, hw + 0.55 * D, zc, O, D, col)
        _gable(o, F2, -pd / 2 - 0.3, pd / 2, hw + 0.85 * D, ze2, O["pediment"], col["roof"], ov=0.1, band=0.16)
        _pediment(o, F2, pd / 2 + 0.25 * D, pd / 2 - 0.5 * D, hw + 0.85 * D, ze2, O["pediment"], cs, col["roof"], D)


def _palaestra_geo(P, o):
    F = Fr(P["c"], P["u"]); O = P["ord"]; D = P["D"]; col = P["col"]; cs = col["stone"]; L, W, dr = P["Lp"], P["Wp"], P["dr"]; z0, zb = P["z0"], P["zb"]
    t = 0.7; zt = z0 + P["h"]; pc = 3.0                                             # depth of the portico at the court
    o.prism(F.rect(-L / 2, L / 2, -W / 2, W / 2), zb, z0 + 0.05, col["stone"])
    outer = [F.w(-L / 2 + t / 2, -W / 2 + t / 2), F.w(L / 2 - t / 2, -W / 2 + t / 2), F.w(L / 2 - t / 2, W / 2 - t / 2), F.w(-L / 2 + t / 2, W / 2 - t / 2)]
    for i in range(4):
        p, q = outer[i], outer[(i + 1) % 4]; Lw = float(np.linalg.norm(q - p))
        hs = [(s, zt - 1.9, 0.8, 0.9, "rect", "opening") for s in np.arange(3.0, Lw - 2.0, 4.5)]
        if i == 1: hs = [h for h in hs if abs(h[0] - Lw / 2) > 3.0] + [(Lw / 2, z0 + 0.05, 2.8, 3.6, "rect", "opening")]
        o.wall(p, q, z0 - 0.05, zt, t, hs, col["wall"])
    cx, cy = L / 2 - dr, W / 2 - dr                                                 # court
    zc_top = zt - (dr - t) * math.tan(math.radians(16)); H_ = max(3.2, zc_top - z0 - (O["architrave"] + O["frieze"] + O["cornice"]) * D)
    zc = z0 + H_; a = P["a"]
    for (x0_, y0_), (x1_, y1_) in (((-cx, -cy), (cx, -cy)), ((cx, -cy), (cx, cy)), ((cx, cy), (-cx, cy)), ((-cx, cy), (-cx, -cy))):
        Lr = math.hypot(x1_ - x0_, y1_ - y0_); k = max(2, int(round(Lr / a)))
        for i in range(k):
            f = i / k; o.column(F.w(x0_ + (x1_ - x0_) * f, y0_ + (y1_ - y0_) * f), D / 2, z0 + 0.05, zc, P["order"], F.u * (x1_ - x0_) + F.v * (y1_ - y0_), cs)
    ze = zc + (O["architrave"] + O["frieze"]) * D; b = 0.55 * D
    for x0_, x1_, y0_, y1_ in ((-cx - b, cx + b, -cy - b, -cy + b), (-cx - b, cx + b, cy - b, cy + b), (-cx - b, -cx + b, -cy + b, cy - b), (cx - b, cx + b, -cy + b, cy - b)):
        o.prism(F.rect(x0_, x1_, y0_, y1_), zc, ze, cs, bot=True)                     # a ring beam over the court columns
    o.face([F.p3(-cx + 0.5 * D, -cy + 0.5 * D, z0 + 0.06), F.p3(cx - 0.5 * D, -cy + 0.5 * D, z0 + 0.06), F.p3(cx - 0.5 * D, cy - 0.5 * D, z0 + 0.06),
            F.p3(-cx + 0.5 * D, cy - 0.5 * D, z0 + 0.06)], col["sand"], "floor")
    O_ = [F.w(-L / 2 - 0.4, -W / 2 - 0.4), F.w(L / 2 + 0.4, -W / 2 - 0.4), F.w(L / 2 + 0.4, W / 2 + 0.4), F.w(-L / 2 - 0.4, W / 2 + 0.4)]
    I_ = [F.w(-cx - 0.55 * D, -cy - 0.55 * D), F.w(cx + 0.55 * D, -cy - 0.55 * D), F.w(cx + 0.55 * D, cy + 0.55 * D), F.w(-cx - 0.55 * D, cy + 0.55 * D)]
    for i in range(4):                                                              # roof sloping toward the court
        j = (i + 1) % 4; o.face([(*O_[i], zt + 0.1), (*O_[j], zt + 0.1), (*I_[j], ze), (*I_[i], ze)], col["roof"], "roof", 0.18)
    ins = [F.w(-cx - pc, -cy - pc), F.w(cx + pc, -cy - pc), F.w(cx + pc, cy + pc), F.w(-cx - pc, cy + pc)]
    for i in range(4):                                                              # the wall of the rooms behind the portico — doors
        p, q = ins[i], ins[(i + 1) % 4]; Lw = float(np.linalg.norm(q - p)); zi = ze + (pc - 0.55 * D) * math.tan(math.radians(16))
        o.wall(q, p, z0, zi, 0.5, [(s, z0 + 0.05, 1.4, 2.6, "rect", "opening") for s in np.arange(3.0, Lw - 2.0, 6.0)], col["wall"])


def _altar_geo(P, o):
    F = Fr(P["c"], P["u"]); col = P["col"]; cs = col["stone"]; L, W, h = P["La"], P["Wa"], P["h"]; z0, zb = P["z0"], P["zb"]
    o.prism(F.rect(-L / 2, L / 2, -W / 2, W / 2), zb, z0 + h, cs)
    n = max(3, int(round(h / 0.2)))
    for i in range(n):                                                              # steps — toward the temple (−u)
        x = -L / 2 - (n - i) * 0.3; o.prism(F.rect(x, x + 0.32, -W / 2 + 0.6, W / 2 - 0.6), zb, z0 + (i + 1) * h / n, cs)
    for (x0_, x1_, y0_, y1_) in ((L / 2 - 0.45, L / 2, -W / 2, W / 2), (-L / 2, L / 2, -W / 2, -W / 2 + 0.45), (-L / 2, L / 2, W / 2 - 0.45, W / 2)):
        o.prism(F.rect(x0_, x1_, y0_, y1_), z0 + h, z0 + h + 1.0, cs)
    o.prism(F.rect(L / 2 - 2.2, L / 2 - 0.8, -0.9, 0.9), z0 + h, z0 + h + 1.0, cs)


def _theatre_geo(P, o):
    F = Fr(P["c"], P["u"]); col = P["col"]; cs = col["stone"]; T = P["T"]; zo = P["zo"]
    th = math.atan2(F.u[1], F.u[0]); sp = math.radians(T["span"]); a0, a1 = th - sp / 2, th + sp / 2
    circ = lambda R_, m=40: [F.c + R_ * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, m, endpoint=False)]
    o.prism(circ(T["ro"] + 1.0), zo - 0.6, zo - 0.02, cs)
    o.prism(circ(T["ro"]), zo - 0.6, zo + 0.02, col["floor"], "floor")
    nw = T["stairs"]; st = [a0 + (a1 - a0) * j / nw for j in range(nw + 1)]           # stairs: at the edges and between the wedges
    def sector(r0, r1, b0, b1, z_lo, z_hi, c_):
        m = max(3, int((b1 - b0) * r1 / 2.2) + 1); A = np.linspace(b0, b1, m)
        poly = [F.c + r1 * np.array([math.cos(t), math.sin(t)]) for t in A] + [F.c + r0 * np.array([math.cos(t), math.sin(t)]) for t in A[::-1]]
        o.prism(poly, z_lo, z_hi, c_)
    for k, (r0, r1, zl, zt) in enumerate(P["rows"]):
        rm = (r0 + r1) / 2; gap = 0.95 / rm
        for j in range(nw):
            b0 = st[j] + (gap / 2 if j > 0 else 0.0); b1 = st[j + 1] - (gap / 2 if j < nw - 1 else 0.0)
            if b1 > b0: sector(r0, r1, b0, b1, zl, zt, cs)
        for j in range(1, nw):                                                      # kerkides: two steps per row
            d = np.array([math.cos(st[j]), math.sin(st[j])]); h_ = (r1 - r0) / 2
            for s_, zz in ((0, zt - T["rise"] / 2), (1, zt)):
                c = F.c + d * (r0 + h_ * (s_ + 0.5)); o.box(c, d, 0.95, h_ + 0.01, zl, zz, col["floor"])
    if P.get("back"):                                                               # retaining wall behind the top row
        r = P["rows"][-1][1] + 1.2; ztop = P["rows"][-1][3]                          # a solid wall, with the cut face dropping off in the middle
        for (ta, za), (tb, zb_) in zip(P["back"][:-1], P["back"][1:]):
            p = F.c + r * np.array([math.cos(ta), math.sin(ta)]); q = F.c + r * np.array([math.cos(tb), math.sin(tb)]); d = q - p; Lq = float(np.linalg.norm(d))
            e = d / max(Lq, 1e-9) * 0.06
            o.wall(q + e, p - e, ztop - 0.6, [(0.0, max(ztop + 1.1, zb_ + 0.3)), (Lq + 0.12, max(ztop + 1.1, za + 0.3))], 2.0, [], cs, "finish")
    # skene with a proskenion
    ro = T["ro"]; ds, hs, pk = P["ds"], P["hs"], P["pk"]; ws = ro + 5.0; xf = -(ro + 2.0 + pk); xb = xf - ds; t = 0.7
    o.prism(F.rect(xb - 0.3, xf + pk + 0.3, -ws - 0.3, ws + 0.3), zo - 0.6, zo + 0.02, cs)
    cr = [F.w(xb + t / 2, -ws + t / 2), F.w(xf - t / 2, -ws + t / 2), F.w(xf - t / 2, ws - t / 2), F.w(xb + t / 2, ws - t / 2)]
    for i in range(4):
        p, q = cr[i], cr[(i + 1) % 4]; Lw = float(np.linalg.norm(q - p)); hs_ = []
        if i == 1: hs_ = [(Lw / 2, zo, 2.2, 3.4, "rect", "opening"), (Lw / 2 - 5.0, zo, 1.6, 3.0, "rect", "opening"), (Lw / 2 + 5.0, zo, 1.6, 3.0, "rect", "opening")] + \
                         [(Lw / 2 + s * 4.2, zo + pk + 3.6, 2.6, 2.6, "rect", "opening") for s in (-1, 0, 1)]
        elif i == 3: hs_ = [(s, zo + hs - 2.4, 0.9, 1.1, "rect", "opening") for s in np.arange(2.0, Lw - 1.5, 3.5)]
        else: hs_ = [(Lw / 2, zo, 1.4, 2.6, "rect", "opening")]
        o.wall(p, q, zo - 0.05, zo + hs, t, hs_, col["wall"])
    _gable(o, Fr(F.w((xb + xf) / 2, 0.0), F.v), -ws - 0.2, ws + 0.2, ds / 2 + 0.3, zo + hs, 14.0, col["roof"], ov=0.2, band=0.18)
    zp = zo + 3.0                                                                   # proskenion — a platform on columns (logeion)
    o.prism(F.rect(xf, xf + pk, -ws + 1.0, ws - 1.0), zp - 0.45, zp, cs, bot=True)
    for y in np.arange(-ws + 1.4, ws - 1.0, 1.9):
        o.column(F.w(xf + pk - 0.3, y), 0.24, zo, zp - 0.45, "doric", F.v, cs)
    o.prism(F.rect(xf + 0.1, xf + pk - 0.5, -ws + 1.0, ws - 1.0), zo, zp - 0.45, col["wall"], "wall")


def geometry(P):
    """Program → primitives (world, z up): prisms, columns, walls, faces, bars, cones, boxes."""
    o = _Out(); k = P["kind"]
    if k in TEMPLES: _temple_geo(P, o)
    elif k == "stoa": _stoa_geo(P, o)
    elif k == "bouleuterion": _bouleuterion_geo(P, o)
    elif k == "palaestra": _palaestra_geo(P, o)
    elif k == "altar": _altar_geo(P, o)
    elif k == "theatre": _theatre_geo(P, o)
    return o.d


# ---------------------------------------------------------------- showcase on site
def plan(place, out=None, kind="all", seed=1, at=None, **kw):
    """Building of the given kind (or "all" — a showcase of all kinds in rows) at the place → classic.json in out."""
    kind = kw.pop("kind", None) or kw.pop("type", None) or kind                   # type — this is how the "building" step passes the kind
    import shutil
    from core.surface.ground import Ground
    G = Ground(place); rng = np.random.default_rng(int(seed)); c = np.asarray(at if at is not None else G.loc.get("center", [0, 0]), float)
    kinds = [k for k in KINDS if k != "theatre"] if kind == "all" else [kind]
    frames = {"peripteros": (34, 18), "prostyle": (20, 11), "amphiprostyle": (16, 9), "in_antis": (12, 7), "podium": (30, 16), "tholos": (16, 16),
              "propylon": (16, 22), "stoa": (12, 60), "bouleuterion": (26, 22), "palaestra": (46, 46), "altar": (6, 10)}
    progs = []; x = 0.0; row_y = 0.0; row_h = 0.0
    for k in kinds:
        L, W = frames[k]
        if x + W > 130: x = 0.0; row_y -= row_h + 14; row_h = 0.0
        cc = c + np.array([x + W / 2 - 60, row_y]); x += W + 10; row_h = max(row_h, L)
        p = program(k, {"c": cc.tolist(), "u": [0.0, -1.0], "L": L, "W": W}, {"height": G.height}, rng)   # facade toward the view (south)
        if p: progs.append(p)
    out = out or os.path.join(arch_parts.ROOT, "out", "classics", f"{kind}_{seed}")
    os.makedirs(out, exist_ok=True)
    R = {"type": "classics", "place": place, "spec": {"kind": kind, "seed": seed}, "buildings": progs, "checks": [], "stats": {"n_buildings": len(progs)}, "out": out}
    json.dump(R, open(os.path.join(out, "classic.json"), "w"), ensure_ascii=False)
    for f in ("relief.npz", "location.json"): shutil.copy(os.path.join(place, f), os.path.join(out, f))
    return R


def main():
    a = sys.argv[1:]
    if not a: raise SystemExit(__doc__)
    kw = dict(x.split("=", 1) for x in a[1:] if "=" in x)
    R = plan(a[0], **kw)
    print("__RESULT__", json.dumps({"out": R["out"], **R["stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
