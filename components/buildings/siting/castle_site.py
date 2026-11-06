"""Reading a site for a castle: where to build, what type of site, where the weak sides are, where the approach comes from.
Any site (relief.npz).

The site type follows castellology (Hill castle / Lowland castle): summit, ridge, promontory, slope, rock, cliff edge;
plain, shore, island (peninsula), valley. It is determined by what lies outside the platform edge: a cliff, a steep
slope, water, a dominating height or a gentle approach (shares and arcs along the edge). The site's suitability follows
Dusterwald 2015 (a level top, height over the surroundings), plus natural protection in 16 directions and a threat from
above.

.venv/bin/python -m components.buildings.siting.castle_site <site folder> [area=3000] [at=x,y|anchor] → site.json next to the sheet (__RESULT__)
In code: read(folder, area=…, at=…) → the site dict (site meters, z up)."""
import json, math, os, sys

import cv2
import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from core.surface.ground import Ground                                                    # noqa: E402

STEP = 1.0                                                                    # m: the analysis grid
NDIR = 16
CLASSES = ("cliff", "steep", "water", "uphill", "gentle")                     # what lies outside the platform edge
TYPE_RU = {"summit": "summit", "ridge": "crest", "spur": "cape", "hillside": "hillside", "rock": "rock",
           "cliff_edge": "cliff edge", "plain": "plain", "shore": "shore", "island": "island",
           "peninsula": "peninsula", "valley": "valley"}


def grid(g, step=STEP, margin=6.0):
    b = g.near_bounds()
    xs = np.arange(b[0] + margin, b[2] - margin + 1e-6, step); ys = np.arange(b[1] + margin, b[3] - margin + 1e-6, step)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def shift(A, dy, dx):
    """A shifted by (dy, dx) cells: B[i, j] = A[i + dy, j + dx], beyond the border — the border value."""
    h, w = A.shape; p = max(abs(dy), abs(dx))
    P = np.pad(A, p, mode="edge")
    return P[p + dy:p + dy + h, p + dx:p + dx + w]


def maps(g, step=STEP, area=3000.0):
    """Site maps on a grid of step: height, slope, water, height above the surroundings, protection by direction, a threat from
    above. Protection and threat are measured outside the future walls — from the castle radius R = √(area/π)."""
    X, Y = grid(g, step)
    H0 = g.height(X, Y).astype(np.float32); W = g.water(X, Y).astype(np.float32)
    s = 2.0 / step; H = cv2.GaussianBlur(H0, (0, 0), s)                      # ground for building — without fine roughness
    gy, gx = np.gradient(H, step); S = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    k = int(60 / step) | 1
    REL = H - cv2.GaussianBlur(H, (k * 2 + 1, k * 2 + 1), k / 2.5)
    R = math.sqrt(area / math.pi)
    drop = np.zeros((NDIR,) + H.shape, np.float32); near = np.zeros_like(drop); up = np.zeros_like(drop)
    for d in range(NDIR):
        a = 2 * math.pi * d / NDIR; ca, sa = math.cos(a), math.sin(a)
        for r in (R, R + 8, R + 16, R + 26, R + 40):                  # the drop downward beyond the walls — up to R + 40 m
            Hs = shift(H, int(round(r * sa / step)), int(round(r * ca / step)))
            drop[d] = np.maximum(drop[d], H - Hs)
            if r <= R + 16: near[d] = np.maximum(near[d], H - Hs)
        for r in (R + 10, R + 30, R + 55, R + 85):                    # what is higher nearby — up to R + 85 m
            up[d] = np.maximum(up[d], shift(H, int(round(r * sa / step)), int(round(r * ca / step))) - H)
    return dict(X=X, Y=Y, H=H, Hraw=H0, W=W, S=S, REL=REL, drop=drop, near=near, up=up, step=step, R=R,
                x0=float(X[0, 0]), y0=float(Y[0, 0]))


def defended(M, h=9.0):
    """The share of directions with natural protection outside: a drop ≥ h within 45 m and ≥ 0.6 h already within 20 m."""
    return ((M["drop"] >= h) & (M["near"] >= 0.6 * h)).mean(0)


def candidates(M, area=3000.0, n=6, sep=60.0):
    """Candidate sites: level over the castle's area, dry, high above the surroundings, protected, not under a dominating height,
    not at the edge."""
    F = defended(M); S, W, REL = M["S"], M["W"], M["REL"]; st = M["step"]
    w_ = max(5, int(2 * math.sqrt(area / math.pi) / st)) | 1
    flat = cv2.blur((S < 15).astype(np.float32), (w_, w_))
    dom = (M["up"].max(0) > 8).astype(np.float32)
    h, w = S.shape; edge = np.zeros_like(S); e = int(40 / st)
    edge[:e] = 1; edge[-e:] = 1; edge[:, :e] = 1; edge[:, -e:] = 1
    score = F + 0.4 * np.tanh(REL / 15) + 0.6 * flat - 0.6 * (W > 0.1) - 0.5 * edge - 0.25 * dom - 0.4 * (S > 20)
    k = int(sep / st) | 1
    peak = (score >= cv2.dilate(score, np.ones((k, k), np.uint8)) - 1e-6) & (S < 15) & (W < 0.05) & (edge < 0.5)
    ij = sorted(zip(*np.nonzero(peak)), key=lambda p: -score[p])
    out = []
    for i, j in ij:
        if all(math.hypot(i - a, j - b) * st >= sep for a, b in [c["ij"] for c in out]):
            out.append({"ij": (int(i), int(j)), "xy": [float(M["X"][i, j]), float(M["Y"][i, j])], "score": float(score[i, j]),
                        "protection": float(F[i, j]), "over_surroundings": float(REL[i, j])})
        if len(out) >= n: break
    return out


def regions(M, ij, area):
    """Platforms around a point at different cuts: level (< 22°), dry, within Δ below the top (Δ = 3…24 m), no farther than 1.9
    castle radii. Returns ([(mask, Δ)], the top elevation, the top cell); the cut is chosen by the edge (evaluate)."""
    st = M["step"]; H, S, W = M["H"], M["S"], M["W"]
    i, j = ij; r = int(15 / st)
    ok = (S < 20) & (W < 0.05)
    win = np.where(ok, H, -1e9)[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]; a, b = np.unravel_index(np.argmax(win), win.shape)
    i, j = max(0, i - r) + a, max(0, j - r) + b; H0 = float(H[i, j])          # start — at the top nearby
    yy, xx = np.mgrid[0:H.shape[0], 0:H.shape[1]]
    R = math.sqrt(area / math.pi) * 1.9
    dist = np.hypot(yy - i, xx - j) * st
    out = []; k = max(3, int(3 / st)) | 1
    for dl in (3, 5, 8, 12, 17, 24):
        m = ((S < 22) & (W < 0.05) & (H > H0 - dl) & (H < H0 + 3) & (dist < R)).astype(np.uint8)
        nlab, lab = cv2.connectedComponents(m, connectivity=4)
        if lab[i, j] == 0: continue
        reg = (lab == lab[i, j]).astype(np.uint8)
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
        nlab, lab = cv2.connectedComponents(reg, connectivity=4)
        if lab[i, j] == 0: continue
        reg = (lab == lab[i, j]).astype(np.uint8)
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        reg = np.zeros_like(reg); cv2.drawContours(reg, cs, -1, 1, -1)          # without holes
        if out and reg.sum() == out[-1][0].sum(): continue
        out.append((reg.astype(bool), dl))
    if not out:
        reg = np.zeros(H.shape, bool); reg[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = True; out = [(reg, 0)]
    return out, H0, (i, j)


def contour(mask, M):
    """The outer outline of a mask — points every 1 cell, counterclockwise (in site meters)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)                    # (x=j, y=i)
    P = np.column_stack([M["x0"] + c[:, 0] * M["step"], M["y0"] + c[:, 1] * M["step"]])
    a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return P if a > 0 else P[::-1]


def sample(M, A, x, y):
    j = np.clip(np.round((x - M["x0"]) / M["step"]).astype(int), 0, A.shape[1] - 1)
    i = np.clip(np.round((y - M["y0"]) / M["step"]).astype(int), 0, A.shape[0] - 1)
    return A[i, j]


def edge_classes(M, P, smooth=9):
    """What lies outside each edge point: a ray outward along the normal — the drop, the steepness, water, a rise. The classes
    are smoothed along the edge."""
    n = len(P); T = np.roll(P, -2, 0) - np.roll(P, 2, 0); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    N = np.column_stack([T[:, 1], -T[:, 0]])                                   # outward (the outline is counterclockwise)
    H, S, W = M["H"], M["S"], M["W"]
    h0 = sample(M, H, P[:, 0], P[:, 1])
    drop = np.zeros(n); drop20 = np.zeros(n); smax = np.zeros(n); wet = np.zeros(n); rise = np.zeros(n)
    for r in np.arange(2, 46, 2.0):
        x, y = P[:, 0] + N[:, 0] * r, P[:, 1] + N[:, 1] * r
        h = sample(M, H, x, y); drop = np.maximum(drop, h0 - h)
        if r <= 20: drop20 = np.maximum(drop20, h0 - h); smax = np.maximum(smax, sample(M, S, x, y))
        if r <= 25: wet = np.maximum(wet, sample(M, W, x, y))
    for r in np.arange(5, 81, 5.0):
        rise = np.maximum(rise, sample(M, H, P[:, 0] + N[:, 0] * r, P[:, 1] + N[:, 1] * r) - h0)
    cls = np.full(n, 4)                                                         # gentle
    cls[(rise >= 5) & (drop < 4)] = 3                                           # uphill: it is higher outside — a threat from above
    cls[(drop >= 6) & (drop20 >= 4)] = 1                                         # steep
    cls[(drop >= 10) & (smax >= 40)] = 0                                         # cliff
    cls[wet >= 0.3] = 2                                                          # water
    k = smooth; ext = np.concatenate([cls[-k:], cls, cls[:k]])                   # the majority within the window — no flicker
    out = np.array([np.bincount(ext[i:i + 2 * k + 1], minlength=5).argmax() for i in range(n)])
    return out, dict(drop=drop, rise=rise, wet=wet, normal=N)


def arcs(cls):
    """Continuous arcs of one class along the closed edge: [(class, start, length)]."""
    n = len(cls)
    if (cls == cls[0]).all(): return [(int(cls[0]), 0, n)]
    s = int(np.nonzero(cls != np.roll(cls, 1))[0][0]); out = []; i = 0
    while i < n:
        c = cls[(s + i) % n]; L = 1
        while i + L < n and cls[(s + i + L) % n] == c: L += 1
        out.append((int(c), (s + i) % n, L)); i += L
    return out


def classify(fr, arc_list, area, elong, rel, n_pts, valley_sides):
    """The site type from the shares of the edge and the shape of the platform."""
    nat = fr["cliff"] + fr["steep"]
    gentle_arcs = [a for a in arc_list if a[0] == 4 and a[2] >= 0.04 * n_pts]
    if fr["water"] >= 0.85: return "island"
    if fr["water"] >= 0.45: return "peninsula"
    if valley_sides: return "valley"
    if nat >= 0.75 and area <= 2500 and rel >= 10: return "rock"
    if fr["uphill"] >= 0.15: return "hillside"
    if fr["water"] >= 0.15 and fr["cliff"] < 0.25: return "shore"
    if nat >= 0.55:
        if len(gentle_arcs) == 1 and fr["gentle"] <= 0.35: return "spur"
        if len(gentle_arcs) == 2 and elong >= 1.8: return "ridge"
        if len(gentle_arcs) <= 1: return "summit"
    if fr["cliff"] >= 0.15 and fr["gentle"] >= 0.3: return "cliff_edge"
    if nat >= 0.3: return "spur" if len(gentle_arcs) == 1 else "summit"
    if fr["water"] >= 0.08: return "shore"
    return "plain"


def approach(g, M, P, centre, b, src=None):
    """Where the approach comes from: an explicit point of the intent (approach), otherwise the start of the site's road, if there
    is one, otherwise the plot edge on the side of the gentlest edge; a path to the center."""
    if isinstance(src, str): src = [float(v) for v in src.replace(";", ",").split(",")[:2]]   # "x,y" from the recipe
    src = np.asarray(src, float)[:2] if src is not None else None     # an explicit approach point (intent: approach) — stronger than the site's roads
    for name, p in (g.loc.get("paths") or {}).items():
        if src is not None or p.get("kind") != "road": continue
        pts = np.asarray(p["pts"], float)
        far = pts[np.argmax(np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]))]
        src = far[:2]; break
    if src is None:
        k = int(np.argmin(b["drop"] - 0.3 * b["rise"]))                       # the gentlest edge
        d = P[k] - centre; d /= max(np.linalg.norm(d), 1e-9)
        nb = g.near_bounds(); src = centre + d * ((nb[2] - nb[0]) / 2 - 15)
    R = g.route(src, centre, max_slope=28.0, step=2.0, margin=40.0)
    if R is None: return {"from": [float(src[0]), float(src[1])], "path": None}
    k = next((i for i, p in enumerate(R) if cv2.pointPolygonTest(P.astype(np.float32).reshape(-1, 1, 2),
                                                                  (float(p[0]), float(p[1])), False) >= 0), len(R) - 1)
    arr = R[k]; e = int(np.argmin(np.hypot(P[:, 0] - arr[0], P[:, 1] - arr[1])))
    return {"from": [float(src[0]), float(src[1])], "path": R[:k + 1].round(2).tolist(), "arrive": arr[:2].round(2).tolist(),
            "edge_i": e, "dir": (np.asarray(arr[:2]) - R[max(0, k - 8), :2]).round(3).tolist()}


def evaluate(M, ij, area):
    """A platform at a point and everything one needs to know about it: the outline, the edge classes, the site type, the site's
    quality for a castle. The platform cut is the one where the edge is better protected at a sufficient area: the walls go
    along the brow, the castle adapts."""
    regs, H0, top = regions(M, ij, area)
    best = None
    for reg, dl in regs:
        P = contour(reg, M); cls, b = edge_classes(M, P)
        fr = {c: float((cls == k).mean()) for k, c in enumerate(CLASSES)}; am = float(reg.sum() * M["step"] ** 2)
        fit = min(am / area, 1.0)
        qs = (fr["cliff"] + fr["steep"] + fr["water"]) * (0.4 + 0.6 * fit) + 0.25 * fit - 0.3 * fr["uphill"] - 0.6 * (fit < 0.2)
        if best is None or qs > best[0] + 1e-9: best = (qs, reg, dl, P, cls, b, fr)
    _, reg, dl, P, cls, b, fr = best
    n = len(P)
    A = arcs(cls)
    ii, jj = np.nonzero(reg); xy = np.column_stack([M["X"][ii, jj], M["Y"][ii, jj]]); centre = xy.mean(0)
    ev, evec = np.linalg.eigh(np.cov((xy - centre).T)) if len(xy) > 2 else (np.ones(2), np.eye(2))
    elong = float(math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))); axis = evec[:, 1]
    area_m2 = float(reg.sum() * M["step"] ** 2); rel = float(M["REL"][top])
    up_dirs = (M["up"][:, top[0], top[1]] > 12)                                 # valley: high on two opposite sides
    valley = bool(any(up_dirs[d] and up_dirs[(d + NDIR // 2) % NDIR] for d in range(NDIR))) and fr["uphill"] >= 0.3
    kind = classify(fr, A, area_m2, elong, rel, n, valley)
    h, w = reg.shape; e = int(8 / M["step"])
    touch = bool(reg[:e].any() or reg[-e:].any() or reg[:, :e].any() or reg[:, -e:].any())
    nat = fr["cliff"] + fr["steep"] + fr["water"]
    fit = min(area_m2 / area, 1.0)
    q = (nat * (0.4 + 0.6 * fit) + 0.25 * fit + 0.2 * math.tanh(rel / 15) - 0.3 * fr["uphill"]
         - 0.6 * (fit < 0.2) - 0.2 * touch)
    return dict(reg=reg, dl=dl, H0=H0, top=top, P=P, cls=cls, b=b, fr=fr, A=A, centre=centre, elong=elong, axis=axis,
                area=area_m2, rel=rel, kind=kind, q=q, touch=touch)


def read(folder, area=3000.0, at=None, want=None, g=None, M=None, approach_from=None):
    """A site for a castle of an area of ~area m²: by the anchor / the point at; otherwise the best of the candidates (want — the
    desired site type)."""
    g = g or Ground(folder); M = M or maps(g, area=area)
    if at is not None:
        xy = g.anchor(at)[:2] if isinstance(at, str) else np.asarray(at, float)
        ij = (int(round((xy[1] - M["y0"]) / M["step"])), int(round((xy[0] - M["x0"]) / M["step"])))
        cand = [{"ij": ij, "xy": [float(xy[0]), float(xy[1])], "score": None}]
    else:
        cand = candidates(M, area)
        if not cand:                                                            # no level dry place — the best point as is
            F = defended(M); i, j = np.unravel_index(np.argmax(F + 0.4 * np.tanh(M["REL"] / 15) - (M["W"] > 0.1)), F.shape)
            cand = [{"ij": (int(i), int(j)), "xy": [float(M["X"][i, j]), float(M["Y"][i, j])], "score": None}]
    ev = []
    for c in cand:
        e = evaluate(M, c["ij"], area); c["type"] = TYPE_RU[e["kind"]]; c["q"] = round(e["q"], 3); ev.append(e)
    pool = [e for e in ev if want and e["kind"] == want] or ev
    E = max(pool, key=lambda e: e["q"])
    ap = approach(g, M, E["P"], E["centre"], E["b"], src=approach_from)
    top = E["top"]; P, cls = E["P"], E["cls"]
    out = {"type": E["kind"], "type_ru": TYPE_RU[E["kind"]], "q": round(E["q"], 3), "centre": E["centre"].round(2).tolist(),
           "top": [float(M["X"][top]), float(M["Y"][top]), E["H0"]],
           "area_m2": round(E["area"], 1), "elong": round(E["elong"], 2), "axis": E["axis"].round(3).tolist(), "rel_m": round(E["rel"], 1),
           "plateau_dz": E["dl"], "edge_frac": {k: round(v, 3) for k, v in E["fr"].items()},
           "outline": P[::2].round(2).tolist(), "edge_cls": cls[::2].tolist(),
           "edge_drop": E["b"]["drop"][::2].round(1).tolist(), "edge_rise": E["b"]["rise"][::2].round(1).tolist(),
           "arcs": [{"cls": CLASSES[c], "i0": s // 2, "n": L // 2, "len_m": round(L * M["step"], 1)} for c, s, L in E["A"]],
           "approach": ap, "candidates": [{k: v for k, v in c.items() if k != "ij"} for c in cand]}
    return out


def main():
    a = sys.argv[1:]; folder = a[0]; kw = dict(x.split("=", 1) for x in a[1:])
    at = kw.get("at")
    if at and "," in at: at = [float(v) for v in at.split(",")]
    S = read(folder, area=float(kw.get("area", 3000)), at=at)
    out = kw.get("out", os.path.join(folder, "castle_site.json"))
    json.dump(S, open(out, "w"), ensure_ascii=False)
    print("__RESULT__", json.dumps({k: S[k] for k in ("type", "type_ru", "area_m2", "elong", "rel_m", "edge_frac")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
