"""Parametric architectural elements — a shared library (castles; later cathedrals and mega-structures):
corbels, machicolations, hoarding gallery, covered wall walk, garderobe, oriel, apse on corbels, bartizan,
"butter churn" tower, wicket gate, water gates and bricked-up gates, buttresses, stair, residential windows
with tracery, portcullis, bretesche, drawbridge with lever arms, chimneys, dormers, half-timbering, stepped
gable, ridge turret. No ornaments.

Pure numpy. a is a mesh accumulator with the methods face(points, color), box(center, axis, width, length,
z0, z1, color), prism(polygon, z0, z1, color, …), quad_strip(A, B, color, closed); K is the colors (STONE,
STONE_D, STONE2, ROOF, PLASTER, WOOD, HOLE, IRON). Wall points are stations along the wall axis, N is the
outward normals, t is the thickness; z up, meters."""
import math

import numpy as np


def arch_pts(s, h, n=8):
    """Arc from the left springing to the right one: two centers on the springing line, r = (h² + s²/4)/s
    (semicircle — h = s/2)."""
    r = (h * h + s * s / 4) / s; xc = r - s / 2; a1 = math.atan2(h, -xc)
    left = [(xc + r * math.cos(math.pi + (a1 - math.pi) * i / n), r * math.sin(math.pi + (a1 - math.pi) * i / n)) for i in range(n + 1)]
    return left + [(-x, z) for x, z in reversed(left[:-1])]


def tangent(n):
    n = np.asarray(n, float); return np.array([-n[1], n[0]])


def along(P, step, margin=0.8):
    """Points every step along the polyline P (inset from the ends) and the tangents."""
    P = np.asarray(P, float); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))]); L = s[-1]
    if L < 2 * margin + 0.1: return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    k = max(1, int((L - 2 * margin) // step)); u = np.linspace(margin + (L - 2 * margin - (k - 1) * step) / 2, L - margin, k) if k > 1 else np.array([L / 2])
    u = np.clip(u, 0, L); X = np.interp(u, s, P[:, 0]); Y = np.interp(u, s, P[:, 1])
    T = np.column_stack([np.interp(np.minimum(u + 0.5, L), s, P[:, 0]) - np.interp(np.maximum(u - 0.5, 0), s, P[:, 0]),
                         np.interp(np.minimum(u + 0.5, L), s, P[:, 1]) - np.interp(np.maximum(u - 0.5, 0), s, P[:, 1])])
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    return np.column_stack([X, Y]), T, u


def interp_s(P, vals, u):
    P = np.asarray(P, float); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    return np.interp(u, s, np.asarray(vals, float))


def opening(a, c, n, z0, w, h, K, top="rect", eps=0.04, col=None):
    """Opening (dark) on a vertical face: c is a point of the face, n points outward; the top is
    rect/round/pointed/slit."""
    n = np.asarray(n, float); n /= max(np.linalg.norm(n), 1e-9); t = tangent(n); c = np.asarray(c, float) + n * eps
    if top in ("rect", "slit") or w < 0.5:
        prof = [(-w / 2, 0), (w / 2, 0), (w / 2, h), (-w / 2, h)]
    else:
        rise = w / 2 if top == "round" else w * 0.75
        prof = [(-w / 2, 0), (w / 2, 0)] + [(x, h - rise + z) for x, z in arch_pts(w, rise, 6)][::-1]
    a.face([(*(c + t * x), z0 + z) for x, z in prof], col or K["HOLE"])


def ribbon(a, c, n, curve, width, K, col=None, eps=0.07):
    """Strip of width `width` along a curve (u, z) in the plane of a face (c is the face origin, n points outward):
    bars of tracery and half-timbering."""
    n = np.asarray(n, float); n /= max(np.linalg.norm(n), 1e-9); t = tangent(n); c = np.asarray(c, float) + n * eps
    Q = np.asarray(curve, float)
    for (u1, z1), (u2, z2) in zip(Q[:-1], Q[1:]):
        dx, dz = u2 - u1, z2 - z1; L = math.hypot(dx, dz)
        if L < 1e-6: continue
        px, pz = -dz / L * width / 2, dx / L * width / 2
        a.face([(*(c + t * (u1 + px)), z1 + pz), (*(c + t * (u2 + px)), z2 + pz), (*(c + t * (u2 - px)), z2 - pz), (*(c + t * (u1 - px)), z1 - pz)],
               col or K["STONE"])


def tracery(a, c, n, z0, w, h, K, col=None):
    """Window tracery (after Havemann 2004, simplified): a transom, two lancets, an oculus with a quatrefoil — over
    the dark opening."""
    rise = w * 0.75; sp = h - rise; bar = max(0.08, w * 0.07); col = col or K["STONE"]
    ribbon(a, c, n, [(0, z0), (0, z0 + sp + rise * 0.35)], bar, K, col)
    for k in (-1, 1):
        ribbon(a, c, n, [(k * w / 4 + x, z0 + sp + z) for x, z in arch_pts(w / 2 - bar, (w / 2 - bar) * 0.8, 5)], bar, K, col)
    ro = w * 0.17; zc = z0 + sp + rise * 0.62
    ribbon(a, c, n, [(ro * math.cos(t_), zc + ro * math.sin(t_)) for t_ in np.linspace(0, 2 * math.pi, 17)], bar * 0.8, K, col)


def corbels(a, p, n, z_top, K, tiers=3, w=0.55, step_out=0.28, step_h=0.32, col=None):
    """Corbel: stepped projections, the top is z_top, the projection is tiers × step_out from the face p."""
    n = np.asarray(n, float); p = np.asarray(p, float)
    for k in range(tiers):
        depth = step_out * (k + 1); z1 = z_top - (tiers - 1 - k) * step_h
        a.box(p + n * depth / 2, n, w, depth, z1 - step_h, z1, col or K["STONE_D"])


def merlon_row(a, P, N, z, K, mw=1.7, gap=0.8, h=1.2, th=0.6, col=None):
    """Merlons along the polyline P (outer edge), normals N, bottom z (a number or an array per P point)."""
    pts, T, u = along(P, mw + gap, margin=gap / 2)
    zz = interp_s(P, np.broadcast_to(np.asarray(z, float), (len(P),)), u) if len(pts) else []
    Nn = np.column_stack([interp_s(P, N[:, 0], u), interp_s(P, N[:, 1], u)]) if len(pts) else []
    for q, tg, z0, nn in zip(pts, T, zz, Nn):
        nn = nn / max(np.linalg.norm(nn), 1e-9)
        a.box(q - nn * th / 2, tg, th, mw, z0, z0 + h, col or K["STONE"])


# ---------------------------------------------------------------- wall section elements
def machicolation(a, P, N, walk, t, K, tiers=3, spacing=1.8, proj=0.75):
    """Machicolations: corbels under the projecting parapet, openings downward between the corbels; merlons along
    the projecting edge."""
    P = np.asarray(P, float); N = np.asarray(N, float); t = np.broadcast_to(np.asarray(t, float), (len(P),)); walk = np.asarray(walk, float)
    Po = P + N * (t / 2)[:, None]
    pts, T, u = along(Po, spacing, margin=0.6)
    zz = interp_s(Po, walk, u)
    for q, tg, z in zip(pts, T, zz):
        nn = np.array([tg[1], -tg[0]])
        corbels(a, q, nn, z - 0.45, K, tiers=tiers, step_out=proj / tiers, step_h=0.38)
    zb = walk - 0.45; zt = walk + 1.2
    A_out = [(*p, z) for p, z in zip(Po + N * proj, zb)]; B_out = [(*p, z) for p, z in zip(Po + N * proj, zt)]
    A_in = [(*p, z) for p, z in zip(Po - N * 0.05, zt)]; B_in = [(*p, z) for p, z in zip(Po - N * 0.55, zt)]
    a.quad_strip(A_out, B_out, K["STONE"])                                         # outer face of the projecting parapet
    a.quad_strip(B_out, A_in, K["STONE"]); a.quad_strip(A_in, B_in, K["STONE"])
    a.quad_strip([(*p, z) for p, z in zip(Po - N * 0.55, zt)], [(*p, z) for p, z in zip(Po - N * 0.55, walk)], K["STONE"])
    a.quad_strip([(*p, z) for p, z in zip(Po, zb)], [(*p, z) for p, z in zip(Po + N * proj, zb)], K["HOLE"])   # slots downward (bottom)
    merlon_row(a, Po + N * proj, N, zt, K, th=0.55)


def hoarding(a, P, N, walk, t, K, bay=1.3, out=1.7):
    """Hoarding gallery: beams through the wall, deck, plank wall with loopholes, roof over the walkway."""
    P = np.asarray(P, float); N = np.asarray(N, float); t = np.broadcast_to(np.asarray(t, float), (len(P),)); walk = np.asarray(walk, float)
    Po = P + N * (t / 2)[:, None]; pts, T, u = along(Po, bay, margin=0.4); zz = interp_s(Po, walk, u)
    for q, tg, z in zip(pts, T, zz):
        nn = np.array([tg[1], -tg[0]])
        a.box(q + nn * (out / 2 - 0.2), nn, 0.24, out + 0.4, z - 0.55, z - 0.3, K["WOOD"])          # beam
    zf = walk - 0.3; zt = walk + 2.2; zr = walk + 3.3
    Pout = Po + N * out
    a.quad_strip([(*p, z) for p, z in zip(Po, zf)], [(*p, z) for p, z in zip(Pout, zf)], K["WOOD"])         # deck
    a.quad_strip([(*p, z) for p, z in zip(Pout, zf - 0.15)], [(*p, z) for p, z in zip(Pout, zt)], K["WOOD"])  # wall
    a.quad_strip([(*p, z) for p, z in zip(Pout + N * 0.25, zt - 0.1)], [(*p, z) for p, z in zip(Po - N * (t * 0.5)[:, None], zr)], K["ROOF"])
    for q, tg, z in zip(pts, T, zz):                                                                           # loopholes
        nn = np.array([tg[1], -tg[0]]); opening(a, q + nn * out, nn, z + 0.7, 0.35, 0.6, K)
    for e, sgn in ((0, -1), (-1, 1)):                                                                          # end faces
        p0, p1, z = Po[e], Pout[e], walk[e]
        a.face([(*p0, zf[e]), (*p1, zf[e]), (*p1, zt[e]), (*p0, zr[e])], K["WOOD"])


def covered_walk(a, P, N, walk, t, K, post=2.6, out=0.35, eave=2.55):
    """Covered wall walk: a roof over the walkway along the wall, on posts on the inner side; out, eave —
    projection and eave height (over machicolations — up to their parapet: the two-tier defense of Pierrefonds)."""
    P = np.asarray(P, float); N = np.asarray(N, float); t = np.broadcast_to(np.asarray(t, float), (len(P),)); walk = np.asarray(walk, float)
    Pi = P - N * (t / 2 - 0.2)[:, None]; pts, T, u = along(Pi, post, margin=0.5); zz = interp_s(Pi, walk, u)
    for q, tg, z in zip(pts, T, zz): a.box(q, tg, 0.22, 0.22, z, z + eave + 0.35, K["WOOD"])
    a.quad_strip([(*p, z + eave) for p, z in zip(P + N * (t / 2 + out)[:, None], walk)], [(*p, z + eave + 0.6) for p, z in zip(P - N * (t / 2 + 0.4)[:, None], walk)], K["ROOF"])


def arcade(a, p0, p1, deck, w, n, zg, K, pointed=True, roof=True, walls=2.6):
    """Arcaded gallery (Danzker, Marienburg): n spans from p0 (wall) to p1 (tower); piers from the ground zg(x, y)
    to the deck, with pointed arches between them (springing, vault, spandrels up to the deck), the deck,
    gallery walls with windows, a gable roof."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); d = p1 - p0; L = float(np.linalg.norm(d)); d /= max(L, 1e-9)
    nr = np.array([-d[1], d[0]]); span = L / n; pier = float(np.clip(span * 0.26, 1.4, 2.6)); op = span - pier
    for i in range(n + 1):                                                        # piers
        c = p0 + d * (min(max(i * span, pier / 2), L - pier / 2)); zb = float(zg(*c)) - 0.8
        a.box(c, d, w, pier, zb, deck - 0.5, K["STONE"])
    rise = op * (0.75 if pointed else 0.5)
    for i in range(n):                                                            # arches: the springing is below the deck by the vault + 1.2 m
        cm = p0 + d * ((i + 0.5) * span); spring = deck - 0.5 - 1.2 - rise; zgr = float(zg(*cm))
        spring = max(spring, zgr + 2.0)
        prof = arch_pts(op, rise, 8)
        for (x1, z1), (x2, z2) in zip(prof[:-1], prof[1:]):
            xm = (x1 + x2) / 2; zm = spring + min(z1, z2)
            a.box(cm + d * xm, d, w, abs(x2 - x1) + 0.03, zm, deck - 0.5, K["STONE"])   # vault and spandrels as strips; empty under the arch
    a.box((p0 + p1) / 2, d, w + 0.3, L, deck - 0.5, deck, K["STONE_D"])          # deck
    for sgn in (-1, 1):                                                           # gallery walls (at the bridge — a parapet without windows)
        a.box((p0 + p1) / 2 + nr * sgn * (w / 2 - 0.2), d, 0.4, L, deck, deck + walls, K["STONE"])
        if walls >= 2.0:
            for k in range(1, int(L / 3.2)):
                q = p0 + d * (k * L / int(L / 3.2)) + nr * sgn * (w / 2); opening(a, q, nr * sgn, deck + 1.0, 0.5, 1.1, K)
    if roof:
        e0, e1 = p0 - d * 0.2, p1 + d * 0.2; rh = (w / 2 + 0.4) * 0.7
        for sgn in (-1, 1):
            a.face([(*(e0 + nr * sgn * (w / 2 + 0.4)), deck + 2.5), (*(e1 + nr * sgn * (w / 2 + 0.4)), deck + 2.5), (*e1, deck + 2.5 + rh), (*e0, deck + 2.5 + rh)], K["ROOF"])


def garderobe(a, p, n, z, t, K):
    """Garderobe on corbels: a booth projecting outward, a pitched roof, a chute down the wall."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); tg = tangent(n); po = np.asarray(p, float) + n * t / 2
    for s in (-0.55, 0.55): corbels(a, po + tg * s, n, z, K, tiers=3, w=0.4, step_out=0.42, step_h=0.35)
    a.box(po + n * 0.65, n, 1.6, 1.3, z, z + 2.3, K["STONE"])
    a.face([(*(po - tg * 0.95), z + 2.6), (*(po + tg * 0.95), z + 2.6), (*(po + tg * 0.95 + n * 1.45), z + 2.15), (*(po - tg * 0.95 + n * 1.45), z + 2.15)], K["ROOF"])
    opening(a, po + n * 1.3, n, z + 1.0, 0.25, 0.6, K)
    opening(a, po, n, z - 3.2, 0.45, 3.0, K)                                                                  # chute


def oriel(a, p, n, z0, z1, K, faces=3, w=3.0, col=None, pitch=55):
    """Oriel: a polygonal projection on a stepped corbel base, windows on the faces, a roof sloping to the wall."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); tg = tangent(n); p = np.asarray(p, float); dep = w * 0.45; col = col or K["PLASTER"]
    if faces == 3:
        Q = [p - tg * w / 2, p - tg * w * 0.25 + n * dep, p + tg * w * 0.25 + n * dep, p + tg * w / 2]
    else:
        Q = [p - tg * w / 2] + [p + tg * (w / 2) * math.cos(math.pi - math.pi * i / 4) + n * dep * math.sin(math.pi * i / 4) for i in range(1, 4)] + [p + tg * w / 2]
    Q = np.array(Q)
    for k in range(3):                                                             # corbel base — in steps toward the wall
        f0, f1 = 1.0 - 0.3 * k, 1.0 - 0.3 * (k + 1)
        a.quad_strip([(*q, z0 - 0.32 * k) for q in p + (Q - p) * f0], [(*q, z0 - 0.32 * k) for q in p + (Q - p) * f1], K["STONE_D"])
        a.quad_strip([(*q, z0 - 0.32 * k) for q in p + (Q - p) * f1], [(*q, z0 - 0.32 * (k + 1)) for q in p + (Q - p) * f1], K["STONE_D"])
    a.face([(*q, z0) for q in Q], K["STONE_D"])
    a.quad_strip([(*q, z0) for q in Q], [(*q, z1) for q in Q], col)
    for q0, q1 in zip(Q[:-1], Q[1:]):
        m = (q0 + q1) / 2; nn = np.array([(q1 - q0)[1], -(q1 - q0)[0]]); nn /= max(np.linalg.norm(nn), 1e-9)
        if nn @ n > 0.05: opening(a, m, nn, z0 + 0.8, min(0.9, np.hypot(*(q1 - q0)) * 0.6), max(1.0, z1 - z0 - 1.6), K)
    apex = (*p, z1 + dep * math.tan(math.radians(pitch)))
    for q0, q1 in zip(Q[:-1], Q[1:]): a.face([(*q0, z1), (*q1, z1), apex], K["ROOF"])


def chapel_bay(a, p, n, z0, z1, K, r=3.0, corbelled=False, pitch=55):
    """Chapel apse projecting through the wall: 5/8 of an octagon, tall pointed windows, a half-tent roof; on
    corbels if it is high up."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); tg = tangent(n); p = np.asarray(p, float)
    Q = np.array([p + tg * r * math.cos(math.pi - math.pi * i / 4) + n * r * math.sin(math.pi * i / 4) for i in range(5)])
    if corbelled:
        for k in range(4):
            f0, f1 = 1.0 - 0.22 * k, 1.0 - 0.22 * (k + 1)
            a.quad_strip([(*(p + (q - p) * f0), z0 - 0.5 * k) for q in Q], [(*(p + (q - p) * f1), z0 - 0.5 * (k + 1)) for q in Q], K["STONE_D"])
    a.face([(*q, z0) for q in Q], K["STONE_D"])
    a.quad_strip([(*q, z0) for q in Q], [(*q, z1) for q in Q], K["STONE"])
    for q0, q1 in zip(Q[:-1], Q[1:]):
        m = (q0 + q1) / 2; nn = np.array([(q1 - q0)[1], -(q1 - q0)[0]]); nn /= max(np.linalg.norm(nn), 1e-9)
        if nn @ n > 0.2:
            h = (z1 - z0) * 0.62; wv = min(1.0, np.hypot(*(q1 - q0)) * 0.45)
            opening(a, m, nn, z0 + (z1 - z0) * 0.22, wv, h, K, "pointed"); tracery(a, m, nn, z0 + (z1 - z0) * 0.22, wv, h, K)
    for q in Q[1:-1]:                                                              # buttresses at the apse corners
        u = (q - p) / max(np.linalg.norm(q - p), 1e-9); a.box(q + u * 0.35, u, 0.5, 0.7, z0 if not corbelled else z0, z0 + (z1 - z0) * 0.8, K["STONE"])
    apex = (*p, z1 + r * math.tan(math.radians(pitch)) * 0.8)
    for q0, q1 in zip(Q[:-1], Q[1:]): a.face([(*q0, z1), (*q1, z1), apex], K["ROOF"])


def bartizan(a, p, z_top, r, K, roof="cone", pitch=60, out=None):
    """Bartizan: a small turret on stepped corbel rings; a conical roof or merlons."""
    p = np.asarray(p, float); n_ = 14
    ring = lambda rr: [p + rr * np.array([math.cos(2 * math.pi * i / n_), math.sin(2 * math.pi * i / n_)]) for i in range(n_)]
    zc = z_top - 1.0
    for k, (r0, r1) in enumerate(((0.55, 0.72), (0.72, 0.87), (0.87, 1.0))):     # stepped corbel rings
        z_lo = zc - 0.3 * (3 - k); z_hi = zc - 0.3 * (2 - k)
        a.quad_strip([(*q, z_lo) for q in ring(r * r0)], [(*q, z_lo) for q in ring(r * r1)], K["STONE_D"], closed=True)
        a.quad_strip([(*q, z_lo) for q in ring(r * r1)], [(*q, z_hi) for q in ring(r * r1)], K["STONE_D"], closed=True)
    a.face([(*q, zc - 0.9) for q in ring(r * 0.55)], K["STONE_D"])
    a.quad_strip([(*q, zc) for q in ring(r)], [(*q, z_top + 2.0) for q in ring(r)], K["STONE"], closed=True)
    if out is not None: opening(a, p + np.asarray(out) * r, out, z_top + 0.4, 0.2, 1.0, K, "slit")
    if roof == "flat":
        a.face([(*q, z_top + 2.0) for q in ring(r)], K["STONE_D"])
    else:
        apex = (*p, z_top + 2.0 + r * 1.1 * math.tan(math.radians(pitch)))
        R = ring(r * 1.12)
        for i in range(n_): a.face([(*R[i], z_top + 1.9), (*R[(i + 1) % n_], z_top + 1.9), apex], K["ROOF"])


def butterfass(a, c, r, z0, h, K, roof="cone", pitch=60, plan="round"):
    """"Butter churn": a smaller tower on the platform of the bergfried (keep)."""
    c = np.asarray(c, float); n_ = 20 if plan == "round" else 4
    rr = r if plan == "round" else r / math.cos(math.pi / 4)
    P = [c + rr * np.array([math.cos(2 * math.pi * i / n_ + (math.pi / 4 if n_ == 4 else 0)), math.sin(2 * math.pi * i / n_ + (math.pi / 4 if n_ == 4 else 0))]) for i in range(n_)]
    a.quad_strip([(*q, z0) for q in P], [(*q, z0 + h) for q in P], K["STONE"], closed=True)
    for k in range(4):
        ang = 2 * math.pi * k / 4 + 0.4; v = np.array([math.cos(ang), math.sin(ang)])
        opening(a, c + v * r, v, z0 + h * 0.45, 0.6, 1.2, K, "round")
    if roof in ("flat",):
        a.face([(*q, z0 + h) for q in P], K["STONE_D"])
    else:
        E = [c + (q - c) * 1.1 for q in P]; apex = (*c, z0 + h + rr * math.tan(math.radians(pitch)) * (1.3 if roof == "spire" else 1.0))
        for i in range(n_): a.face([(*E[i], z0 + h - 0.1), (*E[(i + 1) % n_], z0 + h - 0.1), apex], K["ROOF"])


def postern(a, p, n, z, t, K, w=1.1, h=2.2):
    """Wicket gate: a small arched opening near the ground, framed."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); po = np.asarray(p, float) + n * t / 2
    ribbon(a, po, n, [(x, z + zz) for x, zz in [(-w / 2 - 0.15, 0)] + [(x_, h - w / 2 + z_ ) for x_, z_ in arch_pts(w + 0.3, w / 2 + 0.15, 6)] + [(w / 2 + 0.15, 0)]],
           0.3, K, K["STONE_D"], eps=0.05)
    opening(a, po, n, z, w, h, K, "round", eps=0.08)


def water_gate(a, p, n, z, t, K, w=2.6, h=2.8):
    n = np.asarray(n, float); n /= np.linalg.norm(n); po = np.asarray(p, float) + n * t / 2
    ribbon(a, po, n, [(x, z + zz) for x, zz in [(x_, h - w / 2 + z_) for x_, z_ in arch_pts(w + 0.5, w / 2 + 0.25, 8)]], 0.45, K, K["STONE_D"], eps=0.05)
    opening(a, po, n, z - 0.3, w, h + 0.3, K, "round", eps=0.09)
    tg = tangent(n)
    for k in range(4): a.box(po + n * (0.5 + 0.4 * k), tg, w, 0.4, z - 0.25 * (k + 1) - 0.3, z - 0.25 * k - 0.3 + 0.01, K["STONE_D"])


def blocked_gate(a, p, n, z, t, K, w=3.4, h=4.6):
    """Bricked-up gate: the framing arch and the infill in a different masonry."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); po = np.asarray(p, float) + n * t / 2; tg = tangent(n)
    prof = [(-w / 2, 0), (w / 2, 0)] + [(x, h - w / 2 + zz) for x, zz in arch_pts(w, w / 2, 8)][::-1]
    a.face([(*(po + n * 0.04 + tg * x), z + zz) for x, zz in prof], K["STONE2"])
    ribbon(a, po, n, [(x, z + zz) for x, zz in [(-w / 2 - 0.3, 0)] + [(x_, h - w / 2 + z_) for x_, z_ in arch_pts(w + 0.6, w / 2 + 0.3, 8)] + [(w / 2 + 0.3, 0)]],
           0.55, K, K["STONE_D"], eps=0.06)


def buttresses(a, P, N, base, walk, t, K, spacing=6.0, w=1.6, d=1.8):
    """Sloping buttresses along a stretch on a slope: from the bottom up to 0.6 of the height, the top is beveled."""
    P = np.asarray(P, float); N = np.asarray(N, float); tt = np.broadcast_to(np.asarray(t, float), (len(P),))
    Po = P + N * (tt / 2)[:, None]; pts, T, u = along(Po, spacing, margin=2.0)
    zb = interp_s(Po, base, u); zw = interp_s(Po, walk, u)
    for q, tg, z0, z1 in zip(pts, T, zb, zw):
        nn = np.array([tg[1], -tg[0]]); zt = z0 + (z1 - z0) * 0.62
        B0 = [q - tg * w / 2, q + tg * w / 2, q + tg * w / 2 + nn * d, q - tg * w / 2 + nn * d]
        B1 = [q - tg * w / 2, q + tg * w / 2, q + tg * w / 2 + nn * 0.35, q - tg * w / 2 + nn * 0.35]
        a.quad_strip([(*b, z0) for b in B0], [(*b, zt) for b in B1], K["STONE"], closed=True)
        a.face([(*b, zt) for b in B1], K["STONE"])


def pinnacle(a, p, z0, w, h, K, col=None):
    """Pinnacle (Roriczer, 1486): a square shaft of width w and height h and a small pyramid three times as tall as
    the width."""
    p = np.asarray(p, float); a.box(p, (1.0, 0.0), w, w, z0, z0 + h, col or K["STONE"])
    q = [p + np.array(v) * w / 2 for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4): a.face([(*q[i], z0 + h), (*q[(i + 1) % 4], z0 + h), (*p, z0 + h + 3.2 * w)], K["STONE_D"])


def pinnacle_buttresses(a, P, N, base, walk, t, K, spacing=8.0, w=1.3, d=1.5, parapet=1.1):
    """Gothic buttresses along a wall: from the bottom to the walkway in two offsets, with a pinnacle above the
    parapet at the top."""
    P = np.asarray(P, float); N = np.asarray(N, float); tt = np.broadcast_to(np.asarray(t, float), (len(P),))
    Po = P + N * (tt / 2)[:, None]; pts, T, u = along(Po, spacing, margin=2.5)
    zb = interp_s(Po, base, u); zw = interp_s(Po, walk, u)
    for q, tg, z0, z1 in zip(pts, T, zb, zw):
        nn = np.array([tg[1], -tg[0]]); zm = z0 + (z1 - z0) * 0.55
        for (za, zc, dd) in ((z0, zm, d), (zm, z1 - 0.4, d * 0.62)):            # two offsets
            B = [q - tg * w / 2, q + tg * w / 2, q + tg * w / 2 + nn * dd, q - tg * w / 2 + nn * dd]
            a.quad_strip([(*b, za) for b in B], [(*b, zc) for b in B], K["STONE"], closed=True); a.face([(*b, zc) for b in B], K["STONE_D"])
        pinnacle(a, q + nn * d * 0.31, z1 - 0.4, w * 0.62, parapet + 1.6, K)


def skybridge(a, p0, p1, z, w, K, pointed=True):
    """Bridge between tower tops (Ravenloft, Lothric): a deck, parapets, a single-span vault under the deck."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); dv = p1 - p0; L = float(np.linalg.norm(dv)); dv /= max(L, 1e-9)
    nr = np.array([-dv[1], dv[0]])
    a.box((p0 + p1) / 2, dv, w, L, z - 0.7, z, K["STONE_D"])
    for sgn in (-1, 1): a.box((p0 + p1) / 2 + nr * sgn * (w / 2 - 0.15), dv, 0.3, L, z, z + 1.1, K["STONE"])
    rise = L * (0.22 if pointed else 0.16); prof = arch_pts(L, rise, 10)
    for (x1, z1_), (x2, z2_) in zip(prof[:-1], prof[1:]):                         # vault: strips from the arc to the deck
        xm = (x1 + x2) / 2; zlo = z - 0.7 - rise + min(z1_, z2_)
        a.box((p0 + p1) / 2 + dv * xm, dv, w * 0.9, abs(x2 - x1) + 0.03, zlo, z - 0.7, K["STONE"])


def stairs(a, p, along_d, n_in, z0, z1, t, K, w=1.3, rise=0.2, run=0.3):
    """Stair from the courtyard to the wall walk along the inner face."""
    along_d = np.asarray(along_d, float); along_d /= np.linalg.norm(along_d); n_in = np.asarray(n_in, float); n_in /= np.linalg.norm(n_in)
    k = max(2, int(math.ceil((z1 - z0) / rise))); L = k * run
    start = np.asarray(p, float) + n_in * (t / 2 + w / 2) - along_d * L
    for i in range(k):
        c = start + along_d * (run * (i + 0.5)); a.box(c, along_d, w, run + 0.01, z0 - 0.3, z0 + (i + 1) * (z1 - z0) / k, K["STONE_D"])


def palace_windows(a, P, N, t, floor, storeys, sh, K, tracery_on=True, rhythm=3.8, top="pointed"):
    """Residential windows in the curtain wall: the upper floors (the first is blind), with tracery for Gothic."""
    P = np.asarray(P, float); N = np.asarray(N, float); tt = np.broadcast_to(np.asarray(t, float), (len(P),))
    Po = P + N * (tt / 2)[:, None]; pts, T, u = along(Po, rhythm, margin=1.5)
    for q, tg in zip(pts, T):
        nn = np.array([tg[1], -tg[0]])
        for st_ in range(1, max(2, storeys)):
            z0 = floor + st_ * sh + 0.7; h = min(2.8, sh - 1.1); w = 1.35
            opening(a, q, nn, z0, w, h, K, top if tracery_on else "rect")
            if tracery_on: tracery(a, q, nn, z0, w, h, K)


def portcullis(a, p, d, z0, w, h, K, raised=0.55):
    """Portcullis in the gateway: bars every 0.3 m, crossbars every 0.6 m; raised."""
    d = np.asarray(d, float); d /= np.linalg.norm(d); p = np.asarray(p, float); zb = z0 + h * raised
    for x in np.arange(-w / 2 + 0.15, w / 2, 0.3): a.box(p + d * x, d, 0.09, 0.09, zb, z0 + h, K["IRON"])
    for z in np.arange(zb + 0.3, z0 + h, 0.6): a.box(p, d, 0.07, w, z, z + 0.08, K["IRON"])


def breteche(a, p, n, z, K, w=2.6):
    """Bretesche: a projecting loophole booth on corbels over the gateway."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); tg = tangent(n); p = np.asarray(p, float)
    for s in (-w / 2 + 0.3, 0.0, w / 2 - 0.3): corbels(a, p + tg * s, n, z, K, tiers=3, w=0.4, step_out=0.35, step_h=0.35)
    a.box(p + n * 0.55, n, w, 1.1, z, z + 1.9, K["STONE"])
    a.face([(*(p - tg * (w / 2 + 0.1)), z + 2.25), (*(p + tg * (w / 2 + 0.1)), z + 2.25), (*(p + tg * (w / 2 + 0.1) + n * 1.25), z + 1.95), (*(p - tg * (w / 2 + 0.1) + n * 1.25), z + 1.95)], K["ROOF"])
    opening(a, p + n * 1.1, n, z + 0.7, 0.22, 0.8, K, "slit")


def drawbridge(a, gp, d, o, zdeck, wd, length, K, beams=True, zgate_top=None):
    """Drawbridge span at the gate (lowered) and the lever arms in slots over the gateway with chains (from the
    14th c.)."""
    d = np.asarray(d, float); o = np.asarray(o, float); gp = np.asarray(gp, float)
    c = gp + o * (length / 2 + 0.4); a.box(c, o, wd + 0.2, length, zdeck - 0.25, zdeck + 0.06, K["WOOD"])
    if beams and zgate_top is not None:
        for s in (-1, 1):
            q = gp + d * s * (wd / 2 + 0.35)
            opening(a, q + o * 0.05, o, zgate_top - 0.2, 0.32, 3.2, K)                                      # slot
            z = zgate_top + 2.6; tip = q + o * (length + 0.4)
            a.box((q + tip) / 2 + o * 0.2, o, 0.3, length + 0.8, z, z + 0.3, K["WOOD"])
            a.box(tip, d, 0.06, 0.06, zdeck + 0.06, z, K["IRON"])                                          # chain


# ---------------------------------------------------------------- buildings
def chimney(a, p, z0, z1, K, w=0.9, d=0.7, axis=(1, 0)):
    a.box(p, axis, d, w, z0, z1, K["STONE_D"]); a.box(p, axis, d + 0.2, w + 0.2, z1, z1 + 0.25, K["STONE"])


def dormer(a, p, n, z_eave, K, w=1.4, pitch=50):
    """Dormer window on a roof slope: a front wall with a window, a small gable roof."""
    n = np.asarray(n, float); n /= np.linalg.norm(n); tg = tangent(n); p = np.asarray(p, float); h = 1.5
    a.box(p - n * 0.6, n, w, 1.4, z_eave, z_eave + h, K["PLASTER"])
    opening(a, p + n * 0.1, n, z_eave + 0.3, w * 0.5, 0.9, K)
    zr = z_eave + h + w / 2 * math.tan(math.radians(pitch))
    for s in (-1, 1):
        a.face([(*(p + tg * s * (w / 2 + 0.15) + n * 0.2), z_eave + h - 0.1), (*(p + tg * s * (w / 2 + 0.15) - n * 1.5), z_eave + h - 0.1),
                (*(p - n * 1.5), zr), (*(p + n * 0.2), zr)], K["ROOF"])


def fachwerk(a, p0, p1, n, z0, z1, K, post=1.15):
    """Half-timbering of a face from z0 to z1: sill beam, top plate, posts at the spacing post, braces at the
    corners (dark wood over plaster)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); n = np.asarray(n, float); n /= np.linalg.norm(n)
    L = float(np.hypot(*(p1 - p0))); tg = (p1 - p0) / max(L, 1e-9); c = p0
    curves = [[(0, z0 + 0.1), (L, z0 + 0.1)], [(0, z1 - 0.1), (L, z1 - 0.1)], [(0, (z0 + z1) / 2), (L, (z0 + z1) / 2)]]
    for x in np.arange(0.12, L, post): curves.append([(x, z0), (x, z1)])
    for x0, x1 in ((0.12, min(L, 1.4)), (L - 0.12, max(0, L - 1.4))): curves.append([(x0, z0 + 0.1), (x1, (z0 + z1) / 2)])
    for cv in curves:
        Q = [(u * 1.0, z) for u, z in cv]
        ribbon_t(a, c, tg, n, Q, 0.16, K["WOOD"])


def ribbon_t(a, c, tg, n, curve, width, col, eps=0.06):
    """Like ribbon, but with an explicit face tangent (u along tg from c)."""
    c = np.asarray(c, float) + n * eps
    for (u1, z1), (u2, z2) in zip(curve[:-1], curve[1:]):
        dx, dz = u2 - u1, z2 - z1; L = math.hypot(dx, dz)
        if L < 1e-6: continue
        px, pz = -dz / L * width / 2, dx / L * width / 2
        a.face([(*(c + tg * (u1 + px)), z1 + pz), (*(c + tg * (u2 + px)), z2 + pz), (*(c + tg * (u2 - px)), z2 - pz), (*(c + tg * (u1 - px)), z1 - pz)], col)


def stepped_gable(a, pa, pb, n, z_eave, z_ridge, K, steps=5, col=None):
    """Stepped gable over the end wall pa–pb (n points outward): steps above the roof slope."""
    pa, pb = np.asarray(pa, float), np.asarray(pb, float); n = np.asarray(n, float); n /= np.linalg.norm(n)
    L = float(np.hypot(*(pb - pa))); tg = (pb - pa) / L; m = (pa + pb) / 2; H = z_ridge - z_eave + 0.8
    for k in range(steps):
        f0 = k / steps; f1 = (k + 1) / steps; half = L / 2 * (1 - f0)
        z0 = z_eave + H * f0; z1 = z_eave + H * f1
        for s in (-1, 1):
            q = m + tg * s * (half - L / (4 * steps)); a.box(q, tg, 0.5, L / (2 * steps), z0, z1, col or K["PLASTER"])
    a.face([(*(pa + n * 0.02), z_eave), (*(pb + n * 0.02), z_eave), (*(m + n * 0.02), z_ridge)], col or K["PLASTER"])


def roof_turret(a, p, z0, K, r=0.9, h=2.4, spire=3.0):
    """Ridge turret (Dachreiter): an octagon and a spire."""
    p = np.asarray(p, float); P = [p + r * np.array([math.cos(math.pi / 8 + i * math.pi / 4), math.sin(math.pi / 8 + i * math.pi / 4)]) for i in range(8)]
    a.quad_strip([(*q, z0) for q in P], [(*q, z0 + h) for q in P], K["WOOD"], closed=True)
    for i in range(8):
        v = (P[i] + P[(i + 1) % 8]) / 2 - p; v /= np.linalg.norm(v); opening(a, (P[i] + P[(i + 1) % 8]) / 2, v, z0 + h * 0.35, 0.35, h * 0.45, K)
        a.face([(*P[i], z0 + h), (*P[(i + 1) % 8], z0 + h), (*p, z0 + h + spire)], K["ROOF"])


def arrow_loops(a, P, N, walk, t, K, step=3.6, cross=False):
    """Loopholes in the curtain wall at a height of 2 m below the walkway; late ones are cruciform (with a
    transverse slit)."""
    P = np.asarray(P, float); N = np.asarray(N, float); tt = np.broadcast_to(np.asarray(t, float), (len(P),))
    Po = P + N * (tt / 2)[:, None]; pts, T, u = along(Po, step, margin=1.5); zz = interp_s(Po, walk, u)
    for q, tg, z in zip(pts, T, zz):
        nn = np.array([tg[1], -tg[0]]); opening(a, q, nn, z - 2.6, 0.2, 1.7, K, "slit")
        if cross: opening(a, q, nn, z - 1.9, 0.7, 0.18, K, "rect", eps=0.05)


def putlogs(a, P, N, base, walk, t, K, rows=3, step=2.0):
    """Scaffold putlog holes: rows of small square openings on the outer face (every step m, rows 1.6 m apart below
    the walkway)."""
    P = np.asarray(P, float); N = np.asarray(N, float); tt = np.broadcast_to(np.asarray(t, float), (len(P),))
    Po = P + N * (tt / 2)[:, None]; pts, T, u = along(Po, step, margin=1.0); zw = interp_s(Po, walk, u); zb = interp_s(Po, base, u)
    for q, tg, z1, z0 in zip(pts, T, zw, zb):
        nn = np.array([tg[1], -tg[0]])
        for k in range(rows):
            z = z1 - 4.0 - 1.6 * k
            if z > z0 + 1.5: opening(a, q, nn, z, 0.22, 0.22, K, "rect", eps=0.03)
