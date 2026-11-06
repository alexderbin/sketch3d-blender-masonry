"""Clay geometry of the levels — only numpy and shapely (the project's Python; Blender only draws, levels/bl_clay.py).

The model is a rectangular block [X0, X1] × [Y0, Y1] (like a museum cross-section): everything is clipped by its faces, the
four side faces of the block are the faces of the ground section by era layers (holes — cellars, tunnels, the lower parts
of buildings). Bodies of two kinds:
  prism(outline, z0, z1)           — a vertical prism (walls, slabs, supports, steps, decks);
  extrude(profile, axis, a0, a1)   — a profile in the plane across the axis (v, z), extruded along the x or y axis (vaults, bridges, roofs);
voids (void=True) — the same bodies inside out and without caps on the block faces: visible only through the section.
A face carries a class (code CLS) and a tone 0–1 (depth shade, era) → the clay color at build time (bl_clay.py)."""
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box as sbox
from shapely.ops import unary_union

CLS = {"rock": 0, "fill": 1, "ground": 2, "wall": 3, "slab": 4, "roof": 5, "void": 6, "street": 7, "deck": 8, "ruin": 9,
       "stair": 10, "pillar": 11, "water": 12, "retain": 13}
EPS = 1e-6


def polys(g):
    """Polygons of a shapely geometry (empty → [])."""
    if g is None or g.is_empty: return []
    if isinstance(g, Polygon): return [g]
    if isinstance(g, MultiPolygon): return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and p.area > EPS]


def tris(poly):
    """Triangles of a polygon with holes (constrained Delaunay) → [(3, 2)]."""
    out = []
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        c = np.asarray(t.exterior.coords)[:3]
        e1, e2 = c[1] - c[0], c[2] - c[0]
        if abs(e1[0] * e2[1] - e1[1] * e2[0]) > 1e-9: out.append(c)
    return out


class Mesh:
    """Accumulator: vertices, faces (triangles and quads), a class and a tone per face, a level (for the exploded view)."""

    def __init__(self, block):
        self.V = []; self.F = []; self.C = []; self.T = []; self.L = []; self.H = []
        self.X0, self.Y0, self.X1, self.Y1 = block; self.rect = sbox(*block)
        self.tone = 1.0; self.level = 0; self.tint = (1.0, 1.0, 1.0)

    def face(self, pts, cls, flip=False):
        pts = [tuple(map(float, p)) for p in pts]
        if flip: pts = pts[::-1]
        i0 = len(self.V); self.V += pts; self.F.append(list(range(i0, i0 + len(pts))))
        self.C.append(CLS[cls] if isinstance(cls, str) else cls); self.T.append(self.tone); self.L.append(self.level); self.H.append(self.tint)

    def cap(self, poly, to3, cls, normal):
        """A flat polygon face (with holes): to3(u, v) → (x, y, z); the winding follows the normal normal (outward from the body)."""
        for t in tris(poly):
            P = [to3(u, v) for u, v in t]
            n = np.cross(np.subtract(P[1], P[0]), np.subtract(P[2], P[0]))
            self.face(P, cls, flip=float(np.dot(n, normal)) < 0)

    # ------------------------------------------------------------ bodies
    def prism(self, foot, z0, z1, cls, void=False, top=True, bottom=True, cls_top=None):
        """A vertical prism over an outline (shapely or points), clipped by the block; void — inside out, without caps on the block
        faces."""
        if z1 - z0 < EPS: return
        g = foot if hasattr(foot, "geom_type") else Polygon(foot)
        for p in polys(g.intersection(self.rect)):
            p = shapely.geometry.polygon.orient(p, 1.0)
            for ring in [p.exterior] + list(p.interiors):
                C = np.asarray(ring.coords)[:-1]; n = len(C)
                for i in range(n):
                    a, b = C[i], C[(i + 1) % n]
                    if void and self._on_block(a, b): continue                       # a face along the block edge — open to the section
                    self.face([(*a, z0), (*b, z0), (*b, z1), (*a, z1)], cls, flip=void)
            sg = -1.0 if void else 1.0
            if top: self.cap(p, lambda u, v: (u, v, z1), cls_top or cls, (0, 0, sg))
            if bottom: self.cap(p, lambda u, v: (u, v, z0), cls, (0, 0, -sg))

    def box(self, x0, x1, y0, y1, z0, z1, cls, **kw):
        if x1 - x0 > EPS and y1 - y0 > EPS: self.prism(sbox(x0, y0, x1, y1), z0, z1, cls, **kw)

    def extrude(self, prof, axis, a0, a1, cls, void=False, caps=True):
        """A profile (v, z) is a shapely polygon in the plane across the axis; extruded along the axis from a0 to a1 (clipped by the
        block: along the axis — by an interval, across it — by the profile). Axis x: v = y; axis y: v = x."""
        lo, hi = (self.X0, self.X1) if axis == "x" else (self.Y0, self.Y1)
        vlo, vhi = (self.Y0, self.Y1) if axis == "x" else (self.X0, self.X1)
        a0c, a1c = max(a0, lo), min(a1, hi)
        if a1c - a0c < EPS: return
        g = prof.intersection(sbox(vlo, -1e5, vhi, 1e5))
        P3 = (lambda a, v, z: (a, v, z)) if axis == "x" else (lambda a, v, z: (v, a, z))
        for p in polys(g):
            p = shapely.geometry.polygon.orient(p, 1.0)
            for ring in [p.exterior] + list(p.interiors):
                C = np.asarray(ring.coords)[:-1]; n = len(C)
                for i in range(n):
                    (va, za), (vb, zb) = C[i], C[(i + 1) % n]
                    if void and (abs(va - vb) < EPS and (abs(va - vlo) < EPS or abs(va - vhi) < EPS)): continue
                    q = [P3(a0c, va, za), P3(a0c, vb, zb), P3(a1c, vb, zb), P3(a1c, va, za)]
                    self.face(q, cls, flip=(axis == "y") != void)
            ax = np.array((1.0, 0, 0) if axis == "x" else (0, 1.0, 0)) * (-1.0 if void else 1.0)
            for a, at_edge, sg in ((a0c, a0c <= lo + EPS, -1.0), (a1c, a1c >= hi - EPS, 1.0)):
                if not caps or (void and at_edge): continue
                self.cap(p, lambda v, z, a=a: P3(a, v, z), cls, ax * sg)

    def _on_block(self, a, b):
        return ((abs(a[0] - b[0]) < EPS and (abs(a[0] - self.X0) < EPS or abs(a[0] - self.X1) < EPS)) or
                (abs(a[1] - b[1]) < EPS and (abs(a[1] - self.Y0) < EPS or abs(a[1] - self.Y1) < EPS)))

    # ------------------------------------------------------------ ground
    def terrain(self, layers, step=0.5, holes=(), cls_top="ground", tones=None, z_bottom=None, classes=None, tints=None, top_cls=None):
        """Ground by layers: layers — functions g_k(x, y) (growing upward: nature, era fills…). The top is the last layer over the
        grid (the cells under buildings are holes, polygons are skipped); the block's sides are a section by layers (minus the
        section holes self.cuts — voids and the lower parts of buildings); the bottom is a plane."""
        X = np.arange(self.X0, self.X1 + EPS, step); Y = np.arange(self.Y0, self.Y1 + EPS, step); X[-1] = self.X1; Y[-1] = self.Y1
        Xg, Yg = np.meshgrid(X, Y); Z = np.broadcast_to(np.asarray(layers[-1](Xg, Yg), float), Xg.shape)
        zb = (Z.min() - 4.0) if z_bottom is None else min(z_bottom, Z.min() - 4.0); self.z_bottom = zb
        hole = unary_union([h if hasattr(h, "geom_type") else Polygon(h) for h in holes]) if holes else None
        keep_t = self.tint
        for j in range(len(Y) - 1):
            for i in range(len(X) - 1):
                xm, ym = (X[i] + X[i + 1]) / 2, (Y[j] + Y[j + 1]) / 2
                if hole is not None and hole.contains(shapely.geometry.Point(xm, ym)): continue
                c, self.tint = (top_cls(xm, ym) if top_cls else (cls_top, keep_t))
                self.face([(X[i], Y[j], Z[j, i]), (X[i + 1], Y[j], Z[j, i + 1]), (X[i + 1], Y[j + 1], Z[j + 1, i + 1]),
                           (X[i], Y[j + 1], Z[j + 1, i])], c)
        self.tint = keep_t
        self.cap(sbox(self.X0, self.Y0, self.X1, self.Y1), lambda u, v: (u, v, zb), "rock", (0, 0, -1))
        for side in ("y0", "y1", "x0", "x1"): self.section(side, layers, zb, tones, classes=classes, tints=tints)

    def section(self, side, layers, zb, tones=None, step=0.1, classes=None, tints=None):
        """A block face by layers: the polygon of layer k in the (u, z) plane between g_{k−1} and g_k minus the cuts (self.cuts[side])."""
        if side in ("y0", "y1"):
            yy = self.Y0 if side == "y0" else self.Y1; U = np.arange(self.X0, self.X1 + EPS, step); U[-1] = self.X1
            f = lambda g, u: g(u, np.full_like(u, yy)); to3 = lambda u, z: (u, yy, z); nrm = (0, -1 if side == "y0" else 1, 0)
        else:
            xx = self.X0 if side == "x0" else self.X1; U = np.arange(self.Y0, self.Y1 + EPS, step); U[-1] = self.Y1
            f = lambda g, u: g(np.full_like(u, xx), u); to3 = lambda u, z: (xx, u, z); nrm = (-1 if side == "x0" else 1, 0, 0)
        cut = unary_union(self.cuts.get(side, [])) if getattr(self, "cuts", None) else None
        prev = np.full(len(U), zb)
        for k, g in enumerate(layers):
            cur = np.maximum(np.broadcast_to(np.asarray(f(g, U), float), U.shape), prev)
            if np.any(cur - prev > 1e-3):
                P = Polygon(list(zip(U, cur)) + list(zip(U[::-1], prev[::-1]))).buffer(0)
                if cut is not None: P = P.difference(cut)
                keep = self.tone, self.tint; self.tone = tones[k] if tones else 1.0; self.tint = tints[k] if tints else keep[1]
                for p in polys(P): self.cap(p, to3, (classes[k] if classes else ("rock" if k == 0 else "fill")), nrm)
                self.tone, self.tint = keep
            prev = cur

    def add_cut(self, side, poly):
        """A hole in a section face (u, z): the lower part of a building, a cellar, a tunnel."""
        if not hasattr(self, "cuts"): self.cuts = {}
        self.cuts.setdefault(side, []).append(poly)

    def cut_box(self, x0, x1, y0, y1, z0, z1):
        """Holes of the block faces that the box [x0, x1] × [y0, y1] × [z0, z1] crosses."""
        for side, (a, b, val, lo, hi) in {"y0": (x0, x1, self.Y0, y0, y1), "y1": (x0, x1, self.Y1, y0, y1),
                                           "x0": (y0, y1, self.X0, x0, x1), "x1": (y0, y1, self.X1, x0, x1)}.items():
            if lo - EPS <= val <= hi + EPS: self.add_cut(side, sbox(a, z0, b, z1))

    def cut_prism(self, poly, z0, z1):
        """Holes of the block faces from a vertical prism of any outline: the intersection of the outline with the face line × [z0, z1]."""
        lines = {"y0": (LineString([(self.X0, self.Y0), (self.X1, self.Y0)]), 0), "y1": (LineString([(self.X0, self.Y1), (self.X1, self.Y1)]), 0),
                 "x0": (LineString([(self.X0, self.Y0), (self.X0, self.Y1)]), 1), "x1": (LineString([(self.X1, self.Y0), (self.X1, self.Y1)]), 1)}
        for side, (ln, k) in lines.items():
            g = poly.intersection(ln)
            for seg in (getattr(g, "geoms", None) or ([g] if not g.is_empty else [])):
                if seg.geom_type != "LineString": continue
                b = seg.bounds; u0, u1 = (b[0], b[2]) if k == 0 else (b[1], b[3])
                if u1 - u0 > EPS: self.add_cut(side, sbox(u0, z0, u1, z1))

    def cut_profile(self, prof, axis, a0, a1):
        """Holes of the block faces from an extruded profile (v, z): the ends — the profile; the sides — a strip along the axis."""
        if axis == "x":
            for side, val in (("x0", self.X0), ("x1", self.X1)):
                if a0 - EPS <= val <= a1 + EPS: self.add_cut(side, prof)
            for side, val in (("y0", self.Y0), ("y1", self.Y1)):
                z = _zs_at(prof, val)
                if z: self.add_cut(side, sbox(a0, z[0], a1, z[1]))
        else:
            for side, val in (("y0", self.Y0), ("y1", self.Y1)):
                if a0 - EPS <= val <= a1 + EPS: self.add_cut(side, prof)
            for side, val in (("x0", self.X0), ("x1", self.X1)):
                z = _zs_at(prof, val)
                if z: self.add_cut(side, sbox(a0, z[0], a1, z[1]))

    # ------------------------------------------------------------ output
    def arrays(self, explode=0.0, poche=0.66):
        """→ V (n, 3), triangles (m, 3), class, tone, level per triangle; explode — lifts a level by explode·level (m);
        poche — the tone of faces lying in the section planes (block faces): the cut is darker, as in an architectural section."""
        V = np.asarray(self.V, np.float64); tri = []; c = []; t = []; l = []
        planes = ((0, self.X0), (0, self.X1), (1, self.Y0), (1, self.Y1))
        for f, ci, ti, li in zip(self.F, self.C, self.T, self.L):
            P = V[f]
            if any(np.all(np.abs(P[:, a] - val) < 1e-4) for a, val in planes): ti = ti * poche
            for k in range(1, len(f) - 1): tri.append((f[0], f[k], f[k + 1])); c.append(ci); t.append(ti); l.append(li)
        tri = np.asarray(tri, np.int64); L = np.asarray(l, np.int32)
        self._tint = np.asarray([h for f, h in zip(self.F, self.H) for _ in range(len(f) - 2)], np.float32).reshape(-1, 3)
        if explode:
            lv = np.zeros(len(V), np.int32); lv[tri.ravel()] = np.repeat(L, 3); V = V.copy(); V[:, 2] += explode * lv
        return V, tri, np.asarray(c, np.int32), np.asarray(t, np.float32), L

    def save(self, path, **meta):
        import json
        V, T, C, Tn, L = self.arrays()
        np.savez_compressed(path, verts=V.astype(np.float32), tris=T.astype(np.int32), cls=C, tone=Tn, level=L, tint=self._tint,
                            meta=json.dumps(meta, ensure_ascii=False))
        return path


def _zs_at(prof, v):
    """The z interval of a profile (v, z) on the line v = const → (z0, z1) or None."""
    g = prof.intersection(LineString([(v, -1e5), (v, 1e5)]))
    if g.is_empty: return None
    b = g.bounds; return (b[1], b[3]) if b[3] - b[1] > EPS else None


def arch_profile(v0, v1, z0, z_spring, kind="round", n=12):
    """Profile of a vaulted void: a rectangle [v0, v1] × [z0, z_spring] + a semicircular (or pointed) arch on top."""
    w = v1 - v0; c = (v0 + v1) / 2
    if kind == "flat": return sbox(v0, z0, v1, z_spring)
    r = w / 2; A = np.linspace(np.pi, 0, n + 1)
    k = 1.25 if kind == "pointed" else 1.0
    top = [(c + r * np.cos(a), z_spring + r * k * np.sin(a)) for a in A]
    return Polygon([(v0, z0)] + top + [(v1, z0)])


def arcade_profile(u0, u1, z0, z1, span, pier, z_spring, kind="round"):
    """Profile of a bridge or a wall with arches (u, z): the bar [u0, u1] × [z0, z1] minus the spans span with piers pier."""
    P = sbox(u0, z0, u1, z1); L = u1 - u0; n = max(1, int((L - pier) // (span + pier)))
    s = (L - pier * (n + 1)) / n
    for i in range(n):
        a = u0 + pier + i * (s + pier)
        P = P.difference(arch_profile(a, a + s, z0 - 1.0, z_spring, kind))
    return P
