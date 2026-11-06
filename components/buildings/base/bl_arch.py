"""Shared architectural geometry in Blender (temples — temples/bl_temple.py; then castles and cities): a mesh
accumulator with per-face color, a wall with real openings (a polygon with holes — mathutils
tessellate_polygon, no boolean operations), arches by compass, ribbed and groin vaults of a bay on four
corners, an apse vault, a flying buttress, a pinnacle (Roriczer), tracery (two lancets and an oculus,
Havemann) and a rose window, roofs (gable, shed, cone), a solid of revolution, ground following the site
relief. The rules are tried out in base/runs/bay/bl_bay.py."""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils import geometry as MG


SCENE = [None]                                                                   # scene of the step-by-step build (None — the current one); set by the type's builder


def coll(name):
    sc = SCENE[0] or bpy.context.scene
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in sc.collection.children: sc.collection.children.link(c)
    return c


from core.contract.contract import ROLES   # object contract — a single table (no copies)
ROLE_NAME = {v: k for k, v in ROLES.items()}                                     # 0–7 are masonry codes (bl_stones.ROLES); 8–10 are architecture
ROLE_ROUGH = {0: 0.85, 1: 0.75, 2: 0.7, 3: 0.8, 4: 0.4, 5: 0.05, 6: 0.95, 7: 0.9, 8: 0.8, 9: 1.0, 10: 0.05}


class Col(tuple):
    """Face color with a role (sk_role): the palette carries meaning — the material is assigned by role in a
    separate slot (not one for everything: the shader limit is 16 textures). The shade keeps the role."""
    role = None

    def __new__(cls, rgb, role=None):
        o = super().__new__(cls, tuple(float(v) for v in tuple(rgb)[:3])); o.role = role if role is not None else getattr(rgb, "role", None); return o


def rc(rgb, role):
    """Color with a role: rc((r, g, b), "roof")."""
    return Col(rgb, ROLES[role] if isinstance(role, str) else role)


def shade(c, k): return Col(tuple(float(min(1, max(0, v * k))) for v in c), getattr(c, "role", None))


def mat_role(role):
    """Material of the role sk_<name> (shared with the masonry): the color is the Col attribute (previs), the
    roughness depends on the role."""
    name = f"sk_{ROLE_NAME.get(role, role)}"; m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name); nt = m.node_tree; bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"; nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
        bs.inputs["Roughness"].default_value = ROLE_ROUGH.get(role, 0.85)
    return m


class Acc:
    """Mesh accumulator: vertices, faces, face color → one object (the color is the "Col" attribute per corner);
    the face role (sk_role) comes from the color (Col.role) or from the accumulator's default; the block number
    sk_stone = 0 (not a masonry stone); materials — one slot per role. item is the item number for the following
    faces (a house in a batch of houses): if set, the object gets a FACE INT attribute item_attr (−1 — no
    number)."""

    def __init__(self, role=0):
        self.v = []; self.f = []; self.c = []; self.role = ROLES[role] if isinstance(role, str) else role
        self.it = []; self.item = None; self.item_attr = "sk_item"

    def face(self, pts, col):
        i0 = len(self.v); self.v += [tuple(map(float, p)) for p in pts]; self.f.append(list(range(i0, i0 + len(pts)))); self.c.append(col)
        self.it.append(self.item)

    def tri(self, P, idx, col):
        """Triangles by indices into the point list P (a single insertion of vertices)."""
        i0 = len(self.v); self.v += [tuple(map(float, p)) for p in P]
        for a, b, c in idx: self.f.append([i0 + a, i0 + b, i0 + c]); self.c.append(col); self.it.append(self.item)

    def quad_strip(self, A, B, col, closed=False):
        n = len(A); m = n if closed else n - 1
        for i in range(m):
            j = (i + 1) % n; self.face([A[i], A[j], B[j], B[i]], col)

    def prism(self, poly, z0, z1, col, top=True, bottom=False, scale_top=1.0):
        P = np.asarray(poly, float); c = P.mean(0)
        lo = [(*p, z0) for p in P]; hi = [(*(c + (p - c) * scale_top), z1) for p in P]
        self.quad_strip(lo, hi, col, closed=True)
        if top: self.face(hi, col)
        if bottom: self.face(lo[::-1], col)
        return hi

    def box(self, c, d, w, l, z0, z1, col):
        """Bar: center c, axis d, length l along d, width w across, from z0 to z1."""
        d = np.asarray(d, float); d = d / max(np.linalg.norm(d), 1e-9); n = np.array([-d[1], d[0]]); c = np.asarray(c, float)
        self.prism([c - d * l / 2 - n * w / 2, c + d * l / 2 - n * w / 2, c + d * l / 2 + n * w / 2, c - d * l / 2 + n * w / 2], z0, z1, col)

    def obj(self, name, coll_, smooth=False):
        """→ the object; smooth — shared vertices (welded within 1 mm) and smooth shading (for vaults: no stripes along the faces)."""
        V = np.asarray(self.v, float); keep = []
        for i, f in enumerate(self.f):                                           # without degenerate faces
            P = V[f]; n = np.cross(P - P[0], np.roll(P, -1, 0) - P[0]).sum(0)
            if float(n @ n) > 1e-10: keep.append(i)
        f_ = [self.f[i] for i in keep]; c_ = [self.c[i] for i in keep]; verts = self.v
        it_ = [self.it[i] for i in keep] if len(self.it) == len(self.f) else [None] * len(keep)
        if smooth:
            _, first, inv = np.unique(np.round(V / 1e-3).astype(np.int64), axis=0, return_index=True, return_inverse=True)
            inv = inv.ravel(); verts = V[first].tolist(); g_ = []; cc_ = []; ii_ = []
            for f, c, t in zip(f_, c_, it_):                                     # converged vertices (the vault keystone): quad → triangle
                q = [int(inv[j]) for j in f]; q = [x for i, x in enumerate(q) if x != q[i - 1]]
                if len(q) >= 3 and len(set(q)) == len(q): g_.append(q); cc_.append(c); ii_.append(t)
            f_, c_, it_ = g_, cc_, ii_
        me = bpy.data.meshes.new(name); me.from_pydata(verts, [], f_)
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")              # marks go before validate (it also cleans the face layers)
        cols = np.repeat(np.array(c_, np.float32), [len(f) for f in f_], axis=0)
        ca.data.foreach_set("color", np.column_stack([cols, np.ones(len(cols), np.float32)]).ravel())
        rl = np.array([self.role if getattr(c, "role", None) is None else c.role for c in c_], np.int32)
        me.attributes.new("sk_role", "INT", "FACE").data.foreach_set("value", rl)
        me.attributes.new("sk_stone", "INT", "FACE").data.foreach_set("value", np.zeros(len(f_), np.int32))
        if any(t is not None for t in it_):
            me.attributes.new(self.item_attr, "INT", "FACE").data.foreach_set("value", np.array([-1 if t is None else int(t) for t in it_], np.int32))
        me.validate(); me.update()
        if smooth: me.shade_smooth()
        me.color_attributes.active_color = me.color_attributes["Col"]
        rl = np.empty(len(me.polygons), np.int32); me.attributes["sk_role"].data.foreach_get("value", rl)
        slot = {r: i for i, r in enumerate(sorted(set(rl.tolist())))}
        for r in slot: me.materials.append(mat_role(r))
        me.polygons.foreach_set("material_index", np.array([slot[r] for r in rl.tolist()], np.int32)); me.update()
        ob = bpy.data.objects.new(name, me); coll_.objects.link(ob); return ob


# ---------------------------------------------------------------- arches
def arch_pts(s, h, n=10):
    """Arc from the left springing (−s/2, 0) to the right one (s/2, 0) through the keystone (0, h): two centers on
    the springing line, r = (h² + s²/4)/s (semicircle — h = s/2; equilateral pointed — h = s·√3/2)."""
    r = (h * h + s * s / 4) / s; xc = r - s / 2; a1 = math.atan2(h, -xc)
    left = [(xc + r * math.cos(math.pi + (a1 - math.pi) * i / n), r * math.sin(math.pi + (a1 - math.pi) * i / n)) for i in range(n + 1)]
    return left + [(-x, z) for x, z in reversed(left[:-1])]


RISE = {"round": 0.5, "pointed": 0.75, "lancet": 1.1, "segment": 0.12, "ogee": 0.9, "horseshoe": 0.866, "parabolic": 1.15, "stepped": 0.5,
        "flat": 0.0, "rect": 0.0, "slit": 0.0}


def arch_rise(kind, s):
    """Rise of an opening's arch by kind: semicircle 0.5 of the span, pointed 0.75, lancet 1.1, ogee 0.9, horseshoe
    0.87, parabola 1.15, stepped 0.5, segmental 0.12, lintel 0."""
    return s * RISE.get(kind, 0.75)


def arch_curve(kind, w, rise, n=10):
    """Top of an opening from the left springing (−w/2, 0) to the right one (w/2, 0) by arch kind; z is above the
    springing. Ogee — an S-curve with a point; horseshoe — a 240° arc (wider than the span); parabola — from the
    springings to the keystone (Gaudi); stepped — corbelled stones (dwarves)."""
    if kind in ("round", "pointed", "lancet", "segment"): return arch_pts(w, rise, n)
    if kind == "ogee":
        L = [(-w / 2 + (w / 2) * (1 - math.cos(math.pi * t)) / 2, rise * t) for t in np.linspace(0, 1, n + 1)]
        return L + [(-x, z) for x, z in reversed(L[:-1])]
    if kind == "horseshoe":
        r = w / (2 * math.cos(math.radians(30))); zc = r * 0.5
        sc = rise / (zc + r)
        return [(r * math.cos(math.radians(t)), (zc + r * math.sin(math.radians(t))) * sc) for t in np.linspace(210, -30, 2 * n + 1)]
    if kind == "parabolic":
        return [(float(x), rise * (1 - (2 * x / w) ** 2)) for x in np.linspace(-w / 2, w / 2, 2 * n + 1)]
    if kind == "stepped":
        k = 3; d = w / 8; r1 = rise / k; pts = [(-w / 2, 0.0)]
        for i in range(k):
            x = -w / 2 + i * d; pts += [(x, (i + 1) * r1), (x + d, (i + 1) * r1)]
        return pts + [(-x, z) for x, z in reversed(pts)]
    return [(-w / 2, 0.0), (w / 2, 0.0)]


def hole_outline(sc, z0, w, h, kind, n=10):
    """Outline of an opening in the wall plane (s, z): straight jambs up to the springings and an arch of any kind
    (arch_curve) or a rectangle; circle — a circle (rose window, oculus) centered at (sc, z0 + h/2) with
    diameter h."""
    if kind == "circle":
        r = h / 2; m = 4 * n
        return [(sc + r * math.cos(2 * math.pi * i / m), z0 + r + r * math.sin(2 * math.pi * i / m)) for i in range(m)]
    rs = min(arch_rise(kind, w), h * 0.95); zs = z0 + h - rs
    if rs < 1e-3: return [(sc - w / 2, z0), (sc + w / 2, z0), (sc + w / 2, z0 + h), (sc - w / 2, z0 + h)]
    top = [(sc + x, zs + z) for x, z in arch_curve(kind, w, rs, n)]
    out = [(sc - w / 2, z0), (sc + w / 2, z0)] + top[::-1]
    return [p for i, p in enumerate(out) if i == 0 or abs(p[0] - out[i - 1][0]) + abs(p[1] - out[i - 1][1]) > 1e-6]


# ---------------------------------------------------------------- wall with openings
def wall_panel(a, p0, p1, z0, z1, t, holes, col, col_jamb=None, z1_end=None, top=None):
    """Wall along p0 → p1 (world, xy) of thickness t (equally on both sides), from z0 to z1 (at p1 — z1_end: a
    slope; top — the top polyline [(s, z)] from s=0 to s=L: a gable); holes — through openings with jambs: (s
    from p0, z of the bottom, width, height, kind) or {"pts": [(s, z)…]} — a custom outline. The face and the
    back are triangles of a polygon with holes (tessellate_polygon), no boolean operations. An opening that
    crosses the wall edge is skipped. → the number of skipped openings."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); L = float(np.linalg.norm(p1 - p0))
    if L < 1e-6: return 0
    d = (p1 - p0) / L; nrm = np.array([d[1], -d[0]]); z1e = z1 if z1_end is None else z1_end
    top = [(0.0, z1), (L, z1e)] if top is None else [(float(s_), float(z_)) for s_, z_ in top]
    ts = np.array([q[0] for q in top]); tz = np.array([q[1] for q in top])
    outer = [(0.0, z0), (L, z0)] + top[::-1]
    loops = [outer]; skipped = 0
    for h_ in holes:
        H = list(h_["pts"]) if isinstance(h_, dict) else hole_outline(*h_)
        hs = np.array([q[0] for q in H]); hz = np.array([q[1] for q in H])
        if hs.min() > 0.05 and hs.max() < L - 0.05 and hz.min() > z0 + 0.01 and np.all(hz < np.interp(hs, ts, tz) - 0.05): loops.append(H)
        else: skipped += 1
    P2 = [p for lp in loops for p in lp]
    tris = MG.tessellate_polygon([[Vector((x, z, 0.0)) for x, z in lp] for lp in loops])
    def W3(s_, z_, off): q = p0 + d * s_ + nrm * off; return (q[0], q[1], z_)
    front = [W3(x, z, t / 2) for x, z in P2]; back = [W3(x, z, -t / 2) for x, z in P2]
    a.tri(front, [(i, j, k) for i, j, k in tris], col); a.tri(back, [(i, k, j) for i, j, k in tris], col)
    cj = col_jamb or shade(col, 0.82)
    for li, lp in enumerate(loops):                                              # jambs of the openings and the wall ends
        n = len(lp)
        for i in range(n):
            j = (i + 1) % n; A, Bp = lp[i], lp[j]
            a.face([W3(*A, t / 2), W3(*Bp, t / 2), W3(*Bp, -t / 2), W3(*A, -t / 2)], cj if li else col)
    return skipped


def porch(a, p_c, nrm, z0, w, h, kind, depth, col, col_in=None, k=1.6, gable=True):
    """Portal: a frame in front of the wall (projection depth) with an opening k× wider, a splay (stepped
    archivolts) from the frame to the opening in the wall, a wimperg gable over the arch (Gothic). p_c is the
    middle of the opening's bottom on the wall's outer face, nrm points outward."""
    p_c = np.asarray(p_c, float); nrm = np.asarray(nrm, float); nrm = nrm / max(np.linalg.norm(nrm), 1e-9); tg = np.array([nrm[1], -nrm[0]])
    wo = w * k; ho = h + (wo - w) * 0.5; ci = col_in or shade(col, 0.86)
    inner = hole_outline(0.0, z0, w, h, kind); outer = hole_outline(0.0, z0, wo, ho, kind)
    half = wo / 2 + 0.22 * wo; zt = z0 + ho + 0.12 * wo; zg = zt + (half * 1.1 if gable else 0.0)
    prof = [(0.0, zt), (half, zg), (2 * half, zt)] if gable else [(0.0, zt), (2 * half, zt)]
    q0 = p_c - tg * half + nrm * depth / 2; q1 = p_c + tg * half + nrm * depth / 2
    wall_panel(a, q0, q1, z0 - 0.02, zt, depth, [{"pts": [(x + half, z) for x, z in outer]}], col, top=prof)
    def W3(x, z, off): q = p_c + tg * x + nrm * off; return (q[0], q[1], z)
    n = len(inner); steps = 3                                                    # stepped splay
    for s_ in range(steps):
        f0, f1 = s_ / steps, (s_ + 1) / steps
        A = [W3(xo + (xi - xo) * f0, zo + (zi - zo) * f0, depth * (1 - f0)) for (xo, zo), (xi, zi) in zip(outer, inner)]
        B = [W3(xo + (xi - xo) * f1, zo + (zi - zo) * f1, depth * (1 - f1) + 0.001) for (xo, zo), (xi, zi) in zip(outer, inner)]
        for i in range(1, n):                                                    # without the bottom edge (threshold)
            j = (i + 1) % n; a.face([A[i], A[j], B[j], B[i]], ci)


def steps(a, p_c, d, w, z_lo, z_hi, col, rise=0.17, tread=0.34):
    """Stair from z_lo up to z_hi in direction d (toward the building): p_c is the middle of the top edge; the
    steps are beams."""
    p_c = np.asarray(p_c, float); d = np.asarray(d, float); d = d / max(np.linalg.norm(d), 1e-9)
    n = max(1, int(round((z_hi - z_lo) / rise))); r_ = (z_hi - z_lo) / n
    for i in range(n):
        c = p_c - d * (n - i - 0.5) * tread; a.box(c, d, w, tread + 0.01, z_lo - 0.3, z_lo + (i + 1) * r_, col)


# ---------------------------------------------------------------- vaults
def rib_vault(a, C4, z_spring, rise, col, col_rib=None, kind="rib4", n=8, m=8, rib_w=0.35, band_w=None, a_rib=None):
    """Vault of a bay on four corners C4 (world, xy, in order around; sides 0 and 2 are the transverse ones): four
    webs from the sides to the center (as in the trial: groin — cylinders along the sides; ribbed — edges along
    the diagonals, a horizontal ridge); ribs — the diagonals and the transverse ones; the groin vault has only
    the transverse arches (band_w)."""
    C4 = [np.asarray(p, float) for p in C4]; Cc = sum(C4) / 4; D = max(np.linalg.norm(C4[2] - C4[0]), np.linalg.norm(C4[3] - C4[1]))
    ar = a_rib or a; cr = col_rib or shade(col, 0.8)
    if kind == "flat" or rise < 1e-3:                                             # flat ceiling with beams — coffers (dwarves)
        a.face([(*p, z_spring) for p in C4], col)
        for f in (0.0, 0.5, 1.0):
            p, q = C4[0] + (C4[3] - C4[0]) * f, C4[1] + (C4[2] - C4[1]) * f; d = q - p; L_ = float(np.linalg.norm(d))
            if L_ > 1e-6: ar.box((p + q) / 2, d, rib_w * 1.6, L_, z_spring - rib_w * 2.2, z_spring, cr)
        for f in (1 / 3, 2 / 3):
            p, q = C4[0] + (C4[1] - C4[0]) * f, C4[3] + (C4[2] - C4[3]) * f; d = q - p; L_ = float(np.linalg.norm(d))
            if L_ > 1e-6: ar.box((p + q) / 2, d, rib_w * 1.2, L_, z_spring - rib_w * 1.6, z_spring, cr)
        return
    if kind == "barrel":                                                         # a cylinder along the bay, transverse arches at the edges
        S0 = float(np.linalg.norm(C4[1] - C4[0])); rows = []
        for g in np.linspace(0, 1, 3):
            p0, p1 = C4[0] + (C4[3] - C4[0]) * g, C4[1] + (C4[2] - C4[1]) * g
            rows.append([(*(p0 + (p1 - p0) * f), z_spring + _arch_z(S0, rise, (f - 0.5) * S0)) for f in np.linspace(0, 1, 2 * n + 1)])
        for A_, B_ in zip(rows[:-1], rows[1:]): a.quad_strip(A_, B_, col)
        for k in (0, 2): _rib(ar, C4[k], C4[(k + 1) % 4], z_spring, rise, band_w or rib_w, cr)
        return
    for k in range(4):
        P, Q = C4[k], C4[(k + 1) % 4]; S = float(np.linalg.norm(Q - P))
        grid = []
        for j in range(m + 1):
            v = j / m; zd = _arch_z(D, rise, (1 - v) * D / 2); row = []
            for i in range(n + 1):
                f = i / n; base = P + (Q - P) * f; x = base + (Cc - base) * v; aa = (f - 0.5) * S
                z = _arch_z(S, rise, aa * (1 - v)) if kind == "groin" else zd + _arch_z(S, rise, aa) / max(rise, 1e-6) * (rise - zd)
                row.append((x[0], x[1], z_spring + z))
            grid.append(row)
        for j in range(m):
            for i in range(n): a.face([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]], col)
    if kind != "groin":                                                          # ribs: diagonals
        for k in range(2):
            P = C4[k]; Q = C4[k + 2]; _rib(ar, P, Q, z_spring, rise, rib_w, cr)
    for k in (0, 2):                                                             # transverse ones (for the groin vault — the transverse arches)
        P, Q = C4[k], C4[(k + 1) % 4]; _rib(ar, P, Q, z_spring, rise, band_w or rib_w, cr)
    if kind == "star":                                                           # star vault: a ridge rib and tiercerons to its thirds
        m0, m2 = (C4[0] + C4[1]) / 2, (C4[2] + C4[3]) / 2; R1, R2 = m0 + (m2 - m0) / 3, m0 + 2 * (m2 - m0) / 3; zc = z_spring + rise - 0.05
        beam(ar, [(m0 + (m2 - m0) * f, zc) for f in np.linspace(0, 1, 5)], rib_w * 0.8, cr)
        for P, R in ((C4[0], R1), (C4[1], R1), (C4[2], R2), (C4[3], R2)):
            beam(ar, [(P + (R - P) * f, z_spring + rise * math.sqrt(max(0.0, 1 - (1 - f) ** 2))) for f in np.linspace(0, 1, 10)], rib_w * 0.8, cr)


def radial_vault(a, c, ring, z_spring, rise, col, col_rib=None, rf=None, n=8, m=8, rib_w=0.35, ribs=True, a_rib=None):
    """Vault over a polygon (apse, choir hemicycle, chapel): triangular webs from the chords of the ring to the
    keystone above c; ribs — quarter arcs from the ring supports to the keystone; wall arcs of the chords — rise
    rf (by default the same as the vault's)."""
    c = np.asarray(c, float); R = [np.asarray(p, float) for p in ring]
    for P, Q in zip(R[:-1], R[1:]):
        S = float(np.linalg.norm(Q - P)); rf_ = min(rf or rise, rise); grid = []
        for j in range(m + 1):
            v = j / m; ze = rise * math.sqrt(max(0.0, 1 - (1 - v) ** 2)); row = []
            for i in range(n + 1):
                f = i / n; base = P + (Q - P) * f; x = base + (c - base) * v; aa = (f - 0.5) * S
                z = ze + _arch_z(S, rf_, aa) / max(rf_, 1e-6) * (rf_ - ze) if ze < rf_ else ze
                row.append((x[0], x[1], z_spring + z))
            grid.append(row)
        for j in range(m):
            for i in range(n): a.face([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]], col)
    if ribs:
        cr = col_rib or shade(col, 0.8)
        for P in R:
            pts = [(P + (c - P) * f, z_spring + rise * math.sqrt(max(0.0, 1 - (1 - f) ** 2))) for f in np.linspace(0, 1, 12)]
            beam(a_rib or a, pts, rib_w, cr)


def beam(a, pts, w, col, drop=0.05):
    """Beam of square section w × w under a polyline [(xy, z)] (a rib): the sides and the bottom."""
    P = [np.asarray(x, float) for x, _ in pts]; Z = [float(z) for _, z in pts]
    d = P[-1] - P[0]; d = d / max(np.linalg.norm(d), 1e-9); nr = np.array([-d[1], d[0]])
    A = [(*(x + nr * w / 2), z - drop) for x, z in zip(P, Z)]; B = [(*(x - nr * w / 2), z - drop) for x, z in zip(P, Z)]
    A2 = [(*(x + nr * w / 2), z - drop - w) for x, z in zip(P, Z)]; B2 = [(*(x - nr * w / 2), z - drop - w) for x, z in zip(P, Z)]
    a.quad_strip(A, A2, col); a.quad_strip(B2, B, col); a.quad_strip(A2, B2, col)


def _arch_z(s, h, x):
    r = (h * h + s * s / 4) / s; xc = r - s / 2
    return math.sqrt(max(0.0, r * r - (abs(x) + xc) ** 2))


def _rib(a, P, Q, z_spring, rise, w, col, n=12):
    """Rib: a beam along an arc from P to Q (the span is the distance, the rise is the same as the vault's)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float); L = float(np.linalg.norm(Q - P))
    beam(a, [(P + (Q - P) * f, z_spring + _arch_z(L, rise, (f - 0.5) * L)) for f in np.linspace(0, 1, n + 1)], w, col)


# ---------------------------------------------------------------- exterior: flying buttresses, pinnacles, roofs
def pinnacle(a, p, z0, w, h, col, col_cap=None):
    """Pinnacle (Roriczer): a square shaft and a small pyramid three times as tall as the width."""
    p = np.asarray(p, float); a.box(p, (1.0, 0.0), w, w, z0, z0 + h, col)
    q = [p + np.array(v) * w / 2 for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4): a.face([(*q[i], z0 + h), (*q[(i + 1) % 4], z0 + h), (*p, z0 + h + 3.2 * w)], col_cap or col)


def flyer(a, pp, pw, z_pier, z_wall, w, th, col):
    """Flying buttress: from the pier pp (top z_pier) to the wall pw (at the vault springing z_wall): the top is a
    straight slope, the bottom is a quarter arch."""
    pp = np.asarray(pp, float); pw = np.asarray(pw, float); L = float(np.linalg.norm(pw - pp)); d = (pw - pp) / max(L, 1e-9); nr = np.array([-d[1], d[0]])
    n = 10; top = []; bot = []
    for i in range(n + 1):
        f = i / n; x = pp + (pw - pp) * f; zt = z_pier + (z_wall - z_pier) * f + th * 0.5
        z_lo = z_pier - th * 1.6; z_hi = z_wall - th
        zb = z_lo + (z_hi - z_lo) * math.sqrt(max(0.0, 1 - (1 - f) ** 2))          # quarter arch: up from the pier, gently sloping toward the wall
        top.append((x, zt)); bot.append((x, min(zb, zt - th * 0.8)))
    for s_ in (-1, 1):
        a.quad_strip([(*(x + nr * s_ * w / 2), z) for x, z in top], [(*(x + nr * s_ * w / 2), z) for x, z in bot], col)
    a.quad_strip([(*(x + nr * w / 2), z) for x, z in top], [(*(x - nr * w / 2), z) for x, z in top], col)
    a.quad_strip([(*(x - nr * w / 2), z) for x, z in bot], [(*(x + nr * w / 2), z) for x, z in bot], col)


def roof_gable(a, p0, p1, half_w, z_eave, pitch, col, ov=0.6, hip0=False, hip1=False):
    """Gable roof along p0 → p1 of width 2·half_w; a hip at the end if hip is set."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); L = float(np.linalg.norm(p1 - p0)); d = (p1 - p0) / max(L, 1e-9); n_ = np.array([-d[1], d[0]])
    hw = half_w + ov; zr = z_eave + half_w * math.tan(math.radians(pitch))
    r0 = p0 + d * (half_w if hip0 else -ov); r1 = p1 - d * (half_w if hip1 else -ov)
    e = [p0 - d * ov - n_ * hw, p1 + d * ov - n_ * hw, p1 + d * ov + n_ * hw, p0 - d * ov + n_ * hw]
    a.face([(*e[0], z_eave - 0.3), (*e[1], z_eave - 0.3), (*r1, zr), (*r0, zr)], col)
    a.face([(*e[2], z_eave - 0.3), (*e[3], z_eave - 0.3), (*r0, zr), (*r1, zr)], col)
    if hip0: a.face([(*e[3], z_eave - 0.3), (*e[0], z_eave - 0.3), (*r0, zr)], col)
    if hip1: a.face([(*e[1], z_eave - 0.3), (*e[2], z_eave - 0.3), (*r1, zr)], col)
    return zr


def roof_lean(a, p0, p1, n_out, w, z_low, z_high, col, ov=0.5):
    """Shed roof along p0 → p1: the high edge at the wall, the low one w outward (n_out)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); n_out = np.asarray(n_out, float); d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    a.face([(*(p0 - d * ov), z_high), (*(p1 + d * ov), z_high), (*(p1 + d * ov + n_out * (w + ov)), z_low - 0.2), (*(p0 - d * ov + n_out * (w + ov)), z_low - 0.2)], col)


def roof_cone(a, ring, z, h, col, ov=1.08):
    """Cone (tent) over a ring: overhang ×ov, the apex h above the center."""
    P = np.asarray(ring, float); c = P.mean(0); E = [c + (p - c) * ov for p in P]; apex = (*c, z + h)
    for i in range(len(E)): a.face([(*E[i], z - 0.2), (*E[(i + 1) % len(E)], z - 0.2), apex], col)


def roof_fan(a, apex, ring, z_eave, col):
    """Fan roof from the eaves ring (world, xy; an open polyline) to the vertex apex (x, y, z): the half-cone of an
    apse, a tent."""
    R = [np.asarray(p, float) for p in ring]
    for P, Q in zip(R[:-1], R[1:]): a.face([(*P, z_eave), (*Q, z_eave), tuple(apex)], col)


def lathe(a, c, prof, z0, col, n=24):
    R = [[(c[0] + r_ * math.cos(2 * math.pi * i / n), c[1] + r_ * math.sin(2 * math.pi * i / n), z0 + h_) for i in range(n)] for r_, h_ in prof]
    for A, B in zip(R[:-1], R[1:]): a.quad_strip(A, B, col, closed=True)


def bar(a, p_c, nrm, pts, w, dpt, col):
    """Tracery bar along a polyline [(x, z)] in the window plane (x runs along the wall from p_c): section w in the
    plane × dpt along the normal."""
    p_c = np.asarray(p_c, float); nrm = np.asarray(nrm, float); nrm = nrm / max(np.linalg.norm(nrm), 1e-9); tg = np.array([-nrm[1], nrm[0]])
    P = np.asarray(pts, float)
    if len(P) < 2: return
    T_ = np.gradient(P, axis=0); T_ /= np.maximum(np.linalg.norm(T_, axis=1, keepdims=True), 1e-9); N_ = np.stack([-T_[:, 1], T_[:, 0]], 1)
    Aq = P + N_ * w / 2; Bq = P - N_ * w / 2
    def W3(x, z, off): q = p_c + tg * x + nrm * off; return (q[0], q[1], z)
    Af = [W3(*q, dpt / 2) for q in Aq]; Ab = [W3(*q, -dpt / 2) for q in Aq]; Bf = [W3(*q, dpt / 2) for q in Bq]; Bb = [W3(*q, -dpt / 2) for q in Bq]
    a.quad_strip(Af, Bf, col); a.quad_strip(Bb, Ab, col); a.quad_strip(Ab, Af, col); a.quad_strip(Bf, Bb, col)


def _circle(xc, zc, r, n=20):
    return [(xc + r * math.cos(2 * math.pi * i / n), zc + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]


def tracery(a, p_c, nrm, z0, w, h, col, th=0.18, kind="pointed", style="bar"):
    """Window tracery from bars along arcs (Havemann 2004, simplified): bar — a mullion, two lancets, an oculus
    with a quatrefoil; flamboyant — ogee lancets and "fish bladders" (S-bars) in the head; grid — a grid of
    mullions and transoms (Perpendicular); flowing — parabolic lancets and a leaf in the middle (elves); plate
    and none — without bars (plate tracery — by openings in the wall)."""
    if style in ("none", "plate", None): return
    rs = min(arch_rise(kind, w), h * 0.9); zs = z0 + h - rs
    if style == "grid":                                                          # Perpendicular: 3 mullions, 2 transoms
        for x in (-w / 4, 0.0, w / 4): bar(a, p_c, nrm, [(x, z0), (x, zs + rs * (0.55 if x else 0.75))], th * 0.8, th * 1.2, col)
        for f in (0.35, 0.68): bar(a, p_c, nrm, [(-w / 2, z0 + (zs - z0) * f), (w / 2, z0 + (zs - z0) * f)], th * 0.7, th * 1.1, col)
        for x in (-3 * w / 8, -w / 8, w / 8, 3 * w / 8):
            bar(a, p_c, nrm, [(x + xx, zs - 0.15 * w + zz) for xx, zz in arch_curve("ogee", w / 4, 0.18 * w, 5)], th * 0.6, th, col)
        return
    lk = "ogee" if style == "flamboyant" else ("parabolic" if style == "flowing" else kind)
    rl = min(arch_rise(lk, w / 2) * 0.62, rs * 0.6)
    bar(a, p_c, nrm, [(0.0, z0), (0.0, zs + rl)], th, th * 1.3, col)               # mullion
    for s_ in (-1, 1):                                                           # lancet heads
        bar(a, p_c, nrm, [(s_ * w / 4 + x, zs + z) for x, z in arch_curve(lk, w / 2, rl, 8)], th * 0.85, th * 1.2, col)
    ro = 0.21 * w; zc = zs + rs - 1.12 * ro
    if style == "flamboyant":                                                    # two S-bars — a flame
        for s_ in (-1, 1):
            pts = [(s_ * (0.05 * w + 0.3 * w * math.sin(math.pi * f) * (1 - f)), zs + rl + (rs - rl) * 0.92 * f) for f in np.linspace(0, 1, 9)]
            bar(a, p_c, nrm, pts, th * 0.7, th, col)
        return
    if style == "flowing":                                                       # a leaf (mandorla) in the middle
        lf = [(0.16 * w * math.sin(math.pi * f), zs + rl + (rs - rl) * 0.9 * f) for f in np.linspace(0, 1, 9)]
        bar(a, p_c, nrm, lf, th * 0.7, th, col); bar(a, p_c, nrm, [(-x, z) for x, z in lf], th * 0.7, th, col)
        return
    if zc - ro > zs + rl * 0.7:                                                  # the oculus fits above the lancets
        bar(a, p_c, nrm, _circle(0.0, zc, ro), th * 0.8, th * 1.1, col)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            bar(a, p_c, nrm, _circle(dx * ro * 0.42, zc + dz * ro * 0.42, ro * 0.4, 14), th * 0.5, th, col)


def rose(a, p_c, nrm, zc, r, col, n=12, th=0.25):
    """Rose window: a rim, an inner ring, spokes and small circles between the spokes."""
    bar(a, p_c, nrm, _circle(0.0, zc, r * 0.97, 48), th * 1.2, th * 1.4, col)
    bar(a, p_c, nrm, _circle(0.0, zc, r * 0.3, 24), th, th * 1.2, col)
    for i in range(n):
        t_ = 2 * math.pi * i / n
        bar(a, p_c, nrm, [(r * 0.3 * math.cos(t_), zc + r * 0.3 * math.sin(t_)), (r * 0.96 * math.cos(t_), zc + r * 0.96 * math.sin(t_))], th * 0.8, th * 1.1, col)
        t2 = t_ + math.pi / n
        bar(a, p_c, nrm, _circle(r * 0.68 * math.cos(t2), zc + r * 0.68 * math.sin(t2), r * 0.15, 14), th * 0.6, th, col)


# ---------------------------------------------------------------- shapes of temples and structures
def bar3(a, p, q, w, col, w2=None):
    """Beam of square section w (at q — w2) between points p and q (x, y, z): spire ribs, horns, the crane."""
    p = np.asarray(p, float); q = np.asarray(q, float); d = q - p; L = float(np.linalg.norm(d))
    if L < 1e-6: return
    d /= L; up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, up); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1); w2 = w if w2 is None else w2
    R0 = [p + (e1 * x + e2 * y) * w / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    R1 = [q + (e1 * x + e2 * y) * w2 / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    a.quad_strip([tuple(v) for v in R0], [tuple(v) for v in R1], col, closed=True); a.face([tuple(v) for v in R1], col)


def tube(a, pts, radii, col, n=8, cap=True):
    """Tube along a polyline pts (x, y, z) with radii radii (branch buttresses, horns, forge chimneys); the rings
    are across the tangent."""
    P = np.asarray(pts, float); R = np.broadcast_to(np.asarray(radii, float), (len(P),)); rings = []
    for i in range(len(P)):
        t_ = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; t_ /= max(np.linalg.norm(t_), 1e-9)
        up = np.array([0.0, 0.0, 1.0]) if abs(t_[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(t_, up); e1 /= np.linalg.norm(e1); e2 = np.cross(t_, e1)
        rings.append([tuple(P[i] + (e1 * math.cos(2 * math.pi * j / n) + e2 * math.sin(2 * math.pi * j / n)) * R[i]) for j in range(n)])
    for A, B in zip(rings[:-1], rings[1:]): a.quad_strip(A, B, col, closed=True)
    if cap and R[-1] > 1e-3: a.face(rings[-1], col)


def ring_bar(a, c, r, z, w, col, n=8, phase=math.pi / 8):
    """Ring beam (a band of an openwork spire, a rim)."""
    c = np.asarray(c, float)
    Q = [c + r * np.array([math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n)]) for i in range(n)]
    for p, q in zip(Q, Q[1:] + Q[:1]):
        d = q - p; L = float(np.linalg.norm(d))
        if L > 1e-6: a.box((p + q) / 2, d, w, L + w * 0.5, z - w / 2, z + w / 2, col)


def spike(a, p, z0, w, h, col, lean=(0.0, 0.0)):
    """Spike: a thin four-sided pyramid (base w, height h, apex tilt lean)."""
    p = np.asarray(p, float); q = [p + np.array(v) * w / 2 for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]; apex = (p[0] + lean[0], p[1] + lean[1], z0 + h)
    for i in range(4): a.face([(*q[i], z0), (*q[(i + 1) % 4], z0), apex], col)


def spike_row(a, P, z, step, w, h, col, closed=False):
    """Spikes along a polyline P (world, xy) at height z with spacing step."""
    P = [np.asarray(p, float) for p in P] + ([np.asarray(P[0], float)] if closed else [])
    for p0, p1 in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(p1 - p0)); n = max(1, int(L // step))
        for i in range(n): spike(a, p0 + (p1 - p0) * (i + 0.5) / n, z, w, h, col)


def slab(a, p, n_out, tg, prof, w, col):
    """Profile slab: prof — [(s outward, z)] in the plane perpendicular to the wall at point p (n_out points
    outward, tg runs along the wall), thickness w along tg (a fin, a blade, a stair cheek)."""
    p = np.asarray(p, float); n_out = np.asarray(n_out, float); tg = np.asarray(tg, float)
    F = [(*(p + n_out * s_ + tg * w / 2), z) for s_, z in prof]; B = [(*(p + n_out * s_ - tg * w / 2), z) for s_, z in prof]
    a.face(F, col); a.face(B[::-1], col); a.quad_strip(F, B, col, closed=True)


def fin(a, p, n_out, z0, z1, d0, d1, w, col):
    """Fin-blade at a wall: projection d0 at the bottom, d1 at the top (tapering), thickness w (forge, dark order, hive)."""
    n_out = np.asarray(n_out, float); tg = np.array([-n_out[1], n_out[0]])
    slab(a, p, n_out, tg, [(0.0, z0), (d0, z0), (d1, z1 - 0.15 * (z1 - z0)), (d1 * 0.4, z1), (0.0, z1)], w, col)


def claw(a, pp, pw, z_pier, z_wall, w, th, col, col_spike=None):
    """Claw (dark temple, hell): a flying buttress with a concave top and a hook spike on the pier."""
    pp = np.asarray(pp, float); pw = np.asarray(pw, float); L = float(np.linalg.norm(pw - pp)); d = (pw - pp) / max(L, 1e-9); nr = np.array([-d[1], d[0]])
    top = []; bot = []
    for f in np.linspace(0, 1, 13):
        x = pp + (pw - pp) * f; zt = z_pier + (z_wall - z_pier) * f ** 0.55 + th * 0.4
        zb = z_pier - th * 1.4 + (z_wall - th - (z_pier - th * 1.4)) * math.sqrt(max(0.0, 1 - (1 - f) ** 2)) ** 1.6
        top.append((x, zt)); bot.append((x, min(zb, zt - th * 0.6)))
    for s_ in (-1, 1): a.quad_strip([(*(x + nr * s_ * w / 2), z) for x, z in top], [(*(x + nr * s_ * w / 2), z) for x, z in bot], col)
    a.quad_strip([(*(x + nr * w / 2), z) for x, z in top], [(*(x - nr * w / 2), z) for x, z in top], col)
    a.quad_strip([(*(x - nr * w / 2), z) for x, z in bot], [(*(x + nr * w / 2), z) for x, z in bot], col)
    spike(a, pp - d * th * 0.3, z_pier + th * 0.4, w * 1.1, th * 4.0, col_spike or col, lean=tuple(-d * th * 1.6))


def branch(a, pp, pw, z_pier, z_wall, w, col):
    """Branch (elves): a flying buttress in a smooth arc, thicker at the pier, thinner at the wall, with a hump
    upward."""
    pp = np.asarray(pp, float); pw = np.asarray(pw, float); mid = (pp + pw) / 2; zm = max(z_pier, z_wall) + 0.25 * float(np.linalg.norm(pw - pp))
    pts = []; rad = []
    for f in np.linspace(0, 1, 12):
        b0 = (1 - f) ** 2; b1 = 2 * (1 - f) * f; b2 = f * f; xy = pp * b0 + mid * b1 + pw * b2
        pts.append((xy[0], xy[1], z_pier * b0 + zm * b1 + z_wall * b2)); rad.append(w * (0.75 - 0.4 * f))
    tube(a, pts, rad, col, n=8)


def stack(a, c, r, z0, z1, col, col_band, col_soot=Col((0.05, 0.05, 0.05), 4)):
    """Forge chimney: a cylinder with bands, a rim and soot on top (forge cult, hive)."""
    lathe(a, c, [(r, 0.0), (r, z1 - z0)], z0, col, n=16)
    for f in (0.3, 0.62, 0.9): lathe(a, c, [(r * 1.12, 0.0), (r * 1.12, 0.35 * r)], z0 + (z1 - z0) * f, col_band, n=16)
    lathe(a, c, [(r * 1.18, 0.0), (r * 1.18, 0.5 * r), (r * 0.82, 0.5 * r)], z1, col_band, n=16)
    lathe(a, c, [(r * 0.82, 0.48 * r), (0.01, 0.2 * r)], z1, col_soot, n=16)


def crane(a, p, z, size, col, out=(1.0, 0.0)):
    """Medieval crane on a tower stump (Cologne, ~400 years): a post, a brace, a booth."""
    p = np.asarray(p, float); o = np.asarray(out, float); o = o / max(np.linalg.norm(o), 1e-9); w = 0.06 * size
    a.box(p, (1.0, 0.0), 0.35 * size, 0.35 * size, z, z + 0.3 * size, col)
    bar3(a, (*p, z + 0.3 * size), (*p, z + 1.0 * size), w * 1.6, col)
    bar3(a, (*p, z + 0.85 * size), (*(p + o * 0.9 * size), z + 1.25 * size), w * 1.2, col)
    bar3(a, (*(p + o * 0.85 * size), z + 1.22 * size), (*(p + o * 0.85 * size), z + 0.5 * size), w * 0.3, col)


def merlon_row(a, P, z, s, col, closed=True):
    """Merlons along a polyline (fortress cathedral, dwarves): merlon s, gap 0.8 s, height 1.2 s, thickness 0.5 s."""
    P = [np.asarray(p, float) for p in P] + ([np.asarray(P[0], float)] if closed else [])
    for p0, p1 in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(p1 - p0)); d = (p1 - p0) / max(L, 1e-9); n = int(L // (1.8 * s))
        for i in range(n):
            a.box(p0 + d * (L - n * 1.8 * s + 0.8 * s) / 2 + d * (i * 1.8 * s + s / 2), d, 0.5 * s, s, z, z + 1.2 * s, col)


def octagon(c, r, phase=math.pi / 8, n=8):
    c = np.asarray(c, float); rr = r / math.cos(math.pi / n)
    return [c + rr * np.array([math.cos(phase + 2 * math.pi * i / n), math.sin(phase + 2 * math.pi * i / n)]) for i in range(n)]


def dome_profile(kind, r, h, n=10):
    """Dome profile [(radius, height)] from the springing to the apex: hemisphere, pointed (Florence), onion (Rus),
    saucer."""
    if kind == "onion":
        return [(r * 0.86, 0.0), (r * 1.16, 0.2 * h), (r * 1.08, 0.42 * h), (r * 0.66, 0.66 * h), (r * 0.22, 0.86 * h), (0.02, h)]
    if kind == "pointed":
        rr = (h * h + r * r) / (2 * r); xc = rr - r                               # pointed: two centers on the springing
        return [(r - (rr - math.sqrt(max(0.0, rr * rr - z * z))), z) for z in np.linspace(0, h, n + 1)][:-1] + [(0.02, h)]
    if kind == "leaf":                                                           # keel-shaped (elves): a bulge and a long point
        return [(r * (1 - f) ** 1.4 * (1 + 0.55 * math.sin(math.pi * min(f / 0.55, 1.0))), f * h) for f in np.linspace(0, 1, n + 1)][:-1] + [(0.01, h)]
    return [(r * math.cos(t), h * math.sin(t)) for t in np.linspace(0, math.pi / 2, n + 1)][:-1] + [(0.02, h)]


def dome(a, ar, c, r, z0, h, kind, cols, ribs=0, lantern=True):
    """Dome by profile (hemisphere, pointed, onion, leaf, ribs) with a lantern and a finial."""
    lathe(ar, c, dome_profile("hemi" if kind == "ribbed" else kind, r, h), z0, cols["roof"], n=24)
    if kind == "ribbed" or ribs:                                                 # dome ribs (forge cult, Florence)
        for i in range(ribs or 16):
            t_ = 2 * math.pi * i / (ribs or 16); u = np.array([math.cos(t_), math.sin(t_)]); prof = dome_profile("hemi", r * 1.03, h * 1.03, 8)
            pts = [(*(np.asarray(c) + u * rr), z0 + zz) for rr, zz in prof]
            for p_, q_ in zip(pts[:-1], pts[1:]): bar3(a, p_, q_, 0.06 * r, cols["accent"])
    if lantern and kind not in ("onion", "leaf"):
        rl = 0.16 * r; lathe(a, c, [(rl, 0.0), (rl, 0.4 * r * 0.6)], z0 + h * 0.97, cols["stone"], n=12)
        lathe(ar, c, dome_profile("hemi", rl * 1.1, rl * 1.0), z0 + h * 0.97 + 0.24 * r, cols["roof"], n=12)
        bar3(a, (*c, z0 + h * 0.97 + 0.24 * r + rl), (*c, z0 + h * 0.97 + 0.24 * r + rl + 0.5 * rl + 0.5), 0.12 * rl + 0.05, cols["accent"])
    else:
        bar3(a, (*c, z0 + h * 0.98), (*c, z0 + h + 0.25 * r + 0.5), 0.05 * r + 0.05, cols["accent"])


def drum(a, c, r, z0, h, nwin, cols, t=1.0, n=16, win_kind="round"):
    """Dome drum: n faces with through windows on every other face (light from inside), a cornice."""
    P = octagon(c, r, phase=0.0, n=n)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]; L = float(np.linalg.norm(q - p)); holes = []
        if nwin and i % max(1, n // nwin) == 0 and L > 0.8:
            holes.append((L / 2, z0 + 0.25 * h, min(0.45 * L, 0.35 * h), 0.55 * h, win_kind))
        wall_panel(a, p, q, z0, z0 + h, t, holes, cols["stone"])
    lathe(a, c, [(r / math.cos(math.pi / n) + t, 0.0), (r / math.cos(math.pi / n) + t, 0.35 + 0.02 * h)], z0 + h - 0.2, cols["dark"], n=n)


def pendentives(a, c, half, z_spring, col, m=8):
    """Pendentives: spherical triangles in the corners of a square (side 2·half) from the springings (corners) to
    the dome ring (radius half, height z_spring + half); a sphere of radius half·√2 centered at the springings
    (Hagia Sophia)."""
    c = np.asarray(c, float); R = half * math.sqrt(2)
    for k in range(4):
        phi = math.pi / 4 + k * math.pi / 2; rows = []
        for th in np.linspace(phi - math.pi / 4, phi + math.pi / 4, m + 1):
            u = np.array([math.cos(th), math.sin(th)]); rmax = half / max(abs(u[0]), abs(u[1])); row = []
            for f in np.linspace(0, 1, m + 1):
                rr = half + (rmax - half) * f; row.append((*(c + u * rr), z_spring + math.sqrt(max(0.0, R * R - rr * rr))))
            rows.append(row)
        for A, B in zip(rows[:-1], rows[1:]): a.quad_strip(A, B, col)


def crown(a, ar, c, P, half, plan, z, kind, h, cols, pinnacles=False, fort=False, seed=0.0, s=1.0):
    """Tower top above the top tier (outline P, center c, half-width half, top height h): a platform (parapet,
    pinnacles or merlons), a needle, an openwork needle (Freiburg), a leaf (elves), a cluster of spires, a
    pyramid, a rhombic helm (Sompting), a tent with a cupola (Rus), a dome with a lantern, an onion, steps
    (dwarves), spikes, horns (Orthanc), blades (Barad-dur), a stump (unfinished), fragments (ruin), a chimney."""
    c = np.asarray(c, float); st, dk, rf, ac = cols["stone"], cols["dark"], cols["roof"], cols.get("accent", cols["roof"])
    P = [np.asarray(p, float) for p in P]; sq = plan == "square"
    def parapet(hp=1.2 * s):
        a.quad_strip([(*p, z) for p in P] + [(*P[0], z)], [(*p, z + hp) for p in P] + [(*P[0], z + hp)], st)
    if kind in ("flat", "broken", "stack"):
        if kind == "broken":                                                     # fragments: an uneven top
            for i, (p, q) in enumerate(zip(P, P[1:] + P[:1])):
                f_ = (math.sin(seed * 12.9898 + i * 78.233) * 43758.5453) % 1.0
                a.box((p + q) / 2, q - p, 1.0 * s, float(np.linalg.norm(q - p)) * 0.8, z, z + (0.5 + 2.5 * f_) * s, st)
            return
        parapet()
        if fort: merlon_row(a, P, z + 1.2 * s, 0.9 * s, st)
        if pinnacles and not fort:
            for p in P: pinnacle(a, c + (p - c) * 0.92, z, 0.24 * half, 0.4 * half, st, rf)
        return
    if kind in ("needle", "tent", "cluster", "openwork"):
        P8 = octagon(c, half * 0.9)
        if kind == "openwork":                                                   # through lattice: ribs and bands
            apex = np.array([*c, z + h])
            for p in P8: bar3(a, (*p, z), apex, 0.05 * half + 0.1, st)
            for f in np.linspace(0.12, 0.85, 6): ring_bar(a, c, half * 0.9 * (1 - f), z + h * f, 0.04 * half + 0.08, st)
            bar3(a, (*c, z + h * 0.85), apex + np.array([0, 0, 0.06 * h]), 0.06 * half + 0.1, st)
        else: roof_fan(ar, (*c, z + h), P8 + [P8[0]], z, rf if kind != "cluster" else ac)
        if kind == "tent": lathe(ar, c, dome_profile("onion", 0.16 * half, 0.22 * h), z + h * 0.92, ac, n=12)
        if kind == "cluster" or (pinnacles and sq):                              # corner spires (a cluster) or pinnacles
            for p in (P if sq else P[::2]):
                q = c + (p - c) * 0.84; r_ = 0.2 * half
                roof_fan(ar, (*q, z + h * (0.45 if kind == "cluster" else 0.18)), octagon(q, r_) + [octagon(q, r_)[0]], z, rf)
        return
    if kind == "leaf":
        lathe(ar, c, dome_profile("leaf", half * 0.92, h), z, ac, n=16); return
    if kind == "pyramid":
        roof_fan(ar, (*c, z + h), P + [P[0]], z, rf); return
    if kind == "helm" and sq:                                                    # rhombic helm: gables on the faces, rhombs toward the apex
        G = [((P[i] + P[(i + 1) % 4]) / 2) for i in range(4)]; zg = z + 0.3 * h; apex = (*c, z + h)
        for i in range(4):
            a.face([(*P[i], z), (*P[(i + 1) % 4], z), (*G[i], zg)], st)
            ar.face([(*P[i], z), (*G[i], zg), apex, (*G[i - 1], zg)], rf)
        return
    if kind in ("dome", "onion"):
        if kind == "dome": parapet(0.8 * s)
        dome(a, ar, c, half * (0.86 if kind == "dome" else 0.8), z + (0.8 * s if kind == "dome" else 0.0), h, "hemi" if kind == "dome" else "onion",
             {"stone": st, "roof": rf if kind == "dome" else ac, "accent": ac, "dark": dk}, lantern=kind == "dome"); return
    if kind == "stepped":
        for i in range(3):
            hh = half * (0.82 - 0.2 * i); Q = [c + (p - c) * (hh / half) for p in P] if sq else octagon(c, hh)
            a.prism(Q, z + h * i / 3, z + h * (i + 1) / 3, st)
        return
    if kind == "spikes":
        n = max(8, int(sum(float(np.linalg.norm(q - p)) for p, q in zip(P, P[1:] + P[:1])) / (1.8 * s)))
        L_ = P + [P[0]]
        for i in range(n):
            f = i / n * (len(L_) - 1); j = int(f); p = L_[j] + (L_[j + 1] - L_[j]) * (f - j)
            spike(a, c + (p - c) * 0.95, z, 0.5 * s, h * (0.35 + 0.25 * ((i * 7) % 3) / 2), dk, lean=tuple((p - c) / max(np.linalg.norm(p - c), 1e-9) * 0.3 * s))
        spike(a, c, z, 0.45 * half, h, dk); return
    if kind == "horns":                                                          # four horns (Orthanc): faceted, tapering, pointing outward
        cs_ = [P[i] for i in range(0, len(P), max(1, len(P) // 4))][:4]
        for q in cs_:
            u = (q - c) / max(np.linalg.norm(q - c), 1e-9); bw = max(1.2 * s, 0.32 * half); pts = []; rad = []
            for f in np.linspace(0, 1, 8):
                cc = q - u * bw * 0.6 + u * (0.45 * half * math.sin(math.pi * f * 0.8)); pts.append((*cc, z + h * f)); rad.append(bw * (1 - 0.88 * f) * 0.6)
            tube(a, pts, rad, dk, n=6)
        return
    if kind == "blades":                                                         # crown of blades (Barad-dur)
        nb = max(8, int(2 * math.pi * half / (2.6 * s)))
        for i in range(nb):
            u = np.array([math.cos(2 * math.pi * i / nb), math.sin(2 * math.pi * i / nb)]); tg = np.array([-u[1], u[0]]); q = c + u * half * 0.95
            hb = h * (0.7 + 0.3 * (i % 2)); tip = q + u * hb * 0.18
            for s_ in (-0.15, 0.15): a.face([(*(q - tg * 0.9 * s + u * s_), z), (*(q + tg * 0.9 * s + u * s_), z), (*(tip + u * s_), z + hb)], dk)
        return
    if kind == "stump":                                                          # temporary roof of an unfinished building
        roof_fan(ar, (*c, z + h), P + [P[0]], z, rc((0.30, 0.22, 0.16), "wood")); return
    roof_fan(ar, (*c, z + h), P + [P[0]], z, rf)


# ---------------------------------------------------------------- parts of structures
def offset_poly(P, d):
    """Convex polygon (counter-clockwise) inset by d: each vertex is the intersection of the shifted edges."""
    P = [np.asarray(p, float) for p in P]; n = len(P); out = []
    for i in range(n):
        a_, b_, c_ = P[i - 1], P[i], P[(i + 1) % n]
        e1 = b_ - a_; e2 = c_ - b_; n1 = np.array([-e1[1], e1[0]]) / max(np.linalg.norm(e1), 1e-9); n2 = np.array([-e2[1], e2[0]]) / max(np.linalg.norm(e2), 1e-9)
        out.append(b_ + d * (n1 + n2) / max(1.0 + float(n1 @ n2), 1e-3))
    return out


def seg_d(g, a, b):
    g, a, b = (np.asarray(v, float) for v in (g, a, b)); e = b - a; t = float(np.clip((g - a) @ e / max(float(e @ e), 1e-12), 0, 1)); return float(np.linalg.norm(g - a - e * t))


def ring_runs(P, gaps):
    """Ring P minus the gaps [(point, width)] → a list of open polylines (a parapet and an edge with passages at
    the stair flights)."""
    P = [np.asarray(p, float) for p in P]; n = len(P); runs = []; cur = [P[0]]
    for i in range(n):
        a_, b_ = P[i], P[(i + 1) % n]; L_ = float(np.linalg.norm(b_ - a_)); d = (b_ - a_) / max(L_, 1e-9)
        cuts = []                                                                # gaps on an edge; merge the overlapping ones
        for s0, s1 in sorted((float((np.asarray(g) - a_) @ d) - w / 2, float((np.asarray(g) - a_) @ d) + w / 2) for g, w in gaps if seg_d(g, a_, b_) < 1.5):
            if cuts and s0 <= cuts[-1][1]: cuts[-1] = (cuts[-1][0], max(cuts[-1][1], s1))
            else: cuts.append((s0, s1))
        for s0, s1 in cuts:
            s0, s1 = max(0.0, s0), min(L_, s1)
            if s1 <= s0: continue
            cur.append(a_ + d * s0)
            if len(cur) > 1: runs.append(cur)
            cur = [a_ + d * s1]
        cur.append(b_)
    if gaps and runs and np.linalg.norm(runs[0][0] - P[0]) < 1e-6 and len(cur) > 1: runs[0] = cur[:-1] + runs[0]   # join across the zero vertex
    elif len(cur) > 1: runs.append(cur)
    return runs if gaps else [P + [P[0]]]


def tier(a, P, z0, z1, batter, col, col_top=None, parapet=0.0, tablero=False, col_band=None, gaps=()):
    """Platform tier: sloped walls (the top is narrower by batter·height), the top platform, a parapet; tablero —
    the Maya profile: a sloping talud and a vertical tablero panel on top with a projection; gaps — breaks of
    the parapet at the stair flights
    [(point, width)]."""
    P = [np.asarray(p, float) for p in P]; h = z1 - z0
    if tablero:
        zt = z0 + 0.6 * h; Pm = offset_poly(P, batter * 0.6 * h); Pv = offset_poly(Pm, -0.04 * h)
        a.quad_strip([(*p, z0) for p in P + [P[0]]], [(*p, zt) for p in Pm + [Pm[0]]], col)
        a.quad_strip([(*p, zt) for p in Pm + [Pm[0]]], [(*p, zt) for p in Pv + [Pv[0]]], col_band or col)
        a.quad_strip([(*p, zt) for p in Pv + [Pv[0]]], [(*p, z1) for p in Pv + [Pv[0]]], col_band or col)
        Pt = Pv
    else:
        Pt = offset_poly(P, batter * h); a.quad_strip([(*p, z0) for p in P + [P[0]]], [(*p, z1) for p in Pt + [Pt[0]]], col)
    a.face([(*p, z1) for p in Pt], col_top or col)
    if parapet > 0:                                                              # parapet — a ring or segments between the passages
        Pi = offset_poly(Pt, 0.5 * parapet); c_ = np.mean(Pt, axis=0)
        for run in ring_runs(Pt, gaps):
            Ri = [c_ + (q - c_) * 0.0 for q in run]
            for k_, q in enumerate(run):                                         # inner edge: the same point shifted inward
                j = int(np.argmin([np.linalg.norm(q - x) for x in Pt])); Ri[k_] = q + (Pi[j] - Pt[j]) if np.linalg.norm(q - Pt[j]) < 1e-6 else q + offset_dir(Pt, q, 0.5 * parapet)
            a.quad_strip([(*p, z1) for p in run], [(*p, z1 + parapet) for p in run], col)
            a.quad_strip([(*p, z1 + parapet) for p in run], [(*p, z1 + parapet) for p in Ri], col)
            a.quad_strip([(*p, z1 + parapet) for p in Ri], [(*p, z1) for p in Ri], col)
    return Pt


def offset_dir(P, q, d):
    """Shift of the point q on an edge of the convex polygon P inward by d."""
    P = [np.asarray(p, float) for p in P]; best = None
    for i in range(len(P)):
        a_, b_ = P[i], P[(i + 1) % len(P)]
        if seg_d(q, a_, b_) < 1e-3: e = b_ - a_; best = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-9); break
    return best * d if best is not None else np.zeros(2)


def flight(a, p0, p1, z0, z1, w, col, cheek=0.0, col_cheek=None, rise=0.17, ground=None):
    """Stair flight from p0 (bottom, the middle of the first step) to p1 (top) from z0 to z1: steps as beams down
    to the ground (ground), cheek walls on the sides of height cheek above the steps."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); d = p1 - p0; L = float(np.linalg.norm(d))
    if L < 1e-6: return
    d /= L; nn = np.array([-d[1], d[0]]); n = max(1, int(round((z1 - z0) / rise))); tr = L / n; r_ = (z1 - z0) / n
    zg = z0 - 0.5 if ground is None else ground
    for i in range(n):
        c = p0 + d * (i + 0.5) * tr; a.box(c, d, w, tr + 0.01, zg, z0 + (i + 1) * r_, col)
    if cheek > 0:
        cc = col_cheek or shade(col, 0.82); t_ = 0.12 * w + 0.2
        for s_ in (-1, 1):
            q0 = p0 + nn * s_ * (w / 2 + t_ / 2); q1 = p1 + nn * s_ * (w / 2 + t_ / 2)
            A_ = [(*(q0 + nn * s_ * t_ / 2 * k_), zg) for k_ in (-1, 1)]
            for k_ in (-1, 1):
                q0k = q0 + nn * k_ * t_ / 2; q1k = q1 + nn * k_ * t_ / 2
                a.face([(*q0k, zg), (*q1k, zg), (*q1k, z1 + cheek), (*q0k, z0 + cheek)], cc)
            a.face([(*(q0 - nn * t_ / 2), z0 + cheek), (*(q1 - nn * t_ / 2), z1 + cheek), (*(q1 + nn * t_ / 2), z1 + cheek), (*(q0 + nn * t_ / 2), z0 + cheek)], cc)


def spiral(a, c, r_in, r_out, z0, z1, a0, turns, col, parapet=1.0, n_per=48):
    """Spiral ramp (Samarra, Tower of Babel): a strip between the radii r_in(t) and r_out(t), rising from z0 to z1
    over turns revolutions from angle a0 (counter-clockwise), a parapet along the outer edge. r_in, r_out are
    numbers or pairs (bottom, top)."""
    c = np.asarray(c, float); ri = r_in if isinstance(r_in, (list, tuple)) else (r_in, r_in); ro = r_out if isinstance(r_out, (list, tuple)) else (r_out, r_out)
    N = max(8, int(n_per * turns)); A_ = []; B_ = []; Pp = []
    for i in range(N + 1):
        t = i / N; ang = a0 + 2 * math.pi * turns * t; u = np.array([math.cos(ang), math.sin(ang)]); z = z0 + (z1 - z0) * t
        r1 = ri[0] + (ri[1] - ri[0]) * t; r2 = ro[0] + (ro[1] - ro[0]) * t
        A_.append((*(c + u * r1), z)); B_.append((*(c + u * r2), z)); Pp.append((*(c + u * r2), z + parapet))
    a.quad_strip(A_, B_, col); a.quad_strip(B_, Pp, shade(col, 0.9))
    a.quad_strip([(x, y, z - 0.6) for x, y, z in B_], B_, shade(col, 0.85))


def redented(c, half, notch=0.18, ax=(1.0, 0.0)):
    """Square with cut-out corners (cruciform — the plan of a prang)."""
    c = np.asarray(c, float); ax = np.asarray(ax, float); ax = ax / max(np.linalg.norm(ax), 1e-9); pv = np.array([-ax[1], ax[0]]); m = half * (1 - notch)
    pts = [(half, -m), (half, m), (m, m), (m, half), (-m, half), (-m, m), (-half, m), (-half, -m), (-m, -m), (-m, -half), (m, -half), (m, -m)]
    return [c + ax * x + pv * y for x, y in pts]


def prang(a, ar, c, half, z0, h, cols, tiers=6, ax=(1.0, 0.0)):
    """Prang (Angkor): a body cruciform in plan, tiers with cornices narrowing along a convex curve (the
    "corncob"), a lotus bud."""
    zb = z0 + 0.32 * h; a.prism(redented(c, half, ax=ax), z0, zb, cols["stone"])
    zc = zb
    for i in range(tiers):
        t0, t1 = i / tiers, (i + 1) / tiers; f0 = 1 - t0 ** 1.5 * 0.82; f1 = 1 - t1 ** 1.5 * 0.82; hz = (h * 0.6) / tiers
        a.prism(redented(c, half * f0 * 1.04, ax=ax), zc, zc + hz * 0.25, cols["dark"])
        a.prism(redented(c, half * f0 * 0.96, ax=ax), zc + hz * 0.25, zc + hz, cols["stone"], scale_top=f1 / f0)
        zc += hz
    lathe(ar, c, dome_profile("onion", half * 0.2, h - (zc - z0)), zc, cols["accent"], n=12)


def minaret(a, ar, c, r, z0, h, cols, balconies=(0.55, 0.8)):
    """Minaret (an order temple — like the minarets of Hagia Sophia): a faceted shaft with a slight taper, balconies, a
    needle."""
    lathe(a, c, [(r, 0.0), (r * 0.82, 0.86 * h)], z0, cols["stone"], n=16)
    for f in balconies:
        zz = z0 + f * h; rr = r * (1 - 0.18 * f)
        lathe(a, c, [(rr * 1.0, -0.25 * r), (rr * 1.45, 0.0), (rr * 1.45, 0.45 * r)], zz, cols["dark"], n=16)
    roof_fan(ar, (*c, z0 + h), octagon(c, r * 0.8, n=16) + [octagon(c, r * 0.8, n=16)[0]], z0 + 0.86 * h, cols["accent"])


def obelisk(a, ar, c, base, z0, h, col, col_tip=None, ax=(1.0, 0.0)):
    """Obelisk: a square shaft tapering to 0.65 of the base, a pyramidion (Karnak; dark order; occult circle)."""
    c = np.asarray(c, float); ax = np.asarray(ax, float); ax = ax / max(np.linalg.norm(ax), 1e-9); pv = np.array([-ax[1], ax[0]])
    def sq(s_): return [c + (ax * x + pv * y) * s_ / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    a.prism(sq(base * 1.4), z0, z0 + 0.06 * h, shade(col, 0.85))
    a.prism(sq(base), z0 + 0.06 * h, z0 + 0.92 * h, col, top=False, scale_top=0.65)
    roof_fan(ar, (*c, z0 + h), sq(base * 0.65) + [sq(base * 0.65)[0]], z0 + 0.92 * h, col_tip or col)


def column(a, c, r, z0, z1, col, kind="round", col_cap=None):
    """Column: round, square, papyrus (flared capital, Karnak), twisted (darkness)."""
    c = np.asarray(c, float); h = z1 - z0; cc = col_cap or shade(col, 0.9)
    if kind == "square":
        a.box(c, (1.0, 0.0), 2 * r, 2 * r, z0, z1 - 0.08 * h, col); a.box(c, (1.0, 0.0), 2.4 * r, 2.4 * r, z1 - 0.08 * h, z1, cc); return
    if kind == "papyrus":
        lathe(a, c, [(r * 1.15, 0.0), (r * 1.0, 0.04 * h), (r * 0.92, 0.8 * h)], z0, col, n=16)
        lathe(a, c, [(r * 0.92, 0.0), (r * 1.6, 0.13 * h), (r * 1.25, 0.17 * h), (r * 1.25, 0.2 * h)], z0 + 0.8 * h, cc, n=16); return
    if kind == "twisted":
        pts = [(*(c + 0.35 * r * np.array([math.cos(6 * f), math.sin(6 * f)])), z0 + f * h) for f in np.linspace(0, 1, 13)]
        tube(a, pts, r * 0.8, col, n=8); return
    lathe(a, c, [(r * 1.2, 0.0), (r, 0.05 * h), (r * 0.9, 0.92 * h), (r * 1.25, 0.96 * h), (r * 1.25, h)], z0, col, n=16)


def gallery(a, ar, P, z0, w, h, cols, post=3.0, kind="square", arch="flat", doors=()):
    """Enclosure gallery (Angkor): along the outline P (convex, counter-clockwise) — a blind outer wall with window
    openings, inside it a row of posts at spacing w, a gallery roof sloping inward; a floor; doors — [(point,
    width)] passages under the gopuras."""
    P = [np.asarray(p, float) for p in P]; Pi = offset_poly(P, w); n = len(P)
    for i in range(n):
        p, q = P[i], P[(i + 1) % n]; L_ = float(np.linalg.norm(q - p)); k = max(1, int(L_ / (post * 2.2))); holes = []
        dd = (q - p) / max(L_, 1e-9); dr = [(float((np.asarray(g) - p) @ dd), gw) for g, gw in doors if seg_d(g, p, q) < 1.5]
        for j in range(k):
            s_ = (j + 0.5) * L_ / k; ww = min(0.35 * L_ / k, 0.3 * h)
            if not any(abs(s_ - sd) < (ww + gw) / 2 + 0.4 for sd, gw in dr): holes.append((s_, z0 + 0.3 * h, ww, 0.4 * h, arch if arch != "flat" else "rect"))
        for sd, gw in dr: holes.append((sd, z0, gw, min(0.75 * h, max(2.6, 0.6 * h)), arch if arch != "flat" else "rect"))
        wall_panel(a, p, q, z0 - 0.3, z0 + h, 0.6 + 0.04 * w, holes, cols["stone"])
        pi_, qi_ = Pi[i], Pi[(i + 1) % n]; Li = float(np.linalg.norm(qi_ - pi_)); m = max(2, int(Li / post))
        for j in range(m + 1): column(a, pi_ + (qi_ - pi_) * j / m, 0.11 * w + 0.15, z0, z0 + 0.82 * h, cols["stone"], kind)
        ar.face([(*p, z0 + h + 0.3), (*q, z0 + h + 0.3), (*qi_, z0 + 0.82 * h), (*pi_, z0 + 0.82 * h)], cols["roof"])
        a.face([(*pi_, z0 + 0.82 * h - 0.01), (*qi_, z0 + 0.82 * h - 0.01), (*q, z0 + h * 0.98), (*p, z0 + h * 0.98)], cols["dark"])


def pylon(a, ar, c, ax, w, h, d, z0, cols, gate_w=None, cavetto=True, blind=True):
    """Pylon (Egypt): two trapezoidal towers with a batter on both sides of the gate, the entrance lower between
    them, a cavetto cornice; blind — the passage is closed by a dark face (a false gate), otherwise it is
    through (per the space scheme)."""
    c = np.asarray(c, float); ax = np.asarray(ax, float); ax = ax / max(np.linalg.norm(ax), 1e-9); pv = np.array([-ax[1], ax[0]])
    gw = gate_w or 0.18 * w; tw = (w - gw) / 2
    for s_ in (-1, 1):
        cc = c + pv * s_ * (gw / 2 + tw / 2)
        P = [cc + ax * x * d / 2 + pv * y * tw / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        Pt = offset_poly(P, 0.07 * h); a.quad_strip([(*p, z0) for p in P + [P[0]]], [(*p, z0 + h) for p in Pt + [Pt[0]]], cols["stone"])
        if cavetto:
            Pc = offset_poly(Pt, -0.04 * h); a.quad_strip([(*p, z0 + 0.94 * h) for p in Pt + [Pt[0]]], [(*p, z0 + h) for p in Pc + [Pc[0]]], cols["dark"])
            a.face([(*p, z0 + h) for p in Pc], cols["dark"])
        else: a.face([(*p, z0 + h) for p in Pt], cols["dark"])
    gh = 0.62 * h; G = [c + ax * x * d * 0.4 + pv * y * gw / 2 for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    a.quad_strip([(*p, z0 + gh * 0.75) for p in G + [G[0]]], [(*p, z0 + gh) for p in G + [G[0]]], cols["stone"]); a.face([(*p, z0 + gh) for p in G], cols["dark"])
    a.face([(*p, z0 + gh * 0.75) for p in G[::-1]], cols["stone"])                 # the lower face of the lintel is the ceiling of the passage
    if blind: a.face([(*(c - ax * d * 0.41 + pv * y * gw / 2), z0 + z) for y, z in ((-1, 0.0), (1, 0.0), (1, gh * 0.75), (-1, gh * 0.75))], cols["hole"])


def stupa(a, ar, c, r, z0, cols):
    """Stupa (Borobudur): a low drum, a bell, a square harmika, a chhatra spire."""
    lathe(a, c, [(r * 1.1, 0.0), (r * 1.1, 0.15 * r), (r * 0.95, 0.25 * r)], z0, cols["stone"], n=16)
    lathe(a, c, [(r * 0.95 * math.sqrt(max(0.0, 1 - f * f)) + 0.02, 0.25 * r + f * 0.95 * r) for f in np.linspace(0, 1, 8)], z0, cols["stone"], n=16)
    a.box(c, (1.0, 0.0), 0.32 * r, 0.32 * r, z0 + 1.15 * r, z0 + 1.4 * r, cols["stone"])
    roof_fan(ar, (*c, z0 + 2.2 * r), octagon(c, 0.1 * r) + [octagon(c, 0.1 * r)[0]], z0 + 1.4 * r, cols["stone"])


def menhir(a, c, w, h, z0, col, seed=0.0):
    """Menhir: an uneven stone, tapering upward, slightly tilted."""
    c = np.asarray(c, float); rnd = lambda i: (math.sin(seed * 9.17 + i * 12.9898) * 43758.5453) % 1.0
    P0 = [c + np.array([math.cos(2 * math.pi * i / 6), math.sin(2 * math.pi * i / 6)]) * w * (0.4 + 0.25 * rnd(i)) for i in range(6)]
    lean = np.array([rnd(7) - 0.5, rnd(8) - 0.5]) * 0.3 * w
    P1 = [c + lean + (p - c) * (0.45 + 0.2 * rnd(i + 10)) for i, p in enumerate(P0)]
    a.quad_strip([(*p, z0 - 0.3) for p in P0 + [P0[0]]], [(*p, z0 + h * (0.92 + 0.08 * rnd(20))) for p in P1 + [P1[0]]], col)
    a.face([(*p, z0 + h * (0.92 + 0.08 * rnd(20))) for p in P1], col)


def altar(a, c, s, z0, col, col_top=None, steps=3):
    """Altar on a stepped platform."""
    c = np.asarray(c, float)
    for i in range(steps):
        e = s * (1.9 - 0.3 * i); a.box(c, (1.0, 0.0), e, e, z0 + i * 0.3 * s / steps * 2, z0 + (i + 1) * 0.3 * s / steps * 2, col)
    zt = z0 + 0.6 * s; a.box(c, (1.0, 0.0), 0.55 * s, 1.0 * s, zt, zt + 0.45 * s, col_top or col)


def staged(at, ar, c, ax, stages, spire, cols, z_off=0.0, s=1.0, pinnacles=False, buttresses=False, fort=False, seed=0.0, open_k="pointed"):
    """Tower by tiers in world coordinates (the plan is temple_plan.tower_plan): prism tiers with openings, bands,
    corner buttresses, pinnacles at the transition to the octagon, the top is crown. Tier heights start from
    z_off."""
    c = np.asarray(c, float); ax = np.asarray(ax, float); ax = ax / max(np.linalg.norm(ax), 1e-9); pv = np.array([-ax[1], ax[0]])
    def foot_(half, plan):
        if plan == "square": return [c + ax * x * half + pv * y * half for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        if plan == "round": return octagon(c, half, phase=0.0, n=16)
        return octagon(c, half, phase=math.atan2(ax[1], ax[0]) + math.pi / 8)
    for k, sg in enumerate(stages):
        P = foot_(sg["half"], sg["plan"]); n = len(P); z0 = z_off + sg["z0"]; z1 = z_off + sg["z1"]
        a = at
        a.quad_strip([(*p, z0) for p in P + [P[0]]], [(*p, z1) for p in P + [P[0]]], cols["stone"])
        if sg.get("kind") in ("belfry", "lantern"):
            hz = sg["z1"] - sg["z0"]
            for i in range(n):
                p, q = P[i], P[(i + 1) % n]; ln = float(np.linalg.norm(q - p)); nn = (p + q) / 2 - c; nn /= max(np.linalg.norm(nn), 1e-9)
                if sg["plan"] == "round" and i % 2: continue
                cnt = 1 if sg["plan"] != "square" else 2
                for j in range(cnt):
                    x = (j + 0.5) / cnt; wv = min(0.7 * ln / cnt, 0.22 * 2 * sg["half"]); cw = p + (q - p) * x + nn * 0.05; tg = np.array([-nn[1], nn[0]])
                    a.face([(*(cw + tg * xx), z0 + 0.18 * hz + zz) for xx, zz in hole_outline(0.0, 0.0, wv, 0.6 * hz if open_k != "round" else 0.4 * hz, open_k, 6)], cols["hole"])
        Pb = [c + (p - c) * (1 + 0.5 * s / max(np.linalg.norm(p - c), 1e-6)) for p in P]
        a.quad_strip([(*p, z1 - 0.45 * s) for p in Pb + [Pb[0]]], [(*p, z1) for p in Pb + [Pb[0]]], cols["dark"]); a.face([(*p, z1) for p in Pb], cols["dark"])
        if buttresses and sg["plan"] == "square" and k <= 1:
            bw_ = 0.12 * 2 * sg["half"]; bd_ = 0.16 * 2 * sg["half"]; zt = z1 - 2.5 * s
            for p in P:
                n1 = ax * (1.0 if (p - c) @ ax > 0 else -1.0); n2 = pv * (1.0 if (p - c) @ pv > 0 else -1.0)
                for dd, other in ((n1, n2), (n2, n1)):
                    a.box(p + dd * bd_ / 2 - other * bw_ / 2, dd, bw_, bd_, z0, z0 + (zt - z0) * 0.55, cols["stone"])
                    a.box(p + dd * bd_ * 0.35 - other * bw_ / 2, dd, bw_, bd_ * 0.7, z0 + (zt - z0) * 0.55, zt, cols["stone"])
                    if pinnacles: pinnacle(a, p + dd * bd_ * 0.35 - other * bw_ / 2, zt, bw_ * 0.8, 1.2 * bw_, cols["stone"], cols["roof"])
        if k < len(stages) - 1:
            nx = stages[k + 1]
            if nx["plan"] == "poly8" and sg["plan"] == "square" and pinnacles:
                for p in P: pinnacle(a, c + (p - c) * 0.86, z1, 0.16 * 2 * sg["half"], 0.45 * (nx["z1"] - nx["z0"]), cols["stone"], cols["roof"])
            elif sg["half"] - nx["half"] > 0.3 * s:
                a.quad_strip([(*p, z1) for p in P + [P[0]]], [(*p, z1 + 1.1 * s) for p in P + [P[0]]], cols["stone"])
    top = stages[-1]; P = foot_(top["half"], top["plan"])
    if spire["kind"] == "prang": prang(at, ar, c, top["half"], z_off + top["z1"], spire["h"], cols, ax=ax)
    elif spire["kind"] == "minaret": minaret(at, ar, c, top["half"] * 0.7, z_off + top["z1"], spire["h"], cols)
    else: crown(at, ar, c, P, top["half"], top["plan"], z_off + top["z1"], spire["kind"], spire["h"], cols, pinnacles=pinnacles, fort=fort, seed=seed, s=s)


# ---------------------------------------------------------------- ground following the site relief
class Relief:
    """Site relief (relief.npz): the near plot and the belt, height joined with a 12 m seam (as Ground does)."""

    def __init__(self, path):
        Z = np.load(path); self.HN = Z["near"].astype(np.float64); self.nx0, self.ny0, self.nres = (float(v) for v in Z["near_o"])
        self.HF = Z["far"].astype(np.float64); self.fx0, self.fy0, self.fres = (float(v) for v in Z["far_o"]); self.LEVEL = Z["level"].astype(np.float64)
        self.NB = (self.nx0, self.ny0, self.nx0 + (self.HN.shape[1] - 1) * self.nres, self.ny0 + (self.HN.shape[0] - 1) * self.nres)

    @staticmethod
    def _bil(A, x0, y0, res, x, y):
        gx = np.clip((np.asarray(x) - x0) / res, 0, A.shape[1] - 1.001); gy = np.clip((np.asarray(y) - y0) / res, 0, A.shape[0] - 1.001)
        ix, iy = gx.astype(int), gy.astype(int); fx, fy = gx - ix, gy - iy
        return A[iy, ix] * (1 - fx) * (1 - fy) + A[iy, ix + 1] * fx * (1 - fy) + A[iy + 1, ix] * (1 - fx) * fy + A[iy + 1, ix + 1] * fx * fy

    def height(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float); NB = self.NB
        zf = self._bil(self.HF, self.fx0, self.fy0, self.fres, x, y); zn = self._bil(self.HN, self.nx0, self.ny0, self.nres, x, y)
        d = np.minimum(np.minimum(x - NB[0], NB[2] - x), np.minimum(y - NB[1], NB[3] - y)); w = np.clip(d / 12.0, 0, 1); w = w * w * (3 - 2 * w)
        return np.where(d > 0, zn * w + zf * (1 - w), zf)


def ground(R, centre, half, step, hole=None, name="ground"):
    """Ground: a mesh over the relief nodes (the near one); the far ring with a hole lies under the near one (on
    cliffs the coarse mesh does not poke out)."""
    if hole is None:
        x0 = R.nx0 + math.ceil((centre[0] - half - R.nx0) / step) * step; y0 = R.ny0 + math.ceil((centre[1] - half - R.ny0) / step) * step
        xs = np.arange(x0, centre[0] + half + 1e-6, step); ys = np.arange(y0, centre[1] + half + 1e-6, step)
    else:
        xs = np.arange(centre[0] - half, centre[0] + half + 1e-6, step); ys = np.arange(centre[1] - half, centre[1] + half + 1e-6, step)
    X, Y = np.meshgrid(xs, ys); Zg = R.height(X, Y); ny, nx = X.shape
    if hole is not None: Zg = np.where((np.abs(X - centre[0]) < hole + 4.5) & (np.abs(Y - centre[1]) < hole + 4.5), Zg - 3.0, Zg)
    gy, gx = np.gradient(Zg, step); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    grass = np.array([0.36, 0.42, 0.25]); dirt = np.array([0.45, 0.40, 0.30]); rock = np.array([0.50, 0.48, 0.45])
    tt = np.clip((sl - 20) / 20, 0, 1)[..., None]; colv = grass * (1 - tt) + rock * tt
    colv = np.where((sl < 3)[..., None], colv * 0.6 + dirt * 0.4, colv)
    me = bpy.data.meshes.new(name); V = np.column_stack([X.ravel(), Y.ravel(), Zg.ravel()])
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]; F = np.stack([i, i + 1, i + nx + 1, i + nx], -1).reshape(-1, 4)
    if hole is not None:
        cx = (X[:-1, :-1] + X[1:, 1:]) / 2; cy = (Y[:-1, :-1] + Y[1:, 1:]) / 2
        F = F[((np.abs(cx - centre[0]) > hole) | (np.abs(cy - centre[1]) > hole)).ravel()]
    me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 4)); me.update()
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    ca.data.foreach_set("color", np.column_stack([colv.reshape(-1, 3), np.ones(nx * ny)]).astype(np.float32).ravel())
    me.color_attributes.active_color = ca
    ob = bpy.data.objects.new(name, me); coll("terrain").objects.link(ob); return ob
