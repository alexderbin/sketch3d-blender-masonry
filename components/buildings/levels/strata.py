"""Levels over time — the model and the rules. The project's Python only.

The World is the model block, the ground of the eras (layers g_k(x, y): nature, fills), decks (slabs, bridges, tiers),
streets and terraces (walkable surfaces), buildings (outline, storeys, fate), voids (cellars, tunnels, vaults under a
street), other bodies (retaining walls, supports, stairs). One rule for all cases — `relate`: a building's storey against
all the grounds around it:
  * an opening per span of a side — a window if the ground outside is lower than the sill; blank if higher (the storey
    is "in the ground" on this side);
  * an entrance — where a walkable surface (ground, street, deck) meets the storey at its elevation; across a gap of up
    to 8 m — a bridge;
  * a cellar into the rock — where a side of the storey is entirely in the ground and there is enough cover above the
    tunnel.
Time — `epochs`: a fill raises the ground, the old goes down; a building's fate — raise the entrance, cut down to the new
ground (the lower part becomes a basement or the foundation of the new building) or bury; the new stands on the old
walls (repeating the outline) or across them."""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box as sbox

from components.buildings.levels import geom as G

SIDES = (("s", (0.0, -1.0)), ("e", (1.0, 0.0)), ("n", (0.0, 1.0)), ("w", (-1.0, 0.0)))


class World:
    def __init__(self, block, layers, tones=None):
        self.block = block; self.layers = list(layers); self.tones = tones
        self.decks = []; self.streets = []; self.buildings = []; self.voids = []; self.solids = []; self.notes = []
        self.digs = []                                                           # excavations: (polygon, bottom elevation[, era]) — they cut all layers
        self.layer_cls = None; self.layer_tints = None; self.top_cls = None      # the class and tint of a layer in the section, the class of a cell's top
        self.links = []                                                          # walkability links: a stair, a ramp, a lift, a gate (walk.py)
        self.dark = None                                                         # the depth shade field: f(x, y, z) → 0–1

    # ------------------------------------------------------------ grounds
    def layer(self, k):
        """Layer k with excavations: inside an excavation the ground is no higher than its bottom (all eras — a cut also removes the
        old layers)."""
        g = self.layers[k]
        if not self.digs: return g
        digs = list(self.digs)

        def f(X, Y, g=g):
            X = np.asarray(X, float); Y = np.asarray(Y, float); z = np.broadcast_to(np.asarray(g(X, Y), float), np.broadcast(X, Y).shape).copy()
            for dg in digs:
                m = shapely.contains_xy(dg[0], X, Y); z[m] = np.minimum(z[m], dg[1])
            return z
        return f

    def ground(self, x, y, k=-1):
        return float(np.asarray(self.layer(k)(np.asarray([x], float), np.asarray([y], float))).ravel()[0])

    def walk_at(self, x, y, z, tol=0.45):
        """Walkable surfaces at a point at elevation z ± tol: ground, street, deck, a flat terrace roof → [(kind, z)]."""
        out = []; g = self.ground(x, y)
        if abs(g - z) <= tol: out.append(("ground", g))
        p = Point(x, y)
        for s in self.streets:
            if s["poly"].contains(p) and abs(s["z"] - z) <= tol: out.append((s.get("kind", "street"), s["z"]))
        for d in self.decks:
            if d["poly"].contains(p) and abs(d["z"] - z) <= tol: out.append(("deck", d["z"]))
        for b in self.buildings:
            if b.get("roof") in ("terrace", "street") and b["poly"].contains(p) and abs(b["z_top"] - z) <= tol: out.append(("roof", b["z_top"]))
        return out

    def slab_at(self, x, y, z0, z1):
        """A deck (slab) crosses the interval [z0, z1] at a point — the window is blocked by the slab."""
        p = Point(x, y)
        return any(d["poly"].contains(p) and d["z"] - d["t"] < z1 and d["z"] > z0 for d in self.decks)

    def solid_at(self, x, y, z0, z1, me=None):
        """Another building occupies a point at elevations [z0, z1] — a shared wall (blank). Neighbors are by extent (self._near)."""
        cand = self._near if getattr(self, "_near", None) is not None else self.buildings
        def inside(b):
            if not b.get("rot"): return b["x0"] - 0.05 <= x <= b["x1"] + 0.05 and b["y0"] - 0.05 <= y <= b["y1"] + 0.05
            bx = b["poly"].bounds
            return bx[0] - 0.05 <= x <= bx[2] + 0.05 and bx[1] - 0.05 <= y <= bx[3] + 0.05 and b["poly"].buffer(0.05).contains(Point(x, y))
        return any(b is not me and b["z_found"] < z1 and (b["z_cut"] if b.get("z_cut") is not None else b["z_top"]) > z0 and inside(b)
                   for b in cand if b.get("state") != "gone")


# ---------------------------------------------------------------- building
def building(x0, x1, y0, y1, z0, floors, fh=3.2, t=0.6, s=0.3, cellars=0, ch=3.0, roof="flat", epoch=0, kind="house",
             bay=3.0, caves=0, **kw):
    """A box building: storeys from z0 upward (floors × fh), cellars downward (cellars × ch, vaults); roof: flat | terrace | gable."""
    F = [{"z": z0 + i * fh, "h": fh, "kind": "floor"} for i in range(floors)]
    C = [{"z": z0 - (i + 1) * ch, "h": ch, "kind": "cellar"} for i in range(cellars)]
    b = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "poly": sbox(x0, y0, x1, y1), "floors": C[::-1] + F, "fh": fh, "t": t, "s": s,
         "roof": roof, "epoch": epoch, "kind": kind, "bay": bay, "caves": caves, "state": "whole",
         "z0": z0, "z_found": z0 - cellars * ch - 0.6, "z_top": z0 + floors * fh, "z_cut": None}
    ang = kw.pop("angle", 0.0)
    b.update(kw)
    if ang:                                                                    # a rotated building: x0…y1 are its own axes, the outline is in the world
        from shapely import affinity
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; b["rot"] = (cx, cy, float(ang))
        b["poly"] = affinity.rotate(b["poly"], ang, origin=(cx, cy), use_radians=True)
    return b


def floor_fit(z_low, z_up, lo=3.0, hi=3.7, tol=0.15, k_max=14):
    """Storeys between the lower ground and the upper walkable surface (a bridge, a street on the ridge): a floor both at the
    ground and at the elevation z_up.
    → (z0, k, fh): k storeys of height fh from z0 to z_up; z0 is not higher than z_low + tol (the house does not hang), sunk if
    needed (the bottom of the storey is in the ground on the slope side). k = 0 — the level change is less than a storey: the
    house stands at the upper elevation."""
    if z_up - z_low < lo * 0.5: return z_up + tol, 0, lo
    for k in range(1, k_max + 1):
        if z_up - k * hi > z_low + tol: continue
        z0 = min(z_up - k * lo, z_low + tol)
        return z0, k, (z_up - z0) / k
    return z_low + tol, 0, lo


def to_world(b, p):
    """A point in the building's own axes → the world (for an unrotated one — the same)."""
    r = b.get("rot")
    if not r: return (float(p[0]), float(p[1]))
    cx, cy, a = r; c, s_ = math.cos(a), math.sin(a); dx, dy = p[0] - cx, p[1] - cy
    return (cx + c * dx - s_ * dy, cy + s_ * dx + c * dy)


def side_line(b, side, local=False):
    """A side: start, end, outward normal (clockwise from the south: s — y0, e — x1, n — y1, w — x0); for a rotated building —
    in the world (local=True — in its own axes, for construction)."""
    x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
    a, c, n = {"s": ((x0, y0), (x1, y0), (0, -1)), "e": ((x1, y0), (x1, y1), (1, 0)), "n": ((x1, y1), (x0, y1), (0, 1)),
               "w": ((x0, y1), (x0, y0), (-1, 0))}[side]
    if local or not b.get("rot"): return a, c, n
    ang = b["rot"][2]; cs, sn = math.cos(ang), math.sin(ang)
    return to_world(b, a), to_world(b, c), (cs * n[0] - sn * n[1], sn * n[0] + cs * n[1])


def bays(L, bay, margin=0.9):
    n = max(1, int((L - 2 * margin) // bay)); s = (L - 2 * margin) / n
    return [margin + s * (i + 0.5) for i in range(n)]


def relate(W, b, sill=0.9, win=1.5, door=2.3, reach=8.0, stoop_max=1.3):
    """A building's storeys against the grounds: span openings, entrances, bridges, blank sides.
    → b["open"][storey][side] = [(u, kind)]."""
    b["open"] = []; b["doors"] = []; b["bridges"] = []
    top = b["z_cut"] if b["z_cut"] is not None else b["z_top"]
    r = reach + 2.0                                                              # neighbors by extent — for solid_at
    bb = b["poly"].bounds
    W._near = [o for o in W.buildings if (lambda ob: ob[2] > bb[0] - r and ob[0] < bb[2] + r and ob[3] > bb[1] - r and ob[1] < bb[3] + r)(o["poly"].bounds)]
    for fi, f in enumerate(b["floors"]):
        z = f["z"]; row = {}
        if z >= top - 0.5: b["open"].append(row); continue
        for side, _ in SIDES:
            a, c, n = side_line(b, side); a = np.asarray(a, float); c = np.asarray(c, float); n = np.asarray(n, float)
            L = float(np.linalg.norm(c - a)); d = (c - a) / L; cells = []
            for u in bays(L, b["bay"]):
                p = a + d * u + n * (b["t"] + 0.5)
                g = W.ground(*p)
                if f["kind"] == "cellar" or g > z + sill or W.solid_at(*p, z + sill, z + sill + win, me=b) \
                        or W.slab_at(*p, z + sill, z + sill + win):
                    kind = "blind"
                else:
                    kind = "window"
                walk = W.walk_at(*p, z) if f["kind"] == "floor" and not W.solid_at(*p, z + 0.3, z + 2.0, me=b) else []   # not into a neighbor
                stoop = f["kind"] == "floor" and not walk and z - stoop_max <= g < z - 0.45 and not W.solid_at(*p, g, z, me=b)
                cells.append([float(u), kind, bool(walk), float(z - g) if stoop else 0.0])
            # entrance: a span with a walkable surface at the storey's elevation — the one nearest the middle of the side;
            # otherwise a porch (the ground below the threshold up to stoop_max: steps outward)
            cand = [c_ for c_ in cells if c_[2]] or [c_ for c_ in cells if c_[3] > 0]
            if cand:
                u = min(cand, key=lambda c_: abs(c_[0] - L / 2)); u[1] = "door"
                b["doors"].append({"floor": fi, "side": side, "u": u[0], **({"stoop": u[3]} if not u[2] else {})})
            elif f["kind"] == "floor" and abs(b.get("rot", (0, 0, 0.0))[2]) < 0.3 and z > W.ground(*(a + d * L / 2 + n * (b["t"] + 0.5))) + 1.0:
                br = _bridge(W, b, a + d * L / 2, n, z, reach)                 # a walkable surface across a gap — a bridge
                if br:
                    mid = min(cells, key=lambda c_: abs(c_[0] - L / 2)); mid[1] = "door"
                    b["doors"].append({"floor": fi, "side": side, "u": mid[0], "bridge": True}); b["bridges"].append(br)
            row[side] = [(c_[0], c_[1]) for c_ in cells]
        b["open"].append(row)
    W._near = None
    return b


def _bridge(W, b, p0, n, z, reach):
    """A ray from the middle of a side along the normal: the first walkable surface at elevation z (a street, a deck, a terrace
    roof) farther than 0.8 m and nearer than reach, with neither ground nor a building at that elevation on the way →
    a bridge {from, to, z}."""
    for dist in np.arange(b["t"] + 0.8, reach, 0.25):
        p = p0 + n * dist
        if W.ground(*p) > z + 0.5 or W.solid_at(*p, z, z + 2.0, me=b): return None
        w = W.walk_at(*p, z)
        if w and (any(k != "ground" for k, _ in w) or _flat(W, p)):          # ground only if gentle (no bridge onto a sheer cliff)
            return {"a": (p0 + n * b["t"]).tolist(), "b": p.tolist(), "z": float(z)}
    return None


def _flat(W, p, r=0.75, grade=0.6):
    gx = (W.ground(p[0] + r, p[1]) - W.ground(p[0] - r, p[1])) / (2 * r); gy = (W.ground(p[0], p[1] + r) - W.ground(p[0], p[1] - r)) / (2 * r)
    return max(abs(gx), abs(gy)) <= grade


def caves(W, b, n_max, length=(6.0, 12.0), width=3.0, cover=2.0, rng=None):
    """Cellars into the rock: a side of the storey entirely in the ground (above the vault by cover) → a tunnel along the
    normal, the side with the greatest cover (into the hill); the length lasts while there is enough cover above the vault.
    Bottom to top over the storeys, no more than n_max. → W.voids."""
    made = 0
    if b.get("rot"): return 0                                                  # a tunnel along the world axes — a rotated building has none
    for fi, f in enumerate(b["floors"]):
        if made >= n_max: break
        z = f["z"]; zs = z + 2.0; ztop = zs + width / 2; best = None
        for side, _ in SIDES:
            a, c, n = side_line(b, side); a = np.asarray(a, float); c = np.asarray(c, float); n = np.asarray(n, float)
            L = float(np.linalg.norm(c - a)); d = (c - a) / L; m = a + d * L / 2
            Lc = 0.0; covers = []
            for dist in np.arange(0.5, length[1], 0.5):
                q = m + n * (b["t"] + dist); g = W.ground(*q)
                if g < ztop + cover or W.solid_at(*q, z, ztop): break
                Lc = dist; covers.append(g - ztop)
            if Lc >= length[0] and (best is None or np.mean(covers) > best[0]): best = (float(np.mean(covers)), side, m, n, Lc, L)
        if best is None: continue
        _, side, m, n, Lc, L = best
        axis = "x" if abs(n[0]) > 0.5 else "y"; k = 0 if axis == "x" else 1
        a0, a1 = sorted((m[k] + n[k] * b["t"] * 0.5, m[k] + n[k] * (b["t"] + Lc))); v0, v1 = m[1 - k] - width / 2, m[1 - k] + width / 2
        W.voids.append({"kind": "cave", "axis": axis, "a0": float(a0), "a1": float(a1), "v0": float(v0), "v1": float(v1),
                        "z0": z, "zs": zs, "arch": "round", "of": b["id"], "floor": fi})
        row = b["open"][fi].get(side)
        if row:
            u = min(row, key=lambda c_: abs(c_[0] - L / 2)); row[row.index(u)] = (u[0], "arch")
        made += 1
    return made


# ---------------------------------------------------------------- time
def fate(W, b, z_new, rng, keep_raise=0.3, cut=0.5):
    """The ground has risen to z_new at a building: raise the entrance (the storeys below go into the ground — the relate rule
    will make them blank), cut down to the new ground (the lower part becomes a basement or a foundation), or bury it at
    random. A temple / an important building — cut down and repeat."""
    if b["z_top"] <= z_new + 0.5 or (b["kind"] != "temple" and rng.random() > keep_raise + cut):
        b["state"] = "buried"; b["z_cut"] = float(min(b["z_top"], z_new - rng.uniform(0.2, 1.5))); return "buried"
    if b["kind"] == "temple" or rng.random() < cut / (keep_raise + cut):
        b["state"] = "cut"; b["z_cut"] = float(z_new); return "cut"
    b["state"] = "raised"; return "raised"


def check(W):
    """Checks before the model: an entrance on every intact building, no windows into the ground, supports under a deck, cover
    over a cellar."""
    bad = []
    for b in W.buildings:
        if "open" not in b or b.get("via") is not None: continue              # the link was not computed (a distant mass) or the entrance is from inside the lower one
        if b["state"] in ("whole", "raised") and not b.get("doors"): bad.append(f"house {b['id']}: no entrance on any storey")
        for fi, row in enumerate(b.get("open", [])):
            for side, cells in row.items():
                a, c, n = side_line(b, side); a = np.asarray(a, float); c = np.asarray(c, float); n = np.asarray(n, float)
                d = (c - a) / max(np.linalg.norm(c - a), 1e-9)
                for u, k in cells:
                    if k == "window" and W.ground(*(a + d * u + n * (b["t"] + 0.5))) > b["floors"][fi]["z"] + 0.95:
                        bad.append(f"house {b['id']} storey {fi}: a window into the ground")
    for dk in W.decks:
        if dk.get("span") and dk["span"] > 24.0: bad.append(f"deck {dk.get('name')}: a span of {dk['span']:.0f} m without a support")
    for v in W.voids:
        if v["kind"] == "cave":
            mids = [(v["a0"] + v["a1"]) / 2, (v["v0"] + v["v1"]) / 2]
            x, y = (mids if v["axis"] == "x" else mids[::-1])
            if W.ground(x, y) < v["zs"] + (v["v1"] - v["v0"]) / 2 + 1.0: bad.append(f"cellar of house {v['of']}: too little cover")
    return bad


# ---------------------------------------------------------------- geometry
def emit(W, lod=2):
    """World → a clay mesh (geom.Mesh): the ground by layers with the section holes, buildings (walls with openings per relate,
    slabs, cellar vaults, roofs), voids, decks, streets, bridges, other bodies. lod 1 — without openings and stairs (distant
    masses)."""
    M = G.Mesh(W.block)
    holes = []
    for b in W.buildings:
        if b["state"] == "gone": continue
        if b["state"] == "cut":                                                  # the remainder is underground: the ground above it is intact
            M.cut_box(b["x0"], b["x1"], b["y0"], b["y1"], b["z_found"] + 0.6, b["z_cut"])
        elif b["state"] != "buried":
            if b["z_found"] < W.ground((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2) + 0.5:   # a hole in the terrain — for buildings standing on the ground
                holes.append(b["poly"].buffer(-min(b["t"], 0.45)))
                if b.get("rot"): M.cut_prism(b["poly"], b["z_found"] + 0.6, 1e4)
                else: M.cut_box(b["x0"], b["x1"], b["y0"], b["y1"], b["z_found"] + 0.6, 1e4)
        else:
            for wb in _walls_boxes(b, b["z_found"] + 0.6, b["z_cut"]): M.cut_box(*wb)
    for v in W.voids:
        prof = G.arch_profile(v["v0"], v["v1"], v["z0"], v["zs"], v.get("arch", "round"))
        M.cut_profile(prof, v["axis"], v["a0"], v["a1"])
    for s in W.solids:
        if s.get("cut_ground") or s.get("void"):
            if s["type"] == "prism": M.cut_prism(s["poly"], s["z0"], s["z1"])
            else: M.cut_profile(s["prof"], s["axis"], s["a0"], s["a1"])
    deep = [v["z0"] for v in W.voids] + [b["z_found"] for b in W.buildings if b["state"] != "gone"] + \
           [s_["z0"] for s_ in W.solids if s_["type"] == "prism"]
    holes += list(getattr(W, "terrain_holes", []))                            # the ground top is drawn by someone else (the cavea in rings)
    M.terrain([W.layer(k) for k in range(len(W.layers))], step=W.__dict__.get("step", 0.5), holes=holes, tones=W.tones,
              classes=W.layer_cls, tints=W.layer_tints, top_cls=W.top_cls, z_bottom=(min(deep) - 4.0) if deep else None)
    if getattr(W, "auto_retain", False): _auto_retain(M, W, holes)
    for v in W.voids:
        M.tone = _dark(W, (v["a0"] + v["a1"]) / 2, (v["v0"] + v["v1"]) / 2, v["z0"], v["axis"]) * 0.85
        M.extrude(G.arch_profile(v["v0"], v["v1"], v["z0"], v["zs"], v.get("arch", "round")), v["axis"], v["a0"], v["a1"], "void", void=True)
    for s in W.streets:
        if s.get("draw", True):
            M.tone = s.get("tone", 1.0); M.prism(s["poly"], s["z"] - s.get("t", 0.25), s["z"], s.get("cls", "street"))
            bx = s["poly"].bounds; M.cut_box(bx[0], bx[2], bx[1], bx[3], s["z"] - s.get("t", 0.25), s["z"])   # no flicker in the section
    for d in W.decks:
        if d.get("draw") is False: continue                                    # an invisible deck: a chamber floor — for walkability only
        M.tone = d.get("tone", 1.0); M.level = d.get("level", 0); M.tint = tuple(d.get("tint", (1.0, 1.0, 1.0)))
        M.prism(d["poly"], d["z"] - d["t"], d["z"], "deck")
        for pl in d.get("pillars", []): M.prism(pl["poly"], pl["z0"], d["z"] - d["t"], "pillar")
    M.level = 0
    for s in W.solids:
        M.tone = s.get("tone", 1.0); M.level = s.get("level", 0); M.tint = tuple(s.get("tint", (1.0, 1.0, 1.0)))
        if s["type"] == "prism": M.prism(s["poly"], s["z0"], s["z1"], s["cls"], void=bool(s.get("void")))
        else: M.extrude(s["prof"], s["axis"], s["a0"], s["a1"], s["cls"], void=bool(s.get("void")))
    M.level = 0; M.tint = (1.0, 1.0, 1.0)
    for b in W.buildings:
        if b["state"] == "gone": continue
        if b.get("rot"):                                                        # rotated: built in its own axes without clipping by the block,
            i0 = len(M.V); keep = (M.X0, M.Y0, M.X1, M.Y1, M.rect)              # then the vertices are rotated around the center
            M.X0, M.Y0, M.X1, M.Y1 = -1e6, -1e6, 1e6, 1e6; M.rect = sbox(-1e6, -1e6, 1e6, 1e6)
            _emit_building(M, W, b, lod)
            M.X0, M.Y0, M.X1, M.Y1, M.rect = keep
            for k in range(i0, len(M.V)):
                v = M.V[k]; w = to_world(b, v[:2]); M.V[k] = (w[0], w[1], v[2])
        else:
            _emit_building(M, W, b, lod)
        for d in b.get("doors", []):
            if d.get("stoop"): _stoop(M, W, b, d)
    return M


def _stoop(M, W, b, d, w=1.6, rise=0.18, tread=0.32):
    """A porch: steps from the ground to the threshold in front of the door (outward along the side's normal)."""
    a, c, n = side_line(b, d["side"]); a = np.asarray(a, float); c = np.asarray(c, float); n = np.asarray(n, float)
    L = float(np.linalg.norm(c - a)); dd = (c - a) / L; p = a + dd * d["u"]; z = b["floors"][d["floor"]]["z"]
    k = max(1, int(round(d["stoop"] / rise))); M.tone = b.get("tone", 1.0)
    for i in range(k):
        q0 = p + n * (tread * (k - 1 - i)); q1 = p + n * (tread * (k - i))
        P = [q0 - dd * w / 2, q1 - dd * w / 2, q1 + dd * w / 2, q0 + dd * w / 2]
        M.prism(Polygon([tuple(v) for v in P]).buffer(0), z - d["stoop"] - 0.3, z - rise * (k - 1 - i), "stair")


def _auto_retain(M, W, holes, step=0.5, drop=1.0, t=0.55, par=0.9):
    """A retaining wall everywhere the ground drops by more than drop between neighboring grid nodes and it is not a house (a
    building outline or its hole in the terrain): the links along the drop merge; the height — from the lower ground to the
    upper one + a parapet."""
    import shapely
    from shapely.ops import unary_union
    X = np.arange(M.X0, M.X1 + 1e-6, step); Y = np.arange(M.Y0, M.Y1 + 1e-6, step); Xg, Yg = np.meshgrid(X, Y)
    Z = np.asarray(W.layer(-1)(Xg, Yg), float)
    bl = unary_union([b["poly"].buffer(0.3) for b in W.buildings if b["state"] not in ("buried", "gone")] + list(holes))
    inb = shapely.contains_xy(bl, Xg, Yg) if not bl.is_empty else np.zeros_like(Z, bool)
    zone = getattr(W, "retain_zone", None)                                    # where to put walls (otherwise — everywhere)
    if zone is not None: inb |= ~shapely.contains_xy(zone, Xg, Yg)
    for axis in (0, 1):                                                       # 0 — a drop between neighbors along x, 1 — along y
        dz = np.diff(Z, axis=1 - axis) if axis == 0 else np.diff(Z, axis=0)
        ok = (np.abs(dz) > drop)
        nb = (inb[:, :-1] | inb[:, 1:]) if axis == 0 else (inb[:-1, :] | inb[1:, :])
        ok &= ~nb
        lo = np.minimum(Z[:, :-1], Z[:, 1:]) if axis == 0 else np.minimum(Z[:-1, :], Z[1:, :])
        hi = np.maximum(Z[:, :-1], Z[:, 1:]) if axis == 0 else np.maximum(Z[:-1, :], Z[1:, :])
        rows, cols = ok.shape
        if axis == 0:                                                         # a wall along y at x between nodes i and i+1
            for i in range(cols):
                j = 0
                while j < rows:
                    if not ok[j, i]: j += 1; continue
                    j0 = j
                    while j + 1 < rows and ok[j + 1, i]: j += 1
                    xm = (X[i] + X[i + 1]) / 2
                    M.box(xm - t / 2, xm + t / 2, Y[j0] - step / 2, Y[j] + step / 2, float(lo[j0:j + 1, i].min()) - 0.3,
                          float(hi[j0:j + 1, i].max()) + par, "retain")
                    j += 1
        else:                                                                 # a wall along x at y between nodes j and j+1
            for j in range(rows):
                i = 0
                while i < cols:
                    if not ok[j, i]: i += 1; continue
                    i0 = i
                    while i + 1 < cols and ok[j, i + 1]: i += 1
                    ym = (Y[j] + Y[j + 1]) / 2
                    M.box(X[i0] - step / 2, X[i] + step / 2, ym - t / 2, ym + t / 2, float(lo[j, i0:i + 1].min()) - 0.3,
                          float(hi[j, i0:i + 1].max()) + par, "retain")
                    i += 1


def _dark(W, a, v, z, axis="x"):
    if W.dark is None: return 1.0
    x, y = (a, v) if axis == "x" else (v, a)
    return float(W.dark(x, y, z))


def _walls_boxes(b, z0, z1):
    t = b["t"]; x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
    if z1 - z0 < 0.05: return []
    return [(x0, x1, y0, y0 + t, z0, z1), (x0, x1, y1 - t, y1, z0, z1), (x0, x0 + t, y0 + t, y1 - t, z0, z1), (x1 - t, x1, y0 + t, y1 - t, z0, z1)]


def _emit_building(M, W, b, lod, sill=0.9, win=1.5, door=2.3, ww=1.1, dw=1.5):
    t, s = b["t"], b["s"]; x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
    top = b["z_cut"] if b["z_cut"] is not None else b["z_top"]
    M.level = b.get("level", 0); M.tint = tuple(b.get("tint", (1.0, 1.0, 1.0)))
    if b["state"] == "buried":
        M.tone = b.get("tone", 1.0) * 0.92
        for wb in _walls_boxes(b, b["z_found"] + 0.6, top): M.box(*wb, "ruin")
        return
    zf0 = b["floors"][0]["z"]
    M.tone = b.get("tone", 1.0) * _dark(W, (x0 + x1) / 2, (y0 + y1) / 2, zf0)
    lod = min(lod, b.get("lod", lod))
    if lod == 0:                                                                 # a distant mass: a solid box and a roof
        M.box(x0, x1, y0, y1, b["z_found"], top, "wall")
        if b["z_cut"] is None: _roof(M, b, top)
        M.level = 0; M.tint = (1.0, 1.0, 1.0); return
    M.box(x0, x1, y0, y1, b["z_found"], zf0 - s, "wall")                         # foundation
    st = _stair_strip(b) if lod >= 2 else None
    for fi, f in enumerate(b["floors"]):
        z, h = f["z"], f["h"]
        if z >= top - 0.05: break
        zt = min(z + h, top); M.tone = b.get("tone", 1.0) * _dark(W, (x0 + x1) / 2, (y0 + y1) / 2, z)
        M.level = b.get("level", 0) + f.get("level", 0)
        slab = sbox(x0 + t, y0 + t, x1 - t, y1 - t)
        if st is not None and fi > 0: slab = slab.difference(st["hole"])         # the stair opening in the floor slab
        M.prism(slab, z - s, z, "slab")                                          # the storey floor
        if st is not None and fi + 1 < len(b["floors"]) and b["floors"][fi + 1]["z"] < top - 0.05:
            _flight(M, st, z, b["floors"][fi + 1]["z"], fi % 2 == 1)
        row = b["open"][fi] if fi < len(b.get("open", [])) else {}
        for side, _ in SIDES:
            a, c, n = side_line(b, side, local=True); a = np.asarray(a, float); c = np.asarray(c, float); n = np.asarray(n, float)
            L = float(np.linalg.norm(c - a)); d = (c - a) / L
            holes = []
            if lod >= 2:
                for u, k in row.get(side, []):
                    if k == "window": holes.append((u - ww / 2, u + ww / 2, z + sill, min(z + sill + win, zt - 0.3)))
                    elif k == "door": holes.append((u - dw / 2, u + dw / 2, z, min(z + door, zt - 0.3)))
                    elif k == "arch": holes.append((u - 1.4, u + 1.4, z, min(z + 2.6, zt - 0.2)))
            # the side wall: from corner to corner along the outer outline (the corners belong to s and n), minus the openings
            u0, u1 = (0.0, L) if side in ("s", "n") else (t, L - t)
            _wall_strip(M, a, d, n, t, u0, u1, z, zt, holes)
        if f["kind"] == "cellar" and b.get("vaults", True):                    # cellar vaults (in modern times — flat)
            _vault(M, b, z, h)
    if b["z_cut"] is None:
        _roof(M, b, top)
    elif b.get("covered") or b.get("under") is not None:                      # the floor over a cut: the floor of whatever is above, or a street
        M.box(x0, x1, y0, y1, top - s, top, "slab")
    M.level = 0; M.tint = (1.0, 1.0, 1.0)


def _stair_strip(b, w=1.25, run=5.2):
    """A stair strip along the inner side of the wall farthest from the section (n): flights back and forth through the storeys,
    an opening in the slab."""
    t = b["t"]; x0, x1, y0, y1 = b["x0"] + t, b["x1"] - t, b["y0"] + t, b["y1"] - t
    if x1 - x0 < run + 1.5 or y1 - y0 < 3 * w: return None
    xa = x0 + 0.6; xb = min(xa + run, x1 - 0.6); ya, yb = y1 - w, y1
    return {"xa": xa, "xb": xb, "ya": ya, "yb": yb, "hole": sbox(xa - 0.1, ya - 0.05, xb + 0.1, yb)}


def _flight(M, st, z0, z1, back, rise=0.19):
    """A flight from z0 to z1 along the strip (forth — from xa to xb, back — from xb to xa): prism steps on a 0.25 m stringer."""
    n = max(2, int(round((z1 - z0) / rise))); dx = (st["xb"] - st["xa"]) / n; dz = (z1 - z0) / n
    for i in range(n):
        xa = st["xa"] + dx * i if not back else st["xb"] - dx * (i + 1)
        zt = z0 + dz * (i + 1); M.box(xa, xa + dx, st["ya"], st["yb"], max(z0, zt - dz - 0.25), zt, "stair")


def _wall_strip(M, a, d, n, t, u0, u1, z0, z1, holes):
    """A wall of thickness t (inward from the outer face) from u0 to u1 along a side, z0–z1, with rectangular openings
    (u, u, z, z)."""
    holes = sorted([h for h in holes if h[1] > u0 and h[0] < u1 and h[3] > h[2]], key=lambda h: h[0])

    def piece(ua, ub, za, zb):
        if ub - ua < 1e-3 or zb - za < 1e-3: return
        p0 = a + d * ua; p1 = a + d * ub; q0 = p0 - n * t; q1 = p1 - n * t
        M.prism(Polygon([tuple(p0), tuple(p1), tuple(q1), tuple(q0)]).buffer(0), za, zb, "wall")

    cur = u0
    for ha, hb, hz0, hz1 in holes:
        ha, hb = max(ha, u0), min(hb, u1)
        piece(cur, ha, z0, z1); piece(ha, hb, z0, hz0); piece(ha, hb, hz1, z1); cur = hb
    piece(cur, u1, z0, z1)


def _vault(M, b, z, h, span=4.2, pier=0.6):
    """Cellar vaults: a row of barrel vaults along one axis (span ≤ span, with supporting walls pier between them) — the mass
    between the intrados and the floor of the storey above; the axis is b["vault"] or along the long side."""
    t = b["t"]; x0, x1, y0, y1 = b["x0"] + t, b["x1"] - t, b["y0"] + t, b["y1"] - t
    axis = b.get("vault") or ("x" if (x1 - x0) > (y1 - y0) + 0.5 else "y")
    v0, v1, a0, a1 = (y0, y1, x0, x1) if axis == "x" else (x0, x1, y0, y1)
    w = v1 - v0; n = max(1, int(math.ceil((w + pier) / (span + pier)))); s_ = (w - pier * (n - 1)) / n
    zc = z + h - b["s"]; r = s_ / 2; zs = max(z + 1.2, zc - r - 0.25)
    mass = sbox(v0, z, v1, zc)
    for i in range(n):
        a = v0 + i * (s_ + pier); mass = mass.difference(G.arch_profile(a, a + s_, z - 1.0, zs, "round"))
    if not mass.is_empty: M.extrude(mass, axis, a0, a1, "wall")


def _roof(M, b, top):
    x0, x1, y0, y1, t = b["x0"], b["x1"], b["y0"], b["y1"], b["t"]
    if b["roof"] == "street":                                                # the roof is the pavement of the street above the house (no parapet)
        M.box(x0, x1, y0, y1, top - b["s"], top, "street"); return
    if b["roof"] in ("flat", "terrace"):
        M.box(x0, x1, y0, y1, top - b["s"], top, "roof")
        par = 1.0 if b["roof"] == "terrace" else 0.6
        for wb in _walls_boxes(b, top, top + par): M.box(*wb, "roof")
    else:                                                                      # gabled along the long axis, a 40° slope
        axis = b.get("ridge") or ("x" if (x1 - x0) >= (y1 - y0) else "y")
        v0, v1, a0, a1 = (y0, y1, x0, x1) if axis == "x" else (x0, x1, y0, y1)
        ang = math.radians(b.get("pitch", 40)); hh = (v1 - v0) / 2 * math.tan(ang); th = 0.35 / math.cos(ang)
        outer = Polygon([(v0 - 0.4, top - 0.3), (v1 + 0.4, top - 0.3), ((v0 + v1) / 2, top + hh)])
        inner = Polygon([(v0 + t, top), (v1 - t, top), ((v0 + v1) / 2, top + hh - th - t * math.tan(ang))])
        M.extrude(outer.difference(inner), axis, a0 - 0.4, a1 + 0.4, "roof")      # the roof is a shell with an attic under it
        for a_ in (a0, a1 - t):                                                      # gables
            M.extrude(inner, axis, a_, a_ + t, "wall")
        M.box(x0, x1, y0, y1, top - b["s"], top, "slab")


def bridge_solid(br, width=2.0, th=0.6, arch=True):
    """A bridge from a house wall to a street: a bar with an arch beneath (the profile along the bridge) → a body for W.solids."""
    (ax, ay), (bx, by) = br["a"], br["b"]; z = br["z"]
    if abs(bx - ax) >= abs(by - ay):
        u0, u1 = sorted((ax, bx)); v = ay; axis_prof = "y"                    # the profile in (x, z), extruded along y
        prof = sbox(u0, z - th - (1.2 if arch else 0), u1, z)
        if arch and u1 - u0 > 2.0: prof = prof.difference(G.arch_profile(u0 + 0.3, u1 - 0.3, z - 6.0, z - th - 0.25 - (u1 - u0 - 0.6) / 2))
        return {"type": "extrude", "prof": prof, "axis": axis_prof, "a0": v - width / 2, "a1": v + width / 2, "cls": "deck"}
    u0, u1 = sorted((ay, by)); v = ax
    prof = sbox(u0, z - th - (1.2 if arch else 0), u1, z)
    if arch and u1 - u0 > 2.0: prof = prof.difference(G.arch_profile(u0 + 0.3, u1 - 0.3, z - 6.0, z - th - 0.25 - (u1 - u0 - 0.6) / 2))
    return {"type": "extrude", "prof": prof, "axis": "x", "a0": v - width / 2, "a1": v + width / 2, "cls": "deck"}


def stairs(W, x0, x1, y, z0, z1, width=1.6, axis="x", **kw):
    """A stair — both bodies (stair_solids) and a walkability link (W.links: bottom ↔ top), so that walk.py knows where it
    leads."""
    W.solids += stair_solids(x0, x1, y, z0, z1, width=width, axis=axis, **kw)
    a, b = ((x0, y, z0), (x1, y, z1)) if axis == "x" else ((y, x0, z0), (y, x1, z1))
    if not hasattr(W, "links"): W.links = []
    W.links.append({"kind": "stair", "a": a, "b": b})


def stair_solids(x0, x1, y, z0, z1, width=1.6, rise=0.18, tread=0.30, axis="x", cls="stair"):
    """An outer stair from (x0, z0) to (x1, z1) along an axis, of width width (along the other axis centered at y): the steps
    are prisms down to z0."""
    n = max(1, int(round(abs(z1 - z0) / rise))); dz = (z1 - z0) / n; L = x1 - x0; dx = L / n; out = []
    for i in range(n):
        a, b_ = sorted((x0 + dx * i, x0 + dx * (i + 1)))
        zt = z0 + dz * (i + 1); zb = min(z0, z1) - 0.4
        poly = sbox(a, y - width / 2, b_, y + width / 2) if axis == "x" else sbox(y - width / 2, a, y + width / 2, b_)
        out.append({"type": "prism", "poly": poly, "z0": zb, "z1": zt, "cls": cls})
    return out


def slice_png(M, path, side="y1", offset=0.06, scale=12, pad=20):
    """A section of the mesh by a plane near a block face (offset inward by offset): triangle segments by class → PNG (a check
    without Blender). Color is the class; a thick line — walls and slabs."""
    from PIL import Image, ImageDraw
    V, T, C, _, _ = M.arrays()
    if side in ("y1", "y0"):
        val = (M.Y1 - offset) if side == "y1" else (M.Y0 + offset); k, ku = 1, 0
    else:
        val = (M.X1 - offset) if side == "x1" else (M.X0 + offset); k, ku = 0, 1
    P = V[T]; d = P[:, :, k] - val; segs = []
    for i in np.nonzero((d.min(1) < 0) & (d.max(1) > 0))[0]:
        pts = []
        for e in ((0, 1), (1, 2), (2, 0)):
            da, db = d[i, e[0]], d[i, e[1]]
            if (da < 0) != (db < 0):
                tt = da / (da - db); q = P[i, e[0]] + (P[i, e[1]] - P[i, e[0]]) * tt; pts.append((q[ku], q[2]))
        if len(pts) == 2: segs.append((pts, int(C[i])))
    if not segs: return None
    allp = np.array([p for s_, _ in segs for p in s_]); lo = allp.min(0); hi = allp.max(0)
    Wd = int((hi[0] - lo[0]) * scale) + 2 * pad; Ht = int((hi[1] - lo[1]) * scale) + 2 * pad
    im = Image.new("RGB", (Wd, Ht), (250, 248, 244)); dr = ImageDraw.Draw(im)
    col = {0: (120, 110, 100), 1: (170, 150, 120), 2: (110, 150, 90), 3: (40, 40, 40), 4: (40, 40, 40), 5: (150, 60, 50),
           6: (90, 60, 160), 7: (60, 120, 160), 8: (40, 110, 170), 9: (160, 110, 60), 10: (200, 120, 40), 11: (60, 60, 120),
           12: (40, 120, 200), 13: (90, 90, 90)}
    f = lambda q: (pad + (q[0] - lo[0]) * scale, Ht - pad - (q[1] - lo[1]) * scale)
    for (pa, pb), c in segs: dr.line([f(pa), f(pb)], fill=col.get(c, (0, 0, 0)), width=2 if c in (3, 4, 9, 13) else 1)
    im.save(path); return path
