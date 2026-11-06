"""Timber — layout of wooden wall and fence structures: fachwerk, log house, cladding, palisade, wattle (the timber part of
the Constructions component; knowledge — timber.yaml). Pure numpy: works in the project's Python and in Blender.

A wall is a Panel: the unrolling of a face (u along the wall left to right from outside, z is height) as in masonry;
storeys (floor levels), openings (u0, u1, z0, z1), outer corners at the edges. The layout → elements in (u, z):
  beam   — a quadrilateral (post, sill, wall plate, rail, brace) with an offset from the plane;
  field  — fachwerk infill (plaster) between the beams, openings cut out;
  log    — a log-house course (u0, u1, middle z, radius; overhang at an outer corner; neighboring walls shifted by half
           a course);
  plank  — cladding and a batten.
A fence is a line on the ground: a palisade (logs tight together with a point), wattle (stakes and rods interwoven).
Element number: 8·10⁸ + wall·10⁴ + number.

  .venv/bin/python -m components.constructions.timber.timber resolve <folder> [fachwerk=fachwerk] [buildings=log_oblo] [fences=…] out=<plan.json>
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "timber.yaml")
SID_BASE = 800_000_000


def load_library(path=LIB):
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    base, kinds = d["base"], {}
    for name, kw in d["kinds"].items():
        k = dict(base); k.update({a: b for a, b in kw.items() if a != "title"}); k["name"] = name; k["title"] = kw.get("title", name)
        kinds[name] = k
    return kinds, d


try:
    KINDS, LIBRARY = load_library()
except ImportError:
    KINDS, LIBRARY = {}, {}


def _rng(*key):
    return np.random.default_rng([int(abs(k)) & 0x7FFFFFFF for k in key])


def rect(u0, u1, z0, z1):
    return [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]


class Panel:
    """Wall: p0 → p1 (x, y) — the bottom of the face left to right from outside, n — outward (x, y), z0…z1; storeys — floor
    levels (≥ z0); openings [(u0, u1, z0, z1)]; corners — whether the edges are outer ones (left, right); the course phase
    (0 | 0.5) — by the wall direction."""

    def __init__(self, p0, p1, z0, z1, n=None, storeys=(), openings=(), corners=(True, True), name="wall"):
        self.p0 = np.asarray(p0, float)[:2]; self.p1 = np.asarray(p1, float)[:2]; d = self.p1 - self.p0
        self.L = float(np.hypot(*d)); self.t = d / max(self.L, 1e-9)
        nn = np.array([self.t[1], -self.t[0]]) if n is None else np.asarray(n, float)[:2]
        self.n = nn / max(np.linalg.norm(nn), 1e-9)
        self.z0, self.z1 = float(z0), float(z1); self.name = name
        lv = sorted({float(z) for z in storeys if self.z0 - 1e-6 <= z < self.z1 - 0.5} | {self.z0})
        self.levels = lv + [self.z1]
        self.openings = [tuple(map(float, o[:4])) for o in openings]; self.corners = tuple(corners)
        self.phase = 0.5 if abs(self.n[0]) >= abs(self.n[1]) else 0.0

    def world(self, u, z, d):
        u, z, d = np.broadcast_arrays(np.asarray(u, float), np.asarray(z, float), np.asarray(d, float))
        xy = self.p0 + u[..., None] * self.t + d[..., None] * self.n
        return np.concatenate([xy, z[..., None]], -1)


def _subtract(r, o):
    """Rectangle r (u0, u1, z0, z1) minus rectangle o → up to 4 rectangles."""
    u0, u1, z0, z1 = r; a0, a1, b0, b1 = o
    if a1 <= u0 or a0 >= u1 or b1 <= z0 or b0 >= z1: return [r]
    out = []
    if b0 > z0: out.append((u0, u1, z0, b0))
    if b1 < z1: out.append((u0, u1, b1, z1))
    m0, m1 = max(z0, b0), min(z1, b1)
    if a0 > u0: out.append((u0, a0, m0, m1))
    if a1 < u1: out.append((a1, u1, m0, m1))
    return [x for x in out if x[1] - x[0] > 0.01 and x[3] - x[2] > 0.01]


# ── fachwerk ────────────────────────────────────────────────────────────────────────────────────────────────────
def fachwerk(pn, k, seed=1):
    """Frame of the wall storeys: sill and wall plate, posts (corners, opening jambs, the rest no sparser than the "step"),
    rails (at openings — at the sill and the lintel, otherwise at the "levels"), braces by the "pattern" (mann — at corner
    posts over half a storey, crosses — under windows), infill fields between the beams (openings cut out). → elements."""
    sw, sd = k["post"]; ph, pd = k["threshold"]; th, td = k["plate"]; rh, rd = k["rail"]; bw, bd = k["brace"]
    out = []; L = pn.L
    for s, (zb, zt) in enumerate(zip(pn.levels[:-1], pn.levels[1:])):
        if zt - zb < 1.2: continue
        out.append(dict(t="beam", kind="threshold", poly=rect(0, L, zb, zb + ph), d=pd))
        out.append(dict(t="beam", kind="plate", poly=rect(0, L, zt - th, zt), d=td))
        ya, yb = zb + ph, zt - th
        ops = [o for o in pn.openings if o[3] > ya + 0.1 and o[2] < yb - 0.1]
        fixed = [sw / 2, L - sw / 2] + [x for o in ops for x in (o[0] - sw / 2, o[1] + sw / 2)]
        fixed = sorted(x for x in fixed if sw / 2 - 1e-6 <= x <= L - sw / 2 + 1e-6)
        posts = []
        for x in fixed:
            if not posts or x - posts[-1] > 0.35: posts.append(x)
        full = [posts[0]]
        for a, b in zip(posts[:-1], posts[1:]):
            inside = any(o[0] - 0.01 <= (a + b) / 2 <= o[1] + 0.01 for o in ops)
            n = 0 if inside else int(math.ceil((b - a) / float(k["step"]) - 1e-6)) - 1
            full += [a + (b - a) * (i + 1) / (n + 1) for i in range(max(n, 0))] + [b]
        for x in full: out.append(dict(t="beam", kind="post", poly=rect(x - sw / 2, x + sw / 2, ya, yb), d=sd))
        for i, (a, b) in enumerate(zip(full[:-1], full[1:])):
            ua, ub = a + sw / 2, b - sw / 2
            if ub - ua < 0.05: continue
            bo = [o for o in ops if o[0] < ub - 0.01 and o[1] > ua + 0.01]
            rails = sorted({o[2] - rh for o in bo if o[2] - rh > ya + 0.1} | {o[3] for o in bo if o[3] < yb - rh - 0.05}) if bo else \
                    [zb + lv for lv in k["levels"] if ya + 0.25 < zb + lv < yb - rh - 0.25]
            for zr in rails: out.append(dict(t="beam", kind="rail", poly=rect(ua, ub, zr, zr + rh), d=rd))
            bands = [ya] + [z for zr in rails for z in (zr, zr + rh)] + [yb]
            fields = [(ua, ub, z0_, z1_) for z0_, z1_ in zip(bands[0::2], bands[1::2]) if z1_ - z0_ > 0.05]
            for o in bo:
                fields = [f for r in fields for f in _subtract(r, (o[0], o[1], o[2], o[3]))]
            for f in fields: out.append(dict(t="field", poly=rect(*f)))
            corner_bay = (i == 0 and pn.corners[0]) or (i == len(full) - 2 and pn.corners[1])
            if k["pattern"] == "mann" and corner_bay and not bo and len(full) > 2:   # a brace from a corner post to the next one over half a storey
                zt_ = min(yb, ya + (yb - ya) * 0.62)
                if i == 0: p, q = (ua, ya), (ub, zt_)
                else: p, q = (ub, ya), (ua, zt_)
                out.append(dict(t="beam", kind="brace", poly=_diag(p, q, bw), d=bd))
            win = [o for o in bo if o[2] - rh > ya + 0.35]                         # a window (the sill is above the floor), not a door
            if k["pattern"] == "crosses" and win:                                    # St. Andrew's cross under the window
                z_ = min(o[2] for o in win) - rh
                out.append(dict(t="beam", kind="brace", poly=_diag((ua, ya), (ub, z_), bw), d=bd))
                out.append(dict(t="beam", kind="brace", poly=_diag((ub, ya), (ua, z_), bw), d=bd))
    return out


def _diag(p, q, w):
    """Brace p → q of width w: a parallelogram with horizontal ends (a cut into the sill and the rail)."""
    p, q = np.asarray(p, float), np.asarray(q, float); d = q - p; L = float(np.hypot(*d))
    if L < 1e-6: return rect(p[0], p[0] + w, p[1], p[1] + w)
    s = abs(d[1]) / L; hx = (w / 2) / max(s, 0.2)                                # half-width along the horizontal
    return [(p[0] - hx, p[1]), (p[0] + hx, p[1]), (q[0] + hx, q[1]), (q[0] - hx, q[1])]


# ── log house ───────────────────────────────────────────────────────────────────────────────────────────────────
def srub(pn, k, seed=1, ri=0):
    """Courses of a wall from the bottom up: the course height is a fraction of the diameter (the groove); at an outer corner —
    an overhang of "remainder" diameters (round notch) or the end flush (lap joint); neighboring walls are shifted by half a
    course (phase); at an opening the log is cut and surrounded by a casing. → elements."""
    dmin, dmax = k["d"]; hc = float(k["crown"]) * (dmin + dmax) / 2; out = []
    rng = _rng(seed, ri, 13); z = pn.z0 + hc * pn.phase; n = 0
    while z + hc * 0.5 < pn.z1:
        d = float(rng.uniform(dmin, dmax)); zc = z + hc / 2; ext = float(k["remainder"]) * d
        e0 = -ext if pn.corners[0] else 0.0; e1 = pn.L + ext if pn.corners[1] else pn.L
        cuts = sorted((o[0], o[1]) for o in pn.openings if o[2] < zc < o[3])
        a = e0
        for c0, c1 in cuts + [(e1, e1)]:
            if c0 - a > 0.15: out.append(dict(t="log", u0=a, u1=c0, zc=zc, r=d / 2, course=n, ends=(a == e0 and pn.corners[0], c0 == e1 and pn.corners[1])))
            a = max(a, c1)
        z += hc; n += 1
    for o in pn.openings:                                                        # casing: boards around the opening outline
        u0, u1, z0, z1 = o; w = 0.12
        for poly in (rect(u0 - w, u0, z0 - w, z1 + w), rect(u1, u1 + w, z0 - w, z1 + w), rect(u0, u1, z1, z1 + w), rect(u0, u1, z0 - w, z0)):
            out.append(dict(t="beam", kind="casing", poly=poly, d=0.05))
    return out


# ── cladding ────────────────────────────────────────────────────────────────────────────────────────────────────
def boards(pn, k, seed=1, ri=0):
    rng = _rng(seed, ri, 19); out = []; u = 0.0; w0, w1 = k["w"]; bw, bd = k["batten"]; seams = []
    while u < pn.L - 0.02:
        w = min(float(rng.uniform(w0, w1)), pn.L - u); out.append(dict(t="beam", kind="plank", poly=rect(u, u + w, pn.z0, pn.z1), d=float(k["t"])))
        u += w; seams.append(u)
    for s in seams[:-1]: out.append(dict(t="beam", kind="batten", poly=rect(s - bw / 2, s + bw / 2, pn.z0, pn.z1), d=float(k["t"]) + bd))
    for o in pn.openings:                                                        # an opening is subtracted as a rectangle from each board
        nxt = []
        for e in out:
            q = e["poly"]; r = (q[0][0], q[1][0], q[0][1], q[2][1])
            nxt += [dict(t="beam", kind=e["kind"], poly=rect(*x), d=e["d"]) for x in _subtract(r, o)]
        out = nxt
    return out


def panel_items(pn, k, seed=1, ri=0):
    g = k["generator"]
    items = fachwerk(pn, k, seed) if g == "fachwerk" else srub(pn, k, seed, ri) if g == "log_house" else boards(pn, k, seed, ri) if g == "cladding" else []
    for i, e in enumerate(items):
        e["sid"] = SID_BASE + (ri % 9000) * 10_000 + (i % 10_000)
        e["tint"] = round(float(_rng(seed, ri, i, 5).normal(1.0, float(k["spread"]))), 4)
    return items


# ── fences along a line ─────────────────────────────────────────────────────────────────────────────────────────
def along(pts, step, start=0.0):
    """Points of a polyline every step along its length: [(x, y, tangent)]."""
    P = np.asarray(pts, float)[:, :2]; seg = np.hypot(*np.diff(P, axis=0).T); s = np.concatenate([[0.0], np.cumsum(seg)]); out = []
    for x in np.arange(start, s[-1] + 1e-9, step):
        i = min(int(np.searchsorted(s, x, side="right")) - 1, len(seg) - 1); f = (x - s[i]) / max(seg[i], 1e-9)
        p = P[i] + (P[i + 1] - P[i]) * f; t = (P[i + 1] - P[i]) / max(seg[i], 1e-9); out.append((p, t))
    return out


def palisade(pts, k, seed=1, ri=0):
    """Palisade: logs tight together along the line (the step is the diameter), random heights, a point; → [{xy, r, h, tip,
    sid}]."""
    rng = _rng(seed, ri, 23); dmin, dmax = k["d"]; out = []; s = 0.0; P = np.asarray(pts, float)
    total = float(np.sum(np.hypot(*np.diff(P[:, :2], axis=0).T)))
    while s < total:
        d = float(rng.uniform(dmin, dmax)); at = along(pts, 1e9, s + d / 2)
        if not at: break
        p, t = at[0]
        out.append(dict(xy=p.tolist(), r=d / 2, h=float(rng.uniform(*k["h"])), tip=float(k["tip"]) * float(rng.uniform(0.8, 1.2)),
                        sid=SID_BASE + 9_000_000 + (ri % 90) * 10_000 + len(out), tint=round(float(rng.normal(1.0, float(k["spread"]))), 4)))
        s += d + 0.01
    return out


def wattle(pts, k, seed=1, ri=0):
    """Wattle: stakes every "step", rods in interwoven rows (a row passes a stake now outside, now inside; the neighboring row
    the opposite way). → (stakes [{xy, r, h}], rods [{pts3: [(x, y, z)], r}])."""
    rng = _rng(seed, ri, 29); st = along(pts, float(k["step"])); h = float(rng.uniform(*k["h"])); rr = float(k["rod"]) / 2
    stakes = [dict(xy=p.tolist(), r=float(rng.uniform(*k["d"])) / 2, h=h + float(rng.uniform(0.0, 0.15)), sid=SID_BASE + 9_500_000 + (ri % 40) * 10_000 + i,
                   tint=round(float(rng.normal(1.0, float(k["spread"]))), 4)) for i, (p, t) in enumerate(st)]
    rods = []
    for j, z in enumerate(np.arange(0.15, h - 0.05, 2.2 * rr)):
        pts3 = []
        for i, (p, t) in enumerate(st):
            n = np.array([-t[1], t[0]]); side = 1 if (i + j) % 2 == 0 else -1; off = stakes[i]["r"] + rr
            pts3.append((*(p + n * off * side), float(z)))
        rods.append(dict(pts3=pts3, r=rr, sid=SID_BASE + 9_800_000 + (ri % 20) * 10_000 + j, tint=round(float(rng.normal(1.0, float(k["spread"]))), 4)))
    return stakes, rods


# ── step plan ───────────────────────────────────────────────────────────────────────────────────────────────────
DESC_FILES = ("constructions.yaml", "masonry.yaml")
PART_KEYS = ("fachwerk", "buildings", "fences")                                 # fences — the default fence kind (a knob); in the description fences is a list of fences
FENCE_GEN = ("palisade", "wattle")                                             # fence generators (timber.yaml)


def _detail(v):
    if v not in ("cam", "all"): raise SystemExit(f"detail level '{v}': cam | all")
    return v


def resolve(folder, kw=None):
    """Timber plan of a building: <folder>/constructions.yaml — the "timber:" section: fachwerk — the kind for buildings
    marked as half-timbered (upper storeys), buildings — the kind for all buildings (whole walls: log house, cladding),
    houses — the settlement house kind → walls, fences — fences as a list {path: [[x, y]…], kind}; a fence without kind
    takes the kind of the fences knob (palisade). detail (cam | all), far — the far-view px per element, seed. Command-line
    keys override the description; the fences are validated here (fence kind, path ≥ 2 points)."""
    import yaml
    kw = dict(kw or {}); path = next((os.path.join(folder, f) for f in DESC_FILES if os.path.exists(os.path.join(folder, f))), "") if folder else ""
    d = (yaml.safe_load(open(path, encoding="utf-8")) or {}) if path else {}
    c = dict(d.get("timber") or {})
    lines = c.get("fences") if isinstance(c.get("fences"), list) else []      # description fences are a list; fences as a string is the default kind
    if any(not isinstance(f, dict) for f in lines): raise SystemExit("timber: the description's fences must be a list {path: [[x, y]…], kind}")
    parts = {"fachwerk": "fachwerk", "fences": "palisade"}
    parts.update({p_: c[p_] for p_ in PART_KEYS if isinstance(c.get(p_), str) and c[p_]}); parts.update({p_: kw[p_] for p_ in PART_KEYS if kw.get(p_)})
    if str(parts.get("fachwerk")) in ("no", "0", "none"): parts.pop("fachwerk")
    fences = [dict(f, kind=f.get("kind") or parts["fences"]) for f in lines]     # a fence without a kind of its own takes the kind of the fences knob
    bad = sorted({str(v) for v in parts.values() if str(v) not in KINDS} | {str(f["kind"]) for f in fences if str(f["kind"]) not in KINDS})
    if bad: raise SystemExit(f"timber: no kind {bad}; available: {', '.join(KINDS)} (components/constructions/timber/timber.yaml)")
    fence_kinds = [n for n, k in KINDS.items() if k["generator"] in FENCE_GEN]
    if parts["fences"] not in fence_kinds: raise SystemExit(f"timber: fences={parts['fences']} — not a fence kind; fences: {', '.join(fence_kinds)}")
    wrong = [i for i, f in enumerate(fences) if f["kind"] not in fence_kinds or len(f.get("path") or []) < 2]
    if wrong: raise SystemExit(f"timber: fences {wrong} — need a path of ≥ 2 points and a fence kind ({', '.join(fence_kinds)})")
    houses = dict(LIBRARY.get("houses") or {}); houses.update(c.get("houses") or {})   # settlement house kind → walls (an izba is a log house)
    houses = {t: k for t, k in houses.items() if k and str(k) not in ("no", "0", "none")}
    used = {v for p_, v in parts.items() if p_ != "fences"} | set(houses.values()) | {f["kind"] for f in fences}
    plan = dict(timber=dict(parts=parts, houses=houses, detail=_detail(kw.get("detail") or c.get("detail") or "cam"), far=float(kw.get("Tfar", c.get("far", 2.0))),
                            seed=int(kw.get("seed", c.get("seed", 1))), fences=fences),
                kinds={n: KINDS[n] for n in sorted(used) if n in KINDS} if kw.get("kinds") != "all" else dict(KINDS),
                substances=LIBRARY.get("substances", {}), version=LIBRARY.get("version"), description=path or None)
    return plan


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "resolve":
    KW = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    plan = resolve(sys.argv[2], KW)
    if KW.get("out"): json.dump(plan, open(KW["out"], "w"), ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False))
