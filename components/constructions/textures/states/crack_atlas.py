"""Crack atlas by growth, like real cracks (a Voronoi network read as scribbles, and the Poly Haven scans' height maps hold
pits and fragments, not a network). The model follows published crack models (growth from a seed: the heading drifts
slowly, the deviation from the heading returns to it — a crack keeps its direction and is finely ragged; tapering toward
the end, branching at 25–60°, stopping at a crossing crack → a T-junction, the main sign of a real network),
simplified, on a torus (the atlas is seamless). The heading drifts instead of accumulating a turn (an accumulated turn
produced looping "worms").
Output — library/textures/states/cracks_<name>/: mask.png (the crack order in the value: main 1.0, branch 0.66, hairline
0.33 — the logic mask opens the main ones first), height.png (depth: a blurred core), meta.json.
.venv/bin/python -m components.constructions.textures.states.crack_atlas [name=stone] [px=2048] [m=2.0] [seed=1]"""
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)

# order → (number of seeds per atlas, length in px (min, max), width in px at the start, tortuosity rad/step, branching
# per step)
LEVELS = {1.0: (6, (700, 1800), 5.0, 0.10, 0.004), 0.66: (0, (100, 420), 2.6, 0.12, 0.0), 0.33: (80, (40, 180), 1.4, 0.14, 0.0)}


def grow(px=2048, seed=1):
    rng = np.random.default_rng(seed)
    occ = np.zeros((px, px), np.int32)            # who occupies a cell: the crack number (to stop at another crack)
    level = np.zeros((px, px), np.float32); width = np.zeros((px, px), np.float32)
    cracks = []; cid = [0]

    def walk(x, y, ang, L, w0, wig, lvl, br, parent):
        cid[0] += 1; me = cid[0]; pts = []
        step = 1.5; n = int(L / step); head = ang; dev = 0.0
        for i in range(n):
            head += rng.normal(0, 0.012)                                # the heading drifts slowly
            dev = 0.82 * dev + rng.normal(0, wig)                       # the deviation returns to the heading: a crack keeps its direction
            a = head + dev
            jit = rng.normal(0, 0.45)                                   # fine jaggedness across the direction of travel
            x = (x + step * math.cos(a) - jit * math.sin(a)) % px; y = (y + step * math.sin(a) + jit * math.cos(a)) % px
            ix, iy = int(x) % px, int(y) % px
            o = occ[iy, ix]
            if o and o != me and o != parent and i > 3:               # ran into another crack — a T-junction, the end
                pts.append((x, y, 0.0)); break
            w = w0 * (1.0 - i / n) ** 0.6 + 0.6                        # thinner toward the end
            pts.append((x, y, w))
            if br and rng.random() < br and lvl >= 1.0:                # branch: at 30–70°, thinner, shorter
                sgn = 1 if rng.random() < 0.5 else -1
                walk(x, y, a + sgn * math.radians(rng.uniform(25, 60)), rng.uniform(*LEVELS[0.66][1]),
                     w * 0.55, LEVELS[0.66][3], 0.66, 0.0, me)
        cracks.append((me, lvl, pts))
        for (cx, cy, cw) in pts:                                       # occupy cells (with a margin for the width)
            r = max(1, int(cw))
            occ[max(0, int(cy) - r):int(cy) + r + 1, max(0, int(cx) - r):int(cx) + r + 1] = me
        return me

    for lvl in (1.0, 0.33):
        nseed, (l0, l1), w0, wig, br = LEVELS[lvl]
        for _ in range(nseed):
            x, y = rng.random(2) * px; a = rng.random() * 2 * math.pi; L = rng.uniform(l0, l1)
            walk(x, y, a, L / 2, w0, wig, lvl, br, 0)                  # in both directions from the seed
            walk(x, y, a + math.pi, L / 2, w0, wig, lvl, br, 0)
    # raster: a torus — draw 3×3 copies of the segment, cut out the middle
    big = np.zeros((px * 3, px * 3), np.float32); lv = np.zeros_like(big)
    for _, lvl, pts in cracks:
        for (x0, y0, w0), (x1, y1, w1) in zip(pts[:-1], pts[1:]):
            if abs(x1 - x0) > px / 2 or abs(y1 - y0) > px / 2: continue   # a jump across the edge of the torus
            th = max(1, int(round((w0 + w1) / 2)))
            for dx in (0, px, 2 * px):
                for dy in (0, px, 2 * px):
                    p0 = (int((x0 + dx) * 4), int((y0 + dy) * 4)); p1 = (int((x1 + dx) * 4), int((y1 + dy) * 4))
                    cv2.line(big, p0, p1, 1.0, th, cv2.LINE_AA, shift=2)
                    cv2.line(lv, p0, p1, float(lvl), th, cv2.LINE_8, shift=2)
    m = big[px:2 * px, px:2 * px]; lvm = lv[px:2 * px, px:2 * px]
    mask = np.clip(m, 0, 1) * np.maximum(lvm, 0.33 * (m > 0))          # the value is the crack order
    height = 1.0 - cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 1.6) * 0.9 # depth: a blurred core
    return mask, height, len(cracks)


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    name, px, size, seed = kw.get("name", "stone"), int(kw.get("px", 2048)), float(kw.get("m", 2.0)), int(kw.get("seed", 1))
    mask, height, n = grow(px, seed)
    d = os.path.join(ROOT, "library", "textures", "states", f"cracks_{name}"); os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(d, "height.png"), (height * 255).astype(np.uint8))
    rel = lambda f: os.path.relpath(os.path.join(d, f), ROOT)
    meta = {"id": f"cracks_{name}", "source": "crack_atlas.py (crack growth)", "license": "own", "size_m": [size, size],
            "kind": "cracks", "n_cracks": n, "seed": seed, "maps": {"mask": rel("mask.png"), "height": rel("height.png")}}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)
    ip = os.path.join(ROOT, "library", "textures", "states", "index.json"); idx = json.load(open(ip)) if os.path.exists(ip) else {}
    idx[meta["id"]] = meta; json.dump(idx, open(ip, "w"), ensure_ascii=False, indent=1)
    print(meta["id"], "n_cracks", n, "frac", round(float((mask > 0.1).mean()), 4))
