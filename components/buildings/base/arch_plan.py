"""Shared decisions of the building plan — part of the "base" core: a number from a range (uni), a per-name
seed (irng), a tower by tiers from the style genes (tower_plan; tops CROWN_HALF, CROWN_SHARE). Temples and
structures both call them from the core. The function bodies are kept as they were (the reference)."""
import os
import sys
import zlib

import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.style import style_genes as TS                                                       # noqa: E402

CROWN_HALF = {"dome": (0.9, 1.2), "onion": (1.7, 2.3), "stepped": (1.0, 1.4), "spikes": (0.9, 1.4), "horns": (1.0, 1.5),
              "blades": (0.8, 1.2), "stump": (0.25, 0.35)}                     # top sized by the tier's half-width
CROWN_SHARE = {"pyramid": (0.15, 0.25), "helm": (0.16, 0.24), "tent": (0.32, 0.45)}   # top as a share of the tower height


def uni(r, rng):
    return float(rng.uniform(*r)) if isinstance(r, (list, tuple)) else float(r)


def irng(seed, name):
    """A per-name seed (choosing a family's genes does not shift the neighboring decisions)."""
    return np.random.default_rng((int(seed) * 1000003 + zlib.crc32(name.encode())) % (2 ** 32))


def tower_plan(h, side, z0, z_f, TSg, k, rng, crown=None, octagon=None):
    """Tower by tiers: the base up to the top of the facade (for a crossing tower — up to the ridge) → the belfry
    tiers (square, each narrower) → an octagon with pinnacles at the corners (Freiburg, Ulm) → the top:
    platform, needle, openwork needle, leaf, cluster of spires, pyramid, rhombic helm, tent, dome, onion, steps,
    spikes, horns, blades."""
    sp = crown or TS.choose(TSg["spire"], rng); sh_ = TSg["stage_h"] * k ** 0.85
    if sp in ("needle", "openwork", "leaf", "cluster"): f_ = uni(TSg["spire_k"], rng); hs = (h - z0) * f_
    elif sp in CROWN_SHARE: f_ = uni(CROWN_SHARE[sp], rng); hs = (h - z0) * f_
    elif sp in CROWN_HALF: f_ = uni(CROWN_HALF[sp], rng); hs = side / 2 * 0.8 * f_
    else: f_ = 0.0; hs = 0.0
    zm = h - hs; zcap = z0 + 0.97 * float(TSg.get("slender", 9.0)) * side       # masonry is no slimmer than the style norm — the rest goes to the spire
    if zm > zcap:
        if sp in ("needle", "openwork", "leaf", "cluster", "tent"): hs = h - zcap
        zm = zcap
    z_f = min(z_f, zm - 0.8 * sh_)
    free = zm - z_f; n = int(np.clip(round(free / sh_), 1, TSg.get("max_stages", 3)))
    oct_ = octagon if octagon is not None else (n >= 2 and rng.random() < TSg["octagon"])
    st = [{"plan": "square", "half": round(side / 2, 2), "z0": round(z0, 2), "z1": round(z_f, 2), "kind": "base"}]
    half = side / 2; z = z_f
    for i in range(n):
        last = i == n - 1; plan = "poly8" if (oct_ and last) else "square"
        half = half * (0.84 if plan == "poly8" else (TSg["taper"] if i else 0.97))
        st.append({"plan": plan, "half": round(half, 2), "z0": round(z, 2), "z1": round(z + free / n, 2),
                   "kind": "lantern" if plan == "poly8" else "belfry", "open": TSg["open"]})
        z += free / n
    if sp in CROWN_HALF: hs = st[-1]["half"] * f_                               # top sized by the half-width of the top tier
    why = (f"{len(st)} tiers (base up to {z_f:.0f} m, " + ", ".join(s_["plan"] for s_ in st[1:]) + f"), top {sp}"
           + (f" {hs:.0f} m" if hs else "") + (", octagon with pinnacles" if oct_ else ""))
    return {"stages": st, "spire": {"kind": sp, "h": round(hs, 2)}, "pinnacles": bool(TSg["pinnacles"]), "buttresses": bool(TSg["buttresses"]),
            "h": round(st[-1]["z1"] + hs, 1), "why": why}
