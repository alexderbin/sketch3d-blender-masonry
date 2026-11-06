"""Tower by tiers: base → tiers → crown → roof. One grammar for the main tower, the wall towers, a mage's tower
and a villain's dark tower; the style is in castles/castles.yaml → tower_styles (a chain of tier plans,
taper, transitions, windows, crown, spire, turrets, the maximum slenderness). The markup is assembled by
castles/bl_castle.py (staged_tower).

stages(plan tower, style, rng, floor) → {"stages": [{plan, r, z0, z1, trans, win}], "crown": {...}, "spire": {...},
"turrets": {...}, "why": "…"}; a plan tower has r, base, top, plan, role."""
import math

import numpy as np


def pick(w, rng):
    k = list(w); p = np.array([w[x] for x in k], float); return k[int(rng.choice(len(k), p=p / p.sum()))]


def stages(t, sty, rng, storey=3.8, role="keep"):
    """Tiers of a tower: the count follows the masonry height (a tier is 2–3 floors), the bottom one taller than
    the rest; a tier's plan follows the style's chain starting from the base plan; each tier is narrower than
    the one below (the style's taper), a transition is a ledge with a parapet, pinnacles at the corners (square
    → polygon), a balcony gallery or a corbel-out; the crown and roof are chosen by the style's weights, the spire height depends on the top radius. A tier is at least two
    floors high, the upper one sits inside the lower (support)."""
    H = float(t["top"]) - float(t["base"]); r0 = float(t["r"]); base = float(t["base"])
    lo, hi = sty["stages"]
    if role != "keep": lo, hi = max(1, lo - 1), max(1, hi - 2)                      # wall towers are simpler than the main one
    n = int(np.clip(round(H / (storey * 2.6)), lo, hi)); n = max(1, min(n, int(H // (storey * 2))))
    w = np.array([1.35] + [1.0] * (n - 1)) * (1 + rng.uniform(-0.12, 0.12, n)); hs = H * w / w.sum()
    chain = sty["chain"].get(t["plan"], [t["plan"]])
    taper = float(rng.uniform(*sty["taper"]))
    out = []; z = base; r = r0
    for k in range(n):
        plan = chain[min(k, len(chain) - 1)]
        out.append({"plan": plan, "r": round(r, 2), "z0": round(z, 2), "z1": round(z + hs[k], 2), "trans": None, "win": sty["windows"]})
        z += hs[k]
        if k < n - 1:
            nxt = chain[min(k + 1, len(chain) - 1)]
            tr = pick(sty["trans"], rng)
            if tr == "pinnacles" and not (plan == "square" and nxt != "square"): tr = "setback"   # pinnacles — at the corners of a square under a polygon
            out[-1]["trans"] = tr
            r = r * (taper if tr != "corbel" else 1.06)                           # corbel-out — wider on corbels (the top, an archers' gallery)
    top = out[-1]
    crown = pick(sty["crown"], rng); spire = sty.get("keep_spire") if role == "keep" and sty.get("keep_spire") else pick(sty["spire"], rng)
    if spire == "flat" and crown == "none": crown = "merlons"                       # a flat top has merlons
    if crown in ("horns", "crown", "spikes") and spire == "needle": spire = "flat" if rng.random() < 0.5 else spire
    kk = sty["spire_k"].get(spire, [1.0, 1.0]); sh = float(rng.uniform(*kk)) * float(top["r"]) if spire != "flat" else 0.0
    tur = {"n": int(rng.choice([2, 4])), "r": round(float(np.clip(top["r"] * 0.22, 0.9, 2.2)), 2)} if rng.random() < sty["turrets"] else None
    why = (f"{n} tiers ({', '.join(s_['plan'] for s_ in out)}), taper {taper:.2f}, transitions: "
           + ", ".join(s_["trans"] for s_ in out[:-1] if s_["trans"]) + f"; crown {crown}, roof {spire}" + (f" {sh:.0f} m" if sh else ""))
    return {"stages": out, "crown": {"kind": crown}, "spire": {"kind": spire, "h": round(sh, 2)}, "turrets": tur, "why": why}
