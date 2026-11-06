"""Temple biography — uniqueness: building campaigns by the eras of the line (what was built first and in which style), the unfinished
last campaign (a tower stump, a crane; the nave not built), a special landmark element of the silhouette, annexes, condition (ruin),
irregularity — each decision with a reason. Knowledge — temples.yaml → story, signatures, additions, states. The deviation of the
choir axis is decided by the plan from the ground (temple_plan.py)."""
import os
import sys

import numpy as np
import yaml

from components.buildings import arch_parts                                                              # noqa: E402
from components.buildings.style import style_genes as TS                                                       # noqa: E402

TK = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "temples.yaml")))   # temple knowledge: biography, canon
ST = TK["story"]; SIG = TK["signatures"]; ADD = TK["additions"]
PARTS_RU = {"east": "choir and east end", "transept": "transept and crossing", "nave": "nave", "west": "facade and towers"}


def chain_weight(name, fams):
    """Weight of a rule for a family, and if the rule has none — for its base family (hell → dark → …)."""
    while name:
        if name in fams: return float(fams[name])
        name = TS.SK["families"].get(name, {}).get("base")
    return 0.0


def weight(rule, S):
    return sum(q * chain_weight(n, rule.get("families", {})) for n, q in zip(S["_families"], S["_weights"]))


def story(S, rng, kn, layout):
    """S — style (before the genes are chosen); kn — knobs {history, wealth, ruin, irregular} (None — from the seed and the family);
    layout — what the plan has {plan, towers, transept, crossing, flyers}. → the biography and the markup records."""
    notes = []
    def note(what, value, why, alt=None):
        notes.append({"step": "history", "what": what, "value": value, "why": why, **({"otherwise": alt} if alt else {})})
    hist = float(kn["history"]) if kn.get("history") is not None else float(rng.uniform(0.15, 0.9))
    wealth = float(kn["wealth"]) if kn.get("wealth") is not None else float(rng.uniform(0.3, 0.85))
    ruin = float(kn.get("ruin") or 0.0); irr = float(kn["irregular"]) if kn.get("irregular") is not None else float(S.get("irregular", 0.15))
    basil = layout["plan"] == "basilica"
    # ---- campaigns: the order of the parts and the era of each
    n_c = int(np.clip(1 + round(hist * 3 + rng.uniform(-0.4, 0.4)), 1, 4)) if basil else 1
    east_first = bool(rng.random() < ST["start_east"])
    order = ["east", "transept", "nave", "west"] if east_first else ["nave", "transept", "east", "west"]
    groups = {1: [order], 2: [order[:3], order[3:]], 3: [order[:2], order[2:3], order[3:]], 4: [[x] for x in order]}[n_c]
    i_main = next(i for i, g in enumerate(groups) if "nave" in g)
    p_shift = 0.35 + 0.5 * hist; steps = [0] * n_c
    for i in range(i_main - 1, -1, -1): steps[i] = steps[i + 1] - int(rng.random() < p_shift)
    for i in range(i_main + 1, n_c): steps[i] = steps[i - 1] + int(rng.random() < p_shift)
    camps = []
    for i, g in enumerate(groups):
        fam = S["_name"] if steps[i] == 0 else TS.era_family(S, steps[i])
        camps.append({"i": i, "parts": g, "family": fam, "step": steps[i]})
    part_fam = {p: c["family"] for c in camps for p in c["parts"]}
    if n_c > 1:
        note("campaigns", " → ".join(f"{'+'.join(PARTS_RU[p] for p in c['parts'])}: {c['family']}" for c in camps),
             ("started with the choir — services are needed at once" if east_first else "started with the nave — the east end was occupied by the old church (Amiens)")
             + f"; facade and towers last; the era changes between campaigns with a share of {p_shift:.2f} (knob history {hist:.2f}): "
             "parts in the style of their time (Le Mans — a Romanesque nave and a Gothic choir)", "history=0 — in a single campaign")
    # ---- the unfinished last campaign
    unfinished = None
    if n_c >= 2 and hist >= 0.4 and rng.random() < ST["unfinished"]:
        nt = layout["towers"]; r_ = rng.random()
        if nt >= 2: kind = "one_tower" if r_ < 0.6 else ("both_towers" if r_ < 0.85 or not east_first else "nave")
        elif nt == 1: kind = "tower" if (r_ < 0.8 or not east_first) else "nave"
        else: kind = "nave" if east_first else None
        if kind:
            crane = bool(rng.random() < ST["crane"]) and kind != "nave"
            unfinished = {"kind": kind, "crane": crane, "which": int(rng.integers(0, 2)), "level": round(float(rng.uniform(0.45, 0.95)), 2),
                          "nave_bays": int(rng.integers(1, 3))}
            note("unfinished", {"one_tower": "one tower is a stump", "both_towers": "both towers are stumps", "tower": "the tower is a stump",
                               "nave": "nave not built — a temporary wall"}[kind] + (", crane on the stump" if crane else ""),
                 "the last campaign is unfinished: Cologne — a stump with a crane for ~400 years, Strasbourg — one tower, Ulm — spire of 1890, "
                 "Beauvais — the nave was never built", "history= below 0.4 — completed")
    # ---- special landmark element
    cand = {}
    for k, r in SIG.items():
        w = weight(r, S); need = r.get("needs")
        if w <= 0 or (need == "crossing" and not layout["crossing"]) or (need == "towers2" and layout["towers"] < 2) \
                or (need == "transept" and not layout["transept"]): continue
        if k in ("twin_different", "great_spire") and unfinished and unfinished["kind"] in ("one_tower", "both_towers", "tower"): continue
        cand[k] = w
    signature = None
    if cand:
        k = TS.choose(cand, rng); signature = {"kind": k, "why": SIG[k]["why"]}
        note("special element", k, SIG[k]["why"] + f"; out of {len(cand)} suitable for the style (family weights)", "seed=")
    # ---- additions
    adds = []
    if basil:
        n_max = 1 + int(round(3 * wealth)); pool = []
        for k, r in ADD.items():
            w = weight(r, S)
            if w <= 0 or (r.get("needs") == "flyers" and not layout["flyers"]): continue
            if k == "cloister" and layout.get("tight"): continue
            if rng.random() < min(0.95, w * (0.45 + 0.8 * wealth)): pool.append(k)
        rng.shuffle(pool); adds = [{"kind": k, "why": ADD[k]["why"]} for k in pool[:n_max]]
        if adds:
            note("annexes", ", ".join(a["kind"] for a in adds), "by family and wealth (knob wealth "
                 f"{wealth:.2f}: up to {n_max}); " + "; ".join(a["why"] for a in adds), "wealth=")
    side = -1 if rng.random() < 0.7 else 1                                       # cloister and chapter house — usually on the south side (by the warm wall)
    # ---- condition and irregularity
    state = {"ruin": ruin, "nave_roofless": ruin >= TK["states"]["ruin_nave"], "spires_fallen": ruin >= TK["states"]["ruin_spire"]}
    if ruin > 0:
        note("state", ("nave without roof and vaults; " if state["nave_roofless"] else "") + ("spires collapsed" if state["spires_fallen"] else "decays"),
             f"knob ruin {ruin:.2f}: spires and wooden roofs fall first, then the nave vaults (Tintern)", "ruin=0")
    return {"history": round(hist, 2), "wealth": round(wealth, 2), "irregular": round(irr, 2), "east_first": east_first,
            "campaigns": camps, "part_family": part_fam, "unfinished": unfinished, "signature": signature, "additions": adds,
            "side": side, "state": state, "notes": notes}
