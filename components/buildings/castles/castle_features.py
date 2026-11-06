"""Unique elements of a castle: wall stretches and their surroundings → elements by the rules of
castles/features.yaml. Uniqueness comes from the function and the surroundings (what is behind the wall and
in front of it), landmarks, the history of rebuilds and the variety within a kind. Each element is recorded
with a reason (why).

Called from castle_plan.plan(): place(C, ctx) appends elements (features) to the plan, edits walls
(superstructure), adds the walls and towers of the zwinger and the barbican, the flags of buildings;
checks(C, ctx) — the checks of logic and uniqueness."""
import math, os

import numpy as np
import yaml

FK = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.yaml")))
LIVING = {"palas", "palas_big", "hall", "governor", "wing", "convent", "tower_house"}


def normals(P, closed):
    """Outward normals at the wall points (the outline is counter-clockwise): a mitered joint, as in the build."""
    P = np.asarray(P, float)
    tn = (np.roll(P, -1, 0) - P) if closed else np.vstack([P[1:] - P[:-1], P[-1:] - P[-2:-1]])
    tp = (P - np.roll(P, 1, 0)) if closed else np.vstack([P[1:2] - P[:1], P[1:] - P[:-1]])
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9); tp /= np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-9)
    N = np.column_stack([tn[:, 1], -tn[:, 0]]) + np.column_stack([tp[:, 1], -tp[:, 0]])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)


def sections(C, ctx):
    """Wall stretches between towers and gates: stations, length, normal, what is outside, where it faces, what is
    behind the wall."""
    G2, S, st, thr, expo, inside = ctx["G2"], ctx["S"], ctx["st"], ctx["thr"], ctx["exposure"], ctx["inside"]
    gate0 = np.asarray(C["gates"][0]["xy"]); out = []
    wardp = {w["id"]: np.asarray(w["poly"]) for w in C["wards"]}
    ringed = any(w.get("ring") for w in C["wards"])                              # concentric: the inner ring faces the space between the walls
    for wi, w in enumerate(C["walls"]):
        if w["kind"] in ("zwinger", "barbican", "retaining"): continue
        P = np.asarray(w["pts"], float); n = len(P); closed = bool(w["closed"])
        N = normals(P, closed); flip = False
        if not closed:                                                           # cut and barrier: outward — toward the outer court / the threat
            m = P[n // 2]; k = n // 2
            if w["kind"] == "cross" and "inner" in wardp and inside(wardp["inner"], m[0] + N[k, 0] * 4, m[1] + N[k, 1] * 4): N = -N; flip = True
            if w["kind"] == "barrier" and (N[k] @ thr) < 0: N = -N; flip = True
        att = set()
        for t in C["towers"]:
            d = np.hypot(*(P - np.asarray(t["on"])).T); i = int(np.argmin(d))
            if d[i] < max(2.5, t["r"] * 1.2): att.add(i)
        for g in C["gates"]:
            d = np.hypot(*(P - np.asarray(g["xy"])).T); i = int(np.argmin(d))
            if d[i] < 3.0: att.add(i)
        cuts = sorted(att)
        if closed:
            if not cuts: cuts = [0, n // 3, 2 * n // 3]
            ranges = [list(range(a, b + 1)) if b > a else list(range(a, n)) + list(range(0, b + 1)) for a, b in zip(cuts, cuts[1:] + cuts[:1])]
        else:
            pts = [0] + [c for c in cuts if 0 < c < n - 1] + [n - 1]; ranges = [list(range(a, b + 1)) for a, b in zip(pts[:-1], pts[1:])]
        walk = np.asarray(w["walk"]); base = np.asarray(w["base"]); t = np.asarray(w["t"]) if isinstance(w["t"], list) else np.full(n, w["t"])
        ab = set(w.get("absorbed", []))                                          # became the wall of a house — not a stretch
        if ab:
            rr = []
            for ii in ranges:
                cur = []
                for j in ii:
                    if j in ab:
                        if cur: rr.append(cur); cur = []
                    else: cur.append(j)
                if cur: rr.append(cur)
            ranges = rr
        for ii in ranges:
            if len(ii) < 3: continue
            Q = P[ii]; L = float(np.hypot(*np.diff(Q, axis=0).T).sum())
            if L < 6: continue
            Nm = N[ii].mean(0); Nm /= max(np.linalg.norm(Nm), 1e-9)
            zo = G2.height(Q[:, 0] + N[ii, 0] * (t[ii] / 2 + 2.5), Q[:, 1] + N[ii, 1] * (t[ii] / 2 + 2.5))
            zi = G2.height(Q[:, 0] - N[ii, 0] * (t[ii] / 2 + 2.0), Q[:, 1] - N[ii, 1] * (t[ii] / 2 + 2.0))
            ex = [expo(S, Q[k] + N[ii][k] * 4) for k in range(0, len(ii), max(1, len(ii) // 6))]
            exm = max(dict.fromkeys(ex), key=ex.count)                         # on a tie — the first along the wall (not set order)
            if ringed and w["ward"] == "inner": exm = "lists"
            s_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
            slope = float(np.max(np.abs(np.diff(walk[ii])) / np.maximum(np.diff(s_), 1e-6))) if len(ii) > 1 else 0.0
            behind = []
            for bi, b in enumerate(C["buildings"]):
                R = np.asarray(b["poly"]); bm = (R[0] + R[1]) / 2
                d = np.hypot(*(Q - bm).T); k = int(np.argmin(d))
                if d[k] < t[ii][k] / 2 + 2.5 and 0 < k < len(ii) - 1: behind.append((bi, b["kind"], ii[k]))
            mid = Q[len(Q) // 2]
            dq = np.hypot(*np.diff(Q, axis=0).T); ris = int(((dq < 1e-3) & (np.abs(np.diff(walk[ii])) > 0.05)).sum())   # stairstep ledges
            out.append({"id": len(out), "wall": wi, "ii": ii, "L": round(L, 1), "mid": mid, "N": Nm, "ward": w["ward"], "kind": w["kind"], "risers": ris,
                        "outer": not (ringed and w["ward"] == "inner") and w["kind"] != "cross",
                        "exposure": exm, "threat": float(Nm @ thr), "east": float(Nm @ np.array([1.0, 0.0])),
                        "h_out": float(np.mean(walk[ii] - zo)), "h_in": float(walk[ii][len(ii) // 2] - zi[len(ii) // 2]),
                        "depth": float(np.max(walk[ii] - base[ii])), "slope": slope,
                        "behind": behind, "gate_d": float(np.hypot(*(mid - gate0))), "closed": closed, "flip": flip})
    return out


def pick_culture(rule, cu, rng):
    p = rule.get("cultures", {}).get(cu, 0.0) if "cultures" in rule else 1.0
    return rng.random() < p


def place(C, ctx):
    """Place the elements. ctx: G2, S, st, spec, thr, rng, exposure, inside, wall_profile, out."""
    st, sp, rng, thr, G2 = ctx["st"], ctx["spec"], ctx["rng"], ctx["thr"], ctx["G2"]
    cu, era = sp["culture"], sp["era"]; kn = st.get("knobs", {}); age = kn.get("age", 0.5); wealth = kn.get("wealth", 0.5)
    small = st.get("small", True)                                                # goal "background" — without small things
    F = []; secs = sections(C, ctx); used = {s["id"]: set() for s in secs}
    def add(kind, why, **kw):
        f = dict(kind=kind, why=why, **kw); F.append(f); return f
    def free_i(s, i, gap=7.0):
        """A stretch station no closer than gap m to the already placed point elements on the same wall (otherwise the
        nearest free one)."""
        P = np.asarray(C["walls"][s["wall"]]["pts"]); taken = [P[f["i"]] for f in F if f.get("wall") == s["wall"] and "i" in f]
        order = sorted(s["ii"][1:-1], key=lambda j: abs(s["ii"].index(j) - (s["ii"].index(i) if i in s["ii"] else len(s["ii"]) // 2)))
        for j in order:
            if all(np.hypot(*(P[j] - q)) >= gap for q in taken): return int(j)
        return None
    def era_ok(r): return "eras" not in r or era in r["eras"]

    # ---------------- gate: portcullis, bretesche, drawbridge, barbican
    G = FK["gates"]
    for gi, g in enumerate(C["gates"]):
        if g.get("kind") != "simple": add("portcullis", "portcullis in the gateway", gate=gi)
        if gi == 0 and rng.random() < G["breteche"]["prob"]: add("breteche", "over the main gateway — to strike from above", gate=gi)
    if C["bridges"]:
        add("drawbridge", "at the moat — a drawbridge span" + (", lever arms in slots (from the 14th c.)" if era in G["drawbridge"]["beams_eras"] else ""),
            gate=0, beams=era in G["drawbridge"]["beams_eras"])
    bb = G["barbican"]
    if era in bb["eras"] and sp["scale"] in bb["min_scale"] and rng.random() < bb["prob"] and not st.get("bastions"):
        f = barbican(C, ctx)
        if f: F.append(f)

    # ---------------- landmarks: butter churn, wedge, bartizans
    T = FK["towers"]; keep = C.get("keep")
    if keep and keep["kind"] in T["butterfass"]["keeps"] and not keep.get("stages") and pick_culture(T["butterfass"], cu, rng):
        keep["butterfass"] = {"k": round(float(rng.uniform(0.55, 0.68)), 3), "h": round(float(rng.uniform(0.22, 0.32)) * (keep["top"] - keep["base"]), 2)}
        add("butterfass", "the bergfried is topped with a smaller tower — prestige and height (a late superstructure)", keep=True)
    if pick_culture(T["prow"], cu, rng):
        cand = [(ti, t) for ti, t in enumerate(C["towers"]) if t["role"] in ("corner", "flank") and t.get("dir") and not t.get("stages")]
        if keep and keep["rule"] in ("threat", "corner_threat") and keep["plan"] in ("square", "poly8", "round") and not keep.get("stages"):
            keep["prow"] = True; add("prow", "the main tower with a wedge toward the threat — to deflect projectiles", keep=True)
        elif cand:
            ti, t = max(cand, key=lambda q: float(np.dot(np.array([q[1]["dir"][1], -q[1]["dir"][0]]), thr)))
            t["prow"] = True; add("prow", "a tower on the threat side — with a wedge pointing outward", tower=ti)
    if era_ok(T["bartizans"]) and pick_culture(T["bartizans"], cu, rng):
        targets = []
        if keep and keep["plan"] in ("square", "poly8") and keep.get("roof") == "flat" and not keep.get("stages"): targets.append(("keep", None))
        for ti, t in enumerate(C["towers"]):
            if t["role"] == "gate" and t["plan"] == "square" and not t.get("stages"): targets.append(("tower", ti))
        for kind, ti in targets[:3]:
            add("bartizans", "corner turrets on corbels — a view from the corners", **({"keep": True} if kind == "keep" else {"tower": ti}),
                r=round(float(rng.uniform(1.2, 1.7)), 2))

    # ---------------- function: apse, residential windows, oriel, garderobe, wicket gate, water gate
    R = FK["sections"]
    for s in secs:
        chap = [b for b in s["behind"] if b[1] in R["chapel_bay"]["behind"]]
        if chap and s["east"] >= math.cos(math.radians(R["chapel_bay"]["facing_east_deg"])) and "chapel_bay" not in [f["kind"] for f in F]:
            bi, _, i = chap[0]; C["buildings"][bi]["apse_out"] = True
            add("chapel_bay", "a chapel at a wall facing east — the apse projects through the wall", wall=s["wall"], i=int(i), building=bi,
                sec=s["id"]); used[s["id"]].add("chapel_bay")
    liv = [s for s in secs if any(b[1] in R["palace_windows"]["behind"] for b in s["behind"]) and s["exposure"] in R["palace_windows"]["exposure"]
           and s["threat"] < 0.5]
    n_pw = R["palace_windows"].get("max_sections", 2) if wealth >= 0.3 else (1 if wealth >= 0.12 else 0)   # windows are a luxury (knob wealth)
    for s in sorted(liv, key=lambda q: (q["exposure"] != "cliff", q["threat"]))[:n_pw]:
        bi, kind, i = next(b for b in s["behind"] if b[1] in R["palace_windows"]["behind"]); b = C["buildings"][bi]
        add("palace_windows", f"behind the wall {kind}, outside {s['exposure']} — rare residential windows in the curtain (no stained glass)", wall=s["wall"],
            sec=s["id"], building=bi, floor=b["floor"], storeys=b["storeys"], tracery=False,
            ii=[j for j in s["ii"] if abs(j - i) <= int(b["L"] / 3)]); used[s["id"]].add("palace_windows")
    if liv and wealth >= 0.2:
        s = max(liv, key=lambda q: -q["threat"]); bi, kind, i = next(b for b in s["behind"] if b[1] in R["oriel"]["behind"]) \
            if any(b[1] in R["oriel"]["behind"] for b in s["behind"]) else (None, None, None)
        if bi is not None:
            b = C["buildings"][bi]
            add("oriel", f"oriel {kind} — on the safe side", wall=s["wall"], i=int(i), sec=s["id"], building=bi,
                z0=round(b["floor"] + st["storey"], 2), z1=round(b["eave"] - 0.4, 2), faces=int(rng.choice([3, 5])),
                w=round(float(rng.uniform(2.4, 3.4)) * st["k"] ** 0.4, 2)); used[s["id"]].add("oriel")
    gar = [s for s in secs if s["exposure"] in R["garderobe"]["exposure"] and s["outer"] and any(b[1] in R["garderobe"]["behind"] for b in s["behind"])]
    for s in gar[: int(rng.integers(R["garderobe"]["count"][0], R["garderobe"]["count"][1] + 1))]:
        bi, kind, i = next(b for b in s["behind"] if b[1] in R["garderobe"]["behind"])
        j = s["ii"][min(len(s["ii"]) - 2, max(1, s["ii"].index(i) + int(rng.integers(-3, 4)) if i in s["ii"] else len(s["ii"]) // 3))]
        j = free_i(s, j)
        if j is None: continue
        b = C["buildings"][bi]
        add("garderobe", f"at {kind}, outside {s['exposure']} — a garderobe on corbels", wall=s["wall"], i=int(j), sec=s["id"],
            z=round(b["floor"] + st["storey"] * (1 if b["storeys"] > 1 else 0) + 0.3, 2)); used[s["id"]].add("garderobe")
    if not gar:                                                                  # garderobe over the moat — at a dwelling on the ring
        cand = [s for s in secs if any(b[1] in R["garderobe"]["behind"] for b in s["behind"]) and C["ditches"] and s["outer"]]
        if cand:
            s = cand[0]; bi, kind, i = next(b for b in s["behind"] if b[1] in R["garderobe"]["behind"])
            add("garderobe", f"at {kind} — a garderobe over the moat", wall=s["wall"], i=int(i), sec=s["id"], z=round(C["buildings"][bi]["floor"] + 0.3, 2))
            used[s["id"]].add("garderobe")
    quiet = sorted([s for s in secs if s["gate_d"] >= R["postern"]["min_gate_dist"] and s["exposure"] in ("steep", "gentle", "water")
                    and s["outer"]], key=lambda q: (q["threat"], -q["gate_d"]))
    if quiet and free_i(quiet[0], quiet[0]["ii"][len(quiet[0]["ii"]) // 2]) is not None:
        s = quiet[0]; i = free_i(s, s["ii"][len(s["ii"]) // 2])
        add("postern", "wicket gate on a quiet side, far from the gate", wall=s["wall"], i=int(i), sec=s["id"]); used[s["id"]].add("postern")
    wat = [s for s in secs if s["exposure"] in R["water_gate"]["exposure"] and s["outer"]]
    if wat:
        s = wat[0]; add("water_gate", "a stretch by the water — a water gate", wall=s["wall"], i=int(s["ii"][len(s["ii"]) // 2]), sec=s["id"])
        used[s["id"]].add("water_gate")

    # ---------------- defense: machicolations or a gallery (one per stretch), covered walk, zwinger
    hm = R["machicolation"]; hh = R["hoarding"]; hc = R["covered_walk"]
    thr_secs = sorted([s for s in secs if (s["exposure"] in hm["exposure"] or (s["exposure"] == "lists" and s["threat"] > 0.3))
                       and not used[s["id"]] & {"palace_windows", "chapel_bay"} and not s["risers"]],     # stretches with ledges have their own merlons, no galleries
                      key=lambda q: -(q["threat"] + 0.6 * (q["gate_d"] < 35)))
    m_ok = era_ok(hm) and pick_culture(hm, cu, rng); h_ok = era_ok(hh) and pick_culture(hh, cu, rng)
    for k, s in enumerate(thr_secs):
        if m_ok and (k < max(1, round(len(thr_secs) * hm["share"])) or s["gate_d"] < 35):
            tiers = int(rng.integers(hm["tiers"][0], hm["tiers"][1] + 1))
            add("machicolation", "stretch toward the threat" + (", near the gate" if s["gate_d"] < 35 else "") + " — machicolations", wall=s["wall"], ii=s["ii"],
                sec=s["id"], tiers=tiers); used[s["id"]].add("machicolation")
        elif h_ok and k < max(1, round(len(thr_secs) * hh["share"])) + 1:
            add("hoarding", "stretch toward the threat — a hoarding gallery on beams", wall=s["wall"], ii=s["ii"], sec=s["id"],
                bay=round(float(rng.uniform(1.1, 1.5)), 2)); used[s["id"]].add("hoarding")
    h2 = R.get("two_level", {})
    if era_ok(h2) and pick_culture(h2, cu, rng):                                 # Pierrefonds: a covered walk over the machicolations
        for f in [f for f in F if f["kind"] == "machicolation"]:
            add("covered_walk", "two-tier defense (Pierrefonds): machicolations with a covered walk above them", wall=f["wall"], ii=f["ii"], sec=f["sec"],
                over_mach=True); used[f["sec"]].add("covered_walk")
    if pick_culture(hc, cu, rng):
        for s in secs:
            if not used[s["id"]] & {"machicolation", "hoarding"} and not s["risers"] and rng.random() < hc["share"] and s["L"] > 10:
                add("covered_walk", "covered walk along the wall", wall=s["wall"], ii=s["ii"], sec=s["id"]); used[s["id"]].add("covered_walk")
    hz = R["zwinger"]
    if era_ok(hz) and pick_culture(hz, cu, rng) and sp["scale"] in ("medium", "large", "royal") and not st.get("bastions"):
        zs = [s for s in thr_secs if s["L"] >= hz["min_len"] and s["closed"] and s["ward"] == C["gates"][0].get("ward", "inner")
              and not any(f.get("sec") == s["id"] and f["kind"] in ("postern",) for f in F)]
        for s in zs[: int(rng.integers(hz["count"][0], hz["count"][1] + 1))]:
            f = zwinger(C, ctx, s, hz)
            if f: F.append(f); used[s["id"]].add("zwinger")

    hpb = R.get("pinnacle_buttresses", {})                                        # fantasy Gothic: buttresses with pinnacles
    if hpb and pick_culture(hpb, cu, rng):
        for s in secs:
            if s["outer"] and s["L"] > 10 and rng.random() < hpb.get("share", 0.85) and not used[s["id"]] & {"machicolation", "hoarding", "covered_walk", "chapel_bay"}:
                add("pinnacle_buttresses", "fantasy Gothic: buttresses with pinnacles (Roriczer) — the wall soars upward", wall=s["wall"], ii=s["ii"],
                    sec=s["id"], spacing=hpb["spacing"]); used[s["id"]].add("pinnacle_buttresses")
    hl = R.get("arrow_loops", {"share": 0.7}); hp = R.get("putlogs", {"share": 0.35})
    for s in secs:                                                               # loopholes in the curtain — toward the threat, where there is no gallery or machicolations
        if small and s["threat"] > -0.2 and s["L"] > 9 and not used[s["id"]] & {"hoarding", "machicolation", "palace_windows"} and rng.random() < hl["share"]:
            add("arrow_loops", "stretch toward the threat without a gallery — loopholes in the curtain" + (" (cruciform)" if era in ("late", "future") else ""),
                wall=s["wall"], ii=s["ii"], sec=s["id"], cross=era in ("late", "future"), step=round(float(rng.uniform(3.0, 4.5)), 2))
            used[s["id"]].add("arrow_loops")
    for s in secs:                                                               # putlog holes — a trace of construction (rows)
        if small and s["L"] > 12 and rng.random() < hp["share"] * min(1.6, 2 * age):
            add("putlogs", "scaffold putlog holes in rows — a trace of construction", wall=s["wall"], ii=s["ii"], sec=s["id"],
                rows=int(rng.integers(2, 4)), step=round(float(rng.uniform(1.6, 2.4)), 2)); used[s["id"]].add("putlogs")

    # ---------------- structure: buttresses on a slope, stepped walkway, stairs to the wall
    hb = R["buttresses"]                                                         # retaining ones — on a slope with soil, not on rock
    deep = sorted([s for s in secs if s["exposure"] in ("steep", "gentle", "uphill") and s["depth"] >= hb["base_depth"] + st["wall_h"] * 0.4],
                  key=lambda q: -q["depth"])
    for s in deep[:3]:
        add("buttresses", f"foundation {s['depth']:.0f} m deep on a slope — retaining buttresses", wall=s["wall"], ii=s["ii"], sec=s["id"],
            spacing=hb["spacing"]); used[s["id"]].add("buttresses")
    for s in secs:
        if s["slope"] > 0.18 and not C["walls"][s["wall"]].get("steps"):          # a stairstep wall already has ledges
            add("stepped_walk", "steep stretch — a stepped walkway", wall=s["wall"], ii=s["ii"], sec=s["id"]); used[s["id"]].add("stepped_walk")
    for w_id in dict.fromkeys(s["ward"] for s in secs):                          # courts in the order of the stretches (not set order)
        ws = sorted([s for s in secs if s["ward"] == w_id and s["L"] > 12 and not s["behind"] and s["closed"] and 2.0 <= s["h_in"] <= 10.0],
                    key=lambda q: q["gate_d"])
        for s in ws[: int(rng.integers(*FK["sections"]["stairs"]["per_ward"]) ) + 1]:
            i = s["ii"][len(s["ii"]) // 2]
            add("stairs", "stair from the court to the wall", wall=s["wall"], i=int(i), sec=s["id"], left=bool(rng.random() < 0.5))
            used[s["id"]].add("stairs")

    # ---------------- history: a superstructure with filled-in merlons, a bricked-up gate
    hr = FK["history"]["raised"]
    if rng.random() < min(0.95, hr["prob"] * min(1.5, 2 * age)):
        core = [s for s in secs if s["ward"] == "inner" and s["closed"] and s["L"] > 12 and not used[s["id"]] & {"machicolation", "hoarding"}
                and not s["risers"]]                                             # superstructure — on a level walkway (do not break the stairstep)
        core = sorted(core, key=lambda q: -q["threat"])
        for s in core[: int(rng.integers(hr["count"][0], hr["count"][1] + 1))]:
            dz = float(rng.uniform(*hr["dz"])); w = C["walls"][s["wall"]]; walk = np.asarray(w["walk"])
            n = len(s["ii"]); tap = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / 2.0)
            old = walk[s["ii"]].copy(); walk[s["ii"]] = old + dz * tap; w["walk"] = walk.round(2).tolist()
            add("raised", f"wall raised by {dz:.1f} m — the old merlons are filled in (a rebuild)", wall=s["wall"], ii=s["ii"], sec=s["id"],
                old=old.round(2).tolist()); used[s["id"]].add("raised")
    if rng.random() < min(0.95, FK["history"]["blocked_gate"]["prob"] * min(1.6, 2 * age)):
        cand = [s for s in secs if s["ward"] == "inner" and s["L"] > 12 and not used[s["id"]] and s["gate_d"] > 25]
        if cand:
            s = max(cand, key=lambda q: q["threat"]); i = s["ii"][len(s["ii"]) // 2]
            add("blocked_gate", "the former gate is bricked up — the entrance was moved", wall=s["wall"], i=int(i), sec=s["id"]); used[s["id"]].add("blocked_gate")

    # ---------------- buildings: chimneys, dormers, half-timbering, stepped gables, ridge turret, tracery
    B = FK["buildings"]
    fw = rng.random() < B["fachwerk"]["cultures"].get(cu, 0.0); sg = rng.random() < B["stepped_gable"]["cultures"].get(cu, 0.0)
    for bi, b in enumerate(C["buildings"]):
        if b["kind"] in B["chimneys"]["kinds"] and b["roof"] != "flat":
            b["chimneys"] = int(rng.integers(B["chimneys"]["per"][0], B["chimneys"]["per"][1] + 1))
        if b["kind"] in B["dormers"]["kinds"] and b["roof"] in ("gable", "hip") and b.get("pitch", 0) >= B["dormers"]["min_pitch"]:
            b["dormers"] = True
        if fw and b["kind"] in B["fachwerk"]["kinds"] and b["storeys"] >= 2 and b["roof"] != "flat": b["fachwerk"] = 1
        if sg and b["roof"] == "gable" and b["kind"] not in ("chapel", "chapel_big"): b["stepped_gable"] = True
        if b["kind"] in B["roof_turret"]["kinds"] and b["roof"] != "flat": b["roof_turret"] = True
        if era in B["tracery"]["eras"] and b["kind"] in ("chapel", "chapel_big"): b["tracery"] = True       # stained glass is a luxury: the chapel
        tt = B.get("turrets", {})
        if b["kind"] in tt.get("kinds", []) and rng.random() < tt.get("cultures", {}).get(cu, 0.3) * (0.6 + wealth):
            b["turrets"] = {"n": int(rng.choice([2, 4])), "r": round(float(rng.uniform(1.0, 1.4)), 2)}   # turrets at the corners (Eltz)
    halls = [b for b in C["buildings"] if b["kind"] in ("palas", "palas_big", "hall") and era in B["tracery"]["eras"]]
    if halls and rng.random() < B["tracery"].get("hall_prob", 0.5) * min(1.6, 2 * wealth): halls[0]["tracery"] = "hall"   # and one hall — one tier toward the court
    hd = FK["sections"].get("dansker", {})
    if hd and rng.random() < hd.get("purposes", {}).get(sp["purpose"], hd.get("prob", 0.0)) * (0.5 + wealth):
        f = dansker(C, ctx, secs, used, hd)
        if f: F.append(f)
    hsb = T.get("sky_bridges", {})
    if hsb and pick_culture(hsb, cu, rng):
        F.extend(sky_bridges(C, ctx, hsb))
    roofscape(C, ctx, F)
    C["sections"] = [{k: (v.round(3).tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k != "behind"}
                     | {"behind": [b[1] for b in s["behind"]], "features": sorted(used[s["id"]])} for s in secs]
    return F


def roofscape(C, ctx, F):
    """Tower roofs: a roof is expensive, in classic castles ≤ 65 % of the towers are roofed; one tall roof is a
    landmark, the rest are low (≤ 0.38 of the tower height). Roofs go at the corners and by dwellings; lower
    down, toward the threat — open platforms with merlons."""
    sp, rng, thr = ctx["spec"], ctx["rng"], ctx["thr"]; RS = FK["roofs"]
    wealth = ctx["st"].get("knobs", {}).get("wealth", 0.5)
    share = min(RS["max_share"], RS["share"].get(sp["culture"], 0.5) * (0.6 + 0.8 * wealth)); cu = sp["culture"]   # roofs — by wealth
    if C.get("keep") and C["keep"].get("stages"):                                # the tiered main tower is a landmark by its silhouette
        F.append({"kind": "landmark_spire", "why": "main tower in tiers — the vertical of the castle (" + C["keep"].get("why", "") + ")", "keep": True})
    towers = [t for t in C["towers"] if t["role"] in ("corner", "flank", "gate", "barbican", "zwinger") and not t.get("stages")]   # tiered ones have their own roof
    if not towers: return
    living = [np.asarray(b["c"]) for b in C["buildings"] if b["kind"] in LIVING]
    def rank(t):
        p = np.asarray(t["xy"]); d_liv = min([np.hypot(*(p - q)) for q in living] or [60.0])
        o = np.array([t["dir"][1], -t["dir"][0]]) if t.get("dir") else np.zeros(2)
        return {"corner": 0.0, "gate": 0.4, "flank": 0.8, "barbican": 1.2, "zwinger": 1.5}[t["role"]] + d_liv / 40 + 0.5 * float(o @ thr) + 0.3 * rng.random()
    order = sorted(range(len(towers)), key=lambda k: rank(towers[k])); n_roof = int(round(share * len(towers)))
    roofed = set(order[:n_roof])
    for k, t in enumerate(towers):                                              # a pair of gate towers — alike
        if t["role"] == "gate" and any(towers[j]["role"] == "gate" and j in roofed for j in range(len(towers)) if abs(j - k) == 1): roofed.add(k)
    cap_n = int(math.floor(RS["max_share"] * len(towers) + 1e-9))
    for k in sorted(roofed, key=lambda k: -rank(towers[k])):                     # over the norm — remove from the least important ones
        if len(roofed) <= cap_n: break
        if towers[k]["role"] == "gate":
            pair = [j for j in roofed if towers[j]["role"] == "gate" and abs(j - k) <= 1]
            if len(roofed) - len(pair) >= 0: roofed -= set(pair)
        else: roofed.discard(k)
    kind_by_plan = RS["kind"].get(cu, {})
    keep = C.get("keep"); land = None
    if keep and keep.get("stages"): land = None
    elif keep and keep.get("roof") not in (None, "flat") and keep["kind"] != "spire" and not keep.get("butterfass"): land = ("keep", keep)
    elif roofed:
        k0 = max(roofed, key=lambda k: towers[k]["top"] - towers[k]["base"]); land = ("tower", towers[k0])
    for k, t in enumerate(towers):
        body = float(t["top"]) - float(t["base"])
        if k not in roofed:
            t["roof"] = "flat"; continue
        rk = kind_by_plan.get(t["plan"], "cone" if t["plan"] in ("round", "D") else "pyramid"); t["roof"] = rk
        pitch = math.radians(np.clip(RS["pitch"].get(cu, 52), 36, 62)); cap = RS["cap"]
        t["roof_h"] = round(float(min(t["r"] * math.tan(pitch), cap * body)), 2)
    if land is not None:
        _, t = land; body = float(t["top"]) - float(t["base"])
        if t.get("roof") in (None, "flat"): t["roof"] = "cone" if t["plan"] in ("round", "D") else "pyramid"
        t["roof_h"] = round(float(min(t["r"] * math.tan(math.radians(np.clip(RS["pitch"].get(cu, 52) + 8, 45, 70))), RS["cap_landmark"] * body)), 2)
        t["landmark"] = True
        F.append({"kind": "landmark_roof", "why": "one tall roof is a silhouette landmark; the rest are low",
                  **({"keep": True} if land[0] == "keep" else {"tower": C["towers"].index(t)})})
    C["roof_share"] = round(sum(1 for t in towers if t.get("roof") not in (None, "flat")) / len(towers), 2)


def zwinger(C, ctx, s, hz):
    """Zwinger: a low wall in front of a stretch at hz.offset m, its ends turn toward the main wall, with turrets
    at the ends."""
    st, G2 = ctx["st"], ctx["G2"]; w = C["walls"][s["wall"]]; P = np.asarray(w["pts"])[s["ii"]]
    N = normals(np.asarray(w["pts"]), True)[s["ii"]]; t = st["wall_t"]; off = t / 2 + hz["offset"] * st["k"] ** 0.5
    Z = P + N * off
    a0, a1 = P[0] + N[0] * (t / 2 + 0.5), P[-1] + N[-1] * (t / 2 + 0.5)
    L = np.vstack([a0, Z[1:-1] if len(Z) > 4 else Z, a1])
    if len(L) < 3: return None
    zst = dict(st, wall_h=hz["height"] * st["k"] ** 0.6, wall_t=max(1.4, t * 0.55))
    pts, walk, base = ctx["wall_profile"](G2, L, 0, len(L) - 1, zst, None, 1.0)
    C["walls"].append({"ward": s["ward"], "kind": "zwinger", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                       "base": base.round(2).tolist(), "t": round(zst["wall_t"], 2), "closed": False})
    for q in (Z[0], Z[-1]):
        zz = float(G2.height(*q)); C["towers"].append({"xy": q.round(3).tolist(), "on": q.round(3).tolist(), "plan": "round", "r": round(2.3 * st["k"] ** 0.5, 2),
                                                        "up": 2.5, "roof": "flat", "dir": None, "role": "zwinger", "ward": s["ward"], "lean": 0.0,
                                                        "base": round(zz - 1.0, 2), "walk": round(zz + zst["wall_h"], 2), "top": round(zz + zst["wall_h"] + 2.5, 2)})
    return {"kind": "zwinger", "why": "in front of the stretch toward the threat — a zwinger: a low wall, a field of fire", "wall": s["wall"], "sec": s["id"],
            "zwall": len(C["walls"]) - 1}


def barbican(C, ctx):
    """Barbican beyond the moat: walls on the sides of the neck and a front wall with a gate, two turrets at the
    corners."""
    st, G2 = ctx["st"], ctx["G2"]; g = C["gates"][0]; gp = np.asarray(g["xy"]); o = np.asarray(g["out"]); d = np.asarray(g["dir"])
    span = 0.0
    if C["bridges"]: span = float(np.hypot(*(np.asarray(C["bridges"][0]["pts"][1]) - gp)))
    base = gp + o * (max(span, st["wall_t"] / 2 + 2) + 1.0)
    Wb = g["w"] + 2 * (2.2 * st["k"] ** 0.5 + 2.5); Db = 11.0 * st["k"] ** 0.5
    q = [base - d * Wb / 2, base - d * Wb / 2 + o * Db, base + d * Wb / 2 + o * Db, base + d * Wb / 2]
    xs = np.linspace(min(p[0] for p in q), max(p[0] for p in q), 8); ys = np.linspace(min(p[1] for p in q), max(p[1] for p in q), 8)
    X, Y = np.meshgrid(xs, ys)
    if float(np.percentile(G2.slope(X, Y), 80)) > 24 or float(G2.water(X, Y).max()) > 0.3: return None
    if any(D for D in C["ditches"] if min(np.hypot(*(np.asarray(D["pts"]) - (base + o * Db / 2)).T)) < D["w"] / 2 + 2): return None
    L = np.array(q)
    zst = dict(st, wall_h=st["wall_h"] * 0.65, wall_t=max(1.6, st["wall_t"] * 0.7))
    pts, walk, base_z = ctx["wall_profile"](G2, L, 0, 3, zst, None, 1.0)
    C["walls"].append({"ward": "barbican", "kind": "barbican", "pts": pts.round(3).tolist(), "walk": walk.round(2).tolist(),
                       "base": base_z.round(2).tolist(), "t": round(zst["wall_t"], 2), "closed": False})
    fg = base + o * Db
    C["gates"].append({"xy": fg.round(3).tolist(), "edge": -1, "dir": d.round(4).tolist(), "out": o.round(4).tolist(), "ward": "barbican",
                       "kind": "simple", "w": g["w"], "h": g["h"], "z": round(float(G2.height(*fg)), 2), "walk": round(float(np.max(walk)), 2)})
    for p in (q[1], q[2]):
        zz = float(G2.height(*p)); r = 2.4 * st["k"] ** 0.5
        C["towers"].append({"xy": p.round(3).tolist(), "on": p.round(3).tolist(), "plan": "round", "r": round(r, 2), "up": 3.0, "roof": "flat",
                            "dir": d.round(4).tolist(), "role": "barbican", "ward": "barbican", "lean": 0.0,
                            "base": round(zz - 1.0, 2), "walk": round(zz + zst["wall_h"], 2), "top": round(zz + zst["wall_h"] + 3.0, 2)})
    return {"kind": "barbican", "why": "in front of the main gate beyond the moat — a barbican: the approach neck", "gate": 0, "wall": len(C["walls"]) - 1}


def r_at(t, z):
    """Tower radius at height z (for a tiered tower — the radius of the tier)."""
    for sg in t.get("stages") or []:
        if float(sg["z0"]) - 1e-6 <= z <= float(sg["z1"]) + 1e-6: return float(sg["r"])
    return float(t["r"])


def sky_bridges(C, ctx, hsb):
    """Bridges between the tops of tall towers (Ravenloft, Lothric): pairs of tiered towers (with the main tower
    first) at the span distance hsb.span between the faces; the bridge is below the top tier of the smaller
    tower, above the ridge of the buildings and the wall walkway, and does not touch other towers."""
    rng = ctx["rng"]; T = ([("keep", C["keep"])] if C.get("keep") and C["keep"].get("stages") else []) + \
        [(i, t) for i, t in enumerate(C["towers"]) if t.get("stages")]
    cand = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            (ia, a), (ib, b) = T[i], T[j]; pa, pb = np.asarray(a["xy"], float), np.asarray(b["xy"], float); dd = float(np.hypot(*(pb - pa)))
            top = min(float(a["stages"][-1]["z0"]), float(b["stages"][-1]["z0"])) - 1.0     # under the top tier of the smaller one
            z = top - 2.0; ra, rb = r_at(a, z), r_at(b, z); span = dd - ra - rb
            if not hsb["span"][0] <= span <= hsb["span"][1]: continue
            u = (pb - pa) / max(dd, 1e-9); line = pa + u * np.linspace(ra + 0.5, dd - rb - 0.5, 24)[:, None]; ok = True
            for bd in C["buildings"]:
                R_ = np.asarray(bd["poly"], float); ridge = float(bd["eave"]) + float(bd["D"]) * 0.6 * math.tan(math.radians(max(float(bd.get("pitch", 45)), 20)))
                if z < ridge + 2.0 and ctx["inside"](R_, line[:, 0], line[:, 1]).any(): ok = False; break
            for w in C["walls"] if ok else []:
                P = np.asarray(w["pts"], float); dmin = np.hypot(P[:, None, 0] - line[None, :, 0], P[:, None, 1] - line[None, :, 1])
                k_ = np.unravel_index(int(np.argmin(dmin)), dmin.shape)
                if dmin[k_] < 2.0 and z < float(w["walk"][k_[0]]) + 3.0: ok = False; break
            for k_, (ic, c_) in enumerate(T + [(None, t_) for t_ in C["towers"] if not t_.get("stages")]):
                if not ok or c_ is a or c_ is b: continue
                if np.hypot(line[:, 0] - c_["xy"][0], line[:, 1] - c_["xy"][1]).min() < float(c_["r"]) + 1.5 and z < float(c_["top"]) + 2: ok = False
            if ok: cand.append(((ia == "keep") * 2.0 + span / 40 + rng.random() * 0.3, ia, ib, pa + u * ra, pb - u * rb, z, span))
    cand.sort(key=lambda q: -q[0]); out = []; used_t = set(); n_max = int(rng.integers(hsb["count"][0], hsb["count"][1] + 1))
    for _, ia, ib, p0, p1, z, span in cand:
        if len(out) >= n_max or ia in used_t or ib in used_t: continue
        used_t |= {ia, ib}
        out.append({"kind": "sky_bridge", "why": f"bridge between tower tops at a height of {z - float(C['towers'][ib]['base'] if ib != 'keep' else 0):.0f} m — "
                    "a walkway over the court (Ravenloft)", "a": ia, "b": ib, "p0": p0.round(3).tolist(), "p1": p1.round(3).tolist(), "z": round(z, 2),
                    "w": 2.8, "span": round(span, 1)})
    return out


def dansker(C, ctx, secs, used, hd):
    """Projecting tower on an arcade (Danzker, Marienburg: a latrine tower 60 m from the wall on five arches): a
    stretch of the outer wall by a dwelling, not toward the threat; the tower stands farther along the normal,
    where the ground is lowest or there is water (≤ 2.5 m); a gallery on pointed arches at walkway height; the
    path does not touch towers, walls or the road."""
    st, G2, inside = ctx["st"], ctx["G2"], ctx["inside"]; t_w = st["wall_t"]; k = st["k"]
    road = np.asarray(C.get("road") or [[1e9, 1e9, 0]], float)[:, :2]; best = None
    for s in secs:
        if not s["outer"] or s["threat"] > 0.3 or s["L"] < 12 or used[s["id"]] & {"palace_windows", "chapel_bay", "oriel", "garderobe"}: continue
        if s["exposure"] not in hd.get("exposure", ["water", "steep", "cliff", "gentle"]): continue
        w = C["walls"][s["wall"]]; P = np.asarray(w["pts"], float); N = normals(P, bool(w["closed"]))
        i = s["ii"][len(s["ii"]) // 2]; n = N[i]; p0 = P[i] + n * (t_w / 2)
        liv = any(b[1] in hd.get("behind", []) for b in s["behind"])
        for d in np.arange(hd["dist"][0], hd["dist"][1] + 0.1, 2.0) * k ** 0.5:
            p1 = p0 + n * d; zg = float(G2.height(*p1)); wat = float(G2.water(*p1))
            if wat > 2.5: break
            line = p0 + n * np.linspace(3.0, d + 6, 24)[:, None]
            if any(inside(np.asarray(wd["poly"]), q[0], q[1]) for wd in C["wards"] for q in line[::4]): break
            if min((np.hypot(*(np.asarray(t_["xy"]) - q)) - t_["r"] for t_ in C["towers"] for q in line[::3]), default=99) < 3.0: break
            if min(np.hypot(road[:, 0] - q[0], road[:, 1] - q[1]).min() for q in line[::3]) < 6.0: break
            deck = float(w["walk"][i]) - 1.2; side = float(np.clip(10.5 * k ** 0.5, 8.0, 14.0))
            zf = G2.height(p1[0] + np.array([-1, 1, 1, -1]) * side / 2, p1[1] + np.array([-1, -1, 1, 1]) * side / 2)
            hgt = deck - float(zf.min())                                          # by the bottom of the tower's footprint — not on a bank cliff
            if hgt > 26 or hgt < 6 or zg - float(zf.min()) > 4.0 or float(G2.water(*p1)) > 2.5: continue
            sc = hgt / 10 + 1.5 * (wat > 0.2) + 1.0 * liv + d / 60 - 0.5 * s["threat"]
            if best is None or sc > best[0]: best = (sc, s, i, p0, p1, deck, zg, wat, d, liv)
    if best is None: return None
    _, s, i, p0, p1, deck, zg, wat, d, liv = best
    side = round(float(np.clip(10.5 * k ** 0.5, 8.0, 14.0)), 2); n_a = max(2, int(round(d / (9.0 * k ** 0.4))))
    zz = G2.height(p1[0] + np.array([-1, 1, 1, -1]) * side / 2, p1[1] + np.array([-1, -1, 1, 1]) * side / 2)
    used[s["id"]].add("dansker")
    return {"kind": "dansker", "why": "projecting tower on an arcade (Danzker): by a dwelling, not toward the threat, — above " + ("water" if wat > 0.2 else "hillside"),
            "wall": s["wall"], "i": int(i), "sec": s["id"], "p0": p0.round(3).tolist(), "p1": p1.round(3).tolist(), "deck": round(deck, 2),
            "w": round(3.4 * k ** 0.4, 2), "arches": n_a, "side": side, "base": round(float(zz.min()) - 1.0, 2), "top": round(deck + 6.5 * k ** 0.5, 2),
            "roof": "flat" if st["flat_roofs"] else "pyramid", "len": round(float(d), 1)}


def checks(C, ctx):
    """Logic of the elements and uniqueness of the stretches."""
    out = []; F = C.get("features", []); secs = C.get("sections", [])
    def chk(what, ok, **kw): out.append(dict({"what": what, "ok": bool(ok)}, **kw))
    sig = [(tuple(s["features"]), round(s["h_out"] / 3), s["exposure"]) for s in secs if s["L"] >= 10]
    uniq = len(set(sig)) / max(1, len(sig))
    chk("uniqueness of wall stretches ≥ 0.6 (different sets of elements, height, side)", uniq >= 0.6, frac=round(uniq, 2), n_plots=len(sig))
    land = {f["kind"] for f in F} & {"butterfass", "prow", "oriel", "chapel_bay", "barbican", "bartizans", "zwinger", "drawbridge", "raised", "landmark_roof",
                                     "landmark_spire", "dansker"}
    rs = C.get("roof_share")
    if rs is not None: chk("roofs: ≤ 65 % of the towers are roofed (a roof is expensive)", rs <= FK["roofs"]["max_share"] + 1e-6, frac=rs)
    chk("landmarks ≥ 2", len(land) >= 2, present=sorted(land))
    byid = {s["id"]: s for s in secs}
    for f in F:
        s = byid.get(f.get("sec"))
        if s is None: continue
        if f["kind"] == "garderobe": chk("garderobe — over a cliff, a steep slope, water or a moat, by a dwelling", s["exposure"] in ("cliff", "steep", "water") or bool(C["ditches"]), side=s["exposure"])
        if f["kind"] in ("palace_windows", "oriel"): chk(f"{f['kind']}: not toward the threat", s["threat"] < 0.5, threat=round(s["threat"], 2))
        if f["kind"] == "chapel_bay": chk("chapel apse — facing east (±70°)", s["east"] >= math.cos(math.radians(70)), east=round(s["east"], 2))
        if f["kind"] == "postern": chk("wicket gate — no closer than 40 m to the gate", s["gate_d"] >= 40, m=round(s["gate_d"]))
        if f["kind"] in ("machicolation", "hoarding"):
            chk(f"{f['kind']}: toward the threat or near the gate", s["exposure"] in ("gentle", "uphill") or (s["exposure"] == "lists" and s["threat"] > 0.3) or s["gate_d"] < 35,
                side=s["exposure"])
        if f["kind"] in ("postern", "water_gate", "garderobe"): chk(f"{f['kind']}: on the outer circuit", s.get("outer", True))
    return out
