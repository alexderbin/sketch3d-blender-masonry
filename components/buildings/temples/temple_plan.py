"""The plan of a basilica temple: the form language is the genes of a family (style/style_genes.py,
style/styles.yaml), the biography is the construction campaigns, an unfinished state, a special element, additions, a
condition (temples/temple_story.py); a module per part by its campaign → wall tiers, windows under the vault arcs, parts,
tiered towers, buttresses and flying buttresses; the site (the axis to the east, leveling, an undercroft, a deflection of
the choir axis by the ground) → the decision log and the checks. Centric temples and ziggurats are edifices
(edifices/edifice_plan.py). The build is temples/bl_temple.py.

.venv/bin/python -m components.buildings.temples.temple_plan <place> [style=family|a+b|a:0.6,b:0.4] [seed=1] [at=x,y] [out=folder]
  [history= wealth= ruin= irregular=] [gene=value …]   → <out>/temple.json (+ relief.npz, location.json), __RESULT__
Coordinates are local: u — along the axis from west to east (0 — the axis of the west wall), v — across (+ toward the
north), z — above the cathedral floor (the floor is frame.floor above the platform; for a hive — on top of the podium)."""
import json, math, os, shutil, sys

import numpy as np

from components.buildings import arch_parts                                                              # noqa: E402
ROOT = arch_parts.ROOT
from components.buildings.siting import castle_site as CS                                                       # noqa: E402
from components.buildings.style import style_genes as TS                                                       # noqa: E402
from components.buildings.temples import temple_space as TSP                                                     # noqa: E402
from components.buildings.temples import temple_story as TST                                                     # noqa: E402
from components.buildings.base.arch_plan import CROWN_HALF, CROWN_SHARE, irng, tower_plan, uni           # noqa: E402,F401  (shared with the edifices — in the base)
from core.surface.ground import Ground                                                      # noqa: E402

TK = TST.TK                                                                    # temples knowledge (temples/temples.yaml)
EAST_RU = {"apse": "apse", "chevet": "ambulatory with chapels", "flat": "flat", "polygon": "polygonal apse"}
VR = {"round": 0.5, "pointed": 0.62, "lancet": 0.78, "ogee": 0.62, "horseshoe": 0.5, "parabolic": 0.75, "stepped": 0.5, "flat": 0.0}
WR = {"round": 0.5, "pointed": 0.75, "lancet": 1.1, "ogee": 0.9, "horseshoe": 0.87, "parabolic": 1.15, "stepped": 0.5, "flat": 0.0}


def vrise(arch, vault, span):
    """Vault rise: flat — 0; otherwise by the arch kind (semicircular 0.5 of the span, pointed 0.62, lancet 0.78…)."""
    return 0.0 if vault == "flat" else span * VR.get(arch, 0.62)


def arch_z(s, h, x):
    """The height of an arc (span s, rise h) above the springing at a point x from the middle: by compass, two centers on the
    springing line."""
    if h <= 1e-6: return 0.0
    r = (h * h + s * s / 4) / s; xc = r - s / 2
    return math.sqrt(max(0.0, r * r - (abs(x) + xc) ** 2))


def win_arch(kind, w): return w * WR.get(kind, 0.75)


def fit_window(win, x0, kind, z_spr, R, span, gap, flat=None):
    """A window under the wall arc of a vault: the window arc (center x0 from the bay middle) is everywhere lower than the vault
    arc by the clearance gap (for a flat ceiling and a barrel vault — lower by flat); otherwise the window is shortened from
    the top. → the window and the margin (m)."""
    w = win["w"]; ra = min(win_arch(kind, w), win["h"] * 0.9); zs = win["z0"] + win["h"] - ra; worst = 1e9
    for x in np.linspace(-w / 2, w / 2, 11):
        top = zs + (arch_z(w, ra, x) if ra > 1e-6 else 0.0)
        lim = flat - gap if flat is not None else z_spr + arch_z(span, R, x0 + x) - gap
        worst = min(worst, lim - top)
    if worst < 0: win = dict(win, h=round(max(0.5, win["h"] + worst), 2))
    return win, round(worst, 2)


def window(z_lo, z_hi, w, hf, kind):
    """A window in the zone [z_lo, z_hi]: width w, height a fraction of the zone (not lower than the arch + 0.6 m), pressed to the top."""
    h = max(hf * (z_hi - z_lo), min(win_arch(kind, w) + 0.6, z_hi - z_lo))
    return {"w": round(w, 2), "z0": round(z_hi - h, 2), "h": round(h, 2)}


def module_of(S_, W, A, k, kq, rng):
    """A part's module by the family of its campaign (the span W and the side aisle A are shared — the same foundations):
    heights, wall tiers, windows under the vault arcs, the triforium, the structure."""
    H = W * uni(S_["HW"], rng); B = S_["B"] * W; hall = S_["HA"] >= 0.99; Ha = H * S_["HA"]
    t = S_["t"] * (k if k > 1 else 1.0); tc = t if hall else round(t * 0.8, 2)
    arch = S_["arch"]; vault = S_["vault"]; roof = S_.get("roof", "gable")
    pier = S_["pier"] * W; ab = int(S_["aisle_bays"]); bay_a = B / ab
    rise = vrise(arch, vault, W); spring = H - rise; ra = vrise(arch, vault, A); spring_a = Ha - ra
    ext = 0.06 * W + 0.8 * kq; He = H + ext; Ea = He if hall else Ha + 0.06 * A + 0.8 * kq; ov = 0.5 * kq
    pa = float(S_.get("aisle_pitch", 20)) if roof == "gable" else 0.0; pitch = float(S_["roof_pitch"]) if roof == "gable" else 0.0
    z_lean = He if hall else Ea + (A + t / 2 - tc / 2 + ov) * math.tan(math.radians(pa)) + (0.0 if roof == "gable" else 1.2 * kq)
    el = S_["elevation"]
    if hall: z_arc = z_tri = z_tri0 = H
    else:
        z_arc = Ha - 0.3 * kq                                                    # arcade — the wall arc of the side-aisle vault
        f_t = el["tribune"] / max(el["tribune"] + el["clerestory"], 1e-6); z_tri0 = z_arc + (spring - z_arc) * f_t
        z_tri = max(z_tri0, z_lean + 0.3 * kq)                                   # the triforium hides the side-aisle roof
    half = (W / 2 + A + t / 2 + ov) if hall else (W / 2 + tc / 2 + ov)
    ridge = He + half * math.tan(math.radians(pitch)) if roof == "gable" else He + 1.3 * kq
    wk = arch; Wn = S_["win"]; nf = int(S_["buttress"]["flyers"]); bw = S_["buttress"]["w"] * W; fits = []
    flat_n = H - 0.5 * kq if vault == "flat" else (spring - 0.3 * kq if vault == "barrel" else None)
    flat_a = Ha - 0.5 * kq if vault == "flat" else (spring_a - 0.3 * kq if vault == "barrel" else None)
    cl = None
    if not hall and Wn["clerestory"][0] > 0:
        w_c = min(Wn["clerestory"][0] * bay_a, bay_a - 0.8 * pier - 0.3 * kq)
        z_hi = flat_n if flat_n is not None else spring + rise * (0.86 if arch not in ("round", "horseshoe", "stepped") else 0.75)
        if z_hi - (z_tri + 0.4 * kq) > 1.5 * kq:
            cl = window(z_tri + 0.4 * kq, z_hi, w_c, Wn["clerestory"][1], wk)
            for x0 in [(j + 0.5) * bay_a - B / 2 for j in range(ab)]:
                cl, f_ = fit_window(cl, x0, wk, spring, rise, B, 0.35 * kq, flat_n); fits.append(("clerestory", f_))
    w_a = min(Wn["aisle"][0] * (bay_a - bw), bay_a - pier - 0.4 * kq) if not hall else min(Wn["aisle"][0] * B, B - bw - 0.4 * kq)
    ai = window(2.0 * kq, (flat_a if flat_a is not None else spring_a + ra * 0.86), w_a, Wn["aisle"][1], wk)
    ai, f_ = fit_window(ai, 0.0, wk, spring_a, ra, bay_a if not hall else B, 0.3 * kq, flat_a); fits.append(("side aisle", f_))
    choir_win = None
    if not S_["choir_aisles"]:                                                   # a choir without aisles: windows of two tiers (Romanesque) or one tall one
        if hall: choir_win = [window(2.0 * kq, spring + rise * 0.86, min(Wn["aisle"][0] * B, B - bw - 0.4 * kq), Wn["aisle"][1], wk)]
        else: choir_win = [window(2.6 * kq, Ha, min(0.3 * B, 2.4 * kq), 0.55, wk), cl] if cl else [ai]
        choir_win[-1], f_ = fit_window(choir_win[-1], 0.0, wk, spring, rise, B, 0.35 * kq, flat_n); fits.append(("choir", f_))
    tri = None
    if not hall and S_["tri_n"] and z_tri - z_arc > 1.6 * kq:
        tri = {"n": int(S_["tri_n"]), "z0": round(z_arc + 0.5 * kq, 2), "h": round(z_tri - z_arc - 0.8 * kq, 2),
               "w": round((bay_a - pier) / int(S_["tri_n"]) * 0.62, 2)}
    bt = S_["buttress"]
    return {"family": S_["_name"], "W": round(W, 2), "H": round(H, 2), "A": round(A, 2), "Ha": round(Ha, 2), "B": round(B, 2), "t": round(t, 2),
            "tc": round(tc, 2), "pier": round(pier, 2), "pier_kind": S_.get("pier_kind", "compound"), "rise": round(rise, 2), "spring": round(spring, 2),
            "ra": round(ra, 2), "spring_a": round(spring_a, 2), "z_arc": round(z_arc, 2), "z_tri": round(z_tri, 2), "z_lean": round(z_lean, 2),
            "Ea": round(Ea, 2), "He": round(He, 2), "ridge": round(ridge, 2), "ov": round(ov, 2), "k": round(k, 2), "kq": round(kq, 3),
            "arch": arch, "vault": vault, "roof": roof, "ab": ab, "hall": hall, "roof_pitch": pitch, "aisle_pitch": pa,
            "win": {"cl": cl, "ai": ai, "choir": choir_win, "tracery": Wn["tracery"]}, "tri": tri,
            "buttress": {"depth": round(bt["depth"] * W, 2), "w": round(bw, 2), "flyers": nf, "kind": bt.get("kind", "flyer"), "th": round(0.075 * W, 2)},
            "colors": S_["colors"], "spikes": float(S_.get("spikes", 0.0)), "organic": float(S_.get("organic", 0.0)),
            "fortified": float(S_.get("fortified", 0.0)), "industrial": float(S_.get("industrial", 0.0)), "pinnacle_k": float(S_.get("pinnacle_k", 1.0)),
            "pinnacles": bool(S_["tower"]["pinnacles"]), "tower_open": S_["tower"]["open"], "fits": fits, "z_tri0": round(z_tri0, 2)}


def stump(tw, level, crane):
    """An unfinished tower: the masonry stops at a fraction of the base, a temporary low roof; a crane on the stump (Cologne)."""
    zt = tw["stages"][0]["z0"] + (tw["stages"][0]["z1"] - tw["stages"][0]["z0"]) * level
    if level > 0.9 and len(tw["stages"]) > 1:                                    # reached the first tier of the bell tower
        s1 = dict(tw["stages"][1]); s1["z1"] = round(s1["z0"] + (s1["z1"] - s1["z0"]) * 0.6, 2); st = [tw["stages"][0], s1]
    else: st = [dict(tw["stages"][0], z1=round(zt, 2))]
    return dict(tw, stages=st, spire={"kind": "stump", "h": round(st[-1]["half"] * 0.3, 2)}, unfinished=True, crane=crane,
                h=round(st[-1]["z1"], 1), why=tw["why"] + f"; unfinished — a stump of {st[-1]['z1']:.0f} m" + (", crane" if crane else ""))


def basilica(S, Sp, St, seed, rng, k, kq, W, A, note):
    """The parts, towers, crossing, buttresses and additions of a basilica — by the campaign modules (a style per part)."""
    pf = St["part_family"]; over = S.get("_over", {})
    fam = {}; mods = {}; mkey = {}
    for p in ("nave", "transept", "east", "west"):
        spec = pf.get(p, S["_name"])
        if spec not in fam:
            fam[spec] = Sp if spec == S["_name"] else TS.pick(TS.resolve(spec, over), irng(seed, spec))
            key = f"m{len(mods)}"; mods[key] = module_of(fam[spec], W, A, k, kq, irng(seed, "mod:" + spec)); mkey[spec] = key
    F = {p: fam[pf.get(p, S["_name"])] for p in ("nave", "transept", "east", "west")}
    MK = {p: mkey[pf.get(p, S["_name"])] for p in ("nave", "transept", "east", "west")}
    Mn, Mt, Me, Mw = (mods[MK[p]] for p in ("nave", "transept", "east", "west"))
    sig = (St["signature"] or {}).get("kind"); unf = St["unfinished"]; state = St["state"]
    for key, M in mods.items():
        note("module", f"{key}: {M['family']}", f"span {W:.1f} m, vault {M['H']:.1f} m (×{M['H'] / W:.2f}), arch {M['arch']}, vault {M['vault']}, "
             f"side aisle {M['Ha']:.1f} m, bay {M['B']:.1f} m; arcade up to {M['z_arc']:.1f}, triforium up to {M['z_tri']:.1f}, springing {M['spring']:.1f}, "
             f"ridge {M['ridge']:.0f} m", "arcade — the wall arc of the side-aisle vault; triforium — at the side-aisle roof; windows — under the arcs of the vaults; "
             "the span and the side aisle are shared — the same foundations, the heights and shapes follow the campaign's era", "style=, history=")
    # ---- parts along the axis
    nb = int(rng.integers(F["nave"]["nave_bays"][0], F["nave"]["nave_bays"][1] + 1))
    if unf and unf["kind"] == "nave": nb = unf["nave_bays"]
    mb = int(rng.integers(F["east"]["choir_bays"][0], F["east"]["choir_bays"][1] + 1))
    kb = int(rng.integers(Sp["transept"][0], Sp["transept"][1] + 1))
    has_x = kb > 0 or F["transept"]["crossing"]["kind"] != "none"
    Bn, Bc, Bt = Mn["B"], Me["B"], Mt["B"]
    u_n1 = nb * Bn; u_x1 = u_n1 + (W if has_x else 0.0); u_c1 = u_x1 + mb * Bc
    ch_ais = bool(Sp["choir_aisles"])
    parts = {"nave": {"u0": 0.0, "u1": round(u_n1, 2), "bays": nb, "mod": MK["nave"]},
             "crossing": {"u0": round(u_n1, 2), "u1": round(u_x1, 2), "mod": MK["transept"]} if has_x else None,
             "choir": {"u0": round(u_x1, 2), "u1": round(u_c1, 2), "bays": mb, "aisles": ch_ais, "mod": MK["east"], "bend": 0.0},
             "transept": {"bays": kb, "v1": round(W / 2 + kb * Bt, 2), "mod": MK["transept"]} if kb else None,
             "west": {"mod": MK["west"]}}
    east = F["east"]["east"]; tE = Me["t"]
    if east == "chevet" and not ch_ais: ch_ais = True; parts["choir"]["aisles"] = True   # the ambulatory continues the choir aisles: without them the ends of the ambulatory are open
    nch = int(rng.choice([c for c in range(F["east"]["chapels"][0], F["east"]["chapels"][1] + 1) if c % 2 == 1] or [0])) if F["east"]["chapels"][1] else 0
    ea = {"kind": east, "u": round(u_c1, 2), "r": round(W / 2, 2), "mod": MK["east"]}
    if east == "chevet":
        n = max(3, nch); r2 = W / 2 + A; ch2 = 2 * r2 * math.sin(math.pi / (2 * n)); rc = min(0.5 * ch2 - 0.25 * Me["buttress"]["w"], 0.62 * A); s_ = 0.35 * rc
        ea.update(n=n, r2=round(r2, 2), chapels=nch, rc=round(rc, 2), straight=round(s_, 2)); u_end = u_c1 + r2 + tE / 2 + (s_ + rc + tE / 2 if nch else 0.0)
    elif east == "apse": ea.update(n=11, top=round(Me["spring"] + 0.6 * Me["rise"] + 0.5 * kq, 2)); u_end = u_c1 + W / 2 + tE / 2
    elif east == "polygon": ea.update(n=4); u_end = u_c1 + W / 2 + tE / 2
    else:
        lady = 2 * Bc if "gothic_en" in F["east"]["_name"] else 0.0
        ea.update(n=0, lady=round(lady, 2), lady_w=round(0.72 * W, 2), lady_h=round(Me["Ha"] * 1.15, 2)); u_end = u_c1 + tE / 2 + lady
    parts["east"] = ea
    note("plan", "parts", f"nave {nb} bays, " + (f"transept {kb} per arm, " if kb else "no transept, ") + f"choir {mb}"
         + ("" if ch_ais else " (no aisles)") + f", east end — {EAST_RU[east]}" + (f" ({nch})" if nch else ""),
         "east end — by the style of the choir's campaign: Romanesque — apse, France — ambulatory with an odd number of chapels, England — flat, hall church — "
         "polyhedron", "style=, seed=")
    # ---- west end: portal, towers, rose, screen
    wt = dict(Sp["west"]); ntw = int(wt.get("towers", 0)); TSw = F["west"]["tower"]
    if unf and unf["kind"] == "nave": ntw = 0                                    # Beauvais: there is no facade — a temporary wall
    lo_p, hi_p = TK["canon"]["portal_h"]
    ph = float(np.clip(0.23 * Mn["H"], lo_p * kq, hi_p * kq)); ph = math.ceil(ph * 10) / 10 if ph <= lo_p * kq + 1e-9 else math.floor(ph * 10) / 10
    portal = {"h": round(ph, 1), "w": round(min(ph * 0.55, W * 0.6), 1), "n": int(wt["portals"]) if (ntw or wt.get("screen")) else 1}
    towers = []; z_f = Mw["He"] + (0.1 * Mw["H"] if F["west"]["arch"] != "round" else 0.0); irr = St["irregular"]
    def tw_h(base): return base * (1 + irr * float(rng.uniform(-0.12, 0.12)))
    if ntw >= 2:                                                                 # above the first side-aisle bay: the inner face is behind the nave wall
        sq = max(A + Mw["t"] / 2 - Mw["tc"] / 2 + 0.8 * kq, Mn["B"] / Mn["ab"] + Mw["t"] / 2 + 0.45 * kq) * float(Sp.get("tower_side_k", 1.0))
        crowns = [None, None]
        if irr > 0.3 and rng.random() < irr: crowns = [TS.choose(TSw["spire"], rng), TS.choose(TSw["spire"], rng)]
        fams2 = [F["west"], F["west"]]
        if sig == "twin_different":                                              # Chartres: one is plainer (earlier), the other richer (later)
            older = TS.era_family(S, min(c["step"] for c in St["campaigns"]) - 1); newer = TS.era_family(S, max(c["step"] for c in St["campaigns"]) + 1)
            fams2 = [TS.pick(TS.resolve(older, over), irng(seed, "tw0" + older)), TS.pick(TS.resolve(newer, over), irng(seed, "tw1" + newer))]
            crowns = [TS.choose(fams2[0]["tower"]["spire"], rng), "needle" if fams2[1]["tower"]["spire"] != "flat" else TS.choose(fams2[1]["tower"]["spire"], rng)]
            if crowns[0] == crowns[1]: crowns[0] = "flat" if crowns[1] != "flat" else "needle"
        for i_, sgn in enumerate((1, -1)):
            h_ = tw_h(Mn["H"] * float(uni(wt["tower_k"], rng)) * uni([0.92, 1.08], rng)); fz = fams2[i_]
            if sig == "great_spire": h_ *= (1.35 if i_ == 0 else 0.82)
            cr = crowns[i_] if crowns[i_] else (("needle" if i_ == 0 else "flat") if sig == "great_spire" else None)
            tp = tower_plan(h_, sq, 0.0, z_f, fz["tower"], k, rng, crown=cr)
            towers.append({"u": round(sq / 2 - Mw["t"] / 2 - 0.4 * kq, 2), "v": round(sgn * (W / 2 + Mw["tc"] / 2 + 0.05 * kq + sq / 2), 2), "side": round(sq, 2),
                           "role": "west", "family": fz["_name"], "colors": fz["colors"], **tp})
        if ntw >= 4:                                                              # the second pair — at the corners of the transept and the choir (Speyer, Bamberg)
            sq2 = sq * 1.05
            for sgn in (1, -1):
                h_ = tw_h(Mn["H"] * float(uni(wt["tower_k"], rng)) * 0.82); tp = tower_plan(h_, sq2, 0.0, Me["He"], TSw, k, rng)
                towers.append({"u": round(u_x1 + Me["t"] / 2 + sq2 / 2 + 0.05, 2), "v": round(sgn * (W / 2 + A + Me["t"] / 2 + sq2 / 2 + 0.05), 2),
                               "side": round(sq2, 2), "role": "east", "family": F["west"]["_name"], "colors": F["west"]["colors"], **tp})
    elif ntw == 1:
        sq = max(W * 0.9, (A + Mw["t"]) * 1.6); h_ = tw_h(Mn["H"] * float(uni(wt["tower_k"], rng)) * (1.3 if sig == "great_spire" else 1.0))
        tp = tower_plan(h_, sq, 0.0, Mw["He"] + 0.1 * Mw["H"], TSw, k, rng, crown="needle" if sig in ("great_spire", "openwork") else None, octagon=True)
        towers.append({"u": round(-Mw["t"] / 2 - sq / 2 - 0.05, 2), "v": 0.0, "side": round(sq, 2), "role": "west", "family": F["west"]["_name"],
                       "colors": F["west"]["colors"], **tp})
    for tw in towers:                                                            # a side portal in the tower base: within the face and the side aisle
        if tw["role"] == "west" and ntw >= 2 and int(wt["portals"]) >= 3:
            sg_ = 1.0 if tw["v"] > 0 else -1.0; lo_ = max(abs(tw["v"]) - tw["side"] / 2, W / 2) + 0.4 * kq; hi_ = min(abs(tw["v"]) + tw["side"] / 2, W / 2 + A) - 0.4 * kq
            wd = min(portal["w"] * 0.75, hi_ - lo_)
            if wd >= 1.2 * kq: tw["portal"] = {"v": round(sg_ * (lo_ + hi_) / 2, 2), "w": round(wd, 2), "h": round(portal["h"] * 0.8, 2)}
    if sig == "openwork":
        for tw in towers:
            if tw["spire"]["kind"] in ("needle", "flat", "leaf"):
                tw["spire"] = {"kind": "openwork", "h": round(max(tw["spire"]["h"], 0.4 * (tw["h"] - tw["stages"][0]["z0"])), 2)}
    if unf and unf["kind"] in ("one_tower", "both_towers", "tower"):
        for i_, tw in enumerate(towers):
            if tw["role"] != "west": continue
            if unf["kind"] in ("both_towers", "tower") or i_ == unf["which"]:
                main_ = unf["kind"] != "both_towers" or i_ == unf["which"]
                towers[i_] = stump(tw, unf["level"] if main_ else unf["level"] * 0.55, unf["crane"] and main_)
    if state["spires_fallen"]:
        for tw in towers:
            if tw["spire"]["kind"] in ("needle", "openwork", "leaf", "cluster", "tent", "pyramid", "helm"): tw["spire"] = {"kind": "broken", "h": 0.0}; tw["fallen"] = True
    if towers:
        note("facade", "towers", "; ".join(f"{tw['role']} {tw['h']:.0f} m ({tw['family']}): {tw['why']}" for tw in towers),
             "towers follow the style of the facade's campaign; irregularity and a special element change the pair (Chartres — different, Strasbourg — one huge)", "seed=, history=")
    rose = None
    if F["west"]["west"].get("rose") and ntw != 1:
        r_ = F["west"]["west"]["rose"] * W * 0.9; zr = min((Mw["z_tri"] + Mw["spring"]) / 2 + 0.05 * Mw["H"], z_f - r_ - 1.0 * kq); rose = {"r": round(r_, 2), "z": round(zr, 2)}
    screen = None
    if wt.get("screen"):
        screen = {"half": round(W / 2 + A + Mw["t"] / 2 + wt["screen"] * A, 2), "z": round(Mw["He"] + 0.12 * Mw["H"], 2), "turrets": int(wt.get("turrets", 0)),
                  "turret_r": round(0.11 * W, 2), "turret_h": round(Mw["He"] + 0.42 * Mw["H"], 2)}
    # ---- crossing
    cr = F["transept"]["crossing"]; crossing = None; ck = cr["kind"]
    if sig == "lantern": ck = "lantern"
    if sig == "crossing_dome": ck = "dome"
    if has_x and ck != "none":
        h_ = Mt["H"] * float(cr["k"]) * (1.3 if (sig == "great_spire" and not towers) else 1.0)
        if ck == "spire":
            crossing = {"kind": "spire", "h": round(h_, 1), "r": round(0.17 * W, 2), "z0": round(Mt["ridge"] - 2.0 * kq, 2),
                        "z1": round(Mt["ridge"] + 0.14 * (h_ - Mt["ridge"]), 2)}
        elif ck == "lantern":                                                    # Ely: a full-width octagon and a wooden lantern
            crossing = {"kind": "lantern", "r": round(W / 2 + Mt["tc"] / 2, 2), "z0": round(Mt["He"] - 1.0 * kq, 2), "z1": round(Mt["He"] + 0.22 * Mt["H"], 2),
                        "r2": round(0.42 * W, 2), "z2": round(Mt["He"] + 0.42 * Mt["H"], 2), "h": round(Mt["He"] + 0.5 * Mt["H"], 1)}
        elif ck == "dome":                                                       # Florence: a drum with windows, a dome, a lantern
            dk = TS.choose({"pointed": 0.6, "hemi": 0.4}, rng) if F["transept"]["arch"] != "round" else "hemi"
            dh = W / 2 * (1.25 if dk == "pointed" else 1.0); zd = Mt["ridge"] - 0.5 * kq
            crossing = {"kind": "dome", "dome": dk, "r": round(W / 2 + Mt["tc"] / 2, 2), "z0": round(Mt["He"] - 1.0 * kq, 2), "zd": round(zd, 2),
                        "dh": round(dh, 2), "windows": 16, "h": round(zd + dh + 0.25 * W, 1)}
        else:
            oct_ = ck == "tower" and F["transept"]["arch"] == "round"
            tp = tower_plan(h_, W + Mt["tc"] + 0.6 * kq, Mt["He"] - 1.0 * kq, Mt["ridge"] + 0.4 * kq, F["transept"]["tower"], k, rng,
                            crown="needle" if ck == "tower_spire" else ("pyramid" if oct_ else None), octagon=False)
            if oct_:
                for s_ in tp["stages"][1:]: s_["plan"] = "poly8"
                tp["why"] = tp["why"].replace("square", "poly8")
            crossing = {"kind": ck, "u": round((u_n1 + u_x1) / 2, 2), "v": 0.0, "side": round(W + Mt["tc"] + 0.6 * kq, 2), "family": F["transept"]["_name"],
                        "colors": F["transept"]["colors"], "role": "crossing", **tp}
        if state["spires_fallen"] and crossing["kind"] in ("spire", "tower_spire"):
            crossing = None if crossing["kind"] == "spire" else dict(crossing, spire={"kind": "broken", "h": 0.0}, fallen=True)
        if crossing:
            note("crossing", crossing["kind"], f"{crossing.get('h', 0):.0f} m", "by the style of the transept's campaign: France — a flèche, England — a tower with a spire, "
                 "Romanesque — an octagonal tower; an octagonal lantern (Ely) and a dome on a drum (Florence) are special elements", "style=, seed=")
    # ---- buttresses and flying buttresses (by the part's module)
    buts = []
    def in_tower(u, v, pad):
        return any(abs(u - tw["u"]) < tw["side"] / 2 + pad and abs(v - tw["v"]) < tw["side"] / 2 + pad for tw in towers)
    fort = sig == "fortress" or max(M_["fortified"] for M_ in mods.values()) >= 0.6
    def add_but(M, part, u, v, n, kind, z_top, w_=None, d_=None, span=None, tiers=None, pin=None, tower=False):
        bw, bd = M["buttress"]["w"], M["buttress"]["depth"]
        if in_tower(u, v, (w_ or bw) / 2 + 0.6 * kq) or in_tower(u + n[0] * (d_ or bd), v + n[1] * (d_ or bd), 0.6 * kq): return
        form = M["buttress"]["kind"] if kind == "flyer" else ("lesene" if M["buttress"]["kind"] == "lesene" else "buttress")
        if fort: form = "turret"
        b_ = {"u": round(u, 2), "v": round(v, 2), "n": [round(n[0], 4), round(n[1], 4)], "kind": kind, "form": form, "w": round(w_ or bw, 2),
              "d": round(d_ or bd, 2), "z_top": round(z_top, 2), "part": part, "mod": mkey[M["family"]]}
        if span is not None and not fort: b_["span"] = round(span, 2); b_["tiers"] = tiers; b_["th"] = M["buttress"]["th"]
        elif span is not None: b_["kind"] = "wall"; b_["z_top"] = round(M["Ea"] + 0.3 * (M["He"] - M["Ea"]), 2)
        if pin is None: pin = M["pinnacles"]
        if pin: b_["pinnacle"] = round(3.6 * 0.6 * (w_ or bw) * (1.5 if sig == "pinnacle_forest" else 1.0) * M["pinnacle_k"], 2)
        if tower and not fort:
            ts_ = max(bw * 1.5, 0.6 * (d_ or bd)); b_["tower"] = {"side": round(ts_, 2), "z1": round(z_top + 0.22 * M["H"], 2), "spire": round(2.6 * ts_, 2)}
        buts.append(b_)
    def flyer_tiers(M, span):
        out_ = []
        for i in range(M["buttress"]["flyers"]):
            zw = max(M["spring"] + M["rise"] * (-0.1 + 0.42 * i), M["z_lean"] + M["buttress"]["th"] + 0.6 * kq + i * 0.42 * max(M["rise"], 0.25 * M["W"]))
            out_.append({"z_wall": round(zw, 2), "z_pier": round(zw - span * math.tan(math.radians(24)), 2)})
        return out_
    bt_tow = bool(Sp.get("buttress_towers"))
    def aisle_line(M, part, u, sgn, full):
        aisle_out = W / 2 + A + M["t"] / 2; span_f = A + M["t"] / 2 - M["tc"] / 2
        if M["hall"]: add_but(M, part, u, sgn * aisle_out, (0.0, sgn), "wall", M["He"] - 1.5 * kq); return
        if M["buttress"]["flyers"] and full:
            tiers = flyer_tiers(M, span_f); z_top = max(t_["z_pier"] for t_ in tiers) + M["buttress"]["th"] + 1.0 * kq
            add_but(M, part, u, sgn * aisle_out, (0.0, sgn), "flyer", z_top, span=span_f, tiers=tiers, tower=bt_tow)
        else: add_but(M, part, u, sgn * aisle_out, (0.0, sgn), "wall", M["Ea"] - 0.4 * kq, pin=False if M["buttress"]["kind"] == "lesene" else None)
    for sgn in (1, -1):
        ab_n = Mn["ab"]; bay_n = Bn / ab_n
        for j in range(nb * ab_n + 1):
            if j == 0 or (has_x and kb and j == nb * ab_n): continue
            if unf and unf["kind"] == "nave" and j == nb * ab_n: continue
            aisle_line(Mn, "nave", j * bay_n, sgn, j % ab_n == 0)
        if ch_ais:
            ab_c = Me["ab"]; bay_c = Bc / ab_c
            for j in range(mb * ab_c + 1):
                if (j == 0 and kb) or (j == mb * ab_c and east == "chevet"): continue
                aisle_line(Me, "choir", u_x1 + j * bay_c, sgn, j % ab_c == 0)
        else:
            for j in range(mb + 1):
                if j == 0 and kb: continue
                add_but(Me, "choir", u_x1 + j * Bc, sgn * (W / 2 + Me["t"] / 2), (0.0, sgn), "wall", (Me["He"] - 1.5 * kq) if Me["hall"] else (Me["Ea"] - 0.4 * kq),
                        d_=Me["buttress"]["depth"] if Me["hall"] else 0.07 * W, pin=Me["pinnacles"] and Me["hall"])
            if not kb and Mn["hall"]: add_but(Mn, "nave", u_n1 + Mn["t"] / 2, sgn * (W / 2 + A * 0.5), (1.0, 0.0), "wall", Mn["He"] - 1.5 * kq, d_=Mn["buttress"]["depth"] * 0.6)
        if kb:
            v1 = W / 2 + kb * Bt; tT = Mt["t"]
            for j in range(1, kb + 1):
                v = sgn * (W / 2 + j * Bt)
                if abs(v) < W / 2 + A + tT / 2 + Mt["buttress"]["w"]: continue
                for uu, nu in ((u_n1 - tT / 2, -1.0), (u_x1 + tT / 2, 1.0)):
                    add_but(Mt, "transept", uu, v - sgn * (0.5 * Mt["buttress"]["w"] if j == kb else 0.0), (nu, 0.0), "wall", Mt["spring"] + 0.3 * Mt["rise"],
                            d_=Mt["buttress"]["depth"] * 0.8)
            for uu in (u_n1 + Mt["buttress"]["w"] / 2, u_x1 - Mt["buttress"]["w"] / 2):
                add_but(Mt, "transept", uu, sgn * (v1 + tT / 2), (0.0, sgn), "wall", Mt["spring"] + 0.3 * Mt["rise"], d_=Mt["buttress"]["depth"] * 0.8)
    if east == "chevet":
        n = ea["n"]; r2 = ea["r2"]; span_f = A + Me["t"] / 2 - Me["tc"] / 2
        for i in range(1, n):
            a_ = -math.pi / 2 + math.pi * i / n; nn = (math.cos(a_), math.sin(a_)); rr = r2 + Me["t"] / 2
            if Me["buttress"]["flyers"]:
                tiers = flyer_tiers(Me, span_f); z_top = max(t_["z_pier"] for t_ in tiers) + Me["buttress"]["th"] + 1.0 * kq
                add_but(Me, "choir", u_c1 + rr * nn[0], rr * nn[1], nn, "flyer", z_top, span=span_f, tiers=tiers, tower=bt_tow,
                        d_=Me["buttress"]["depth"] + (ea["straight"] + ea["rc"] * 0.6 if nch else 0.0))
            else: add_but(Me, "choir", u_c1 + rr * nn[0], rr * nn[1], nn, "wall", Me["Ea"] - 0.4 * kq)
        for sgn in (1, -1):
            if Me["buttress"]["flyers"]:
                tiers = flyer_tiers(Me, span_f); z_top = max(t_["z_pier"] for t_ in tiers) + Me["buttress"]["th"] + 1.0 * kq
                add_but(Me, "choir", u_c1, sgn * (W / 2 + A + Me["t"] / 2), (0.0, sgn), "flyer", z_top, span=span_f, tiers=tiers, tower=bt_tow)
    elif east in ("apse", "polygon"):
        n = ea["n"]; R_ = W / 2 + Me["t"] / 2
        for i in range(1, n):
            if east == "apse" and i not in (2, 4, 7, 9): continue
            a_ = -math.pi / 2 + math.pi * i / n; nn = (math.cos(a_), math.sin(a_)); f_ = math.cos(math.pi / (2 * n)) if east == "polygon" else 1.0
            add_but(Me, "choir", u_c1 + R_ * nn[0] * f_, R_ * nn[1] * f_, nn, "wall", (Me["He"] - 1.5 * kq) if east == "polygon" else Me["spring"],
                    d_=Me["buttress"]["depth"] if east == "polygon" else 0.07 * W, pin=Me["pinnacles"] and east == "polygon")
    else:
        for sgn in (1, -1):
            ao = W / 2 + A + Me["t"] / 2
            add_but(Me, "choir", u_c1 + Me["t"] / 2, sgn * (ao - Me["buttress"]["w"] / 2), (1.0, 0.0), "wall", Me["Ea"] - 0.4 * kq, d_=Me["buttress"]["depth"] * 0.7)
            add_but(Me, "choir", u_c1 + Me["t"] / 2, sgn * (W / 2 + Me["tc"] / 2 + Me["buttress"]["w"] / 2), (1.0, 0.0), "wall", Me["spring"], d_=Me["buttress"]["depth"] * 0.6)
    if ntw <= 1 and not screen:                                                  # west corners without a pair of towers
        for sgn in (1, -1):
            vv = W / 2 + A + Mw["t"] / 2 - Mw["buttress"]["w"] / 2
            add_but(Mw, "west", -Mw["t"] / 2, sgn * vv, (-1.0, 0.0), "wall", (Mw["He"] - 1.5 * kq) if Mw["hall"] else Mw["Ea"] - 0.4 * kq, d_=Mw["buttress"]["depth"] * 0.7)
            if not Mw["hall"]: add_but(Mw, "west", -Mw["t"] / 2, sgn * (W / 2 + Mw["tc"] / 2 + Mw["buttress"]["w"] / 2), (-1.0, 0.0), "wall", Mw["spring"], d_=Mw["buttress"]["depth"] * 0.6)
    note("structure", "supports", f"{len(buts)} buttresses; shapes: " + ", ".join(sorted({b_['form'] for b_ in buts}))
         + (f"; flying buttresses — {max(M_['buttress']['flyers'] for M_ in mods.values())} tiers" if any(b_['kind'] == 'flyer' for b_ in buts) else "")
         + ("; fortress: tower buttresses instead of flying buttresses (Albi)" if fort else ""),
         "buttress depth — a fraction of the span (Huerta 2006); a flying buttress — from the pier outside the side aisle to the nave wall at the springing of the vault; the pier form — by "
         "the part's family (pilaster strips, buttresses, flying buttresses, claws, branches, towers)", "style=")
    # ---- special elements and additions: places
    ex = {}; side = St["side"]; aisle_out = W / 2 + A + Mn["t"] / 2; bdn = Mn["buttress"]["depth"]
    if sig == "campanile":                                                       # Pisa: a separate bell tower, to the southwest
        sq = max(0.55 * W, 7.0 * kq); h_ = Mn["H"] * 2.2 * uni([0.9, 1.15], rng)
        tp = tower_plan(h_, sq, 0.0, min(0.35 * h_, Mn["He"]), F["west"]["tower"], k, rng)
        ex["campanile"] = {"u": round(-0.2 * W, 2), "v": round(side * (aisle_out + bdn + 8.0 * kq + sq / 2), 2), "side": round(sq, 2), "role": "campanile",
                           "family": F["west"]["_name"], "colors": F["west"]["colors"], **tp}
    if sig == "baptistery":
        r_b = 0.62 * W; ex["baptistery"] = {"u": round(-(portal["w"] * 1.8 + 14.0 * kq + r_b), 2), "v": 0.0, "r": round(r_b, 2), "h": round(0.55 * Mn["H"], 2),
                                            "dh": round(r_b * 0.95, 2)}
    if sig == "westwork" and len([tw for tw in towers if tw["role"] == "west"]) == 2:
        ex["westwork"] = {"side": round(W + Mw["tc"] + 0.6 * kq, 2), "z0": round(Mw["He"] - 1.0 * kq, 2), "z1": round(z_f + 0.35 * Mw["H"], 2), "spire": round(0.25 * W, 2)}
    if sig == "transept_towers" and kb:
        tt = []
        for sgn in (1, -1):
            v1 = W / 2 + kb * Bt; sq = W + Mt["t"] + 0.6 * kq; h_ = Mt["ridge"] + 0.55 * Mt["H"]
            tp = tower_plan(h_, sq, Mt["He"] - 1.0 * kq, Mt["ridge"] + 0.4 * kq, F["transept"]["tower"], k, rng, octagon=False)
            tt.append({"u": round((u_n1 + u_x1) / 2, 2), "v": round(sgn * (v1 - sq / 2 + Mt["t"] / 2), 2), "side": round(sq, 2), "role": "transept",
                       "family": F["transept"]["_name"], "colors": F["transept"]["colors"], **tp})
        ex["transept_towers"] = tt
    for a in St["additions"]:
        kd = a["kind"]
        if kd == "chapels":                                                      # between the nave buttresses
            bays = [j for j in range(1, nb - (1 if kb else 0)) if not in_tower((j + 0.5) * Bn, aisle_out + bdn / 2, 0.0)]
            if bays and Mn["buttress"]["flyers"]: ex["chapels"] = {"bays": bays, "depth": round(bdn * 0.9, 2), "h": round(min(Mn["Ea"] * 0.92, Mn["Ha"] * 0.95), 2)}
        elif kd == "chapter_house" and kb:
            r_ch = 0.62 * W; v1 = W / 2 + kb * Bt; L_v = 6.0 * kq
            ex["chapter_house"] = {"u": round((u_n1 + u_x1) / 2, 2), "v": round(side * (v1 + Mt["t"] / 2 + L_v + r_ch), 2), "r": round(r_ch, 2),
                                   "n": 8 if rng.random() < 0.6 else 10, "h": round(0.5 * Mn["H"], 2), "vest_w": round(4.0 * kq, 2), "vest_l": round(L_v, 2)}
        elif kd == "cloister" and nb >= 3:
            v0 = aisle_out + bdn + 1.5 * kq; L_ = min(u_n1 - (towers[0]["side"] if towers else 0.0) - 2.0 * kq, 9.0 * W)
            if L_ > 2.4 * W:
                wk_ = 0.32 * W; lo_, hi_ = v0 + wk_ + 1.0 * kq, min(v0 + L_ - wk_ - 1.0 * kq, W / 2 + kb * Bt - Mt["t"] / 2 - 1.2 * kq)   # a door from the transept arm
                ex["cloister"] = {"u0": round(u_n1 - L_, 2), "u1": round(u_n1 - 0.5 * kq, 2), "v0": round(side * v0, 2), "v1": round(side * (v0 + L_), 2),
                                  "walk": round(wk_, 2), "h": round(min(0.32 * Mn["H"], 9.0 * kq), 2), "side": side,
                                  "door_v": round(side * (lo_ + hi_) / 2, 2) if kb and hi_ - lo_ > 1.8 * kq else None}
        elif kd == "porch" and portal["n"]:                                          # in front of the portal: with one tower — in front of its face
            d_ = 1.4 * portal["w"] + 2.0 * kq; u1p = (towers[0]["u"] - towers[0]["side"] / 2) if ntw == 1 else -Mw["t"] / 2
            ex["porch"] = {"d": round(d_, 2), "w": round(portal["w"] * 1.9, 2), "h": round(portal["h"] * 1.5, 2), "u0": round(u1p - d_, 2), "u1": round(u1p, 2)}
        elif kd == "sacristy":
            ex["sacristy"] = {"u0": round(u_x1 + 0.5 * Bc, 2), "u1": round(u_x1 + min(2.0, mb) * Bc, 2), "side": -side, "d": round(0.6 * A + 2.0 * kq, 2),
                              "h": round(Me["Ea"] * 0.6, 2)}
        elif kd == "stair_turrets" and kb:
            v1 = W / 2 + kb * Bt; r_t = 0.075 * W + 0.5 * kq; zt = Mt["He"] + 0.18 * Mt["H"]
            ex["turrets"] = [[round(uu, 2), round(sgn * (v1 + Mt["t"] / 2 + r_t * 0.3), 2), round(r_t, 2), round(zt, 2)]
                             for sgn in (1, -1) for uu in (u_n1 - Mt["t"] / 2 - r_t * 0.3, u_x1 + Mt["t"] / 2 + r_t * 0.3)]
    clo = ex.get("cloister"); cp_ = ex.get("campanile")
    if clo and cp_:                                                              # the campanile is not in the cloister: to the other side
        vlo_, vhi_ = sorted((clo["v0"], clo["v1"]))
        if clo["u0"] - cp_["side"] / 2 < cp_["u"] < clo["u1"] + cp_["side"] / 2 and vlo_ - cp_["side"] / 2 < cp_["v"] < vhi_ + cp_["side"] / 2: cp_["v"] = -cp_["v"]
    if clo:                                                                      # remove the supports inside the cloister: its walls and roof hold themselves
        vlo_, vhi_ = sorted((clo["v0"], clo["v1"]))
        buts[:] = [b_ for b_ in buts if not (clo["u0"] - 0.5 < b_["u"] + b_["n"][0] * b_["d"] < clo["u1"] + 0.5 * kq + Mt["t"] and vlo_ - 0.5 < b_["v"] + b_["n"][1] * b_["d"] < vhi_ + 0.5)]
        if ex.get("turrets"): ex["turrets"] = [t_ for t_ in ex["turrets"] if not (clo["u0"] - t_[2] < t_[0] < clo["u1"] + Mt["t"] + t_[2] and vlo_ - t_[2] < t_[1] < vhi_ + t_[2])]
    if ex:
        note("special and additions", ", ".join(ex), "places: the campanile — at the southwest corner, the baptistery — on the axis in front of the facade (Pisa), the chapels — "
             "in the nave bays between the piers, the chapter house — behind the transept end on the cloister side, the cloister — along the nave from the transept, the porch — in front of "
             "the portal, the sacristy — by the choir, the turrets — at the corners of the transept", "seed=, wealth=")
    return {"modules": mods, "MK": MK, "F": F, "parts": parts, "towers": towers, "crossing": crossing, "portal": portal, "rose": rose, "screen": screen,
            "buttresses": buts, "extras": ex, "u_end": u_end, "west": wt, "z_f": z_f, "fort": fort, "has_x": has_x, "nb": nb, "kb": kb}


def plan(folder, out=None, style="gothic_fr", seed=1, at=None, **kw):
    rng = np.random.default_rng(int(seed)); logic = []

    def note(step, what, value, why, alt=None):
        logic.append({"step": step, "what": what, "value": value, "why": why, **({"otherwise": alt} if alt else {})})
    kn = {k_: (float(kw.pop(k_)) if k_ in kw else None) for k_ in ("history", "wealth", "ruin", "irregular")}
    over = TS.parse_over(kw); S = TS.resolve(style, over); S["_over"] = over
    Sp = TS.pick(S, irng(seed, S["_name"])); Sp["_over"] = over
    pk = Sp.get("plan", "basilica")
    if pk != "basilica": raise SystemExit(f"{style}: the plan {pk} is an edifice, build it with edifices/edifice_plan.py")
    k = uni(Sp.get("scale", 1.0), rng); kq = k ** 0.5
    note("style", S["_name"], f"arch {Sp.get('arch')}, vault {Sp.get('vault')}, tower tops {Sp['tower']['spire']}, supports {Sp['buttress'].get('kind')}, spikes "
         f"{Sp.get('spikes', 0)}, organic {Sp.get('organic', 0)}, fortress {Sp.get('fortified', 0)}, factory {Sp.get('industrial', 0)}"
         + (f"; scale ×{k:.1f}" if k > 1 else ""), "the form language — the genes of a family (historical, fantasy and sci-fi are points of one space); "
         "variants follow the seed", "style=a+b, gene=value")
    W = uni(Sp["W"], rng) * k; A = Sp["A"] * W
    layout = {"plan": pk, "towers": int(Sp["west"].get("towers", 0)), "transept": bool(Sp["transept"][1]),
              "crossing": bool(Sp["transept"][1]) or Sp["crossing"]["kind"] != "none", "flyers": int(Sp["buttress"]["flyers"]) > 0}
    St = TST.story(S, rng, kn, layout); logic += St["notes"]
    BZ = basilica(S, Sp, St, seed, rng, k, kq, W, A, note)
    M0 = BZ["modules"][BZ["MK"]["nave"]]; u_end = BZ["u_end"]; towers = BZ["towers"]; ex = BZ["extras"]
    u_west = min([-M0["t"] / 2 - 0.4 * kq] + [tw["u"] - tw["side"] / 2 for tw in towers] + ([ex["campanile"]["u"] - ex["campanile"]["side"] / 2] if ex.get("campanile") else [])
                 + ([ex["baptistery"]["u"] - ex["baptistery"]["r"]] if ex.get("baptistery") else []) + ([ex["porch"]["u0"]] if ex.get("porch") else []))
    vs = [W / 2 + A + M0["t"] / 2 + M0["buttress"]["depth"]] + ([BZ["parts"]["transept"]["v1"] + M0["t"] / 2] if BZ["kb"] else []) \
        + ([BZ["screen"]["half"]] if BZ["screen"] else []) + [abs(tw["v"]) + tw["side"] / 2 for tw in towers]
    v_lo, v_hi = -max(vs), max(vs)
    if ex.get("campanile"): e_ = ex["campanile"]; v_lo, v_hi = min(v_lo, e_["v"] - e_["side"] / 2), max(v_hi, e_["v"] + e_["side"] / 2)
    if ex.get("chapter_house"): ch = ex["chapter_house"]; v_lo, v_hi = min(v_lo, ch["v"] - ch["r"]), max(v_hi, ch["v"] + ch["r"])
    if ex.get("cloister"): cl_ = ex["cloister"]; v_lo, v_hi = min(v_lo, cl_["v0"], cl_["v1"]), max(v_hi, cl_["v0"], cl_["v1"])
    for tw in ex.get("transept_towers", []): v_lo, v_hi = min(v_lo, tw["v"] - tw["side"] / 2), max(v_hi, tw["v"] + tw["side"] / 2)
    podium = None
    if Sp.get("podium"):
        pl = int(rng.integers(Sp["podium"]["levels"][0], Sp["podium"]["levels"][1] + 1)); ph_ = M0["H"] * uni(Sp["podium"]["h"], rng)
        run = ph_ / TK["canon"]["step"] * 0.3; mg = max(W * 0.5, run + 0.15 * W)
        podium = {"levels": pl, "h": round(ph_, 1), "margin": round(mg, 1), "stair_w": round(W * 0.8, 1), "run": round(run, 1)}
        note("structure", "podium", f"{pl} tiers of {ph_:.0f} m, terrace {mg:.0f} m, flight {run:.0f} m", "hive: the cathedral = a fortress (Blanche) — on a bastion podium; "
             "steps sized for a person (0.17 m), a flight rests on the terrace of the lower one", "seed=")
    # ---- site: the platform, the axis to the east
    pad_p = podium["margin"] * podium["levels"] if podium else 0.0
    if podium: u_west -= pad_p + podium["run"] + 6.0
    u_w0 = u_west - (8.0 * kq if not podium else 0.0)
    Lx = u_end - u_w0 + pad_p + 12; Wy = (v_hi - v_lo) + 2 * pad_p + 12
    G = Ground(folder); Ssite = CS.read(folder, area=min(Lx * Wy * 0.8, 60000.0), at=at, g=G)
    c = np.asarray(Ssite["centre"], float); ax = np.array([1.0, 0.0]); best = -1.0
    ax_site = np.asarray(Ssite.get("axis", [1.0, 0.0]), float); ax_site /= max(np.linalg.norm(ax_site), 1e-9)
    for cand in (ax_site, -ax_site):                                             # the long side of the platform, if it is within 35° of east
        if cand[0] >= math.cos(math.radians(35)) and cand[0] > best: ax, best = cand.copy(), cand[0]
    pv = np.array([-ax[1], ax[0]]); u_mid = (u_end + pad_p + u_w0) / 2; v_mid = (v_lo + v_hi) / 2; origin = c - ax * u_mid - pv * v_mid
    note("place", "platform and axis", f"{Ssite['type_ru']}, axis at {math.degrees(math.atan2(ax[1], ax[0])):+.0f}° from east",
         "the altar to the east; the axis turns to the long side of the platform if it is within 35° of east", "at=x,y — another site")
    # the deflection of the choir axis follows the ground (Quimper: straight ahead is a marsh)
    # or occasionally an old foundation
    bend = 0.0; u_x1 = BZ["parts"]["choir"]["u0"]; Lc = u_end - u_x1; d0 = uni(TK["story"]["bend"]["deg"], rng)
    def badness(deg):
        a_ = math.radians(deg); Q = []
        for uu in np.linspace(0, Lc, 9):
            for vv in np.linspace(-W / 2 - A, W / 2 + A, 5):
                Q.append(origin + ax * (u_x1 + uu * math.cos(a_) - vv * math.sin(a_)) + pv * (uu * math.sin(a_) + vv * math.cos(a_)))
        Q = np.asarray(Q); z = G.height(Q[:, 0], Q[:, 1]); wet = G.water(Q[:, 0], Q[:, 1])
        return float(np.percentile(z, 95) - np.percentile(z, 5)) + 30.0 * float(np.mean(wet > 0.1))
    b0 = badness(0.0); bp, bm = badness(d0), badness(-d0)
    if min(bp, bm) < b0 - 2.5:
        bend = d0 if bp < bm else -d0
        note("history", "choir axis", f"deflected by {bend:+.1f}°", f"straight ahead the ground is worse (level change and water: {b0:.1f} versus {min(bp, bm):.1f}) — "
             "the choir was turned toward firm ground (Quimper: 8–10°, a marsh)", "at=")
    elif rng.random() < float(TK["story"]["bend"]["p_old"]) * (0.5 + St["irregular"] + St["history"]):
        bend = round(d0 * (1 if rng.random() < 0.5 else -1) * 0.6, 2)
        note("history", "choir axis", f"deflected by {bend:+.1f}°", "the choir was set on the foundation of an old church — the axes diverged", "irregular=0")
    BZ["parts"]["choir"]["bend"] = round(bend, 2)
    # leveling: the platform with margins goes to the median, a cut of up to 8 m
    # (on a slope the floor sits at the top, with an undercroft below), a 12 m batter
    pad = 6.0 * kq; Hn = G.Hn.copy(); b = G.near_bounds(); ny, nx = Hn.shape
    X, Y = np.meshgrid(b[0] + np.arange(nx) * G.nres, b[1] + np.arange(ny) * G.nres)
    def to_local(X_, Y_):
        U_ = (X_ - origin[0]) * ax[0] + (Y_ - origin[1]) * ax[1]; V_ = (X_ - origin[0]) * pv[0] + (Y_ - origin[1]) * pv[1]
        if bend:                                                                 # the choir — in its own (rotated) frame
            a_ = -math.radians(bend); m_ = U_ > u_x1; du, dv = U_ - u_x1, V_
            U_ = np.where(m_, u_x1 + du * math.cos(a_) - dv * math.sin(a_), U_); V_ = np.where(m_, du * math.sin(a_) + dv * math.cos(a_), V_)
        return U_, V_
    def dout(U_, V_):
        return np.maximum.reduce([np.zeros_like(U_), (u_w0 - pad) - U_, U_ - (u_end + pad + pad_p), (v_lo - pad - pad_p) - V_, V_ - (v_hi + pad + pad_p)])
    U, V = to_local(X, Y)
    m = dout(U, V) <= 0; mb = (U > u_west) & (U < u_end) & (V > v_lo) & (V < v_hi)
    zmed = float(np.median(Hn[m])) if m.any() else float(G.height(*c))
    zhi = float(np.percentile(Hn[mb], 99.5)) if mb.any() else zmed
    z0 = max(zmed, zhi - 8.0)
    def level(Hg, d_):
        f = np.clip(1 - d_ / 12.0, 0, 1); f = f * f * (3 - 2 * f)
        return Hg - np.clip(Hg - z0, -8, 8) * f
    Hn = level(Hn, dout(U, V))
    rel = float(np.percentile(np.abs(G.Hn[m] - z0), 95)) if m.any() else 0.0
    above = float(np.percentile(Hn[mb], 99.5) - z0) if mb.any() else 0.0; under = float(z0 - np.min(Hn[mb])) if mb.any() else 0.0
    out = out or os.path.join(ROOT, "out/temple", os.path.basename(folder.rstrip("/")) + f"_{S['_name'].replace(':', '_').replace(',', '_')}_{seed}")
    os.makedirs(out, exist_ok=True)
    zz = dict(np.load(os.path.join(folder, "relief.npz"))); zz["near"] = Hn.astype(zz["near"].dtype)
    fb = G.far_bounds(); Fy, Fx = G.Hf.shape
    FX, FY = np.meshgrid(fb[0] + np.arange(Fx) * G.fres, fb[1] + np.arange(Fy) * G.fres); FU, FV = to_local(FX, FY)
    zz["far"] = level(G.Hf, np.maximum(dout(FU, FV) - G.fres, 0.0)).astype(zz["far"].dtype)
    np.savez_compressed(os.path.join(out, "relief.npz"), **zz); shutil.copy(os.path.join(folder, "location.json"), os.path.join(out, "location.json"))
    floor = TK["canon"]["plinth_steps"] * TK["canon"]["step"] + (podium["levels"] * podium["h"] if podium else 0.0)
    note("place", "alignment", f"platform at elevation {z0:.1f} m, cut/fill up to {rel:.1f} m; the floor {floor:.2f} m higher"
         + (f"; under the temple — an undercroft up to {under:.0f} m" if under > 2 else ""), "a temple on a level platform on a plinth of 5 steps; on a slope the floor is at "
         "the upper level (a cut of up to 8 m), with an undercroft on a terrace below (Bamberg, Laon); beyond the platform edge — a 12 m batter", "at=")
    # ---- checks
    checks = []
    def chk(what, ok, **kw_): checks.append(dict({"what": what, "ok": bool(ok)}, **kw_))
    lo_p, hi_p = TK["canon"]["portal_h"]; portal = BZ["portal"]
    chk("portal sized for a person (5–14 m, for a mega-cathedral × scale^0.5)", lo_p - 1e-6 <= portal["h"] / kq <= hi_p + 1e-6, portal_m=portal["h"])
    for key, M in BZ["modules"].items():
        lo, hi = TS.resolve(M["family"], over)["HW"]
        chk(f"{key} {M['family']}: height to span {lo}–{hi}", lo - 1e-6 <= M["H"] / M["W"] <= hi + 1e-6, ratio=round(M["H"] / M["W"], 2))
        if M["win"]["cl"]: chk(f"{key}: the clerestory window above the side-aisle roof", M["win"]["cl"]["z0"] >= M["z_lean"] + 0.2 * kq and M["win"]["cl"]["h"] >= 1.5 * kq,
                               bottom_windows=M["win"]["cl"]["z0"], roof_top=round(M["z_lean"], 1))
        chk(f"{key}: windows under the vault arcs (trimmed by no more than 3 m)", min(f_ for _, f_ in M["fits"]) > -3.0 * kq, margin_m={n_: f_ for n_, f_ in M["fits"]})
        chk(f"{key}: the side aisle is lower than the nave (except hall churches)", M["Ha"] < M["H"] or M["hall"], lateral_m=M["Ha"])
    flys = [b_ for b_ in BZ["buttresses"] if b_["kind"] == "flyer"]
    if flys: chk("flying buttresses above the side-aisle roof", all(b_["tiers"][0]["z_wall"] - b_["th"] >= BZ["modules"][b_["mod"]]["z_lean"] + 0.3 * kq for b_ in flys))
    ea = BZ["parts"]["east"]
    if ea["kind"] == "chevet": chk("chapels: an odd number, between the piers", ea["chapels"] % 2 == 1 and ea["rc"] > 1.2 * kq, n_chapels=ea["chapels"], radius_m=ea["rc"])
    for tw in BZ["towers"] + ([BZ["crossing"]] if BZ["crossing"] and BZ["crossing"].get("stages") else []) + ex.get("transept_towers", []) \
            + ([ex["campanile"]] if ex.get("campanile") else []):
        top_m = tw["stages"][-1]["z1"] - tw["stages"][0]["z0"]
        sl_ = float(TS.resolve(tw.get("family", S["_name"]), over)["tower"].get("slender", 9.0))
        chk(f"tower ({tw.get('role', '')}): masonry slenderness ≤ {sl_:g}, the upper tier on the lower one", top_m / tw["side"] <= sl_ + 1e-6 and all(
            b_["half"] <= a_["half"] + 1e-6 for a_, b_ in zip(tw["stages"][:-1], tw["stages"][1:])), slenderness=round(top_m / tw["side"], 1))
    chk("the choir axis is deflected by no more than 10°", abs(bend) <= 10.0 + 1e-6, degrees=bend)
    pod_h = podium["levels"] * podium["h"] if podium else 0.0
    chk("the ground is not above the floor, the undercroft is not higher than 0.6 of the vault", above <= 0.5 + pod_h and under <= 0.6 * M0["H"] + pod_h, above_floor_m=round(above, 1), undercroft_m=round(under, 1))
    # ---- result
    hmax = max([tw["h"] for tw in towers] + [BZ["crossing"]["h"] if BZ["crossing"] else 0.0] + [M_["ridge"] for M_ in BZ["modules"].values()]
               + [tw["h"] for tw in ex.get("transept_towers", [])] + ([ex["campanile"]["h"]] if ex.get("campanile") else []))
    R = {"style": S["_name"], "plan": pk, "seed": int(seed), "k": round(k, 2), "place": folder, "out": out,
         "frame": {"origin": origin.round(3).tolist(), "ax": ax.round(5).tolist(), "pv": pv.round(5).tolist(), "z0": round(z0, 2), "under": round(under, 2),
                   "floor": round(floor, 2), "u_west": round(u_west, 2), "u_end": round(u_end, 2), "v_lo": round(v_lo, 2), "v_hi": round(v_hi, 2),
                   "v_half": round(max(-v_lo, v_hi), 2)},
         "story": {k_: St[k_] for k_ in ("history", "wealth", "irregular", "east_first", "campaigns", "unfinished", "signature", "additions", "side", "state")},
         "modules": BZ["modules"], "module": M0, "parts": BZ["parts"], "west": BZ["west"], "portal": portal, "rose": BZ["rose"], "screen": BZ["screen"],
         "crossing": BZ["crossing"], "towers": towers, "buttresses": BZ["buttresses"], "extras": ex, "fort": BZ["fort"],
         "buttress": {"depth": M0["buttress"]["depth"], "w": M0["buttress"]["w"], "flyers": M0["buttress"]["flyers"], "th": M0["buttress"]["th"],
                      "towers": bool(Sp.get("buttress_towers"))},
         "podium": podium, "length": round(u_end, 1), "colors": Sp["colors"], "logic": logic, "checks": checks,
         "site": {k_: Ssite[k_] for k_ in ("type", "type_ru", "centre", "area_m2")}}
    Gp = Ground(out)                                                             # a platform on a slope: an undercroft terrace with a drop to the ground
    UU, VV = np.meshgrid(np.arange(u_west - 2, u_end + 2, 2.0), np.arange(v_lo - 2, v_hi + 2, 2.0))
    zmin = float(Gp.height(origin[0] + ax[0] * UU + pv[0] * VV, origin[1] + ax[1] * UU + pv[1] * VV).min())
    R["terrace"] = {"z": round(z0 + 0.3, 2), "e": round(3.0 * kq, 2), "zmin": round(zmin, 2)} if (not podium and z0 - zmin > 1.5) else None
    SP = TSP.spaces(R, Gp.height); R["spaces"] = SP.to_dict()                    # space: cells, transitions, checks — before the model
    checks += [dict(c_, what="space: " + c_["what"]) for c_ in R["spaces"]["checks"]]
    R["stats"] = {"length_m": round(u_end - u_west, 1), "vault_m": round(M0["H"], 1), "span_m": round(W, 1), "n_towers": len(towers), "highest_m": round(hmax, 1),
                  "n_buttresses": len(BZ["buttresses"]), "n_campaigns": len(St["campaigns"]), "special": (St["signature"] or {}).get("kind"),
                  "annexes": list(ex), "n_rooms": R["spaces"]["stats"]["n_cells"], "n_entrances": R["spaces"]["stats"]["n_entrances"],
                  "checks": f"{sum(c_['ok'] for c_ in checks)}/{len(checks)}"}
    json.dump(R, open(os.path.join(out, "temple.json"), "w"), ensure_ascii=False, default=float)
    return R


def main():
    a = sys.argv[1:]; kw = dict(x.split("=", 1) for x in a[1:])
    at = kw.pop("at", None)
    if at and "," in at: at = [float(v) for v in at.split(",")]
    R = plan(a[0], out=kw.pop("out", None), at=at, **kw)
    print("__RESULT__", json.dumps({"out": R["out"], **R["stats"], "non_passing": [c["what"] for c in R["checks"] if not c["ok"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
