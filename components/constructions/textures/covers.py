"""Covering substance — the knowledge of the "textures" step (covers.yaml: a layout substance → a scan and how to lay it) →
covers.json (there is no YAML in Blender). The kinds and tones of the component's parts come from their knowledge,
read-only (like substances.py from packs.yaml): roofing (roofing/coverings.yaml), paving (paving/paving.yaml), timber
(timber/timber.yaml): kind → substance, piece role, second substance (paving bedding, half-timbering infill); substance
tones — for faces without the Col tone (paving bedding). Reconciliation: the scans exist in
library/materials/surfaces.yaml, all part substances are covered (except skip), knobs are within limits.
.venv/bin/python -m components.constructions.textures.covers [check]  → covers.json; check — exit code 1 if the file is
stale or there are remarks"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(HERE, "covers.yaml")
OUT = os.path.join(HERE, "covers.json")
SURFACES = os.path.join(ROOT, "library", "materials", "surfaces.yaml")


def parts():
    """Object property of a part → {kinds: {kind: {substance, role, second}}, tones: {substance: rgb}}; second — the paving
    bedding or the half-timbering infill."""
    from components.constructions.paving import paving as P
    from components.constructions.roofing import roofing as R
    from components.constructions.timber import timber as T
    out = {}
    for prop, mod, second in (("sk_roofing", R, None), ("sk_paving", P, "bedding"), ("sk_timber", T, "infill")):
        kinds = {n: {"substance": k["substance"], "role": k.get("role"), "second": k.get(second) if second else None} for n, k in mod.KINDS.items()}
        out[prop] = {"kinds": kinds, "tones": mod.LIBRARY.get("substances") or {}}
    return out


def build():
    d = yaml.safe_load(open(SRC, encoding="utf-8"))
    base = d.get("defaults") or {}
    subs = {name: {**base, **(spec or {})} for name, spec in (d.get("substances") or {}).items()}
    return {"_": "covering substance — components/constructions/textures/covers.py from covers.yaml and the parts' knowledge", "version": d.get("version"),
            "defaults": base, "substances": subs, "skip": d.get("skip") or [], "parts": parts()}


def problems(t):
    """Scans that are not in the surface list; part substances without a row (except skip); knobs out of limits. → [lines]."""
    scans = set((yaml.safe_load(open(SURFACES, encoding="utf-8")) or {}).get("surfaces") or {})
    out = [f"{n}: scan {s['scan']} is not in surfaces.yaml" for n, s in t["substances"].items() if s.get("scan") not in scans]
    for prop, p in t["parts"].items():
        need = {k["substance"] for k in p["kinds"].values()} | {k["second"] for k in p["kinds"].values() if k["second"]}
        out += [f"{prop}: substance {n} — no row in covers.yaml" for n in sorted(need - set(t["substances"]) - set(t["skip"]))]
    for n, s in t["substances"].items():
        lo, hi = s["rough"]
        if not 0.0 <= lo <= hi <= 1.0: out.append(f"{n}: rough {s['rough']} — need 0 ≤ from ≤ to ≤ 1")
        if s["rot"] not in ("free", "grain", "long"): out.append(f"{n}: rot {s['rot']} — free | grain | long")
        if not 0.0 <= s["chroma"] <= 1.0 or not 0.0 <= s["metal"] <= 1.0: out.append(f"{n}: chroma and metal — 0…1")
    return out


if __name__ == "__main__":
    new = build(); bad = problems(new)
    for b in bad: print("  ✗", b)
    if sys.argv[1:2] == ["check"]:
        old = json.load(open(OUT)) if os.path.exists(OUT) else {}
        same = all(old.get(k) == new[k] for k in ("substances", "skip", "parts"))
        print("covers.json —", "agrees with covers.yaml and the parts' knowledge" if same else "stale: .venv/bin/python -m components.constructions.textures.covers")
        sys.exit(0 if same and not bad else 1)
    json.dump(new, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(OUT, len(new["substances"]), "substances", "" if not bad else f"— {len(bad)} remarks")
