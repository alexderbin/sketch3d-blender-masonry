"""Substance by masonry pack — the knowledge of the "textures" step taken from the masonry knowledge
(../masonry/packs.yaml: pack → substance mat, face face, kind kind; the masonry code and knowledge are only read, never
changed) → substances.json (there is no YAML in Blender).
Rules:
  substance → surface scan (library/materials/surfaces.yaml): stone — rock_surface, granite — rock_boulder_dry,
  brick — clay_plaster (fired clay), tuff — rock_pitted_mossy, concrete — concrete, glaze — rock_01;
  the stones differ (face "rubble" or a cell layout: rubble, dry-stone, polygonal, cyclopean, ishigaki) — several
  scans, one per stone (fieldstones of different rocks); ashlar and brick — a single one (one quarry, one kiln).
.venv/bin/python -m components.constructions.textures.substances [check]  → substances.json; "check" — exit code 1 if
the file is stale against packs.yaml"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACKS = os.path.join(HERE, "..", "masonry", "packs.yaml")
OUT = os.path.join(HERE, "substances.json")
BY_MAT = {"stone": ["rock_surface"], "granite": ["rock_boulder_dry"], "brick": ["clay_plaster"],
          "tufa": ["rock_pitted_mossy"], "concrete": ["concrete"], "glazed": ["rock_01"]}
MIXED = {"stone": ["rock_surface", "rock_boulder_dry", "rock_01"], "granite": ["rock_boulder_dry", "rock_01", "rock_surface"]}


def build():
    d = yaml.safe_load(open(PACKS))
    base = d.get("base") or {}
    out = {}
    for name, pk in (d.get("packs") or {}).items():
        p = {**base, **(pk or {})}
        mat, face, kind = p.get("mat", "stone"), p.get("face", "ashlar"), p.get("kind", "courses")
        mixed = face == "rubble" or kind == "cells"
        scans = (MIXED.get(mat) if mixed else None) or BY_MAT.get(mat) or BY_MAT["stone"]
        out[name] = {"substance": mat, "facing": face, "kind": kind, "scans": scans, "stones_varied": mixed}
    return {"_": "substance by masonry pack — components/constructions/textures/substances.py from ../masonry/packs.yaml", "packs": out}


if __name__ == "__main__":
    new = build()
    if sys.argv[1:2] == ["check"]:
        old = json.load(open(OUT)) if os.path.exists(OUT) else {}
        same = old.get("packs") == new["packs"]
        print("substances.json —", "agrees with packs.yaml" if same else "stale: .venv/bin/python " + os.path.relpath(__file__))
        sys.exit(0 if same else 1)
    json.dump(new, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(OUT, len(new["packs"]), "n_packs")
