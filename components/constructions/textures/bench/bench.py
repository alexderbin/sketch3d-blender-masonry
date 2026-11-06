"""Check of the "textures" step (without Blender): (1) substances.json agrees with ../masonry/packs.yaml — every masonry
pack has a substance and scans, the scans are listed in library/materials/surfaces.yaml and downloaded; (2) the masonry
tone is preserved — by the albedo frames of the stand, if a folder of shots is given (→ check_tone.py).
.venv/bin/python -m components.constructions.textures.bench.bench [stand shots folder] → lines "ok / NO", exit code 1 if there is a NO"""
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); TX = os.path.dirname(HERE)
ROOT = TX
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
PY = os.path.join(ROOT, ".venv", "bin", "python")
res = []


def check(name, ok, info=""):
    res.append(ok); print(("ok  " if ok else "NO ") + name + (f" — {info}" if info else ""), flush=True)


r = subprocess.run([PY, "-m", "components.constructions.textures.substances", "check"], capture_output=True, text=True)
check("substances.json agrees with the masonry packs", r.returncode == 0, r.stdout.strip())
sub = json.load(open(os.path.join(TX, "substances.json")))["packs"]
surf = (yaml.safe_load(open(os.path.join(ROOT, "library", "materials", "surfaces.yaml"))) or {}).get("surfaces") or {}
need = sorted({s for v in sub.values() for s in v["scans"]})
miss = [s for s in need if s not in surf]
check("pack scans are described in surfaces.yaml", not miss, ", ".join(miss))
nodl = [s for s in need if not os.path.exists(os.path.join(ROOT, "library", "textures", "surface", s, "meta.json"))]
check("pack scans are downloaded (surface_tex.py fetch)", not nodl, ", ".join(nodl))
r = subprocess.run([PY, "-m", "components.constructions.textures.covers", "check"], capture_output=True, text=True)
check("covers.json agrees with covers.yaml and with the kinds of roofing, paving and timber", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
cov = {s["scan"] for s in json.load(open(os.path.join(TX, "covers.json")))["substances"].values()}
nodl = sorted(s for s in cov if not os.path.exists(os.path.join(ROOT, "library", "textures", "surface", s, "meta.json")))
check("covering scans are downloaded (surface_tex.py fetch)", not nodl, ", ".join(nodl))
if len(sys.argv) > 1:
    r = subprocess.run([PY, "-m", "components.constructions.textures.bench.check_tone", sys.argv[1]], capture_output=True, text=True)
    check("masonry tone is preserved (stand albedo, tolerance 6 %)", r.returncode == 0, r.stdout.strip()[:300])
sys.exit(0 if all(res) else 1)
