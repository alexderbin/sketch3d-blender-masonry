#!/bin/bash
# Substance stand for roofing: the roofing plan (all kinds of coverings.yaml) → the "roofing" step → "before" frames → covering substance (bl_cover) →
# "after" frames → sheets. From the project root through guard (Eevee; bash — a rented machine has no zsh):
#   .venv/bin/python -m core.services.guard heavy --wait 3600 --what "textures: roofing" -- bash components/constructions/textures/bench/roofs/run.sh out/textures/roofs [kinds] [res=1280x720]
set -e
ROOT=$(cd "$(dirname "$0")" && pwd); while [ ! -d "$ROOT/core" ] && [ "$ROOT" != "/" ]; do ROOT=$(dirname "$ROOT"); done
cd "$ROOT"; PY="$ROOT/.venv/bin/python"                                          # project root — where core/ is
OUT=${1:-out/textures/roofs}; shift || true
mkdir -p "$OUT"
"$PY" -m components.constructions.roofing.roofing resolve "$OUT" kinds=all out="$OUT/plan_roofs.json" > /dev/null
"$PY" -m components.constructions.textures.covers check
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/constructions/textures/bench/roofs/bl_roofs_tex.py -- "$OUT" "$@" > "$OUT/run.log" 2>&1 \
  || { grep -E "Error|Traceback|line [0-9]" "$OUT/run.log" | tail -20; exit 1; }
grep -E "__RESULT__" "$OUT/run.log" | cut -c1-600
"$PY" -m components.constructions.textures.bench.roofs.sheet "$OUT"
