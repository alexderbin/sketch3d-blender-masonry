#!/bin/bash
# Substance stand for paving and timber: plans (all kinds of paving.yaml and timber.yaml) → the "paving" and "timber" steps on the layer of its own scene →
# "before" frames → covering substance (bl_cover) → "after" frames → sheets. From the project root through guard (Eevee):
#   .venv/bin/python -m core.services.guard heavy --wait 3600 --what "textures: paving and timber" -- bash components/constructions/textures/bench/ground/run.sh out/textures/ground [paving:cobble,timber:fachwerk] [res=1280x720]
set -e
ROOT=$(cd "$(dirname "$0")" && pwd); while [ ! -d "$ROOT/core" ] && [ "$ROOT" != "/" ]; do ROOT=$(dirname "$ROOT"); done
cd "$ROOT"; PY="$ROOT/.venv/bin/python"                                          # project root — where core/ is
OUT=${1:-out/textures/ground}; shift || true
mkdir -p "$OUT"
"$PY" -m components.constructions.paving.paving resolve "$OUT" kinds=all out="$OUT/plan_paving.json" > /dev/null
"$PY" -m components.constructions.timber.timber resolve "$OUT" kinds=all out="$OUT/plan_timber.json" > /dev/null
"$PY" -m components.constructions.textures.covers check
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/constructions/textures/bench/ground/bl_ground_tex.py -- "$OUT" "$@" > "$OUT/run.log" 2>&1 \
  || { grep -E "Error|Traceback|line [0-9]" "$OUT/run.log" | tail -20; exit 1; }
grep -E "__RESULT__" "$OUT/run.log" | cut -c1-600
"$PY" -m components.constructions.textures.bench.ground.sheet "$OUT"
