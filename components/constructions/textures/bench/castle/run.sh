#!/bin/bash
# Construction substance on the castle: the castle_textures.yaml pipeline (relief → castle → camera → masonry, roofing, paving, timber —
# all elements → textures → daylight) → close-ups from the faces of construction objects (bl_views.py, the .blend of the "light" step) → sheet.
# From the project root through guard (heavy: all construction elements; on a rented machine — minutes):
#   .venv/bin/python -m core.services.guard heavy --wait 3600 --what "textures: castle" -- bash components/constructions/textures/bench/castle/run.sh out/textures/castle [views] [res=1280x720]
# views — only the close-ups from a finished run (the pipeline is not touched). The "frames" output fails on a rented machine —
# the pipeline runs up to the "light" step (--until light); the pipeline frames are made on the Mac.
set -e
ROOT=$(cd "$(dirname "$0")" && pwd); while [ ! -d "$ROOT/core" ] && [ "$ROOT" != "/" ]; do ROOT=$(dirname "$ROOT"); done
cd "$ROOT"; PY="$ROOT/.venv/bin/python"
OUT=${1:-out/textures/castle}; shift || true
if [ "$1" = "views" ]; then shift; else
  "$PY" -m core.manager.pipeline components/constructions/textures/bench/castle/castle_textures.yaml --out "$OUT" --until light | tail -3
fi
BL=$(ls -d "$OUT"/*_light 2>/dev/null | tail -1)/scene.blend
[ -f "$BL" ] || { echo "no .blend of the 'light' step: $BL"; exit 1; }
blender -b "$BL" --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/constructions/textures/bench/castle/bl_views.py -- "$OUT" "$@" > "$OUT/views.log" 2>&1 \
  || { grep -E "Error|Traceback|line [0-9]" "$OUT/views.log" | tail -20; exit 1; }
grep -E "__RESULT__" "$OUT/views.log" | cut -c1-400
"$PY" -m components.constructions.textures.bench.castle.sheet "$OUT"
