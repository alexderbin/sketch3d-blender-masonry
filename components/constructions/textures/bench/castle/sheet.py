"""Sheet of castle close-ups (bl_views.py): the views at full size two per row and the pipeline frames (frames) small.
.venv/bin/python -m components.constructions.textures.bench.castle.sheet <run> → <run>/views/sheet_views.jpg"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw

from components.constructions.textures.bench.roofs.sheet import font, label

RUN = sys.argv[1] if len(sys.argv) > 1 else "out/textures/castle"
V = os.path.join(RUN, "views"); F2 = font(26)
NAMES = {"yard": "courtyard at human eye level", "fachwerk": "half-timbered facade", "roof": "roof from wall height", "road": "road by the gate",
         "wall": "outer wall up close"}
R = json.load(open(os.path.join(V, "views.json")))
W, H = 960, 540; cols = 2; names = [n for n in NAMES if n in R and os.path.exists(os.path.join(V, f"{n}.png"))]
frames = sorted(glob.glob(os.path.join(RUN, "*_frames", "**", "*.png"), recursive=True))[:8]
rows = (len(names) + cols - 1) // cols; fr = (len(frames) + 3) // 4
S = Image.new("RGB", (cols * (W + 6), 48 + rows * (H + 6) + fr * (H // 2 + 6)), (255, 255, 255))
ImageDraw.Draw(S).text((10, 10), "Construction substance on the castle: masonry, roofing, paving, timber (the 'textures' step)", fill=(0, 0, 0), font=F2)
for i, n in enumerate(names):
    S.paste(label(Image.open(os.path.join(V, f"{n}.png")).convert("RGB").resize((W, H), Image.LANCZOS), NAMES[n]), ((i % cols) * (W + 6), 48 + (i // cols) * (H + 6)))
y0 = 48 + rows * (H + 6)
for i, p in enumerate(frames):
    S.paste(Image.open(p).convert("RGB").resize((W // 2 - 3, H // 2), Image.LANCZOS), ((i % 4) * (W // 2 + 3), y0 + (i // 4) * (H // 2 + 6)))
S.save(os.path.join(V, "sheet_views.jpg"), quality=88); print("sheet:", os.path.join(V, "sheet_views.jpg"))
