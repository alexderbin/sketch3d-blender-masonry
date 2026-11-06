"""Sheets of the paving and timber substance stand (bl_ground_tex.py): "before | after" pairs by kind
(sheet_<part>_<view>.jpg) and 1:1 crops of the "after" close-up (crop_macro.jpg). The caption is the kind, the
substance, the scan.
.venv/bin/python -m components.constructions.textures.bench.ground.sheet <out>"""
import json
import os
import sys

from PIL import Image, ImageDraw

from components.constructions.textures.bench.roofs.sheet import font, label

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/textures/ground"
F2 = font(26)
R = json.load(open(os.path.join(OUT, "run_ground_tex.json")))
COV = json.load(open(os.path.join(HERE, "..", "..", "covers.json")))["substances"]


def cap(key):
    s = R[key]["substance"]; return f"{R[key]['title']} · {s} · {COV.get(s, {}).get('scan', '—')}"


def stem(key): return key.replace(":", "_")


def grid(tiles, head, path, cols, W=576, H=324):
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * (W + 6), 48 + rows * (H + 6)), (255, 255, 255)); ImageDraw.Draw(S).text((10, 10), head, fill=(0, 0, 0), font=F2)
    for i, (p, t) in enumerate(tiles):
        if not os.path.exists(p): continue
        S.paste(label(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), t), ((i % cols) * (W + 6), 48 + (i // cols) * (H + 6)))
    S.save(path, quality=88); print("sheet:", path)


for part, views in (("paving", ("court", "macro", "road")), ("timber", ("facade", "macro", "fence"))):
    keys = [k for k in R if k.startswith(part + ":")]
    for v in views:
        tiles = []
        for k in keys:
            if os.path.exists(os.path.join(OUT, f"{stem(k)}_{v}_1.png")):
                tiles += [(os.path.join(OUT, f"{stem(k)}_{v}_0.png"), f"before · {R[k]['title']}"), (os.path.join(OUT, f"{stem(k)}_{v}_1.png"), f"after · {cap(k)}")]
        if tiles: grid(tiles, f"Before | after — {part}: {v}", os.path.join(OUT, f"sheet_{part}_{v}.jpg"), 4)
crops = []
for k in R:
    p = os.path.join(OUT, f"{stem(k)}_macro_1.png")
    if os.path.exists(p):
        im = Image.open(p).convert("RGB"); w, h = im.size
        crops.append(label(im.crop((w // 2 - 240, h // 2 - 135, w // 2 + 240, h // 2 + 135)), cap(k)))
if crops:
    cols = 4; rows = (len(crops) + cols - 1) // cols
    S = Image.new("RGB", (cols * 486, 48 + rows * 276), (255, 255, 255)); ImageDraw.Draw(S).text((10, 10), "Close-up 1:1 (frame center)", fill=(0, 0, 0), font=F2)
    for i, c in enumerate(crops): S.paste(c, ((i % cols) * 486, 48 + (i // cols) * 276))
    S.save(os.path.join(OUT, "crop_macro.jpg"), quality=90); print("sheet:", os.path.join(OUT, "crop_macro.jpg"))
