"""Sheets of the roofing substance stand (bl_roofs_tex.py): per view — the "after" frames of all kinds (sheet_<view>.jpg);
"before | after" pairs up close and in depth (sheet_pairs_macro.jpg, sheet_pairs_deep.jpg); 1:1 crops of the close-up
(crop_macro.jpg) — repetition and grain are visible only at full size. The caption is the kind, the substance, the scan.
.venv/bin/python -m components.constructions.textures.bench.roofs.sheet <out>"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/textures/roofs"


def font(size):
    """Arial (macOS), DejaVu (rented machine, Linux), otherwise the built-in font."""
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


F, F2 = font(18), font(26)
HEAD = {"macro": "Roof substance up close (2.5 m): its own piece of the scan per tile, the layout tone",
        "close": "Roof substance: hip and ridge", "all": "Roof substance: house, tower with a pyramid, round one with a cone; figure — 1.8 m",
        "deep": "Near roof and the same group 90 m away (the far one is a shell with a row pattern)"}


def label(im, t):
    d = ImageDraw.Draw(im); w = d.textlength(t, font=F)
    d.rectangle([4, 4, 12 + w, 28], fill=(255, 255, 255)); d.text((8, 6), t, fill=(0, 0, 0), font=F)
    return im


def cap(n):
    R, COV = DATA; s = R[n]["substance"]; return f"{R[n]['title']} · {s} · {COV.get(s, {}).get('scan', '—')}"


def grid(tiles, head, path, cols, W=576, H=324):
    rows = (len(tiles) + cols - 1) // cols
    S = Image.new("RGB", (cols * (W + 6), 48 + rows * (H + 6)), (255, 255, 255)); ImageDraw.Draw(S).text((10, 10), head, fill=(0, 0, 0), font=F2)
    for i, (p, t) in enumerate(tiles):
        if not os.path.exists(p): continue
        im = label(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS), t)
        S.paste(im, ((i % cols) * (W + 6), 48 + (i // cols) * (H + 6)))
    S.save(path, quality=88); print("sheet:", path)


DATA = None


def main():
    global DATA
    R = json.load(open(os.path.join(OUT, "run_roofs_tex.json")))
    DATA = (R, json.load(open(os.path.join(HERE, "..", "..", "covers.json")))["substances"])
    names = list(R)
    for v in ("macro", "close", "all", "deep"):
        grid([(os.path.join(OUT, f"{n}_{v}_1.png"), cap(n)) for n in names], HEAD[v], os.path.join(OUT, f"sheet_{v}.jpg"), 4)
    for v in ("macro", "deep"):
        tiles = []
        for n in names: tiles += [(os.path.join(OUT, f"{n}_{v}_0.png"), f"before · {R[n]['title']}"), (os.path.join(OUT, f"{n}_{v}_1.png"), f"after · {cap(n)}")]
        grid(tiles, f"Before | after — {HEAD[v]}", os.path.join(OUT, f"sheet_pairs_{v}.jpg"), 4)
    crops = []                                                                       # 1:1 crops of the center of the close-up
    for n in names:
        p = os.path.join(OUT, f"{n}_macro_1.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB"); w, h = im.size; c = im.crop((w // 2 - 240, h // 2 - 135, w // 2 + 240, h // 2 + 135))
            crops.append(label(c, cap(n)))
    if crops:
        cols = 4; rows = (len(crops) + cols - 1) // cols
        S = Image.new("RGB", (cols * 486, 48 + rows * 276), (255, 255, 255)); ImageDraw.Draw(S).text((10, 10), "Close-up 1:1 (frame center)", fill=(0, 0, 0), font=F2)
        for i, c in enumerate(crops): S.paste(c, ((i % cols) * 486, 48 + (i // cols) * 276))
        S.save(os.path.join(OUT, "crop_macro.jpg"), quality=90); print("sheet:", os.path.join(OUT, "crop_macro.jpg"))


if __name__ == "__main__":
    main()
