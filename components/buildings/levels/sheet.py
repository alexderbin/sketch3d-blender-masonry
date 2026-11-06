"""Clay sheet of the levels: images with captions in a grid, a title and lines of notes at the bottom (PIL).

.venv/bin/python -m components.buildings.levels.sheet <output.jpg> "<title>" <image>::<caption> … [cols=3] [w=760] [note=…]"""
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def sheet(out, title, items, cols=3, w=760, notes=(), h=None):
    """Cells of one size w × h (by default — by the most frequent aspect ratio); an image is fitted with margins."""
    big, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 17)
    ims = [(Image.open(p).convert("RGB"), cap) for p, cap in items]
    if h is None:
        from collections import Counter
        r = Counter(round(im.height / im.width, 2) for im, _ in ims).most_common(1)[0][0]; h = int(w * r)
    caps = [_wrap(cap, small, w - 4) for _, cap in ims]                     # a caption wider than the cell — wrapped by words
    notes = [line for n in notes for line in _wrap(n, small, cols * w + (cols - 1) * 8 - 4)]
    rows = (len(ims) + cols - 1) // cols; cap_h = 8 + 22 * max(len(c) for c in caps); top = 48; bot = 26 * len(notes) + (12 if notes else 0)
    S = Image.new("RGB", (cols * w + (cols + 1) * 8, top + rows * (h + cap_h + 8) + bot), (255, 255, 255)); d = ImageDraw.Draw(S)
    d.text((10, 10), title, fill=(20, 20, 20), font=big)
    for i, (im, cap) in enumerate(ims):
        r, c = divmod(i, cols); x = 8 + c * (w + 8); y = top + r * (h + cap_h + 8)
        k = min(w / im.width, h / im.height); t = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
        S.paste(t, (x + (w - t.width) // 2, y + (h - t.height) // 2))
        for j, line in enumerate(caps[i]): d.text((x + 2, y + h + 4 + 22 * j), line, fill=(30, 30, 30), font=small)
    for j, n in enumerate(notes): d.text((10, top + rows * (h + cap_h + 8) + 6 + 26 * j), n, fill=(40, 40, 40), font=small)
    S.save(out, quality=90); return out


def _wrap(text, font, width):
    out = [""]
    for word in text.split(" "):
        t = (out[-1] + " " + word).strip()
        if font.getlength(t) <= width or not out[-1]: out[-1] = t
        else: out.append(word)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]; out, title = a[0], a[1]; kw = {}; items = []; notes = []
    for x in a[2:]:
        if "::" in x: p, cap = x.split("::", 1); items.append((p, cap))
        elif x.startswith("note="): notes.append(x[5:])
        elif "=" in x: k, v = x.split("=", 1); kw[k] = int(v)
    print(sheet(out, title, items, notes=notes, **kw))
