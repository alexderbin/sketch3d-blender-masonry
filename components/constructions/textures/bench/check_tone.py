"""Masonry tone is preserved: the average color of the stone substance (stand albedo frames: Base Color emission,
everything except the stones is black) of a variant against "before" (the Col tone of the masonry without substance).
The scan pattern is normalized to 1 and does not touch the weathering spread of the tone, only edge wear and chips are
lighter → a tolerance per channel.
.venv/bin/python -m components.constructions.textures.bench.check_tone <shots folder> [variant=v2] [tolerance=0.06] → JSON, exit code 1 — violated"""
import json
import os
import sys

import numpy as np
from PIL import Image


def mean_rgb(path):
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)      # the shot is sRGB (the Standard view)
    m = lin.max(2) > 0.004                                              # stones; black (mass, slab, sky) is skipped
    return lin[m].mean(0), float(m.mean())


def main(src, var="v2", tol=0.06):
    out, ok = {}, True
    for view in ("near", "close"):
        p0, p1 = os.path.join(src, f"before_albedo_{view}.png"), os.path.join(src, f"{var}_albedo_{view}.png")
        if not (os.path.exists(p0) and os.path.exists(p1)):
            continue
        (c0, f0), (c1, f1) = mean_rgb(p0), mean_rgb(p1)
        r = c1 / np.maximum(c0, 1e-6)
        good = bool(np.all(np.abs(r - 1) <= tol))
        ok &= good
        out[view] = {"before": [round(float(x), 4) for x in c0], var: [round(float(x), 4) for x in c1],
                     "ratio": [round(float(x), 3) for x in r], "frac_stone": [round(f0, 3), round(f1, 3)], "within_tolerance": good}
    out["result"] = ok and bool(out)
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["result"] else 1


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    sys.exit(main(sys.argv[1], kw.get("variant", "v2"), float(kw.get("tolerance", 0.06))))
