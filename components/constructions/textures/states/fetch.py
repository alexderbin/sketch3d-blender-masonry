"""Download of the state assets (sources.yaml): ambientCG — the archive <id>_<res>.zip (map images), Poly Haven — maps
through their API. → library/textures/states/<id>/<files>, meta.json (maps by meaning: color, opacity, normal, rough,
height, ao), index.json.
.venv/bin/python -m components.constructions.textures.states.fetch [--list]"""
import io
import json
import os
import sys
import urllib.request
import zipfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "library", "textures", "states")
UA = {"User-Agent": "sketch3d/1.0 (+local pipeline)"}
KIND = (("opacity", "opacity"), ("normalgl", "normal"), ("nor_gl", "normal"), ("roughness", "rough"), ("rough", "rough"),
        ("displacement", "height"), ("color", "color"), ("diffuse", "color"), ("ambientocclusion", "ao"))


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        return r.read()


def _kind(fn):
    f = fn.lower()
    return next((k for key, k in KIND if key in f), None)


def fetch(show=False):
    cfg = yaml.safe_load(open(os.path.join(HERE, "sources.yaml")))
    ip = os.path.join(OUT, "index.json"); idx = json.load(open(ip)) if os.path.exists(ip) else {}
    mb = 0.0
    acg = cfg.get("ambientcg") or {}
    for aid, spec in (acg.get("assets") or {}).items():
        d = os.path.join(OUT, aid); os.makedirs(d, exist_ok=True)
        maps = {}
        if not any(f.endswith(".jpg") for f in os.listdir(d)):
            b = _get(f"https://ambientcg.com/get?file={aid}_{acg.get('res', '1K-JPG')}.zip"); mb += len(b) / 1e6
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                for n in z.namelist():
                    if n.lower().endswith((".jpg", ".png")) and _kind(n):
                        open(os.path.join(d, os.path.basename(n)), "wb").write(z.read(n))
        for f in sorted(os.listdir(d)):
            k = _kind(f)
            if k and f.lower().endswith((".jpg", ".png")): maps.setdefault(k, os.path.relpath(os.path.join(d, f), ROOT))
        meta = {"id": aid, "source": "ambientCG", "license": "CC0", **spec, "maps": maps}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1); idx[aid] = meta
        print(aid, sorted(maps), flush=True)
    ph = cfg.get("polyhaven") or {}
    for aid, spec in (ph.get("assets") or {}).items():
        d = os.path.join(OUT, aid); os.makedirs(d, exist_ok=True)
        res = ph.get("res", "2k"); files = None; maps = {}
        info = json.loads(_get(f"https://api.polyhaven.com/info/{aid}"))
        for m in spec.get("maps", ["Diffuse", "nor_gl", "Rough"]):
            p = os.path.join(d, f"{aid}_{m}_{res}.jpg")
            if not os.path.exists(p):
                files = files or json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))
                b = _get(files[m][res]["jpg"]["url"]); mb += len(b) / 1e6; open(p, "wb").write(b)
            maps[_kind(m) or m] = os.path.relpath(p, ROOT)
        meta = {"id": aid, "source": "Poly Haven", "license": "CC0", "size_m": [round(x / 1000, 3) for x in info.get("dimensions", [1000, 1000])],
                **{k: v for k, v in spec.items() if k != "maps"}, "maps": maps}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1); idx[aid] = meta
        print(aid, sorted(maps), meta["size_m"], flush=True)
    json.dump(idx, open(ip, "w"), ensure_ascii=False, indent=1)
    print(f"downloaded {mb:.1f} MB")


if __name__ == "__main__":
    fetch("--list" in sys.argv)
