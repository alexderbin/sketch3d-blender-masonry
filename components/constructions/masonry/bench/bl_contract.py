"""The contract of the masonry step in Blender (called by bench.py --blender; standalone: blender -b -P core/blender/run.py -- components/constructions/masonry/bench/bl_contract.py).
A piece of the kinds sheet's wall, three packs (ashlar, brick, cyclopean) as blocks with a body and one by camera (rows, a slab): attributes and output slots, block numbers,
a rebuild repeating one-to-one, the textures step (../../textures/bl_textures.run) accepts the masonry output. Prints OK/FAIL lines."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
from components.constructions.masonry import bl_stones as BS  # noqa: E402
from components.constructions.masonry import masonry as M  # noqa: E402

WORK = tempfile.mkdtemp(prefix="masonry_contract_")


def check(name, ok, value):
    print(("OK   " if ok else "FAIL ") + name, json.dumps(value, ensure_ascii=False, default=str), flush=True)


def plan(packs, **kw):
    args = [f"{k}={v}" for k, v in kw.items()]
    r = subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), "-m", "components.constructions.masonry.masonry", "resolve", WORK, "packs=all", *args],
                       capture_output=True, text=True, check=True)
    p = json.loads(r.stdout); p["packs"] = {n: p["packs"][n] for n in packs}; return p


def build(pack, view="all", cams=(), fpx=900.0):
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob, do_unlink=True)
    coll = bpy.data.collections.new("t"); bpy.context.scene.collection.children.link(coll)
    S = M.Strip([(-3.0, 0.0), (0.0, 0.0), (0.0, 6.0)], [0.0] * 3, [4.5] * 3, 1.0, side=1, closed=False)
    item = dict(tag=pack, strip=S.spec(), pack=pack, seed=3, sid0=1, gates=[dict(u=1.5, w=1.0, z=1.0, h=1.9, sill=True)], putlogs=False, body=True)
    st, objs = BS.run([item], plan([pack], detail=view), coll, cams=cams, fpx=fpx, work=WORK, log=lambda *a: None)
    return st, objs[pack]


def vhash(ob):
    v = np.empty(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get("co", v)
    return hashlib.sha1(np.round(v, 5).tobytes()).hexdigest()[:16]


for pack in ("ashlar", "brick_english", "cyclopean"):
    st, o = build(pack)
    ob = o["blocks"]; me = ob.data
    at = {a.name: a.domain for a in me.attributes}
    slots = [s.material.name for s in ob.material_slots]
    sid = np.empty(len(me.polygons), np.int32); me.attributes["sk_stone"].data.foreach_get("value", sid)
    role = np.empty(len(me.polygons), np.int32); me.attributes["sk_role"].data.foreach_get("value", role)
    check(f"{pack}: attributes and slots", at.get("sk_role") == "FACE" and at.get("sk_stone") == "FACE" and "Col" in at
          and all(s.startswith("sk_") for s in slots) and "body" in o, {"slots": slots, "attributes": sorted(at)})
    n_blocks = len(set(sid[(sid > 0) & (sid < 500_000)].tolist()))
    check(f"{pack}: a block number on every stone", n_blocks == st["blocks"], {"n_stones": st["stones"], "n_blocks": st["blocks"], "n_numbers": n_blocks})
    h1 = vhash(ob); _, o2 = build(pack); h2 = vhash(o2["blocks"])
    check(f"{pack}: a rebuild repeats one-to-one", h1 == h2, {"digest": h1})

st, o = build("ashlar", view="cam", cams=[(2.0, -1.9, 1.5)], fpx=60.0)      # a small frame: a band near the camera, rows farther away
check("by camera: blocks up close, rows and a slab at a distance", "plate" in o and 0 < st["blocks"] < st["stones"], {"n_stones": st["stones"], "n_blocks": st["blocks"]})

surf = os.path.join(ROOT, "library", "textures", "surface", "rock_surface")
if os.path.isdir(surf):
    from components.constructions.textures import bl_textures as TX
    _, o = build("ashlar")
    info = TX.run(bpy.context.scene, {"wall": "rock_surface", "finish": "rock_01", "mortar": "sand"})
    names = {s.material.name.split(".")[0] for s in o["blocks"].material_slots}
    check("textures accept the masonry output", info["roles"].get("wall", 0) >= 1 and any(n.startswith("sk_surf_wall") for n in names)   # name format: sk_surf_<role>_<scan>
          and "sk_rnd" in o["blocks"].data.attributes, {"roles": info["roles"], "slots": sorted(names)})
else:
    check("textures accept the masonry output", True, "no scans (core/surface/surface_tex.py fetch) — skipped")
