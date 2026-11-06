"""Reference of the masonry part of the Constructions component: the result is fixed and reproducible.

1. Checks of the trial layout (check_layout.py) and of the kinds (check_types.py).
2. Layout fingerprints — the number of stones and a hash of the polygons (4 digits) — against golden.json: all the packs of library/masonry/
   packs.yaml on a piece of the kinds sheet's wall (a corner 3 + 6 m, a window with a sill) and wall 0 of the castle cx_order_castle (both sides;
   the input is frozen by the surfaces layer bench/castle_cx_surfaces.json (wall 0 of the plan, written by Buildings)).
   A mismatch means a change altered the result: if it is deliberate, rebuild the reference with the --save key.
3. The building plan (masonry.py resolve, constructions.yaml): parts, packs, step-line keys, sections of the following steps — as they are;
   the towers knob is the tower pack, 0|no — towers without masonry (the plan has no tower pack), anything else — a failure with a reason.
4. With the --blender key — the step contract in Blender (bl_contract.py): attributes and output slots, block numbers, a rebuild repeating
   one-to-one, the textures step accepts the masonry output.

.venv/bin/python -m components.constructions.masonry.bench.bench [--save] [--blender]   → lines OK/FAIL, __RESULT__, exit code = number of failures
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import core.paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is
from components.constructions.masonry import masonry as M  # noqa: E402

PY = os.path.join(ROOT, ".venv", "bin", "python")
GOLD = os.path.join(HERE, "golden.json")
SURF = os.path.join(HERE, "castle_cx_surfaces.json")                            # input frozen: the surfaces layer of the castle cx_order_castle, wall 0
res = []


def check(name, ok, value):
    res.append({"check": name, "ok": bool(ok), "value": value}); print(("OK   " if ok else "FAIL ") + name, value, flush=True)


def sub(script):
    r = subprocess.run([PY, os.path.join(HERE, script)], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("__RESULT__")), "__RESULT__ {}")
    return json.loads(line.split(" ", 1)[1]), r.returncode


def fingerprint(L):
    h = hashlib.sha1()
    for s in L["stones"] + L["features"]:
        h.update(json.dumps([[round(x, 4), round(y, 4)] for x, y in s["poly"]]).encode())
    return dict(n_stones=len(L["stones"]), n_surrounds=len(L["features"]), digest=h.hexdigest()[:16])


def prints():
    out = {}
    S = M.Strip([(-3.0, 0.0), (0.0, 0.0), (0.0, 6.0)], [0.0] * 3, [4.5] * 3, 1.0, side=1, closed=False)
    for name, pk in M.PACKS.items():
        out[f"kind:{name}"] = fingerprint(M.layout(S, pk, gates=[dict(u=1.5, w=1.0, z=1.0, h=1.9, sill=True)], seed=3, putlogs=False))
    if os.path.exists(SURF):                                                      # the castle wall comes from the surfaces layer (the castle plan is not read)
        lay = json.load(open(SURF))
        for side, (it, pk, seed) in enumerate(zip(sorted(lay["items"], key=lambda i: i["side"]), ("ashlar", "rubble"), (1, 2))):
            out[f"castle:wall0:{'outer' if side == 0 else 'inner'}"] = fingerprint(M.layout(M.strip_from(it["strip"]), M.PACKS[pk], seed=seed, gaps=it["gaps"],
                                                                                         putlogs=M.PACKS[pk].get("putlog") is not None))
    return out


def main():
    save = "--save" in sys.argv
    for script in ("check_layout.py", "check_types.py"):
        r, rc = sub(script)
        check(f"{script}", rc == 0, r)
    fp = prints()
    if save or not os.path.exists(GOLD):
        json.dump(dict(version=M.LIBRARY.get("version"), imprints=fp), open(GOLD, "w"), ensure_ascii=False, indent=1)
        print("reference written:", GOLD)
    gold = json.load(open(GOLD))["imprints"]
    diff = {k: (gold.get(k), v) for k, v in fp.items() if gold.get(k) != v}
    check("layout fingerprints = reference", not diff and set(gold) == set(fp), {"n_layouts": len(fp), "otherwise": sorted(diff)[:6],
                                                                              "not in the reference": sorted(set(fp) - set(gold))[:4]})
    with tempfile.TemporaryDirectory() as d:                                     # building plan: description + step-line keys
        open(os.path.join(d, "constructions.yaml"), "w").write("masonry:\n  packs: {outer: bossage}\n  detail: all\ntextures: {wall: rock_surface}\n")
        plan = M.resolve(d, {"inner": "rubble_random", "Tlo": "5"})
        k = plan["masonry"]
        check("plan: description + keys", k["parts"] == {"outer": "bossage", "inner": "rubble_random", "towers": "ashlar"} and k["detail"] == "all"
              and k["thresholds"][0] == 5.0 and set(plan["packs"]) == {"bossage", "rubble_random", "ashlar"}
              and plan.get("textures") == {"wall": "rock_surface"}, {"parts": k["parts"], "detail": k["detail"], "thresholds": k["thresholds"]})
    with tempfile.TemporaryDirectory() as d:                                     # towers knob: the tower pack; 0|no — towers without masonry
        off = [M.resolve(d, {"towers": v})["masonry"]["parts"] for v in ("0", "no")]
        on = M.resolve(d, {"towers": "bossage"})["masonry"]["parts"].get("towers")
        open(os.path.join(d, "constructions.yaml"), "w").write("masonry:\n  packs: {towers: 0}\n")
        desc_off = "towers" not in M.resolve(d, {})["masonry"]["parts"]
        try: M.resolve(d, {"towers": "1"}); one = "no failure"
        except SystemExit as e: one = str(e)
        check("plan: towers — the tower pack, 0|no — without masonry", all("towers" not in p for p in off) and on == "bossage" and desc_off
              and "no pack ['1']" in one, {"0|no": off[0], "bossage": on, "description towers: 0": desc_off, "towers=1": one[:30]})
    if "--blender" in sys.argv:
        r = subprocess.run(["blender", "-b", "--python-exit-code", "1", "-P", core.paths.RUN, "--", os.path.join(HERE, "bl_contract.py")], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("OK   ") or line.startswith("FAIL "):
                name, _, val = line[5:].partition(" {")
                check("Blender: " + name.strip(), line.startswith("OK"), "{" + val if val else "")
        if r.returncode and not any(l.startswith("FAIL") for l in r.stdout.splitlines()):
            check("Blender: run", False, r.stderr[-400:])
    ok = sum(x["ok"] for x in res)
    print("__RESULT__", json.dumps({"ok": ok, "total": len(res)}, ensure_ascii=False))
    sys.exit(len(res) - ok)


main()
