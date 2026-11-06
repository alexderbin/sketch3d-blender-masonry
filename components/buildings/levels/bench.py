"""Levels bench: each case over a matrix of seeds — the checks (strata.check) and the connectivity of the walkable surface
(walk.report), without a mesh and Blender. A lesson from the castles: a matrix finds what a single example does not show.

.venv/bin/python -m components.buildings.levels.bench [cases=a,b] [seeds=1-8]   → a table and failures (exit code 1 if any)"""
import sys
import time

from components.buildings.levels import cases as C
from components.buildings.levels import strata as S
from components.buildings.levels import walk as WK

SEEDED = {"slope_row", "strata_town", "hive", "hillside", "matera", "raised_street", "plate", "viaduct", "kowloon", "underground", "arena", "old_town", "cliff_town", "living_bridge"}


def run(names, seeds):
    bad = []; rows = []
    for name in names:
        for sd in (seeds if name in SEEDED else [None]):
            t0 = time.time()
            try:
                W, _ = C.CASES[name](**({"seed": sd} if sd is not None else {}))
                chk = S.check(W)
                big = (W.block[2] - W.block[0]) * (W.block[3] - W.block[1]) > 4e4
                rep, *_ = WK.report(W, step=2.0 if big else 1.0)
                ok = not chk and rep["unreachable_n_storeys"] == 0 and not rep["unreachable_decks"] and rep["n_parts"] == 1
                pr = next((n.split(": ", 1)[1] for n in W.notes if n.startswith("unreachable removed")), "—")
                rows.append((name, sd, len(chk), rep["n_parts"], rep["unreachable_n_storeys"], len(rep["unreachable_decks"]), time.time() - t0, pr))
                if not ok: bad.append((name, sd, chk[:3], rep["examples"][:3], rep["unreachable_decks"][:3]))
            except Exception as e:                                           # noqa: BLE001 — the bench collects all failures
                rows.append((name, sd, -1, -1, -1, -1, time.time() - t0, "—")); bad.append((name, sd, [f"{type(e).__name__}: {e}"], [], []))
    print(f"{'case':14} {'seed':>4} {'fail':>4} {'parts':>4} {'unr.fl':>6} {'unr.dk':>5} {'s':>5}  removed")
    for r in rows: print(f"{r[0]:14} {str(r[1]):>4} {r[2]:>4} {r[3]:>4} {r[4]:>6} {r[5]:>5} {r[6]:>5.1f}  {r[7]}")
    print(f"failures {len(bad)} of {len(rows)}")
    for b in bad: print("  FAIL", b)
    return bad


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    names = kw.get("cases", ",".join(C.CASES)).split(",")
    a, b = (kw.get("seeds", "1-8").split("-") + [None])[:2]
    seeds = list(range(int(a), int(b or a) + 1))
    sys.exit(1 if run(names, seeds) else 0)
