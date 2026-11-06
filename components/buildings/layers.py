"""Layers of the buildings component for its neighbors (formats — see the header of core/layers.py): occupancy
(occupancy.json) and anchors (anchors.json) — from the type plans: castle (castle.json), settlement (ensemble.json + host
castle), temple, edifice, classic (classic.json). Written by the "building" step (plan) into the run's scene;
neighbors read core.layers, not the buildings plans.
    .venv/bin/python -m components.buildings.layers <plan folder> <scene folder>   → the number of elements and anchors"""
import json
import math
import os
import sys

from core import layers as L


def _circle(c, r, n=16):
    return [[round(c[0] + r * math.cos(2 * math.pi * i / n), 3), round(c[1] + r * math.sin(2 * math.pi * i / n), 3)] for i in range(n)]


def _castle(C, pre, items, anchors, z):
    S = C.get("site") or {}
    if S.get("outline"): items.append({"kind": "site", "poly": S["outline"], "of": pre})
    for w in C.get("wards") or []:
        if w.get("poly"): items.append({"kind": "yard", "poly": w["poly"], "of": pre, "id": w.get("id", ""), "level": w.get("level")})
    for w in C.get("walls") or []:
        pts = [p[:2] for p in w.get("pts") or []]
        if len(pts) < 2: continue
        t = w.get("t"); tw = max(t) if isinstance(t, list) and t else float(t or 2.0)
        items.append({"kind": "wall", "line": pts + ([pts[0]] if w.get("closed") else []), "w": round(tw + 1.0, 2), "of": pre})
    for t in ([C["keep"]] if isinstance(C.get("keep"), dict) and C["keep"].get("xy") else []) + list(C.get("towers") or []):   # keep first
        items.append({"kind": "tower", "poly": _circle(t["xy"], float(t.get("r", 3.0)) + 0.5), "of": pre, "xy": t["xy"], "r": t.get("r")})
    for b in C.get("buildings") or []:                                  # yard buildings: base, floor, cornice, storeys, fachwerk (from a given storey)
        if b.get("poly"): items.append({"kind": "building", "poly": b["poly"], "of": pre, "type": b.get("kind", "building"),
                                        **{k: b[k] for k in ("base", "floor", "eave", "storeys", "fachwerk") if b.get(k) is not None}})
    if len(C.get("road") or []) >= 2: items.append({"kind": "road", "line": C["road"], "w": 4.0, "of": pre})   # points x, y, z
    c, top = S.get("centre"), S.get("top")
    if c: anchors[f"{pre}.centre"] = [c[0], c[1], float(top[2]) if top and len(top) > 2 else z(c)]
    for i, g in enumerate(C.get("gates") or []):
        anchors[f"{pre}.gate" + (f".{i}" if i else "")] = [g["xy"][0], g["xy"][1], float(g["z"]) if g.get("z") is not None else z(g["xy"])]
    k = C.get("keep")
    if k: anchors[f"{pre}.keep"] = [k["xy"][0], k["xy"][1], float(k.get("base", 0.0)) + 0.5 * (float(k.get("top", 0.0)) - float(k.get("base", 0.0)))]


def _town(E, pre, items, anchors, z):
    for h in E.get("houses") or []:
        F = h.get("F")
        if not F: continue
        it = {"kind": "house", "poly": F, "of": pre, "type": h.get("kind"),
              "h": round(float(h.get("h0", 2.5)) + float(h.get("floors", 1)) * float(h.get("fh", 2.5)), 2)}   # height to the cornice — the shadow
        if h.get("z0") is not None:                                     # walls: base, floor, cornice, storeys, fachwerk (from a given storey)
            fl = float(h["z0"]); n = max(1, int(h.get("floors", 1)))
            it.update(base=float(h.get("zb", fl)), floor=fl, eave=round(fl + float(h.get("h0", 2.8)) + (n - 1) * float(h.get("fh", 2.8)), 3),
                      storeys=n, fachwerk=h.get("fach") or None)
        de = h.get("door_edge", h.get("front")); de = None if de in (None, "None") else int(de)
        if de is not None and 0 <= de < len(F):
            (x0, y0), (x1, y1) = F[de][:2], F[(de + 1) % len(F)][:2]; it["door"] = [round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3)]
        items.append(it)
    for p in E.get("plots") or []:
        if not p.get("poly"): continue
        P_, T_, W_ = (list(p.get("influence") or []) + [0.0, 0.3, 0.5])[:3]          # the plot's influences: power, trade, wealth
        items.append({"kind": "plot", "poly": p["poly"], "of": pre, "power": P_, "trade": T_, "wealth": W_})
    deg = {}
    for e in (E.get("streets") or {}).get("edges") or []:
        if len(e.get("pts") or []) >= 2: items.append({"kind": "street", "line": [q[:2] for q in e["pts"]], "w": float(e.get("w", 4.0)), "of": pre})
        for k in (e.get("a"), e.get("b")): deg[k] = deg.get(k, 0) + 1
    for i, nd in enumerate((E.get("streets") or {}).get("nodes") or []):   # street ends — the starts of paths into the countryside (trails, grazing)
        q = nd if isinstance(nd, (list, tuple)) else (nd.get("p") or nd.get("pt") or [nd.get("x"), nd.get("y")])
        if deg.get(i, 0) <= 1 and q and q[0] is not None: anchors[f"{pre}.end.{i}"] = [q[0], q[1], z(q)]
    for i, q in enumerate(E.get("squares") or []):
        for poly in q.get("polys") or []:
            items.append({"kind": "square", "poly": poly, "of": pre})
        P = (q.get("polys") or [[]])[0]
        if P: c = [sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P)]; anchors[f"{pre}.square.{i}"] = [c[0], c[1], z(c)]
    w = E.get("wall")
    if isinstance(w, dict) and len(w.get("pts") or []) >= 2: items.append({"kind": "wall", "line": [p[:2] for p in w["pts"]], "w": 3.0, "of": pre})
    S = E.get("site") or {}
    if S.get("boundary"): items.append({"kind": "town", "poly": S["boundary"], "of": pre})   # the boundary of the intent is not occupancy: the reader chooses the kinds
    if S.get("centre"): anchors[f"{pre}.centre"] = [S["centre"][0], S["centre"][1], z(S["centre"])]
    for i, g in enumerate(E.get("gates") or []): anchors[f"{pre}.gate.{i}"] = [g[0], g[1], z(g)]


def _single(C, pre, items, anchors, z):
    """Temple, edifice: the footprint is a circle by area, the anchor is the center."""
    S = C.get("site") or {}; c = S.get("centre")
    if not c: return
    r = math.sqrt(max(float(S.get("area_m2") or 400.0), 1.0) / math.pi) * 1.15
    items.append({"kind": "building", "poly": _circle(c, r, 24), "of": pre}); anchors[f"{pre}.centre"] = [c[0], c[1], z(c)]


def _classic(K, pre, items, anchors, z):
    """Classic (a single building or a showcase of kinds): occupancy — the outline of each program (the type's footprint via the
    entry: an L × W frame along the facade axis, a tholos — a circle), the anchor — the middle of all outlines."""
    from components.buildings import buildings as B
    footprint = B.module("classics").footprint
    pts = []
    for P in K.get("buildings") or []:
        poly = [[round(float(q[0]), 3), round(float(q[1]), 3)] for q in (P.get("fp") or footprint(P))]
        items.append({"kind": "building", "poly": poly, "of": pre}); pts += poly
    if pts:
        c = [(min(q[0] for q in pts) + max(q[0] for q in pts)) / 2, (min(q[1] for q in pts) + max(q[1] for q in pts)) / 2]
        anchors[f"{pre}.centre"] = [round(c[0], 3), round(c[1], 3), z(c)]


def write(plan_dir, scene):
    """Occupancy and anchors of a building plan → occupancy.json, anchors.json in the scene. → (elements, anchors)."""
    try:
        from core.surface.ground import Ground
        G = Ground(scene); z = lambda p: round(float(G.height(float(p[0]), float(p[1]))), 3)
    except Exception:                                                  # noqa: BLE001 — without terrain the anchor height is 0
        z = lambda p: 0.0
    items, anchors = [], {}
    j = lambda d, f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else None
    E = j(plan_dir, "ensemble.json"); C = None
    if E is not None:
        _town(E, "town", items, anchors, z)
        src = os.path.join(plan_dir, "influences.npz")                        # the settlement's influence raster as a layer (trade at facades and squares)
        if os.path.exists(src):
            import shutil; shutil.copy(src, L.path(scene, "influence"))
        hf = (E.get("host") or {}).get("folder")
        C = j(hf, "castle.json") if hf else None
        if C is not None: _castle(C, "castle", items, anchors, z)
    elif j(plan_dir, "castle.json") is not None:
        _castle(j(plan_dir, "castle.json"), "castle", items, anchors, z)
    for f, pre in (("temple.json", "temple"), ("edifice.json", "edifice")):
        P1 = j(plan_dir, f)
        if P1 is not None: _single(P1, pre, items, anchors, z)
    K = j(plan_dir, "classic.json")
    if K is not None: _classic(K, "classic", items, anchors, z)
    pal = next((P.get("colors") for P in (j(plan_dir, f) for f in ("castle.json", "edifice.json", "temple.json")) if P and P.get("colors")), None)
    if pal is None and E is not None and C is not None: pal = C.get("colors")       # a settlement takes the palette of its host castle
    L.write(scene, "occupancy", {"items": items, "palette": pal or {}}); L.write(scene, "anchors", anchors)
    return len(items), len(anchors)


if __name__ == "__main__":
    n, a = write(sys.argv[1], sys.argv[2])
    print(f"occupancy: {n} elements; anchors: {a}")
