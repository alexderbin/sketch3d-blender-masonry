"""Editing the ground under an ensemble's buildings — the buildings core (the terrain edits under buildings belong to the
buildings side; refreshing the ground belongs to the site component). Streets follow a smoothed surface (level across),
squares are a least-squares plane, a platform (hive) sits at a given elevation; the edges are blended. Site masks: road
(the street bed), pave (paved), pad (squares and houses — the site component plants no undergrowth there). numpy and
OpenCV only; the caller supplies the geometry (shapely)."""
import os
import sys
from components.buildings import arch_parts
import shutil

import cv2
import numpy as np


def _parts(g, min_area=0.01):
    if g is None or g.is_empty: return []
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area]


def raster(geom, x0, y0, res, shape):
    """A geometry mask on the terrain grid (uint8 0/255; holes are empty)."""
    M = np.zeros(shape, np.uint8)
    for p in _parts(geom):
        cv2.fillPoly(M, [np.round((np.asarray(p.exterior.coords) - (x0, y0)) / res).astype(np.int32)], 255)
        for h in p.interiors: cv2.fillPoly(M, [np.round((np.asarray(h.coords) - (x0, y0)) / res).astype(np.int32)], 0)
    return M


def _feather(M, res, width):
    d = cv2.distanceTransform((~M).astype(np.uint8) * 255, cv2.DIST_L2, 5) * res; w = np.clip(1 - d / width, 0, 1)
    return w * w * (3 - 2 * w)


def grade(base, out, streets_g, squares, pave, pads, flat=None):
    """The settlement terrain from the site base → out (relief.npz, location.json, the site texture): streets, squares, a
    platform, masks."""
    z = dict(np.load(os.path.join(base, "relief.npz"))); H = z["near"].astype(np.float64); x0, y0, res = (float(v) for v in z["near_o"])
    shp = H.shape
    Ms = raster(streets_g, x0, y0, res, shp)
    if Ms.any():
        S = cv2.GaussianBlur(H, (0, 0), 4.0 / res); w = _feather(Ms > 0, res, 1.5); H = H * (1 - w) + S * w
    for sq in squares:
        M = raster(sq, x0, y0, res, shp) > 0
        if M.sum() < 8: continue
        iy, ix = np.nonzero(M); X = x0 + ix * res; Y = y0 + iy * res
        cf = np.linalg.lstsq(np.column_stack([X, Y, np.ones_like(X)]), H[iy, ix], rcond=None)[0]
        gy, gx = np.mgrid[0:shp[0], 0:shp[1]]; Pz = cf[0] * (x0 + gx * res) + cf[1] * (y0 + gy * res) + cf[2]
        w = _feather(M, res, 2.5); H = H * (1 - w) + Pz * w
    if flat is not None:
        geom, zf = flat; w = _feather(raster(geom.buffer(2.0), x0, y0, res, shp) > 0, res, 6.0); H = H * (1 - w) + zf * w
    z["near"] = H.astype(z["near"].dtype)
    def put(k, M):
        old = z[k].astype(np.uint8) if k in z else np.zeros(shp, np.uint8); z[k] = np.maximum(old, M).astype(np.uint8)
    put("road", Ms); put("pave", raster(pave, x0, y0, res, shp)); put("pad", raster(pads, x0, y0, res, shp))
    np.savez_compressed(os.path.join(out, "relief.npz"), **z)
    shutil.copy(os.path.join(base, "location.json"), os.path.join(out, "location.json"))
    for f in ("relief_tex.png", "relief_tex.json"):
        if os.path.exists(os.path.join(base, f)): shutil.copy(os.path.join(base, f), os.path.join(out, f))


def add_pad(out, geom):
    """The terrain's pad mask is extended with house outlines."""
    z = dict(np.load(os.path.join(out, "relief.npz"))); x0, y0, res = (float(v) for v in z["near_o"])
    M = raster(geom, x0, y0, res, z["near"].shape); z["pad"] = np.maximum(z.get("pad", np.zeros_like(M)).astype(np.uint8), M)
    np.savez_compressed(os.path.join(out, "relief.npz"), **z)


def reshape(out, items):
    """Editing the settlement terrain after grade (landmarks): items — [{"geom", "mode": cut | set, "z": a number or z(X, Y),
    "feather": m}]. cut — the ground is no higher than z inside (a theatre cavea in a slope); set — a platform at z, the edges
    blended over feather outside (an acropolis platform, an orchestra). The pad mask is extended."""
    z = dict(np.load(os.path.join(out, "relief.npz"))); H = z["near"].astype(np.float64); x0, y0, res = (float(v) for v in z["near_o"]); shp = H.shape
    gy, gx = np.mgrid[0:shp[0], 0:shp[1]]; X = x0 + gx * res; Y = y0 + gy * res; pad = np.zeros(shp, np.uint8)
    for it in items:
        M = raster(it["geom"], x0, y0, res, shp) > 0
        if not M.any(): continue
        Zt = it["z"](X, Y) if callable(it["z"]) else np.full(shp, float(it["z"]))
        if it["mode"] == "cut": H = np.where(M, np.minimum(H, Zt), H)
        else:
            w = _feather(M, res, float(it.get("feather", 3.0))); H = H * (1 - w) + Zt * w
        pad = np.maximum(pad, M.astype(np.uint8) * 255)
    z["near"] = H.astype(z["near"].dtype); z["pad"] = np.maximum(z.get("pad", np.zeros(shp, np.uint8)).astype(np.uint8), pad)
    np.savez_compressed(os.path.join(out, "relief.npz"), **z)
