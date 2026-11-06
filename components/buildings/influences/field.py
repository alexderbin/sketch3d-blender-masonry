"""Influence field as a raster — numpy only (the project's Python and Blender). The plan folder's file `influences.npz`:
x0, y0 — the grid corner (m, center of the first cell), step — the step (m); P, W — (tiers, ny, nx), T — (ny, nx): uint8,
0–255 ↔ 0–1; z0, dz — hive tiers (the floor of tier k is z0 + k·dz; without tiers — a single tier); L — the district
reading thresholds (power, trade, trade_W, nobility, poor). Grid row j — y0 + j·step (south → north), column i — x0 + i·step.
  sample(F, X, deck=None, z=None) → (P, T, W) at points (bilinear; the tier by index or by height z);
  label(P, T, W, L) — the reading: power, trade, craft, nobility, middle, poor."""
import os

import numpy as np

LABELS = ("power", "trade", "craft", "nobility", "middle", "poor")
COLORS = {"power": (128, 70, 150), "trade": (226, 136, 48), "craft": (150, 96, 60), "nobility": (238, 204, 92),
          "middle": (206, 196, 172), "poor": (104, 118, 142)}
FILE = "influences.npz"
KEYS = ("power", "trade", "trade_W", "nobility", "poor")


def _u8(a): return np.clip(np.round(np.asarray(a, float) * 255.0), 0, 255).astype(np.uint8)


def save(path, x0, y0, step, P, T, W, L, z0=0.0, dz=0.0):
    """P, W — (tiers, ny, nx) or (ny, nx); T — (ny, nx); L — the district thresholds (a dict)."""
    P = np.asarray(P, float); W = np.asarray(W, float)
    if P.ndim == 2: P = P[None]; W = W[None]
    np.savez_compressed(path, x0=float(x0), y0=float(y0), step=float(step), P=_u8(P), T=_u8(T), W=_u8(W),
                        z0=float(z0), dz=float(dz), L=np.array([float(L[k]) for k in KEYS]))


def load(path):
    if os.path.isdir(path): path = os.path.join(path, FILE)
    Z = np.load(path)
    return {"x0": float(Z["x0"]), "y0": float(Z["y0"]), "step": float(Z["step"]), "P": Z["P"].astype(np.float32) / 255.0,
            "T": Z["T"].astype(np.float32) / 255.0, "W": Z["W"].astype(np.float32) / 255.0, "z0": float(Z["z0"]), "dz": float(Z["dz"]),
            "L": dict(zip(KEYS, Z["L"].tolist()))}


def _bilinear(A, fx, fy):
    ny, nx = A.shape; fx = np.clip(fx, 0, nx - 1); fy = np.clip(fy, 0, ny - 1)
    i0 = np.minimum(np.floor(fx).astype(int), nx - 2) if nx > 1 else np.zeros_like(fx, int)
    j0 = np.minimum(np.floor(fy).astype(int), ny - 2) if ny > 1 else np.zeros_like(fy, int)
    tx = fx - i0; ty = fy - j0; i1 = np.minimum(i0 + 1, nx - 1); j1 = np.minimum(j0 + 1, ny - 1)
    return (A[j0, i0] * (1 - tx) * (1 - ty) + A[j0, i1] * tx * (1 - ty) + A[j1, i0] * (1 - tx) * ty + A[j1, i1] * tx * ty)


def sample(F, X, deck=None, z=None):
    """(P, T, W) at points X (N, 2); the tier — the index deck or by height z (hive); outside the grid — the grid edge."""
    X = np.atleast_2d(np.asarray(X, float)); fx = (X[:, 0] - F["x0"]) / F["step"]; fy = (X[:, 1] - F["y0"]) / F["step"]
    nd = F["P"].shape[0]
    if deck is None and z is not None and nd > 1 and F["dz"] > 0:
        deck = np.clip(np.floor((np.asarray(z, float) - F["z0"]) / F["dz"] + 1e-6), 0, nd - 1).astype(int)
    if deck is None: deck = np.full(len(X), nd - 1 if nd > 1 else 0)          # without a tier — the top one (for a hive the plan field is the top)
    deck = np.broadcast_to(np.asarray(deck, int), (len(X),))
    P = np.zeros(len(X)); W = np.zeros(len(X))
    for k in np.unique(deck):
        m = deck == k; P[m] = _bilinear(F["P"][k], fx[m], fy[m]); W[m] = _bilinear(F["W"][k], fx[m], fy[m])
    return P, _bilinear(F["T"], fx, fy), W


def label(P, T, W, L):
    """Reading of the three numbers (scalars or arrays) → a district (a string or a list of strings)."""
    P, T, W = (np.asarray(v, float) for v in (P, T, W)); sc = P.ndim == 0
    P, T, W = np.atleast_1d(P), np.atleast_1d(T), np.atleast_1d(W)
    out = np.where(P >= L["power"], "power",
          np.where(T >= L["trade"], np.where(W >= L["trade_W"], "trade", "craft"),
          np.where(W >= L["nobility"], "nobility", np.where(W <= L["poor"], "poor", "middle"))))
    return str(out[0]) if sc else [str(v) for v in out]
