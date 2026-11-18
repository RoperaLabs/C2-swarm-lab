"""Ground-truth checks used only for generation and metrics; never given to a brain.

voxelize: conservative occupancy (inflated by the drone radius). feasible: A and B joined through free space.
coverage: plan-view cover, the fraction of xy cells whose vertical segment touches a primitive.
"""
import math

import numpy as np
from scipy import ndimage

RES = {"mixto": 0.5, "urbano": 1.0, "bosque": 1.0, "interior": 0.25}
INFLATE = 0.6   # m: iris collision radius 0.384 + 0.2 margin
ALT = 2.0       # m above A and B


def _centres(world, res):
    lo, hi = world.bounds
    axes = [lo[a] + res * (np.arange(max(int((hi[a] - lo[a]) / res), 1)) + 0.5) for a in range(3)]
    return axes


def voxelize(world, res, inflate=INFLATE):
    """occ (nx, ny, nz) bool: True where a drone of radius `inflate` centred anywhere in the voxel could touch."""
    xs, ys, zs = _centres(world, res)
    thr = inflate + res * math.sqrt(3) / 2
    occ = np.empty((len(xs), len(ys), len(zs)), bool)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    yz = np.c_[gy.ravel(), gz.ravel()]
    for i, x in enumerate(xs):
        pts = np.c_[np.full(len(yz), x), yz]
        occ[i] = (world.sdf(pts, max_dist=thr) < thr).reshape(len(ys), len(zs))
    return occ


def feasible(world, res, inflate=INFLATE, alt=ALT):
    occ = voxelize(world, res, inflate)
    labels, _ = ndimage.label(~occ, structure=np.ones((3, 3, 3)))
    lo = world.bounds[0]
    ids = []
    for name in ("A", "B"):
        idx = np.floor((np.array(world.anchors[name]) + (0, 0, alt) - lo) / res).astype(int)
        idx = np.clip(idx, 0, np.array(occ.shape) - 1)
        ids.append(labels[tuple(idx)])
    return ids[0] != 0 and ids[0] == ids[1]


def coverage(world, res, z_top=None):
    xs, ys, _ = _centres(world, res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.c_[gx.ravel(), gy.ravel()]
    z0 = world.ground(xy) + 0.05
    top = world.bounds[1][2] - 0.05 if z_top is None else z_top
    return float((world.segment_crossings(np.c_[xy, z0], np.c_[xy, np.full(len(xy), top)]) > 0).mean())
