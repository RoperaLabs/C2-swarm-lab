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


def shortest_paths(world, starts, b, res, inflate=INFLATE):
    """Shortest collision-free paths from each start to b (world coordinates) for a drone of radius `inflate`: one
    Dijkstra from b over the 26-connected free voxels, then each path string-pulled with clearance checks.
    Returns a list of (waypoints (k, 3), length), (None, inf) where no path exists.
    ponytail: memory grows with the voxel count; coarsen `res` for the 200 m biomes if it bites."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    occ = voxelize(world, res, inflate)
    shape, lo = occ.shape, world.bounds[0]
    idx = np.full(shape, -1, np.int64)
    free = ~occ
    idx[free] = np.arange(free.sum())
    rows, cols, w = [], [], []
    for dx, dy, dz in [(x, y, z) for x in (0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) > (0, 0, 0)]:
        src = [slice(max(0, -d), s - max(0, d)) for d, s in zip((dx, dy, dz), shape)]
        dst = [slice(max(0, d), s - max(0, -d)) for d, s in zip((dx, dy, dz), shape)]
        u, v = idx[tuple(src)], idx[tuple(dst)]
        ok = (u >= 0) & (v >= 0)
        rows.append(u[ok])
        cols.append(v[ok])
        w.append(np.full(ok.sum(), res * math.sqrt(dx * dx + dy * dy + dz * dz)))
    n = int(free.sum())
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()

    def node(p):
        return idx[tuple(np.clip(np.floor((np.asarray(p) - lo) / res).astype(int), 0, np.array(shape) - 1))]

    t = node(b)
    if t < 0:
        return [(None, math.inf)] * len(starts)
    dist, pred = dijkstra(g, directed=False, indices=t, return_predecessors=True)
    centres = np.argwhere(free) * res + lo + res / 2

    def clear(p, q):
        m = max(int(np.linalg.norm(q - p) / (res / 4)), 2)
        return world.sdf(p + np.linspace(0, 1, m)[:, None] * (q - p), max_dist=inflate).min() >= inflate - 1e-9

    out = []
    for a in starts:
        k = node(a)
        if k < 0 or not np.isfinite(dist[k]):
            out.append((None, math.inf))
            continue
        chain = []
        while k >= 0:
            chain.append(centres[k])
            k = pred[k]
        pts = [np.asarray(a, float)] + chain[1:-1] + [np.asarray(b, float)]
        way, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not clear(pts[i], pts[j]):
                j -= 1
            way.append(pts[j])
            i = j
        way = np.array(way)
        out.append((way, float(np.linalg.norm(np.diff(way, axis=0), axis=1).sum())))
    return out
