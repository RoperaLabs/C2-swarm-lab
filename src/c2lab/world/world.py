"""`World`: a c2lab-world/1 scene (primitives + terrain + geofence) with batched geometric queries."""
import hashlib
import json
import math

import numpy as np

from . import primitives as K

FORMAT = "c2lab-world/1"
_GRID = 4.0  # m, broad-phase cell for bounded SDF queries


def _round(x):
    if isinstance(x, float):
        return round(x, 6) + 0.0
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def quat_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pack(p):
    """One primitive dict -> (type id, P row, AABB)."""
    t, row = K.TYPES[p["type"]], np.zeros(17)
    if t == K.CAPSULE:
        a, b, r = np.array(p["a"], float), np.array(p["b"], float), p["r"]
        row[1:4], row[13:16], row[16] = a, b, r
        return t, row, np.r_[np.minimum(a, b) - r, np.maximum(a, b) + r]
    c, R = np.array(p["c"], float), quat_matrix(p.get("q", (1, 0, 0, 0)))
    row[1:4], row[4:13] = c, R.T.ravel()
    if t == K.BOX:
        row[13:16] = p["h"]
        lo, hi = -np.array(p["h"], float), np.array(p["h"], float)
    elif t == K.CYLINDER:
        row[13:15] = p["r"], p["hh"]
        lo, hi = np.array([-p["r"], -p["r"], -p["hh"]]), np.array([p["r"], p["r"], p["hh"]])
    elif t == K.SPHERE:
        row[13] = p["r"]
        lo, hi = -np.full(3, p["r"]), np.full(3, p["r"])
    elif t == K.CONE:
        row[13:15] = p["r"], p["h"]
        lo, hi = np.array([-p["r"], -p["r"], 0.0]), np.array([p["r"], p["r"], p["h"]])
    else:
        row[13:17] = p["R"], p["r"], math.sin(p["arc"]), math.cos(p["arc"])
        e = p["R"] + p["r"]
        lo, hi = np.array([-e, -e, -p["r"]]), np.array([e, e, p["r"]])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = corners @ R.T + c
    return t, row, np.r_[w.min(0), w.max(0)]


class World:
    TERRAIN = K.TERRAIN

    def __init__(self, d):
        self._d = d
        prims = d["primitives"]
        packed = [_pack(p) for p in prims]
        self._T = np.array([x[0] for x in packed], np.int64)
        self._P = np.array([x[1] for x in packed]).reshape(-1, 17)
        self._AB = np.array([x[2] for x in packed]).reshape(-1, 6)
        lo, hi = (np.array(b, float) for b in d["bounds"])
        self.bounds = (lo, hi)
        self.anchors = {k: tuple(v) for k, v in d["anchors"].items()}
        self.tags = [p.get("tag", "") for p in prims]
        H = np.array(d["terrain"]["heights"], float)
        cx, cy = (hi[0] - lo[0]) / (H.shape[0] - 1), (hi[1] - lo[1]) / (H.shape[1] - 1)
        lx = np.abs(np.diff(H, axis=0)).max(initial=0.0) / cx
        ly = np.abs(np.diff(H, axis=1)).max(initial=0.0) / cy
        self._H, self._g = H, np.array([lo[0], lo[1], cx, cy, 1 / math.sqrt(1 + lx * lx + ly * ly)])
        self._build_grid()

    def _build_grid(self):
        lo = np.minimum(self.bounds[0], self._AB[:, :3].min(0, initial=np.inf)) - _GRID
        hi = np.maximum(self.bounds[1], self._AB[:, 3:].max(0, initial=-np.inf)) + _GRID
        dim = np.maximum(np.ceil((hi - lo) / _GRID).astype(np.int64), 1)
        cells = [[] for _ in range(int(dim.prod()))]
        for i, ab in enumerate(self._AB):
            a = np.floor((ab[:3] - lo) / _GRID).astype(int)
            b = np.floor((ab[3:] - lo) / _GRID).astype(int)
            for x in range(a[0], b[0] + 1):
                for y in range(a[1], b[1] + 1):
                    for z in range(a[2], b[2] + 1):
                        cells[(x * dim[1] + y) * dim[2] + z].append(i)
        self._grid0, self._gdim = lo, dim
        self._cstart = np.r_[0, np.cumsum([len(c) for c in cells])].astype(np.int64)
        self._items = np.array([i for c in cells for i in c], np.int64)

    @classmethod
    def from_dict(cls, d):
        d = _round(dict(d))
        d["format"] = FORMAT
        return cls(d)

    def to_dict(self):
        return json.loads(json.dumps(self._d))

    @property
    def hash(self):
        return hashlib.sha256(json.dumps(self._d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def sdf(self, p, terrain=True, max_dist=np.inf):
        p = np.ascontiguousarray(p, float).reshape(-1, 3)
        out = np.empty(len(p))
        K.sdf_points(self._T, self._P, self._AB, self._H, self._g, terrain, self._grid0, _GRID, self._gdim,
                     self._cstart, self._items, p, float(max_dist), out)
        return out

    def ground(self, xy):
        xy = np.ascontiguousarray(xy, float).reshape(-1, 2)
        out = np.empty(len(xy))
        K.ground_points(self._H, self._g, xy, out)
        return out

    def raycast(self, o, d, max_range):
        o = np.ascontiguousarray(o, float).reshape(-1, 3)
        d = np.ascontiguousarray(d, float).reshape(-1, 3)
        r = np.broadcast_to(np.asarray(max_range, float), len(o)).copy()
        dist, prim, normal = np.empty(len(o)), np.empty(len(o), np.int64), np.empty((len(o), 3))
        K.raycast(self._T, self._P, self._AB, self._H, self._g, o, d, r, dist, prim, normal)
        return dist, prim, normal

    def segment_crossings(self, a, b):
        a = np.ascontiguousarray(a, float).reshape(-1, 3)
        b = np.ascontiguousarray(b, float).reshape(-1, 3)
        out = np.empty(len(a), np.int64)
        K.crossings(self._T, self._P, self._AB, self._H, self._g, a, b, out)
        return out
