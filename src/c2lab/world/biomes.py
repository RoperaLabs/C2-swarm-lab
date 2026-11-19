"""Procedural biomes. `density` is the target plan-view cover (oracle.coverage); objects are placed until it is
reached or nothing more fits. Every random draw comes from rng(seed, "world", attempt); an attempt the oracle
rejects (A and B not connected) is redrawn with the next attempt."""
import math
import time

import numpy as np

from c2lab.seeds import rng as stream

from . import oracle
from .world import World

CLEAR = {"mixto": 4.0, "urbano": 10.0, "bosque": 6.0, "interior": 3.0, "swarm_dil_v2": 10.0}   # m, free radius around A and B
MAX_ATTEMPTS = 20
_OVERLAP = 0.3   # a new object may cover at most this share of already-covered cells


def _quat(M):
    """Rotation matrix (columns = local axes in the world) -> (w, x, y, z)."""
    t = np.trace(M)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [s / 4, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(M)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + M[i, i] - M[j, j] - M[k, k])
        q = [0.0] * 4
        q[0], q[1 + i] = (M[k, j] - M[j, k]) / s, s / 4
        q[1 + j], q[1 + k] = (M[j, i] + M[i, j]) / s, (M[k, i] + M[i, k]) / s
    return [float(v) for v in q]


def yaw_q(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def ring_q(yaw):
    """Vertical ring: axis horizontal along `yaw`, kept arc centred on world +z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return _quat(np.array([[-s, 0, c], [c, 0, s], [0, 1, 0]], float))


class _Builder:
    def __init__(self, biome, rng, size, density, heights=None):
        self.rng, self.size, self.density = rng, [float(v) for v in size], density
        self.res = oracle.RES[biome]
        nx, ny = max(int(size[0] / self.res), 1), max(int(size[1] / self.res), 1)
        self.X, self.Y = np.meshgrid((np.arange(nx) + 0.5) * self.res, (np.arange(ny) + 0.5) * self.res, indexing="ij")
        self.cov = np.zeros((nx, ny), bool)
        self.H = np.zeros((2, 2)) if heights is None else heights
        self.prims = []
        clear, m = CLEAR[biome], CLEAR[biome] + 1.0
        self.anchors = {"A": (m, rng.uniform(m, size[1] - m)), "B": (size[0] - m, rng.uniform(m, size[1] - m))}
        self.clear = np.zeros_like(self.cov)
        for x, y in self.anchors.values():
            self.clear |= (self.X - x) ** 2 + (self.Y - y) ** 2 <= (clear + 0.5) ** 2

    def ground(self, x, y):
        nx, ny = self.H.shape
        fx = min(max(x / self.size[0] * (nx - 1), 0.0), nx - 1.0)
        fy = min(max(y / self.size[1] * (ny - 1), 0.0), ny - 1.0)
        i, j = min(int(fx), nx - 2), min(int(fy), ny - 2)
        u, v = fx - i, fy - j
        H = self.H
        return float((1 - u) * (1 - v) * H[i, j] + u * (1 - v) * H[i + 1, j] + (1 - u) * v * H[i, j + 1]
                     + u * v * H[i + 1, j + 1])

    def rect(self, cx, cy, hx, hy, yaw=0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.X - cx, self.Y - cy
        return (np.abs(c * dx + s * dy) <= hx) & (np.abs(-s * dx + c * dy) <= hy)

    def disk(self, cx, cy, r):
        return (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= r * r

    def fits(self, mask, overlap=_OVERLAP):
        if (mask & self.clear).any():
            return False
        n = mask.sum()
        return n == 0 or (mask & self.cov).sum() <= overlap * n

    def add(self, prims, mask=None):
        self.prims += prims
        if mask is not None:
            self.cov |= mask

    def full(self):
        return self.cov.mean() >= self.density

    def fill(self, propose, tries=4000):
        """Call propose() (returns (prims, mask) or None) until the cover target is met."""
        for _ in range(tries):
            if self.full():
                return
            got = propose()
            if got is not None and self.fits(got[1]):
                self.add(*got)

    def to_dict(self, biome, seed):
        anchors = {k: [x, y, self.ground(x, y)] for k, (x, y) in self.anchors.items()}
        return {"biome": biome, "seed": seed, "density": self.density, "bounds": [[0, 0, 0], self.size],
                "anchors": anchors, "terrain": {"heights": self.H.tolist()}, "primitives": self.prims}


def _box(tag, c, h, yaw=0.0):
    return {"type": "box", "tag": tag, "c": [float(v) for v in c], "h": [float(v) for v in h], "q": yaw_q(yaw)}


def _cyl(tag, x, y, z0, r, height):
    return {"type": "cylinder", "tag": tag, "c": [x, y, z0 + height / 2], "r": r, "hh": height / 2, "q": [1, 0, 0, 0]}


def _local(x, y, yaw, u, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * u - s * v, y + s * u + c * v


# --- mixto ------------------------------------------------------------------------------------------------------

def _mixto(b):
    r, (sx, sy, sz) = b.rng, b.size

    def propose():
        x, y, yaw = r.uniform(0, sx), r.uniform(0, sy), r.uniform(0, math.pi)
        kind = r.choice(["pillar", "column", "wall", "slab", "ring", "arch", "sphere"],
                        p=[0.25, 0.15, 0.15, 0.15, 0.12, 0.1, 0.08])
        if kind == "pillar":
            a, h = r.uniform(0.5, 1.25), min(r.uniform(8, 18), sz - 1)
            return [_box("pillar", (x, y, h / 2), (a, a, h / 2), yaw)], b.rect(x, y, a, a, yaw)
        if kind == "column":
            rad, h = r.uniform(0.5, 1.2), min(r.uniform(8, 18), sz - 1)
            return [_cyl("pillar", x, y, 0.0, rad, h)], b.disk(x, y, rad)
        if kind == "wall":
            L, t, H = r.uniform(8, 20), r.uniform(0.3, 0.5), min(r.uniform(6, 15), sz - 1)
            w, h = r.uniform(1.5, min(4, L - 2)), r.uniform(1.5, min(4, H - 1.5))
            u0, z0 = r.uniform(-L / 2 + 0.5, L / 2 - 0.5 - w), r.uniform(0.5, H - h - 0.5)
            parts = [(-L / 2, u0, 0, H), (u0 + w, L / 2, 0, H), (u0, u0 + w, 0, z0), (u0, u0 + w, z0 + h, H)]
            prims = []
            for a0, a1, b0, b1 in parts:
                if a1 - a0 > 1e-3 and b1 - b0 > 1e-3:
                    cx, cy = _local(x, y, yaw, (a0 + a1) / 2, 0.0)
                    prims.append(_box("wall", (cx, cy, (b0 + b1) / 2), ((a1 - a0) / 2, t / 2, (b1 - b0) / 2), yaw))
            return prims, b.rect(x, y, L / 2, t / 2, yaw)
        if kind == "slab":
            hx, hy, hz = r.uniform(2, 5), r.uniform(2, 5), r.uniform(0.15, 0.3)
            return [_box("slab", (x, y, r.uniform(4, min(15, sz - 1))), (hx, hy, hz), yaw)], b.rect(x, y, hx, hy, yaw)
        if kind == "ring":
            R, t = r.uniform(1.5, 4), r.uniform(0.2, 0.5)
            if sz - R - 1 < R + t + 0.5:
                return None
            z = r.uniform(max(3, R + t + 0.5), min(15, sz - R - 1))
            prim = {"type": "torus", "tag": "ring", "c": [x, y, z], "R": R, "r": t, "arc": math.pi, "q": ring_q(yaw)}
            return [prim], b.rect(x, y, t, R + t, yaw)
        if kind == "arch":
            R, t = r.uniform(5, min(15, sz - 2)), r.uniform(0.3, 0.8)
            prim = {"type": "torus", "tag": "arch", "c": [x, y, 0.0], "R": R, "r": t, "arc": math.pi / 2, "q": ring_q(yaw)}
            return [prim], b.rect(x, y, t, R + t, yaw)
        rad = r.uniform(0.8, 3)
        return [{"type": "sphere", "tag": "sphere", "c": [x, y, r.uniform(rad + 0.5, sz - rad - 1)], "r": rad}], \
            b.disk(x, y, rad)

    b.fill(propose)


# --- urbano -----------------------------------------------------------------------------------------------------

def _cuts(r, n, street=(10, 16), block=(30, 45)):
    """Alternating streets and blocks along one axis: list of (start, end) of the blocks."""
    out, pos = [], r.uniform(*street)
    while pos < n:
        end = min(pos + r.uniform(*block), n)
        if end - pos > 12:
            out.append((pos, end))
        pos = end + r.uniform(*street)
    return out


def _urbano(b):
    r, (sx, sy, sz) = b.rng, b.size
    xb, yb = _cuts(r, sx), _cuts(r, sy)
    blocks = [(x0, x1, y0, y1) for x0, x1 in xb for y0, y1 in yb]
    r.shuffle(blocks)
    parks = blocks[: max(1, int(0.15 * len(blocks)))]
    trees = []
    for x0, x1, y0, y1 in parks:
        for _ in range(int((x1 - x0) * (y1 - y0) / 60)):
            trees.append((r.uniform(x0 + 2, x1 - 2), r.uniform(y0 + 2, y1 - 2)))
    for x, y in trees:
        rad, ht = r.uniform(2, 4), r.uniform(3, 6)
        mask = b.disk(x, y, rad)
        if not b.full() and b.fits(mask, overlap=0.5):
            b.add([_cyl("trunk", x, y, 0.0, r.uniform(0.15, 0.3), ht),
                   {"type": "sphere", "tag": "crown", "c": [x, y, ht + rad * 0.8], "r": rad}], mask)
    tall, lots = [], blocks[len(parks):]
    for _ in range(3000):
        if b.full() or not lots:
            break
        x0, x1, y0, y1 = lots[int(r.integers(len(lots)))]
        if r.random() < 0.5:                      # a building over most of the block
            m = r.uniform(1, 2)
            hx, hy, cx, cy = (x1 - x0) / 2 - m, (y1 - y0) / 2 - m, (x0 + x1) / 2, (y0 + y1) / 2
        else:
            hx, hy = r.uniform(5, min(17.5, (x1 - x0) / 2)), r.uniform(5, min(17.5, (y1 - y0) / 2))
            cx, cy = r.uniform(x0 + hx, x1 - hx), r.uniform(y0 + hy, y1 - hy)
        mask = b.rect(cx, cy, hx, hy)
        if not b.fits(mask, overlap=0.0):
            continue
        h = float(np.clip(15 * math.exp(r.normal(0, 0.5)), 4, 0.9 * sz))
        prims = [_box("building", (cx, cy, h / 2), (hx, hy, h / 2))]
        if r.random() < 0.3:
            top = r.uniform(0.2, 0.4) * h
            prims.append(_box("building", (cx, cy, h + top / 2), (0.7 * hx, 0.7 * hy, top / 2)))
        b.add(prims, mask)
        tall.append((cx, cy, hx, hy, h))
    # street lights joined by cables, along the street below every other block row
    for k, (y0, _) in enumerate(yb):
        if k % 2 or y0 < 6:
            continue
        y, xs = y0 - 3.0, np.arange(r.uniform(3, 10), sx - 3, r.uniform(20, 30))
        h, prev = r.uniform(7, 10), None
        for x in xs:
            if b.clear[min(int(x / b.res), b.cov.shape[0] - 1), min(int(y / b.res), b.cov.shape[1] - 1)]:
                prev = None
                continue
            b.add([_cyl("pole", float(x), y, 0.0, r.uniform(0.12, 0.2), h)])
            if prev is not None:
                b.add([{"type": "capsule", "tag": "cable", "a": [prev, y, h - 1.5], "b": [float(x), y, h - 1.5],
                        "r": r.uniform(0.01, 0.03)}])
            prev = float(x)
    # footbridges between facing buildings across a street (along x)
    for i, (ax, ay, ahx, ahy, ah) in enumerate(tall):
        for bx, by, bhx, bhy, bh in tall[i + 1:]:
            gap0, gap1 = ax + ahx, bx - bhx
            y0, y1 = max(ay - ahy, by - bhy), min(ay + ahy, by + bhy)
            if 8 < gap1 - gap0 < 18 and y1 - y0 > 4 and min(ah, bh) > 9 and r.random() < 0.3:
                z, w = r.uniform(6, min(ah, bh) - 2), r.uniform(1, 2)
                mask = b.rect((gap0 + gap1) / 2, (y0 + y1) / 2, (gap1 - gap0) / 2, w)
                if not (mask & b.clear).any():
                    b.add([_box("bridge", ((gap0 + gap1) / 2, (y0 + y1) / 2, z), ((gap1 - gap0) / 2, w, 0.25))],
                          mask)


# --- bosque -----------------------------------------------------------------------------------------------------

def _terrain(r, size, cell):
    nx, ny = int(size[0] / cell) + 1, int(size[1] / cell) + 1
    X, Y = np.meshgrid(np.linspace(0, size[0], nx), np.linspace(0, size[1], ny), indexing="ij")
    H = np.zeros((nx, ny))
    for _ in range(int(r.integers(4, 8))):
        cx, cy, s = r.uniform(0, size[0]), r.uniform(0, size[1]), r.uniform(15, 50)
        H += r.uniform(-1, 1) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    H -= H.min()
    return np.round(H / max(H.max(), 1e-9) * r.uniform(2, 8), 3)


def _bosque(b):
    r, (sx, sy, sz) = b.rng, b.size
    clearings = [(r.uniform(0, sx), r.uniform(0, sy), r.uniform(8, 15)) for _ in range(int(r.integers(2, 5)))]
    trunks = np.empty((0, 2))

    def propose():
        nonlocal trunks
        x, y = r.uniform(0, sx), r.uniform(0, sy)
        if any((x - cx) ** 2 + (y - cy) ** 2 < rr * rr for cx, cy, rr in clearings):
            return None
        if len(trunks) and ((trunks - (x, y)) ** 2).sum(1).min() < 4.0:
            return None
        g = b.ground(x, y)
        if r.random() < 0.6:
            tr, h, cr = r.uniform(0.15, 0.4), r.uniform(4, 10), r.uniform(2, 5)
            prims = [_cyl("trunk", x, y, g - 0.5, tr, h + 0.5),
                     {"type": "sphere", "tag": "crown", "c": [x, y, g + h + 0.6 * cr], "r": cr}]
        else:
            tr, h, cr, z0 = r.uniform(0.15, 0.3), r.uniform(6, 15), r.uniform(1.5, 3.5), r.uniform(1, 3)
            prims = [_cyl("trunk", x, y, g - 0.5, tr, 0.7 * h + 0.5),
                     {"type": "cone", "tag": "crown", "c": [x, y, g + z0], "r": cr, "h": h - z0, "q": [1, 0, 0, 0]}]
        if prims[-1]["c"][2] + cr > sz - 0.5:
            return None
        trunks = np.vstack([trunks, (x, y)])
        return prims, b.disk(x, y, cr)

    b.fill(propose, tries=6000)
    for _ in range(int(sx * sy / 400)):   # undergrowth and fallen logs
        x, y = r.uniform(0, sx), r.uniform(0, sy)
        if b.clear[min(int(x / b.res), b.cov.shape[0] - 1), min(int(y / b.res), b.cov.shape[1] - 1)]:
            continue
        g = b.ground(x, y)
        if r.random() < 0.7:
            rad = r.uniform(0.3, 1.0)
            b.add([{"type": "sphere", "tag": "bush", "c": [x, y, g + 0.6 * rad], "r": rad}])
        else:
            L, rad, yaw = r.uniform(4, 10), r.uniform(0.2, 0.4), r.uniform(0, math.pi)
            x1, y1 = _local(x, y, yaw, L, 0.0)
            if 0 < x1 < sx and 0 < y1 < sy and not (b.rect(*_local(x, y, yaw, L / 2, 0.0), L / 2, rad, yaw)
                                                    & b.clear).any():
                b.add([{"type": "capsule", "tag": "log", "a": [x, y, g + rad], "b": [x1, y1, b.ground(x1, y1) + rad],
                        "r": rad}])


# --- interior ---------------------------------------------------------------------------------------------------

def _interior(b):
    r, (sx, sy, sz) = b.rng, b.size
    ceil, t = sz - 0.3, 0.15
    b.add([_box("ceiling", (sx / 2, sy / 2, sz - 0.15), (sx / 2, sy / 2, 0.15)),
           _box("wall", (sx / 2, t, ceil / 2), (sx / 2, t, ceil / 2)),
           _box("wall", (sx / 2, sy - t, ceil / 2), (sx / 2, t, ceil / 2)),
           _box("wall", (t, sy / 2, ceil / 2), (t, sy / 2, ceil / 2)),
           _box("wall", (sx - t, sy / 2, ceil / 2), (t, sy / 2, ceil / 2))],
          b.rect(sx / 2, t, sx / 2, t) | b.rect(sx / 2, sy - t, sx / 2, t) | b.rect(t, sy / 2, t, sy / 2)
          | b.rect(sx - t, sy / 2, t, sy / 2))
    step = r.uniform(8, 10)
    for x in np.arange(step, sx - 1, step):
        for y in np.arange(step, sy - 1, step):
            m = b.rect(x, y, 0.25, 0.25)
            if b.fits(m, overlap=1.0):
                b.add([_box("column", (x, y, ceil / 2), (0.25, 0.25, ceil / 2))], m)
    edge = CLEAR["interior"] + 4.5                       # beams and lamps stay out of the A/B columns
    for x in np.arange(edge + r.uniform(0, 3), sx - edge, r.uniform(4, 6)):        # beams and hanging lamps
        b.add([_box("beam", (x, sy / 2, ceil - 0.25), (0.15, sy / 2 - 2 * t, 0.25))])
        for y in np.arange(r.uniform(3, 6), sy - 2, r.uniform(6, 9)):
            z = r.uniform(ceil - 2.5, ceil - 1.5)
            b.add([{"type": "capsule", "tag": "cable", "a": [x, y, z], "b": [x, y, ceil - 0.5], "r": 0.01},
                   {"type": "sphere", "tag": "lamp", "c": [x, y, z - 0.2], "r": 0.25}])
    # partition walls across x with one or two doors
    lines = np.arange(step, sx - 1, step)                  # partitions stand on column lines (cross-aisles)
    for x in sorted(r.choice(lines, min(len(lines), int(r.integers(1, 3))), replace=False)):
        doors = sorted(r.uniform(2, sy - 4, 2))
        edges, y, prims = [], 2 * t, []
        for d0 in doors:
            if d0 > y:
                edges.append((y, d0))
            w, h = r.uniform(1.8, 2.5), r.uniform(2.1, 2.5)
            prims.append(_box("wall", (x, d0 + w / 2, (h + ceil) / 2), (0.1, w / 2, (ceil - h) / 2)))
            y = d0 + w
        edges.append((y, sy - 2 * t))
        for y0, y1 in edges:
            if y1 - y0 > 0.05:
                prims.append(_box("wall", (x, (y0 + y1) / 2, ceil / 2), (0.1, (y1 - y0) / 2, ceil / 2)))
        m = b.rect(x, sy / 2, 0.1, sy / 2)
        if not (m & b.clear).any():
            b.add(prims, m)

    # back-to-back rack rows (2 x ~1.1 m deep) between column lines, cross-aisles at the columns
    depth, aisle, h = r.uniform(2.0, 2.4), r.uniform(2.5, 3.5), min(r.uniform(4, 6), ceil - 1)
    edges = np.r_[2 * t, np.arange(step, sx - 1, step), sx - 2 * t]
    racks = []
    for y in np.arange(2 * t + aisle + depth / 2, sy - 2 * t - aisle - depth / 2, depth + aisle):
        for a0, a1 in zip(edges[:-1], edges[1:]):
            L = a1 - a0 - 1.2
            if L > 3:
                racks.append(((a0 + a1) / 2, float(y), L))
    r.shuffle(racks)
    for x, y, L in racks:
        if b.full():
            break
        mask = b.rect(x, y, L / 2, depth / 2)
        if not b.fits(mask, overlap=0.0):
            continue
        prims = []
        for u in np.linspace(-L / 2, L / 2, max(round(L / 2.5), 1) + 1):
            for v in (-depth / 2, depth / 2):
                prims.append(_box("rack", (x + u, y + v, h / 2), (0.04, 0.04, h / 2)))
        for z in np.arange(r.uniform(0.2, 0.5), h, r.uniform(1.5, 2.0)):
            prims.append(_box("rack", (x, y, z), (L / 2, depth / 2, 0.025)))
        b.add(prims, mask)


BUILD = {"mixto": _mixto, "urbano": _urbano, "bosque": _bosque, "interior": _interior}


def z_top(biome, size):
    """Upper end of the coverage segments: under the ceiling and beams indoors, the top of the arena elsewhere."""
    return size[2] - 0.3 - 0.55 if biome == "interior" else None


def merge_boxes(blocked):
    """Greedy cover of a boolean voxel grid by disjoint boxes (x runs, then rows in y, then layers in z): list of
    (lo, hi) voxel index ranges, hi exclusive."""
    left, out = blocked.copy(), []
    for x, y, z in np.argwhere(blocked):
        if not left[x, y, z]:
            continue
        x1 = x + 1
        while x1 < left.shape[0] and left[x1, y, z]:
            x1 += 1
        y1 = y + 1
        while y1 < left.shape[1] and left[x:x1, y1, z].all():
            y1 += 1
        z1 = z + 1
        while z1 < left.shape[2] and left[x:x1, y:y1, z1].all():
            z1 += 1
        left[x:x1, y:y1, z:z1] = False
        out.append(((x, y, z), (x1, y1, z1)))
    return out


def swarm_dil_v2(seed):
    """The GRACE-3D 2.0.0 terrain of swarm-dil (sim3d.world.Mission defaults: 40 x 40 x 10 voxels of 5 m) for the same
    seed, as merged boxes on flat ground; A and B sit under the centres of its start and goal voxels."""
    from sim3d.world import Mission
    from swarm_dil.nav3.world3 import make_world
    m = Mission()
    blocked, start, goal, _ = make_world(m, np.random.default_rng(seed))
    c = m.cell_m
    prims = [_box("block", (np.add(lo, hi) / 2 * c).tolist(), (np.subtract(hi, lo) / 2 * c).tolist())
             for lo, hi in merge_boxes(blocked)]
    size = [float(n * c) for n in m.shape]
    return World.from_dict({"biome": "swarm_dil_v2", "seed": seed, "density": float(blocked[:, :, 0].mean()),
                            "bounds": [[0, 0, 0], size], "terrain": {"heights": np.zeros((2, 2)).tolist()},
                            "anchors": {k: [(v[0] + 0.5) * c, (v[1] + 0.5) * c, 0.0]
                                        for k, v in (("A", start), ("B", goal))},
                            "primitives": prims})


def generate(biome, size_m, density, seed):
    t0 = time.perf_counter()
    if biome == "swarm_dil_v2":                     # fixed size and terrain: size_m and density do not apply
        w = swarm_dil_v2(seed)
        w.attempts, w.seconds = 1, time.perf_counter() - t0
        return w
    for attempt in range(MAX_ATTEMPTS):
        r = stream(seed, "world", attempt)
        heights = _terrain(r, size_m, r.uniform(3, 5)) if biome == "bosque" else None
        b = _Builder(biome, r, size_m, density, heights)
        BUILD[biome](b)
        w = World.from_dict(b.to_dict(biome, seed))
        if oracle.feasible(w, oracle.RES[biome]):
            w.attempts, w.seconds = attempt + 1, time.perf_counter() - t0
            return w
    raise RuntimeError(f"{biome} seed {seed}: no feasible world in {MAX_ATTEMPTS} attempts")
