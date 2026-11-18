import math

import numpy as np
import pytest

from c2lab.world import World
from c2lab.world.world import quat_matrix

RNG = np.random.default_rng(0)
QT = [0.9, 0.2, -0.3, 0.25]                       # a generic tilt, normalised by the World
ARCH_Q = [0.5, 0.5, 0.5, 0.5]                     # local x -> y, y -> z, z -> x (convention example)

PRIMS = {
    "box": {"type": "box", "c": [0, 0, 0], "h": [1.0, 0.6, 0.4], "q": QT},
    "cylinder": {"type": "cylinder", "c": [0, 0, 0], "r": 0.7, "hh": 1.1, "q": QT},
    "capsule": {"type": "capsule", "a": [-1, -0.5, 0.2], "b": [1, 0.6, -0.3], "r": 0.3},
    "sphere": {"type": "sphere", "c": [0, 0, 0], "r": 0.9},
    "cone": {"type": "cone", "c": [0, 0, -0.8], "r": 0.8, "h": 1.6, "q": QT},
    "torus": {"type": "torus", "c": [0, 0, 0], "R": 1.0, "r": 0.25, "arc": 2.0, "q": QT},
}


def world(prims, heights=((-50.0, -50.0), (-50.0, -50.0)), bounds=((-5, -5, -5), (5, 5, 5))):
    return World.from_dict({"biome": "t", "seed": 0, "density": 0.0, "bounds": bounds,
                            "anchors": {"A": [0, 0, 0], "B": [1, 1, 0]},
                            "terrain": {"heights": [list(r) for r in heights]},
                            "primitives": [dict(p, tag="t") for p in prims]})


def local(p, pts):
    if p["type"] == "capsule":
        return pts
    R = quat_matrix(p.get("q", (1, 0, 0, 0)))
    return (pts - p["c"]) @ R


def inside(p, pts):
    """Membership written independently of the SDFs."""
    x, y, z = local(p, pts).T
    t = p["type"]
    if t == "box":
        return (np.abs(np.c_[x, y, z]) <= p["h"]).all(1)
    if t == "sphere":
        return np.sqrt(x * x + y * y + z * z) <= p["r"]
    if t == "cylinder":
        return (np.hypot(x, y) <= p["r"]) & (np.abs(z) <= p["hh"])
    if t == "cone":
        return (z >= 0) & (z <= p["h"]) & (np.hypot(x, y) <= p["r"] * (1 - z / p["h"]))
    if t == "capsule":
        a, b = np.array(p["a"]), np.array(p["b"])
        h = np.clip((pts - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        return np.linalg.norm(pts - a - h[:, None] * (b - a), axis=1) <= p["r"]
    ang = np.arctan2(x, y)                           # angle from +y
    tube = (np.hypot(x, y) - p["R"]) ** 2 + z * z <= p["r"] ** 2
    ends = [np.linalg.norm(np.c_[x, y, z] - p["R"] * np.array([s * math.sin(p["arc"]), math.cos(p["arc"]), 0]), axis=1)
            <= p["r"] for s in (-1, 1)]
    return (tube & (np.abs(ang) <= p["arc"])) | ends[0] | ends[1]


def surface(p, n=400_000):
    """Dense points on the surface: rejection on a thin shell found by bisection between in/out samples."""
    pts = RNG.uniform(-2.2, 2.2, (n, 3))
    ins = inside(p, pts)
    a, b = pts[ins][: n // 8], pts[~ins][: n // 8]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    for _ in range(40):
        mid = (a + b) / 2
        im = inside(p, mid)
        a = np.where(im[:, None], mid, a)
        b = np.where(im[:, None], b, mid)
    return (a + b) / 2


@pytest.mark.parametrize("name", PRIMS)
def test_sign_matches_membership(name):
    p, pts = PRIMS[name], RNG.uniform(-2.2, 2.2, (10_000, 3))
    s = world([p]).sdf(pts, terrain=False)
    assert np.array_equal(s <= 0, inside(p, pts))


@pytest.mark.parametrize("name", PRIMS)
def test_sdf_is_exact_distance(name):
    p, w, h = PRIMS[name], world([PRIMS[name]]), 1e-6
    surf = surface(p)
    pts = RNG.uniform(-2.2, 2.2, (3000, 3))
    pts = pts[~inside(p, pts)][:500]
    s = w.sdf(pts, terrain=False)
    ref = np.array([np.linalg.norm(surf - q, axis=1).min() for q in pts])
    assert np.all(s <= ref + 1e-6)                  # never more than the distance to a true surface point
    g = np.stack([(w.sdf(pts + e, terrain=False) - w.sdf(pts - e, terrain=False)) / (2 * h) for e in np.eye(3) * h], 1)
    g /= np.linalg.norm(g, axis=1)[:, None]
    # ... and the point at that distance along the gradient is on the surface (sign test: sdf = 0 is the boundary)
    assert np.abs(w.sdf(pts - s[:, None] * g, terrain=False)).max() <= 1e-5


@pytest.mark.parametrize("name", PRIMS)
def test_gradient_norm(name):
    w, h = world([PRIMS[name]]), 1e-5
    pts = RNG.uniform(-2.2, 2.2, (2000, 3))
    g = np.stack([(w.sdf(pts + e, terrain=False) - w.sdf(pts - e, terrain=False)) / (2 * h)
                  for e in np.eye(3) * h], 1)
    assert np.linalg.norm(g, axis=1).max() <= 1 + 1e-3


def test_terrain_bound_and_ground():
    H = [[0, 1, 0], [2, 0, 3], [0, 1, 0]]
    w = world([], heights=H, bounds=((0, 0, 0), (10, 10, 10)))
    assert np.allclose(w.ground([[0, 0], [5, 5], [10, 5], [5, 10], [2.5, 0]]), [0, 0, 1, 3, 1])
    pts = RNG.uniform(0, 10, (2000, 3))
    s, gz = w.sdf(pts), w.ground(pts[:, :2])
    assert np.array_equal(s < 0, pts[:, 2] < gz)
    assert np.all(np.abs(s) <= np.abs(pts[:, 2] - gz) + 1e-12)
    assert np.allclose(world([], heights=[[0, 0], [0, 0]]).ground([[1, 2]]), 0)


def scene():
    prims = [dict(p, c=[x, y, z]) if "c" in p else p for p, (x, y, z) in
             zip(PRIMS.values(), [(-3, -3, 1), (0, -3, 1.5), (0, 0, 0), (3, -3, 1), (-3, 3, 1), (3, 3, 1.5)])]
    prims[2] = {"type": "capsule", "a": [-1, -0.5, 1.2], "b": [1, 0.6, 0.7], "r": 0.3}
    prims.append({"type": "capsule", "a": [-4, 0, 3], "b": [4, 0.5, 3.2], "r": 0.02})   # a cable
    return world(prims, heights=[[0, 0.3, 0], [0.4, 0, 0.2], [0, 0.5, 0]])


def test_raycast_matches_dense_reference():
    w = scene()
    n, rng = 2000, 8.0
    o = RNG.uniform(-4.5, 4.5, (n, 3))
    o[:, 2] = RNG.uniform(1.5, 4.5, n)
    o = o[w.sdf(o) > 0.05]
    d = RNG.normal(size=(len(o), 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    dist, prim, normal = w.raycast(o, d, rng)
    ts = np.arange(0, rng, 0.002)
    for k in range(len(o)):
        s = w.sdf(o[k] + ts[:, None] * d[k])
        neg = np.flatnonzero(s <= 0)
        if neg.size == 0:                       # reference misses: a hit may only be a near-touch
            assert np.isinf(dist[k]) or s.min() <= 0.002
            continue
        if np.isinf(dist[k]):                   # kernel misses: the reference may only graze
            assert s.min() >= -0.002
            continue
        lo, hi = ts[neg[0]] - 0.002, ts[neg[0]]
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if w.sdf(o[k] + mid * d[k])[0] <= 0 else (mid, hi)
        assert abs(dist[k] - hi) <= 1e-3, (k, dist[k], hi)


def test_raycast_normals_and_fixed_cases():
    w = scene()
    o = np.array([[0.0, -3.0, 6.0], [3.0, 3.0, 1.5], [-3.0, -3.0, 1.0]])
    d = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    dist, prim, normal = w.raycast(o, d, 20.0)
    assert dist[2] == 0.0 and np.all(normal[2] == 0)            # origin inside the box
    hit = np.isfinite(dist) & (dist > 0)
    assert np.all(np.einsum("ij,ij->i", normal[hit], d[hit]) < 0)
    # ring hole: through the centre of a full vertical torus, along its axis
    ring = world([{"type": "torus", "c": [0, 0, 2], "R": 1.0, "r": 0.2, "arc": math.pi, "q": ARCH_Q}],
                 heights=[[0, 0], [0, 0]])
    assert np.isinf(ring.raycast([[-4, 0, 2]], [[1, 0, 0]], 8)[0][0])
    assert ring.raycast([[-4, 1, 2]], [[1, 0, 0]], 8)[1][0] == 0
    # grazing a box face
    b = world([{"type": "box", "c": [0, 0, 0], "h": [1, 1, 1], "q": [1, 0, 0, 0]}])
    assert np.isclose(b.raycast([[-3, 0, 1 - 1e-4]], [[1, 0, 0]], 8)[0][0], 2.0)
    assert np.isinf(b.raycast([[-3, 0, 1 + 1e-4]], [[1, 0, 0]], 8)[0][0])


def test_arch_convention():
    w = world([{"type": "torus", "c": [0, 0, 0], "R": 5, "r": 0.5, "arc": math.pi / 2, "q": ARCH_Q}],
              bounds=((-10, -10, -10), (10, 10, 10)))
    s = w.sdf([[0, 5, 0], [0, -5, 0], [0, 0, 5], [0, 0, -5]], terrain=False)
    assert (s[:3] < 0).all() and s[3] > 0


def test_segment_crossings_bruteforce():
    w = scene()
    a = RNG.uniform(-4.5, 4.5, (2000, 3))
    b = RNG.uniform(-4.5, 4.5, (2000, 3))
    a[:, 2], b[:, 2] = RNG.uniform(0.6, 4.5, 2000), RNG.uniform(0.6, 4.5, 2000)
    got = w.segment_crossings(a, b)
    singles = [world([p], heights=[[-50, -50], [-50, -50]]) for p in w.to_dict()["primitives"]]
    terr = world([], heights=w.to_dict()["terrain"]["heights"])
    ts = np.linspace(0, 1, 4001)
    for k in range(0, 2000, 10):
        pts = a[k] + ts[:, None] * (b[k] - a[k])
        mins = [s.sdf(pts, terrain=False).min() for s in singles] + [terr.sdf(pts).min()]
        ref, graze = sum(m <= 0 for m in mins), sum(abs(m) <= 1e-3 for m in mins)
        assert abs(got[k] - ref) <= graze, k


def test_max_dist_clamp():
    w = scene()
    pts = RNG.uniform(-5, 5, (3000, 3))
    for m in (0.3, 1.0, 4.0):
        assert np.allclose(w.sdf(pts, max_dist=m), np.minimum(w.sdf(pts), m))
        assert np.allclose(w.sdf(pts, terrain=False, max_dist=m), np.minimum(w.sdf(pts, terrain=False), m))


def test_roundtrip_and_pinned_hash():
    w = scene()
    assert World.from_dict(w.to_dict()).hash == w.hash
    small = world([{"type": "sphere", "c": [1.0000001, 2, 3], "r": 0.5}])
    assert small.to_dict()["primitives"][0]["c"] == [1.0, 2, 3]
    assert small.hash == PINNED


PINNED = "f3b8e6a8b91ab13ad1143112df2e45f92f6d0b1360a967445ae40ad394c5be75"
