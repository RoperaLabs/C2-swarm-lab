"""Ray and SDF throughput on a seeded synthetic 200 x 200 x 60 m scene of 500 primitives. Target: >= 300 000 rays/s."""
import math
import time

import numpy as np

from c2lab.world import World

rng = np.random.default_rng(0)


def yaw_q():
    a = rng.uniform(0, 2 * math.pi)
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def c(z=(1, 40)):
    return [float(rng.uniform(5, 195)), float(rng.uniform(5, 195)), float(rng.uniform(*z))]


prims = ([{"type": "box", "c": c((2, 20)), "h": [float(x) for x in rng.uniform(1, 8, 3)], "q": yaw_q()} for _ in range(400)]
         + [{"type": "cylinder", "c": c(), "r": 0.5, "hh": 4.0, "q": yaw_q()} for _ in range(40)]
         + [{"type": "sphere", "c": c(), "r": 2.0} for _ in range(20)]
         + [{"type": "capsule", "a": (a := c()), "b": [a[0] + 15, a[1], a[2]], "r": 0.03} for _ in range(20)]
         + [{"type": "cone", "c": c(), "r": 2.0, "h": 6.0, "q": yaw_q()} for _ in range(10)]
         + [{"type": "torus", "c": c(), "R": 3.0, "r": 0.4, "arc": math.pi, "q": yaw_q()} for _ in range(10)])
w = World.from_dict({"biome": "bench", "seed": 0, "density": 0.0, "bounds": [[0, 0, 0], [200, 200, 60]],
                     "anchors": {"A": [1, 1, 0], "B": [199, 199, 0]}, "terrain": {"heights": [[0, 0], [0, 0]]},
                     "primitives": [dict(p, tag="x") for p in prims]})
orig = []
while len(orig) < 10:
    p = np.array([c((2, 20))])
    if w.sdf(p)[0] > 1.0:
        orig.append(p[0])
d = rng.normal(size=(512, 3))
d /= np.linalg.norm(d, axis=1)[:, None]
o, d = np.repeat(orig, 512, 0), np.tile(d, (10, 1))
w.raycast(o, d, 10.0)
t = time.perf_counter()
for _ in range(20):
    dist, _, _ = w.raycast(o, d, 10.0)
rate = 20 * len(o) / (time.perf_counter() - t)
print(f"raycast: {rate:,.0f} rays/s (hits {np.isfinite(dist).mean():.0%})")
p = np.array(orig)
w.sdf(p, max_dist=2.0)
t = time.perf_counter()
for _ in range(2000):
    w.sdf(p, max_dist=2.0)
print(f"sdf(10 points, max_dist=2): {(time.perf_counter() - t) / 2000 * 1e6:.1f} us/call")
