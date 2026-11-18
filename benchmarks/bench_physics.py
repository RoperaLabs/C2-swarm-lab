"""Ticks per second of 10 iris drones (model + autopilot) in open space. Target: >= 24 000 (20x real time at 1200 Hz)."""
import time

import numpy as np

from c2lab.brain import Velocity
from c2lab.physics import Drones


class OpenWorld:
    bounds = ((-1e4, -1e4, 0.0), (1e4, 1e4, 1e3))

    def sdf(self, p, terrain=True, max_dist=np.inf):
        return np.minimum(p[:, 2] if terrain else np.full(len(p), np.inf), max_dist)

    def ground(self, xy):
        return np.zeros(len(xy))


for mode in ("quad", "pointmass"):
    d = Drones("iris", 10, np.random.default_rng(0), OpenWorld(), mode=mode, wind=(3, 0, 0), gust_sigma=1.0)
    d.reset(np.c_[np.arange(10) * 5.0, np.zeros(10), np.full(10, 50.0)], np.zeros(10), landed=False)
    for i in range(10):
        d.set_setpoint(i, Velocity((4.0, 1.0, 0.0)))
    for k in range(1200):                       # warm-up: numba compiles on the first calls
        d.step(k)
    n = 24000
    t = time.perf_counter()
    for k in range(1200, 1200 + n):
        d.step(k)
    rate = n / (time.perf_counter() - t)
    print(f"{mode}: {rate:,.0f} ticks/s = {rate / 1200:.1f}x real time")
