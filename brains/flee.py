"""Evader: when drones of another team are in sight, run directly away from them (weighted by closeness) while
staying inside the operating area; otherwise hold position. speed = 0 gives a stationary target."""
import numpy as np

from .common import detections
from .potential_field import PotentialField


class Flee(PotentialField):
    def __init__(self, speed=3.0, horizon=10.0, **kw):
        super().__init__(speed=speed, **kw)
        self.horizon = horizon

    def target(self, p, R, obs):
        rel, dist, cls = detections(obs, R)
        foe = cls == 1
        if not foe.any():
            return p
        away = -(rel[foe] / np.maximum(dist[foe], 0.5)[:, None] ** 2).sum(0)
        away[2] *= 0.3                                                 # mostly horizontal escapes
        return p + away / max(np.linalg.norm(away), 1e-9) * self.horizon
