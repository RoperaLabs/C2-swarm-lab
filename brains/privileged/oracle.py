"""Follows the oracle's shortest path (local frame, spec.mission["oracle_path"]) at a fixed speed. Needs an
accurate pose. Privileged: exists only to validate the metrics."""
import numpy as np

from c2lab.brain import Action, Velocity

from ..common import climbing, clip, pose


class Oracle:
    privileged = True

    def __init__(self, speed=2.5, reach=0.3):
        self.speed, self.reach = speed, reach

    def reset(self, spec):
        self.path, self.k = np.array(spec.mission["oracle_path"], float), 0

    def step(self, obs):
        ps = pose(obs)
        if ps is None:
            return Action()
        p = ps[0]
        if climbing(p):
            return Action(Velocity(climbing(p)))
        while self.k < len(self.path) - 1 and np.linalg.norm(self.path[self.k] - p) < self.reach:
            self.k += 1
        return Action(Velocity(tuple(float(x) for x in clip((self.path[self.k] - p) * 2.0, self.speed))))
