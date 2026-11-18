"""Random heading every `period` seconds at cruise altitude; no sensing beyond pose."""
import numpy as np

from c2lab.brain import Action, Velocity

from .common import pose


class RandomWalk:
    def __init__(self, speed=1.5, period=3.0, alt=3.0):
        self.speed, self.period, self.alt = speed, period, alt

    def reset(self, spec):
        self.rng, self.next, self.v = np.random.default_rng(spec.seed), 0.0, np.zeros(3)

    def step(self, obs):
        if obs.t >= self.next:
            a = self.rng.uniform(0, 2 * np.pi)
            self.v, self.next = np.array([np.cos(a), np.sin(a), 0.0]) * self.speed, obs.t + self.period
        ps = pose(obs)
        vz = 0.0 if ps is None else float(np.clip(self.alt - ps[0][2], -1, 1))
        return Action(Velocity((float(self.v[0]), float(self.v[1]), vz)))
