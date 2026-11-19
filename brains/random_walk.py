"""Random waypoint inside the operating area (or 20 m away when none is given) every `period` seconds, at cruise
altitude; avoidance from PotentialField."""
import numpy as np

from .potential_field import PotentialField


class RandomWalk(PotentialField):
    def __init__(self, speed=1.5, period=8.0, alt=3.0, **kw):
        super().__init__(speed=speed, **kw)
        self.period, self.alt = period, alt

    def reset(self, spec):
        super().reset(spec)
        self.next, self.wp = 0.0, None

    def target(self, p, R, obs):
        t = obs.t
        if t >= self.next or self.wp is None:
            if self.area is not None:
                lo, hi = self.area
                xy = self.rng.uniform(lo[:2] + 2, hi[:2] - 2)
            else:
                a = self.rng.uniform(0, 2 * np.pi)
                xy = p[:2] + 20 * np.array([np.cos(a), np.sin(a)])
            self.wp, self.next = np.r_[xy, self.alt], t + self.period
        return self.wp
