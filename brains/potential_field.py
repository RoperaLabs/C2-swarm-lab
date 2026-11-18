"""Artificial potential field: pulled to the goal, pushed away from what the ToF arrays see; a seeded random
nudge (sideways and vertical) when the distance to the goal stops shrinking, to leave local minima. With a detector it also keeps `sep` metres from the
drones it sees."""
import numpy as np

from c2lab.brain import Action, Velocity

from .common import climbing, clip, obstacles, pose, zone_dirs


class PotentialField:
    def __init__(self, speed=2.0, safe=2.5, k_rep=4.0, sep=3.0, stuck=30, decay=0.99, climb=1.0):
        self.speed, self.safe, self.k_rep, self.sep = speed, safe, k_rep, sep
        self.stuck, self.decay, self.climb = stuck, decay, climb

    def reset(self, spec):
        self.goal = np.array(spec.mission["goal_local"], float)
        self.dirs = zone_dirs(spec.sensors["tof"]) if "tof" in spec.sensors else None
        self.rng, self.nudge = np.random.default_rng(spec.seed), np.zeros(3)
        self.best, self.since = np.inf, 0

    def step(self, obs):
        ps = pose(obs)
        if ps is None:
            return Action()
        p, v, R = ps
        if climbing(p):
            return Action(Velocity(climbing(p)))
        to = self.goal - p
        cmd = to / max(np.linalg.norm(to), 1.0) * self.speed
        for o in obstacles(obs, self.dirs, R, self.safe):
            d = np.linalg.norm(o)
            cmd -= self.k_rep * (self.safe - d) / self.safe * o / max(d, 1e-3)
        det = obs.readings.get("detector")
        if det and len(det[-1].data):
            az, el, r = det[-1].data[:, 0], det[-1].data[:, 1], det[-1].data[:, 2]
            rel = (np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)] * r[:, None]) @ R.T
            for o, d in zip(rel, r):
                if d < self.sep:
                    cmd -= self.k_rep * (self.sep - d) / self.sep * o / max(d, 1e-3)
        dist = np.linalg.norm(to)
        if dist < self.best - 0.5:
            self.best, self.since = dist, 0
        self.since += 1
        if self.since > self.stuck:                      # no 0.5 m of progress for `stuck` steps: new nudge
            self.nudge = np.r_[self.rng.normal(0, 1, 2), self.rng.normal(0, self.climb)] * 2 * self.speed
            self.best, self.since = dist, 0
        self.nudge *= self.decay
        return Action(Velocity(tuple(float(x) for x in clip(cmd + self.nudge, self.speed))))
