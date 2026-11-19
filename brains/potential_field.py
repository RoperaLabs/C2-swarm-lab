"""Artificial potential field: pulled to a goal, pushed away from what the ToF arrays see, from teammates the detector
sees (closer than `sep`) and from the edges of the operating area when the mission gives one; a seeded random nudge
(sideways and vertical) when the distance to the goal stops shrinking, to leave local minima.

Subclasses move the goal by overriding `target(p, R, obs)`: pursue, flee, random walk and sector sweep reuse the
same avoidance."""
import numpy as np

from c2lab.brain import Action, Velocity

from .common import climbing, clip, detections, obstacles, pose, zone_dirs

EDGE = 4.0   # m from the operating-area edges where their push starts


class PotentialField:
    def __init__(self, speed=2.0, safe=2.5, k_rep=4.0, sep=3.0, k_sep=8.0, stuck=30, decay=0.97, climb=0.5):
        self.speed, self.safe, self.k_rep, self.sep, self.k_sep = speed, safe, k_rep, sep, k_sep
        self.stuck, self.decay, self.climb = stuck, decay, climb

    def reset(self, spec):
        m = spec.mission
        self.goal = np.array(m["goal_local"], float) if "goal_local" in m else None
        self.area = (np.array(m["area_local"][0]), np.array(m["area_local"][1])) if "area_local" in m else None
        self.dirs = zone_dirs(spec.sensors["tof"]) if "tof" in spec.sensors else None
        self.rng, self.nudge = np.random.default_rng(spec.seed), np.zeros(3)
        self.best, self.since = np.inf, 0

    def target(self, p, R, obs):
        return self.goal

    def step(self, obs):
        ps = pose(obs)
        if ps is None:
            return Action()
        p, v, R = ps
        if climbing(p):
            return Action(Velocity(climbing(p)))
        goal = self.target(p, R, obs)
        if goal is None:
            return Action()
        if self.area is not None:                       # never aim outside the operating area
            lo, hi = self.area
            goal = np.clip(goal, lo + EDGE, hi - EDGE)
        to = goal - p
        cmd = to / max(np.linalg.norm(to), 1.0) * self.speed
        for o in obstacles(obs, self.dirs, R, self.safe):
            d = np.linalg.norm(o)
            cmd -= self.k_rep * (self.safe - d) / self.safe * o / max(d, 1e-3)
        rel, dist, cls = detections(obs, R)
        for o, d in zip(rel[cls == 0], dist[cls == 0]):
            if d < self.sep:
                cmd -= self.k_sep * (self.sep - d) / self.sep * o / max(d, 1e-3)
        if self.area is not None:
            lo, hi = self.area
            for a in range(2):                          # walls of the area (the ground is the ToF's job)
                cmd[a] += self.k_rep * (max(0.0, EDGE - (p[a] - lo[a])) - max(0.0, EDGE - (hi[a] - p[a]))) / EDGE
            cmd[2] -= self.k_rep * max(0.0, EDGE - (hi[2] - p[2])) / EDGE
        dist_goal = np.linalg.norm(to)
        if dist_goal < self.best - 0.5:
            self.best, self.since = dist_goal, 0
        self.since += 1
        if self.since > self.stuck:                      # no 0.5 m of progress for `stuck` steps: new nudge
            self.nudge = np.r_[self.rng.normal(0, 1, 2), self.rng.normal(0, self.climb)] * self.speed
            self.best, self.since = dist_goal, 0
        self.nudge *= self.decay
        return Action(Velocity(tuple(float(x) for x in clip(cmd + self.nudge, self.speed))))
