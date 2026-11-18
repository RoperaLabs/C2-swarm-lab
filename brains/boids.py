"""Reynolds boids with a goal: separation and cohesion from what the detector sees, alignment from the velocities
the neighbours broadcast (6 bytes each), obstacle avoidance from the ToF arrays."""
import struct

import numpy as np

from c2lab.brain import Action, Velocity

from .common import climbing, clip, obstacles, pose, zone_dirs


class Boids:
    def __init__(self, speed=2.0, sep=2.5, w_sep=1.5, w_coh=0.3, w_ali=0.5, w_goal=1.0, safe=2.5, k_rep=4.0):
        self.speed, self.sep, self.safe, self.k_rep = speed, sep, safe, k_rep
        self.w = (w_sep, w_coh, w_ali, w_goal)

    def reset(self, spec):
        self.goal = np.array(spec.mission["goal_local"], float)
        self.dirs = zone_dirs(spec.sensors["tof"]) if "tof" in spec.sensors else None

    def step(self, obs):
        ps = pose(obs)
        if ps is None:
            return Action()
        p, v, R = ps
        if climbing(p):
            return Action(Velocity(climbing(p)))
        w_sep, w_coh, w_ali, w_goal = self.w
        to = self.goal - p
        cmd = w_goal * to / max(np.linalg.norm(to), 1.0) * self.speed
        det = obs.readings.get("detector")
        if det and len(det[-1].data):
            az, el, r = det[-1].data[:, 0], det[-1].data[:, 1], det[-1].data[:, 2]
            rel = (np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)] * r[:, None]) @ R.T
            cmd += w_coh * 0.2 * rel.mean(0)
            near = rel[r < self.sep]
            if len(near):
                cmd -= w_sep * (near / np.maximum(np.linalg.norm(near, axis=1), 0.1)[:, None] ** 2).sum(0)
        vel = [struct.unpack("<3e", pk.data) for pk in obs.inbox if len(pk.data) == 6]
        if vel:
            cmd += w_ali * (np.mean(vel, 0) - v)
        for o in obstacles(obs, self.dirs, R, self.safe):
            d = np.linalg.norm(o)
            cmd -= self.k_rep * (self.safe - d) / self.safe * o / max(d, 1e-3)
        return Action(Velocity(tuple(float(x) for x in clip(cmd, self.speed))), (struct.pack("<3e", *v),))
