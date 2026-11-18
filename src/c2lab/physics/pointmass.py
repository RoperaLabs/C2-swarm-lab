"""Point-mass model: same interface and same outer loop as `Quad`, with the dynamics replaced by bounded acceleration.

Acceleration is limited by the tilt and vertical bounds of the autopilot, velocity by the profile limits. The
attitude carries only the heading. ponytail: wind is ignored; add it as a relative-velocity drag if needed.
"""
import numpy as np

from .autopilot import limit_accel
from .profiles import G, RHO
from .quad import quat_yaw

_K_YAW = 4.0   # 1/s, heading P gain


class PointMass:
    def __init__(self, prof, n, dt):
        self.prof, self.n, self.dt = prof, n, dt
        self.p, self.v, self.w, self.f = (np.zeros((n, 3)) for _ in range(4))
        self.q = quat_yaw(np.zeros(n))
        self.yaw = np.zeros(n)
        self.omega = np.zeros((n, 4))
        self.energy = np.zeros(n)
        self.f_rest = np.zeros((n, 3))
        self.act = np.ones(n)
        self._k = prof.fm * prof.eta * np.sqrt(2 * RHO * np.pi * prof.prop_radius ** 2)

    def reset(self, p0, yaw0, omega0):
        self.p[:], self.v[:], self.w[:], self.f[:], self.f_rest[:] = p0, 0.0, 0.0, 0.0, 0.0
        self.yaw[:], self.q[:], self.omega[:], self.energy[:] = yaw0, quat_yaw(yaw0), omega0, 0.0

    def set_active(self, act):
        self.act[:] = act

    def cos_tilt(self, idx):
        return np.ones(len(idx))

    def freeze(self, idx, landed):
        self.v[idx] = self.w[idx] = self.omega[idx] = self.f[idx] = 0.0
        self.f_rest[idx] = (0.0, 0.0, G) if landed else 0.0

    def step(self, a_des, yaw_sp):
        pr, dt, act = self.prof, self.dt, self.act[:, None]
        a = limit_accel(a_des.copy(), pr)
        v0 = self.v.copy()
        self.v += a * (dt * act)
        h = np.hypot(self.v[:, 0], self.v[:, 1])
        self.v[:, :2] *= np.minimum(1.0, pr.vmax_xy / np.maximum(h, 1e-9))[:, None]
        np.clip(self.v[:, 2], -pr.vmax_down, pr.vmax_up, out=self.v[:, 2])
        self.p += self.v * dt
        a = (self.v - v0) / dt                                   # acceleration actually applied
        err = (yaw_sp - self.yaw + np.pi) % (2 * np.pi) - np.pi
        rate = np.clip(_K_YAW * err, -pr.yaw_rate_max, pr.yaw_rate_max) * self.act
        self.yaw += rate * dt
        self.q[:] = quat_yaw(self.yaw)
        self.w[:, 2] = rate
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        a[:, 2] += G
        self.f[:, 0], self.f[:, 1], self.f[:, 2] = c * a[:, 0] + s * a[:, 1], c * a[:, 1] - s * a[:, 0], a[:, 2]
        self.f *= act
        self.f += self.f_rest
        t = pr.mass * np.linalg.norm(a, axis=1) / 4 * self.act   # thrust per rotor
        self.omega[:] = np.sqrt(t / pr.kf)[:, None]
        self.energy += 4 * t ** 1.5 / self._k * dt
