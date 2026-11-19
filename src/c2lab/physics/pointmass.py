"""Point-mass model: same interface and same outer loop as `Quad`, with the dynamics replaced by bounded acceleration.

Acceleration is limited by the tilt and vertical bounds of the autopilot, velocity by the profile limits. The
attitude carries only the heading. ponytail: wind is ignored; add it as a relative-velocity drag if needed.
"""
import math

import numpy as np
from numba import njit

from .autopilot import accel_limits
from .profiles import G, RHO
from .quad import quat_yaw

_K_YAW = 4.0   # 1/s, heading P gain


@njit(cache=True)
def _step(p, v, yaw, q, w, f, f_rest, omega, energy, act, a_des, yaw_sp, prm):
    """One tick for every drone. prm: dt, vmax_xy, vmax_down, vmax_up, az_max, tan_tilt, yaw_rate_max, mass, kf, k."""
    dt, vxy, vdown, vup, az_max, tan_tilt, yaw_max, mass, kf, k = prm
    for i in range(p.shape[0]):
        ax, ay, az = a_des[i, 0], a_des[i, 1], min(max(a_des[i, 2], -az_max), az_max)
        lim, h = (G + az) * tan_tilt, math.hypot(ax, ay)
        if h > lim:
            ax, ay = ax * lim / max(h, 1e-9), ay * lim / max(h, 1e-9)
        s = dt * act[i]
        vx, vy, vz = v[i, 0] + ax * s, v[i, 1] + ay * s, v[i, 2] + az * s
        h = math.hypot(vx, vy)
        if h > vxy:
            vx, vy = vx * vxy / h, vy * vxy / h
        vz = min(max(vz, -vdown), vup)
        ax, ay, az = (vx - v[i, 0]) / dt, (vy - v[i, 1]) / dt, (vz - v[i, 2]) / dt + G
        v[i, 0], v[i, 1], v[i, 2] = vx, vy, vz
        p[i, 0] += vx * dt
        p[i, 1] += vy * dt
        p[i, 2] += vz * dt
        err = (yaw_sp[i] - yaw[i] + math.pi) % (2 * math.pi) - math.pi
        rate = min(max(_K_YAW * err, -yaw_max), yaw_max) * act[i]
        yaw[i] += rate * dt
        q[i, 0], q[i, 1], q[i, 2], q[i, 3] = math.cos(yaw[i] / 2), 0.0, 0.0, math.sin(yaw[i] / 2)
        w[i, 2] = rate
        c, sn = math.cos(yaw[i]), math.sin(yaw[i])
        f[i, 0] = (c * ax + sn * ay) * act[i] + f_rest[i, 0]
        f[i, 1] = (c * ay - sn * ax) * act[i] + f_rest[i, 1]
        f[i, 2] = az * act[i] + f_rest[i, 2]
        t = mass * math.sqrt(ax * ax + ay * ay + az * az) / 4 * act[i]
        for j in range(4):
            omega[i, j] = math.sqrt(t / kf)
        energy[i] += 4 * t ** 1.5 / k * dt


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
        az_max, tilt = accel_limits(prof)
        self._prm = (dt, prof.vmax_xy, prof.vmax_down, prof.vmax_up, az_max, float(tilt), prof.yaw_rate_max,
                     prof.mass, prof.kf, float(self._k))

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
        _step(self.p, self.v, self.yaw, self.q, self.w, self.f, self.f_rest, self.omega, self.energy, self.act,
              np.ascontiguousarray(a_des, float), np.ascontiguousarray(yaw_sp, float), self._prm)
