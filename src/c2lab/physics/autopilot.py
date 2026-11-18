"""PX4-style cascade on the *estimated* state, vectorised over n drones.

Outer loop (50 Hz): position P + velocity PI -> desired acceleration -> thrust vector and desired attitude.
Attitude loop (600 Hz) and rate loop (every tick) run inside the compiled tick of `quad.py`: geometric P on SO(3)
-> desired body rates and collective thrust, then rate PI -> torques -> allocation inverse -> motor commands.

ponytail: no derivative term in the rate loop (no estimator noise model yet) and plain clipping instead of
mixer desaturation; add them here if a noisy estimator or saturated manoeuvres need it.
"""
import numpy as np

from .profiles import G
from .quad import mixer

_TILT_MARGIN = 0.85   # the commanded tilt stays below tilt_max so that attitude overshoot does not cross it
_AZ_MAX = 0.5 * G     # vertical acceleration limit, +- (also used by the point mass)


def accel_limits(prof):
    return _AZ_MAX, G * np.tan(_TILT_MARGIN * prof.tilt_max)


def limit_accel(a, prof):
    """Clip a (n, 3) acceleration in place: vertical bound, then horizontal bound tied to the thrust tilt."""
    np.clip(a[:, 2], -_AZ_MAX, _AZ_MAX, out=a[:, 2])
    lim = (G + a[:, 2]) * np.tan(_TILT_MARGIN * prof.tilt_max)
    h = np.hypot(a[:, 0], a[:, 1])
    a[:, :2] *= np.minimum(1.0, lim / np.maximum(h, 1e-9))[:, None]
    return a


class Autopilot:
    def __init__(self, prof, n, dt):
        self.prof, self.n, self.dt = prof, n, dt
        self.dt_pos = round(1 / (50 * dt)) * dt
        self.mode_vel = np.zeros(n, bool)
        self.v_cmd, self.body, self.yaw_rate = np.zeros((n, 3)), np.zeros(n, bool), np.zeros(n)
        self.p_tgt, self.yaw_sp = np.zeros((n, 3)), np.zeros(n)
        self.i_vel = np.zeros((n, 3))
        self.a_des = np.zeros((n, 3))
        self.AP = np.zeros((n, 22))           # inner-loop state, one array for the compiled tick; views below
        self.rd_t = self.AP[:, 0:9].reshape(n, 3, 3)
        self.f_des, self.w_des, self.i_rate = self.AP[:, 9:12], self.AP[:, 12:15], self.AP[:, 15:18]
        self.wrench = self.AP[:, 18:22]
        j = np.array(prof.inertia)
        self.kpj, self.kij = np.array(prof.kp_rate) * j, np.array(prof.ki_rate) * j * dt
        self.ij_max = 3.0 * self.kpj                            # integral clamp, torque
        self.w_max = np.array([8.0, 8.0, prof.yaw_rate_max])
        self.kp_att = prof.kp_att
        self.ainv = np.linalg.inv(mixer(prof))
        self.w2max = prof.omega_max ** 2

    def arm(self, idx, yaw):
        """Hover-level state for drones idx with heading yaw (reset and take-off)."""
        c, s = np.cos(yaw), np.sin(yaw)
        self.rd_t[idx] = 0.0
        self.rd_t[idx, 0, 0], self.rd_t[idx, 0, 1] = c, s
        self.rd_t[idx, 1, 0], self.rd_t[idx, 1, 1] = -s, c
        self.rd_t[idx, 2, 2] = 1.0
        self.a_des[idx], self.i_vel[idx], self.i_rate[idx], self.w_des[idx] = 0.0, 0.0, 0.0, 0.0
        self.f_des[idx] = (0.0, 0.0, self.prof.mass * G)
        self.wrench[idx] = (self.prof.mass * G, 0.0, 0.0, 0.0)

    def outer(self, p, v, yaw):
        """50 Hz: setpoint -> velocity target -> acceleration -> desired attitude."""
        pr, dt = self.prof, self.dt_pos
        c, s = np.cos(yaw), np.sin(yaw)
        vc = self.v_cmd
        rot = np.stack([c * vc[:, 0] - s * vc[:, 1], s * vc[:, 0] + c * vc[:, 1], vc[:, 2]], axis=1)
        vt = np.where(self.mode_vel[:, None], np.where(self.body[:, None], rot, vc), pr.kp_pos * (self.p_tgt - p))
        h = np.hypot(vt[:, 0], vt[:, 1])
        vt[:, :2] *= np.minimum(1.0, pr.vmax_xy / np.maximum(h, 1e-9))[:, None]
        np.clip(vt[:, 2], -pr.vmax_down, pr.vmax_up, out=vt[:, 2])
        self.yaw_sp += self.yaw_rate * dt
        self.yaw_sp[:] = (self.yaw_sp + np.pi) % (2 * np.pi) - np.pi
        e = vt - v
        i_new = np.clip(self.i_vel + pr.ki_vel * dt * e, -_AZ_MAX, _AZ_MAX)
        raw = pr.kp_vel * e + i_new
        a = limit_accel(raw.copy(), pr)
        free = (np.abs(a - raw) < 1e-9).all(1)            # anti-windup: integrate only while not saturated
        self.i_vel[free] = i_new[free]
        self.a_des[:] = a
        f = self.f_des
        f[:, :2], f[:, 2] = a[:, :2] * pr.mass, (a[:, 2] + G) * pr.mass
        zd = f / np.linalg.norm(f, axis=1)[:, None]
        cy, sy = np.cos(self.yaw_sp), np.sin(self.yaw_sp)
        yd = np.stack([-zd[:, 2] * sy, zd[:, 2] * cy, zd[:, 0] * sy - zd[:, 1] * cy], axis=1)   # z_d x x_c
        yd /= np.linalg.norm(yd, axis=1)[:, None]
        xd = np.cross(yd, zd)
        self.rd_t[:, 0], self.rd_t[:, 1], self.rd_t[:, 2] = xd, yd, zd
