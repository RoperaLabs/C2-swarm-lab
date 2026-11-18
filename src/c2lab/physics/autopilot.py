"""PX4-style cascade on the *estimated* state, vectorised over n drones.

Outer loop (50 Hz): position P + velocity PI -> desired acceleration -> thrust vector and desired attitude.
Attitude loop (600 Hz): geometric P on SO(3) -> desired body rates, collective thrust.
Rate loop (every tick): PI -> torques -> allocation inverse -> motor speed commands.

ponytail: no derivative term in the rate loop (no estimator noise model yet) and plain clipping instead of
mixer desaturation; add them here if a noisy estimator or saturated manoeuvres need it.
"""
import numpy as np

from .profiles import G
from .quad import mixer

_TILT_MARGIN = 0.85   # the commanded tilt stays below tilt_max so that attitude overshoot does not cross it
_AZ_MAX = 0.5 * G     # vertical acceleration limit, +- (also used by the point mass)
_V = np.zeros((9, 3))                                          # vec(Re) -> vee(Re - Re^T) / 2
for _k, (_a, _b) in enumerate([(7, 5), (2, 6), (3, 1)]):
    _V[_a, _k], _V[_b, _k] = 0.5, -0.5


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
        self.a_des, self.f_des = np.zeros((n, 3)), np.zeros((n, 3))
        self.rd_t = np.zeros((n, 3, 3))
        self.w_des, self.i_rate = np.zeros((n, 3)), np.zeros((n, 3))
        self.wrench = np.zeros((n, 4))
        j = np.array(prof.inertia)
        self._kpj, self._kij = np.array(prof.kp_rate) * j, np.array(prof.ki_rate) * j * dt
        self._ij_max = 3.0 * self._kpj                          # integral clamp, torque
        self._w_max = np.array([8.0, 8.0, prof.yaw_rate_max])
        self._kp_att = -prof.kp_att
        self._ainv = np.linalg.inv(mixer(prof))
        self._w2max = prof.omega_max ** 2
        self._cmd, self._e, self._tmp = np.zeros((n, 4)), np.zeros((n, 3)), np.zeros((n, 3))

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

    def attitude(self, R):
        """600 Hz: desired body rates and collective thrust from the estimated attitude R (n, 3, 3)."""
        re = np.matmul(self.rd_t, R)
        np.clip(re.reshape(self.n, 9) @ _V * self._kp_att, -self._w_max, self._w_max, out=self.w_des)
        np.clip(np.einsum("ij,ij->i", self.f_des, R[:, :, 2]), 0.0, None, out=self.wrench[:, 0])
        np.clip(self.i_rate, -self._ij_max, self._ij_max, out=self.i_rate)

    def rates(self, w):
        """Every tick: motor speed commands (n, 4), rad/s."""
        e, wr = self._e, self.wrench
        np.subtract(self.w_des, w, out=e)
        np.multiply(e, self._kij, out=self._tmp)
        self.i_rate += self._tmp
        np.multiply(e, self._kpj, out=wr[:, 1:])
        wr[:, 1:] += self.i_rate
        c = np.matmul(wr, self._ainv, out=self._cmd)
        np.clip(c, 0.0, self._w2max, out=c)
        return np.sqrt(c, out=c)
