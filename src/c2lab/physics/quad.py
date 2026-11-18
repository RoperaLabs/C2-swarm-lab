"""Rigid-body quadrotor, vectorised over n drones. Convention: ENU world, FLU body, quaternion (w, x, y, z) body to world.

ponytail: no ground effect, no quadratic parasitic drag and no wake between drones; add them as extra terms in
`Quad.step` (force on the body) if a scenario needs them.
"""
import numpy as np

from .profiles import G

_BLOCK = 64  # ticks of thrust kept before they are folded into the energy integral


def _qmul(a, b):
    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                     a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                     a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                     a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def _kq(scale):
    """K[i, j, k]: coefficient of q_i * w_j in component k of scale * (q x (0, w))."""
    k = np.zeros((4, 3, 4))
    for i in range(4):
        for j in range(3):
            qi, wj = np.zeros(4), np.zeros(4)
            qi[i], wj[j + 1] = 1.0, 1.0
            k[i, j] = scale * _qmul(qi, wj)
    return k.reshape(4, 12)


def _rot_matrix():
    """C (16, 9): vec(q q^T) @ C = vec(R(q)); homogeneous in q, so it needs no normalisation."""
    c = np.zeros((4, 4, 9))
    for e, terms in enumerate([
            [(0, 0, 1), (1, 1, 1), (2, 2, -1), (3, 3, -1)], [(1, 2, 2), (0, 3, -2)], [(1, 3, 2), (0, 2, 2)],
            [(1, 2, 2), (0, 3, 2)], [(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, -1)], [(2, 3, 2), (0, 1, -2)],
            [(1, 3, 2), (0, 2, -2)], [(2, 3, 2), (0, 1, 2)], [(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, 1)]]):
        for i, j, s in terms:
            c[i, j, e] += s
    return c.reshape(16, 9)


_C = _rot_matrix()


def quat_to_R(q):
    """(n, 4) quaternions -> (n, 3, 3) rotation matrices."""
    return ((q[:, :, None] * q[:, None, :]).reshape(len(q), 16) @ _C).reshape(len(q), 3, 3)


def quat_yaw(yaw):
    out = np.zeros((len(yaw), 4))
    out[:, 0], out[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    return out


def mixer(prof):
    """A (4, 4) with omega^2 @ A = (thrust, torque x, torque y, torque z) in the body frame."""
    x, y = np.array(prof.rotors).T
    return np.stack([np.full(4, prof.kf), y * prof.kf, -x * prof.kf, np.array(prof.spin) * prof.km], axis=1)


class Quad:
    def __init__(self, prof, n, rng, wind, gust_sigma, gust_tau, dt):
        self.prof, self.n, self.dt, self.rng = prof, n, dt, rng
        self.p = np.zeros((n, 3))
        self._vw = np.zeros((n, 6))                  # v and w share one array: one update for both
        self.v, self.w = self._vw[:, :3], self._vw[:, 3:]
        self.q = np.zeros((n, 4))
        self.q[:, 0] = 1.0
        self.R = np.tile(np.eye(3), (n, 1, 1))
        self.f = np.zeros((n, 3))
        self.f_rest = np.zeros((n, 3))               # (0, 0, g) for landed drones, 0 otherwise
        self.omega = np.zeros((n, 4))
        self._energy = np.zeros(n)
        self._buf, self._i = np.zeros((_BLOCK, n, 4)), 0
        self.A = mixer(prof)
        self._a_up, self._a_dn = (1 - np.exp(-dt / prof.tau_up)), (1 - np.exp(-dt / prof.tau_down))
        self._kq = _kq(0.5 * dt)
        j = np.array(prof.inertia)
        self._inv_j = 1 / j
        self._gyro = np.array([j[1] - j[2], j[2] - j[0], j[0] - j[1]]) / j       # w x (J w) / J, up to w1 * w2
        self._cd = np.array([prof.c_xy, prof.c_xy, prof.c_z])
        self._gvec = np.array([0.0, 0.0, -G])
        self._wind0 = np.tile(np.asarray(wind, float), (n, 1))
        self._windy = bool(np.any(self._wind0)) or gust_sigma > 0
        self._gust = np.zeros((n, 3))
        self._wind = self._wind0.copy()
        self._ou = None
        if gust_sigma > 0:
            a = np.exp(-dt / gust_tau)
            self._ou = (a, gust_sigma * np.sqrt(1 - a * a))
            self._noise, self._ni = rng.standard_normal((_BLOCK, n, 3)), 0
        self.act = np.ones(n)                        # 1 while integrated (alive and not landed)
        self._act_col, self._act_dt = self.act[:, None], self.act[:, None] * dt
        self._d, self._d4 = np.zeros((n, 4)), np.zeros((n, 4))
        self._dvw, self._acc = np.zeros((n, 6)), np.zeros((n, 3, 1))
        self._F, self._tmp3 = np.zeros((n, 3)), np.zeros((n, 3))
        self._dq = np.zeros((n, 1, 4))

    @property
    def energy(self):
        self._flush()
        return self._energy

    def _flush(self):
        t = self._buf[:self._i]
        k = self.prof.fm * self.prof.eta * np.sqrt(2 * 1.225 * np.pi * self.prof.prop_radius ** 2)
        self._energy += (t * np.sqrt(t)).sum(axis=(0, 2)) * (self.dt / k)
        self._i = 0

    def reset(self, p0, yaw0, omega0):
        self._flush()
        self._energy[:] = 0.0
        self.p[:], self._vw[:], self.f[:], self.f_rest[:] = p0, 0.0, 0.0, 0.0
        self.q[:] = quat_yaw(yaw0)
        self.R[:] = quat_to_R(self.q)
        self.omega[:] = omega0
        self._gust[:] = 0.0

    def set_active(self, act):
        self.act[:] = act
        self._act_dt[:] = self.act[:, None] * self.dt

    def cos_tilt(self, idx):
        return self.R[idx, 2, 2]

    def freeze(self, idx, landed):
        """Stop integrating drones idx; a landed drone also levels out (keeps only its heading)."""
        self._vw[idx], self.omega[idx], self.f[idx] = 0.0, 0.0, 0.0
        self.f_rest[idx] = (0.0, 0.0, G) if landed else 0.0
        if landed:
            self.q[idx] = quat_yaw(np.arctan2(self.R[idx, 1, 0], self.R[idx, 0, 0]))
            self.R[idx] = quat_to_R(self.q[idx])

    def step(self, cmd):
        """One tick: motors, forces, semi-implicit Euler (v, w first, then p, q with the new values)."""
        dt, om, d, R = self.dt, self.omega, self._d, self.R
        np.multiply(cmd, self._act_col, out=cmd)
        np.subtract(cmd, om, out=d)
        if self.prof.tau_up == self.prof.tau_down:
            d *= self._a_up
        else:
            d *= np.where(d > 0, self._a_up, self._a_dn)
        om += d
        w2 = np.multiply(om, om, out=self._d4)
        wr = w2 @ self.A                                       # (n, 4): thrust, torque x, y, z
        np.multiply(w2, self.prof.kf, out=self._buf[self._i])  # per-rotor thrust, kept for the energy
        self._i += 1
        if self._i == _BLOCK:
            self._flush()
        # force on the body: rotor drag + thrust
        if self._ou is not None:
            a, b = self._ou
            self._gust *= a
            self._gust += b * self._noise[self._ni]
            self._ni += 1
            if self._ni == _BLOCK:
                self._noise, self._ni = self.rng.standard_normal((_BLOCK, self.n, 3)), 0
            np.add(self._wind0, self._gust, out=self._wind)
        vrel = self.v - self._wind if self._windy else self.v
        F = np.multiply(np.matmul(vrel[:, None, :], R)[:, 0], self._cd, out=self._F)
        F *= -om.sum(1)[:, None]
        F[:, 2] += wr[:, 0]
        np.multiply(F, 1.0 / self.prof.mass, out=self.f)
        self.f += self.f_rest                                    # a landed drone feels the ground reaction
        dvw = self._dvw
        np.matmul(R, self.f[:, :, None], out=self._acc)
        np.add(self._acc[:, :, 0], self._gvec, out=dvw[:, :3])
        w = self.w
        g = np.multiply(w[:, (1, 2, 0)], w[:, (2, 0, 1)], out=self._tmp3)
        g *= self._gyro
        np.multiply(wr[:, 1:], self._inv_j, out=dvw[:, 3:])
        dvw[:, 3:] -= g
        dvw *= self._act_dt
        self._vw += dvw
        np.multiply(self.v, dt, out=self._tmp3)
        self.p += self._tmp3
        t3 = (self.q @ self._kq).reshape(self.n, 3, 4)
        np.matmul(w[:, None, :], t3, out=self._dq)
        self.q += self._dq[:, 0]
        self.q /= np.sqrt(np.einsum("ij,ij->i", self.q, self.q))[:, None]
        R[:] = quat_to_R(self.q).reshape(self.n, 3, 3)
