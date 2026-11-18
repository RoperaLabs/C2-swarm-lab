"""Rigid-body quadrotor, vectorised over n drones. Convention: ENU world, FLU body, quaternion (w, x, y, z) body to world.

One compiled kernel (`_tick`) runs a whole tick for every drone: the autopilot's attitude loop (when due) and rate
loop, the motors, the forces, a semi-implicit Euler step and the cheap part of the collision check.

ponytail: no ground effect, no quadratic parasitic drag and no wake between drones; add them as extra terms on F in
`_tick` if a scenario needs them.
"""
import math

import numpy as np
from numba import njit

from .profiles import G, RHO

_BLOCK = 64  # gust noise is drawn in blocks of this many ticks


def quat_to_R(q):
    """(n, 4) quaternions -> (n, 3, 3) rotation matrices."""
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def quat_yaw(yaw):
    out = np.zeros((len(yaw), 4))
    out[:, 0], out[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    return out


def mixer(prof):
    """A (4, 4) with omega^2 @ A = (thrust, torque x, torque y, torque z) in the body frame."""
    x, y = np.array(prof.rotors).T
    return np.stack([np.full(4, prof.kf), y * prof.kf, -x * prof.kf, np.array(prof.spin) * prof.km], axis=1)


@njit(cache=True)
def _tick(S, AP, DY, use_est, do_att, prm):
    """One tick for every drone. S: model state (see `Quad`), AP: autopilot inner state, DY: estimator yaw error,
    prm: packed parameters (see `Quad.bind`). Returns True when some drone needs the full contact check."""
    n = len(S)
    kp_att, w2max, kf, mass, a_up, a_dn, dt, e_k, radius = (prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6],
                                                           prm[7], prm[8])
    need = False
    eR = np.empty(9)
    for i in range(n):
        if S[i, 32] == 0.0:
            continue
        if use_est:     # estimated attitude = yaw error applied to the true one
            c, s = math.cos(DY[i]), math.sin(DY[i])
            for b in range(3):
                eR[b], eR[3 + b], eR[6 + b] = (c * S[i, 13 + b] - s * S[i, 16 + b], s * S[i, 13 + b] + c * S[i, 16 + b],
                                               S[i, 19 + b])
        else:
            eR[:] = S[i, 13:22]
        ew = S[i, 6:9]
        if do_att:      # attitude loop: Re = Rd^T R, rates from vee(Re - Re^T) / 2, collective thrust
            re = np.zeros(9)
            for a in range(3):
                for b in range(3):
                    re[3 * a + b] = AP[i, 3 * a] * eR[b] + AP[i, 3 * a + 1] * eR[3 + b] + AP[i, 3 * a + 2] * eR[6 + b]
            vee = (0.5 * (re[7] - re[5]), 0.5 * (re[2] - re[6]), 0.5 * (re[3] - re[1]))
            for a in range(3):
                AP[i, 12 + a] = min(max(-kp_att * vee[a], -prm[18 + a]), prm[18 + a])
                AP[i, 15 + a] = min(max(AP[i, 15 + a], -prm[15 + a]), prm[15 + a])
            AP[i, 18] = max(AP[i, 9] * eR[2] + AP[i, 10] * eR[5] + AP[i, 11] * eR[8], 0.0)
        for a in range(3):  # rate loop
            e = AP[i, 12 + a] - ew[a]
            AP[i, 15 + a] += e * prm[12 + a]
            AP[i, 19 + a] = e * prm[9 + a] + AP[i, 15 + a]
        thrust, tx, ty, tz, osum, power = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for m in range(4):  # allocation inverse, first-order motors, rotor forces and power
            c = 0.0
            for a in range(4):
                c += AP[i, 18 + a] * prm[36 + 4 * a + m]
            d = math.sqrt(min(max(c, 0.0), w2max)) - S[i, 22 + m]
            S[i, 22 + m] += d * (a_up if d > 0 else a_dn)
            om = S[i, 22 + m]
            w2 = om * om
            thrust += w2 * prm[52 + 4 * m]
            tx += w2 * prm[53 + 4 * m]
            ty += w2 * prm[54 + 4 * m]
            tz += w2 * prm[55 + 4 * m]
            osum += om
            t_m = kf * w2
            power += t_m * math.sqrt(t_m)
        S[i, 33] += power * dt / e_k
        vr0, vr1, vr2 = S[i, 3] - S[i, 34], S[i, 4] - S[i, 35], S[i, 5] - S[i, 36]
        for a in range(3):  # specific force in the body frame: rotor drag on the air-relative velocity + thrust
            vb = S[i, 13 + a] * vr0 + S[i, 16 + a] * vr1 + S[i, 19 + a] * vr2
            S[i, 26 + a] = -osum * prm[27 + a] * vb / mass + S[i, 29 + a]
        S[i, 28] += thrust / mass
        for a in range(3):
            S[i, 3 + a] += (S[i, 13 + 3 * a] * S[i, 26] + S[i, 14 + 3 * a] * S[i, 27] + S[i, 15 + 3 * a] * S[i, 28]
                            - (G if a == 2 else 0.0)) * dt
        w0, w1, w2_ = S[i, 6], S[i, 7], S[i, 8]
        S[i, 6] += (tx * prm[21] - prm[24] * w1 * w2_) * dt
        S[i, 7] += (ty * prm[22] - prm[25] * w2_ * w0) * dt
        S[i, 8] += (tz * prm[23] - prm[26] * w0 * w1) * dt
        for a in range(3):
            S[i, a] += S[i, 3 + a] * dt
        qw, qx, qy, qz = S[i, 9], S[i, 10], S[i, 11], S[i, 12]   # q += dt/2 q x (0, w), normalised
        ox, oy, oz = S[i, 6], S[i, 7], S[i, 8]
        h = 0.5 * dt
        qw, qx, qy, qz = (qw - h * (qx * ox + qy * oy + qz * oz), qx + h * (qw * ox + qy * oz - qz * oy),
                          qy + h * (qw * oy - qx * oz + qz * ox), qz + h * (qw * oz + qx * oy - qy * ox))
        nq = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = qw / nq, qx / nq, qy / nq, qz / nq
        S[i, 9], S[i, 10], S[i, 11], S[i, 12] = qw, qx, qy, qz
        S[i, 13], S[i, 14], S[i, 15] = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)
        S[i, 16], S[i, 17], S[i, 18] = 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)
        S[i, 19], S[i, 20], S[i, 21] = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
        S[i, 37] -= math.sqrt(S[i, 3] ** 2 + S[i, 4] ** 2 + S[i, 5] ** 2) * dt   # cheap collision screening
        if S[i, 37] <= 0.0:
            need = True
        for a in range(3):
            if S[i, a] < prm[30 + a] or S[i, a] > prm[33 + a]:
                need = True
    for i in range(n):
        if S[i, 32] == 0.0:
            continue
        for j in range(i + 1, n):
            if S[j, 32] != 0.0 and ((S[i, 0] - S[j, 0]) ** 2 + (S[i, 1] - S[j, 1]) ** 2
                                    + (S[i, 2] - S[j, 2]) ** 2) < 4 * radius * radius:
                need = True
    return need


class Quad:
    def __init__(self, prof, n, rng, wind, gust_sigma, gust_tau, dt):
        self.prof, self.n, self.dt, self.rng = prof, n, dt, rng
        # all per-drone state in one array S for the compiled tick; the attributes are views of it
        self.S = np.zeros((n, 38))
        S = self.S
        self.p, self.v, self.w, self.q = S[:, 0:3], S[:, 3:6], S[:, 6:9], S[:, 9:13]
        self.R = S[:, 13:22].reshape(n, 3, 3)
        self.omega, self.f, self.f_rest = S[:, 22:26], S[:, 26:29], S[:, 29:32]   # f_rest: (0, 0, g) when landed
        self.act, self.energy, self.wind, self.budget = S[:, 32], S[:, 33], S[:, 34:37], S[:, 37]
        self.q[:] = quat_yaw(np.zeros(n))
        self.R[:] = quat_to_R(self.q)
        self.act[:] = 1.0                            # 1 while integrated (alive and not landed)
        self.A = mixer(prof)
        self._a_up, self._a_dn = 1 - np.exp(-dt / prof.tau_up), 1 - np.exp(-dt / prof.tau_down)
        j = np.array(prof.inertia)
        self._inv_j = 1 / j
        self._gyro = np.array([j[2] - j[1], j[0] - j[2], j[1] - j[0]]) / j     # (w x Jw) / J, per w_a w_b
        self._cd = np.array([prof.c_xy, prof.c_xy, prof.c_z])
        self._e_k = prof.fm * prof.eta * np.sqrt(2 * RHO * np.pi * prof.prop_radius ** 2)
        self._wind0 = np.tile(np.asarray(wind, float), (n, 1))
        self.wind[:] = self._wind0
        self._gust = np.zeros((n, 3))
        self._ou = None
        if gust_sigma > 0:
            a = np.exp(-dt / gust_tau)
            self._ou = (a, gust_sigma * np.sqrt(1 - a * a))
            self._noise, self._ni = rng.standard_normal((_BLOCK, n, 3)), 0

    def reset(self, p0, yaw0, omega0):
        self.p[:], self.v[:], self.w[:], self.f[:], self.f_rest[:] = p0, 0.0, 0.0, 0.0, 0.0
        self.q[:] = quat_yaw(yaw0)
        self.R[:] = quat_to_R(self.q)
        self.omega[:], self.energy[:], self._gust[:] = omega0, 0.0, 0.0

    def set_active(self, act):
        self.act[:] = act

    def cos_tilt(self, idx):
        return self.R[idx, 2, 2]

    def freeze(self, idx, landed):
        """Stop integrating drones idx; a landed drone also levels out (keeps only its heading)."""
        self.v[idx], self.w[idx], self.omega[idx], self.f[idx] = 0.0, 0.0, 0.0, 0.0
        self.f_rest[idx] = (0.0, 0.0, G) if landed else 0.0
        if landed:
            self.q[idx] = quat_yaw(np.arctan2(self.R[idx, 1, 0], self.R[idx, 0, 0]))
            self.R[idx] = quat_to_R(self.q[idx])

    def bind(self, ap, lo, hi):
        """Pack the parameters read by the compiled tick (autopilot gains, model constants, geofence)."""
        pr = self.prof
        self._prm = np.r_[ap.kp_att, ap.w2max, pr.kf, pr.mass, self._a_up, self._a_dn, self.dt, self._e_k, pr.radius,
                          ap.kpj, ap.kij, ap.ij_max, ap.w_max, self._inv_j, self._gyro, self._cd, lo, hi,
                          ap.ainv.ravel(), self.A.ravel()]
        self._ap = ap
        self._noyaw = np.zeros(1)

    def tick(self, dyaw, do_att):
        """One tick; dyaw (n,) = estimator yaw error, or None for the true attitude."""
        if self._ou is not None:
            a, b = self._ou
            self._gust *= a
            self._gust += b * self._noise[self._ni]
            self._ni += 1
            if self._ni == _BLOCK:
                self._noise, self._ni = self.rng.standard_normal((_BLOCK, self.n, 3)), 0
            np.add(self._wind0, self._gust, out=self.wind)
        return _tick(self.S, self._ap.AP, self._noyaw if dyaw is None else dyaw, dyaw is not None, do_att, self._prm)
