"""`Drones`: n drones with one model (quad or point mass), the autopilot, ground contact and collision events."""
import numpy as np

from c2lab.brain import Position, Velocity

from .autopilot import Autopilot
from .pointmass import PointMass
from .profiles import PROFILES
from .quad import Quad, quat_to_R

_MAX_DIST = 5.0                     # m, cap of the safe-distance budget (re-check at least every 5 m flown)
_LAND_SPEED = 1.0                   # m/s, faster ground contact is a crash
_LAND_COS_TILT = np.cos(np.radians(20.0))
_CLIMB = 0.2                        # m/s or m of climb that triggers a take-off


def _yaw(R):
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


class Drones:
    def __init__(self, profile, n, rng, world, mode="quad", wind=(0.0, 0.0, 0.0), gust_sigma=0.0, gust_tau=2.0,
                 dt=1 / 1200):
        self.prof = PROFILES[profile] if isinstance(profile, str) else profile
        self.n, self.dt, self.world, self.mode = n, dt, world, mode
        self.model = (Quad(self.prof, n, rng, wind, gust_sigma, gust_tau, dt) if mode == "quad"
                      else PointMass(self.prof, n, dt))
        self.ap = Autopilot(self.prof, n, dt)
        self._k_pos, self._k_att = round(1 / (50 * dt)), round(1 / (600 * dt))
        self.alive, self.landed = np.ones(n, bool), np.zeros(n, bool)
        self._budget = self.model.budget if mode == "quad" else np.full(n, -1.0)
        self._est_p = np.zeros((n, 3))
        self._lo, self._hi = (np.asarray(b, float) for b in world.bounds)
        if mode == "quad":
            self.model.bind(self.ap, self._lo, self._hi)

    p = property(lambda s: s.model.p)
    v = property(lambda s: s.model.v)
    q = property(lambda s: s.model.q)
    w = property(lambda s: s.model.w)
    f = property(lambda s: s.model.f)
    omega = property(lambda s: s.model.omega)
    energy = property(lambda s: s.model.energy)

    def reset(self, p0, yaw0, landed=True):
        p0, yaw0 = np.array(p0, float), np.asarray(yaw0, float)
        r = self.prof.radius
        if landed:
            p0[:, 2] = self.world.ground(p0[:, :2]) + r
        self.model.reset(p0, yaw0, 0.0 if landed else self.prof.hover_omega)
        idx = np.arange(self.n)
        self.ap.arm(idx, yaw0)
        self.ap.mode_vel[:], self.ap.p_tgt[:], self.ap.yaw_sp[:], self.ap.yaw_rate[:] = False, p0, yaw0, 0.0
        self.alive[:], self.landed[:] = True, landed
        self._est_p[:] = p0
        self._budget[:] = -1.0
        if landed:
            self.model.freeze(idx, True)
        self._sync_active()

    def set_setpoint(self, i, sp):
        ap = self.ap
        ap.yaw_rate[i] = 0.0
        if isinstance(sp, Velocity):
            ap.mode_vel[i], ap.v_cmd[i], ap.body[i], ap.yaw_rate[i] = True, sp.v, sp.frame == "body", sp.yaw_rate
        else:
            ap.mode_vel[i] = False
            ap.p_tgt[i] = self._est_p[i] + (sp.dp if isinstance(sp, Position) else 0.0)
            if isinstance(sp, Position) and sp.yaw is not None:
                ap.yaw_sp[i] = sp.yaw

    def _sync_active(self):
        self.model.set_active((self.alive & ~self.landed).astype(float))

    def _stop(self, idx, landed):
        self.model.freeze(idx, landed)
        if landed:
            self.landed[idx] = True
            self.model.p[idx, 2] = self.world.ground(self.model.p[idx, :2]) + self.prof.radius
            self.ap.mode_vel[idx], self.ap.p_tgt[idx] = False, self.model.p[idx]
        else:
            self.alive[idx] = False
        self._sync_active()

    def step(self, tick, est=None):
        """Advance one tick; est (attributes p, v, q, w) feeds the autopilot, None = true state."""
        m, ap, events = self.model, self.ap, []
        if est is None:
            ep, ev, ew = m.p, m.v, m.w
            eR = m.R if self.mode == "quad" else quat_to_R(m.q)
        else:
            ep, ev, ew, eR = est.p, est.v, est.w, quat_to_R(est.q)
        self._est_p[:] = ep

        if self.landed.any():
            climb = np.where(ap.mode_vel, ap.v_cmd[:, 2], ap.p_tgt[:, 2] - ep[:, 2]) > _CLIMB
            up = np.flatnonzero(self.landed & self.alive & climb)
            if up.size:
                self.landed[up] = False
                m.f_rest[up] = 0.0
                m.omega[up] = self.prof.hover_omega         # motors spooled up on arming
                ap.arm(up, _yaw(eR[up]))
                self._budget[up] = -1.0
                self._sync_active()
                events += [(tick, int(i), "takeoff") for i in up]

        if tick % self._k_pos == 0:
            ap.outer(ep, ev, _yaw(eR))
        if self.mode == "quad":
            E = None if est is None else np.c_[eR.reshape(-1, 9), ew]
            need = m.tick(E, tick % self._k_att == 0)
        else:
            m.step(ap.a_des, ap.yaw_sp)
            self._budget -= np.linalg.norm(m.v, axis=1) * self.dt
            need = True
        if need:
            fly = np.flatnonzero(self.alive & ~self.landed)
            if fly.size:
                events += self._contacts(tick, fly)
        return events

    def _contacts(self, tick, fly):
        m, r, w, events = self.model, self.prof.radius, self.world, []
        p = m.p
        out = fly[((p[fly] < self._lo) | (p[fly] > self._hi)).any(1)]
        if out.size:
            self._stop(out, False)
            events += [(tick, int(i), "geofence") for i in out]
            fly = np.setdiff1d(fly, out)

        chk = fly[self._budget[fly] <= 0]
        if chk.size:
            pc = p[chk]
            hit = w.sdf(pc, terrain=False, max_dist=_MAX_DIST) < r
            if hit.any():
                bad = chk[hit]
                self._stop(bad, False)
                events += [(tick, int(i), "crash_obstacle") for i in bad]
                chk, pc = chk[~hit], pc[~hit]
            low = (pc[:, 2] - w.ground(pc[:, :2]) <= r) & (m.v[chk, 2] < 0)
            if low.any():
                g = chk[low]
                soft = (np.linalg.norm(m.v[g], axis=1) <= _LAND_SPEED) & (m.cos_tilt(g) >= _LAND_COS_TILT)
                if soft.any():
                    self._stop(g[soft], True)
                    events += [(tick, int(i), "landed") for i in g[soft]]
                if (~soft).any():
                    self._stop(g[~soft], False)
                    events += [(tick, int(i), "crash_terrain") for i in g[~soft]]
                chk = chk[~low]
            if chk.size:
                self._budget[chk] = w.sdf(p[chk], terrain=True, max_dist=_MAX_DIST) - r

        fly = np.flatnonzero(self.alive & ~self.landed)
        if fly.size > 1:
            d = np.linalg.norm(p[fly, None] - p[None, fly], axis=2)
            np.fill_diagonal(d, np.inf)
            bad = fly[(d < 2 * r).any(1)]
            if bad.size:
                self._stop(bad, False)
                events += [(tick, int(i), "crash_drone") for i in bad]
        return events
