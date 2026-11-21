"""On-board sensors. Each samples the true state at its own rate, adds its own seeded noise and reports in the drone's
frames; nothing else of the world reaches a brain. Defaults are typical datasheet values (VL53L5CX, Livox Mid-360,
RealSense D435, BMI088, u-blox M10, BMP388); every one can be overridden in the scenario."""
import math

import numpy as np
from numba import njit

from c2lab.brain import Reading
from c2lab.physics.profiles import G
from c2lab.physics.quad import quat_to_R
from c2lab.scenario import PHYSICS_HZ
from c2lab.seeds import rng as stream
from c2lab.world.primitives import sphere_hits

MOUNTS = {"front": (0, 0), "left": (90, 0), "back": (180, 0), "right": (-90, 0), "up": (0, 90), "down": (0, -90)}

DEFAULTS = {
    "tof": {"arrays": ["front"], "zones": 8, "fov_deg": 45.0, "range_m": 4.0, "min_m": 0.02, "hz": 15, "sub": 2,
            "sigma_rel": 0.03, "sigma_min": 0.005, "dropout": 0.01, "drones": True},
    "lidar": {"channels": 16, "vfov_deg": [-7.0, 52.0], "hres_deg": 2.0, "range_m": 40.0, "min_m": 0.1, "hz": 10,
              "sigma": 0.02, "dropout": 0.01, "drones": True},
    "depth": {"width": 64, "height": 48, "hfov_deg": 87.0, "vfov_deg": 58.0, "range_m": 10.0, "min_m": 0.3, "hz": 15,
              "sigma_k": 0.01, "drones": True},
    "detector": {"fov_deg": 90.0, "range_m": 30.0, "hz": 10, "p_detect": 0.95, "sigma_deg": 1.0, "sigma_rel": 0.05,
                 "false_rate": 0.02, "p_confuse": 0.02},
    "pose": {"profile": "ekf", "hz": 50, "drift": 0.01, "max_h": 4.0},
    "imu": {"hz": 400, "acc_density": 1.7e-3, "gyro_density": 2.4e-4, "acc_bias_walk": 1e-4, "gyro_bias_walk": 1e-5},
    "gnss": {"hz": 5, "sigma_h": 1.5, "sigma_v": 3.0, "tau_s": 60.0, "sigma_vel": 0.1, "denied": [], "sky": False},
    "baro": {"hz": 50, "sigma": 0.15, "drift": 0.01},
    "mag": {"hz": 50, "sigma_deg": 2.0},
    "prior_map": {"res_m": 2.0, "sigma": 0.3, "omit": ["cable", "pole", "lamp", "bush"]},
    "battery": {"hz": 1},
}
CAPACITY_J = {"cf2": 250e-3 * 3.7 * 3600, "iris": 5.1 * 11.1 * 3600}
_POSE = {"ideal": (0.0, 0.0, 0.0), "ekf": (0.05, 0.03, math.radians(0.5)), "vio": (0.02, 0.03, math.radians(0.5)),
         "flow": (0.0, 0.0, math.radians(0.5))}   # white noise: position, velocity, yaw


def _dir(az, el):
    """Unit vectors for azimuth/elevation (deg) in the sensor frame (x forward, y left, z up)."""
    az, el = np.broadcast_arrays(np.radians(az), np.radians(el))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def _mount(name, d):
    yaw, pitch = (math.radians(a) for a in MOUNTS[name])
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])    # positive pitch looks up
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return d @ (Rz @ Ry).T


def _pinhole(w, h, hfov, vfov, sub=1):
    """Directions through a w x h grid (sub x sub samples per cell), shape (h, w, sub*sub, 3), sensor frame."""
    u = (np.arange(w * sub) + 0.5) / (w * sub) * 2 - 1
    v = (np.arange(h * sub) + 0.5) / (h * sub) * 2 - 1
    U, V = np.meshgrid(u * math.tan(math.radians(hfov / 2)), -v * math.tan(math.radians(vfov / 2)))
    d = np.stack([np.ones_like(U), -U, V], -1)          # image x to the right = -y (left)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(h, sub, w, sub, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, sub * sub, 3)


def _ro(a):
    a = np.array(a, float)
    a.flags.writeable = False
    return a


class Sensors:
    def __init__(self, cfg, n, seed, ids, world, drones, labels, home, profile="iris"):
        self.n, self.world, self.drones = n, world, drones
        self.labels, self.home = np.asarray(labels), np.asarray(home, float)
        self.cfg = {k: {**DEFAULTS[k], **v} for k, v in cfg.items()}
        self.rng = {k: [stream(seed, "agent", int(i), k) for i in ids] for k in self.cfg}
        self.period = {k: max(round(PHYSICS_HZ / c["hz"]), 1) for k, c in self.cfg.items() if "hz" in c}
        self.buf = [{k: [] for k in self.cfg} for _ in range(n)]
        self.capacity = CAPACITY_J.get(profile, 1.0)
        self._rays = {}
        self.hits = None                      # list: the coverage mission collects ray hit points here
        self.peers, self.g0 = None, 0          # peers() -> (p, alive, labels) of every drone in the run
        if "tof" in self.cfg:
            c = self.cfg["tof"]
            d = _pinhole(c["zones"], c["zones"], c["fov_deg"], c["fov_deg"], c["sub"])
            self._rays["tof"] = np.stack([_mount(m, d) for m in c["arrays"]])          # (a, z, z, s, 3)
        if "lidar" in self.cfg:
            c = self.cfg["lidar"]
            az = np.arange(0, 360, c["hres_deg"])
            el = np.linspace(*c["vfov_deg"], c["channels"])
            self._rays["lidar"] = _dir(az[None, :], el[:, None])                      # (ch, n_az, 3)
        if "depth" in self.cfg:
            c = self.cfg["depth"]
            self._rays["depth"] = _pinhole(c["width"], c["height"], c["hfov_deg"], c["vfov_deg"])[:, :, 0]
        self._pose_err = np.zeros((n, 7))      # position error (3), velocity error (3), yaw error
        self._drift = np.zeros((n, 3))
        self._last_p = np.array(drones.p)
        self._imu_bias = np.zeros((n, 6))
        self._gnss_err = np.zeros((n, 3))
        self._baro_drift = np.zeros(n)
        self.err = None                       # estimator errors for the autopilot (None = ideal)
        if "pose" in self.cfg and self.cfg["pose"]["profile"] != "ideal":
            e = self._pose_err
            self.err = type("PoseError", (), {"dp": e[:, :3], "dv": e[:, 3:6], "dyaw": e[:, 6]})()
        if "prior_map" in self.cfg:              # the map a survey would give: no thin objects
            d = world.to_dict()
            omit = set(self.cfg["prior_map"]["omit"])
            d["primitives"] = [q for q in d["primitives"] if q.get("tag") not in omit]
            self._survey = type(world).from_dict(d)

    def public(self, name):
        return {k: v for k, v in self.cfg[name].items() if k not in ("dropout",)}

    def pop(self, i):
        out = {k: tuple(v) for k, v in self.buf[i].items() if v}
        for v in self.buf[i].values():
            v.clear()
        return out

    def _put(self, name, t, data):
        for i in range(self.n):
            self.buf[i][name].append(Reading(t, _ro(data[i])))

    def step(self, tick):
        t = tick / PHYSICS_HZ
        if "pose" in self.cfg:
            if tick % self.period["pose"] == 0:
                self._pose_update()
        for name in self.cfg:
            if name == "prior_map":
                if tick == 0:
                    self._put(name, t, [self._prior_map(i) for i in range(self.n)])
            elif tick % self.period[name] == 0:
                self._put(name, t, getattr(self, "_" + name)())

    # --- ray sensors -------------------------------------------------------------------------------------------
    def _crowd(self):
        """Positions, liveness and team labels of every drone the sensors can see, and this team's offset."""
        if self.peers is None:
            return self.drones.p, self.drones.alive, self.labels, 0
        return (*self.peers(), self.g0)

    def _cast(self, local, rng_m, drones=True):
        d = self.drones
        sh = local.shape[:-1]
        flat = local.reshape(-1, 3)
        R = d.model.R if hasattr(d.model, "R") else quat_to_R(d.q)
        dirs = np.matmul(R, flat.T).transpose(0, 2, 1)
        o = np.ascontiguousarray(d.p)
        dirs = np.ascontiguousarray(dirs.reshape(-1, 3))
        dist = self.world.cast(o, dirs, rng_m)
        if drones:                                               # other drones are spheres to the sensor
            P, A, _, off = self._crowd()
            far = np.inf if self.hits is not None else float(rng_m)    # beyond the range only coverage sees a hit
            sphere_hits(o, dirs, len(flat), off, np.ascontiguousarray(P), A, d.prof.radius, far, dist)
        if self.hits is not None:
            ok = np.isfinite(dist)
            self.hits.append(np.repeat(o, len(flat), 0)[ok] + dirs[ok] * dist[ok][:, None])
        return dist.reshape((self.n,) + sh)

    def _noisy(self, name, dist, sigma, lo, hi, dropout=0.0):
        out = dist.copy()
        for i in range(self.n):
            r = self.rng[name][i]
            out[i] += r.standard_normal(dist[i].shape) * sigma[i]
            if dropout:
                out[i][r.random(dist[i].shape) < dropout] = np.nan
        out[(dist < lo) | (dist > hi) | ~np.isfinite(dist)] = np.nan
        return out

    def _tof(self):
        c = self.cfg["tof"]
        dist = self._cast(self._rays["tof"], c["range_m"], c["drones"]).min(-1)               # nearest target in each zone
        return self._noisy("tof", dist, np.maximum(c["sigma_min"], c["sigma_rel"] * np.nan_to_num(dist, posinf=0)),
                           c["min_m"], c["range_m"], c["dropout"])

    def _lidar(self):
        c = self.cfg["lidar"]
        dist = self._cast(self._rays["lidar"], c["range_m"], c["drones"])
        return self._noisy("lidar", dist, np.full(dist.shape, c["sigma"]), c["min_m"], c["range_m"], c["dropout"])

    def _depth(self):
        c, rays = self.cfg["depth"], self._rays["depth"]
        z = self._cast(rays, c["range_m"] / rays[..., 0].min(), c["drones"]) * rays[..., 0]     # range -> depth along the axis
        return self._noisy("depth", z, c["sigma_k"] * np.nan_to_num(z, posinf=0) ** 2, c["min_m"], c["range_m"])

    # --- detector ----------------------------------------------------------------------------------------------
    def _detector(self):
        """Per drone, the others in range and FoV (body frame); then one line-of-sight query for all of them; then the
        noisy detections, drawn by `_detect` exactly as the per-candidate Python loop of v0.3.0 drew them."""
        c, d = self.cfg["detector"], self.drones
        R = d.model.R if hasattr(d.model, "R") else quat_to_R(d.q)
        half = math.radians(c["fov_deg"] / 2)
        P, A, L, off = self._crowd()
        everyone, cands = np.arange(len(P)), {}
        for i in range(self.n):
            others = np.flatnonzero(A & (everyone != off + i))
            if len(others) and d.alive[i]:
                rel = (P[others] - d.p[i]) @ R[i]                                     # body frame
                dist = np.linalg.norm(rel, axis=1)
                ang = np.arccos(np.clip(rel[:, 0] / np.maximum(dist, 1e-9), -1, 1))
                cand = np.flatnonzero((dist <= c["range_m"]) & (ang <= half))
                if cand.size:
                    cands[i] = (rel[cand], dist[cand], others[cand])
        if cands:
            js = np.concatenate([x[2] for x in cands.values()])
            vis = self.world.segment_crossings(np.repeat(d.p[list(cands)], [len(x[2]) for x in cands.values()], 0),
                                               P[js]) == 0
            vis = np.split(vis, np.cumsum([len(x[2]) for x in cands.values()])[:-1])
            for (i, (rel, dist, js)), v in zip(list(cands.items()), vis):
                cands[i] = (rel[v], dist[v], js[v])
        s_ang = math.radians(c["sigma_deg"])
        out = []
        for i in range(self.n):
            r, rows = self.rng["detector"][i], []
            if i in cands:
                rel, dist, js = cands[i]
                x, y, z = (rel[:, a].tolist() for a in range(3))
                az = np.array([math.atan2(b, a) for a, b in zip(x, y)])
                el = np.array([math.atan2(h, math.hypot(a, b)) for a, b, h in zip(x, y, z)])
                cls = (L[js] != self.labels[i]).astype(float)
                rows.append(_detect(r, az, el, dist, cls, float(c["p_detect"]), s_ang, float(c["sigma_rel"]),
                                    float(c["p_confuse"])))
            alarms = [(r.uniform(-half, half), r.uniform(-half, half), r.uniform(0, c["range_m"]), float(r.integers(2)),
                       r.uniform(0.3, 0.6)) for _ in range(r.poisson(c["false_rate"]))]
            out.append(np.concatenate(rows + [np.array(alarms, float).reshape(-1, 5)]))
        return out

    # --- state estimation --------------------------------------------------------------------------------------
    def _pose_update(self):
        c, d = self.cfg["pose"], self.drones
        sp, sv, syaw = _POSE[c["profile"]]
        e = self._pose_err
        moved = np.linalg.norm(d.p - self._last_p, axis=1)
        self._last_p = np.array(d.p)
        h = d.p[:, 2] - self.world.ground(d.p[:, :2])
        for i in range(self.n):
            r = self.rng["pose"][i]
            if c["profile"] == "vio":
                self._drift[i] += r.normal(0, c["drift"] * moved[i] + 1e-12, 3)
                e[i, :3] = self._drift[i] + r.normal(0, sp, 3)
            elif c["profile"] == "flow":
                sig = 0.5 if h[i] > c["max_h"] else 0.02 * max(h[i], 0.1)
                e[i, 3:6] = r.normal(0, sig, 3)
                e[i, 3:6][2] = 0.0                                            # height comes from a range finder
                e[i, :3] += e[i, 3:6] / c["hz"]
            else:
                e[i, :3] = r.normal(0, sp, 3) if sp else 0.0
            if c["profile"] != "flow":
                e[i, 3:6] = r.normal(0, sv, 3) if sv else 0.0
            e[i, 6] = r.normal(0, syaw) if syaw else 0.0

    def _pose(self):
        d, e = self.drones, self._pose_err
        c, s = np.cos(e[:, 6] / 2), np.sin(e[:, 6] / 2)        # yaw error applied in the world frame
        q = np.asarray(d.q)
        q = np.stack([c * q[:, 0] - s * q[:, 3], c * q[:, 1] - s * q[:, 2], c * q[:, 2] + s * q[:, 1],
                      c * q[:, 3] + s * q[:, 0]], 1)
        data = np.c_[d.p + e[:, :3] - self.home, d.v + e[:, 3:6], q, d.w]
        if self.cfg["pose"]["profile"] == "flow":
            h = d.p[:, 2] - self.world.ground(d.p[:, :2])
            data[h > self.cfg["pose"]["max_h"], 3:6] = np.nan
        return data

    def _imu(self):
        c, d = self.cfg["imu"], self.drones
        dt = 1 / c["hz"]
        walk = np.repeat([c["acc_bias_walk"], c["gyro_bias_walk"]], 3) * math.sqrt(dt)
        white = np.repeat([c["acc_density"], c["gyro_density"]], 3) * math.sqrt(c["hz"])
        z = np.array([r.standard_normal(12) for r in self.rng["imu"]])
        self._imu_bias += z[:, :6] * walk
        return np.c_[d.f, d.w] + self._imu_bias + z[:, 6:] * white

    def _gnss(self):
        c, d = self.cfg["gnss"], self.drones
        a = math.exp(-1 / (c["hz"] * c["tau_s"]))
        sig = np.array([c["sigma_h"], c["sigma_h"], c["sigma_v"]]) / math.sqrt(2)   # per-axis, horizontal CEP-ish
        out = np.c_[d.p, d.v].astype(float)
        fix = np.ones(self.n, bool)
        for cx, cy, rad in c["denied"]:
            fix &= (d.p[:, 0] - cx) ** 2 + (d.p[:, 1] - cy) ** 2 > rad * rad
        if c["sky"]:
            top = np.c_[d.p[:, :2], np.full(self.n, self.world.bounds[1][2] + 1)]
            fix &= self.world.segment_crossings(d.p, top) == 0
        for i in range(self.n):
            r = self.rng["gnss"][i]
            self._gnss_err[i] = a * self._gnss_err[i] + math.sqrt(1 - a * a) * sig * r.standard_normal(3)
            out[i, :3] += self._gnss_err[i]
            out[i, 3:] += r.normal(0, c["sigma_vel"], 3)
        out[~fix] = np.nan
        return out

    def _baro(self):
        c, d = self.cfg["baro"], self.drones
        out = d.p[:, 2] - self.home[:, 2]
        for i in range(self.n):
            r = self.rng["baro"][i]
            self._baro_drift[i] += r.normal(0, c["drift"] / math.sqrt(c["hz"]))
            out[i] += self._baro_drift[i] + r.normal(0, c["sigma"])
        return out

    def _mag(self):
        c, d = self.cfg["mag"], self.drones
        q = np.asarray(d.q)
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return np.array([yaw[i] + self.rng["mag"][i].normal(0, math.radians(c["sigma_deg"])) for i in range(self.n)])

    def _battery(self):
        return np.clip(1 - np.asarray(self.drones.energy) / self.capacity, 0, 1)

    def _prior_map(self, i):
        """2.5D height map (georeferenced, bounds origin) without thin objects, with noise."""
        c, w = self.cfg["prior_map"], self._survey
        lo, hi = w.bounds
        xs = np.arange(lo[0] + c["res_m"] / 2, hi[0], c["res_m"])
        ys = np.arange(lo[1] + c["res_m"] / 2, hi[1], c["res_m"])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        o = np.c_[X.ravel(), Y.ravel(), np.full(X.size, hi[2] + 1)]
        dist, _, _ = w.raycast(o, np.tile([0.0, 0.0, -1.0], (X.size, 1)), hi[2] + 2)
        height = np.where(np.isfinite(dist), o[:, 2] - dist, w.ground(o[:, :2]))
        return (height + self.rng["prior_map"][i].normal(0, c["sigma"], X.size)).reshape(X.shape)


@njit(cache=True)
def _detect(r, az, el, rng, cls, p_detect, s_ang, s_rel, p_confuse):
    """Detections of the visible candidates (rows of `Reading.data`). Draws from r in the order of v0.3.0: detected?,
    azimuth, elevation, confusion, range; numba's Generator draws exactly what numpy's draws from the same state."""
    out = np.empty((len(az), 5))
    m = 0
    for k in range(len(az)):
        if r.random() > p_detect:
            continue
        out[m, 0] = az[k] + r.normal(0.0, s_ang)
        out[m, 1] = el[k] + r.normal(0.0, s_ang)
        out[m, 3] = cls[k]
        if r.random() < p_confuse:
            out[m, 3] = 1.0 - cls[k]
        out[m, 2] = rng[k] * (1 + r.normal(0.0, s_rel))
        out[m, 4] = p_detect
        m += 1
    return out[:m]
