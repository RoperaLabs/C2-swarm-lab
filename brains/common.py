"""Helpers shared by the baseline brains. Brains may not import the simulator, so the sensor geometry they need
(zone directions of a mounted ToF array) is rebuilt here from the public parameters in AgentSpec.sensors."""
import math

import numpy as np

MOUNTS = {"front": (0, 0), "left": (90, 0), "back": (180, 0), "right": (-90, 0), "up": (0, 90), "down": (0, -90)}


def rot(q):
    """Quaternion (w, x, y, z) -> rotation matrix (body to local)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def zone_dirs(tof):
    """Body-frame unit vectors of every ToF zone centre, shape (arrays, zones, zones, 3)."""
    z, t = tof["zones"], math.tan(math.radians(tof["fov_deg"] / 2))
    u = ((np.arange(z) + 0.5) / z * 2 - 1) * t
    U, V = np.meshgrid(u, -u)
    d = np.stack([np.ones_like(U), -U, V], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    out = []
    for m in tof["arrays"]:
        yaw, pitch = (math.radians(a) for a in MOUNTS[m])
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        out.append(d @ R.T)
    return np.array(out)


def pose(obs):
    """Latest pose reading as (p, v, R) in the local frame, or None."""
    r = obs.readings.get("pose")
    if not r:
        return None
    x = r[-1].data
    return x[:3], x[3:6], rot(x[6:10])


def obstacles(obs, dirs, R, rng_max):
    """Local-frame vectors to the obstacles seen by the latest ToF reading, shape (k, 3)."""
    r = obs.readings.get("tof")
    if not r or dirs is None:
        return np.empty((0, 3))
    d = r[-1].data
    ok = np.isfinite(d) & (d < rng_max)
    return (dirs[ok] * d[ok][:, None]) @ R.T


def clip(v, vmax):
    n = np.linalg.norm(v)
    return v if n <= vmax else v * (vmax / n)


TAKEOFF = 1.5   # m above the take-off point before a baseline starts its manoeuvre


def climbing(p):
    """Action velocity for the take-off phase, or None once airborne enough."""
    return (0.0, 0.0, 1.0) if p[2] < TAKEOFF else None


def detections(obs, R, min_conf=0.7):
    """Latest confident detector rows as local-frame vectors (k, 3), ranges (k,) and classes (k,): 0 = own team,
    1 = other. Rows under `min_conf` (the detector's false alarms report low confidence) are dropped."""
    r = obs.readings.get("detector")
    rows = r[-1].data[r[-1].data[:, 4] >= min_conf] if r and len(r[-1].data) else np.empty((0, 5))
    if not len(rows):
        return np.empty((0, 3)), np.empty(0), np.empty(0)
    az, el, rng_m, cls = (rows[:, k] for k in range(4))
    rel = (np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)] * rng_m[:, None]) @ R.T
    return rel, rng_m, cls
