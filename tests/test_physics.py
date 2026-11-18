import dataclasses

import numpy as np
import pytest

from c2lab.brain import Velocity
from c2lab.physics import PROFILES, Drones
from c2lab.physics.profiles import G
from c2lab.physics.autopilot import Autopilot
from c2lab.physics.quad import Quad

BOX_C, BOX_H = np.array([20.0, 0.0, 5.0]), np.array([1.0, 5.0, 5.0])


class FakeWorld:
    """Flat ground at z = 0 and one axis-aligned box (WorldQuery subset)."""
    bounds = ((-60.0, -60.0, 0.0), (60.0, 60.0, 60.0))

    def sdf(self, p, terrain=True, max_dist=np.inf):
        q = np.abs(p - BOX_C) - BOX_H
        d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        if terrain:
            d = np.minimum(d, p[:, 2])
        return np.minimum(d, max_dist)

    def ground(self, xy):
        return np.zeros(len(xy))


def make(profile="iris", n=1, p0=((0.0, 0.0, 10.0),), landed=False, seed=0, **kw):
    d = Drones(profile, n, np.random.default_rng(seed), FakeWorld(), **kw)
    d.reset(np.array(p0, float), np.zeros(n), landed=landed)
    return d


def run(d, seconds, t0=0, record=None):
    events, ticks = [], round(seconds / d.dt)
    for k in range(t0, t0 + ticks):
        events += d.step(k)
        if record is not None:
            record.append((d.p.copy(), d.v.copy(), d.model.R.copy() if d.mode == "quad" else None))
    return events, t0 + ticks


def test_free_fall():
    prof = dataclasses.replace(PROFILES["iris"], c_xy=0.0, c_z=0.0)
    m, ap = Quad(prof, 1, np.random.default_rng(0), (0, 0, 0), 0.0, 2.0, 1 / 1200), Autopilot(prof, 1, 1 / 1200)
    ap.kpj[:] = ap.kij[:] = 0.0                       # zero wrench -> motors stay stopped
    m.bind(ap, np.full(3, -1e9), np.full(3, 1e9))
    m.reset(np.array([[0.0, 0.0, 100.0]]), np.zeros(1), 0.0)
    for _ in range(1200):
        m.tick(None, False)
    assert abs(m.v[0, 2] + G * 1.0) < 1e-9
    assert np.abs(m.f).max() < 1e-9


@pytest.mark.parametrize("profile", ["iris", "cf2"])
def test_hover(profile):
    d = make(profile)
    d.set_setpoint(0, None)
    ev, _ = run(d, 30)
    assert ev == []
    assert np.linalg.norm(d.p[0] - (0, 0, 10)) < 0.01
    assert 0 <= d.omega.min() and d.omega.max() <= PROFILES[profile].omega_max


@pytest.mark.parametrize("profile,vt", [("iris", 5.0), ("cf2", 1.0)])
def test_velocity_step(profile, vt):
    d, rec = make(profile, p0=((-30.0, 30.0, 10.0),)), []
    d.set_setpoint(0, Velocity((vt, 0.0, 0.0)))
    run(d, 6, record=rec)
    vx = np.array([v[0, 0] for _, v, _ in rec])
    t = np.arange(1, len(vx) + 1) * d.dt
    tilt = np.array([np.arccos(np.clip(R[0, 2, 2], -1, 1)) for _, _, R in rec])
    assert vx.max() <= 1.1 * vt
    assert t[np.argmax(vx >= 0.9 * vt)] <= 1.5
    assert abs(vx[round(5 / d.dt) - 1] - vt) <= 0.05
    assert tilt.max() <= PROFILES[profile].tilt_max


def test_wind_hold():
    d, rec = make(wind=(5.0, 0.0, 0.0), gust_sigma=1.0, gust_tau=2.0), []
    d.set_setpoint(0, None)
    run(d, 60, record=rec)
    err = np.array([np.linalg.norm(p[0] - (0, 0, 10)) for p, _, _ in rec])
    assert np.sqrt((err ** 2).mean()) <= 0.5


def _steps(dt):
    d = make(dt=dt)
    t = 0
    for v in [(3, 0, 0), (0, 2, 1), (-2, -2, 0)]:
        d.set_setpoint(0, Velocity(v))
        _, t = run(d, 10 / 3, t0=t)
    return d.p[0].copy()


def test_dt_convergence():
    assert np.linalg.norm(_steps(1 / 1200) - _steps(1 / 2400)) < 0.01


def test_obstacle_event_on_exact_tick():
    d, w = make(p0=((10.0, 0.0, 5.0),)), FakeWorld()
    d.set_setpoint(0, Velocity((2.0, 0.0, 0.0)))
    first, events = None, []
    for k in range(round(10 / d.dt)):
        events += d.step(k)
        if first is None and w.sdf(d.p, terrain=False)[0] < d.prof.radius:
            first = k
        if events:
            break
    assert events == [(first, 0, "crash_obstacle")] and not d.alive[0]


def test_drone_and_geofence_events():
    d = make(n=2, p0=((-10.0, 30.0, 10.0), (-4.0, 30.0, 10.0)))
    d.set_setpoint(0, Velocity((2.0, 0.0, 0.0)))
    d.set_setpoint(1, Velocity((-2.0, 0.0, 0.0)))
    ev, _ = run(d, 5)
    assert sorted(e[2] for e in ev) == ["crash_drone", "crash_drone"]
    d = make(p0=((55.0, 0.0, 10.0),))
    d.set_setpoint(0, Velocity((5.0, 0.0, 0.0)))
    ev, _ = run(d, 5)
    assert [e[2] for e in ev] == ["geofence"]


@pytest.mark.parametrize("vz,kind", [(-0.5, "landed"), (-1.5, "crash_terrain")])
def test_landing(vz, kind):
    d = make(p0=((0.0, 0.0, 3.0),))
    d.set_setpoint(0, Velocity((0.0, 0.0, vz)))
    ev, _ = run(d, 12)
    assert [e[2] for e in ev] == [kind]


def test_takeoff_from_ground():
    d = make(landed=True)
    d.set_setpoint(0, Velocity((0.0, 0.0, 1.0)))
    ev, _ = run(d, 3)
    assert [e[2] for e in ev] == ["takeoff"] and d.p[0, 2] > 1.5


def test_energy():
    d = make()
    d.set_setpoint(0, Velocity((2.0, 1.0, 0.5)))
    e = 0.0
    for k in range(2400):
        d.step(k)
        e += d.prof.power(d.prof.kf * d.omega ** 2)[0] * d.dt
    assert abs(d.energy[0] - e) <= 1e-6 * e
    assert abs(PROFILES["cf2"].hover_power - 7) <= 0.15 * 7
    assert abs(PROFILES["iris"].hover_power - 170) <= 0.15 * 170


def test_deterministic():
    runs = []
    for _ in range(2):
        d = make(n=3, p0=((0, 0, 10), (5, 5, 10), (-5, 5, 10)), wind=(2, 1, 0), gust_sigma=1.0, seed=7)
        for i in range(3):
            d.set_setpoint(i, Velocity((1.0, -1.0, 0.2)))
        run(d, 10)
        runs.append(np.concatenate([d.p, d.v, d.q, d.w, d.omega], 1))
    assert np.array_equal(*runs)


def test_point_mass_limits():
    d, rec = make(mode="pointmass", p0=((-55.0, 30.0, 10.0),)), []
    d.set_setpoint(0, Velocity((20.0, 0.0, 0.0)))
    run(d, 8, record=rec)
    v = np.array([v[0] for _, v, _ in rec])
    prof = d.prof
    assert np.linalg.norm(v[:, :2], axis=1).max() <= prof.vmax_xy + 1e-9
    a = np.linalg.norm(np.diff(v[:, :2], axis=0), axis=1) / d.dt
    assert a.max() <= G * np.tan(prof.tilt_max) + 1e-6
    assert abs(v[-1, 0] - prof.vmax_xy) < 1e-6


def test_real_world_contacts():
    from c2lab.world import World
    w = World.from_dict({"biome": "t", "seed": 0, "density": 0.0, "bounds": [[0, 0, 0], [60, 40, 20]],
                         "anchors": {"A": [5, 20, 0], "B": [55, 20, 0]}, "terrain": {"heights": [[0, 0], [0, 0]]},
                         "primitives": [{"type": "box", "tag": "wall", "c": [30, 20, 5], "h": [0.3, 10, 5],
                                         "q": [1, 0, 0, 0]}]})
    d = Drones("iris", 2, np.random.default_rng(0), w)
    d.reset(np.array([[5.0, 20.0, 0.0], [5.0, 5.0, 0.0]]), np.zeros(2), landed=True)
    d.set_setpoint(0, Velocity((0.0, 0.0, 1.0)))
    d.set_setpoint(1, Velocity((0.0, 0.0, 1.0)))
    ev, t = run(d, 4)
    d.set_setpoint(0, Velocity((3.0, 0.0, 0.0)))     # into the wall
    d.set_setpoint(1, Velocity((3.0, 0.0, 0.0)))     # passes beside it (wall spans y 10..30)
    ev2, _ = run(d, 15, t0=t)
    assert sorted(e[2] for e in ev) == ["takeoff", "takeoff"]
    assert [(e[1], e[2]) for e in ev2] == [(0, "crash_obstacle")]
    assert d.p[0, 0] < 30 - 0.3 and d.alive[1] and d.p[1, 0] > 40
