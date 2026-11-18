import math

import numpy as np
import pytest

from c2lab.brain import Reading
from c2lab.physics import Drones
from c2lab.radio import DIL, Radio
from c2lab.sensors import Sensors
from c2lab.world import World

WALL = {"type": "box", "tag": "wall", "c": [13.3, 20.0, 5.0], "h": [0.3, 10.0, 5.0], "q": [1, 0, 0, 0]}
CABLE = {"type": "capsule", "tag": "cable", "a": [30.0, 5.0, 6.0], "b": [50.0, 5.0, 6.0], "r": 0.02}


def world(prims=(WALL, CABLE)):
    return World.from_dict({"biome": "t", "seed": 0, "density": 0.0, "bounds": [[0, 0, 0], [60, 40, 20]],
                            "anchors": {"A": [2, 2, 0], "B": [58, 38, 0]}, "terrain": {"heights": [[0, 0], [0, 0]]},
                            "primitives": [dict(p) for p in prims]})


def rig(cfg, pos=((10.0, 20.0, 5.0),), yaw=None, landed=False, w=None, labels=None, seed=0):
    w = w or world()
    n = len(pos)
    d = Drones("iris", n, np.random.default_rng(0), w)
    d.reset(np.array(pos, float), np.zeros(n) if yaw is None else np.array(yaw, float), landed=landed)
    s = Sensors(cfg, n, seed, list(range(n)), w, d, np.zeros(n) if labels is None else labels, d.p.copy())
    return d, s


def run(d, s, ticks, fly=False):
    for k in range(ticks):
        if fly:
            d.step(k, s.err)
        s.step(k)


def test_toggle_rates_and_stream_independence():
    d, s = rig({"imu": {}, "baro": {}}, landed=True)
    d2, s2 = rig({"imu": {}, "baro": {}, "gnss": {}}, landed=True)
    run(d, s, 1200)
    run(d2, s2, 1200)
    a, b = s.pop(0), s2.pop(0)
    assert set(a) == {"imu", "baro"} and set(b) == {"imu", "baro", "gnss"}
    assert len(a["imu"]) == 400 and len(a["baro"]) == 50 and len(b["gnss"]) == 5
    assert [r.t for r in a["imu"][:3]] == [0.0, 3 / 1200, 6 / 1200]
    assert all(np.array_equal(x.data, y.data) for x, y in zip(a["imu"], b["imu"]))
    assert s.pop(0) == {}


def test_readings_are_plain_and_read_only():
    d, s = rig({k: {} for k in ("tof", "lidar", "depth", "detector", "pose", "imu", "gnss", "baro", "mag",
                                "prior_map", "battery")}, pos=((10.0, 20.0, 5.0), (20, 20, 5)))
    run(d, s, 240, fly=True)
    got = s.pop(0)
    assert len(got) == 11
    for name, readings in got.items():
        for r in readings:
            assert type(r) is Reading and isinstance(r.t, float) and isinstance(r.data, np.ndarray)
            assert not r.data.flags.writeable and r.data.dtype == np.float64, name


@pytest.mark.parametrize("name,cfg,sigma", [
    ("imu", {"acc_bias_walk": 0.0, "gyro_bias_walk": 0.0}, 1.7e-3 * math.sqrt(400)),
    ("baro", {"drift": 0.0}, 0.15),
    ("gnss", {"tau_s": 1e-4, "sigma_h": 1.5, "hz": 400}, 1.5 / math.sqrt(2)),
])
def test_noise_statistics(name, cfg, sigma):
    d, s = rig({name: cfg}, landed=True)
    hz = {"imu": 400, "baro": 50, "gnss": 400}[name]
    run(d, s, round(10_000 / hz * 1200))
    x = np.array([r.data for r in s.pop(0)[name]])
    x = x[:, 0] if x.ndim > 1 else x
    assert abs(x.std() / sigma - 1) <= 0.05


def test_gnss_error_autocorrelation():
    d, s = rig({"gnss": {"tau_s": 0.05, "hz": 400}}, landed=True)
    run(d, s, 60_000 * 3)
    e = np.array([r.data[0] for r in s.pop(0)["gnss"]]) - d.p[0, 0]
    rho = np.corrcoef(e[:-20], e[20:])[0, 1]             # lag 20 samples = 0.05 s = tau
    assert abs(rho - math.exp(-1)) <= 0.1 * math.exp(-1)


def test_tof_against_a_wall_and_out_of_range():
    d, s = rig({"tof": {"arrays": ["front", "back"], "dropout": 0.0}})
    run(d, s, 1)
    tof = s.pop(0)["tof"][0].data
    front, back = tof[0], tof[1]
    assert abs(np.nanmedian(front[3:5, 3:5]) - 3.0) < 3 * 0.09
    assert np.all(front[np.isfinite(front)] >= 3.0 - 4 * 0.09)
    assert np.isnan(back).all()                            # nothing within 4 m behind


def test_depth_on_a_plane():
    d, s = rig({"depth": {"range_m": 10.0}})
    run(d, s, 1)
    z = s.pop(0)["depth"][0].data
    good = z[np.isfinite(z)]
    assert good.size > 0.9 * z.size
    assert abs(good.mean() - 3.0) < 0.01 and abs(good.std() - 0.01 * 9) < 0.02


def test_detector_line_of_sight_and_fov():
    cfg = {"detector": {"false_rate": 0.0, "p_detect": 1.0, "p_confuse": 0.0, "range_m": 50.0}}
    pos = ((10.0, 20.0, 5.0), (12.0, 21.0, 5.0), (20.0, 20.0, 5.0), (5.0, 20.0, 5.0))  # ahead, behind wall, behind
    d, s = rig(cfg, pos=pos, labels=np.array([0, 0, 1, 0]))
    run(d, s, 1)
    rows = s.pop(0)["detector"][0].data
    assert len(rows) == 1                                  # only drone 1: 2 is hidden, 3 is out of the 90 deg FoV
    az, el, rng_m, cls, _ = rows[0]
    assert abs(rng_m - math.hypot(2, 1)) < 0.3 * math.hypot(2, 1) and cls == 0.0
    assert abs(az - math.atan2(1, 2)) < 0.1
    d, s = rig(cfg, pos=((10.0, 20.0, 5.0), (12.0, 20.5, 5.0)), labels=np.array([0, 1]))
    run(d, s, 1)
    assert s.pop(0)["detector"][0].data[0, 3] == 1.0      # other team -> class 1


def test_pose_profiles():
    d, s = rig({"pose": {"profile": "ideal"}})
    run(d, s, 1)
    assert s.err is None and np.allclose(s.pop(0)["pose"][0].data[:3], 0.0)
    errs = []
    for prof in ("ekf", "vio"):
        d, s = rig({"pose": {"profile": prof, "drift": 0.05}})
        from c2lab.brain import Velocity
        d.set_setpoint(0, Velocity((0.0, 3.0, 0.0)))
        run(d, s, 1200 * 5, fly=True)
        errs.append(np.linalg.norm(s.err.dp[0]))
    assert errs[0] < 0.3 and errs[1] > errs[0]             # vio drift grows with distance flown


def test_prior_map_omits_thin_objects():
    d, s = rig({"prior_map": {"sigma": 0.0}})
    run(d, s, 1)
    m = s.pop(0)["prior_map"][0].data                      # 2 m cells, origin at bounds
    assert abs(m[6, 10] - 10.0) < 1e-6                     # wall top (x 13..13.6, y 20)
    assert abs(m[20, 2]) < 1e-6                            # under the cable: ground


def test_battery_and_mag():
    d, s = rig({"battery": {}, "mag": {"sigma_deg": 0.0}}, yaw=[0.7])
    run(d, s, 1200, fly=True)
    got = s.pop(0)
    assert abs(float(got["mag"][-1].data) - 0.7) < 1e-3
    assert 0.99 < float(got["battery"][-1].data) < 1.0


# --- radio ------------------------------------------------------------------------------------------------------

def radio(cfg, n=2, w=None):
    return Radio(cfg, n, np.random.default_rng(0), w or world(()), 0.1)


def deliveries(r, p, trials=2000, payload=b"x" * 10):
    got = 0
    for _ in range(trials):
        r.send(0, [payload])
        got += len(r.slot(p, np.ones(len(p), bool))[1])
    return got / trials


def test_radio_ideal_disco_and_latency():
    p = np.array([[0, 0, 5.0], [80, 0, 5.0]])
    r = radio({"profile": "ideal"})
    r.send(0, [b"hello"])
    assert r.slot(p, np.ones(2, bool)) == [[], []]         # one slot of latency
    inbox = r.slot(p, np.ones(2, bool))
    assert inbox[1][0].data == b"hello" and inbox[1][0].src == 0 and inbox[0] == []
    assert deliveries(radio({"profile": "disco", "range_m": 50.0}), p) == 0.0
    rate = deliveries(radio({"profile": "disco", "range_m": 100.0, "loss": 0.3}), p, 4000)
    assert abs(rate - 0.7) < 3 * math.sqrt(0.21 / 4000)


def test_radio_limits_and_ranging():
    r = radio({"profile": "ideal", "packet_bytes": 8, "max_packets": 2, "ranging": True})
    r.send(0, [b"123456789", b"ok", (1, b"to1"), b"third"])
    assert r.dropped[0] == 9 + 5 and r.sent[0] == 5
    p = np.array([[0, 0, 5.0], [30, 40, 5.0]])
    r.slot(p, np.ones(2, bool))
    got = r.slot(p, np.ones(2, bool))[1]
    assert [x.data for x in got] == [b"ok", b"to1"] and abs(got[0].range_m - 50) < 0.5


def test_dil_v02_matches_link_budget_and_line_of_sight():
    p = np.array([[0, 0, 5.0], [300, 0, 5.0]])
    r = radio({"profile": "dil_v02", "dil": {"shadow_sigma_db": 0.0}})
    rx = DIL["tx_dbm"] - DIL["pl_1m_db"] - 10 * DIL["path_loss_exp"] * math.log10(300)
    expect = 1 / (1 + math.exp(-(rx - DIL["noise_dbm"] - DIL["sinr50_db"]) / DIL["sinr_width_db"]))
    rate = deliveries(r, p, 4000)
    assert abs(rate - expect) < 4 * math.sqrt(expect * (1 - expect) / 4000) + 1e-3
    blocked = radio({"profile": "dil_v02", "dil": {"shadow_sigma_db": 0.0}}, w=world((WALL,)))
    pw = np.array([[10.0, 20.0, 5.0], [16.0, 20.0, 5.0]])
    free = radio({"profile": "dil_v02", "dil": {"shadow_sigma_db": 0.0}}, w=world((WALL,)))
    pf = np.array([[10.0, 35.0, 5.0], [16.0, 35.0, 5.0]])
    assert free._links(pf, np.ones(2, bool), np.zeros(2))[1][1, 0] - \
        blocked._links(pw, np.ones(2, bool), np.zeros(2))[1][1, 0] == pytest.approx(DIL["obstacle_db"])
