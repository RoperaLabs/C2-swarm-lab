import numpy as np

from c2lab.missions import CAPTURE_R, Pursuit
from c2lab.scenario import load
from c2lab.sim import run
from c2lab.world import generate


def test_capture_needs_proximity_for_the_capture_time():
    sc = load("scenarios/pursuit_mixto.toml")
    w = generate("mixto", [60, 40, 20], 0.1, 0)
    m = Pursuit(sc, w, [0, 0, 1], ["pursuer", "pursuer", "evader"])
    m.start(np.zeros((3, 3)), [False] * 3)
    alive = np.ones(3, bool)
    near = np.array([[0, 0, 2.0], [30, 0, 2.0], [CAPTURE_R - 0.5, 0, 2.0]])
    far = near.copy()
    far[2, 0] = CAPTURE_R + 0.5
    for k in range(4):                                   # 0.4 s close, then away: no capture
        assert m.update(k, k / 10, near, alive, None) == []
    m.update(4, 0.4, far, alive, None)
    events = [e for k in range(5, 10) for e in m.update(k, k / 10, near, alive, None)]
    assert events == [(2, "captured")] and m.finished() and m.metrics()["capture_rate"] == 1.0


def test_coverage_scores_and_the_station_assigns_sectors():
    sc = load("scenarios/coverage_mixto.toml")
    sc["scenario"]["duration_s"] = 20
    sc["team"][0]["count"] = 3
    r = run(sc)
    m = r["metrics"]
    assert m["surface_voxels"] > 0 and 0 < m["coverage"] <= 1
    assert m["bytes_sent"] >= 3 * 8 and m["bytes_delivered"] >= 3 * 8      # every agent got its 8-byte sector
    assert m["crashes"] == 0


def test_a_to_b_reports_regret():
    sc = load("scenarios/ab_mixto.toml")
    sc["scenario"]["duration_s"] = 5
    m = run(sc)["metrics"]
    assert set(m) >= {"arrival_rate", "regret_mean", "oracle_length", "flown_mean"} and m["oracle_length"] > 40
