import numpy as np

from c2lab.brain import Action, Velocity
from c2lab.scenario import parse
from c2lab.sim import run

BASE = """
[scenario]
name = "t"
seed = 3
duration_s = {dur}
mission = "a_to_b"
[world]
biome = "mixto"
density = 0.1
[radio]
profile = "ideal"
[[team]]
name = "a"
count = {n}
brain = "{brain}"
launch_s = {launch}
sensors.pose = {{ profile = "{pose}" }}
sensors.tof = {{ arrays = ["front"] }}
"""


def scenario(brain, n=3, dur=15, pose="ekf", launch=0.0):
    return parse(BASE.format(brain=brain, n=n, dur=dur, pose=pose, launch=launch))


class Probe:
    """Records what it is given; climbs, then hovers."""
    seen = []

    def reset(self, spec):
        Probe.seen.append(("spec", sorted(spec.sensors), spec.mission["goal_local"]))

    def step(self, obs):
        Probe.seen.append(("obs", sorted(obs.readings), obs.t))
        return Action(Velocity((0.0, 0.0, 1.0)) if obs.t < 3 else None)


class Crash:
    def reset(self, spec):
        pass

    def step(self, obs):
        if obs.t > 2:
            raise RuntimeError("boom")
        return Action(Velocity((0.0, 0.0, 1.0)))


def test_brain_is_one_line_and_sees_only_its_sensors():
    Probe.seen.clear()
    r = run(scenario("test_sim:Probe", n=2, dur=5))
    specs = [s for s in Probe.seen if s[0] == "spec"]
    obs = [s for s in Probe.seen if s[0] == "obs"]
    assert len(specs) == 2 and all(s[1] == ["pose", "tof"] for s in specs)
    assert {tuple(s[1]) for s in obs} <= {("pose", "tof"), ("pose",), ("tof",), ()}
    assert abs(obs[1][2] - obs[0][2]) in (0.0, 0.1)            # 10 Hz brain
    assert r["metrics"]["crashes"] == 0
    other = run(scenario("brains.random_walk:RandomWalk", n=2, dur=5))
    assert other["metrics"]["agents"] == 2


def test_brain_fault_goes_to_failsafe():
    r = run(scenario("test_sim:Crash", n=2, dur=12))
    kinds = [e[2] for e in r["events"]]
    assert r["metrics"]["brain_errors"] == 2 and kinds.count("brain_error") == 2 and kinds.count("landed") == 2


def test_same_seed_same_run():
    a = run(scenario("brains.potential_field:PotentialField", dur=8), record=True)
    b = run(scenario("brains.potential_field:PotentialField", dur=8), record=True)
    assert a["hash"] == b["hash"]


def test_oracle_validates_the_path_metric():
    r = run(scenario("brains.privileged.oracle:Oracle", n=2, dur=60, pose="ideal", launch=4.0))
    m = r["metrics"]
    assert m["privileged"] and m["arrival_rate"] == 1.0
    assert abs(m["flown_mean"] / m["oracle_length"] - 1) < 0.05
    assert np.isfinite(m["time_mean"])
