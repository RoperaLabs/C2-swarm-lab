"""RunClock only paces the loop: the run hash is the same in every mode, with pauses and steps (plan F8.4)."""
import threading
import time

import pytest

from c2lab.clock import RunClock
from c2lab.scenario import parse
from c2lab.sim import run

SC = parse("""
[scenario]
name = "clock"
seed = 5
duration_s = 0.5
mission = "a_to_b"
[world]
biome = "mixto"
density = 0.1
[radio]
profile = "ideal"
[[team]]
name = "a"
count = 3
brain = "brains.boids:Boids"
sensors.pose = { profile = "ekf" }
sensors.tof = { arrays = ["front"] }
""")


@pytest.fixture(scope="module")
def fast_hash():
    return run(SC, record=True)["hash"]


@pytest.mark.parametrize("factor", [0.25, 1.0, 4.0])
def test_realtime_keeps_the_hash_and_the_pace(fast_hash, factor):
    clock = RunClock("realtime", factor)
    assert run(SC, record=True, clock=clock)["hash"] == fast_hash
    st = clock.stats()
    assert st["ticks"] == 600 and st["sim_s"] == 0.5
    assert st["wall_s"] >= 0.95 * (599 / 1200) / factor          # never ahead of the requested pace


def until(cond, timeout=20.0):
    end = time.perf_counter() + timeout
    while not cond():
        assert time.perf_counter() < end, "timed out"
        time.sleep(0.005)


def test_pause_step_and_resume_keep_the_hash(fast_hash):
    clock, out = RunClock("paused"), {}
    th = threading.Thread(target=lambda: out.update(run(SC, record=True, clock=clock)))
    th.start()
    time.sleep(0.2)
    assert clock.stats()["ticks"] == 0 and clock.stats()["real_factor"] == 0.0     # paused before the first tick
    clock.step(5)
    until(lambda: clock.stats()["ticks"] == 5)
    time.sleep(0.05)
    assert clock.stats()["ticks"] == 5                                              # and paused again
    clock.realtime(4.0)
    time.sleep(0.05)
    clock.pause()
    time.sleep(0.05)                                                                # the tick in flight ends
    t1 = clock.stats()["ticks"]
    clock.step(100)
    clock.step(20)
    until(lambda: clock.stats()["ticks"] == t1 + 120)
    time.sleep(0.05)
    assert clock.stats()["ticks"] == t1 + 120
    clock.fast()
    th.join(30)
    assert out["hash"] == fast_hash and clock.stats()["ticks"] == 600


def test_bad_mode_and_factor():
    with pytest.raises(ValueError):
        RunClock("slow")
    with pytest.raises(ValueError):
        RunClock().realtime(20)
