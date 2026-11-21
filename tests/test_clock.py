"""RunClock only paces the loop: the run hash is the same in every mode, with pauses and steps (plan F8.4)."""
import threading
import time

import pytest

from c2lab import clock as clock_module
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


class FakeTime:
    def __init__(self):
        self.now, self.slept = 0.0, 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, s):
        self.now += s
        self.slept += s


def test_realtime_never_idles_when_behind(monkeypatch):
    """A loop slower than the factor runs flat out: heavy sensor ticks never leave the cheap ones sleeping."""
    ft = FakeTime()
    monkeypatch.setattr(clock_module, "time", ft)
    clock = RunClock("realtime", 10.0)                    # 83 us per tick
    for tick in range(12000):
        clock.wait(tick)
        ft.now += 40e-6 if tick % 80 else 8e-3            # 140 us per tick on average
    assert ft.slept == 0.0


def test_bad_mode_and_factor():
    with pytest.raises(ValueError):
        RunClock("slow")
    with pytest.raises(ValueError):
        RunClock().realtime(20)
