"""TickOrder: the order of phases in a tick and when an action takes effect (plan F8.5). Both are part of the
determinism contract, because a run's hash depends on them; these tests fail if either changes."""
import pytest

from c2lab import sim
from c2lab.brain import Action, Velocity
from c2lab.missions import AToB
from c2lab.physics import Drones
from c2lab.radio import Radio
from c2lab.scenario import parse
from c2lab.sensors import Sensors

SC = """
[scenario]
name = "order"
seed = 1
duration_s = {dur}
mission = "a_to_b"
rates_hz = {{ brain = 20, radio = 10, record = 30 }}
[world]
biome = "mixto"
density = 0.1
[radio]
profile = "ideal"
{station}
[[team]]
name = "a"
count = 2
brain = "test_tick_order:{brain}"
command = {{ latency_s = {latency} }}
[[team]]
name = "b"
count = 1
brain = "test_tick_order:{brain}"
"""
STATION = '[ground_station]\nbrain = "test_tick_order:Station"'
LOG = []


class Log:
    def reset(self, spec):
        self.g = spec.agent_id

    def step(self, obs):
        LOG.append(("brain", self.g))
        return Action()


class Station:
    def reset(self, spec):
        pass

    def step(self, obs):
        LOG.append(("station",))
        return Action()


class Climb:
    def reset(self, spec):
        pass

    def step(self, obs):
        return Action(Velocity((0.0, 0.0, 1.0)))


def spy(monkeypatch, cls, name, label):
    fn = getattr(cls, name)

    def wrap(self, *a, **k):
        LOG.append(label(self))
        return fn(self, *a, **k)
    monkeypatch.setattr(cls, name, wrap)


def expected(ticks, kb=60, kr=120, krec=40):
    """The normative order: brains, station and mission on brain ticks -> radio slot -> physics then sensors, team by
    team -> record -> end check on brain ticks. Teams have 2 and 1 drones (agents 0, 1 and 2)."""
    out = []
    for k in range(ticks):
        if k % kb == 0:
            out += [("brain", 0), ("brain", 1), ("brain", 2), ("station",), ("mission",)]
        if k % kr == 0:
            out.append(("radio",))
        out += [("physics", 2), ("sensors", 2), ("physics", 1), ("sensors", 1)]
        if k % krec == 0:
            out.append(("record",))
        if k % kb == 0:
            out.append(("end",))
    return out


def test_phase_order(monkeypatch):
    spy(monkeypatch, Drones, "step", lambda d: ("physics", d.n))
    spy(monkeypatch, Sensors, "step", lambda s: ("sensors", s.n))
    spy(monkeypatch, Radio, "slot", lambda r: ("radio",))
    spy(monkeypatch, AToB, "update", lambda m: ("mission",))
    spy(monkeypatch, AToB, "finished", lambda m: ("end",))
    monkeypatch.setattr(sim, "LIVE_S", -1.0)                 # live() then runs right after every recorded frame
    LOG.clear()
    sim.run(parse(SC.format(dur=0.25, station=STATION, brain="Log", latency=0.0)), record=True,
            live=lambda part: LOG.append(("record",)))
    assert LOG == expected(300)


@pytest.mark.parametrize("latency_s,first", [(0.0, 0), (0.05, 60), (0.01, 12)])
def test_action_takes_effect_after_latency(monkeypatch, latency_s, first):
    """The setpoint computed at the step of tick 0 drives the physics from tick round(latency_s * 1200), on brain
    ticks and between them."""
    seen = []
    step = Drones.step

    def wrap(self, tick, err=None):
        if self.n == 2 and self.ap.mode_vel[0] and not seen:
            seen.append(tick)
        return step(self, tick, err)
    monkeypatch.setattr(Drones, "step", wrap)
    sim.run(parse(SC.format(dur=0.1, station="", brain="Climb", latency=latency_s)))
    assert seen == [first]
