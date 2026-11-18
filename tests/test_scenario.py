from pathlib import Path

import pytest

from c2lab.scenario import ScenarioError, load, parse

ROOT = Path(__file__).resolve().parent.parent
MIN = """
[scenario]
name = "m"
seed = 1
duration_s = 10
mission = "coverage"
[world]
biome = "mixto"
[[team]]
name = "a"
count = 2
brain = "brains.x:X"
"""


def test_example_loads():
    s = load(ROOT / "scenarios" / "ejemplo.toml")
    assert s["scenario"]["rates_hz"]["physics"] == 1200
    assert s["team"][0]["role"] == "nav"
    assert "gnss" not in s["team"][0]["sensors"] and "tof" in s["team"][0]["sensors"]


def test_minimal_defaults():
    s = parse(MIN)
    assert s["scenario"]["rates_hz"] == {"physics": 1200, "brain": 10, "radio": 10, "record": 30}
    assert s["world"] == {"biome": "mixto", "size_m": [60, 40, 20], "density": 0.2}
    assert s["radio"] == {"profile": "dil_v02", "packet_bytes": 64, "ranging": False}
    assert s["team"][0] == {"name": "a", "count": 2, "brain": "brains.x:X", "brain_params": {}, "role": "nav",
                            "drone": "iris", "memory_bytes": None, "sensors": {}}
    assert "ground_station" not in s


def test_sensor_true_and_ground_station():
    s = parse(MIN.replace('brain = "brains.x:X"', 'brain = "brains.x:X"\nsensors = { imu = true, mag = false }')
              + '[ground_station]\nbrain = "brains.g:G"\n')
    assert s["team"][0]["sensors"] == {"imu": {}}
    assert s["ground_station"] == {"brain": "brains.g:G", "brain_params": {}}


def _case(old, new, path, extra=""):
    return pytest.param(MIN.replace(old, new), path, extra, id=path)


@pytest.mark.parametrize("text,path,extra", [
    _case("seed = 1\n", "", "scenario.seed"),
    _case('biome = "mixto"', 'biom = "mixto"', "world.biom"),
    _case('"coverage"', '"foo"', "scenario.mission"),
    _case('mission = "coverage"', 'mission = "coverage"\nrates_hz = { brain = 7 }', "scenario.rates_hz.brain", "7"),
    _case('brain = "brains.x:X"', 'brain = "brains.x:X"\nsensors.tof = { hz = 7 }', "team[0].sensors.tof.hz", "7"),
    _case('"brains.x:X"', '"boids"', "team[0].brain"),
    _case('[[team]]', '[[team]]\nname = "a"\ncount = 1\nbrain = "b.c:D"\n[[team]]', "team[1].name"),
    _case('brain = "brains.x:X"', 'brain = "brains.x:X"\nsensors.sonar = {}', "team[0].sensors.sonar"),
    _case("count = 2", "count = 0", "team[0].count"),
    _case("[world]", '[radio]\npacket_bytes = 0\n[world]', "radio.packet_bytes"),
    _case("[world]", "[foo]\n[world]", "foo"),
    _case("duration_s = 10", "duration_s = inf", "scenario.duration_s"),
    _case('mission = "coverage"', 'mission = "coverage"\nrates_hz = { record = inf }', "scenario.rates_hz.record"),
    _case('mission = "coverage"', 'mission = "coverage"\nrates_hz = { brain = 2400 }', "scenario.rates_hz.brain"),
    _case('biome = "mixto"', 'biome = "mixto"\nsize_m = [60, 40, inf]', "world.size_m"),
])
def test_errors(text, path, extra):
    with pytest.raises(ScenarioError, match=path.replace("[", r"\[").replace("]", r"\]")) as e:
        parse(text)
    assert extra in str(e.value)
