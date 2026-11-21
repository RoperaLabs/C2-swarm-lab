"""Reference runs of v0.3.0, blocking in every phase (plan, principle 2). Each pins the run hash and a digest of
everything the brains received and every event. A mismatch is a broken determinism contract, never a value to
update: a change that must alter them goes out as a new major version with a migration note.

The pins hold on the machine that computed them; math libraries round differently on other systems and CPUs, so a
CI runner on other hardware skips them."""
import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from c2lab import sim
from c2lab.scenario import load, parse
from c2lab.sweep import apply

ROOT = Path(__file__).resolve().parent.parent
PINNED_ON = "Windows | Intel64 Family 6 Model 158 Stepping 9, GenuineIntel"

ALL_SENSORS = """
[scenario]
name = "all_sensors"
seed = 11
duration_s = 12
mission = "a_to_b"
[world]
biome = "mixto"
density = 0.15
wind = [2.0, -1.0, 0.0]
gust_sigma = 0.5
[radio]
profile = "disco"
range_m = 25.0
loss = 0.2
ranging = true
[[team]]
name = "a"
count = 3
drone = "cf2"
brain = "brains.potential_field:PotentialField"
sensors.tof = { arrays = ["front", "down"] }
sensors.lidar = { channels = 4, hres_deg = 10.0, range_m = 15.0 }
sensors.depth = { width = 16, height = 12 }
sensors.detector = { fov_deg = 120 }
sensors.pose = { profile = "flow" }
sensors.imu = true
sensors.gnss = { denied = [[30.0, 20.0, 8.0]], sky = true }
sensors.baro = true
sensors.mag = true
sensors.prior_map = true
sensors.battery = true
[[team]]
name = "b"
count = 3
dynamics = "pointmass"
brain = "brains.boids:Boids"
launch_s = 1.0
sensors.pose = { profile = "vio" }
sensors.detector = { fov_deg = 360, range_m = 20 }
sensors.tof = { arrays = ["front", "left", "right", "back"] }
"""

# name: (scenario, duration override in s, run hash, digest)
REFERENCE = {
    "ab_mixto": ("ab_mixto.toml", 20, "0ac6739608367231750358103e203e02376db7ea19e3950b39e7a9066853829b",
                 "418634caf28f1031dd573c27971f03ff3ac9a1006f935c171f98eaba9443cd5e"),
    "coverage_mixto": ("coverage_mixto.toml", 15, "cb2f661237f434c81ba8fd50406bbdd0b393ec13ff0bf964ed441a33a196d026",
                       "6c48e0d0b465cda45aea1ed3452b99908962d30ef9c992b473480dbdec181d7b"),
    "pursuit_mixto": ("pursuit_mixto.toml", 20, "a14a240bb315fc997e17340e20f079b40620003989535ccd8c07bb967fd00df9",
                      "6c243dcfb83daa1bf7e45a2c6839213c3c711afc3083b79c753d04007027a6a2"),
    "ejemplo": ("ejemplo.toml", 15, "fdf73f74f07904cb94e0089f46054358f63daab9aa30ae0f18d4781ca8cc0761",
                "93bc74166c8d6b8cae672e78f7a7b57c8323fdce5444db497e91f2a48578209c"),
    "grace_bridge": ("grace_bridge.toml", 30, "3b42a2a54e3ebb9f4550051111adb1ec0219fe36de45b5cb45e9740d6ef91465",
                     "0658309ce91d4aab25959bfa212ac0109db0437b7d41746053da0113cf354280"),
    "all_sensors": (ALL_SENSORS, None, "288d495ff2bf86de75a7a2e43eec4761f0a1347b9418388d302f746e72695c13",
                    "16b5c2576aa5e2fb3d7ed5c99401dc94075a1a976924de405641e0fe9a1b3ec4"),
}


class Digest:
    """Wraps a brain and feeds every observation it receives into one shared sha256."""
    h = None

    def __init__(self, brain):
        self.brain = brain

    def __getattr__(self, name):
        return getattr(self.brain, name)

    def reset(self, spec):
        self.brain.reset(spec)

    def step(self, obs):
        h = Digest.h
        h.update(np.float64(obs.t).tobytes())
        for name in sorted(obs.readings):
            h.update(name.encode())
            for r in obs.readings[name]:
                h.update(np.float64(r.t).tobytes() + repr(r.data.shape).encode() + np.ascontiguousarray(r.data).tobytes())
        for pk in obs.inbox:
            h.update(repr((pk.src, pk.data, pk.rssi_dbm, pk.range_m)).encode())
        return self.brain.step(obs)


def reference_run(name, monkeypatch):
    """(run hash, digest of observations and events) of one reference scenario."""
    src, dur = REFERENCE[name][:2]
    sc = parse(src) if src.lstrip().startswith("[") else load(ROOT / "scenarios" / src)
    sc.pop("sweep", None)
    if dur is not None:
        sc = apply(sc, {"scenario.duration_s": dur})
    plain = sim.load_brain
    monkeypatch.setattr(sim, "load_brain", lambda path, params: Digest(plain(path, params)))
    Digest.h = hashlib.sha256()
    r = sim.run(sc, record=True)
    Digest.h.update(repr(r["events"]).encode())
    return r["hash"], Digest.h.hexdigest()


@pytest.mark.parametrize("name", list(REFERENCE))
def test_reference_hashes(name, monkeypatch):
    here = f"{platform.system()} | {platform.processor()}"
    if os.environ.get("CI") and here != PINNED_ON:
        pytest.skip(f"pinned on {PINNED_ON}, not on {here}")
    if name == "grace_bridge":
        pytest.importorskip("sim3d.agent")
    assert reference_run(name, monkeypatch) == REFERENCE[name][2:]
