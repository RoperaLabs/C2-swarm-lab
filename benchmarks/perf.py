"""Throughput of the simulation loop (plan F8.1): drones·tick/s and real-time factor against N.

    python benchmarks/perf.py            full curve: N = 1, 10, 50, 100 · quad, pointmass · sensors on, off
    python benchmarks/perf.py --save     ... and write it to benchmarks/perf_baseline.json
    python benchmarks/perf.py --check    regression set against benchmarks/perf_baseline.json (exit 1 if slower)

Workload: scenarios/ab_mixto.toml with N boids launched together and drone collisions off (the 50- and 100-drone grids
at A are tighter than two iris radii). "on" keeps its sensors (six ToF arrays, EKF pose and an all-round detector),
"off" keeps ideal pose only, so those boids fly blind and some hit obstacles: drones·tick/s counts the drones alive in
the timed window, and `alive` is their mean share. Every configuration runs REPS times on the same seed in this
process with one numba thread, after one untimed run that compiles and loads everything. Only the window after
WARMUP_S simulated seconds is timed, so numba compilation, world generation and the oracle paths stay out. The time is
split by wrapping the components: physics (Drones.step), rays (ToF, lidar and depth casts and the detector), other
sensors, radio, brains (and a ground station), mission, and the rest of the loop (recording, observations, end check).

Scaling up goes across seeds, never threads: one run per process (python -m c2lab sweep, at most 5 processes), each
with NUMBA_NUM_THREADS=1 (plan R1). Runs are independent, and most of a run's cost is Python that threads would not
speed up.
"""
import argparse
import json
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "1"           # before numba is imported

import numba  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # brains/ lives at the project root

from c2lab import sim  # noqa: E402
from c2lab.missions import MISSIONS  # noqa: E402
from c2lab.physics import Drones  # noqa: E402
from c2lab.radio import Radio  # noqa: E402
from c2lab.scenario import load  # noqa: E402
from c2lab.sensors import Sensors  # noqa: E402
from c2lab.sweep import apply  # noqa: E402

SCENARIO = "scenarios/ab_mixto.toml"
BASELINE = ROOT / "benchmarks" / "perf_baseline.json"
CURVE = [(n, dyn, sens) for sens in ("on", "off") for dyn in ("quad", "pointmass") for n in (1, 10, 50, 100)]
CHECK = [(10, "quad", "on"), (50, "quad", "on"), (50, "pointmass", "off")]
WINDOW_S = {1: 120.0, 10: 15.0, 50: 6.0, 100: 4.0}   # timed simulated seconds per run
WARMUP_S = 3.0
REPS = 5
TOLERANCE = 0.15                                 # --check fails below (1 - TOLERANCE) x the baseline rate
PHASES = ("physics", "rays", "sensors", "radio", "brains", "mission")


class W:
    """Stopwatch shared by the wrappers: off during the warm-up, then per-phase seconds."""
    on, warm, first, last, t0, acc = False, 0, 0, 0, 0.0, {}

    @classmethod
    def reset(cls, warm):
        cls.on, cls.warm, cls.acc = False, warm, dict.fromkeys(PHASES, 0.0)


def timed(fn, phase):
    def wrap(*a, **k):
        if not W.on:
            return fn(*a, **k)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            W.acc[phase] += time.perf_counter() - t
    return wrap


def physics(fn):
    step = timed(fn, "physics")

    def wrap(self, tick, *a, **k):
        if not W.on and tick >= W.warm:          # first physics call after the warm-up starts the clock
            W.on, W.first, W.t0 = True, tick, time.perf_counter()
        W.last = tick
        return step(self, tick, *a, **k)
    return wrap


def instrument():
    Drones.step = physics(Drones.step)
    Sensors.step = timed(Sensors.step, "sensors")
    Sensors._cast = timed(Sensors._cast, "rays")
    Sensors._detector = timed(Sensors._detector, "rays")
    Radio.slot, Radio.send = timed(Radio.slot, "radio"), timed(Radio.send, "radio")
    for cls in set(MISSIONS.values()):
        cls.update = timed(cls.update, "mission")
    plain = sim.load_brain

    def load_brain(path, params):
        b = plain(path, params)
        b.step = timed(b.step, "brains")
        return b
    sim.load_brain = load_brain


def scenario(n, dyn, sens, duration):
    sc = load(ROOT / SCENARIO)
    sets = {"team.0.count": n, "team.0.dynamics": dyn, "team.0.launch_s": 0.0, "world.drone_collisions": False,
            "scenario.duration_s": duration}
    if sens == "off":
        sets["team.0.sensors"] = {"pose": {"profile": "ideal"}}
    return apply(sc, sets)


def once(n, dyn, sens):
    sc = scenario(n, dyn, sens, WARMUP_S + WINDOW_S[n])
    phys = sc["scenario"]["rates_hz"]["physics"]
    W.reset(round(WARMUP_S * phys))
    r = sim.run(sc, record=True)
    if not W.on:
        raise RuntimeError(f"N={n} {dyn} {sens}: the run ended during the warm-up")
    wall = time.perf_counter() - W.t0
    ticks = W.last + 1 - W.first
    split = {k: v / wall for k, v in W.acc.items()}
    split["sensors"] -= split["rays"]            # rays are timed inside Sensors.step
    split["rest"] = 1 - sum(split.values())
    alive = float(np.mean([f[3].mean() for f in r["frames"] if f[0] >= WARMUP_S]))
    return {"rate": n * alive * ticks / wall, "rtf": ticks / phys / wall, "split": split, "alive": alive}


def measure(n, dyn, sens, reps):
    runs = [once(n, dyn, sens) for _ in range(reps)]
    rate, rtf = np.array([r["rate"] for r in runs]), np.array([r["rtf"] for r in runs])
    return {"n": n, "dynamics": dyn, "sensors": sens, "sim_s": WINDOW_S[n], "reps": reps,
            "rate_mean": float(rate.mean()), "rate_std": float(rate.std(ddof=1)),
            "cv": float(rate.std(ddof=1) / rate.mean()), "rtf_mean": float(rtf.mean()), "rtf_std": float(rtf.std(ddof=1)),
            "split": {k: round(float(np.mean([r["split"][k] for r in runs])), 4) for k in (*PHASES, "rest")},
            "alive": runs[-1]["alive"]}


def machine():
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "processor": platform.processor(), "cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "numba": numba.__version__}


def document(results, conditions=""):
    """The c2lab-perf/1 record of a set of results; `conditions` notes what the platform does not say (power...)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"format": "c2lab-perf/1", "version": project["version"], "date": time.strftime("%Y-%m-%d"),
            "platform": machine(), "conditions": conditions, "scenario": SCENARIO,
            "seed": load(ROOT / SCENARIO)["scenario"]["seed"], "warmup_s": WARMUP_S, "numba_threads": 1,
            "results": results}


HEAD = (f"{'N':>4} {'dynamics':<9} {'sensors':<7} {'drones*tick/s':>17} {'CV':>6} {'x real':>7}  "
        + " ".join(f"{k:>7}" for k in (*PHASES, "rest")) + "  alive")


def row(r):
    return (f"{r['n']:>4} {r['dynamics']:<9} {r['sensors']:<7} {r['rate_mean']:>9,.0f} +- {r['rate_std']:<6,.0f}"
            f"{r['cv']:>6.1%} {r['rtf_mean']:>7.2f}  " + " ".join(f"{r['split'][k]:>7.1%}" for k in (*PHASES, "rest"))
            + f"  {r['alive']:.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--save", action="store_true", help="write the curve to benchmarks/perf_baseline.json")
    ap.add_argument("--check", action="store_true", help="compare the regression set with the baseline")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", help="also write the results to this JSON file")
    ap.add_argument("--note", default="", help="measuring conditions to record (power source, CPU limits...)")
    a = ap.parse_args()
    instrument()
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if a.check else None
    W.reset(0)
    sim.run(scenario(10, "quad", "on", 2.0), record=True)       # compile and load everything, untimed
    print(HEAD, flush=True)
    results = []
    for cfg in (CHECK if a.check else CURVE):
        results.append(measure(*cfg, a.reps))
        print(row(results[-1]), flush=True)
    doc = document(results, a.note)
    for path in ([BASELINE] if a.save else []) + ([Path(a.out)] if a.out else []):
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if a.check:
        same = {k: base["platform"][k] for k in ("system", "machine", "processor", "cpus")} == \
               {k: doc["platform"][k] for k in ("system", "machine", "processor", "cpus")}
        ref = {(r["n"], r["dynamics"], r["sensors"]): r["rate_mean"] for r in base["results"]}
        print(f"baseline {base['version']} of {base['date']}: {base.get('conditions') or 'no conditions noted'}")
        slow = []
        for r in results:
            ratio = r["rate_mean"] / ref[r["n"], r["dynamics"], r["sensors"]]
            print(f"{r['n']} {r['dynamics']} {r['sensors']}: {ratio:.2f} x baseline")
            slow += [r] * (ratio < 1 - TOLERANCE)
        if not same:
            print(f"baseline platform differs ({base['platform']['processor']}): report only")
        elif slow:
            sys.exit(f"{len(slow)} configuration(s) more than {TOLERANCE:.0%} slower than the baseline")


if __name__ == "__main__":
    main()
