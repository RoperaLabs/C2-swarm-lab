"""F5 acceptance over 20 paired seeds (<= 5 processes, one numba thread each):
- a_to_b: the privileged oracle's regret (flown / shortest - 1) stays within 0.05;
- coverage: central C2 (ground station + sector sweep) against random walk, paired Wilcoxon;
- pursuit: a stationary evader is always caught; a fleeing one is caught less often by 1 pursuer than by 4."""
import multiprocessing
import os
import sys

os.environ["NUMBA_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

SEEDS = range(20)


def job(args):
    from c2lab.scenario import load
    from c2lab.sim import run
    kind, seed = args
    if kind == "oracle":
        sc = load("scenarios/ab_mixto.toml")
        t = sc["team"][0]
        t["brain"], t["brain_params"], t["launch_s"] = "brains.privileged.oracle:Oracle", {}, 4.0
        t["sensors"]["pose"] = {"profile": "ideal"}
    elif kind.startswith("cover"):
        sc = load("scenarios/coverage_mixto.toml")
        if kind == "cover_random":
            sc.pop("ground_station")
            sc["team"][0]["brain"] = "brains.random_walk:RandomWalk"
    else:
        sc = load("scenarios/pursuit_mixto.toml")
        if kind == "still":
            sc["team"][1]["brain_params"] = {"speed": 0.0}
        if kind == "flee1":
            sc["team"][0]["count"] = 1
    sc["scenario"]["seed"] = seed
    return kind, seed, run(sc)["metrics"]


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["oracle", "cover_central", "cover_random", "still", "flee1", "flee4"]
    with ProcessPoolExecutor(5, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(job, [(k, s) for k in kinds for s in SEEDS]))
    m = {k: [r[2] for r in res if r[0] == k] for k in kinds}
    if "oracle" in m:
        reg = np.array([x["regret_mean"] for x in m["oracle"] if x["regret_mean"] is not None])
        print(f"a_to_b oracle: arrival {np.mean([x['arrival_rate'] for x in m['oracle']]):.3f}, regret mean {reg.mean():+.4f}"
              f" max |.| {np.abs(reg).max():.4f}  {'OK' if np.abs(reg.mean()) <= 0.05 else 'OUT'}")
    if "cover_central" in m:
        c, r = (np.array([x["coverage"] for x in m[k]]) for k in ("cover_central", "cover_random"))
        p = wilcoxon(c, r, alternative="greater").pvalue
        print(f"coverage: central {c.mean():.3f} +- {c.std():.3f}, random {r.mean():.3f} +- {r.std():.3f}, "
              f"paired Wilcoxon p = {p:.2g}  {'OK' if p < 0.05 else 'OUT'}")
    if "still" in m:
        still = np.array([x["capture_rate"] for x in m["still"]])
        f1, f4 = (np.array([x["capture_rate"] for x in m[k]]) for k in ("flee1", "flee4"))
        print(f"pursuit: stationary caught {still.mean():.2f}  {'OK' if still.min() == 1 else 'OUT'}; "
              f"fleeing caught by 1 pursuer {f1.mean():.2f}, by 4 {f4.mean():.2f}  {'OK' if f1.mean() < f4.mean() else 'OUT'}")
