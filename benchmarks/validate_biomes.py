"""Biome acceptance over many seeds (<= 5 processes, one numba thread each): feasibility, distinct hashes, attempts,
time, and plan-view cover against the target density."""
import multiprocessing
import os

os.environ["NUMBA_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from c2lab.scenario import BIOMES  # noqa: E402
from c2lab.world import biomes, generate, oracle  # noqa: E402

SEEDS, DENSITIES, DSEEDS = 200, (0.1, 0.2, 0.35), 20


def one(job):
    biome, density, seed = job
    size = BIOMES[biome]
    w = generate(biome, size, density, seed)
    return job, w.hash, w.attempts, w.seconds, oracle.coverage(w, oracle.RES[biome], biomes.z_top(biome, size))


if __name__ == "__main__":
    jobs = [(b, 0.2, s) for b in sorted(BIOMES) for s in range(SEEDS)]
    jobs += [(b, d, s) for b in sorted(BIOMES) for d in DENSITIES if d != 0.2 for s in range(DSEEDS)]
    with ProcessPoolExecutor(5, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    for b in sorted(BIOMES):
        main = [r for r in res if r[0][0] == b and r[0][1] == 0.2]
        att, sec = np.array([r[2] for r in main]), np.array([r[3] for r in main])
        print(f"{b:9s} {len(main)} feasible, {len({r[1] for r in main})} distinct hashes, attempts mean "
              f"{att.mean():.2f} max {att.max()}, time mean {sec.mean():.2f} s max {sec.max():.2f} s")
        for d in DENSITIES:
            cov = np.array([r[4] for r in res if r[0][0] == b and r[0][1] == d][:DSEEDS])
            ok = abs(cov.mean() - d) <= 0.15 * d
            print(f"          density {d:.2f}: cover {cov.mean():.3f} +- {cov.std():.3f}  {'OK' if ok else 'OUT'}")
