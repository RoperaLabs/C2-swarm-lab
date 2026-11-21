"""F7.3-7.4: GRACE-3D 2.0.0 equivalence bridge and realism ladder (scenarios/grace_ladder.toml).

    python benchmarks/grace_ladder.py [--workers 5] [--live]

1. Runs every ladder variant on the 20 paired seeds (resumable sweep into runs/grace_ladder/).
2. Runs swarm-dil's own simulator (sim3d.world.run) on the same seeds: the reference.
3. Scores every run on the swarm-dil terrain in swarm-dil's terms (Outcome.regret): voxels flown, plus the true
   26-lattice distance left for a drone that did not arrive, over the true distance, minus one. `regret_dil` and
   `arrival_dil` are added to each metrics.json.
4. Writes equivalence.md (bridge against swarm-dil) and summary.md (every factor against the bridge, Holm).
"""
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2lab.scenario import load  # noqa: E402
from c2lab.sweep import load_results, summarize, sweep  # noqa: E402

OUT = ROOT / "runs" / "grace_ladder"
PUBLISHED = {"regret": (0.991, 0.144), "arrival": 0.915}     # swarm-dil v2.0.0 README, 20 seeds, mean (± s.e.)


def terrain(seed):
    from sim3d.world import Mission
    from swarm_dil.nav3.world3 import make_world
    m = Mission()
    blocked, start, goal, fields = make_world(m, np.random.default_rng(seed))
    return m, blocked, start, fields[(0, m.shape[2] - 1)]


def reference(seed):
    from sim3d.world import Mission, run
    o = run(Mission(), seed)
    return seed, {"regret": o.regret, "arrival": o.arrival_rate, "bytes": int(o.bytes_sent.sum())}


def regret_dil(seed, per_agent):
    """swarm-dil's regret and arrival rate for one c2lab run on terrain `seed`."""
    m, blocked, start, true = terrain(seed)
    nx, ny, nz = m.shape
    ix = lambda c: (c[0] * ny + c[1]) * nz + c[2]  # noqa: E731
    d_star = true[ix(start)]
    out = []
    for flown, arrived, final in zip(per_agent["flown_m"], per_agent["arrived_s"], per_agent["final_enu"]):
        left = 0.0
        if arrived is None:
            c = np.clip(np.floor(np.array(final) / m.cell_m).astype(int), 0, np.array(m.shape) - 1)
            near = [tuple(int(v) for v in c + np.array(d) - 1) for d in np.ndindex(3, 3, 3)]
            left = min((true[ix(q)] + (q != tuple(c))) for q in near
                       if all(0 <= q[k] < m.shape[k] for k in range(3)))
        out.append((flown / m.cell_m + left) / d_star - 1.0)
    return float(np.mean(out)), float(np.mean([a is not None for a in per_agent["arrived_s"]]))


def mean_se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--live", action="store_true", help="replays written live: viewer/?sweep=../runs/grace_ladder&live=1")
    a = ap.parse_args()
    sc = load(ROOT / "scenarios" / "grace_ladder.toml")
    done = sweep(sc, OUT, a.workers, a.live)
    print(f"{len(done)} runs, {sum(st != 'ok' for _, st in done)} failed")

    seeds = range(*sc["sweep"]["seeds"])
    ref_path = OUT / "swarm_dil.json"
    if ref_path.exists():
        ref = {int(k): v for k, v in json.loads(ref_path.read_text(encoding="utf-8")).items()}
    else:
        with get_context("spawn").Pool(a.workers) as pool:
            ref = dict(pool.map(reference, seeds))
        ref_path.write_text(json.dumps(ref, indent=1), encoding="utf-8")

    biome = {v["name"]: v.get("set", {}).get("world.biome", sc["world"]["biome"]) for v in sc["sweep"]["variant"]}
    for variant, runs in load_results(OUT).items():
        if biome[variant] != "swarm_dil_v2":
            continue
        for seed, m in runs.items():
            path = OUT / variant / str(seed) / "metrics.json"
            m["regret_dil"], m["arrival_dil"] = regret_dil(int(seed), m["per_agent"])
            path.write_text(json.dumps(m, indent=1), encoding="utf-8")

    bridge = load_results(OUT)["bridge"]
    common = sorted(bridge)
    ours_r = [bridge[s]["regret_dil"] for s in common]
    ours_a = [bridge[s]["arrival_dil"] for s in common]
    ref_r = [ref[s]["regret"] for s in common]
    ref_a = [ref[s]["arrival"] for s in common]
    (mr, sr), (ma, sa) = mean_se(ours_r), mean_se(ours_a)
    (rr, rsr), (ra, rsa) = mean_se(ref_r), mean_se(ref_a)
    p_r = wilcoxon(np.subtract(ours_r, ref_r)).pvalue
    d_a = np.subtract(ours_a, ref_a)
    p_a = wilcoxon(d_a).pvalue if np.any(d_a) else 1.0
    pub_r, pub_se = PUBLISHED["regret"]
    lines = [
        f"Equivalence bridge, {len(common)} paired seeds (GRACE-3D 2.0.0 in c2lab against swarm-dil's simulator).", "",
        "| | regret (mean ± s.e.) | arrival (mean ± s.e.) | bytes / mission |", "|---|---|---|---|",
        f"| swarm-dil v2.0.0 published | {pub_r:.3f} ± {pub_se:.3f} | {PUBLISHED['arrival']:.3f} | 13.3 kB |",
        f"| swarm-dil re-run (sim3d.world.run) | {rr:.3f} ± {rsr:.3f} | {ra:.3f} ± {rsa:.3f} | "
        f"{np.mean([ref[s]['bytes'] for s in common]) / 1e3:.1f} kB |",
        f"| c2lab bridge | {mr:.3f} ± {sr:.3f} | {ma:.3f} ± {sa:.3f} | "
        f"{np.mean([bridge[s]['bytes_sent'] for s in common]) / 1e3:.1f} kB |", "",
        f"Paired difference c2lab - swarm-dil: regret {np.mean(ours_r) - rr:+.3f} (Wilcoxon p = {p_r:.3g}), "
        f"arrival {np.mean(d_a):+.3f} (p = {p_a:.3g}).",
        f"Published 95 % interval of the regret: [{pub_r - 1.96 * pub_se:.3f}, {pub_r + 1.96 * pub_se:.3f}]; "
        f"c2lab bridge {'inside' if abs(mr - pub_r) <= 1.96 * pub_se else 'outside'}.",
    ]
    (OUT / "equivalence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    summarize(OUT, ["arrival_rate", "regret_dil", "regret_mean", "bytes_sent", "crashes"], baseline="bridge")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
