"""python -m c2lab run <scenario.toml> [--seed N] [--out DIR]: run one scenario, save it and print its metrics.
(F6 adds sweep, summarize and serve.)"""
import argparse
import json
import sys
import time
from pathlib import Path

from c2lab.record import save
from c2lab.scenario import load
from c2lab.sim import run

sys.path.insert(0, str(Path.cwd()))      # brains/ lives at the project root

ap = argparse.ArgumentParser(prog="python -m c2lab")
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("scenario")
r.add_argument("--seed", type=int)
r.add_argument("--out")
a = ap.parse_args()
sc = load(a.scenario)
if a.seed is not None:
    sc["scenario"]["seed"] = a.seed
t0 = time.perf_counter()
res = run(sc, record=True)
wall = time.perf_counter() - t0
out = save(a.out or f"runs/{sc['scenario']['name']}_{sc['scenario']['seed']}", sc, res)
m = res["metrics"]
print(json.dumps(m, indent=1))
print(f"{out}  {m['seconds']:.1f} s simulated in {wall:.1f} s ({m['seconds'] / wall:.1f}x)  "
      f"viewer: /viewer/?run=../{out.as_posix()}/replay.json")
