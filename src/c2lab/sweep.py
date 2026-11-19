"""Sweeps and their statistics.

A sweep runs every variant of a scenario on the same seeds (paired), in up to 5 processes with one numba thread each,
writing runs/<name>/<variant>/<seed>/metrics.json (or error.json). Runs already on disk are skipped, so a killed sweep
resumes where it stopped. Results do not depend on the number of processes: every run is seeded on its own.

TOML:
    [sweep]
    seeds = [0, 20]                      # range [from, to)
    [[sweep.variant]]
    name = "boids"
    set = { "team.0.brain" = "brains.boids:Boids", "radio.profile" = "disco" }

summarize(): per variant and metric, mean and standard error; paired differences against the first variant with a
one-sample Wilcoxon signed-rank test and Holm's correction over all comparisons; written as CSV and Markdown.
"""
import copy
import csv
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MAX_WORKERS = 5


def apply(sc, overrides):
    """Copy of scenario sc with dotted keys set ("team.0.brain": ...); list items by index."""
    sc = copy.deepcopy(sc)
    for key, value in overrides.items():
        node, parts = sc, key.split(".")
        for p in parts[:-1]:
            node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value
    return sc


def jobs(sc, root):
    sw = sc["sweep"]
    a, b = sw["seeds"]
    variants = sw.get("variant") or [{"name": "base", "set": {}}]
    out = []
    for v in variants:
        for seed in range(a, b):
            run_sc = apply(sc, {**v.get("set", {}), "scenario.seed": seed})
            run_sc.pop("sweep", None)
            out.append((run_sc, str(Path(root) / v["name"] / str(seed))))
    return out


def _one(job):
    sc, out = job
    from c2lab.sim import run
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    try:
        r = run(sc, record=True)
        (path / "metrics.json").write_text(json.dumps({**r["metrics"], "run_hash": r["hash"]}, indent=1),
                                           encoding="utf-8")
        return out, "ok"
    except Exception:
        (path / "error.json").write_text(json.dumps({"error": traceback.format_exc()}), encoding="utf-8")
        return out, "error"


def sweep(sc, root, workers=MAX_WORKERS):
    todo = [j for j in jobs(sc, root) if not (Path(j[1]) / "metrics.json").exists()]
    workers = max(1, min(workers, MAX_WORKERS))
    os.environ["NUMBA_NUM_THREADS"] = "1"           # inherited by the worker processes
    if workers == 1:
        return [_one(j) for j in todo]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_one, todo))


def load_results(root):
    """{variant: {seed: metrics}} from a sweep directory."""
    out = {}
    for f in Path(root).glob("*/*/metrics.json"):
        out.setdefault(f.parent.parent.name, {})[int(f.parent.name)] = json.loads(f.read_text(encoding="utf-8"))
    return out


def holm(p):
    """Holm-Bonferroni adjusted p-values, same order as p."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def summarize(root, metrics=None, baseline=None):
    from scipy.stats import wilcoxon
    res = load_results(root)
    variants = sorted(res, key=lambda v: (v != baseline, v))
    base = baseline or variants[0]
    numeric = sorted({k for runs in res.values() for m in runs.values() for k, v in m.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)})
    metrics = metrics or [k for k in numeric if k not in ("agents", "seconds")]
    rows = []
    for v in variants:
        for k in metrics:
            vals = {s: m[k] for s, m in res[v].items() if isinstance(m.get(k), (int, float))}
            x = np.array(list(vals.values()), float)
            row = {"variant": v, "metric": k, "n": len(x), "mean": float(x.mean()) if len(x) else math.nan,
                   "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan,
                   "diff": math.nan, "p": math.nan}
            if v != base:
                common = sorted(set(vals) & {s for s, m in res[base].items() if isinstance(m.get(k), (int, float))})
                d = np.array([vals[s] - res[base][s][k] for s in common], float)
                if len(d):
                    row["diff"] = float(d.mean())
                    row["p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            rows.append(row)
    tested = [r for r in rows if not math.isnan(r["p"])]
    for r, a in zip(tested, holm([r["p"] for r in tested])):
        r["p_holm"] = float(a)
    for r in rows:
        r.setdefault("p_holm", math.nan)
    root = Path(root)
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Baseline: `{base}`. Paired differences against it; Wilcoxon signed-rank, Holm over {len(tested)} "
             "comparisons.", "", "| variant | metric | n | mean ± s.e. | diff | p (Holm) |", "|---|---|---|---|---|---|"]
    for r in rows:
        diff = "" if math.isnan(r["diff"]) else f"{r['diff']:+.4g}"
        p = "" if math.isnan(r["p_holm"]) else f"{r['p_holm']:.3g}"
        lines.append(f"| {r['variant']} | {r['metric']} | {r['n']} | {r['mean']:.4g} ± {r['se']:.2g} | {diff} | {p} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
