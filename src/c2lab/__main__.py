"""python -m c2lab <command>
    run <scenario.toml> [--seed N] [--out DIR] [--realtime F]   one run, saved with its replay; prints the metrics
    sweep <scenario.toml> [--workers 5] [--out DIR]     every [sweep] variant x seed, resumable
    summarize <sweep dir> [--metric M ...] [--baseline V]  paired statistics -> summary.csv and summary.md
    progress <sweep dir> [--watch S]                    runs done per variant and their running means
    serve [--port 8770]                                 static server for the viewer (project root)
"""
import argparse
import functools
import http.server
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))      # brains/ lives at the project root

from c2lab.scenario import load  # noqa: E402

CLEAR, NL = chr(27) + "[2J" + chr(27) + "[H", chr(10)     # ANSI clear screen + home

ap = argparse.ArgumentParser(prog="python -m c2lab")
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("scenario")
r.add_argument("--seed", type=int)
r.add_argument("--out")
r.add_argument("--live", action="store_true", help="rewrite the replay every few seconds while the run goes on")
r.add_argument("--realtime", type=float, metavar="F", help="pace the run at F x real time (0.1 to 10)")
s = sub.add_parser("sweep")
s.add_argument("scenario")
s.add_argument("--workers", type=int, default=5)
s.add_argument("--out")
s.add_argument("--live", action="store_true", help="each process rewrites its run's replay while it goes on")
m = sub.add_parser("summarize")
m.add_argument("root")
m.add_argument("--metric", nargs="*")
m.add_argument("--baseline")
g = sub.add_parser("progress")
g.add_argument("root")
g.add_argument("--watch", type=float, help="refresh every S seconds until Ctrl+C")
v = sub.add_parser("serve")
v.add_argument("--port", type=int, default=8770)
a = ap.parse_args()

if a.cmd == "run":
    from c2lab.clock import RunClock
    from c2lab.record import save, save_replay
    from c2lab.sim import run
    sc = load(a.scenario)
    sc.pop("sweep", None)
    if a.seed is not None:
        sc["scenario"]["seed"] = a.seed
    out = a.out or f"runs/{sc['scenario']['name']}_{sc['scenario']['seed']}"
    if a.live:
        print(f"live: /viewer/?run=../{Path(out).as_posix()}/replay.json&live=1")
    t0 = time.perf_counter()
    res = run(sc, record=True, live=(lambda part: save_replay(out, sc, part)) if a.live else None,
              clock=RunClock("realtime", a.realtime) if a.realtime else None)
    wall = time.perf_counter() - t0
    out = save(out, sc, res)
    print(json.dumps(res["metrics"], indent=1))
    print(f"{out}  {res['metrics']['seconds']:.1f} s simulated in {wall:.1f} s "
          f"({res['metrics']['seconds'] / wall:.1f}x)  viewer: /viewer/?run=../{out.as_posix()}/replay.json")
elif a.cmd == "sweep":
    from c2lab.sweep import sweep
    sc = load(a.scenario)
    if "sweep" not in sc:
        sys.exit(f"{a.scenario}: no [sweep] table")
    root = a.out or f"runs/{sc['scenario']['name']}"
    t0 = time.perf_counter()
    done = sweep(sc, root, a.workers, a.live)
    bad = [o for o, st in done if st != "ok"]
    print(f"{len(done)} runs in {time.perf_counter() - t0:.0f} s, {len(bad)} failed{': ' + ', '.join(bad) if bad else ''}"
          f"  ->  {root}")
elif a.cmd == "summarize":
    from c2lab.sweep import summarize
    summarize(a.root, a.metric, a.baseline)
    print((Path(a.root) / "summary.md").read_text(encoding="utf-8"))
elif a.cmd == "progress":
    from c2lab.sweep import progress
    while True:
        text = progress(a.root)
        if not a.watch:
            print(text)
            break
        print(CLEAR + time.strftime("%H:%M:%S  ") + a.root + NL + text, flush=True)
        time.sleep(a.watch)
else:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path.cwd()))
    print(f"http://127.0.0.1:{a.port}/viewer/   (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
