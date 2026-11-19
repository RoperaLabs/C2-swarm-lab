"""Run directory: meta.json (scenario, versions, hashes), metrics.json, events.jsonl and replay.json (world + frames,
read by viewer/?run=...)."""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


def save(out_dir, scenario, result):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = result["world"]
    meta = {"scenario": scenario, "world_hash": w.hash, "run_hash": result.get("hash"),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result["metrics"], indent=1), encoding="utf-8")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:
        for tick, g, kind, info in result["events"]:
            f.write(json.dumps({"tick": tick, "agent": g, "kind": kind, "info": info}) + "\n")
    return save_replay(out, scenario, result)


def save_replay(out_dir, scenario, result):
    """replay.json alone (what the viewer reads); a live run rewrites it while it goes on."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = result["world"]
    frames = result.get("frames", [])
    replay = {"world": w.to_dict(), "metrics": result["metrics"],
              "teams": [t["name"] for t in scenario["team"] for _ in range(t["count"])],
              "t": [round(f[0], 3) for f in frames],
              "p": [np.round(f[1], 3).ravel().tolist() for f in frames],
              "q": [np.round(f[2], 4).ravel().tolist() for f in frames],
              "alive": [f[3].astype(int).tolist() for f in frames],
              "links": [[round(t, 2), pairs] for t, pairs in result.get("links", [])],
              "covered": result.get("covered", []),
              "station": result.get("station"),
              "events": [{"t": round(tick / scenario["scenario"]["rates_hz"]["physics"], 3), "agent": g, "kind": kind}
                         for tick, g, kind, _ in result["events"]]}
    tmp = out / "replay.json.tmp"
    tmp.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    try:
        os.replace(tmp, out / "replay.json")       # a viewer polling a live run never reads half a file
    except PermissionError:                        # the server holds it open (Windows): next snapshot retries
        pass
    return out
