import json

import numpy as np
import pytest

from c2lab.scenario import ScenarioError, parse
from c2lab.sweep import apply, holm, load_results, summarize, sweep

SC = """
[scenario]
name = "s"
seed = 0
duration_s = 3
mission = "a_to_b"
[world]
biome = "mixto"
density = 0.1
[radio]
profile = "ideal"
[[team]]
name = "a"
count = 2
brain = "brains.random_walk:RandomWalk"
sensors.pose = { profile = "ekf" }
[sweep]
seeds = [0, 2]
[[sweep.variant]]
name = "rw"
[[sweep.variant]]
name = "pf"
set = { "team.0.brain" = "brains.potential_field:PotentialField" }
"""


def hashes(root):
    return {(v, s): m["run_hash"] for v, runs in load_results(root).items() for s, m in runs.items()}


def test_sweep_is_independent_of_workers_and_resumes(tmp_path):
    sc = parse(SC)
    a, b = tmp_path / "one", tmp_path / "two"
    assert len(sweep(sc, a, workers=1)) == 4
    assert len(sweep(sc, b, workers=2)) == 4
    assert hashes(a) == hashes(b) and len(hashes(a)) == 4
    (a / "pf" / "1" / "metrics.json").unlink()            # a killed sweep: only the missing run is redone
    assert [o for o, _ in sweep(sc, a, workers=1)] == [str(a / "pf" / "1")]
    assert hashes(a) == hashes(b)


def test_summary_files(tmp_path):
    sweep(parse(SC), tmp_path, workers=1)
    rows = summarize(tmp_path, ["flown_mean", "crashes"], baseline="rw")
    assert {(r["variant"], r["metric"]) for r in rows} == {(v, m) for v in ("rw", "pf") for m in ("flown_mean", "crashes")}
    assert (tmp_path / "summary.csv").exists() and "Holm" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_holm_and_overrides():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    sc = apply(parse(SC), {"team.0.count": 5, "radio.profile": "disco"})
    assert sc["team"][0]["count"] == 5 and sc["radio"]["profile"] == "disco"
    with pytest.raises(ScenarioError, match="sweep.seeds"):
        parse(SC.replace("seeds = [0, 2]", "seeds = [3, 1]"))
    json.dumps(sc)                                         # overrides keep the scenario plain data
