# C2-swarm-lab

**A 3-D virtual test range for command-and-control algorithms of drone swarms, and the test bench for new GRACE
versions.**

A swarm algorithm that works in a grid world can fall apart on a real drone, which drifts, brakes late, misreads its
sensors and loses packets. C2-swarm-lab lets you measure that gap one factor at a time, on seeded worlds, on a single
laptop.

- **Physics:** rigid-body quadrotors (motor lag, drag, wind and gusts, autopilot), or point masses to isolate effects.
- **Sensors:** ToF zone arrays, lidar, depth, detector, VIO / GNSS pose, IMU, barometer, magnetometer, battery. Each has
  noise, dropouts and its own rate.
- **Radio:** a DIL channel (path loss, correlated shadowing, obstacle loss, interference, bit rate). An ideal radio is
  also available as a reference.
- **Worlds:** procedural biomes (`mixto`, `urbano`, `bosque`, `interior`), plus swarm-dil's voxel terrain
  (`swarm_dil_v2`). Every world comes from a seed.
- **Missions:** A to B (with regret against an oracle path), area coverage, pursuit, ground station.
- **Statistics:** paired sweeps over seeds, resumable, in up to 5 processes. Wilcoxon signed-rank with Holm's correction.
- **Viewer:** three.js replay in the browser, live while a run or a whole sweep goes on.

## The golden rule

A brain gets only what a real drone would measure: sensor readings with noise, in its own frame and at their own rate,
and radio bytes. It never gets world geometry, grids or global coordinates, unless a sensor that produces them is on
(`gnss`, `prior_map`). `tests/test_isolation.py` enforces this on every brain. The only exception is `brains/privileged/`
(the oracle, used to validate metrics), and its results are marked as privileged.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"        # .venv/bin/python on Linux and macOS
.venv/Scripts/python -m pytest -q
```

One run, saved with its replay:

```bash
.venv/Scripts/python -m c2lab run scenarios/ab_mixto.toml --seed 3
.venv/Scripts/python -m c2lab serve                      # then open the viewer URL the run printed
```

A paired sweep and its statistics:

```bash
.venv/Scripts/python -m c2lab sweep scenarios/sweep_ab_mixto.toml --live   # viewer: ?sweep=../runs/<name>&live=1
.venv/Scripts/python -m c2lab progress runs/sweep_ab_mixto --watch 10
.venv/Scripts/python -m c2lab summarize runs/sweep_ab_mixto
```

## Writing a brain

A brain is a class with two methods, loaded from a scenario as `brain = "brains.mine:Mine"`:

```python
from c2lab.brain import Action, Velocity

class Mine:
    def reset(self, spec):            # AgentSpec: id, team, seed, sensor parameters, limits, mission order
        ...

    def step(self, obs):              # Observation: readings per sensor since the last step, packets heard
        return Action(Velocity((1.0, 0.0, 0.0)), (b"hello",))   # setpoint in the local ENU frame, packets to send
```

Brains may import only `c2lab.brain`, the standard library, NumPy and SciPy. The baselines in `brains/` are a good
starting point: hover, random walk, potential field, boids, sector sweep, central assignment, pursue and flee.
Everything else, from the world and team to sensors, radio, mission and sweep variants, is set in a TOML file under
`scenarios/`.

## GRACE-3D 2.0.0 on a sensed drone (v0.3)

[GRACE-3D 2.0.0](https://github.com/RoperaLabs/swarm-dil) is written for a discrete voxel world. `brains/grace3d.py`
hosts its agent unchanged on a simulated drone, the way firmware would. The agent is pinned to swarm-dil's v2.0.0 commit
and installed with `pip install -e ".[grace]"`. The adapter turns ToF returns into blocked voxels (an occupancy grid
when the returns are noisy), turns each decision into a flown step, and passes radio bytes through untouched.

`benchmarks/grace_ladder.py` first runs an **equivalence bridge**: swarm-dil's idealisations, on its own terrain, on the
same 20 seeds. It then adds **one realism factor at a time**:

| Variant | Arrivals | Difference vs bridge | p (Holm) |
|---|---|---|---|
| Bridge (swarm-dil's idealisations) | 78.5 % | | |
| Quadrotor dynamics, 3 m/s wind | 77 % | −1.5 pp | n.s. |
| Noisy ToF, 2 × 2 subrays | 71 % | −7.5 pp | n.s. |
| ToF sees every direction | 66 % | −12.5 pp | n.s. |
| Ideal radio | 65 % | −13.5 pp | n.s. |
| Everything together | 43.5 % | −35 pp | 0.021 |
| Continuous urban world | 41.5 % | −37 pp | 0.016 |
| Teammates collide | 33.5 % | −45 pp | 0.004 |
| Drifting visual-inertial pose | 20 % | −58.5 pp | 0.006 |

On the same seeds, swarm-dil's own simulator brings 91.5 % home with a regret of 0.99. Measured inside its own lattice,
GRACE's regret on the bridge is 1.21, within swarm-dil's published 95 % interval. The rest of the gap is the price of a
body: aborted steps flown back, diagonals flown as axis legs, and steps refused at corners of zero width.

GRACE-3D shrugs off real dynamics, wind and ToF noise. A drifting pose breaks it, because its lattice no longer lines up
with the world. Teammates as physical obstacles break it too. To test a new GRACE version, see
[`docs/plug_grace.md`](docs/plug_grace.md).

## Layout

| Path | Contents |
|---|---|
| `src/c2lab/` | Simulator: `world/` (biomes, raycast, oracle), `physics/` (quadrotor, point mass, autopilot), `sensors.py`, `radio.py`, `missions.py`, `sim.py`, `sweep.py`, CLI |
| `brains/` | Baseline brains and the GRACE adapter; `privileged/` holds the oracle |
| `scenarios/` | TOML scenarios and sweeps |
| `benchmarks/` | Studies built on sweeps (the GRACE ladder) |
| `viewer/` | Browser replay and live viewer |

## Conventions

SI units throughout. The world frame is ENU and the body frame FLU (REP-103); quaternions are (w, x, y, z). Time runs in
integer physics ticks, at 1200 Hz by default. All randomness comes from `c2lab.seeds.rng(seed, *path)`, so a run is
fully determined by its scenario and seed.

## License

[MIT](LICENSE). Open research on synthetic data only.
