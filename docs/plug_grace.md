# Plugging in a GRACE version

GRACE agents are written for a discrete world: a 26-connected voxel lattice, one move per slot, sensing as blocked
voxels at relative offsets and radio as raw bytes. `brains/grace3d.py` hosts such an agent on a drone the way firmware
would, so the agent's own code runs unchanged. To test a new GRACE version:

1. **Pin it.** Add the release to the `grace` extra in `pyproject.toml` as a commit-pinned git dependency and install
   it into the project environment:

   ```bash
   .venv/Scripts/python -m pip install -e ".[grace]"
   ```

2. **Allow its agent module.** Add the agent-side modules (never its world or simulator) to `GRACE` in
   `tests/test_isolation.py`. For GRACE-3D 2.0.0 these are `sim3d.agent` and `sim3d.geom`.

3. **Adapt the interface, if it changed.** The adapter relies on this agent API:

   | call | meaning |
   |---|---|
   | `Agent(ident, Spec(shape, start, goal, team, launch_at, band, budget, r, seed, ...))` | mission order in lattice coordinates |
   | `observe(hits)` | blocked voxels seen this slot, as offsets from the agent's own voxel |
   | `transmit() -> bytes or None` | at most `budget` bytes, broadcast unchanged |
   | `receive(bytes)` | every packet heard since the previous slot |
   | `decide() -> direction or None` | index into the 26 lattice steps, or hold |
   | `moved(direction, ok)` | whether the step happened |
   | `pos`, `heading`, `t`, `arrived`, `tiles`, `memory_used` | read by the adapter |

   If a version keeps this API, the adapter needs no change. Otherwise copy `brains/grace3d.py` to `brains/graceN.py`
   and change only the calls in `_slot`, `reset` and `state_bytes`.

4. **Point the ladder at it.** In `scenarios/grace_ladder.toml`, the team line is the only change:

   ```toml
   brain = "brains.grace3d:Grace3D"        # or brains.graceN:GraceN
   brain_params = { team = 10, view = "swarm_dil" }
   ```

   Extra keyword arguments in `brain_params` are passed to the agent's `Spec` (for example `K = 14`, `price = 1.0`).

5. **Run it.**

   ```bash
   .venv/Scripts/python benchmarks/grace_ladder.py
   ```

   This writes `runs/grace_ladder/equivalence.md` (the bridge against swarm-dil's own simulator, on swarm-dil's terms)
   and `summary.md` (each realism factor against the bridge, paired over 20 seeds, Wilcoxon with Holm). Use a new
   output folder, or delete the old one, when the brain changes: the sweep resumes any run already present.

## What the adapter does

- **Lattice.** `cell` metres (default 5 m), anchored on the mission area (`area_local`) so the whole team shares it.
  A (`a_local`) and B (`goal_local`) give the start and goal voxels.
- **Slot.** `moved` → `observe` → `transmit` → `receive` → `decide` → fly. A slot ends when the drone reaches the
  target, or after `cell / speed` seconds of holding. After `t_max` slots (500, swarm-dil's horizon) a drone that has
  not arrived holds where it is.
- **Sensing.** ToF returns are placed with the pose sample nearest in time. Exact returns (no range noise, one ray
  per zone) go to the voxel beyond the nearest lattice face. Uncertain returns feed an occupancy grid instead: the
  sensor's datasheet values (`sigma_rel`, `sigma_min`, `sub`) give each return a 3-sigma error. The return, pushed
  that far beyond along its ray, votes for its voxel, and the ray up to that far short of it votes free for every
  voxel it crosses. A voxel reaches GRACE after `OCC_MIN` votes and `OCC_RATIO` times as many as rays through it.
  Returns within that error of the floor are dropped as ground. `view = "swarm_dil"` keeps only swarm-dil's two
  sensor frusta; `view = "all"` keeps every return.
- **Motion.** Velocity setpoints from station to station. Each drone has its own station inside a voxel (a ring of `RING`
  = 0.18 cell, alternate layers above and below the centre), so teammates can share one. A diagonal that grazes a known blocked voxel is flown as axis legs through free voxels. The step is
  refused (`moved(d, False)`) when its target is known blocked or no such legs exist. A step is also abandoned, flying
  back, when a ToF return comes within `margin` of the remaining path or the move times out.

Each of these is a departure from the discrete world. The ladder measures what each one costs.
