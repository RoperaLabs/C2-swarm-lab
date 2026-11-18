# Writing a brain

A brain is any class with `reset(spec)` and `step(obs) -> Action`. It may import only `c2lab.brain`, the standard
library, numpy and scipy (`tests/test_isolation.py` enforces it). Put it under `brains/` and point a team at it:

```toml
[[team]]
name = "blue"
count = 10
brain = "brains.my_brain:MyBrain"       # the only line that changes between algorithms
brain_params = { speed = 2.0 }          # passed to the constructor
sensors.pose = { profile = "ekf" }
sensors.tof = { arrays = ["front", "left", "right"] }
```

## What a brain gets

- `reset(spec: AgentSpec)`: `agent_id`, `team`, `role`, `seed` (draw all randomness from it), `sensors` (public
  parameters of the sensors that are on), `limits` (`brain_hz`, `packet_bytes`, `memory_bytes`, `vmax`...) and
  `mission` (`goal_local`: the goal in the local frame, ENU axes anchored at the take-off point; `goal_enu` only with
  GNSS on).
- `step(obs: Observation)` at `brain_hz`: `obs.t`, `obs.readings[sensor]` (a tuple of `Reading(t, data)` since the
  last step, oldest first, read-only arrays; formats in `.ia/grafo/_indice.md`) and `obs.inbox` (`Packet(src, data,
  rssi_dbm, range_m)` received since the last step).

## What a brain returns

`Action(setpoint, outbox)`:

- `setpoint`: `Velocity((vx, vy, vz), yaw_rate, frame="local" | "body")`, `Position(dp, yaw)` (displacement from the
  current estimated position) or `None` (hold). A landed drone takes off when the setpoint asks to climb.
- `outbox`: `bytes` to broadcast or `(dest, bytes)` for one receiver; each at most `packet_bytes`, at most
  `max_packets` per radio slot, delivered one slot later.

Optional: `state_bytes()` (persistent memory, reported against `memory_bytes`) and `export_map()` (scored by the
coverage mission). A brain that raises is put in failsafe (it lands) and the run continues.

## Example: go to the goal, climbing first

```python
import numpy as np

from c2lab.brain import Action, Velocity


class GoToGoal:
    def __init__(self, speed=2.0):
        self.speed = speed

    def reset(self, spec):
        self.goal = np.array(spec.mission["goal_local"])

    def step(self, obs):
        pose = obs.readings.get("pose")
        if not pose:
            return Action()
        p = pose[-1].data[:3]                       # local position
        if p[2] < 1.5:
            return Action(Velocity((0.0, 0.0, 1.0)))
        to = self.goal - p
        v = to / max(np.linalg.norm(to), 1.0) * self.speed
        return Action(Velocity(tuple(float(x) for x in v)))
```

Run it with `python -m c2lab run scenarios/<file>.toml --seed 3` and open the printed `/viewer/?run=...` URL
(server: `python -m http.server 8770` from the project root).
