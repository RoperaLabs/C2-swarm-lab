"""Pure pursuit: fly at the nearest drone of another team the detector sees; when none is in sight, head for the
mission hint (where the targets were last reported). A sighting counts once it repeats in two consecutive steps,
which filters the detector's friend/foe confusions. Each pursuer aims at its own point `ring` metres around the
target (set by its id), so the pack does not converge on one spot and collide. Avoidance from PotentialField."""
import numpy as np

from .common import detections
from .potential_field import PotentialField


class Pursue(PotentialField):
    def __init__(self, speed=3.0, ring=1.2, **kw):
        super().__init__(speed=speed, **kw)
        self.ring = ring

    def reset(self, spec):
        super().reset(spec)
        self.hint = np.array(spec.mission.get("hint_local", (0.0, 0.0, 2.0)), float)
        self.last = None
        a = 2.4 * spec.agent_id                         # golden-ish angle spreads the pack
        self.offset = self.ring * np.array([np.cos(a), np.sin(a), 0.0])
        self.hint = self.hint + 3 * self.offset                # spread the approach to the hint too

    def target(self, p, R, obs):
        rel, dist, cls = detections(obs, R)
        foe = cls == 1
        if foe.any():
            seen = p + rel[foe][np.argmin(dist[foe])]
            if self.last is not None and np.linalg.norm(seen - self.last) < 3.0:   # confirmed in two steps
                self.hint = seen + self.offset
            self.last = seen
        else:
            self.last = None
        return self.hint
