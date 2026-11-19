"""Coverage worker for a C2 ground station: waits for its sector (x0, y0, x1, y1 in its local frame, 4 x float16 =
8 bytes, sent to it by the station) and sweeps it in lanes `lane` metres apart at cruise altitude. Holds until the
sector arrives. Avoidance from PotentialField."""
import struct

import numpy as np

from .potential_field import PotentialField


class SectorSweep(PotentialField):
    def __init__(self, speed=2.0, lane=8.0, alt=3.0, reach=2.0, **kw):
        super().__init__(speed=speed, **kw)
        self.lane, self.alt, self.reach = lane, alt, reach

    def reset(self, spec):
        super().reset(spec)
        self.route, self.k = None, 0

    def target(self, p, R, obs):
        for pk in obs.inbox:
            if self.route is None and len(pk.data) == 8:
                x0, y0, x1, y1 = struct.unpack("<4e", pk.data)
                ys = np.arange(min(y0, y1) + self.lane / 2, max(y0, y1), self.lane)
                xs = (min(x0, x1) + 1, max(x0, x1) - 1)
                self.route = [np.array([xs[(j + s) % 2], y, self.alt]) for j, y in enumerate(ys) for s in (0, 1)]
        if not self.route:
            return p
        while self.k < len(self.route) - 1 and np.linalg.norm((self.route[self.k] - p)[:2]) < self.reach:
            self.k += 1
        return self.route[self.k]
