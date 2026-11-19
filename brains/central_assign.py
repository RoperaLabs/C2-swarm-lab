"""Ground-station brain: splits the area into one strip along x per agent and keeps sending each agent its strip,
expressed in that agent's local frame (the operator knows the launch points), as 4 x float16 = 8 bytes unicast.
Re-sends round-robin so losses heal; at most `per_step` packets per step (the radio's per-slot limit)."""
import struct

import numpy as np

from c2lab.brain import Action


class CentralAssign:
    def __init__(self, per_step=4):
        self.per_step = per_step

    def reset(self, spec):
        m = spec.mission
        (lo, hi), homes = m["area_enu"], np.array(m["homes_enu"], float)
        self.agents = m["agents"]
        edges = np.linspace(lo[0] + 1, hi[0] - 1, len(self.agents) + 1)
        self.msgs = [struct.pack("<4e", edges[g] - homes[g][0], lo[1] + 1 - homes[g][1],
                                 edges[g + 1] - homes[g][0], hi[1] - 1 - homes[g][1]) for g in range(len(self.agents))]
        self.k = 0

    def step(self, obs):
        out = []
        for _ in range(min(self.per_step, len(self.agents))):
            out.append((self.agents[self.k], self.msgs[self.k]))
            self.k = (self.k + 1) % len(self.agents)
        return Action(None, tuple(out))
