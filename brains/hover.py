"""Does nothing: holds position (and stays on the ground if it starts there)."""
from c2lab.brain import Action


class Hover:
    def reset(self, spec):
        pass

    def step(self, obs):
        return Action()
