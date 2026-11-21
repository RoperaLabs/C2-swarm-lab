"""`RunClock`: how fast the tick loop runs (plan F8.4).

Modes: `fast` (no pacing; the default, and what sweeps use), `realtime` at a factor of 0.1x to 10x, and `paused`;
`step(n)` runs n more ticks and stays paused. The loop calls `wait(tick)` before every tick, and pacing only ever
sleeps there: it never changes what is simulated, so a run's hash is the same in every mode. The control methods are
safe to call from another thread (a viewer). `stats()` reports like Gazebo's /stats: simulated time, wall time, ticks
and the real-time factor over the last second of wall time.
"""
import threading
import time

from c2lab.scenario import PHYSICS_HZ

MODES = ("fast", "realtime", "paused")
MIN_SLEEP_S = 0.002     # realtime: sleep only when at least this far ahead (fewer, longer sleeps)
MAX_LAG_S = 0.1         # realtime: further behind than this, re-anchor instead of catching up in a burst
WINDOW_S = 1.0          # wall seconds over which the real-time factor is measured


class RunClock:
    def __init__(self, mode="fast", factor=1.0):
        if mode not in MODES:
            raise ValueError(f"RunClock mode {mode!r}: must be one of {MODES}")
        self._cv = threading.Condition()
        self.mode, self.factor = mode, _factor(factor)
        self.tick, self.t0, self.real_factor = -1, None, 0.0
        self._budget, self._anchor, self._win = 0, None, None

    def fast(self):
        self._set("fast")

    def realtime(self, factor=1.0):
        self._set("realtime", _factor(factor))

    def pause(self):
        self._set("paused")

    def step(self, n=1):
        """Run n more ticks, then stay paused."""
        with self._cv:
            if self.mode != "paused":
                self._set("paused")
            self._budget += n
            self._cv.notify_all()

    def _set(self, mode, factor=None):
        with self._cv:
            self.mode, self._budget, self._anchor, self._win = mode, 0, None, None
            if factor is not None:
                self.factor = factor
            self._cv.notify_all()

    def wait(self, tick):
        """Return when tick `tick` may run: at once (fast), when its wall time comes (realtime) or when allowed."""
        if self.mode == "paused":
            with self._cv:
                while self.mode == "paused" and not self._budget:
                    self._cv.wait()
                if self.mode == "paused":
                    self._budget -= 1
        now = time.perf_counter()
        if self.mode == "realtime":
            a = self._anchor
            if a is None or now - (a[0] + (tick - a[1]) / (PHYSICS_HZ * self.factor)) > MAX_LAG_S:
                a = self._anchor = (now, tick)
            ahead = a[0] + (tick - a[1]) / (PHYSICS_HZ * self.factor) - now
            if ahead > MIN_SLEEP_S:
                time.sleep(ahead)
                now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        if self._win is None:
            self._win = (now, tick)
        elif now - self._win[0] >= WINDOW_S:
            self.real_factor = (tick - self._win[1]) / PHYSICS_HZ / (now - self._win[0])
            self._win = (now, tick)
        self.tick = tick

    def stats(self):
        ticks = self.tick + 1
        idle = self.mode == "paused" and not self._budget
        return {"mode": self.mode, "factor": self.factor, "ticks": ticks, "sim_s": ticks / PHYSICS_HZ,
                "wall_s": 0.0 if self.t0 is None else time.perf_counter() - self.t0,
                "real_factor": 0.0 if idle else self.real_factor}


def _factor(f):
    if not 0.1 <= f <= 10:
        raise ValueError(f"realtime factor {f}: must be within 0.1 and 10")
    return float(f)
