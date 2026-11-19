"""Missions: where teams start, what each brain is told, what ends an agent's job and how the run is scored. The
information handed to brains respects the golden rule: goals in each drone's local frame (ENU axes, origin at its
take-off point), ENU only for the ground station or with GNSS on.

Contract (used by sim.py):
    spawn(team) -> anchor name;  info(g, home, gnss) -> dict;  station_info() -> dict
    start(homes, privileged)   once, after spawning
    update(tick, t, p, alive, hits) -> [(g, kind)] at brain rate; sets done[g] when agent g's job is over (it lands)
    finished() -> bool;  metrics() -> dict
"""
import math

import numpy as np

from c2lab.world import oracle

GOAL_ALT, GOAL_R = 2.0, 2.0                  # m: goal 2 m above B, reached within 2 m
PATH_RES = {"mixto": 0.5, "urbano": 2.0, "bosque": 1.0, "interior": 0.25}   # oracle grid for the shortest paths
GUIDE_INFLATE = 0.9                          # m, clearance of the path handed to privileged brains
CAPTURE_R, CAPTURE_T = 2.0, 0.5              # m, s: a pursuer this close for this long captures an evader
COVER_RES = 1.0                              # m, surface voxels for the coverage score


class Mission:
    def __init__(self, sc, world, team_of, roles):
        self.sc, self.w = sc, world
        self.team_of, self.roles = np.asarray(team_of), list(roles)    # per agent
        self.N = len(self.team_of)
        self.done = np.zeros(self.N, bool)
        self.biome = sc["world"]["biome"]

    def spawn(self, team):
        return "A"

    def info(self, g, home, gnss):
        return {}

    def station_info(self):
        return {}

    def start(self, homes, privileged):
        self.homes = np.asarray(homes, float)

    def update(self, tick, t, p, alive, hits):
        return []

    def finished(self):
        return bool(self.done.all())

    def metrics(self):
        return {}


class AToB(Mission):
    def start(self, homes, privileged):
        super().start(homes, privileged)
        self.goal = np.array(self.w.anchors["B"], float) + (0, 0, GOAL_ALT)
        res = PATH_RES.get(self.biome, 0.5)
        starts = self.homes + (0, 0, GOAL_ALT)
        narrow = oracle.shortest_paths(self.w, starts, self.goal, res)
        self.short = np.array([length for _, length in narrow])
        self.guide = None
        if any(privileged):
            wide = oracle.shortest_paths(self.w, starts, self.goal, res, GUIDE_INFLATE)
            self.guide = [wd[0] if wd[0] is not None else nr[0] for wd, nr in zip(wide, narrow)]
        self.arrived = np.full(self.N, np.nan)
        self.flown, self.last = np.zeros(self.N), self.homes.copy()

    def info(self, g, home, gnss):
        out = {"goal_local": tuple(self.goal - home)}
        if gnss:
            out["goal_enu"] = tuple(self.goal)
        if self.guide is not None:
            way = self.guide[g] if self.guide[g] is not None else [home + (0, 0, GOAL_ALT)]   # no path: hover
            out["oracle_path"] = [tuple(q - home) for q in way]
        return out

    def update(self, tick, t, p, alive, hits):
        self.flown += np.linalg.norm(p - self.last, axis=1) * ~self.done
        self.last = p.copy()
        hit = np.flatnonzero(~self.done & alive & (np.linalg.norm(p - self.goal, axis=1) < GOAL_R))
        self.arrived[hit], self.done[hit] = t, True
        return [(int(g), "arrived") for g in hit]

    def metrics(self):
        ok = ~np.isnan(self.arrived)
        regret = self.flown[ok] / self.short[ok] - 1
        return {"arrival_rate": float(ok.mean()),
                "time_mean": float(self.arrived[ok].mean()) if ok.any() else None,
                "flown_mean": float(self.flown[ok].mean()) if ok.any() else None,
                "oracle_length": float(np.mean(self.short[np.isfinite(self.short)])),
                "regret_mean": float(regret.mean()) if ok.any() else None}


class Coverage(Mission):
    """Score: share of obstacle-surface voxels touched by any ray of any agent's ray sensors (tof, lidar, depth)."""

    def start(self, homes, privileged):
        super().start(homes, privileged)
        w, r = self.w, COVER_RES
        lo, hi = w.bounds
        self.lo, self.shape = lo, tuple(np.maximum(((hi - lo) / r).astype(int), 1))
        axes = [lo[a] + r * (np.arange(self.shape[a]) + 0.5) for a in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        occ = (w.sdf(g, max_dist=r) < r * math.sqrt(3) / 2).reshape(self.shape)
        free = ~occ
        near_free = np.zeros_like(occ)
        for a in range(3):
            for s in (1, -1):
                near_free |= np.roll(free, s, axis=a)
        self.surface = occ & near_free
        self.seen = np.zeros_like(occ)
        self.curve = []

    def info(self, g, home, gnss):
        lo, hi = self.w.bounds
        out = {"area_local": (tuple(lo - home), tuple(hi - home))}
        if gnss:
            out["area_enu"] = (tuple(lo), tuple(hi))
        return out

    def station_info(self):
        lo, hi = self.w.bounds
        return {"area_enu": (tuple(lo), tuple(hi)), "homes_enu": [tuple(h) for h in self.homes]}

    def update(self, tick, t, p, alive, hits):
        if hits:
            pts = np.concatenate(hits)
            idx = np.floor((pts - self.lo) / COVER_RES).astype(int)
            ok = ((idx >= 0) & (idx < self.shape)).all(1)
            self.seen[tuple(idx[ok].T)] = True
            hits.clear()
        self.curve.append((t, self.coverage()))
        return []

    def coverage(self):
        return float((self.seen & self.surface).sum() / max(self.surface.sum(), 1))

    def finished(self):
        return False

    def metrics(self):
        def first(level):
            return next((t for t, c in self.curve if c >= level), None)
        return {"coverage": self.coverage(), "t50": first(0.5), "t80": first(0.8),
                "surface_voxels": int(self.surface.sum())}


class Pursuit(Mission):
    """Evaders (role "evader", spawned at B) are captured when a pursuer stays within CAPTURE_R for CAPTURE_T."""

    def spawn(self, team):
        return "B" if self.sc["team"][team]["role"] == "evader" else "A"

    def start(self, homes, privileged):
        super().start(homes, privileged)
        self.ev = np.array([r == "evader" for r in self.roles])
        self.close = np.zeros(self.N)
        self.caught = np.full(self.N, np.nan)
        self.min_d = np.full(self.N, np.inf)

    def info(self, g, home, gnss):
        other = "A" if self.ev[g] else "B"
        lo, hi = self.w.bounds
        return {"hint_local": tuple(np.array(self.w.anchors[other], float) + (0, 0, GOAL_ALT) - home),
                "area_local": (tuple(lo - home), tuple(hi - home))}

    def update(self, tick, t, p, alive, hits):
        dt = 1 / self.sc["scenario"]["rates_hz"]["brain"]
        hunters = ~self.ev & alive
        out = []
        for e in np.flatnonzero(self.ev & alive & ~self.done):
            d = np.linalg.norm(p[hunters] - p[e], axis=1).min() if hunters.any() else np.inf
            self.min_d[e] = min(self.min_d[e], d)
            self.close[e] = self.close[e] + dt if d < CAPTURE_R else 0.0
            if self.close[e] >= CAPTURE_T - 1e-9:
                self.caught[e], self.done[e] = t, True
                out.append((int(e), "captured"))
        return out

    def finished(self):
        return bool(self.done[self.ev].all())

    def metrics(self):
        ok = ~np.isnan(self.caught[self.ev])
        return {"capture_rate": float(ok.mean()),
                "capture_time_mean": float(self.caught[self.ev][ok].mean()) if ok.any() else None,
                "min_distance_mean": float(np.mean(self.min_d[self.ev]))}


MISSIONS = {"a_to_b": AToB, "coverage": Coverage, "pursuit": Pursuit}
