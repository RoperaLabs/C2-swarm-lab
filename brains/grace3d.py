"""GRACE-3D 2.0.0 (swarm-dil sim3d.agent, imported unchanged) flown from sensors, as a drone's firmware would host it.

The adapter owns a virtual lattice of `cell` metres anchored on the mission area (area_local), so every drone of the
team shares it; A (a_local) and B (goal_local) give GRACE's start and goal voxels. One GRACE slot is one move:

    moved(previous move) -> observe(ToF hits of the slot) -> transmit -> receive(bytes heard) -> decide -> fly

`observe` gets the voxels holding the ToF returns gathered during the slot (relative to GRACE's own voxel), the radio
passes GRACE's bytes unchanged, and a decided step is flown centre to centre with velocity setpoints. A slot ends when
the drone reaches the target centre, or after `cell / speed` seconds when it holds. After `t_max` slots (swarm-dil's
mission horizon) a drone that has not arrived holds where it is. Deviations from swarm-dil's
discrete world, each a ladder factor:
- a diagonal step whose straight line grazes a known blocked voxel (this slot's hits or GRACE's tiles) is flown as
  axis legs through free voxels of the same unit cube: a body of finite size cannot pass a zero-width corner. It is
  refused (moved(d, False), no motion, so GRACE takes the target for blocked) when the target is known blocked or
  no such legs exist;
- a step is abandoned and flown back (moved(d, False)) when a ToF return comes within `margin` of the remaining
  segment or the move times out;
- pre-launch slots are held on the pad; at launch the drone first climbs to its start voxel;
- view="swarm_dil" keeps only the returns inside swarm-dil's two sensor frusta (60 deg square, one along GRACE's
  heading, one along a perpendicular that rotates with the slot); view="all" keeps every return of the ToF arrays;
- a return goes to the voxel beyond the lattice face nearest to it, on the ray's side;
- returns within GROUND of the area floor are taken for the ground and dropped (the lattice assumes flat ground);
- each drone keeps its own station inside a voxel (a ring of RING cell, alternate layers 0.12 cell above and below
  the centre), so teammates sharing a voxel do not collide.
"""
import math
from itertools import permutations

import numpy as np

from c2lab.brain import Action, Velocity
from sim3d.agent import Agent, Spec
from sim3d.geom import STEPS, blocked_in

from .common import rot, zone_dirs

GROUND = 0.1       # of a cell: returns this close to the area floor are the ground
RING = 0.18        # of a cell: radius of the ring of per-drone stations around a voxel centre
OCC_MIN, OCC_RATIO = 4, 3   # noisy ToF: a voxel is blocked after 4 returns and three times as many as rays through it


class Grace3D:
    def __init__(self, team, cell=5.0, speed=4.0, tol=0.5, margin=0.6, gain=1.2, launch_every=10, t_max=500, view="all",
                 **grace):
        self.team, self.cell, self.speed, self.tol, self.margin, self.gain = team, cell, speed, tol, margin, gain
        self.launch_every, self.t_max, self.view, self.grace = launch_every, t_max, view, grace

    def reset(self, spec):
        m, c = spec.mission, self.cell
        lo, hi = (np.array(x, float) for x in m["area_local"])
        self.lo = lo
        self.shape = tuple(int(round(x)) for x in (hi - lo) / c)
        start = self._vox(np.array(m["a_local"], float))
        start = (start[0], start[1], 0)
        goal = self._vox(np.array(m["goal_local"], float))
        tof = spec.sensors["tof"]
        self.dirs = zone_dirs(tof)
        self.range = tof["range_m"]
        self.sig_rel, self.sig_min = tof.get("sigma_rel", 0.0), tof.get("sigma_min", 0.0)
        sub = tof.get("sub", 1)
        self.ang = math.tan(math.radians(tof["fov_deg"] / tof["zones"] / 2 * (1 - 1 / sub)))   # zone return off centre
        lim = spec.limits
        self.vxy = min(self.speed, lim["vmax"])
        self.vup, self.vdown = lim["vmax_up"], lim["vmax_down"]
        ident = spec.agent_id % self.team
        th = 2 * math.pi * ident / self.team
        self.off = c * np.array([RING * math.cos(th), RING * math.sin(th), 0.12 if ident % 2 else -0.12])
        self.g = Agent(ident, Spec(shape=self.shape, start=start, goal=goal, team=self.team,
                                   launch_at=ident * self.launch_every, band=(0, self.shape[2] - 1),
                                   budget=lim["packet_bytes"], r=self.range / c, seed=spec.seed % 2 ** 31,
                                   **self.grace))
        self.hold_s = c / self.vxy
        self.seen, self.hits, self.inbox, self.pts = set(), set(), [], np.empty((0, 3))
        self.occ, self.free = np.zeros(self.shape, np.int32), np.zeros(self.shape, np.int32)
        self.phase, self.until, self.target, self.pending, self.airborne = "hold", None, None, None, False
        self.route, self.flown = [], []
        self.slots = 0

    def state_bytes(self):
        return self.g.memory_used

    # ------------------------------------------------------------------ lattice
    def _vox(self, x):
        return tuple(int(v) for v in np.floor((x - self.lo) / self.cell))

    def _station(self, v):
        return self.lo + (np.array(v, float) + 0.5) * self.cell + self.off

    def _scan(self, obs, poses):
        """ToF returns of this step, each placed with the pose sample nearest in time -> local points (kept for the
        flight guard) and voxels (gathered for the slot)."""
        ts = np.array([q.t for q in poses])
        pts, rel = [], []
        for r in obs.readings.get("tof", ()):
            x = poses[int(np.abs(ts - r.t).argmin())].data
            d = r.data
            ok = np.isfinite(d) & (d < self.range)
            v = (self.dirs[ok] * d[ok][:, None]) @ rot(x[6:10]).T
            pts.append(x[:3] + v)
            rel.append(v)
        if not pts:
            return
        self.pts, rel = np.concatenate(pts), np.concatenate(rel)
        if not len(rel):
            return
        dist = np.linalg.norm(rel, axis=1)
        err = 3 * np.maximum(self.sig_min, self.sig_rel * dist) + self.ang * dist     # 3-sigma range + zone width
        keep = self._in_view(rel) & (self.pts[:, 2] > self.lo[2] + np.maximum(GROUND * self.cell, err))
        shape = np.array(self.shape)
        if err.max() < 1e-9:                                  # exact returns: the voxel beyond the nearest face
            u = (self.pts - self.lo) / self.cell
            v = np.floor(u).astype(int)
            k = np.abs(u - np.rint(u)).argmin(1)
            i = np.arange(len(u))
            v[i, k] = np.rint(u[i, k]).astype(int) - (rel[i, k] < 0)
            ok = keep & ((v >= 0) & (v < shape)).all(1)
            self.seen.update(map(tuple, v[ok].tolist()))
            return
        # uncertain returns: occupancy evidence. The return, pushed `err` beyond along its ray, votes for its voxel;
        # the ray up to `err` short of it votes free for every voxel it crosses.
        unit = rel / dist[:, None]
        v = np.floor((self.pts + unit * err[:, None] - self.lo) / self.cell).astype(int)
        ok = keep & ((v >= 0) & (v < shape)).all(1)
        np.add.at(self.occ, tuple(v[ok].T), 1)
        s = np.arange(1, int(self.range / (0.25 * self.cell)) + 1) * 0.25 * self.cell
        o = self.pts - rel
        f = np.floor((o[:, None] + unit[:, None] * s[None, :, None] - self.lo) / self.cell).astype(int)
        m = (s[None] < (dist - err)[:, None]) & ((f >= 0) & (f < shape)).all(-1)
        flat = np.ravel_multi_index(tuple(f[m].T), self.shape)
        ray = np.nonzero(m)[0]
        np.add.at(self.free.reshape(-1), np.unique(ray * self.free.size + flat) % self.free.size, 1)
        c = v[ok]
        sure = (self.occ[tuple(c.T)] >= OCC_MIN) & (self.occ[tuple(c.T)] > OCC_RATIO * self.free[tuple(c.T)])
        self.seen.update(map(tuple, c[sure].tolist()))

    def _in_view(self, rel):
        """Mask of the returns a swarm-dil sensor head would see (sim3d.world.zone_rays geometry)."""
        if self.view == "all":
            return np.ones(len(rel), bool)
        axis = np.array(STEPS[self.g.heading], float)
        axis /= np.linalg.norm(axis)
        e1 = np.cross(axis, (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        side, other = ((e1, e2), (-e1, e2), (e2, e1), (-e2, e1))[self.g.t % 4]
        t = math.tan(math.radians(30.0))
        out = np.zeros(len(rel), bool)
        for a, b, c in ((axis, e1, e2), (side, axis, other)):
            x = rel @ a
            out |= (x > 0) & (np.abs(rel @ b) <= t * x) & (np.abs(rel @ c) <= t * x)
        return out

    def _known(self, c):
        return c in self.hits or blocked_in(self.g.tiles, c, None)

    def _legs(self, pos, d):
        """Voxels to fly through for step d (ending at its target), or None if it cannot be flown."""
        step = STEPS[d]
        target = tuple(pos[k] + step[k] for k in range(3))
        if not self.g._inside(target) or self._known(target):
            return None
        subs = {tuple(step[k] if (mask >> k) & 1 else 0 for k in range(3)) for mask in range(1, 8)}
        if not any(self._known((pos[0] + e[0], pos[1] + e[1], pos[2] + e[2])) for e in subs):
            return [target]
        for order in permutations([k for k in range(3) if step[k]]):
            cells, c = [], list(pos)
            for k in order:
                c[k] += step[k]
                cells.append(tuple(c))
            if not any(self._known(x) for x in cells):
                return cells
        return None

    def _timeout(self, d):
        dx, dy, dz = (abs(k) * self.cell for k in STEPS[d])
        vz = self.vup if STEPS[d][2] > 0 else self.vdown
        return 3 * (max(math.hypot(dx, dy) / self.vxy, dz / vz if dz else 0.0) + 2.0)

    # ------------------------------------------------------------------ slot
    def _slot(self, t):
        a = self.g
        if self.pending is not None:
            a.moved(*self.pending)
            self.pending = None
        if not self.airborne and not a.arrived and a.t >= a.s.launch_at:
            self.airborne, self.phase, self.target = True, "fly", self._station(a.pos)
            self.until = math.inf
            return ()
        self.slots += 1
        self.hits, self.seen = self.seen, set()
        a.observe([(c[0] - a.pos[0], c[1] - a.pos[1], c[2] - a.pos[2]) for c in self.hits])
        pkt = a.transmit()
        for data in self.inbox:
            a.receive(data)
        self.inbox = []
        d = a.decide()
        legs = None if d is None else self._legs(a.pos, d)
        if legs:
            self.route = [self._station(c) for c in legs]
            self.flown = [self._station(a.pos)]
            self.target = self.route.pop(0)
            self.phase, self.until, self.pending = "fly", t + self._timeout(d) * len(legs), (d, True)
        else:
            self.phase, self.until, self.pending = "hold", t + self.hold_s, (d, False)
            if self.airborne:
                self.target = self._station(a.pos)
        return (pkt,) if pkt else ()

    def _guard(self, p):
        """A ToF return within `margin` of the rest of the flown segment."""
        if not len(self.pts):
            return False
        seg = self.target - p
        L2 = float(seg @ seg)
        if L2 < 1e-6:
            return False
        s = np.clip((self.pts - p) @ seg / L2, 0.0, 1.0)
        return bool((np.linalg.norm(self.pts - p - s[:, None] * seg, axis=1) < self.margin).any())

    def step(self, obs):
        poses = obs.readings.get("pose")
        if not poses:
            return Action()
        p = poses[-1].data[:3]
        self.pts = np.empty((0, 3))
        self._scan(obs, poses)
        self.inbox += [pk.data for pk in obs.inbox]
        t = obs.t
        if self.until is None:
            self.until = t
        out = ()
        moving = self.pending is not None and self.pending[1] and self.phase == "fly"
        if self.phase == "fly" and np.linalg.norm(self.target - p) < self.tol:
            if self.route:
                self.flown.append(self.target)
                self.target = self.route.pop(0)
            else:
                out = self._slot(t)
        elif self.phase == "hold" and t >= self.until and self.g.t < self.t_max:
            out = self._slot(t)
        elif moving and (t > self.until or self._guard(p)):
            self.pending = (self.pending[0], False)                    # abandoned: fly back to GRACE's voxel
            self.route, self.until = self.flown[::-1], math.inf
            self.target = self.route.pop(0)
        if self.target is None:
            return Action(None, out)
        e = self.gain * (self.target - p)
        h = math.hypot(e[0], e[1])
        if h > self.vxy:
            e[:2] *= self.vxy / h
        e[2] = min(max(e[2], -self.vdown), self.vup)
        return Action(Velocity(tuple(e)), out)
