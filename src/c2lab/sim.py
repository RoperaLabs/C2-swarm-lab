"""Deterministic multi-rate loop (PLAN.md §2.3). In each physics tick: brains (at their rate) -> radio slot ->
physics with the estimator's errors -> sensors -> mission -> recording. Brains see only what `Sensors` and `Radio`
hand them; a brain that raises goes to failsafe (land) and the run continues.

Mission for now: A -> B (F5 generalises it). A drone arrives when it is within GOAL_R of B + GOAL_ALT; it then lands. Drone i of a team is
launched (its brain starts) at i * launch_s.
ponytail: drones of different teams do not collide with each other (each team has its own Drones); one Drones for all
when a scenario mixes teams in close quarters.
"""
import hashlib
import importlib
import time

import numpy as np

from c2lab.brain import AgentSpec, Observation, Velocity
from c2lab.physics import PROFILES, Drones
from c2lab.radio import Radio
from c2lab.sensors import Sensors
from c2lab.seeds import rng as stream
from c2lab.world import generate, oracle
from c2lab.world.biomes import CLEAR

GOAL_ALT, GOAL_R = 2.0, 2.0      # m
SPACING = 2.0                    # m between spawn points
GUIDE_INFLATE = 0.9              # m, clearance of the path handed to privileged brains (tracking margin)


def load_brain(path, params):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)(**params)


def spawn_points(world, n, team, clear):
    """n ground points on a grid centred on A, kept inside A's free zone (team k shifted by k rows)."""
    a = np.array(world.anchors["A"], float)
    side = int(np.ceil(np.sqrt(n)))
    rows = side * (team + 1)
    span = max(side - 1, rows - 1, 1) / 2 * np.sqrt(2)
    step = min(SPACING, (clear - 1.0) / span)
    k = np.arange(n)
    xy = a[:2] + step * (np.c_[k % side, k // side + team * side] - (side - 1) / 2 - [0, team * side / 2])
    return np.c_[xy, world.ground(xy)]


def run(sc, world=None, record=False):
    s, rates = sc["scenario"], sc["scenario"]["rates_hz"]
    seed, phys = s["seed"], rates["physics"]
    w = world or generate(sc["world"]["biome"], sc["world"]["size_m"], sc["world"]["density"], seed)
    goal = np.array(w.anchors["B"], float) + (0, 0, GOAL_ALT)
    teams, g0, oracle_len = [], 0, None
    for k, t in enumerate(sc["team"]):
        n = t["count"]
        d = Drones(t["drone"], n, stream(seed, "team", k, "wind"), w)
        d.reset(spawn_points(w, n, k, CLEAR[sc["world"]["biome"]]), np.zeros(n), landed=True)
        home = d.p.copy()
        sens = Sensors(t["sensors"], n, seed, range(g0, g0 + n), w, d, np.full(n, k), home, t["drone"])
        brains = [load_brain(t["brain"], t["brain_params"]) for _ in range(n)]
        priv = bool(getattr(brains[0], "privileged", False))
        if priv:                                             # ground truth, only for privileged brains
            res = oracle.RES.get(sc["world"]["biome"], 0.5)
            paths = oracle.shortest_paths(w, home + (0, 0, GOAL_ALT), goal, res)
            oracle_len = float(np.mean([length for _, length in paths]))
            wide = oracle.shortest_paths(w, home + (0, 0, GOAL_ALT), goal, res, GUIDE_INFLATE)
            paths = [wd if wd[0] is not None else nr for wd, nr in zip(wide, paths)]   # guide with more margin
        prof = PROFILES[t["drone"]]
        for i, b in enumerate(brains):
            mission = {"goal_local": tuple(goal - home[i])}
            if "gnss" in t["sensors"]:
                mission["goal_enu"] = tuple(goal)
            if priv:
                way = paths[i][0] if paths[i][0] is not None else [home[i] + (0, 0, GOAL_ALT)]   # no path: hover
                mission["oracle_path"] = [tuple(p - home[i]) for p in way]
            b.reset(AgentSpec(
                agent_id=g0 + i, team=t["name"], role=t["role"],
                seed=int(stream(seed, "agent", g0 + i, "brain").integers(2 ** 62)),
                sensors={name: sens.public(name) for name in t["sensors"]},
                limits={"brain_hz": rates["brain"], "packet_bytes": sc["radio"]["packet_bytes"],
                        "memory_bytes": t["memory_bytes"], "vmax": prof.vmax_xy, "vmax_up": prof.vmax_up,
                        "vmax_down": prof.vmax_down},
                mission=mission))
        teams.append({"g0": g0, "n": n, "drones": d, "sensors": sens, "brains": brains, "privileged": priv,
                      "launch": np.arange(n) * t["launch_s"]})          # drone i waits launch_s * i on the ground
        g0 += n
    N = g0
    radio = Radio(sc["radio"], N, stream(seed, "radio"), w, 1 / rates["radio"])
    kb, kr, krec = (phys // rates[x] for x in ("brain", "radio", "record"))
    inbox = [[] for _ in range(N)]
    arrived, failed = np.full(N, np.nan), np.zeros(N, bool)
    flown, last = np.zeros(N), np.concatenate([t["drones"].p for t in teams]).copy()
    cpu, steps, mem = np.zeros(N), np.zeros(N, int), np.zeros(N, int)
    events, frames = [], []

    def all_of(attr):
        return np.concatenate([getattr(t["drones"], attr) for t in teams])

    for tick in range(int(s["duration_s"] * phys)):
        tsec = tick / phys
        if tick % kb == 0:
            for t in teams:
                d = t["drones"]
                for i, b in enumerate(t["brains"]):
                    g = t["g0"] + i
                    if not d.alive[i] or failed[g] or not np.isnan(arrived[g]) or tsec < t["launch"][i]:
                        continue
                    obs = Observation(tsec, t["sensors"].pop(i), tuple(inbox[g]))
                    inbox[g] = []
                    c0 = time.perf_counter()
                    try:
                        act = b.step(obs)
                    except Exception as e:                     # brain fault: failsafe landing, run goes on
                        failed[g] = True
                        events.append((tick, g, "brain_error", repr(e)[:200]))
                        d.set_setpoint(i, Velocity((0.0, 0.0, -0.5)))
                        continue
                    cpu[g] += time.perf_counter() - c0
                    steps[g] += 1
                    if hasattr(b, "state_bytes"):
                        mem[g] = max(mem[g], int(b.state_bytes()))
                    d.set_setpoint(i, act.setpoint)
                    radio.send(g, act.outbox)
            p = all_of("p")
            flown += np.linalg.norm(p - last, axis=1) * np.isnan(arrived)
            last = p.copy()
            hit = np.flatnonzero(np.isnan(arrived) & (np.linalg.norm(p - goal, axis=1) < GOAL_R) & all_of("alive"))
            for g in hit:
                arrived[g] = tsec
                events.append((tick, int(g), "arrived", ""))
                t = next(t for t in teams if t["g0"] <= g < t["g0"] + t["n"])
                t["drones"].set_setpoint(int(g - t["g0"]), Velocity((0.0, 0.0, -0.5)))    # land at B
        if tick % kr == 0:
            for g, box in enumerate(radio.slot(all_of("p"), all_of("alive"))):
                inbox[g] += box
        for t in teams:
            for tk, i, kind in t["drones"].step(tick, t["sensors"].err):
                events.append((tk, t["g0"] + i, kind, ""))
            t["sensors"].step(tick)
        if record and tick % krec == 0:
            frames.append((tsec, all_of("p").copy(), all_of("q").copy(), all_of("alive").copy()))
        alive = all_of("alive")
        if not (alive & np.isnan(arrived) & ~failed).any() and not (alive & ~all_of("landed")).any():
            break

    ok = ~np.isnan(arrived)
    metrics = {
        "agents": N, "privileged": any(t["privileged"] for t in teams), "seconds": tsec,
        "arrival_rate": float(ok.mean()), "time_mean": float(np.nanmean(arrived)) if ok.any() else None,
        "flown_mean": float(flown[ok].mean()) if ok.any() else None, "oracle_length": oracle_len,
        "crashes": sum(e[2].startswith("crash") or e[2] == "geofence" for e in events),
        "brain_errors": int(failed.sum()), "bytes_sent": int(radio.sent.sum()),
        "bytes_delivered": int(radio.delivered.sum()), "bytes_dropped": int(radio.dropped.sum()),
        "brain_ms_mean": float(cpu.sum() / max(steps.sum(), 1) * 1e3), "state_bytes_max": int(mem.max()),
    }
    out = {"metrics": metrics, "events": events, "world": w}
    if record:
        out["frames"] = frames
        out["hash"] = hashlib.sha256(b"".join(f[1].tobytes() + f[2].tobytes() for f in frames)).hexdigest()
    return out
