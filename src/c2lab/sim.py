"""Deterministic multi-rate loop (PLAN.md §2.3). In each physics tick: brains (at their rate) -> radio slot ->
physics with the estimator's errors -> sensors -> mission -> recording. Brains see only what `Sensors` and `Radio`
hand them; a brain that raises goes to failsafe (land) and the run continues. Drone i of a team is launched (its
brain starts) at i * launch_s. An optional ground station is one more radio node with its own brain and no body.

ponytail: drones of different teams do not collide with each other (each team has its own Drones); one Drones for all
when a scenario mixes teams in close quarters.
"""
import hashlib
import importlib
import time

import numpy as np

from c2lab.brain import AgentSpec, Observation, Velocity
from c2lab.missions import MISSIONS
from c2lab.physics import PROFILES, Drones
from c2lab.radio import Radio
from c2lab.sensors import Sensors
from c2lab.seeds import rng as stream
from c2lab.world import generate
from c2lab.world.biomes import CLEAR

SPACING = 2.0          # m between spawn points
GS_GAIN_DB = 12.0      # ground-station antenna gain over a drone antenna (swarm-dil v0.2)
LAND = Velocity((0.0, 0.0, -0.5))
LINK_EVERY = 5         # record the delivered radio links every 5th slot (2 Hz at 10 Hz slots)


def load_brain(path, params):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)(**params)


def spawn_points(world, n, row, clear, anchor="A"):
    """n ground points on a grid centred on an anchor, inside its free zone (row block `row` for the k-th team there)."""
    a = np.array(world.anchors[anchor], float)
    side = int(np.ceil(np.sqrt(n)))
    rows = side * (row + 1)
    span = max(side - 1, rows - 1, 1) / 2 * np.sqrt(2)
    step = min(SPACING, (clear - 1.0) / span)
    k = np.arange(n)
    xy = a[:2] + step * (np.c_[k % side, k // side + row * side] - (side - 1) / 2 - [0, row * side / 2])
    return np.c_[xy, world.ground(xy)]


def run(sc, world=None, record=False):
    s, rates = sc["scenario"], sc["scenario"]["rates_hz"]
    seed, phys = s["seed"], rates["physics"]
    w = world or generate(sc["world"]["biome"], sc["world"]["size_m"], sc["world"]["density"], seed)
    team_of = [k for k, t in enumerate(sc["team"]) for _ in range(t["count"])]
    roles = [t["role"] for t in sc["team"] for _ in range(t["count"])]
    mission = MISSIONS[s["mission"]](sc, w, team_of, roles)
    teams, g0, per_anchor = [], 0, {}
    for k, t in enumerate(sc["team"]):
        n, anchor = t["count"], mission.spawn(k)
        row = per_anchor.get(anchor, 0)
        per_anchor[anchor] = row + 1
        d = Drones(t["drone"], n, stream(seed, "team", k, "wind"), w)
        d.reset(spawn_points(w, n, row, CLEAR[sc["world"]["biome"]], anchor), np.zeros(n), landed=True)
        sens = Sensors(t["sensors"], n, seed, range(g0, g0 + n), w, d, np.full(n, k), d.p.copy(), t["drone"])
        brains = [load_brain(t["brain"], t["brain_params"]) for _ in range(n)]
        teams.append({"g0": g0, "n": n, "cfg": t, "drones": d, "sensors": sens, "brains": brains,
                      "privileged": bool(getattr(brains[0], "privileged", False)),
                      "launch": np.arange(n) * t["launch_s"]})          # drone i waits launch_s * i on the ground
        g0 += n
    N = g0
    homes = np.concatenate([t["drones"].p for t in teams])
    labels = np.array(team_of)

    def peers():
        return (np.concatenate([t["drones"].p for t in teams]),
                np.concatenate([t["drones"].alive for t in teams]), labels)

    for t in teams:                                     # every team's sensors see every drone
        t["sensors"].peers, t["sensors"].g0 = peers, t["g0"]
    mission.start(homes, [t["privileged"] for t in teams for _ in range(t["n"])])
    hits = [] if getattr(mission, "surface", None) is not None else None
    for t in teams:
        if hits is not None:
            t["sensors"].hits = hits                                    # coverage scores the sensor rays
        prof = PROFILES[t["cfg"]["drone"]]
        for i, b in enumerate(t["brains"]):
            g = t["g0"] + i
            b.reset(AgentSpec(
                agent_id=g, team=t["cfg"]["name"], role=t["cfg"]["role"],
                seed=int(stream(seed, "agent", g, "brain").integers(2 ** 62)),
                sensors={name: t["sensors"].public(name) for name in t["cfg"]["sensors"]},
                limits={"brain_hz": rates["brain"], "packet_bytes": sc["radio"]["packet_bytes"],
                        "memory_bytes": t["cfg"]["memory_bytes"], "vmax": prof.vmax_xy, "vmax_up": prof.vmax_up,
                        "vmax_down": prof.vmax_down},
                mission=mission.info(g, homes[g], "gnss" in t["cfg"]["sensors"])))
    gs = sc.get("ground_station")
    station = None
    if gs:
        station = load_brain(gs["brain"], gs["brain_params"])
        station.reset(AgentSpec(agent_id=N, team="ground", role="ground", seed=int(stream(seed, "ground").integers(2 ** 62)),
                                sensors={}, limits={"brain_hz": rates["brain"], "packet_bytes": sc["radio"]["packet_bytes"]},
                                mission={**mission.station_info(), "agents": list(range(N))}))
    gs_pos = np.array(w.anchors["A"], float) + (0, 0, 2.0)
    nodes = N + (station is not None)
    gains = np.r_[np.zeros(N), [GS_GAIN_DB]] if station is not None else None
    radio = Radio(sc["radio"], nodes, stream(seed, "radio"), w, 1 / rates["radio"], gains)
    kb, kr, krec = (phys // rates[x] for x in ("brain", "radio", "record"))
    inbox = [[] for _ in range(nodes)]
    failed = np.zeros(N, bool)
    cpu, steps, mem = np.zeros(N), np.zeros(N, int), np.zeros(N, int)
    events, frames, links = [], [], []

    def all_of(attr):
        return np.concatenate([getattr(t["drones"], attr) for t in teams])

    def node_pos():
        p = all_of("p")
        return np.vstack([p, gs_pos]) if station is not None else p

    def node_alive():
        a = all_of("alive")
        return np.r_[a, True] if station is not None else a

    for tick in range(int(s["duration_s"] * phys)):
        tsec = tick / phys
        if tick % kb == 0:
            for t in teams:
                d = t["drones"]
                for i, b in enumerate(t["brains"]):
                    g = t["g0"] + i
                    if not d.alive[i] or failed[g] or mission.done[g] or tsec < t["launch"][i]:
                        continue
                    obs = Observation(tsec, t["sensors"].pop(i), tuple(inbox[g]))
                    inbox[g] = []
                    c0 = time.perf_counter()
                    try:
                        act = b.step(obs)
                    except Exception as e:                     # brain fault: failsafe landing, run goes on
                        failed[g] = True
                        events.append((tick, g, "brain_error", repr(e)[:200]))
                        d.set_setpoint(i, LAND)
                        continue
                    cpu[g] += time.perf_counter() - c0
                    steps[g] += 1
                    if hasattr(b, "state_bytes"):
                        mem[g] = max(mem[g], int(b.state_bytes()))
                    d.set_setpoint(i, act.setpoint)
                    radio.send(g, act.outbox)
            if station is not None:
                act = station.step(Observation(tsec, {}, tuple(inbox[N])))
                inbox[N] = []
                radio.send(N, act.outbox)
            for g, kind in mission.update(tick, tsec, all_of("p"), all_of("alive"), hits):
                events.append((tick, g, kind, ""))
                t = teams[team_of[g]]
                t["drones"].set_setpoint(g - t["g0"], LAND)            # job over: land where it is
        if tick % kr == 0:
            boxes = radio.slot(node_pos(), node_alive())
            for g, box in enumerate(boxes):
                inbox[g] += box
            if record and (tick // kr) % LINK_EVERY == 0:          # delivered links, thinned for the viewer
                links.append((tsec, sorted({(pk.src, g) for g, box in enumerate(boxes) for pk in box})))
        for t in teams:
            for tk, i, kind in t["drones"].step(tick, t["sensors"].err):
                events.append((tk, t["g0"] + i, kind, ""))
            t["sensors"].step(tick)
        if record and tick % krec == 0:
            frames.append((tsec, all_of("p").copy(), all_of("q").copy(), all_of("alive").copy()))
        if tick % kb == 0:
            alive = all_of("alive")
            working = alive & ~mission.done & ~failed
            if (mission.finished() or not working.any()) and not (alive & ~all_of("landed")).any():
                break

    metrics = {
        "mission": s["mission"], "agents": N, "privileged": any(t["privileged"] for t in teams), "seconds": tsec,
        **mission.metrics(),
        "crashes": sum(e[2].startswith("crash") or e[2] == "geofence" for e in events),
        "brain_errors": int(failed.sum()), "bytes_sent": int(radio.sent.sum()),
        "bytes_delivered": int(radio.delivered.sum()), "bytes_dropped": int(radio.dropped.sum()),
        "brain_ms_mean": float(cpu.sum() / max(steps.sum(), 1) * 1e3), "state_bytes_max": int(mem.max()),
    }
    out = {"metrics": metrics, "events": events, "world": w}
    if record:
        out["frames"], out["links"] = frames, links
        if station is not None:
            out["station"] = gs_pos.tolist()
        if getattr(mission, "seen", None) is not None:                 # covered surface voxels, for the viewer
            idx = np.argwhere(mission.seen & mission.surface)
            out["covered"] = (mission.lo + (idx + 0.5) * mission.COVER_RES).round(2).tolist()
        out["hash"] = hashlib.sha256(b"".join(f[1].tobytes() + f[2].tobytes() for f in frames)).hexdigest()
    return out
