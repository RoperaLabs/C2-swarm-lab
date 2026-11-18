"""Brain contract: the only part of c2lab a brain may import (tests/test_isolation.py).

A brain sees what a real drone would measure and nothing else: noisy sensor readings in its own frames and the
bytes its radio received. It never gets world geometry, a voxel grid or global coordinates unless a sensor that
produces them is switched on (gnss, prior_map).

Units are SI. Frames follow REP-103: the world is ENU (x east, y north, z up), the body is FLU (forward, left, up),
and the local frame is gravity-aligned, ENU-oriented and anchored at the drone's take-off point. Quaternions are
(w, x, y, z), body to frame.
"""
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class AgentSpec:
    """What the drone knows before take-off: identity, hardware and mission order."""
    agent_id: int
    team: str
    role: str                    # "nav", "pursuer", "evader", "ground" (ground station), ...
    seed: int                    # the brain's own seed: a brain needing randomness draws only from it
    sensors: dict                # sensor name -> public parameters (fov_deg, range_m, hz, mount, ...)
    limits: dict                 # brain_hz, packet_bytes, memory_bytes, vmax, amax, ...
    mission: dict                # e.g. {"goal_local": (x, y, z)}; "goal_enu" only with gnss on


@dataclass(frozen=True)
class Reading:
    t: float                     # sample time, s
    data: np.ndarray             # read-only; layout documented per sensor


@dataclass(frozen=True)
class Packet:
    src: int                     # radio address of the sender
    data: bytes
    rssi_dbm: float
    range_m: float | None = None  # only with radio ranging on


@dataclass(frozen=True)
class Observation:
    t: float
    readings: dict = field(default_factory=dict)  # sensor -> tuple[Reading, ...]: samples since the last step, oldest first
    inbox: tuple = ()                             # Packet, ...


@dataclass(frozen=True)
class Velocity:
    v: tuple                     # (vx, vy, vz), m/s
    yaw_rate: float = 0.0        # rad/s
    frame: Literal["local", "body"] = "local"


@dataclass(frozen=True)
class Position:
    dp: tuple                    # displacement from the current estimated position, m, local frame
    yaw: float | None = None     # rad in the local frame; None keeps the heading


@dataclass(frozen=True)
class Action:
    setpoint: Velocity | Position | None = None   # None: hold position (always None for a ground station)
    outbox: tuple = ()           # bytes (broadcast) or (dest, bytes); a payload over packet_bytes is dropped and counted


class Brain(Protocol):
    """A brain is any class with these two methods, built with the team's brain_params as keyword arguments.

    Optional: state_bytes() -> int (persistent memory, checked against memory_bytes) and export_map() (its own
    map, scored by the coverage mission). A ground station implements the same protocol.
    """

    def reset(self, spec: AgentSpec) -> None: ...

    def step(self, obs: Observation) -> Action: ...
