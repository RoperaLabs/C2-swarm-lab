"""Radio channel between drones (and a ground station). Packets sent during one slot are delivered at the next slot,
in sender order. Profiles:

- ideal: every packet arrives.
- disco: within `range_m`, independent loss `loss`.
- dil_v02: the swarm-dil v0.2 link model (path loss, AR(1) shadowing per link, `obstacle_db` per obstacle crossed
  on the 3-D line of sight, logistic packet error on the SINR, airtime contention at the receiver).
"""
import math

import numpy as np

from c2lab.brain import Packet

DIL = {"tx_dbm": 10.0, "path_loss_exp": 2.7, "pl_1m_db": 40.0, "noise_dbm": -110.0, "shadow_sigma_db": 4.0,
       "shadow_corr": 0.9, "sinr50_db": 6.0, "sinr_width_db": 1.5, "obstacle_db": 20.0, "rate_bps": 250_000.0}
RANGE_SIGMA = 0.1   # m, UWB-like ranging noise


def contention(p, airtime, audible=0.05):
    """Share of airtime a receiver gets: 1 while the channel it hears is not saturated (swarm-dil v0.2)."""
    load = airtime + ((p > audible) * airtime[None]).sum(1)
    return np.minimum(1.0, 1.0 / np.maximum(load, 1e-12))


class Radio:
    def __init__(self, cfg, n, rng, world, slot_dt, gains_db=None):
        self.cfg, self.n, self.rng, self.world, self.dt = cfg, n, rng, world, slot_dt
        self.profile = cfg.get("profile", "dil_v02")
        self.packet_bytes, self.max_packets = cfg.get("packet_bytes", 64), cfg.get("max_packets", 4)
        self.ranging = cfg.get("ranging", False)
        self.dil = {**DIL, **cfg.get("dil", {})}
        self.gain = np.zeros(n) if gains_db is None else np.asarray(gains_db, float)
        self.queue, self._inflight = [[] for _ in range(n)], [[] for _ in range(n)]
        self.sent, self.delivered, self.dropped = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        self.shadow = self._sym_noise()

    def _sym_noise(self):
        s = np.triu(self.rng.normal(0, self.dil["shadow_sigma_db"], (self.n, self.n)), 1)
        return s + s.T

    def send(self, i, outbox):
        for item in outbox:
            dest, data = (None, item) if isinstance(item, (bytes, bytearray)) else item
            ok = isinstance(data, (bytes, bytearray)) and len(data) <= self.packet_bytes
            if not ok or len(self.queue[i]) >= self.max_packets:
                self.dropped[i] += len(data) if isinstance(data, (bytes, bytearray)) else 0
                continue
            self.queue[i].append((dest, bytes(data)))
            self.sent[i] += len(data)

    def _links(self, p, alive, airtime):
        """(n, n) delivery probability P(j -> i) and received power (dBm)."""
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        if self.profile == "ideal":
            q, rx = np.ones((self.n, self.n)), np.zeros((self.n, self.n))
        elif self.profile == "disco":
            q = (d <= self.cfg.get("range_m", 100.0)) * (1 - self.cfg.get("loss", 0.0))
            rx = np.zeros((self.n, self.n))
        else:
            r = self.dil
            rho = r["shadow_corr"] ** self.dt
            self.shadow = rho * self.shadow + math.sqrt(1 - rho * rho) * self._sym_noise()
            iu = np.triu_indices(self.n, 1)
            cross = np.zeros((self.n, self.n))
            if len(iu[0]):
                c = self.world.segment_crossings(p[iu[0]], p[iu[1]])
                cross[iu], cross[iu[1], iu[0]] = c, c
            pl = r["pl_1m_db"] + 10 * r["path_loss_exp"] * np.log10(np.maximum(d, 1.0))
            rx = r["tx_dbm"] + self.gain[:, None] + self.gain[None] - pl - r["obstacle_db"] * cross + self.shadow
            q = 1 / (1 + np.exp(-(rx - r["noise_dbm"] - r["sinr50_db"]) / r["sinr_width_db"]))
            q = q * contention(q, airtime)[:, None]
        np.fill_diagonal(q, 0.0)
        q[~alive], q[:, ~alive] = 0.0, 0.0
        return q, rx, d

    def slot(self, p, alive):
        """Deliver what was queued before the previous slot; returns one inbox (list of Packet) per node.

        One uniform draw per (packet, receiver other than the sender), in sender, packet and receiver order. Without
        ranging a broadcast takes its draws in one call, which yields the same numbers, and shares one Packet among
        its receivers when the profile has no received power (Packet is immutable)."""
        out, self._inflight, self.queue = self._inflight, self.queue, [[] for _ in range(self.n)]
        airtime = np.array([8 * sum(len(x) for _, x in q) for q in out]) / (self.dil["rate_bps"] * self.dt)
        prob, rx, d = self._links(np.asarray(p, float), np.asarray(alive, bool), airtime)
        inbox = [[] for _ in range(self.n)]
        for j, packets in enumerate(out):
            for dest, data in packets:
                if dest is None and not self.ranging:
                    to = np.delete(np.arange(self.n), j)
                    to = to[self.rng.random(len(to)) < prob[to, j]]
                    shared = Packet(j, data, 0.0) if self.profile != "dil_v02" else None
                    for i in to.tolist():
                        inbox[i].append(shared or Packet(j, data, float(rx[i, j])))
                    self.delivered[to] += len(data)
                    continue
                for i in (range(self.n) if dest is None else [dest]):
                    if i != j and self.rng.random() < prob[i, j]:
                        rng_m = d[i, j] + self.rng.normal(0, RANGE_SIGMA) if self.ranging else None
                        inbox[i].append(Packet(j, data, float(rx[i, j]), rng_m))
                        self.delivered[i] += len(data)
        return inbox
