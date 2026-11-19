import numpy as np

from brains.grace3d import Grace3D
from c2lab.brain import AgentSpec, Observation, Reading

TOF = {"arrays": ["front", "left", "right", "back", "up", "down"], "zones": 8, "fov_deg": 90.0, "range_m": 25.0}


def brain(view="all", tof=TOF):
    b = Grace3D(team=10, view=view)
    b.reset(AgentSpec(agent_id=3, team="g", role="nav", seed=1, sensors={"tof": tof},
                      limits={"packet_bytes": 64, "vmax": 12.0, "vmax_up": 3.0, "vmax_down": 1.0},
                      mission={"area_local": ((-12.5, -12.5, -0.4), (187.5, 187.5, 49.6)),
                               "a_local": (0.0, 0.0, -0.4), "goal_local": (175.0, 175.0, 1.6)}))
    return b


def test_lattice_from_mission():
    g = brain().g
    assert g.s.shape == (40, 40, 10) and g.start == (2, 2, 0) and g.goal == (37, 37, 0) and g.s.launch_at == 30


def test_returns_go_to_the_voxel_beyond_the_nearest_face():
    b = brain()
    d = np.full((6, 8, 8), np.inf)
    d[0, 3, 3] = 7.5 / b.dirs[0, 3, 3, 0]              # front zone near the axis: a wall face at x = 7.5 m
    d[5, 3, 3] = 2.4 / -b.dirs[5, 3, 3, 2]             # down: the ground 2.4 m below, dropped
    pose = Reading(0.0, np.r_[0.0, 0.0, 2.0, 0, 0, 0, 1.0, 0, 0, 0])
    obs = Observation(0.0, {"tof": (Reading(0.0, d),), "pose": (pose,)})
    b._scan(obs, obs.readings["pose"])
    hit = b.lo + (np.array(sorted(b.seen)[0]) + 0.5) * b.cell
    assert len(b.seen) == 1 and hit[0] > b.pts[0, 0] - 0.01      # the voxel behind the face, not the one in front


def test_noisy_returns_need_occupancy_evidence():
    b = brain(tof={**TOF, "sigma_rel": 0.03, "sigma_min": 0.005, "sub": 2})
    pose = (Reading(0.0, np.r_[0.0, 0.0, 2.0, 0, 0, 0, 1.0, 0, 0, 0]),)

    def scan(x):                                       # front zone near the axis returns at x metres ahead
        d = np.full((6, 8, 8), np.inf)
        d[0, 3, 3] = x / b.dirs[0, 3, 3, 0]
        b.seen = set()
        b._scan(Observation(0.0, {"tof": (Reading(0.0, d),), "pose": pose}), pose)
        return b.seen

    assert not scan(17.5)                              # one return is not enough
    for _ in range(3):
        wall = scan(17.5)
    assert len(wall) == 1 and b.lo[0] + list(wall)[0][0] * b.cell >= 17.5 - 1e-9     # the voxel behind the face
    for _ in range(8):
        assert not scan(6.0)                           # phantom in a voxel the earlier rays crossed: outvoted


def test_blocked_diagonal_flies_axis_legs_or_is_refused():
    b = brain()
    pos = b.g.pos
    up = 16                                            # STEPS[16] = (0, 1, 1)
    assert b._legs(pos, up) == [(2, 3, 1)]
    b.hits = {(2, 3, 0)}
    assert b._legs(pos, up) == [(2, 2, 1), (2, 3, 1)]
    b.hits = {(2, 3, 0), (2, 2, 1)}
    assert b._legs(pos, up) is None
    assert b._legs(pos, 0) is None                     # (-1, -1, -1): below the ground layer


def test_swarm_dil_view_is_two_frusta():
    b = brain("swarm_dil")
    rel = np.array([[10.0, 10.0, 0.0], [-10.0, -10.0, 0.0], [0.0, 0.0, 10.0]])
    assert b._in_view(rel).tolist()[:2] == [True, False]        # heading (1, 1, 0): ahead seen, behind not
    assert brain()._in_view(rel).all()
