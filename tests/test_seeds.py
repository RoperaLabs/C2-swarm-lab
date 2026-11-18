import numpy as np

from c2lab.seeds import rng


def test_same_seed_and_path_repeat():
    assert np.array_equal(rng(3, "wind").random(5), rng(3, "wind").random(5))


def test_other_seed_or_path_differ():
    base = rng(3, "wind").random()
    assert rng(4, "wind").random() != base
    assert rng(3, "radio").random() != base


def test_creation_order_irrelevant():
    alone = rng(1, "agent", 2, "imu").random(3)
    rng(1, "world").random(100)
    rng(1, "agent", 3, "imu")
    assert np.array_equal(rng(1, "agent", 2, "imu").random(3), alone)


def test_int_and_str_parts_differ():
    assert rng(0, "agent", 1, "tof").random() != rng(0, "agent", "1", "tof").random()


PINNED = 0.4645953870616605


def test_pinned_value():
    assert rng(0, "world").random() == PINNED

