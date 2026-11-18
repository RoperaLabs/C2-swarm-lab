import numpy as np
import pytest

from c2lab.scenario import BIOMES
from c2lab.world import biomes, generate, oracle


@pytest.mark.parametrize("biome", sorted(BIOMES))
def test_biome(biome):
    size = BIOMES[biome]
    worlds = [generate(biome, size, 0.2, s) for s in (0, 1)]
    assert generate(biome, size, 0.2, 0).hash == worlds[0].hash
    assert worlds[0].hash != worlds[1].hash
    for w in worlds:
        assert oracle.feasible(w, oracle.RES[biome])
        for x, y, z in w.anchors.values():
            assert abs(z - w.ground([[x, y]])[0]) < 1e-6
            col = np.c_[np.full(20, x), np.full(20, y), np.linspace(z + 0.1, size[2] - 1, 20)]
            if biome == "interior":
                col = col[col[:, 2] < size[2] - 0.3 - biomes.CLEAR[biome]]      # the ceiling closes the column
            assert w.sdf(col, terrain=False).min() >= biomes.CLEAR[biome] - 0.01
        cov = oracle.coverage(w, oracle.RES[biome], biomes.z_top(biome, size))
        assert abs(cov - 0.2) <= 0.15 * 0.2
