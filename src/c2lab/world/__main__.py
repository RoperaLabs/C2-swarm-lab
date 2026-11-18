"""python -m c2lab.world <biome> --seed N [--density D] [--size X Y Z] [--out FILE]: generate and save a world."""
import argparse
import json
from collections import Counter
from pathlib import Path

from c2lab.scenario import BIOMES

from . import biomes, generate, oracle

ap = argparse.ArgumentParser(prog="python -m c2lab.world")
ap.add_argument("biome", choices=sorted(BIOMES))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--density", type=float, default=0.2)
ap.add_argument("--size", type=float, nargs=3)
ap.add_argument("--out")
a = ap.parse_args()
size = a.size or BIOMES[a.biome]
w = generate(a.biome, size, a.density, a.seed)
out = Path(a.out or f"runs/worlds/{a.biome}_{a.seed}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(w.to_dict()), encoding="utf-8")
cov = oracle.coverage(w, oracle.RES[a.biome], biomes.z_top(a.biome, size))
print(f"{out}  hash {w.hash[:12]}  attempts {w.attempts}  {w.seconds:.2f} s  cover {cov:.3f}")
print(dict(Counter(p["type"] for p in w.to_dict()["primitives"])))
