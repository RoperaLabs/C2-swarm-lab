# Performance

How fast the simulation loop runs on one laptop core, how to measure it, and where the time goes.

## Measuring

```bash
.venv/Scripts/python benchmarks/perf.py             # the full curve below, about 15 minutes
.venv/Scripts/python benchmarks/perf.py --check     # three cases against benchmarks/perf_baseline.json
```

`perf.py` runs `scenarios/ab_mixto.toml` with N boids launched together and drone collisions off, so that 50 and
100 iris fit at A. Each case uses quad or point-mass dynamics, with sensors on (six ToF arrays, EKF pose and an
all-round detector) or off (ideal pose only). Every case runs 5 times on seed 0 in one process with one numba thread,
after an untimed run, and only the window after 3 s of warm-up is timed (120 s of it with one drone, 15 s with 10,
6 s with 50 and 4 s with 100): numba compilation, world generation and the oracle paths are left out. Drones·tick/s
counts the drones alive in that window. Boids without sensors fly blind into obstacles, and a run ends when its last
drone is down, so a lone blind drone is timed only while it flies. `--check` fails when a case drops more than 15 %
below the baseline on the same platform; on another platform it only reports.

Measure on AC power. On battery this laptop's CPU either stays at 1.4 GHz or boosts and then throttles between
repetitions, with coefficients of variation of 20 to 35 %.

## The curve, v0.3.0 to F8

Intel Core i7-7700HQ, Windows 10, Python 3.14, NumPy 2.5, numba 0.68, one process, `NUMBA_NUM_THREADS=1`. On AC
power, without turbo, with the CPU at 99 % of its nominal 2.8 GHz during every run. Both versions were measured
interleaved, one case per process, on the same machine state. The real-time factor is simulated seconds per wall
second; CV is the coefficient of variation of drones·tick/s over the 5 runs.

| N | dynamics | sensors | v0.3.0 × real | F8 × real | speed-up | v0.3.0 drones·tick/s | F8 drones·tick/s | CV v0.3.0 / F8 |
|---|---|---|---|---|---|---|---|---|
| 1 | quad | on | 20.04 | 22.74 | 1.13× | 24,048 | 27,282 | 0.9 % / **5.9 %** |
| 10 | quad | on | 4.67 | 7.53 | 1.61× | 56,050 | 90,372 | 1.0 % / 1.0 % |
| 50 | quad | on | 0.75 | 1.73 | 2.31× | 42,429 | 98,020 | 3.2 % / 1.1 % |
| 100 | quad | on | 0.27 | 0.67 | 2.47× | 29,781 | 73,474 | 2.2 % / 0.7 % |
| 1 | pointmass | on | 14.97 | 16.86 | 1.13× | 17,959 | 20,235 | 0.6 % / 0.5 % |
| 10 | pointmass | on | 4.11 | 6.08 | 1.48× | 49,367 | 72,915 | 2.4 % / 4.7 % |
| 50 | pointmass | on | 0.80 | 1.79 | 2.25× | 44,843 | 100,703 | 0.5 % / 1.1 % |
| 100 | pointmass | on | 0.28 | 0.67 | 2.41× | 30,555 | 73,560 | 1.0 % / 2.4 % |
| 1 | quad | off | 36.32 | 35.06 | 0.97× | 43,487 | 41,987 | 2.2 % / 1.0 % |
| 10 | quad | off | 24.42 | 23.33 | 0.96× | 190,260 | 181,761 | 0.6 % / 1.1 % |
| 50 | quad | off | 4.69 | 5.86 | 1.25× | 248,528 | 310,657 | 1.4 % / 0.8 % |
| 100 | quad | off | 1.71 | 2.93 | 1.71× | 204,584 | 350,132 | 1.9 % / 2.6 % |
| 1 | pointmass | off | 24.17 | 23.76 | 0.98× | 28,812 | 28,322 | 3.4 % / 3.9 % |
| 10 | pointmass | off | 18.69 | 18.45 | 0.99× | 143,525 | 141,647 | 3.8 % / 1.8 % |
| 50 | pointmass | off | 5.62 | 7.72 | 1.37× | 289,434 | 397,732 | **10.3 %** / 1.4 % |
| 100 | pointmass | off | 1.93 | 3.51 | 1.82× | 229,696 | 418,354 | 2.6 % / 2.9 % |

The four cases with one drone (with the 120 s window since) and the two noisiest of the first pass (100 quad and 50
point mass with sensors) were measured a second time, once; the table gives that second pass. Two figures stay above 5 % (in bold):
F8 with one quad and sensors, which gave 0.6 % on the first pass, and v0.3.0 with 50 point masses without sensors,
whose repetitions last about a second. Both look like a single slow repetition rather than a trend. The cases that
`--check` watches (10 and 50 quad with sensors, 50 point mass without) stay between 1.0 and 1.4 %.

F8 changes the rays and the radio, so it pays off from 10 drones with sensors and from 50 without, and gives 1.13× with
a single drone and sensors. With one or ten drones without sensors both versions run at the same speed within the
noise (0.96 to 0.99×); the run clock adds 0.3 µs per tick, about 1 % there. Every F8 run keeps the bits of v0.3.0:
`tests/test_reference.py` pins the run hash and a digest of every observation of six reference runs, and the full
runs below keep their hashes.

## The published scenario

`scenarios/ab_mixto.toml` as published (one launch every 2 s, drone collisions on), the whole run timed with numba
already compiled, same machine state as the curve:

| run | v0.3.0 | F8 |
|---|---|---|
| 10 drones, 120 s | 27.4 s (4.4×) | 17.1 s (7.0×) |
| 50 drones, 60 s | 104.1 s (0.58×) | 43.3 s (1.39×) |

Turbo raises every figure but not the ratios: the day before, on AC power with turbo, the 50-drone run took 83.6 s
with v0.3.0 and 50.5 s with F8 before its last two optimizations (0.72× to 1.19×), with the same hash.

## Where the time goes in F8

Share of the timed loop:

| N | dynamics | sensors | physics | rays | other sensors | radio | brains | mission | rest |
|---|---|---|---|---|---|---|---|---|---|
| 1 | quad | on | 44 % | 17 % | 19 % | 2 % | 7 % | 1 % | 10 % |
| 10 | quad | on | 17 % | 39 % | 14 % | 3 % | 22 % | 0 % | 5 % |
| 50 | quad | on | 8 % | 38 % | 11 % | 3 % | 36 % | 0 % | 3 % |
| 100 | quad | on | 8 % | 39 % | 8 % | 2 % | 41 % | 0 % | 2 % |
| 100 | pointmass | off | 18 % | 0 % | 9 % | 14 % | 48 % | 0 % | 11 % |

Rays are the ToF casts, other drones as spheres and the detector; the rest is recording, observations and the end
check. With 50 drones or more, the brains (boids here) take a third or more of the loop.

## What F8 changed

- Ray sensors: the rays of one drone share an origin, so `World.cast` pre-filters the candidate primitives once per
  drone, and computes no normals and allocates nothing per ray.
- Other drones as spheres: a sphere out of reach of the range is skipped, except in coverage, which also scores hits
  beyond the range; each sphere's terms are computed once per drone.
- Flat terrain at height 0 (`mixto`, `urbano`, `interior`, `swarm_dil_v2`): the terrain distance is z, exactly.
- Detector: one line-of-sight query for all drones, and the per-candidate draws in a compiled loop; numba's
  `Generator` draws exactly what NumPy's draws from the same state.
- Radio: a broadcast takes its draws in one call, and without received power its receivers share one immutable
  `Packet`.
- Compiled loops use `prange`: even with one thread they run about twice as fast as plain `njit` loops here.

## Scaling

Parallelise across seeds, not threads: `python -m c2lab sweep` runs up to 5 processes with one numba thread each.
Runs are independent, and most of a run's cost is Python that threads would not speed up.
