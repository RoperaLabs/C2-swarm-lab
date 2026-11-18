# Clúster cerebros (F4)

Código: `src/c2lab/sim.py` (bucle), `src/c2lab/record.py` (grabación), `src/c2lab/__main__.py` (`python -m c2lab run`), `brains/` y `viewer/main.js` (reproducción). Guía: `docs/brain_api.md`. Tests: `tests/test_sim.py`.

## Deltas
N Sim (Opus): bucle multitasa (cerebros → ranura de radio → física con err → sensores → llegada → grabación), failsafe (aterriza) ante excepción, lanzamiento escalonado `launch_s`, llegada a B + 2 m (radio 2 m) y aterrizaje, métricas (llegadas, tiempo, distancia volada, longitud del oráculo, choques, bytes, ms por paso, state_bytes)
N Record: run dir con meta.json, metrics.json, events.jsonl y replay.json; visor `?run=`
N Brain:hover, Brain:random_walk, Brain:potential_field (detector para separación, escape por falta de progreso), Brain:boids (detector + velocidades por radio de 6 B), Brain:oracle (privilegiado)
N Oracle.shortest_paths: un Dijkstra desde la meta (26-conexo, scipy.sparse.csgraph) + tirado de cuerda con holgura; el cerebro oráculo se guía con holgura 0,9 m
R Sim -> Sensors, Radio, Drones, World; R Brain -> BrainContract; R Oracle-brain -> shortest_paths (privilegiado)
A Sensors: los sensores de rayos ven a los demás drones del equipo como esferas (kernel numba sphere_hits)
A Scenario: team.launch_s
A Aceptación F4:
  - cerebro por una línea del TOML, aislamiento, failsafe, misma semilla = mismo hash: tests OK (100 en total)
  - oráculo en mixto 0,2, 20 semillas, 10 drones, launch 4 s: llegadas 98 %; distancia volada / camino óptimo = 1,00 ± 0,01
  - FUERA: potential_field 31 %, boids 21 % (objetivo 70 %); reactivos con ToF de 4 m caen en mínimos locales detrás de muros
  - FUERA: escenario de referencia (urbano, 10 boids, ToF×4, vio, detector, dil_v02) 4,3x tiempo real (objetivo 5x)
