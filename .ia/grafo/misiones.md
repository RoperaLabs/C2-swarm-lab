# Clúster misiones (F5)

Código: `src/c2lab/missions.py` (contrato y misiones), `src/c2lab/sim.py` (las usa), cerebros `brains/pursue.py`, `brains/flee.py`, `brains/sector_sweep.py`, `brains/central_assign.py` (estación de tierra). Escenarios: `scenarios/coverage_mixto.toml`, `scenarios/pursuit_mixto.toml`. Validación: `benchmarks/validate_missions.py [tipos...]`. Tests: `tests/test_missions.py`.

## Deltas
N Mission (contrato): spawn(team) → ancla, info(g, home, gnss), station_info(), start(homes, privileged), update(tick, t, p, alive, hits) → eventos y done[g], finished(), metrics()
N AToB: regret = volado / camino más corto − 1 (un Dijkstra por ejecución, PATH_RES por bioma; urbano a 2 m); guía privilegiada con holgura 0,9 m
N Coverage: puntuación del arnés = vóxeles de superficie (1 m) tocados por algún rayo de los sensores (Sensors.hits); t50, t80
N Pursuit: evasores (role "evader") salen en B; captura = perseguidor a < 2 m durante 0,5 s; capture_rate, tiempo, distancia mínima
N GroundStation: nodo de radio N con cerebro propio y sin cuerpo, ganancia 12 dB; recibe area_enu, homes_enu y agents
N Brain:pursue (avistamiento confirmado en 2 pasos, punto propio a 1,2 m del blanco), Brain:flee, Brain:sector_sweep (franja de 8 B), Brain:central_assign; random_walk y todos heredan la evasión de PotentialField
R Sim -> Mission, R Sensors -> Mission (hits), R central_assign -> sector_sweep (radio unicast)
A Sensors: ven a los drones de todos los equipos (peers); antes el detector solo veía a su equipo
A PotentialField: objetivo recortado al área, empuje de bordes solo laterales y techo, separación k_sep = 8, escape más suave (decay 0,97, magnitud = velocidad), detecciones con confianza ≥ 0,7
A Aceptación (20 semillas emparejadas, 5 procesos):
  - A→B oráculo: llegadas 98 %, regret medio +0,019 (máx 0,073) ✓ (≤ 0,05)
  - cobertura: central 0,797 ± 0,067 frente a aleatoria 0,776 ± 0,051, Wilcoxon emparejado p = 0,011 ✓
  - persecución: evasor que huye capturado 80 % con 1 perseguidor y 100 % con 4 ✓
  - FUERA (por poco): evasor quieto capturado 95 % (objetivo 100 %); la semilla que falla tiene choques contra obstáculos a 3 m/s
