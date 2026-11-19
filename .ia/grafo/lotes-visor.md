# Clúster lotes-visor (F6)

Código: `src/c2lab/sweep.py` (barrido y resumen), `src/c2lab/__main__.py` (CLI), `viewer/main.js` (reproducción completa). Escenario de ejemplo: `scenarios/sweep_ab_mixto.toml`. Tests: `tests/test_sweep.py`.

## Deltas
N Sweep: tabla [sweep] (seeds [desde, hasta), variantes con claves con puntos "team.0.brain"); runs/<nombre>/<variante>/<semilla>/metrics.json (+ run_hash) o error.json; reanudable; ≤ 5 procesos con NUMBA_NUM_THREADS=1
N Summarize: media ± e.e. por variante y métrica; diferencias emparejadas contra la base con Wilcoxon de una muestra y Holm sobre todas las comparaciones; summary.csv y summary.md
N CLI: python -m c2lab run | sweep | summarize | serve
A Viewer: cámara que sigue a un dron, enlaces de radio entregados (2 Hz), estación de tierra, nube de vóxeles cubiertos, lista de eventos con salto, métrica según misión
A Sim/Record: replay.json con links, covered y station
A PotentialField: empuje de los bordes del área = 2·k_sep (16) en el borde; antes podía ganar la repulsión y el dron salía de la geocerca (cobertura)
A Scenario: tabla [sweep] validada
A Aceptación F6:
  - mismo hash por ejecución con 1 y con 2 procesos; reanudación repite solo lo que falta (tests)
  - CPU durante un barrido de 5 procesos: 67-73 % (≤ 85 %) ✓
  - barrido real sweep_ab_mixto 3 × 20 (antes del cambio de borde): oráculo 98 % de llegadas; potential_field 35,5 % frente a boids 22 % (+0,135, p Holm = 0,025), choques 1,1 frente a 2,5 (p Holm = 0,033)
  - no medido: ≥ 30 fps con 50 drones (el panel del navegador va en segundo plano)
  - fuera de alcance: rayos por fotograma del dron seleccionado (engordan la grabación) y modo en vivo (opcional)
