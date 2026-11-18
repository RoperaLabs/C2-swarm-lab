# Clúster cimientos (F0)

Leer antes: `CLAUDE.md`, `.ia/grafo/_indice.md`, `src/c2lab/brain.py` y `scenarios/ejemplo.toml`.
Archivos permitidos: `src/c2lab/seeds.py`, `src/c2lab/scenario.py`, `tests/test_seeds.py`, `tests/test_scenario.py`, `tests/test_isolation.py` y la sección `## Deltas` de este archivo.
Check: `.venv/Scripts/python -m pytest -q` en verde. Solo biblioteca estándar y numpy.

## T0.6 `seeds.py`

`rng(seed: int, *path: str | int) -> np.random.Generator`, construido con `np.random.SeedSequence(seed, spawn_key=...)`. Cada elemento de `path` se convierte en un entero estable: los int se usan tal cual (≥ 0) y los str pasan por `hashlib.blake2b(..., digest_size=8)`. **Nunca** se usa `hash()` de Python, que cambia entre procesos.

Tests (`tests/test_seeds.py`):
1. La misma (seed, path) da los mismos números.
2. Otra seed, u otra path, da números distintos.
3. Crear otros flujos antes, después o en otro orden no cambia los números de un flujo.
4. `("agent", 1, "tof")` y `("agent", "1", "tof")` dan flujos distintos.
5. Valor fijado: `rng(0, "world").random()` es igual a una constante escrita en el test (calcularla una vez tras implementar). Protege la repetibilidad entre versiones.

## T0.7a `scenario.py`

`parse(text: str) -> dict` (con `tomllib`) y `load(path) -> dict`, que lee el archivo y llama a `parse`. Devuelven el escenario validado, con los valores por defecto rellenados y los sensores apagados eliminados. Cualquier error lanza `ScenarioError(ValueError)` con un mensaje que empieza por la ruta completa de la clave: `scenario.rates_hz.brain`, `team[0].sensors.tof.hz`, `world.biom`... Una clave desconocida en cualquier nivel validado es un error que la nombra.

| Ruta | Tipo y regla | Por defecto |
|---|---|---|
| `scenario.name` | str no vacío | obligatorio |
| `scenario.seed` | int ≥ 0 | obligatorio |
| `scenario.duration_s` | número > 0 | obligatorio |
| `scenario.mission` | `a_to_b` \| `coverage` \| `pursuit` | obligatorio |
| `scenario.rates_hz.physics` | int > 0 | 1200 |
| `scenario.rates_hz.brain` / `.radio` / `.record` | número > 0 con periodo entero de ticks | 10 / 10 / 30 |
| `world.biome` | `mixto` \| `urbano` \| `bosque` \| `interior` | obligatorio |
| `world.size_m` | lista de 3 números > 0 | según el bioma: mixto [60, 40, 20], urbano [200, 200, 60], bosque [150, 150, 30], interior [40, 30, 8] |
| `world.density` | 0 < x < 1 | 0.2 |
| `radio.profile` | `ideal` \| `disco` \| `dil_v02` | `dil_v02` (la tabla `[radio]` entera es opcional) |
| `radio.packet_bytes` | int ≥ 1 | 64 |
| `radio.ranging` | bool | false |
| `team` | lista ≥ 1 de tablas `[[team]]` | obligatorio |
| `team[i].name` | str no vacío, único entre equipos | obligatorio |
| `team[i].count` | int ≥ 1 | obligatorio |
| `team[i].brain` | `"modulo.sub:Clase"` (regex `^[A-Za-z_][\w.]*:[A-Za-z_]\w*$`) | obligatorio |
| `team[i].brain_params` | tabla | {} |
| `team[i].role` | str no vacío | "nav" |
| `team[i].drone` | `cf2` \| `iris` | `iris` |
| `team[i].memory_bytes` | int ≥ 0 | None (sin límite) |
| `team[i].sensors` | tabla nombre → tabla \| bool. Nombres válidos: `tof`, `lidar`, `depth`, `detector`, `pose`, `imu`, `gnss`, `baro`, `mag`, `prior_map`, `battery`. `true` equivale a `{}` y `false` elimina el sensor. Si la tabla trae `hz`, se aplica la regla del periodo entero. El resto de parámetros por sensor se acepta tal cual (los valida F3) | {} |
| `ground_station.brain` / `.brain_params` | igual que en team; la tabla es opcional | — |

"Periodo entero de ticks" significa que `physics / f` es entero (tolerancia 1e-9). El mensaje dice la ruta, el valor y la frecuencia de física.

Tests (`tests/test_scenario.py`, con `pytest.mark.parametrize` para los casos de error):
- `scenarios/ejemplo.toml` carga y trae los valores por defecto esperados (`rates_hz.physics == 1200`, `team[0].role == "nav"`, `"gnss"` no está en sensores y `"tof"` sí).
- Un escenario mínimo (solo los campos obligatorios) rellena todos los valores por defecto de la tabla.
- Casos de error. Cada uno comprueba que el mensaje contiene la ruta indicada:
  - falta `scenario.seed`;
  - `world.biom` (clave mal escrita);
  - `scenario.mission = "foo"`;
  - `scenario.rates_hz.brain = 7` (el mensaje contiene también `7`);
  - `team[0].sensors.tof.hz = 7`;
  - `team[0].brain = "boids"`;
  - nombres de equipo duplicados (`team[1].name`);
  - `team[0].sensors.sonar`;
  - `team[0].count = 0`;
  - `radio.packet_bytes = 0`.

## T0.7b `tests/test_isolation.py`

Una función `violations(source: str) -> list[str]` analiza el AST de un archivo y devuelve una línea por cada importación prohibida.
- **Permitido:** la biblioteca estándar (`sys.stdlib_module_names`), `numpy`, `scipy`, `c2lab.brain`, `from c2lab import brain`, `brains.*` y las importaciones relativas.
- **Prohibido:** cualquier otro `c2lab.*`, `import c2lab` a secas, `from c2lab import <otro>`, `importlib`, llamadas a `__import__` y cualquier otro paquete de terceros.

Tests:
1. Todos los `.py` de `brains/` (recursivo, excluyendo `brains/privileged/`) devuelven una lista vacía.
2. Fragmentos sintéticos parametrizados: `import numpy as np`, `from c2lab.brain import Action`, `from c2lab import brain`, `from .common import x` y `import math` pasan. `from c2lab.world import World`, `import c2lab.sim`, `from c2lab import world`, `import c2lab`, `import importlib`, `__import__("c2lab.world")` e `import matplotlib` fallan.

## Deltas
N SeedTree: src/c2lab/seeds.py: rng(seed, *path) con SeedSequence(spawn_key) y blake2b para str; test con valor fijado rng(0,"world").random() = 0.4645953870616605
N Scenario: src/c2lab/scenario.py: parse/load/ScenarioError según el esquema de T0.7a
N IsolationCheck: tests/test_isolation.py: violations(source) por AST; brains/ (sin privileged/) limpio
R IsolationCheck -> BrainContract: único c2lab.* permitido a brains/
A Scenario (Opus): números finitos, periodo >= 1 tick, sección desconocida nombrada sin prefijo; 36 tests
