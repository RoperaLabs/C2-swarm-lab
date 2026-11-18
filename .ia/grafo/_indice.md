# C2-swarm-lab: contrato común (raíz: la carpeta del proyecto)

Banco de pruebas 3D para algoritmos C2 de enjambres de drones. Plan completo: `PLAN.md`. Reglas y convenciones: `CLAUDE.md` (leer antes de nada).

## Clústeres

| clúster | fase | archivo de spec | estado |
|---|---|---|---|
| cimientos | F0 | .ia/grafo/cimientos.md | hecho |
| mundo | F1 | .ia/grafo/mundo.md | pendiente |
| fisica | F2 | .ia/grafo/fisica.md | pendiente |
| sensores | F3 | .ia/grafo/sensores.md | pendiente |
| cerebros | F4 | .ia/grafo/cerebros.md | pendiente |
| misiones | F5 | .ia/grafo/misiones.md | pendiente |
| lotes-visor | F6 | .ia/grafo/lotes-visor.md | pendiente |
| grace | F7 | .ia/grafo/grace.md | pendiente |

## Nodos y contratos entre módulos

Un contrato solo cambia con `BLOQUEO contrato: <qué>` y la aprobación de Opus.

### `BrainContract`: `src/c2lab/brain.py` (F0; congelado en F4)
Tipos `AgentSpec`, `Reading`, `Packet`, `Observation`, `Velocity`, `Position`, `Action` y el protocolo `Brain` (`reset(spec)`, `step(obs) -> Action`). `Observation.readings[sensor]` es una tupla de `Reading`, una por muestra desde el último paso, de la más antigua a la más reciente.

### `SeedTree`: `src/c2lab/seeds.py` (F0)
`rng(seed: int, *path: str | int) -> np.random.Generator`. El generador depende solo de `seed` y `path`, no del orden de creación ni de qué otros flujos existan. Rutas reservadas: `("world",)`, `("mission",)`, `("wind",)`, `("radio",)`, `("agent", i, "<sensor>")` y `("agent", i, "brain")` (de este último sale `AgentSpec.seed`).

### `Scenario`: `src/c2lab/scenario.py` (F0)
`load(path) -> dict` y `parse(text) -> dict` validan y rellenan valores por defecto. Cualquier error lanza `ScenarioError` (subclase de `ValueError`) con un mensaje que nombra la clave completa y el valor, p. ej. `scenario.rates_hz.brain = 7: 1200 Hz de física no da un periodo entero de ticks`. Esquema: `.ia/grafo/cimientos.md`.

### `WorldQuery`: `src/c2lab/world/` (lo implementa F1; F2 y F3 programan contra estas firmas)
Todos los arrays son float64, en metros y ENU.
- `World.sdf(p: (N,3)) -> (N,)`: distancia con signo a la superficie más cercana (negativa dentro). Incluye suelo o terreno y excluye la geocerca.
- `World.raycast(o: (N,3), d: (N,3) unitarios, max_range: float | (N,)) -> (dist (N,), prim (N,) int, normal (N,3))`. Sin impacto dentro del alcance: `dist = inf`, `prim = -1`, `normal = 0`. Origen dentro de un obstáculo: `dist = 0`.
- `World.segment_crossings(a: (N,3), b: (N,3)) -> (N,) int`: número de primitivas que corta cada segmento. Es la línea de vista de la radio y del detector (0 = visible).
- `World.bounds -> (lo (3,), hi (3,))`: la geocerca.
- `World.to_dict()`, `World.from_dict(d)`, `World.hash -> str` (sha256 del JSON canónico).

### `TrueState`: lo produce la física (F2) y lo consumen los sensores y las misiones (F3, F5)
Arrays sobre N drones: `p (N,3)` ENU m, `v (N,3)` m/s, `q (N,4)` wxyz de cuerpo a mundo, `w (N,3)` rad/s en cuerpo, `f (N,3)` fuerza específica en cuerpo (lo que mide un acelerómetro), `alive (N,)` bool.

## Archivos y reglas para agentes

- Escribir solo los archivos permitidos en el spec del clúster. `src/c2lab/brain.py`, `pyproject.toml`, `CLAUDE.md` y este índice son de solo lectura.
- No añadir dependencias. numpy es la única de ejecución; pytest, la de tests.
- Check común: `.venv/Scripts/python -m pytest -q` desde la raíz, en verde.
- Deltas: añadir líneas al final de la sección `## Deltas` del clúster con el formato `N <nodo>` (nuevo), `R <a> -> <b>: <relación>` (arista nueva), `A <nodo>: <cambio>` (actualizado) y `X <nodo>` (eliminado).
