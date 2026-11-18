# C2-swarm-lab: contrato común (raíz: la carpeta del proyecto)

Banco de pruebas 3D para algoritmos C2 de enjambres de drones. Plan completo: `PLAN.md`. Reglas y convenciones: `CLAUDE.md` (leer antes de nada).

## Clústeres

| clúster | fase | archivo de spec | estado |
|---|---|---|---|
| cimientos | F0 | .ia/grafo/cimientos.md | hecho |
| mundo | F1 | .ia/grafo/mundo.md | hecho |
| fisica | F2 | .ia/grafo/fisica.md | hecho |
| sensores | F3 | .ia/grafo/sensores.md | hecho |
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
`from c2lab.world import World, generate`. Todos los arrays son float64, en metros y ENU.
- `World.sdf(p (N,3), terrain=True, max_dist=inf) -> (N,)`: distancia con signo (negativa dentro), recortada a `max_dist`, de modo que un punto sin nada más cerca devuelve `max_dist`. `terrain=False` ignora el terreno. Es exacta para las primitivas; para el terreno es una cota inferior de |d| con el signo correcto. Excluye la geocerca.
- `World.ground(xy (N,2)) -> (N,)`: altura del terreno (bilineal).
- `World.raycast(o (N,3), d (N,3) unitarios, max_range float | (N,)) -> (dist (N,), prim (N,) int, normal (N,3))`: primer impacto. Sin impacto: `inf`, `-1`, `0`. Terreno: `prim = World.TERRAIN` (−2). Origen dentro de un obstáculo: `dist = 0` y `normal = 0`.
- `World.segment_crossings(a (N,3), b (N,3)) -> (N,) int`: número de primitivas que toca cada segmento, +1 si pasa bajo el terreno. Es la línea de vista de la radio y del detector (0 = visible).
- `World.bounds -> (lo (3,), hi (3,))` (geocerca), `World.anchors -> dict` (`{"A": (x,y,z), "B": (x,y,z)}`, en el suelo) y `World.tags -> list[str]` (una por primitiva).
- `World.to_dict()`, `World.from_dict(d)` y `World.hash -> str` (sha256 del JSON canónico).
- `generate(biome, size_m, density, seed) -> World`.

### Formato `c2lab-world/1` (JSON; lo leen el visor y el hash)
```
{"format": "c2lab-world/1", "biome": str, "seed": int, "density": float,
 "bounds": [[0,0,0], [sx,sy,sz]], "anchors": {"A": [x,y,z], "B": [x,y,z]},
 "terrain": {"heights": [[...], ...]},          # nx × ny ≥ 2 × 2 nodos que cubren bounds en xy; plano = [[0,0],[0,0]]
 "primitives": [{"type": ..., "tag": str, ...}]}
box {c, h, q} · cylinder {c, r, hh, q} · capsule {a, b, r} · sphere {c, r} · cone {c, r, h, q} · torus {c, R, r, arc, q}
```
`q` es un cuaternión wxyz de local a mundo. Las convenciones de cada tipo están en `.ia/grafo/mundo.md`. Valores redondeados a 6 decimales.

### `TrueState` y `Physics`: `src/c2lab/physics/` (F2); lo consumen los sensores y las misiones (F3, F5)
`Drones(profile, n, rng, world, mode="quad"|"pointmass", wind, gust_sigma, gust_tau, dt)`. Atributos sobre n drones: `p (n,3)` ENU m, `v (n,3)` m/s, `q (n,4)` wxyz de cuerpo a mundo, `w (n,3)` rad/s en cuerpo, `f (n,3)` fuerza específica en cuerpo (lo que mide un acelerómetro), `alive (n,)`, `landed (n,)`, `omega (n,4)` rad/s y `energy (n,)` J.
Métodos: `reset(p0, yaw0, landed=True)`, `set_setpoint(i, Velocity | Position | None)` y `step(tick, err=None) -> [(tick, i, kind)]`. `err` son los errores del estimador (`Sensors.err`) que ve el autopiloto; `None` significa ideal. Los eventos `kind` son `takeoff`, `landed`, `crash_obstacle`, `crash_terrain`, `crash_drone` y `geofence`. `PROFILES = {"cf2", "iris"}`. Detalle en `.ia/grafo/fisica.md`.

## Archivos y reglas para agentes

- Escribir solo los archivos permitidos en el spec del clúster. `src/c2lab/brain.py`, `pyproject.toml`, `CLAUDE.md` y este índice son de solo lectura.
- No añadir dependencias. De ejecución: numpy, scipy y numba (núcleos calientes); de tests: pytest.
- Check común: `.venv/Scripts/python -m pytest -q` desde la raíz, en verde.
- Deltas: añadir líneas al final de la sección `## Deltas` del clúster con el formato `N <nodo>` (nuevo), `R <a> -> <b>: <relación>` (arista nueva), `A <nodo>: <cambio>` (actualizado) y `X <nodo>` (eliminado).

### `Sensors`: `src/c2lab/sensors.py` (F3)
`Sensors(cfg, n, seed, ids, world, drones, labels, home)`. `cfg` es la tabla `sensors` del equipo (ya validada). `ids` son los agent_id (para las semillas `("agent", id, sensor)`), `labels` el índice de equipo por dron (para el detector) y `home (n,3)` el punto de despegue (origen del marco local).
- `step(tick)`: muestrea los sensores cuyo periodo divide el tick y actualiza `est`.
- `pop(i) -> dict[sensor, tuple[Reading, ...]]`: lecturas del dron i desde la última llamada (arrays de solo lectura).
- `err`: errores del estimador `pose` (`dp (n,3)`, `dv (n,3)`, `dyaw (n,)`, vistas que se actualizan a la tasa de `pose`), o `None` si es ideal o está apagado. Se pasa a `Drones.step` y lo usa el autopiloto.
- `public(name) -> dict`: parámetros públicos para `AgentSpec.sensors`.
Formato de `Reading.data`: `tof (arrays, z, z)` m, NaN sin blanco · `lidar (canales, n_az)` m · `depth (H, W)` profundidad z m · `detector (k, 5)` filas (azimut, elevación, distancia, clase 0 = mismo equipo / 1 = otro, confianza) en cuerpo · `pose (13,)` p_local, v_local, q, w · `imu (6,)` f, w en cuerpo · `gnss (6,)` p, v ENU (NaN sin fix) · `baro ()` altura sobre el despegue · `mag ()` guiñada rad · `prior_map (nx, ny)` alturas 2,5D (solo en el tick 0) · `battery ()` estado de carga 0..1.

### `Radio`: `src/c2lab/radio.py` (F3)
`Radio(cfg, n, rng, world, slot_dt, gains_db=None)`. `send(i, outbox)` encola paquetes; los que superan `packet_bytes`, o pasan de `max_packets` por ranura, se descartan y se cuentan en `dropped`. `slot(p, alive) -> list[list[Packet]]` entrega lo encolado en la ranura anterior: latencia de 1 ranura, en orden de emisor. Perfiles `ideal`, `disco` (`range_m`, `loss`) y `dil_v02` (port de swarm-dil v0.2 con línea de vista 3D `segment_crossings`). `ranging` añade `range_m` con σ 0,1 m. Contadores: `sent`, `delivered` y `dropped` en bytes por nodo.
