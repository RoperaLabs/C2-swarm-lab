# C2-swarm-lab: plan global

Banco de pruebas 3D para algoritmos C2 (mando, control y comunicación) de enjambres de drones. Es la base de pruebas de las próximas versiones de GRACE.

Versión del plan: 2 (2026-11-18). Pendiente de aprobación.

---

## 0. Decisiones de partida

| Tema | Decisión | Origen |
|---|---|---|
| Stack | Simulador sin interfaz en Python (numpy) + visor 3D three.js en el navegador | Decidido el 2026-11-18 |
| Mundo | Espacio continuo con formas variadas y realistas. No se entrega ninguna rejilla privilegiada al cerebro | Decidido el 2026-11-18 |
| Dinámica | Física de cuadricóptero (sólido rígido, motores con retardo, arrastre, viento). Hay además un modo de masa puntual para aislar efectos (§F7.4) | Decidido el 2026-11-18 |
| Nombre | `C2-swarm-lab` | Decidido el 2026-11-18 |
| Visión natural | Cámara de profundidad por trazado de rayos más un **detector de agentes** (rumbo y distancia con ruido; es lo que entrega una cámara con un detector a bordo). Los píxeles RGB quedan en la cartera | Opción por defecto (sin objeción) |
| Radio | Se porta el modelo de swarm-dil v0.2 (`radio.py`: pérdida por trayecto, sombra AR(1), bloqueo de línea de vista y contención) y se añaden los perfiles `ideal` y `disco` | Opción por defecto |
| Visor | Primero reproduce grabaciones. El modo en vivo es opcional en F6 | Opción por defecto |

### Regla de oro: lo que funciona aquí se enchufa a un dron real

El cerebro solo recibe lo que un dron real mediría: lecturas de sensores con ruido, en su marco de referencia y a su frecuencia, y paquetes de bytes recibidos por radio. Nunca recibe la geometría del mundo, una rejilla de vóxeles ni coordenadas globales, salvo que haya encendido un sensor que las produzca (GNSS o mapa previo). Los vóxeles derivados del mundo existen solo para el **oráculo**, que calcula las métricas y la viabilidad, y nunca llegan al cerebro. Si un cerebro necesita un mapa, lo construye él mismo con sus sensores, igual que lo haría en vuelo.

Esta regla se comprueba de tres formas:
1. Un test de importaciones (AST): `brains/` solo puede importar `c2lab.brain`, la biblioteca estándar, numpy y scipy.
2. Un test en ejecución: las observaciones son copias de arrays de solo lectura y tipos planos, sin referencias a objetos del mundo.
3. La información de misión es coherente con los sensores encendidos: sin GNSS, la meta llega en el marco local de despegue.

### Lo que pediste y cómo se configura

| Característica pedida | Clave en el escenario | Opciones |
|---|---|---|
| Ray tracing | `sensors.tof`, `sensors.lidar`, `sensors.depth` | Encendido o apagado, zonas, FoV, alcance, frecuencia y ruido |
| Visión natural | `sensors.depth` + `sensors.detector` | FoV, alcance, probabilidad de detección, falsos positivos |
| Posicionamiento | `sensors.pose` (estimador), `sensors.gnss`, `radio.ranging` | Perfiles `ideal`, `ekf`, `vio` y `flujo`; GNSS con ruido o denegado; distancias UWB entre pares |
| Memoria | `memory_bytes` por equipo | Presupuesto declarado y medido. 0 significa cerebro reactivo puro |
| Conocimiento del entorno | `sensors.prior_map` | Mapa 2,5D degradado: resolución, ruido y omisión de obstáculos finos o voladizos |
| Comunicación y su distancia | `radio.profile`, `radio.packet_bytes` | `ideal`, `disco` (alcance R y pérdida p) o `dil_v02` (realista) |
| Cerebro intercambiable | `brain = "brains.boids:Boids"` | Una línea por equipo. Admite equipos heterogéneos |
| Coordinador | `[ground_station] brain = "..."` | Nodo de tierra con radio y sin vuelo, para comparar C2 centralizado frente a distribuido |
| Objetivos de la prueba | `scenario.mission` | `a_to_b`, `coverage`, `pursuit` (ampliable) |
| Repetibilidad | `scenario.seed` | Misma semilla y mismo cerebro dan la misma grabación bit a bit |

---

## 1. Hardware y entorno (diagnóstico del 2026-11-18)

| Recurso | Valor | Presupuesto (≤ 85 %) |
|---|---|---|
| CPU | i7-7700HQ, 4 núcleos y 8 hilos | **≤ 5 procesos de simulación** en paralelo (con 6 se midió un 87 %) |
| RAM | 15,9 GB (8 GB libres) | ≤ 1 GB por proceso, ≤ 5 GB en total |
| Disco | 622 GB libres | Las grabaciones se comprimen y se pueden purgar |
| GPU | GTX 1050 + HD 630 | Solo para el visor (WebGL) |
| Python | 3.14.5 con py launcher (también 3.12, 3.13 y 3.10) | `.venv` propio del proyecto |
| Paquetes | numpy 2.5.5 y scipy 1.18.1 en venv. **numba 0.68.1 tiene rueda cp314 para Windows** (verificado) | numba solo entra si un benchmark lo exige |
| Node | 24.16 | No se usa: el visor no tiene paso de build |

Aviso: el numpy del site-packages de usuario está roto (`numpy._utils` no existe). El `.venv` lo aísla, pero conviene repararlo o eliminarlo fuera del proyecto.

Concurrencia de agentes (AMTOP): como máximo **3 subagentes** a la vez y **5 procesos** de test o simulación. Ningún servidor ni proceso queda vivo fuera de su tarea.

---

## 2. Arquitectura

### 2.1 Estructura de carpetas

```
C2-swarm-lab/
  CLAUDE.md                 reglas del proyecto (regla de oro, convenciones, comandos)
  PLAN.md                   este documento
  pyproject.toml            numpy; extra test (pytest). scipy, numba y swarm-dil entran en la fase que los necesite
  .ia/grafo/                grafo de contratos y clústeres para los agentes (fuente de verdad)
  src/c2lab/
    brain.py                contrato del cerebro y del coordinador (tipos planos); lo único que ve brains/
    seeds.py                árbol de semillas (SeedSequence) por subsistema
    scenario.py             carga y validación del TOML (claves, valores por defecto, periodos enteros)
    world/
      primitives.py         SDF, AABB, intersección de rayos y serialización de cada primitiva
      raycast.py            consultas por lotes: rayos, segmentos (línea de vista) y SDF de puntos
      biomes.py             generadores procedurales (mixto, urbano, bosque, interior)
      oracle.py             voxelización inflada, viabilidad A-B y camino más corto (solo métricas)
    physics/
      quad.py               cuadricóptero vectorizado sobre N drones, viento y energía
      pointmass.py          modo de masa puntual con límites de velocidad y aceleración
      autopilot.py          cascada posición → velocidad → actitud → velocidad angular → mezclador
    sensors/                un archivo por sensor; todos comparten el patrón (tasa, flujo de ruido, salida)
    radio.py                perfiles ideal, disco y dil_v02; ranging; límites de paquete y ancho de banda
    missions/               a_to_b.py, coverage.py, pursuit.py y el contrato común Mission
    sim.py                  bucle multitasa determinista (ticks enteros)
    record.py               grabación (npz + jsonl + meta) y exportación para el visor
    cli.py                  run, sweep, summarize, export-viewer, serve
  brains/                   cerebros enchufables: baselines, evasor y adaptadores GRACE
  scenarios/                pruebas en TOML
  viewer/                   index.html + main.js + three.js vendorizado (versión fijada)
  tests/
  graphify-out/             grafo del código, regenerable con `graphify update .` (no se versiona)
```

### 2.2 Convenciones (fijadas en F0 y obligatorias)

- **Unidades SI.** Mundo en **ENU** (x este, y norte, z arriba) y cuerpo en **FLU** (adelante, izquierda, arriba), según REP-103. Así el paso a ROS 2 es directo. Cuaterniones en orden (w, x, y, z).
- **Tiempo en ticks enteros** de física: **1200 Hz** por defecto (divisible por 10, 15, 30, 50, 400 y 600, las tasas de los sensores reales). Toda frecuencia (control, sensores, cerebro, radio, grabación) debe dar un periodo entero de ticks; si no, el escenario no carga y el error nombra la clave. No se acumula tiempo en coma flotante.
- **Árbol de semillas:** `SeedSequence(seed)` se divide en mundo, misión, viento, radio y un flujo por agente y por sensor. Encender o apagar un sensor no altera el mundo ni el ruido de ningún otro sensor ni agente.
- **Determinismo:** misma semilla, mismo código y misma plataforma dan un hash idéntico de la grabación. El tiempo de cómputo real del cerebro se mide pero nunca afecta a la simulación. Si se quiere modelar latencia de cómputo, sale de un recuento de operaciones declarado y no del reloj.
- Identidad de commits: ep2si10n con dirección noreply, UTC, sin coautor (salvo orden explícita).

### 2.3 Bucle de simulación (orden dentro de un tick)

```
para cada tick k:
  1. si k es tick de cerebro: construir todas las observaciones (lecturas nuevas + bandeja de radio)
     y después llamar a step() de cada cerebro en orden de id. Es síncrono: el resultado no depende del orden.
     Las consignas pasan al autopiloto y los paquetes salientes a la cola de radio.
  2. autopiloto (a sus tasas) con el estado ESTIMADO → velocidades de motor
  3. física: motores, fuerzas, viento, integración, colisiones → eventos
  4. sensores cuyo periodo divide k: muestrean el estado VERDADERO + su flujo de ruido
  5. radio, si es tick de ranura: entrega los paquetes (latencia ≥ 1 ranura, orden fijo por emisor)
  6. misión: llegadas, capturas, cobertura y fin de episodio
  7. grabación, si es tick de grabación (30 Hz por defecto)
```

Si un cerebro lanza una excepción, ese agente entra en **failsafe** (mantiene posición y aterriza), el evento se registra con su traza y la simulación continúa.

### 2.4 Contrato del cerebro (escrito en F0 en `src/c2lab/brain.py`; congelado en F4)

```python
@dataclass(frozen=True)
class AgentSpec:            # lo que el dron sabe "de fábrica" más su orden de misión
    agent_id: int
    team: str
    role: str               # "nav" | "pursuer" | "evader" | "ground" | ...
    seed: int               # semilla propia del cerebro (sale del árbol de semillas)
    sensors: dict           # sensores encendidos y sus parámetros públicos (FoV, alcance, tasa, montaje)
    limits: dict            # brain_hz, packet_bytes, memory_bytes, vmax, amax
    mission: dict           # p. ej. goal_local=(x, y, z) en el marco de despegue; goal_enu solo con GNSS

@dataclass(frozen=True)
class Observation:
    t: float
    readings: dict          # sensor -> tupla de Reading(t, data de solo lectura), una por muestra desde el último paso
    inbox: tuple            # Packet(src, data: bytes, rssi_dbm, range_m | None)

@dataclass(frozen=True)
class Action:
    setpoint: Velocity | Position | None = None   # Velocity(v, yaw_rate, frame) | Position(dp, yaw); None = mantener
    outbox: tuple = ()      # bytes (difusión) o (dest, bytes); cada uno ≤ packet_bytes

class Brain(Protocol):      # cualquier clase con estos métodos vale; se construye con brain_params del TOML
    def reset(self, spec: AgentSpec) -> None: ...
    def step(self, obs: Observation) -> Action: ...
    # opcionales: state_bytes() -> int (memoria persistente) y export_map() (el mapa propio, para puntuar el mapeado)
```

La consigna de actitud y empuje y el estado de vuelo se quitaron del contrato por ahora: ningún cerebro planificado los usa y añadirlos más tarde no rompe nada.

El coordinador (estación de tierra) implementa el mismo protocolo sin consigna de vuelo: solo recibe radio y, opcionalmente, el mapa previo.

Un paquete que excede `packet_bytes` se descarta y se cuenta como violación, igual que lo rechazaría una radio real. Nunca se trunca en silencio.

### 2.5 Escenario de ejemplo (TOML)

```toml
[scenario]
name = "ab_urbano_10"
seed = 7
duration_s = 180
mission = "a_to_b"
rates_hz = { physics = 1200, brain = 10, radio = 10, record = 30 }

[world]
biome = "urbano"
size_m = [200, 200, 60]
density = 0.25

[radio]
profile = "dil_v02"
packet_bytes = 64
ranging = false

[[team]]
name = "azul"
count = 10
drone = "iris"              # perfil de parámetros físicos
brain = "brains.boids:Boids"
memory_bytes = 256
sensors.tof = { arrays = ["front", "left", "right", "down"], zones = 8, fov_deg = 45, range_m = 4.0, hz = 15 }
sensors.pose = { profile = "vio" }
sensors.detector = { fov_deg = 90, range_m = 30, hz = 10 }
sensors.gnss = false
```

---

## 3. Fases: vista general y dependencias

```
F0 ──┬──> F1 mundo + raycast + visor ──┬──> F3 sensores + radio ──> F4 cerebros ──┬──> F5 misiones ──┬──> F7 GRACE
     └──> F2 física + autopiloto ──────┘                                          └──> F6 lotes + visor ┘
```

F1 y F2 corren en paralelo: F2 se prueba en un mundo vacío y la colisión contra el SDF de F1 se integra al final. F5 y F6 también son paralelas, aunque el resumen estadístico de F6 consume las métricas de F5.

| Fase | Tamaño | Razonamiento de Sonnet | Hito de versión |
|---|---|---|---|
| F0 Cimientos y contrato | S | bajo (el contrato lo diseña Opus) | — |
| F1 Mundo procedural, raycast y visor estático | L | medio-alto | — |
| F2 Física de dron y autopiloto | L | alto | — |
| F3 Sensores y radio | L | medio | — |
| F4 Contrato congelado y cerebros base | M | medio | **v0.1.0 "Primer vuelo"** |
| F5 Misiones y métricas | M | medio | — |
| F6 Lotes, estadística y visor completo | M | bajo-medio | **v0.2.0** |
| F7 GRACE-3D 2.0.0 enchufado y escalera de realismo | L | alto (con análisis de Opus) | **v0.3.0** |

Formato de cada fase: objetivo, entradas, tareas (con el nivel sugerido: H = Haiku, S = Sonnet), criterios de aceptación medibles, entradas y salidas del grafo, y riesgos.

---

## F0. Cimientos y contrato

**Objetivo:** un repositorio vacío pero correcto, donde el contrato del cerebro y la interfaz de consultas del mundo ya estén escritos, para que F1, F2 y F3 programen contra ellos desde el primer día.

| # | Tarea | Nivel |
|---|---|---|
| 0.1 | `git init`, `.gitignore`, `pyproject.toml` (numpy; extra `test`) y `.venv` con 3.14 | Opus |
| 0.2 | `CLAUDE.md` del proyecto: regla de oro, convenciones de §2.2, comandos y protocolo de limpieza | Opus |
| 0.3 | Skills: PonyTail, Graphify, Caveman y Engineering ya están instaladas a nivel de usuario y se citan en `CLAUDE.md`. Grafo de contratos en `.ia/grafo/` (lo leen los agentes) y grafo del código con `graphify update .` | Opus |
| 0.4 | `c2lab/brain.py`: tipos de §2.4 (dataclasses congeladas, sin lógica) | Opus |
| 0.5 | Interfaz del mundo (`WorldQuery`) y estado físico (`TrueState`), escritos como contrato en `.ia/grafo/_indice.md` y no como código vacío. F1 y F2 los implementan | Opus |
| 0.6 | `seeds.py`: árbol de semillas con nombres estables más su test de independencia | S (fullstack) |
| 0.7 | Test de aislamiento (AST sobre `brains/`) y `scenario.py` (carga y validación del TOML: claves, valores por defecto, periodos enteros) | S (fullstack) |

El servidor del visor (`.claude/launch.json`) pasa a F1, junto con el visor: en F0 no hay nada que servir.

**Aceptación**
- `pip install -e .[test]` en un venv limpio y `pytest` en verde (aislamiento, semillas y validación del TOML).
- Un TOML con una frecuencia que no da un periodo entero de ticks se rechaza con un error que nombra la clave, el valor y la frecuencia de física.
- Primer commit con la identidad correcta.

**Grafo** · entra: este PLAN · sale: nodos `BrainContract`, `WorldQuery`, `SeedTree`, `Scenario` y sus aristas `consume` / `produce`.

---

## F1. Mundo procedural, raycast y visor estático

**Objetivo:** mundos 3D continuos, variados y realistas generados por semilla, con consultas geométricas exactas y rápidas, y que se puedan ver en el navegador.

### Primitivas (cada una con SDF exacta, AABB, intersección de rayos, serialización y malla del visor)

| Primitiva | Uso realista | Intersección |
|---|---|---|
| Caja orientada | Edificios, muros, losas, estanterías, vigas | Analítica (slabs en el marco local) |
| Cilindro con tapas | Columnas, silos, troncos, torres | Analítica |
| Cápsula | Postes, **cables** (finos: un reto real para ToF) y ramas | Analítica |
| Esfera | Copas, depósitos | Analítica |
| Cono | Copas de conífera, tejados | Analítica |
| Toro con arco | Anillos y arcos (como en la imagen) | Trazado de esferas sobre la SDF, con un refinado final |
| Campo de alturas | Terreno con colinas y montículos | Avance por pasos más bisección |

No se usa CSG. Un muro con ventanas o puertas se compone de varias cajas, lo que simplifica la SDF, el raycast y el visor.

### Biomas (plantillas parametrizadas por densidad y distribuciones de tamaño)

| Bioma | Contenido | Escala por defecto |
|---|---|---|
| `mixto` | Columnas, anillos, arcos, losas flotantes y muros con ventana (tipo imagen) | 60 × 40 × 20 m |
| `urbano` | Manzanas, edificios de altura log-normal, calles, postes con cables, árboles, pasarelas y puentes | 200 × 200 × 60 m |
| `bosque` | Troncos con copa, sotobosque bajo y terreno ondulado | 150 × 150 × 30 m |
| `interior` | Nave con techo (GNSS inútil), estanterías, columnas, vigas y paredes con puertas | 40 × 30 × 8 m |

Los cuatro biomas comparten la **geocerca** del recinto: salir de ella cuenta como violación.

| # | Tarea | Nivel |
|---|---|---|
| 1.1 | Primitivas: SDF exacta y AABB de los 7 tipos, más serialización a dict canónico | H (fórmulas dadas) |
| 1.2 | Intersección de rayos por tipo, vectorizada sobre lotes de rayos | S |
| 1.3 | `raycast.py`: descarte por esfera de alcance + AABB, narrowphase por tipo y `segment_crossings` (para la línea de vista de la radio) | S |
| 1.4 | Referencia de validación: trazado de esferas por fuerza bruta (lento, solo para tests) | H |
| 1.5 | **Spike de rendimiento:** medir rayos/s con numpy. Si no llega al objetivo, núcleo en numba (`fast`) | S |
| 1.6 | `biomes.py`: `mixto` y `urbano` primero, `bosque` e `interior` después | S |
| 1.7 | `oracle.py`: voxelización inflada por el radio del dron (0,5 m por defecto), conectividad A-B, puntos de salida libres y remuestreo determinista con subsemilla si no es viable | S |
| 1.8 | Hash del mundo (sha256 del JSON canónico) más `world.json` | H |
| 1.9 | Visor estático: three.js vendorizado, órbita, primitivas con instancing, recinto translúcido y A/B marcados, más `.claude/launch.json` para servirlo | S |

**Aceptación**
- Repetibilidad: misma semilla da el mismo hash (100 de 100 semillas por bioma). 200 semillas distintas dan 200 hashes distintos.
- Viabilidad: 200 de 200 semillas por bioma tienen un camino A-B con holgura ≥ radio del dron + 0,2 m y salidas libres. La generación tarda ≤ 2 s por mundo de 500 primitivas.
- SDF: 10⁴ puntos aleatorios por tipo; el signo coincide con la pertenencia analítica en el 100 % y |∇SDF| ≤ 1 + 10⁻⁶ por diferencias finitas. El campo de alturas, que no es exacto, se comprueba solo como cota.
- Raycast frente a la referencia en 10⁵ rayos: acierto/fallo coincide en ≥ 99,99 % y el error de distancia es ≤ 1 mm. Casos analíticos: rayo por el hueco del toro (fallo), rasante a la caja y origen dentro de un obstáculo.
- Rendimiento: **≥ 300 000 rayos/s** con alcance 10 m en una escena de 500 primitivas, en un proceso.
- Variedad: la densidad ocupada por bioma tiene la media configurada ± 10 % y desviación típica > 0 entre semillas.
- Visor: muestra los 4 biomas a ≥ 30 fps con 1000 primitivas en este portátil (comprobado con captura).

**Grafo** · entra: `WorldQuery` · sale: `Primitive:*`, `Biome:*`, `Raycaster`, `Oracle`, `WorldFile`, `Viewer`.

**Riesgos:** el rendimiento de numpy con muchos rayos (mitigación: descarte por alcance y numba como reserva verificada); toros y terreno más lentos (mitigación: aparecen poco y se trazan solo en la región descartada).

---

## F2. Física de dron y autopiloto

**Objetivo:** drones que vuelan como drones, con un autopiloto de bajo nivel que acepta consignas de cerebro, vectorizado sobre N drones.

### Modelo

- **Estado por dron:** posición, velocidad, cuaternión, velocidad angular, velocidad de los 4 rotores y energía consumida.
- **Fuerzas:** empuje Σ k_f·Ω² en el eje z del cuerpo, gravedad, arrastre lineal y cuadrático en ejes de cuerpo y **viento** (medio + ráfagas Ornstein-Uhlenbeck sembradas por dron). El efecto suelo es opcional.
- **Pares:** brazos en X, guiñada por k_m·Ω².
- **Motores:** primer orden (τ_m de 20 a 50 ms) con saturación [Ω_min, Ω_max].
- **Integración:** semi-implícita a 1200 Hz con renormalización del cuaternión. RK4 solo si el test de convergencia lo exige.
- **Energía:** potencia por rotor según la teoría del momento con un rendimiento η. La batería es opcional.
- **Colisión:** el dron es una esfera de radio r_d. Choca si SDF(p) < r_d, si dos drones están a menos de 2·r_d o si sale de la geocerca. Por defecto, choque = dron perdido. Posarse en el suelo al despegar o aterrizar no cuenta como choque.
- **Perfiles de parámetros** (TOML, con la fuente citada): `cf2` (Crazyflie 2.x, 27 g, misma MCU que el presupuesto de GRACE-3D; parámetros de `cf2x.urdf` de gym-pybullet-drones) e `iris` (3DR Iris, 1,5 kg; parámetros de `iris.sdf` del SITL de PX4). Ambas fuentes verificadas el 2026-11-18.
- **Masa puntual:** misma interfaz, solo con límites de velocidad y aceleración. Sirve para depurar algoritmos y para la escalera de realismo de F7.

### Autopiloto (cascada al estilo PX4)

Posición (50 Hz) → velocidad (PID, 50 Hz) → aceleración deseada → actitud (600 Hz) → velocidad angular (PID, 1200 Hz) → mezclador → Ω. Límites: inclinación máxima, velocidad horizontal y vertical máximas, y velocidad de guiñada. **Trabaja con el estado estimado** (el perfil del estimador viene de F3), no con el verdadero; el modo ideal es configurable.

| # | Tarea | Nivel |
|---|---|---|
| 2.1 | Perfiles `cf2` e `iris` con parámetros documentados y su fuente | Opus |
| 2.2 | `quad.py` vectorizado: fuerzas, pares, motores e integración | S |
| 2.3 | Viento (OU sembrado) y energía | H |
| 2.4 | `autopilot.py`: cascada, límites y sintonía por perfil | S |
| 2.5 | `pointmass.py` | H |
| 2.6 | Colisiones contra el SDF de F1, entre drones y con la geocerca, como eventos | S |
| 2.7 | Benchmark y, si falla, núcleos en numba | S |

**Aceptación**
- Caída libre (motores parados, sin arrastre): |a_z + g| < 10⁻⁹.
- Vuelo estacionario (`iris` y `cf2`, estimador ideal, sin viento): deriva < 1 cm en 30 s y motores dentro de sus límites.
- Escalón de velocidad de 0 a 5 m/s (`iris`): sobreoscilación ≤ 10 %, subida al 90 % en ≤ 1,5 s, error estacionario ≤ 0,05 m/s y la inclinación nunca supera el máximo.
- Mantener posición con viento de 5 m/s y ráfagas de σ = 1 m/s: error RMS ≤ 0,5 m.
- Convergencia: al reducir dt a la mitad, la posición tras una maniobra de 10 s cambia < 1 cm.
- Colisión a 2 m/s contra un muro: el evento cae en el tick analítico ± 1. Las colisiones entre drones se detectan.
- Potencia en vuelo estacionario: dentro de ± 10 % de la teoría del momento.
- Determinismo: hash del estado idéntico para la misma semilla.
- Rendimiento: 10 drones con física y autopiloto a 1200 Hz van **≥ 20 veces más rápido que el tiempo real** en un proceso.

**Grafo** · entra: `WorldQuery.sdf` · sale: `Drone`, `DroneProfile:*`, `Autopilot`, `Wind`, `CollisionEvent`.

**Riesgos:** la sintonía del autopiloto por perfil (mitigación: criterios de escalón como test); el coste de 1200 Hz en numpy (mitigación: numba, o 600 Hz si la convergencia lo permite).

---

## F3. Sensores y radio

**Objetivo:** cada capacidad pedida existe como sensor conmutable, con frecuencia propia, ruido sembrado y salida en el marco del dron.

Los valores de la tabla son de partida y se contrastan con la hoja de datos en la tarea 3.1.

| Sensor | Salida (marco) | Parámetros y ruido por defecto |
|---|---|---|
| `tof` multizona (tipo VL53L5CX) | Matriz zonas×zonas de distancias por array montado (cuerpo). Sin blanco: NaN | 8×8 o 4×4, 45°×45°, de 0,02 a 4 m, 15 Hz; σ = máx(5 mm, 3 %·d); k×k subrayos por zona que informan del más cercano; pérdida de zona con probabilidad p |
| `lidar` (giratorio, tipo Mid-360) | Nube de puntos o rangos por haz (cuerpo) | Canales, FoV vertical, resolución, alcance hasta unos 40 m, σ ≈ 2 cm, 10 Hz |
| `depth` (tipo D435) | Imagen de profundidad (cámara) | W×H (64×48 por defecto), 87°×58°, de 0,3 a 10 m, σ ∝ z², 15 Hz |
| `detector` (visión de agentes) | Lista de (azimut, elevación, distancia, clase, confianza) (cuerpo) | FoV, alcance, P_det(d), ruido angular y radial, falsos positivos, confusión amigo/objetivo. **Exige línea de vista** |
| `pose` (estimador del autopiloto) | Posición, velocidad y actitud en el **marco local de despegue** | `ideal`, `ekf` (ruido pequeño), `vio` (deriva ∝ distancia recorrida), `flujo` (válido solo por debajo de una altura máxima) |
| `imu` | Lote de muestras de aceleración y giro desde el último paso (cuerpo) | Densidad de ruido + paseo aleatorio del sesgo, 400 Hz |
| `gnss` | Posición y velocidad ENU | 5 Hz, error Gauss-Markov σ_h ≈ 1,5-2,5 m, latencia, **zonas de denegación**; degradación por cielo obstruido (rayos al cenit) como opción |
| `baro` | Altitud | σ ≈ 0,1-0,3 m y deriva lenta |
| `mag` | Rumbo | σ configurable |
| `prior_map` | Mapa 2,5D de alturas (ENU) entregado en `reset` | Resolución, ruido en altura, omisión de obstáculos finos o voladizos y desplazamiento |
| `battery` | Estado de carga | Desde la energía de F2 |

### Radio (`radio.py`)

| Perfil | Comportamiento |
|---|---|
| `ideal` | Entrega garantizada, sin alcance límite ni pérdidas: aísla el algoritmo |
| `disco` | Alcance R, pérdida de Bernoulli p y tope de ancho de banda |
| `dil_v02` | Port de swarm-dil v0.2: pérdida por trayecto, sombra AR(1) por enlace, **bloqueo de línea de vista con `segment_crossings` 3D**, SINR logística y contención por tiempo de aire |

Común a los tres perfiles: tamaño máximo de paquete (64 B por defecto), número máximo de paquetes por ranura, latencia ≥ 1 ranura, difusión o unidifusión y `ranging` opcional (distancia entre pares con σ ≈ 10 cm, tipo UWB).

| # | Tarea | Nivel |
|---|---|---|
| 3.1 | Patrón común de sensor (tasa, flujo de ruido, montaje y salida) y fichas de parámetros con su fuente | S |
| 3.2 | `tof`, `lidar` y `depth` sobre el raycaster de F1 (todos son el mismo sensor de rayos con distinta disposición) | S |
| 3.3 | `detector` con línea de vista vía `segment_crossings` | S |
| 3.4 | `pose`, `imu`, `gnss`, `baro`, `mag` y `battery` (modelos de ruido) | H × 2 |
| 3.5 | `prior_map` a partir del oráculo con degradación | H |
| 3.6 | `radio.py`: port de dil_v02 + perfiles `ideal` y `disco` + ranging + límites | S |
| 3.7 | Tests de no fuga en ejecución y de independencia de flujos | H |

**Aceptación**
- Conmutación: un sensor apagado no aparece en la observación y no cuesta tiempo. Encender o apagar un sensor deja idénticos el hash del mundo y las muestras de todos los demás sensores (100 %).
- Ruido: 10⁴ muestras estáticas por sensor. σ empírica dentro de ± 5 % de la configurada, sesgo dentro de 3 errores estándar, y autocorrelación de los procesos Gauss-Markov y de paseo aleatorio dentro de ± 10 % en el retardo τ.
- Frecuencias: marcas de tiempo exactas en ticks enteros.
- `tof` contra un muro a distancia y ángulo conocidos: lectura esperada ± ruido. Fuera de alcance: NaN.
- `depth` contra un plano frontal: profundidad constante ± ruido. Las aristas de un cubo proyectan según el modelo estenopeico (± 1 píxel).
- `detector`: P_det empírica dentro de su intervalo de confianza. **0 detecciones a través de muros** en 10⁴ casos. Bordes del FoV correctos.
- Radio: `ideal` entrega el 100 %. `disco` entrega 0 más allá de R y su pérdida cae dentro del intervalo de confianza. `dil_v02` reproduce la curva entrega/distancia de swarm-dil con los mismos parámetros (± 2 %). Los paquetes que exceden el límite se descartan y se cuentan.
- No fuga: las observaciones solo contienen tipos planos y arrays de solo lectura (escaneo de tipos), y la meta va en el marco local cuando no hay GNSS.

**Grafo** · entra: `Raycaster`, `Drone`, `SeedTree` · sale: `Sensor:*`, `Radio:*`, `Observation`.

---

## F4. Contrato congelado y cerebros base: v0.1.0 "Primer vuelo"

**Objetivo:** un enjambre vuela de A a B en un mundo procedural con un cerebro cambiado por una sola línea, y se ve en el visor.

| # | Tarea | Nivel |
|---|---|---|
| 4.1 | `sim.py`: bucle multitasa de §2.3, failsafe y medición del tiempo de `step()` y de `state_bytes()` | S |
| 4.2 | Carga dinámica `"modulo:Clase"` por equipo y equipos heterogéneos | H |
| 4.3 | Cerebros base: `hover`, `random_walk`, `potential_field` (meta + repulsión por ToF o profundidad) y `boids` (Reynolds + meta + evasión; vecinos vía `detector` o vía radio, para mostrar la conmutación) | S |
| 4.4 | `oracle` **privilegiado**: sigue el camino del oráculo. Vive en `brains/privileged/`, excluido a propósito del test de aislamiento, y queda marcado en los resultados. Solo sirve para validar métricas | H |
| 4.5 | `record.py` (meta.json, npz de trayectorias, eventos jsonl) + exportación al visor y reproducción básica (play/pausa) | S |
| 4.6 | `docs/brain_api.md` con un cerebro de ejemplo de ≤ 30 líneas | H |

**Aceptación**
- Cambiar de cerebro es cambiar una línea del TOML. El test de aislamiento pasa con todos los cerebros no privilegiados.
- Un cerebro que lanza una excepción entra en failsafe y la simulación termina con normalidad (test).
- Sanidad en el bioma `mixto` con 20 semillas: `oracle` llega ≥ 95 % y `potential_field` o `boids` llegan ≥ 70 %. No es un listón de calidad, solo prueba que la tubería funciona.
- Misma semilla, mismo hash de grabación.
- **Rendimiento de referencia:** 10 drones, bioma urbano de 500 primitivas, `tof`×4 + `pose` + radio `dil_v02` y cerebro `boids` van **≥ 5 veces más rápido que el tiempo real** en un proceso. 50 drones, ≥ 1 vez.
- El visor reproduce una misión completa.

**Grafo** · entra: todo F1 a F3 · sale: `Sim`, `Brain:*`, `Recording`. El nodo `BrainContract` se marca como congelado.

---

## F5. Misiones y métricas

**Contrato de misión:** `setup(world, rngs, teams)` devuelve salidas, metas y roles. Después vienen `on_tick(state)` (genera eventos), `done(state)` y `metrics(recording)`. La información que se entrega al cerebro respeta la regla de oro.

| Misión | Definición | Métricas |
|---|---|---|
| `a_to_b` | Zona de salida A, región meta B y lanzamiento escalonado opcional | % de llegadas, tiempo, **regret** = longitud recorrida / camino más corto del oráculo − 1, colisiones, bytes enviados, energía |
| `coverage` (mapeado) | Recorrer una región con un tiempo límite | **Cobertura de superficie medida por el arnés** (las superficies se muestrean a puntos y cuentan como vistas cuando un rayo de sensor las toca), tiempo hasta el X %, bytes. Si el cerebro implementa `export_map()`: precisión y exhaustividad de los vóxeles ocupados frente a la verdad |
| `pursuit` | Uno o varios evasores, con su propio cerebro, huyen de los perseguidores. Captura: distancia ≤ r_c durante ≥ t_c | % de capturas, tiempo de captura, distancia mínima en el tiempo, bytes |

| # | Tarea | Nivel |
|---|---|---|
| 5.1 | Contrato `Mission` + `a_to_b` + camino más corto del oráculo (Theta* sobre la rejilla inflada; cota de error documentada) | S |
| 5.2 | `coverage` con muestreo de superficies y puntuación de `export_map` | S |
| 5.3 | `pursuit` con su regla de captura | H |
| 5.4 | Cerebros `flee` (evasor que maximiza la distancia y usa obstáculos para tapar la línea de vista) y `pursue` (persecución pura vía `detector`) | S |
| 5.5 | Estación de tierra con el cerebro `central_assign` (reparte sectores para la cobertura): permite comparar C2 centralizado frente a distribuido | S |

**Aceptación**
- `oracle` en `a_to_b`: regret ≤ 0,05 en 20 semillas, lo que valida la métrica.
- La cobertura es monótona. `central_assign` con radio ideal supera a `random_walk` en 20 semillas emparejadas (test de Wilcoxon, p < 0,05).
- Evasor quieto: `pursue` lo captura en el 100 % de 20 semillas. `flee` contra 1 perseguidor de igual velocidad consigue menos capturas que contra 4 (prueba de sanidad de la dinámica).
- Las métricas aparecen en `metrics.json` de cada ejecución y son idénticas al recalcularlas desde la grabación.

**Grafo** · sale: `Mission:*`, `Metric:*`, `GroundStation`, `Brain:flee`, `Brain:pursue`, `Brain:central_assign`.

---

## F6. Lotes, estadística y visor completo: v0.2.0

| # | Tarea | Nivel |
|---|---|---|
| 6.1 | CLI: `run`, `sweep` (semillas × cerebros × configuraciones de sensores desde un bloque `[sweep]`), `summarize`, `export-viewer` y `serve` | H |
| 6.2 | `sweep` con `ProcessPoolExecutor` (≤ 5 procesos), reanudable (salta las ejecuciones ya hechas) y con fallos registrados | S |
| 6.3 | `summarize`: media ± error estándar, diferencias emparejadas por semilla, Wilcoxon y corrección de Holm (scipy) → CSV + Markdown | H |
| 6.4 | Visor completo: línea de tiempo, velocidad, cámara que sigue a un dron, estelas, enlaces de radio, rayos del dron seleccionado, mapa de cobertura, eventos y panel de métricas | S |
| 6.5 | Opcional: modo en vivo por SSE con `http.server` de la biblioteca estándar, sin dependencias | S |

**Aceptación**
- Un barrido de 20 semillas × 3 cerebros da resultados idénticos con 1 y con 5 procesos (hash por ejecución).
- Al matar un barrido y relanzarlo, solo repite lo pendiente.
- CPU ≤ 85 % durante el barrido (medido).
- El visor va a ≥ 30 fps con 50 drones y estelas.
- `summarize` produce la tabla emparejada con Holm sobre un barrido real.

**Grafo** · sale: `Sweep`, `Summary`, `ViewerReplay`.

---

## F7. GRACE-3D 2.0.0 enchufado y escalera de realismo: v0.3.0

**Objetivo:** que el código real de GRACE-3D 2.0.0 (importado de swarm-dil, sin tocarlo) vuele en C2-swarm-lab alimentado solo por sensores, y medir cuánto le cuesta cada capa de realismo. Esto es lo que usará cada nueva versión de GRACE.

API de GRACE-3D 2.0.0 que hay que adaptar (`swarm-dil/src/sim3d/agent.py`): `Agent(ident, Spec)`, `observe(hits)` (vóxeles bloqueados en desplazamientos relativos), `receive(bytes)`, `transmit() -> bytes` (≤ 64 B), `decide() -> dirección 26-conexa` y `moved(dirección, ok)`. Funciona por ranuras discretas sobre una rejilla de 5 m.

| # | Tarea | Nivel |
|---|---|---|
| 7.1 | `brains/grace3d.py`, el adaptador. Hace de puente entre sensores y GRACE, igual que haría el firmware en un dron real. Rejilla virtual anclada en el despegue (tamaño de vóxel configurable). Pasa las zonas ToF y la pose estimada a desplazamientos de vóxeles bloqueados. Convierte `decide()` en un punto de paso, y este en consignas de velocidad. `moved(ok)` se decide por progreso o por tiempo agotado. Una ranura de GRACE equivale a un paso de cerebro. La radio pasa los bytes sin cambios | S |
| 7.2 | Bioma `swarm_dil_v2`: importa el generador de terreno de swarm-dil y pasa los vóxeles a cajas, fusionadas en bloques grandes para reducir primitivas | H |
| 7.3 | **Puente de equivalencia:** condiciones idealizadas (pose ideal, ToF como en swarm-dil, masa puntual, radio `dil_v02` con los mismos parámetros) sobre las mismas 20 semillas | S |
| 7.4 | **Escalera de realismo**, un factor cada vez sobre semillas emparejadas: (a) mundo continuo `mixto`/`urbano`, (b) ToF con ruido y subrayos, (c) deriva `vio`, (d) dinámica de cuadricóptero con viento, (e) línea de vista 3D en la radio. Se miden Δregret, Δllegadas y Δbytes con Holm | S + análisis de Opus |
| 7.5 | `docs/plug_grace.md`: cómo enchufar GRACE vN (adaptador + una línea de escenario) | H |

**Aceptación**
- El adaptador pasa el test de aislamiento. El código de GRACE se importa de swarm-dil en un commit fijado y no se modifica.
- Equivalencia: llegadas y regret dentro del intervalo de confianza del 95 % de los valores publicados de swarm-dil v2 (91,5 % y 0,99 ± 0,14) en las mismas semillas. Si no se cumple, la discrepancia queda explicada antes de seguir.
- La escalera produce una tabla de efectos con su significación. Esa tabla es la entrada de diseño de la siguiente versión de GRACE.

**Grafo** · sale: `Adapter:GRACE-3D 2.0.0`, `Biome:swarm_dil_v2`, `RealismLadder`, y la arista `GRACE-3D 2.0.0 → validado_en → C2-swarm-lab`.

**Riesgos:** las suposiciones discretas de GRACE (una ranura = un movimiento de vóxel exacto) chocan con la dinámica continua. Mitigación: el adaptador documenta cada desviación y la escalera las cuantifica una a una.

---

## 4. Definición de terminado (cada tarea y cada fase)

1. Los tests nuevos y todos los anteriores en verde, con ≤ 5 procesos.
2. Cada lógica no trivial deja un test que falla si se rompe, y nada más (PonyTail).
3. Sin código muerto, sin comentarios obsoletos y sin dependencias no declaradas.
4. El grafo actualizado con los nodos y aristas de la fase.
5. Limpieza: sin `.tmp`, scripts de verificación ni servidores vivos.
6. Commit con la identidad del proyecto. Opus valida los criterios de aceptación antes de cerrar cada fase.

## 5. Riesgos globales

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Rendimiento de Python con muchos rayos y física a 1200 Hz | Barridos lentos | Descarte por alcance, objetivos medidos en F1, F2 y F4, y numba (rueda cp314 verificada) como reserva |
| Crecimiento del alcance por el realismo | Retraso | Cada capa de realismo va detrás de una opción con perfil `ideal`. La cartera de §6 solo se abre a petición |
| Determinismo entre máquinas | Resultados distintos en otro PC | Garantía declarada solo para la misma plataforma y las mismas versiones (registradas en meta.json) |
| Desajuste del adaptador de GRACE | Conclusiones falsas | El puente de equivalencia es obligatorio antes de la escalera |
| Sintonía del autopiloto en perfiles nuevos | Vuelo inestable | Los criterios de escalón y de viento funcionan como tests de regresión |

## 6. Cartera (fuera de alcance hasta que se pida)

Render RGB y cerebros de visión por computador · puente ROS 2 o Crazyflie (el contrato ya es ENU/FLU en SI) · cerebros en otro proceso o lenguaje (firmware en el bucle) · obstáculos móviles · meteorología que afecte a ToF y LiDAR · estela descendente entre drones · EKF real · misiones de búsqueda y rescate, formación y perímetro · interfaz gymnasium para aprendizaje por refuerzo · nota técnica RL-TR-2026-05 del banco de pruebas, en el formato de RL-TR-2026-03.

## 7. Preguntas abiertas (no bloquean F0 a F2)

1. ¿Se publica en GitHub (RoperaLabs) desde F0 o al llegar a v0.1.0?
2. ¿Qué escala y perfil de dron usamos por defecto: exterior (`urbano` con `iris`) o interior (`interior` con `cf2`)?
3. ¿Quieres la nota técnica RL-TR-2026-05 al cerrar v0.3.0?

## 8. Estado

- **v0.1.0 "Primer vuelo"** (2026-11-19): F0 a F4 cerradas. Aceptada con dos criterios por debajo del objetivo de F4: los cerebros reactivos llegan al 31 % (campo de potencial) y al 21 % (boids) en mixto con densidad 0,2 (objetivo 70 %), y el escenario de referencia va a 4,3x el tiempo real (objetivo 5x). El oráculo valida la tubería y la métrica (98 % de llegadas, distancia volada / óptima = 1,00). Detalle en `.ia/grafo/cerebros.md`.
- **F5 misiones** (2026-11-19): A→B con regret, cobertura medida por el arnés, persecución y estación de tierra. Cumple el oráculo (regret +0,019), la cobertura central frente a aleatoria (p = 0,011) y la persecución 1 frente a 4 (80 % / 100 %). Por debajo: blanco quieto capturado al 95 % (objetivo 100 %). Detalle en `.ia/grafo/misiones.md`.
- **F6 lotes y visor** (2026-11-19): barridos emparejados reanudables, resumen con Wilcoxon y Holm, CLI y visor completo (seguimiento, radio, cobertura, eventos). Resultados independientes del número de procesos; CPU 67-73 %. Sin medir los fps con 50 drones; sin rayos por fotograma ni modo en vivo. Detalle en `.ia/grafo/lotes-visor.md`.
