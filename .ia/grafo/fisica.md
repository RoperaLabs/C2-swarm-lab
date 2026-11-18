# Clúster fisica (F2)

Leer antes: `CLAUDE.md`, `.ia/grafo/_indice.md` (contratos `Physics`, `TrueState` y `WorldQuery`) y `src/c2lab/brain.py` (consignas `Velocity` y `Position`).
Agente A2. Archivos permitidos: `src/c2lab/physics/__init__.py`, `profiles.py`, `quad.py`, `pointmass.py`, `autopilot.py`, `drones.py`, `tests/test_physics.py`, `benchmarks/bench_physics.py` y la sección Deltas.
Solo numpy. Corre **solo** `.venv/Scripts/python -m pytest -q tests/test_physics.py`. `World` aún no existe: los tests usan un `FakeWorld` propio, definido en el test, que cumple el contrato `WorldQuery` (`sdf(p, terrain, max_dist)`, `ground(xy)`, `bounds`) con suelo plano y una caja analítica.

## Perfiles (`profiles.py`, datos con su fuente en un comentario)

| campo | `cf2` | `iris` |
|---|---|---|
| fuente | gym-pybullet-drones `assets/cf2x.urdf` | PX4 SITL gazebo-classic `models/iris/iris.sdf.jinja` |
| masa kg | 0.027 | 1.535 (base 1.5 + IMU 0.015 + 4 rotores × 0.005) |
| J diag kg·m² | 1.4e-5, 1.4e-5, 2.17e-5 | 0.029125, 0.029125, 0.055225 |
| rotores (x, y) m, FLU | (0.028, −0.028), (−0.028, −0.028), (−0.028, 0.028), (0.028, 0.028) | (0.13, −0.22), (−0.13, 0.20), (0.13, 0.22), (−0.13, −0.20) |
| signo del par de guiñada s_i | −1, +1, −1, +1 | −1, −1, +1, +1 (ccw, ccw, cw, cw) |
| kf N/(rad/s)² | 2.8816e-8 (= 3.16e-10 N/rpm²) | 5.84e-6 |
| km N·m/(rad/s)² | 7.2404e-10 (= 7.94e-12 N·m/rpm²) | 3.504e-7 (momentConstant 0.06 × kf) |
| ω_max rad/s | 2273.8 (empuje/peso 2.25) | 1100 |
| τ motor subida / bajada s | 0.02 / 0.02 (estimado, sin fuente) | 0.0125 / 0.025 |
| arrastre de rotor c_xy, c_z (× Σω) | 9.1785e-7, 1.0311e-6 | 1.75e-4, 0 |
| radio de hélice m | 0.0231348 | 0.128 |
| radio de colisión m | 0.06 | 0.384 (envolvente de las puntas de las hélices) |
| FM, η (energía; estimados) | 0.3, 0.5 → ~7 W en estacionario | 0.6, 0.8 → ~170 W en estacionario |
| vmax_xy, vmax_up, vmax_down m/s | 2.0, 1.0, 0.7 | 12, 3, 1.5 (valores por defecto de PX4) |
| inclinación máx., guiñada máx. | 25°, 180°/s | 45°, 90°/s |

## Modelo (`quad.py`, vectorizado sobre n drones)

- Empuje T_i = kf·ω_i² sobre z del cuerpo. Par = Σ r_i × (0, 0, T_i) + Σ s_i·km·ω_i²·ẑ.
- Arrastre en cuerpo: F_d = −(Σω_i)·diag(c_xy, c_xy, c_z)·v_rel,cuerpo, con v_rel = v − viento.
- Viento: vector medio + ráfaga Ornstein-Uhlenbeck por dron (σ, τ), sembrada con el `rng` recibido.
- Motores de primer orden con τ de subida o de bajada según el signo, saturados en [0, ω_max].
- Integración semi-implícita: primero v y ω, después p y q con los valores nuevos; q se renormaliza. dt = 1/1200 por defecto.
- `f` (fuerza específica, lo que mide un acelerómetro) = fuerzas no gravitatorias en cuerpo / m.
- Potencia eléctrica P = Σ T_i^1.5 / (FM·η·√(2ρA)), con ρ = 1.225 y A = π r_hélice². `energy` es su integral.
- Sin efecto suelo, sin arrastre parásito cuadrático y sin estela entre drones: anotarlo con un comentario `ponytail:` y la forma de ampliarlo.

## Masa puntual (`pointmass.py`)

Misma interfaz y mismo control externo que el cuadricóptero. Aceleración limitada (horizontal ≤ g·tan(inclinación máx.); vertical acotada) y velocidad limitada a vmax. q solo lleva guiñada, w = (0, 0, ψ̇) y f = Rᵀ(a + g ẑ). La energía sale del empuje equivalente m·|a + g ẑ| repartido en 4 rotores.

## Autopiloto (`autopilot.py`): cascada estilo PX4, sobre el estado **estimado**

1. Posición P y velocidad PID a 50 Hz (cada 24 ticks). `Position(dp, yaw)` se convierte al recibirla en un objetivo absoluto (estimación + dp). `Velocity` en marco `local` o `body` (body = girado con la guiñada estimada). `None` mantiene la posición donde se recibió. Salida: aceleración deseada, recortada por inclinación y vmax.
2. Vector de empuje m(a + g ẑ) → empuje colectivo + actitud deseada (con la guiñada de la consigna) → actitud P a 600 Hz → velocidades angulares deseadas.
3. Velocidad angular PID a 1200 Hz → pares → mezclador (inversa de la matriz de asignación con kf, km y la geometría) → ω² recortado a [0, ω_max²] → ω_cmd.
4. Las ganancias son por perfil y las sintoniza el agente hasta cumplir los criterios.

## API (`drones.py`, contrato `Physics` del índice)

`Drones(profile, n, rng, world, mode="quad"|"pointmass", wind=(0,0,0), gust_sigma=0.0, gust_tau=2.0, dt=1/1200)`
- Atributos (el `TrueState` del índice): `p`, `v`, `q`, `w`, `f`, `alive`, más `landed`, `omega` y `energy`.
- `reset(p0 (n,3), yaw0 (n,), landed=True)`. Si `landed`, z = ground(xy) + radio de colisión.
- `set_setpoint(i, sp)` con `Velocity | Position | None`.
- `step(tick, est=None) -> list[event]`: un tick. `est` es un objeto con `p`, `v`, `q` y `w` estimados; `None` usa el estado verdadero. Devuelve eventos `(tick, i, kind)` con kind en `takeoff`, `landed`, `crash_obstacle`, `crash_terrain`, `crash_drone` y `geofence`. Un dron caído queda con `alive = False` y deja de integrarse.
- Despegue: un dron posado despega cuando la consigna pide subir (vz > 0,2 m/s o dp_z > 0,2 m). Mientras no, sigue posado y con los motores parados.
- Contacto con el suelo: z − ground ≤ radio. Si |v| ≤ 1 m/s y la inclinación ≤ 20°, se posa (`landed`): queda fijado al suelo con v = w = 0. Si no, `crash_terrain`.
- Obstáculos: `world.sdf(p, terrain=False, max_dist=...)` < radio → `crash_obstacle`. Para no consultar el SDF en cada tick, cada dron guarda un **presupuesto de distancia segura** b = sdf − radio, al que se resta |v|·dt cada tick, y solo se recalcula cuando b ≤ 0. Así no se pierde ninguna colisión.
- Entre drones: distancia < r_i + r_j → `crash_drone` para ambos. Fuera de `bounds` → `geofence` (el dron se pierde).

## Aceptación (`tests/test_physics.py`)

1. Caída libre (motores parados, arrastre a 0, v₀ = 0): |a_z + g| < 1e-9 y |f| < 1e-9.
2. Vuelo estacionario a 10 m, `iris` y `cf2`, estimador ideal y sin viento: deriva < 1 cm en 30 s y ω dentro de [0, ω_max].
3. Escalón de velocidad local x: `iris` de 0 a 5 m/s y `cf2` de 0 a 1 m/s. Sobreoscilación ≤ 10 %, 90 % en ≤ 1,5 s, error ≤ 0,05 m/s a los 5 s y la inclinación nunca supera el máximo.
4. `iris` manteniendo posición con viento de 5 m/s y ráfagas de σ = 1 m/s y τ = 2 s: error RMS ≤ 0,5 m en 60 s.
5. Convergencia: con dt = 1/2400 en lugar de 1/1200, la posición final tras 10 s de escalones de velocidad difiere < 1 cm.
6. Colisión: un `iris` a 2 m/s contra la caja del `FakeWorld` da el evento en **exactamente** el primer tick en que la SDF verdadera < radio (fuerza bruta tick a tick). También: choque frontal de dos drones y salida de la geocerca.
7. Aterrizaje: bajando a 0,5 m/s → `landed`; a 1,5 m/s → `crash_terrain`. Despegue desde posado → `takeoff`.
8. Energía: la integral coincide con la potencia analítica en estacionario × tiempo (≤ 1e-6 relativo). Potencia en estacionario: `cf2` 7 W ± 15 % e `iris` 170 W ± 15 %.
9. Determinismo: misma semilla con ráfagas → estado idéntico bit a bit tras 10 s.
10. Masa puntual: |v| ≤ vmax, |a_xy| ≤ g·tan(inclinación máx.) en todo momento, y alcanza la consigna.

`benchmarks/bench_physics.py`: ticks/s con 10 `iris` (modelo + autopiloto, sin mundo). **Objetivo ≥ 24 000 ticks/s** (20 veces el tiempo real). Si no llega con numpy: devolver `BLOQUEO rendimiento` con la cifra y un perfil por función. No añadir numba.

## Deltas
N Drones: src/c2lab/physics/drones.py (Opus, tras cortarse A2 por límite de gasto): reset/set_setpoint/step, despegue con motores en vuelo estacionario, contacto con el suelo solo bajando, presupuesto de distancia segura con sdf(max_dist=5)
N Autopilot: anti-windup condicional en el lazo de velocidad; ganancias iris kp_vel 2.0 ki_vel 1.0, cf2 2.0/0.5
N Profile:cf2, N Profile:iris (fuentes verificadas); potencia estacionaria 7.1 W / 166 W
R Drones -> WorldQuery (sdf terrain=False/True con max_dist, ground, bounds)
A Aceptación: 15/15 tests (criterios 1-10)
BLOQUEO rendimiento: 3 907 ticks/s quad, 3 600 masa puntual (10 iris) = 3,3x tiempo real; objetivo 24 000. Coste dominado por overhead por llamada numpy (~100 µs/tick). Decisión pendiente: numba / bajar tasa / aceptar
