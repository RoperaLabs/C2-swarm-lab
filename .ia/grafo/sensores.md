# Clúster sensores (F3)

Contratos `Sensors` y `Radio` en `.ia/grafo/_indice.md`. Código: `src/c2lab/sensors.py` y `src/c2lab/radio.py`. Tests: `tests/test_sensors.py`.
Valores por defecto (`sensors.DEFAULTS`): VL53L5CX (tof), Livox Mid-360 (lidar), RealSense D435 (depth), BMI088 (imu), u-blox M10 (gnss) y BMP388 (baro). Todos se pueden sobrescribir en el escenario.

## Deltas
N Sensors (Opus; Sonnet bloqueado por límite de gasto): tof (min por zona sobre 2×2 subrayos), lidar, depth (z), detector (FoV, alcance, línea de vista, clase mismo/otro equipo, falsos positivos, confusión), pose (ideal, ekf, vio con deriva ∝ distancia, flow con validez por altura), imu (densidad + paseo de sesgo), gnss (Gauss-Markov, zonas denegadas, cielo obstruido), baro, mag, prior_map (2,5D sin objetos finos, desde un mundo "topográfico"), battery
N Radio: ideal, disco, dil_v02 (port de swarm-dil v0.2 con segment_crossings 3D), latencia de 1 ranura, límites de tamaño y de paquetes por ranura, ranging
R Sensors -> WorldQuery (raycast, segment_crossings, ground), R Sensors -> Drones (estado verdadero), R Drones -> Sensors.err (el autopiloto vuela sobre la estimación)
A Physics: step(tick, err) con errores del estimador (dp, dv, dyaw); el kernel aplica el error de guiñada; ya no se reconstruye R por tick
A Scenario: radio admite max_packets, range_m y loss
A Aceptación: 15 tests (conmutación, frecuencias, independencia de flujos, solo lectura, σ de imu/baro/gnss ±5 %, autocorrelación Gauss-Markov ±10 %, ToF contra muro, depth contra plano, línea de vista y FoV del detector, perfiles de pose, mapa sin cables, radio ideal/disco/latencia/límites/ranging, dil_v02 = presupuesto de enlace y −20 dB por obstáculo)
A Rendimiento: 10 iris + tof×4 + pose vio + imu + detector en urbano = 5 774 ticks/s (4,8x tiempo real); antes de optimizar 0,9x
