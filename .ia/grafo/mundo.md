# Clúster mundo (F1)

Leer antes: `CLAUDE.md`, `.ia/grafo/_indice.md` (contrato `WorldQuery` y formato `c2lab-world/1`) y `src/c2lab/seeds.py`.
Dependencias de ejecución: numpy y scipy (solo `scipy.ndimage` en el oráculo). Nada más.
Cada tarea corre **solo sus tests** (`.venv/Scripts/python -m pytest -q tests/<suyo>.py`); otros agentes escriben otros tests a la vez.

## Primitivas (marco local: centro `c`, cuaternión `q` wxyz local→mundo)

| type | parámetros | convención | SDF exacta (Inigo Quilez) | rayo |
|---|---|---|---|---|
| `box` | `c`, `h` (3 semiejes), `q` | | sdBox en local | analítico (slabs en local) |
| `cylinder` | `c`, `r`, `hh` (semialtura), `q` | eje = z local | sdCappedCylinder | analítico (cuádrica + tapas) |
| `capsule` | `a`, `b`, `r` | | sdCapsule | analítico (cilindro + 2 esferas) |
| `sphere` | `c`, `r` | | | analítico |
| `cone` | `c` (centro de la base), `r` (radio de la base), `h`, `q` | vértice en `c + h·z_local` | sdCone / sdCappedCone con r2 = 0 | trazado de esferas + bisección |
| `torus` | `c`, `R`, `r`, `arc`, `q` | anillo en el plano xy local (eje = z local); se conserva el arco de semiángulo `arc` centrado en +y local (`arc = π` anillo completo); extremos redondeados | sdCappedTorus | trazado de esferas + bisección |
| terreno | `heights` (nx × ny) | la malla cubre exactamente `bounds` en xy; `heights[i][j]` en x = lo_x + i·(sx/(nx−1)), y = lo_y + j·(sy/(ny−1)); bilineal | no exacta: cota inferior de \|d\| (z − h)/√(1+L²), con L = pendiente máxima | plano analítico si es plano; si no, avance con la cota + bisección |

Ejemplo de convención (debe ser un test): un arco apoyado en el suelo con el eje del anillo = x del mundo y el arco centrado hacia +z del mundo usa la rotación con columnas x_local = (0,1,0), y_local = (0,0,1), z_local = (1,0,0). Con `c` en el suelo, `R` = 5 y `arc` = π/2, los puntos c ± (0, 5, 0) (los pies) y c + (0, 0, 5) (la clave) están dentro, y c + (0, 0, −5) está fuera.

## T1.1-T1.4 Núcleo geométrico (agente A1)

Archivos permitidos: `src/c2lab/world/__init__.py`, `src/c2lab/world/primitives.py`, `src/c2lab/world/world.py`, `tests/test_world.py`, `benchmarks/bench_world.py`, la sección Deltas.
- `World.from_dict(d)` y `to_dict()` (formato del índice; valores redondeados a 6 decimales al construir, de modo que `from_dict(to_dict(w))` es idéntico y el hash es estable), `hash`, `bounds`, `anchors`, `tags` y `TERRAIN = -2`.
- `sdf(p, terrain=True, max_dist=inf)`, `ground(xy)`, `raycast(o, d, max_range)` y `segment_crossings(a, b)` según el índice. Vectorizado por tipo de primitiva. Descarte por AABB y por alcance o `max_dist` antes de evaluar, en bloques de rayos o puntos para acotar la memoria. Sin bucles Python por rayo.
- `__init__.py` exporta `World` (y después `generate`, que añade A4).

Aceptación (`tests/test_world.py`):
1. Signo: 10⁴ puntos aleatorios por tipo cerca de la primitiva. El signo de la SDF coincide al 100 % con un predicado de pertenencia **independiente** (escrito aparte, sin la SDF).
2. Exactitud: 500 puntos exteriores por tipo (salvo el terreno). |sdf| coincide con la distancia mínima a un muestreo denso de la superficie (≥ 10⁵ puntos, incluidas tapas y extremos) con un error ≤ al espaciado del muestreo.
3. |∇sdf| ≤ 1 + 1e-3 por diferencias centrales (h = 1e-5) en todos los tipos. Para el terreno, |∇| ≤ 1 de la cota.
4. Raycast frente a una referencia de muestreo denso a lo largo del rayo (paso de 2 mm sobre `sdf`, con bisección final) en 2×10⁴ rayos de una escena mixta con los 7 tipos, cables incluidos. Toda discrepancia de acierto/fallo debe ser rasante: el mínimo de la SDF a lo largo del rayo de referencia es ≤ 2 mm. Cuando ambos aciertan, la diferencia de distancia es ≤ 1 mm y la normal coincide con ∇sdf normalizado (≤ 1e-3).
5. Casos fijos: rayo por el hueco del toro (fallo), rayo rasante a la cara de una caja, origen dentro de un obstáculo (dist = 0) y el arco del ejemplo de convención.
6. `segment_crossings` frente a fuerza bruta (muestreo denso de cada segmento con la SDF de cada primitiva por separado) en 2000 segmentos: 100 % de coincidencia.
7. `sdf(p, max_dist=m) == minimum(sdf(p), m)` en puntos aleatorios. `ground` es exacto en los nodos y vale 0 en terreno plano.
8. Ida y vuelta `to_dict`/`from_dict` y un **hash fijado** de un mundo pequeño escrito a mano en el test.

`benchmarks/bench_world.py` (imprime cifras y no es un test): escena sintética sembrada en 200×200×60 m con 400 cajas, 40 cilindros, 20 esferas, 20 cápsulas, 10 conos, 10 toros y terreno plano. Mide:
- 10 orígenes libres × 512 rayos uniformes con alcance 10 m, ≥ 20 repeticiones → **rayos/s, objetivo ≥ 300 000**;
- µs por llamada de `sdf` con 10 puntos y `max_dist = 2` (patrón de la física).
Si no llega al objetivo con numpy: **parar** y devolver `BLOQUEO rendimiento` con las cifras y dónde se va el tiempo. No añadir numba.

## T1.9 Visor estático (agente A3)

Archivos permitidos: `viewer/index.html`, `viewer/main.js`, `viewer/sample_world.json`, `.claude/launch.json`, la sección Deltas. `viewer/vendor/` (three.js r170 y OrbitControls) es de solo lectura.
- Un mapa de importaciones lleva `"three"` a `./vendor/three.module.js`. Sin build, sin CDN y sin npm.
- Se sirve con `.venv/Scripts/python.exe -m http.server 8770 --bind 127.0.0.1` desde la raíz (`.claude/launch.json`, configuración `viewer`). La URL es `/viewer/?world=<ruta relativa a viewer/>`; por defecto, `sample_world.json`. También se puede cargar un archivo con un selector o arrastrándolo.
- Mundo en z arriba (`camera.up = (0,0,1)`). La cámara se encuadra a `bounds`. OrbitControls.
- Un `InstancedMesh` por tipo y color (caja, cilindro, esfera, cápsula y cono); cada toro es una malla propia, porque cada `arc` da una geometría distinta. Hay que mapear las convenciones de la tabla: el eje y de las geometrías de three pasa a z local, la base del cono a `c`, y el arco de three empieza en +x mientras aquí está centrado en +y, con longitud 2·arc. El terreno es una malla con las alturas, la geocerca un cubo translúcido con aristas (como la imagen de referencia) y A y B marcadores con etiqueta.
- Color por `tag`, con una paleta por defecto por tipo. Panel con bioma, semilla, número de primitivas por tipo, casillas para ocultar cada tag y contador de fps.
- `sample_world.json`: bounds 40×30×15. Un ejemplar de cada tipo en posiciones separadas: caja girada 30° en guiñada, cilindro inclinado, cápsula oblicua, cono apoyado (vértice arriba), anillo completo vertical y el arco del ejemplo de convención. Terreno con algo de relieve (6×5 nodos) y A/B.
- Rendimiento: ≥ 30 fps con 1000 primitivas en este portátil (GTX 1050). El agente puede generar con un script temporal un JSON de 1000 primitivas para medirlo, y debe borrarlo después.
- El check del agente es que la página cargue sin errores de consola. La comprobación visual la hace Opus en el navegador.

## T1.6-T1.8 Biomas, oráculo y CLI (agente A4, después de A1)

Archivos permitidos: `src/c2lab/world/biomes.py`, `src/c2lab/world/oracle.py`, `src/c2lab/world/__main__.py`, la línea `generate` de `src/c2lab/world/__init__.py`, `tests/test_biomes.py`, `benchmarks/validate_biomes.py` y la sección Deltas.
- `generate(biome, size_m, density, seed) -> World`. El intento k usa `rng(seed, "world", k)` hasta que el oráculo lo da por viable (máximo 20 intentos; si se agotan, error). Rellena `anchors` A y B: puntos en el suelo, A cerca de x mínimo y B cerca de x máximo, separados ≥ 0,7·sx. Alrededor de cada uno queda una **zona libre** (cilindro vertical de radio `clear`, desde el suelo hasta el techo) sin primitivas.
- `density` = **cobertura en planta**: fracción de celdas xy cuyo segmento vertical (de suelo + 5 cm a `z_top`) corta alguna primitiva (`World.segment_crossings`). En el bioma `interior`, `z_top` queda por debajo del techo. El generador coloca objetos hasta alcanzarla, o hasta que no quepan más.
- `oracle.py`: `voxelize(world, res, inflate)`, conservador (ocupado si sdf(centro) < inflate + res·√3/2, con `max_dist`); `feasible(world, res, inflate, alt)`, que comprueba que A + alt·ẑ y B + alt·ẑ están libres y conectados con 26-conectividad (`scipy.ndimage.label`) dentro de la geocerca; y `coverage(world, res, z_top=None)`. Por defecto: `inflate = 0.6` m (radio del iris 0,38 + 0,2 de margen), `alt = 2` m y `res` según el bioma (mixto 0,5; urbano 1,0; bosque 1,0; interior 0,25).
- `python -m c2lab.world <bioma> --seed N [--density D] [--size X Y Z] --out runs/worlds/<bioma>_<N>.json`. Imprime hash, primitivas por tipo, cobertura, intentos y tiempo. Los tamaños por defecto se importan de `c2lab.scenario.BIOMES`, sin duplicarlos.

Biomas (las dimensiones son realistas; la cantidad la fija `density`):
- **mixto** (tipo imagen de referencia; zona libre de 4 m): pilares (cajas de 1 a 2,5 m de lado y 8 a 18 m de alto, más cilindros de r 0,5 a 1,2), anillos (toros de R 1,5 a 4 y r 0,2 a 0,5, centro a 3-15 m, la mayoría verticales), arcos (medio toro apoyado en el suelo, R 5 a 15, r 0,3 a 0,8), losas flotantes (4 a 10 × 4 a 10 × 0,3 a 0,6 m a 4-15 m de altura), muros con una ventana (8 a 20 m de largo, 0,3 a 0,5 de grueso y 6 a 15 de alto; hueco de 1,5 a 4 m formado por 4 cajas) y esferas (r 0,8 a 3). Tags: `pillar`, `ring`, `arch`, `slab`, `wall`, `sphere`.
- **urbano** (zona libre de 10 m): manzanas de 30 a 45 m, calles de 10 a 16 m y un 10-20 % de plazas o parques. Edificios de 1 a 6 por manzana, con planta de 10 a 35 m y altura log-normal (mediana 15 m, σ_log 0,5, recortada a [4, 0,9·sz]); algunos con un retranqueo (otra caja encima). Farolas (cilindros de r 0,12 a 0,2 y 7 a 10 m cada 20-30 m) unidas por **cables** (cápsulas de r 0,01 a 0,03 a 6-8 m). Árboles (tronco cilíndrico + copa esférica). Pasarelas (cajas de 2 a 4 m de ancho entre dos edificios a 6-20 m). Tags: `building`, `pole`, `cable`, `trunk`, `crown`, `bridge`.
- **bosque** (zona libre de 6 m): terreno suave (celda de 3 a 5 m, amplitud de 2 a 8 m, alturas ≥ 0). Frondosos (tronco de r 0,15 a 0,4 y 4 a 10 m + copa esférica de r 2 a 5) y coníferas (tronco + cono de r 1,5 a 3,5 y 6 a 15 m), separados ≥ 2 m, con claros. Arbustos (esferas o conos pequeños) y troncos caídos (cápsulas tumbadas sobre el terreno). Todo se apoya en `ground`. Tags: `trunk`, `crown`, `bush`, `log`.
- **interior** (zona libre de 3 m): techo (losa en la parte alta), **paredes perimetrales** visibles para los sensores, columnas en rejilla de 8 a 10 m, estanterías (montantes + baldas cada 1,5 a 2 m hasta 4-6 m; filas de 10 a 20 m; pasillos de 2,5 a 4 m), tabiques con puertas de 1,5 a 2,5 m de ancho y 2,1 a 2,5 m de alto, vigas bajo el techo y lámparas colgadas (esfera + cable). Tags: `ceiling`, `wall`, `column`, `rack`, `beam`, `lamp`, `cable`.

Aceptación:
- `tests/test_biomes.py` (rápido, 3 semillas por bioma): repetibilidad (misma semilla, mismo hash), semillas distintas dan hashes distintos, viabilidad, zonas libres respetadas (sdf ≥ `clear` − r en la columna de A y B) y A y B en el suelo.
- `benchmarks/validate_biomes.py` (≤ 5 procesos), con informe por bioma:
  - 200 semillas → 100 % viables, 200 hashes distintos, intentos medios y máximos, tiempo medio (objetivo ≤ 2 s; urbano ≤ 4 s);
  - densidades {0,1; 0,2; 0,35} × 20 semillas → cobertura media dentro de ± 15 % relativo de la objetivo.

## Deltas

N Viewer: `viewer/` (index.html, main.js, sample_world.json), three.js r170 vendorizado
R Viewer -> WorldFile(c2lab-world/1): lee el JSON; torus con extremos planos (sin redondear), capsula = cilindro + 2 esferas instanciadas
N World: src/c2lab/world/world.py (Opus, tras cortarse A1 por límite de gasto): from_dict/to_dict/hash, sdf con rejilla de 4 m para max_dist, ground, raycast, segment_crossings
N Primitive:box/cylinder/capsule/sphere/cone/torus + terreno: src/c2lab/world/primitives.py, kernels numba (prange); rayos y segmentos por trazado de esferas sobre SDF exactas
R World -> numba (dependencia aprobada 2026-11-18)
A Aceptación T1.1-T1.4: 25 tests; benchmark 2 015 030 rayos/s (objetivo 300 000), sdf(10 pts, max_dist 2) 24 µs/llamada; usa todos los hilos: en barridos fijar NUMBA_NUM_THREADS por proceso
N Biome:mixto/urbano/bosque/interior: src/c2lab/world/biomes.py (Opus); density = cobertura en planta; urbano árboles antes que edificios; interior con estanterías dobles entre líneas de columnas y tabiques sobre ellas (2 puertas)
N Oracle: src/c2lab/world/oracle.py voxelize/feasible (scipy.ndimage.label 26-conexo)/coverage
N WorldCLI: python -m c2lab.world <bioma> --seed N --density D --out runs/worlds/...
R generate -> SeedTree (rng(seed, "world", intento)), R generate -> Oracle
A Validación (benchmarks/validate_biomes.py, 5 procesos, 1 m 20 s): 200/200 viables y 200 hashes distintos en los 4 biomas; intentos medios 1,0 (interior 1,9, máx 7); tiempo medio 0,10-0,39 s; cobertura ±15 % en 0,1/0,2/0,35 salvo interior 0,35 → 0,255 (capacidad física con pasillos de 2,5-3,5 m)
A Visor comprobado con los 4 biomas (Opus, navegador)
