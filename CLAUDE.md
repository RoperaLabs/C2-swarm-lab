# C2-swarm-lab

Banco de pruebas 3D para algoritmos C2 de enjambres de drones; base de pruebas de las próximas versiones de GRACE.
Plan y fases: `PLAN.md`. Contratos entre módulos y clústeres de trabajo: `.ia/grafo/_indice.md` (fuente de verdad).

## Regla de oro

Un cerebro solo recibe lo que un dron real mediría: lecturas de sensores con ruido, en su marco y a su frecuencia, y bytes de radio. Nunca geometría del mundo, rejillas ni coordenadas globales, salvo que un sensor encendido las produzca (`gnss`, `prior_map`). `brains/` importa solo `c2lab.brain`, la biblioteca estándar, numpy, scipy y el agente de GRACE-3D (`sim3d.agent`, `sim3d.geom`, de swarm-dil; nunca su `sim3d.world`); `tests/test_isolation.py` lo hace cumplir. Los cerebros de `brains/privileged/` (solo para validar métricas) son la única excepción y se marcan en los resultados.

## Convenciones

- Unidades SI. Mundo ENU, cuerpo FLU (REP-103), marco local = ENU anclado al despegue. Cuaterniones (w, x, y, z).
- Tiempo en ticks enteros de física (1200 Hz por defecto). Cada frecuencia da un periodo entero de ticks o el escenario no carga.
- Toda aleatoriedad sale de `c2lab.seeds.rng(seed, *ruta)`. Nunca `np.random` global, `random` global ni el reloj en la lógica de simulación.
- Código, docstrings y comentarios en inglés; planificación y grafo en español.
- Sin rutas locales absolutas, nombres reales ni correos en el contenido de los archivos.

## Comandos

```bash
.venv/Scripts/python -m pip install -e ".[test]"
.venv/Scripts/python -m pytest -q
```

Barridos y tests en paralelo: como máximo 5 procesos (con 6 la CPU llega al 87 %).

## Terminado = 

Tests en verde · cada lógica no trivial deja un test que falla si se rompe · sin código muerto ni dependencias no declaradas · deltas añadidos al clúster en `.ia/grafo/` · sin `.tmp/`, scripts sueltos ni servidores vivos · commit como `ep2si10n <50156072+ep2si10n@users.noreply.github.com>`, fechas UTC, sin coautor.
