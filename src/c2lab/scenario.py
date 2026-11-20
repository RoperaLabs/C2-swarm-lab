"""Scenario loading and validation."""
import math
import re
import tomllib

PHYSICS_HZ = 1200
BIOMES = {"mixto": [60, 40, 20], "urbano": [200, 200, 60], "bosque": [150, 150, 30], "interior": [40, 30, 8],
          "swarm_dil_v2": [200, 200, 50]}
SENSORS = {"tof", "lidar", "depth", "detector", "pose", "imu", "gnss", "baro", "mag", "prior_map", "battery"}
_BRAIN_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


DIL = {"tx_dbm", "path_loss_exp", "pl_1m_db", "noise_dbm", "shadow_sigma_db", "shadow_corr", "sinr50_db",
       "sinr_width_db", "obstacle_db", "rate_bps"}   # radio.dil overrides of the dil_v02 link model


class ScenarioError(ValueError):
    """Invalid scenario; the message starts with the full key path."""


def _err(path, value, why):
    raise ScenarioError(f"{path} = {value!r}: {why}")


def _table(d, path, allowed):
    if not isinstance(d, dict):
        _err(path, d, "debe ser una tabla")
    for k in d:
        if k not in allowed:
            _err(f"{path}.{k}" if path else k, d[k], "clave desconocida")
    return d


def _get(d, path, key, kind, default=..., check=None, why=""):
    """Read d[key] as kind ('int', 'num', 'str', 'bool'), applying default/check."""
    p = f"{path}.{key}"
    if key not in d:
        if default is ...:
            raise ScenarioError(f"{p}: clave obligatoria ausente")
        return default
    v = d[key]
    ok = {"bool": isinstance(v, bool),
          "int": isinstance(v, int) and not isinstance(v, bool),
          "num": isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
          "str": isinstance(v, str)}[kind]
    if not ok:
        _err(p, v, f"debe ser de tipo {kind}" + (" finito" if kind == "num" else ""))
    if check and not check(v):
        _err(p, v, why)
    return v


def _pos(v):
    return v > 0


def _choice(d, path, key, options, default=...):
    return _get(d, path, key, "str", default, lambda v: v in options, f"debe ser uno de {sorted(options)}")


def _ticks(path, hz, phys):
    period = phys / hz
    if period < 1 or abs(period - round(period)) > 1e-9:
        _err(path, hz, f"{phys} Hz de física no da un periodo entero de ticks (>= 1)")


def _brain(d, path):
    out = {"brain": _get(d, path, "brain", "str", check=_BRAIN_RE.match, why="debe ser 'modulo.sub:Clase'")}
    out["brain_params"] = d.get("brain_params", {})
    if not isinstance(out["brain_params"], dict):
        _err(f"{path}.brain_params", out["brain_params"], "debe ser una tabla")
    return out


def _sensors(d, path, phys):
    out = {}
    for name, v in _table(d, path, SENSORS).items():
        p = f"{path}.{name}"
        if v is False:
            continue
        if v is True:
            v = {}
        if not isinstance(v, dict):
            _err(p, v, "debe ser una tabla o bool")
        if "hz" in v:
            hz = _get(v, p, "hz", "num", check=lambda x: x > 0, why="debe ser > 0")
            _ticks(f"{p}.hz", hz, phys)
        out[name] = dict(v)
    return out


def parse(text: str) -> dict:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ScenarioError(f"scenario: TOML inválido: {e}") from e
    _table(raw, "", {"scenario", "world", "radio", "team", "ground_station", "sweep"})
    for k in ("scenario", "world", "team"):
        if k not in raw:
            raise ScenarioError(f"{k}: sección obligatoria ausente")

    s = _table(raw["scenario"], "scenario", {"name", "seed", "duration_s", "mission", "rates_hz"})
    r = _table(s.get("rates_hz", {}), "scenario.rates_hz", {"physics", "brain", "radio", "record"})
    phys = _get(r, "scenario.rates_hz", "physics", "int", PHYSICS_HZ, _pos, "debe ser > 0")
    rates = {"physics": phys}
    for k, dflt in (("brain", 10), ("radio", 10), ("record", 30)):
        rates[k] = _get(r, "scenario.rates_hz", k, "num", dflt, _pos, "debe ser > 0")
        _ticks(f"scenario.rates_hz.{k}", rates[k], phys)
    scen = {
        "name": _get(s, "scenario", "name", "str", check=bool, why="no puede estar vacío"),
        "seed": _get(s, "scenario", "seed", "int", check=lambda v: v >= 0, why="debe ser >= 0"),
        "duration_s": _get(s, "scenario", "duration_s", "num", check=_pos, why="debe ser > 0"),
        "mission": _choice(s, "scenario", "mission", {"a_to_b", "coverage", "pursuit"}),
        "rates_hz": rates,
    }

    w = _table(raw["world"], "world", {"biome", "size_m", "density", "wind", "gust_sigma", "drone_collisions"})
    biome = _choice(w, "world", "biome", BIOMES)
    size = w.get("size_m", BIOMES[biome])
    if not (isinstance(size, list) and len(size) == 3 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in size)):
        _err("world.size_m", size, "debe ser una lista de 3 números > 0")
    world = {"biome": biome, "size_m": list(size),
             "density": _get(w, "world", "density", "num", 0.2, lambda v: 0 < v < 1, "debe cumplir 0 < x < 1"),
             "gust_sigma": _get(w, "world", "gust_sigma", "num", 0.0, lambda v: v >= 0, "debe ser >= 0"),
             "drone_collisions": _get(w, "world", "drone_collisions", "bool", True)}
    wind = w.get("wind", [0.0, 0.0, 0.0])
    if not (isinstance(wind, list) and len(wind) == 3 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in wind)):
        _err("world.wind", wind, "debe ser una lista de 3 números (m/s, ENU)")
    world["wind"] = [float(x) for x in wind]

    ra = _table(raw.get("radio", {}), "radio", {"profile", "packet_bytes", "ranging", "max_packets", "range_m", "loss", "dil"})
    radio = {"profile": _choice(ra, "radio", "profile", {"ideal", "disco", "dil_v02"}, "dil_v02"),
             "packet_bytes": _get(ra, "radio", "packet_bytes", "int", 64, lambda v: v >= 1, "debe ser >= 1"),
             "ranging": _get(ra, "radio", "ranging", "bool", False),
             "max_packets": _get(ra, "radio", "max_packets", "int", 4, lambda v: v >= 1, "debe ser >= 1"),
             "range_m": _get(ra, "radio", "range_m", "num", 100.0, _pos, "debe ser > 0"),
             "loss": _get(ra, "radio", "loss", "num", 0.0, lambda v: 0 <= v < 1, "debe cumplir 0 <= x < 1"),
             "dil": {k: _get(ra["dil"], "radio.dil", k, "num") for k in _table(ra.get("dil", {}), "radio.dil", DIL)}}

    teams, names = raw["team"], set()
    if not isinstance(teams, list) or not teams:
        _err("team", teams, "debe ser una lista no vacía de tablas [[team]]")
    team = []
    for i, t in enumerate(teams):
        p = f"team[{i}]"
        _table(t, p, {"name", "count", "brain", "brain_params", "role", "drone", "memory_bytes", "sensors", "launch_s",
                    "dynamics"})
        name = _get(t, p, "name", "str", check=bool, why="no puede estar vacío")
        if name in names:
            _err(f"{p}.name", name, "nombre de equipo duplicado")
        names.add(name)
        team.append({
            "name": name,
            "count": _get(t, p, "count", "int", check=lambda v: v >= 1, why="debe ser >= 1"),
            **_brain(t, p),
            "role": _get(t, p, "role", "str", "nav", bool, "no puede estar vacío"),
            "drone": _choice(t, p, "drone", {"cf2", "iris"}, "iris"),
            "dynamics": _choice(t, p, "dynamics", {"quad", "pointmass"}, "quad"),
            "memory_bytes": _get(t, p, "memory_bytes", "int", None, lambda v: v >= 0, "debe ser >= 0"),
            "launch_s": _get(t, p, "launch_s", "num", 0.0, lambda v: v >= 0, "debe ser >= 0"),
            "sensors": _sensors(t.get("sensors", {}), f"{p}.sensors", phys),
        })

    out = {"scenario": scen, "world": world, "radio": radio, "team": team}
    if "ground_station" in raw:
        g = _table(raw["ground_station"], "ground_station", {"brain", "brain_params"})
        out["ground_station"] = _brain(g, "ground_station")
    if "sweep" in raw:
        out["sweep"] = _sweep(raw["sweep"])
    return out


def _sweep(sw):
    """[sweep]: seeds = [from, to) and optional [[sweep.variant]] tables with a unique name and a `set` table."""
    _table(sw, "sweep", {"seeds", "variant"})
    seeds = sw.get("seeds")
    if not (isinstance(seeds, list) and len(seeds) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                 for x in seeds) and 0 <= seeds[0] < seeds[1]):
        _err("sweep.seeds", seeds, "debe ser [desde, hasta) con 0 <= desde < hasta")
    variants, names = sw.get("variant", []), set()
    for i, v in enumerate(variants):
        p = f"sweep.variant[{i}]"
        _table(v, p, {"name", "set"})
        name = _get(v, p, "name", "str", check=bool, why="no puede estar vacío")
        if name in names:
            _err(f"{p}.name", name, "nombre de variante duplicado")
        names.add(name)
        if not isinstance(v.get("set", {}), dict):
            _err(f"{p}.set", v["set"], "debe ser una tabla")
    return {"seeds": seeds, "variant": [{"name": v["name"], "set": dict(v.get("set", {}))} for v in variants]}


def load(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return parse(f.read())
