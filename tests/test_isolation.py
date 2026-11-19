import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OK_TOP = set(sys.stdlib_module_names) | {"numpy", "scipy", "brains"}
GRACE = {"sim3d.agent", "sim3d.geom"}      # the GRACE-3D 2.0.0 on-board agent; sim3d.world is its simulator


def violations(source: str) -> list[str]:
    """One line per forbidden import in a brain's source."""
    out = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import):
            mods = [(a.name, ()) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                continue
            mods = [(n.module, tuple(a.name for a in n.names))]
        else:
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "__import__":
                out.append(f"line {n.lineno}: __import__ call")
            continue
        for mod, names in mods:
            top = mod.split(".")[0]
            if top == "importlib":
                bad = True
            elif top == "sim3d":
                bad = mod not in GRACE
            elif top == "c2lab":
                bad = mod != "c2lab.brain" and not (mod == "c2lab" and names and set(names) == {"brain"})
            else:
                bad = top not in OK_TOP
            if bad:
                out.append(f"line {n.lineno}: forbidden import {mod}" + (f" ({', '.join(names)})" if names else ""))
    return out


def test_brains_are_clean():
    files = [f for f in (ROOT / "brains").rglob("*.py") if "privileged" not in f.relative_to(ROOT / "brains").parts]
    for f in files:
        assert violations(f.read_text(encoding="utf-8")) == [], f


@pytest.mark.parametrize("src", ["import numpy as np", "from c2lab.brain import Action", "from c2lab import brain",
                                 "from .common import x", "import math",
                                 "from sim3d.agent import Agent"])
def test_allowed(src):
    assert violations(src) == []


@pytest.mark.parametrize("src", ["from c2lab.world import World", "import c2lab.sim", "from c2lab import world",
                                 "import c2lab", "import importlib", '__import__("c2lab.world")',
                                 "import matplotlib", "from sim3d.world import run", "import sim3d"])
def test_forbidden(src):
    assert violations(src)
