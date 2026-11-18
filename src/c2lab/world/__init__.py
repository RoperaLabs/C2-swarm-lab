"""Procedural 3-D worlds and their geometric queries (contract WorldQuery in .ia/grafo/_indice.md)."""
from .biomes import generate
from .world import World

__all__ = ["World", "generate"]
