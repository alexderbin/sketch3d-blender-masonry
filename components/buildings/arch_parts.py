"""Parts of the "buildings" component (checked by bench/deps.py): the core CORE, the types TYPES, the ensembles
ENSEMBLES — packages components.buildings.<part>. A module calls another part by a package import
(from components.buildings.<part> import <module>) — these are the imports deps.py reads; module(part, name)
is the dotted module name for importlib (the entry of buildings.py: the type registry). Project-wide code
comes from the project core (core.surface.ground, core.layers, …)."""
import os
import sys

COMP = os.path.dirname(os.path.abspath(__file__))
ROOT = COMP
while not os.path.isdir(os.path.join(ROOT, "core")) and ROOT != os.path.dirname(ROOT): ROOT = os.path.dirname(ROOT)   # project root — where core/ is

CORE = ("base", "siting", "style", "space", "influences")   # core: shared by all buildings; does not call the types or the ensembles
TYPES = ("castles", "temples", "edifices", "houses", "classics")          # building types: they call the core, not each other (shared code goes to the core)
ENSEMBLES = ("ensembles",)                                  # streets, cities, hives: the types only through the buildings.py entry
PARTS = CORE + TYPES + ENSEMBLES


PKG = "components.buildings"


def need(file, *parts):
    """Intentionally empty: links between the parts are package imports. No calls are left in the project; deps.py
    would check a declaration only if one appeared."""
    return None


def module(part, name):
    """Dotted module name of a part: module("castles", "castle_plan") → components.buildings.castles.castle_plan."""
    return f"{PKG}.{part}.{name}"
