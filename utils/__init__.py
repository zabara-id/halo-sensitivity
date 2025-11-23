"""
Package init helpers for the utils module.

We register ``utils.subfunctions`` under the top-level name ``subfunctions``
so legacy absolute imports inside the package (e.g. ``from subfunctions import ...``)
continue to work when the package is imported as ``utils``.
"""

from importlib import import_module
import sys

# Expose utils.subfunctions as a top-level module name if it's not already loaded.
if "subfunctions" not in sys.modules:
    _subfunctions = import_module(".subfunctions", __name__)
    sys.modules["subfunctions"] = _subfunctions
