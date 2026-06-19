from __future__ import annotations
import importlib
from functools import lru_cache
from types import ModuleType
from typing import Optional


@lru_cache(maxsize=1)
def load_wilson_extension() -> Optional[ModuleType]:
    """Return the optional compiled Wilson extension if it can be imported.

    The extension is expected to be importable as gnn_graph_sampling._wilson_ext.
    The function caches the result so that the import is attempted at most once.
    """

    try:
        return importlib.import_module('gnn_graph_sampling._wilson_ext')
    except Exception:
        return None
