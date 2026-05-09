from __future__ import annotations

from .training_runs.preflight import recommendations as _recommendations

for _name in dir(_recommendations):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_recommendations, _name)

__all__ = [
    _name
    for _name in dir(_recommendations)
    if not (_name.startswith("__") and _name.endswith("__"))
]
