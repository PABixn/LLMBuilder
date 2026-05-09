from __future__ import annotations

import sys

from ..training_runs.executors.runpod import executor as _executor

sys.modules[__name__] = _executor
