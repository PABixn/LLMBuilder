from __future__ import annotations

import sys

from ..training_runs.executors.runpod import remote_sync as _remote_sync

sys.modules[__name__] = _remote_sync
