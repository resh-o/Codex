"""Pytest configuration: make the backend package importable.

Adds the ``backend`` directory (this file's parent's parent) to ``sys.path`` so
``import app...`` works whether tests are run from the repo root or from
``backend/``.
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
