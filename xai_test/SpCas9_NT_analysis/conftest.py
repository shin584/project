"""Put this subtree on `sys.path` so `test/` can import its modules directly.

Keeps `python -m pytest xai_test/SpCas9_NT_analysis/test -q` working from the
repo root without installing anything, and without the subtree's module names
leaking into the wider `xai_test/` suite.
"""

import sys
from pathlib import Path

SUBTREE_ROOT = Path(__file__).resolve().parent

if str(SUBTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBTREE_ROOT))
