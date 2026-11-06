"""
ref_ranges.py — moved to scripts/common/ranges.py (shared with the unharmonized suggestions).

Kept as a thin alias so the scripts_ref experiments keep importing `ref_ranges`.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common.ranges import *              # noqa: E402,F401,F403
from common.ranges import _RSD_FLOOR     # noqa: E402,F401
