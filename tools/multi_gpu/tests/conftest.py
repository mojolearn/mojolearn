# SPDX-License-Identifier: Apache-2.0
"""The multi-GPU tools (tools/multi_gpu/) are scripts beside these tests, not wheel modules: put them on the path."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
