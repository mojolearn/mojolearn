#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for tools/multi_gpu/verify_parallel_cv.py (a multi-GPU tool, not in the wheel)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "multi_gpu"))
from verify_parallel_cv import main, source_identity  # noqa: E402,F401

if __name__ == '__main__':
    raise SystemExit(main())
