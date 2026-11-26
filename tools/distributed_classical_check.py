#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for tools/multi_gpu/verify_distributed.py (a multi-GPU tool, not in the wheel).

Use the target interpreter/package. Only the tool's own directory is put on the path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "multi_gpu"))
from verify_distributed import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
