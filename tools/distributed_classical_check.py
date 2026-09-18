#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point; implementation ships in the installed wheel.

Use the target interpreter/package. No source-path injection is performed.
"""
from mojolearn._verify_distributed import main

if __name__ == '__main__':
    raise SystemExit(main())
