#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point; the CV verifier ships in the installed wheel."""
from mojolearn._verify_parallel_cv import main, source_identity

if __name__ == '__main__':
    raise SystemExit(main())
