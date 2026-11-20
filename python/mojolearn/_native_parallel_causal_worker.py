# SPDX-License-Identifier: Apache-2.0
"""Installed connection to the compiled persistent causal-layer worker."""
import sys
# Lazy family import (lane lazy-family-import): a worker runs on the loaded
# set the package had before it, every family and unit, as its parent may hand
# it any of them.
from . import _families as _worker_families
_worker_families.load_all()
from .models.parallel_causal_lm import _native_parallel_causal

if __name__ == "__main__":
    _native_parallel_causal("worker", sys.argv)
