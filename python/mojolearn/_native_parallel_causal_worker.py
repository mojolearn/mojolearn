# SPDX-License-Identifier: Apache-2.0
"""Installed connection to the compiled persistent causal-layer worker."""
import sys
from .models.parallel_causal_lm import _native_parallel_causal

if __name__ == "__main__":
    _native_parallel_causal("worker", sys.argv)
