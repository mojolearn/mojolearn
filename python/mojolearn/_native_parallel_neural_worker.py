# SPDX-License-Identifier: Apache-2.0
"""Private process entry; the installed native binding owns the worker loop."""
import sys

from ._parallel_neural import _native_parallel_neural

_native_parallel_neural('worker', sys.argv)
