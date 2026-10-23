# SPDX-License-Identifier: Apache-2.0
"""Private process entry; the installed native binding owns the worker loop."""
import sys

from .parallel_forecasting import _native_parallel_forecast

_native_parallel_forecast('worker', sys.argv)
