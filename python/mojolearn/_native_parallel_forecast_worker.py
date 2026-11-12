# SPDX-License-Identifier: Apache-2.0
"""Private process entry; the installed native binding owns the worker loop."""
import os
import sys

# Lazy family import (lane lazy-family-import): a worker runs on the loaded
# set the package had before it, every family and unit, as its parent may hand
# it any of them.
from . import _families as _worker_families
_worker_families.load_all()
from .parallel_forecasting import _native_parallel_forecast

_native_parallel_forecast('worker', sys.argv)

# glue: process lifecycle only. The native loop returns only after the pool's
# close request, when every reply has been delivered and the owner discards
# this worker's state. End the process here instead of finalizing the
# interpreter and every loaded native runtime, which the owner's reap would
# otherwise wait for (bindings/native_process_pool.mojo, _process_reap_group).
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
