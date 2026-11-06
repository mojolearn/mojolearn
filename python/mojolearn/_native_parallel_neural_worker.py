# SPDX-License-Identifier: Apache-2.0
"""Private process entry; the installed native binding owns the worker loop."""
import os
import sys

from ._parallel_neural import _native_parallel_neural

_native_parallel_neural('worker', sys.argv)

# glue: process lifecycle only. The native loop returns only after the pool's
# close request, when every reply has been delivered and the owner discards
# this worker's state. End the process here instead of finalizing the
# interpreter and every loaded native runtime, which the owner's reap would
# otherwise wait for (bindings/native_process_pool.mojo, _process_reap_group).
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
