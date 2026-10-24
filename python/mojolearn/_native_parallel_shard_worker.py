# SPDX-License-Identifier: Apache-2.0
"""Private process entry of the Python-sharded multi-GPU drivers; the
installed native binding owns the worker loop. The driver modules register
their classes, entries and contexts at import."""
import os
import sys

if os.name == 'posix':
    # glue: a driver that times out (the verifier's cell limit) asks every
    # process of its group for its stacks with SIGUSR1; they go to stderr.
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

from . import parallel_classical, parallel_preprocessing, parallel_gaussian_process, parallel_ensemble  # noqa: F401
from . import parallel_neighbors, parallel_neighbors_reference, parallel_ivf  # noqa: F401
from ._native_parallel_shard import _native_parallel_shard

_native_parallel_shard('worker', sys.argv)
