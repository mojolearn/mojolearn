# SPDX-License-Identifier: Apache-2.0
"""ABI connection for native state/gradient operations of parallel MLP owners."""
import os

from . import _backend
from . import parallel_training as _parallel_buffer_connection
from ._mlp_impl import SmallMLPTrainer
from ._training_impl import AdamW
from .tokenizer import _binding

_native_parallel_mlp = _binding()._module.parallel_mlp_call
try:
    _training_binding = _backend.binding('_mojolearn_training', _backend.default_mode())
except (ImportError, ValueError):
    # glue: an install without the selected tier (a CPU-only install under
    # MOJOLEARN_NUMERIC_MODE=fast) still imports. No SmallMLPTrainer can be
    # built in that process (_mlp_impl refuses the tier), so no parallel MLP
    # owner reaches the unconnected boundary.
    _training_binding = None
if _training_binding is not None:
    _native_parallel_mlp('initialize', (SmallMLPTrainer, AdamW, _training_binding,
                                      os.environ._data, _backend.default_mode(), _backend.vendor()))
