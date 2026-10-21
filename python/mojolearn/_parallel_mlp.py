# SPDX-License-Identifier: Apache-2.0
"""ABI connection for native state/gradient operations of parallel MLP owners."""
import os

from . import _backend
from . import parallel_training as _parallel_buffer_connection
from ._mlp_impl import SmallMLPTrainer
from ._training_impl import AdamW
from .tokenizer import _binding

_native_parallel_mlp = _binding()._module.parallel_mlp_call
_native_parallel_mlp('initialize', (SmallMLPTrainer, AdamW,
                                  _backend.binding('_mojolearn_training', _backend.default_mode()),
                                  os.environ._data, _backend.default_mode(), _backend.vendor()))
