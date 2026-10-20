# SPDX-License-Identifier: Apache-2.0
"""ABI connection for the native legacy parallel neural owner."""
from . import _native_process as _process_connection
from . import _parallel_mlp as _mlp_connection
from ._mlp_impl import SmallMLPTrainer
from ._samba_impl import SambaStack
from ._native_samba_bridge import CONTEXT
from .parallel_training import ParallelNeuralTrainer
from .tokenizer import _binding

_module = _binding()._module
_native_parallel_neural = _module.parallel_neural_call
_native_parallel_neural('initialize', (ParallelNeuralTrainer, SmallMLPTrainer, SambaStack,
                                     _module.samba_native_call, CONTEXT))
