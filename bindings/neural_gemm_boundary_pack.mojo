# SPDX-License-Identifier: Apache-2.0
"""Training binding's neural-only GEMM; preparation and output stay in-call."""
from std.python import Python,PythonObject
from std.python._cpython import GILReleased
from training.neural_gemm_native import neural_gemm_native
from bindings.neural_gemm_boundary_common_unpack import neural_gemm_params

