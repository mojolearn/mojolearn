# SPDX-License-Identifier: Apache-2.0
"""CPU-only training GEMM boundary; no GPU module is imported."""
from std.python import Python,PythonObject
from core.neural_gemm_boundary_host_impl import neural_gemm_checked_host_into
from std.python._cpython import GILReleased
from std.math import isfinite
from core.hostptr import f32_ptr
from bindings.neural_gemm_boundary_common_unpack import neural_gemm_params
from gemm.host.neural_gemm import neural_gemm_host_into


