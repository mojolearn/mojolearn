# SPDX-License-Identifier: Apache-2.0
"""NN24 device launchers for the common scalar normalization profile.

Model RMS forward/backward, residual RMS and RMS/LayerNorm inference select
norm_profile_contract helpers coherently. Extended-options training still
refuses at the existing public API. Default OFF; no compilation, identity,
quality or timing has been run. Standalone centered-LN backward is an optional
component, not a claim of new public LayerNorm training support.
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_mul, identical_mul_add, identical_rsqrt,
)

from transformer.experiments.norm_profile_contract import (
    NN24_NORM_LANES8, NN24_LANES, _merge, _sum, _square, norm_profile_dot,
    norm_profile_forward_row, norm_profile_backward_row, norm_profile_parameter_column,
    norm_profile_host_forward, norm_profile_host_backward,
)

