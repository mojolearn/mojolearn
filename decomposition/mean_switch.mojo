# SPDX-License-Identifier: Apache-2.0
"""Shared device/host dispatch choice for decomposition variance means."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# The launch helper can select a tiled fold. Both columns must select it
# together; otherwise TruncatedSVD explained variance differs in raw bits.
comptime IDN_DECOMP_MEAN_LAUNCH = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_DECOMP_MEAN_LAUNCH_OFF"]()
    or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
