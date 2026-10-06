# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Source-only Apple FAST predictive-variance work grouping."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-L11: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified, OFF.
# Four SIMD groups (128 threads) per query tile instead of eight expose
# more independent groups for a given query count. Each query retains its
# complete ascending training-axis sum; means, solves, variance clamping
# and probability integration are unchanged. No training/query cap is added.
comptime AFCL_L11 = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                    and is_defined["MOJOLEARN_AFCL_L11"]())
comptime AFCL_GP_VAR_TPB = 128 if AFCL_L11 else 256
