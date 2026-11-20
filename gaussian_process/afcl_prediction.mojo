# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Source-only Apple FAST predictive-variance work grouping."""

from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-L11: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified, OFF.
# Four SIMD groups (128 threads) per query tile instead of eight expose
# more independent groups for a given query count. Each query retains its
# complete ascending training-axis sum; means, solves, variance clamping
# and probability integration are unchanged. No training/query cap is added.
comptime AFCL_L11 = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
                    and is_defined["MOJOLEARN_AFCL_L11"]())
comptime AFCL_GP_VAR_TPB = 128 if AFCL_L11 else 256
# TOMBSTONE: MOJOLEARN_AFCL_L11 (never-run Apple FAST programme card) deleted 2026-11-15 by lane/expdel2-19-kernel; code recoverable at 7fa6378a2a.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_L11.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.
# (Sweep 2026-11-15: comptime AFCL_L11 removed; its reader in gaussian_process/checks/kernels.mojo folded to elem_tpb.)
