# SPDX-License-Identifier: Apache-2.0
# Optional public Mamba2 step storage capability. Mutable weight/state bytes are
# refreshed on every step; this resource does not establish immutable weights.
# Uses NativeDescriptorCall. Create has no spans; close has [handle]; step has
# [handle, sixteen state-ABI addresses], [B, d_model, q0], [dt_lo, dt_hi].
comptime MAMBA2_WORKSPACE_ABI_NAME = "mojolearn.mamba2.public_workspace.v1"
comptime MAMBA2_WORKSPACE_CREATE = 0
comptime MAMBA2_WORKSPACE_CLOSE = 1
comptime MAMBA2_WORKSPACE_STEP = 2
comptime MAMBA2_WORKSPACE_OP_COUNT = 3
