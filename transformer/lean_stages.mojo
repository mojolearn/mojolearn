# SPDX-License-Identifier: Apache-2.0
"""Whether a transformer block call may build lean stages (lane
x4-transformer-on-executor, 2026-11-09: moved out of
transformer/transformer_binding_impl.mojo so the attention step op
(autograd/block_step_ops.mojo) and the binding's decode session, which now
runs that op, import it without an import cycle). Bodies unchanged."""

from transformer.block_options import BlockOptions
from transformer.impl.llama.fused_attention import fused_forward_supported_head_dim
from transformer.impl.llama.modeling_llama import ATTN_PATH_EAGER, PLANT_AT_NONE, attention_path_choice


def transformer_lean_stages(hd: Int) -> Bool:
    """Whether this call may skip the `[B, n_heads, L, S]` attention stage
    allocations: the fused attention path will be attempted (FAST or
    IDENTICAL build, supported head_dim, not forced eager) and the trace is off here
    always. The eager fallback grows the buffers on demand."""
    if not fused_forward_supported_head_dim(hd):
        return False
    return attention_path_choice(PLANT_AT_NONE) != ATTN_PATH_EAGER


def _lean_for(hd: Int, opts: BlockOptions) -> Bool:
    """`transformer_lean_stages`, and never lean under a softcap, which
    forces the eager kernels (DEVIATION 2947); the eager fallback would
    grow a lean struct anyway, so this is a cost choice, not a correctness
    one."""
    if opts.has_softcap():
        return False
    return transformer_lean_stages(hd)
