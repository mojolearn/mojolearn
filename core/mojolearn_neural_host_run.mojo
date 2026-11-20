# SPDX-License-Identifier: Apache-2.0
"""RUN half of `_mojolearn_neural_host` (lane rs-neural, wave 8 run split,
from bindings/_mojolearn_neural_host.mojo): the bodies its doors ran with the
GIL released. The doors keep the address and parameter unpacking, release the
GIL around each call exactly where they did, and pack the returned count.
Typed Mojo only, no Python import (tools/python_free_dirs.sh). Same
statements in the same order, so the bits are unchanged."""
from core.hostptr import copy_f32, f32_ptr, read_f32
from core.mojolearn_neural_host_impl import _finite, _transformer_weights_from
from gemm.contract import OP_NT
from gemm.host.neural_gemm import gemm_host_rows
from training.host.mlp_oracle import host_mlp_bias_activation
from transformer.host.transformer_block_host import transformer_host_forward
from transformer.block_options import BlockOptions


def mlp_forward_logits_run(a: List[Int], rows: Int) raises -> Int:
    """The 8-16-3 MLP's logits into a[5]; returns rows * 3, the cells
    written."""
    var x = read_f32(a[0], rows * 8)
    var w1 = read_f32(a[1], 16 * 8)
    var b1 = read_f32(a[2], 16)
    var w2 = read_f32(a[3], 3 * 16)
    var b2 = read_f32(a[4], 3)
    var p1 = gemm_host_rows(x, w1, OP_NT, rows, 16, 8)
    _finite(p1, String("mlp_forward_logits: the first projection"))
    var hidden = host_mlp_bias_activation(p1, b1, rows, 16, 1)
    var p2 = gemm_host_rows(hidden, w2, OP_NT, rows, 3, 16)
    _finite(p2, String("mlp_forward_logits: the second projection"))
    var logits = host_mlp_bias_activation(p2, b2, rows, 3, 0)
    copy_f32(logits.unsafe_ptr(), f32_ptr(a[5]), rows * 3)
    return rows * 3


def transformer_forward_fresh_run(
    a: List[Int], b: Int, l: Int, dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, window: Int,
    opts: BlockOptions, what: String,
) raises -> Int:
    """Stateless prefill from a zero cache of capacity L (or a window ring)
    in `transformer_forward`'s 13 + 11 address layout; only y (a[12]) is
    written. Returns the cached-token count."""
    var w = _transformer_weights_from(a, dm, nh, nkv, hd, it, l, opts, what)
    var x = read_f32(a[0], b * l * dm)
    var out = transformer_host_forward(
        w, x, b, l, l, 0, window, List[Float32](), List[Float32](), False
    )
    copy_f32(out.y.unsafe_ptr(), f32_ptr(a[12]), len(out.y))
    return out.cached_tokens
