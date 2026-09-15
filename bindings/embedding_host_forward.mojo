# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`embedding_forward` on the host, shared by two bindings
(lane/inference-embedding-ivf-cholesky, 2026-09-15).

`bindings/_mojolearn_embedding_host.mojo` (the internal reference binding,
backward included) and `bindings/_mojolearn_embedding_infer_host.mojo` (the
inference binding the wheels ship, no backward) both register
`embedding_forward` from here, so the two binaries look a saved table up
through the same source. Not a binding itself: it registers nothing, and the
host surface tests glob only `_mojolearn_*_host.mojo`.

The contract is `bindings/_mojolearn_embedding.mojo`'s: `addrs` = [weight
(V * d f32, read), ids (T i32, read), y_out (T * d f32, WRITTEN)]; `params`
= [V, d, T]. The refusals are the GPU binding's, in its words and order,
through `embedding/checks/embedding_oracle.mojo`.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, read_f32, read_i32
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    emb_refuse_shape,
    refuse_nonfinite,
)
from embedding.host.embedding_host import host_embedding_forward


def embedding_host_config(
    vocab: Int, width: Int, padding_idx: Int, accumulate: Bool
) raises -> EmbConfig:
    """The GPU binding's `_config`, in its words."""
    if padding_idx != EMB_NO_PADDING_IDX and (padding_idx < 0 or padding_idx >= vocab):
        raise Error(
            String("embedding: padding_idx = ")
            + String(padding_idx)
            + " is outside [0, "
            + String(vocab)
            + ") REFUSED (contract 8; -1 means no padding row)"
        )
    return EmbConfig(vocab, width, padding_idx, accumulate)


def embedding_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Y[t, j] = W[ids[t], j]`, the device gather restated. Returns `T * d`.

    `addrs` = [weight (V * d f32, read), ids (T i32, read), y_out (T * d
    f32, WRITTEN)]; `params` = [V, d, T]. The GPU binding's order."""
    if len(addrs) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(addrs))
        )
    if len(params) != 3:
        raise Error(
            "embedding_forward: params must contain 3 values (V, d, T), got "
            + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var cfg = embedding_host_config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var weight = read_f32(Int(py=addrs[0]), vocab * width)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("W"), weight)
    var yp = f32_ptr(Int(py=addrs[2]))
    with GILReleased(Python()):
        host_embedding_forward(weight, ids, n_positions, cfg, yp)
    return PythonObject(n_positions * width)
