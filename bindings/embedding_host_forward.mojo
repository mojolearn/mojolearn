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

from core.hostptr import f32_ptr, i32_ptr
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids_at,
    emb_refuse_shape,
    refuse_nonfinite_at,
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


def embedding_host_forward_words(a: List[Int], p: List[Int]) raises -> Int:
    """`Y[t, j] = W[ids[t], j]`, the device gather restated. Returns `T * d`.

    `addrs` = [weight (V * d f32, read), ids (T i32, read), y_out (T * d
    f32, WRITTEN)]; `params` = [V, d, T]. The GPU binding's order."""
    if len(a) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(a))
        )
    if len(p) != 3:
        raise Error(
            "embedding_forward: params must contain 3 values (V, d, T), got "
            + String(len(p))
        )
    var vocab = p[0]
    var width = p[1]
    var n_positions = p[2]
    var cfg = embedding_host_config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    # Lane ne3-c-blocks-embedding (rows N160, N158): the table and the ids
    # are read through the caller's buffers, never copied; the refusals keep
    # their words, their order (ids, W, then the y address) and their first
    # index (`emb_refuse_ids_at`, `refuse_nonfinite_at`).
    var wp = f32_ptr(a[0])
    var idsp = i32_ptr(a[1])
    var y_addr = a[2]
    with GILReleased(Python()):
        emb_refuse_ids_at(idsp, n_positions, cfg)
        refuse_nonfinite_at(String("W"), wp, vocab * width)
        var yp = f32_ptr(y_addr)
        host_embedding_forward(wp, idsp, n_positions, cfg, yp)
    return n_positions * width


def embedding_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_host_forward_words(embedding_host_words(addrs), embedding_host_words(params)))


def embedding_host_words(value: PythonObject) raises -> List[Int]:
    # UNPACK of a Python entry's argument words (the Python entries remain
    # for their other callers and the boundary tests).
    var out = List[Int]()
    for i in range(len(value)):  # small-loop(a Python entry's argument words)
        out.append(Int(py=value[i]))
    return out^
