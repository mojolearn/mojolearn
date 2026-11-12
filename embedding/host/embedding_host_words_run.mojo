# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of the host Embedding doors (`bindings/embedding_host_forward.mojo`,
`bindings/_mojolearn_embedding_host.mojo`, and through the first the
inference binding `bindings/_mojolearn_embedding_infer_host.mojo`): the
configuration refusal and the typed bodies the doors run with the GIL
released. Typed Mojo values and addresses only, no Python import
(tools/python_free_dirs.sh). The doors unpack the argument words and release
the GIL around each `*_released` call exactly where they did before the split
(lane rs-embedding-bytelm, 2026-11-12). The bodies moved unchanged, so the
bits are unchanged."""
from core.hostptr import f32_ptr
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids_at,
    refuse_nonfinite_at,
)
from embedding.host.embedding_host import host_embedding_backward, host_embedding_forward


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


def embedding_host_forward_released(
    wp: MutPointer[Float32, MutUntrackedOrigin],
    idsp: MutPointer[Int32, MutUntrackedOrigin],
    n_positions: Int,
    cfg: EmbConfig,
    y_addr: Int,
) raises:
    """The typed body of the host `embedding_forward` that the door runs with
    the GIL released: the refusals in their order (ids, W, then the y
    address), then the gather."""
    emb_refuse_ids_at(idsp, n_positions, cfg)
    refuse_nonfinite_at(String("W"), wp, cfg.vocab * cfg.width)
    var yp = f32_ptr(y_addr)
    host_embedding_forward(wp, idsp, n_positions, cfg, yp)


def embedding_host_backward_released(
    dwp: MutPointer[Float32, MutUntrackedOrigin],
    dyp: MutPointer[Float32, MutUntrackedOrigin],
    ids: List[Int32],
    n_positions: Int,
    cfg: EmbConfig,
    plan: Int,
    cells: Int,
) raises:
    """The typed body of the host `embedding_backward` that the door runs
    with the GIL released: the fold, when there is a cell to write."""
    if cells > 0:
        host_embedding_backward(dwp, dyp, ids, n_positions, cfg, plan)
