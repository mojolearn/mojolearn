# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba inference door (lane rs-neural, wave 8 run split,
from bindings/samba_inference_native.mojo): the Mamba-3 carried-state piece
shapes and names (the host route's primitive entry is held by the stack's
device route, bindings/samba_route_unpack.mojo). Typed Mojo only, no
Python import (tools/python_free_dirs.sh). Same bodies."""
from core.entry_table import EntryRef, entry_ref_call
from core.samba_entry_table import (
    SAMBA_PRIM_EMBEDDING_FORWARD, SAMBA_PRIM_RMS_NORM_FORWARD, SAMBA_PRIM_LINEAR_FORWARD, SAMBA_PRIM_CE_LOSS,
)


def _m3_shapes(b: Int, h: Int) -> List[List[Int]]:
    var result = List[List[Int]]()
    result.append([b, h, 32])
    result.append([b, h, 64, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 128])
    result.append([b, 64, h, 64])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, 64, h])
    result.append([b, h, 128])
    result.append([b, h, 64])
    return result^


def _m3_names() -> List[String]:
    return [String("theta"), String("h"), String("buffer_qrot"), String("buffer_krot"),
            String("buffer_v"), String("buffer_dt"), String("buffer_sig"), String("buffer_adt"),
            String("pending_k"), String("pending_v")]




# --------------------------------------------------------------------------
# RUN of the inference door (bindings/samba_inference_native_door.mojo, door
# split 2026-11-21): the admission checks in the door's words, the words and
# addresses of the stack program's open and forward, the host route's
# embedding and head, and the loss. The door releases the GIL around each
# entry call; nothing here touches the interpreter.
#
# One call's words and one entry call each: no independent units to split
# (the kernels behind the entries are parallel inside).


def samba_allocate_check(batch: Int, capacity: Int) raises:
    if batch < 1 or capacity < 1:
        raise Error("SambaStack.allocate_state: batch_size and max_tokens must be positive")


def samba_ids_rank_check(rank: Int, step: Bool) raises:
    if rank != 2 and not (step and rank == 1):
        raise Error("Samba: inputs must be (B,L) integer ids, or (B,) for step")


def samba_ids_extent_check(b: Int, l: Int, step: Bool) raises:
    if b < 1 or l < 1 or (step and l != 1):
        raise Error("Samba: input batch/length must be positive; step takes one token per row")


def samba_ids_signed(fmt: String) raises -> Bool:
    """Whether the admitted integer format is signed; any other format is
    refused in the door's words."""
    if fmt == "b" or fmt == "h" or fmt == "i" or fmt == "l" or fmt == "q":
        return True
    if fmt == "B" or fmt == "H" or fmt == "I" or fmt == "L" or fmt == "Q":
        return False
    raise Error("Samba: inputs must be native-endian integer ids")


def samba_program_open_words(layers: List[Int], vocab: Int, d_model: Int, n_heads: Int, n_kv_heads: Int,
                             head_dim: Int, intermediate: Int, tie: Bool, n_total: Int, b: Int,
                             smax: Int) -> List[Int]:
    """SAMBA_PROGRAM_OPEN's integer words: [n_layers, layer codes (0 mamba3,
    1 attention), vocab, d_model, n_heads, n_kv_heads, head_dim,
    intermediate, tie, n_total, batch_size, max_tokens]."""
    var n = len(layers)
    var words = List[Int](capacity=n + 11)
    words.append(n)
    for i in range(n):  # driver-loop: over model layer metadata at program creation
        words.append(layers[i])
    words.append(vocab)
    words.append(d_model)
    words.append(n_heads)
    words.append(n_kv_heads)
    words.append(head_dim)
    words.append(intermediate)
    words.append(1 if tie else 0)
    words.append(n_total)
    words.append(b)
    words.append(smax)
    return words^


def samba_program_stateless_addresses(registry: Int, ids: Int, logits: Int) -> List[Int]:
    """[registry, ids, logits, cursors_out (none)] of a stateless forward."""
    var addresses: List[Int] = [registry, ids, logits, 0]
    return addresses^


def samba_program_stateless_words(b: Int, l: Int) -> List[Int]:
    """[B, L, report count (none)] of a stateless forward."""
    var words: List[Int] = [b, l, 0]
    return words^


def samba_program_state_addresses(registry: Int, ids: Int, logits: Int, cursors: Int, reports: List[Int],
                                  pieces: List[Int]) -> List[Int]:
    """[registry, ids, logits, cursors_out], the Mamba-3 report addresses in
    layer order, then the state's piece addresses: one allocation of the
    exact size."""
    var addresses = List[Int](capacity=4 + len(reports) + len(pieces))
    addresses.append(registry)
    addresses.append(ids)
    addresses.append(logits)
    addresses.append(cursors)
    for i in range(len(reports)):  # driver-loop: over the Mamba-3 layers' report addresses
        addresses.append(reports[i])
    for i in range(len(pieces)):  # driver-loop: over the state's piece addresses (per layer)
        addresses.append(pieces[i])
    return addresses^


def samba_program_state_words(b: Int, l: Int, reports: Int, cursors: List[Int]) -> List[Int]:
    """[B, L, report count], then the state's cursor words (two per layer)."""
    var words = List[Int](capacity=3 + len(cursors))
    words.append(b)
    words.append(l)
    words.append(reports)
    for i in range(len(cursors)):  # driver-loop: over the state's cursor words (two per layer)
        words.append(cursors[i])
    return words^


def samba_host_state_check(state_batch: Int, state_layers: Int, b: Int, layers: Int) raises:
    if state_batch != b or state_layers != layers:
        raise Error("Samba: decode state batch or layer count differs")


def samba_host_embed_run(prim: EntryRef, hidden: Int, weight: Int, ids: Int, rows: Int, vocab: Int,
                         d_model: Int) raises:
    """The host route's embedding lookup: `rows` ids into (rows, d_model)."""
    var addresses: List[Int] = [hidden, weight, ids]
    var ints: List[Int] = [rows, vocab, d_model]
    _ = entry_ref_call(prim, SAMBA_PRIM_EMBEDDING_FORWARD, addresses, ints, List[Float64]())


def samba_host_head_run(prim: EntryRef, logits: Int, normalized: Int, hidden: Int, norm_weight: Int, head: Int,
                        rows: Int, vocab: Int, d_model: Int, eps: Float64) raises:
    """The host route's final RMS norm and head projection, in that order."""
    var norm_addresses: List[Int] = [normalized, hidden, norm_weight]
    var norm_ints: List[Int] = [rows, d_model]
    var norm_floats: List[Float64] = [eps]
    _ = entry_ref_call(prim, SAMBA_PRIM_RMS_NORM_FORWARD, norm_addresses, norm_ints, norm_floats)
    var head_addresses: List[Int] = [logits, normalized, head]
    var head_ints: List[Int] = [rows, vocab, d_model]
    _ = entry_ref_call(prim, SAMBA_PRIM_LINEAR_FORWARD, head_addresses, head_ints, List[Float64]())


def samba_loss_rows_check(targets: Int, rows: Int) raises:
    if targets != rows:
        raise Error("SambaStack.loss: targets must contain one class index per logit row")


def samba_ce_loss_run(entry: EntryRef, loss: Int, row_loss: Int, gradient: Int, logits: Int, targets: Int,
                      rows: Int, vocab: Int) raises -> Float64:
    """Mean cross entropy of (rows, vocab) logits. ce_loss words: [n_rows,
    vocab, ignore_index, reduction (2 mean), num_items, want_grad],
    [label_smoothing]. Returns the float32 loss written at `loss`."""
    var addresses: List[Int] = [loss, row_loss, gradient, logits, targets]
    var ints: List[Int] = [rows, vocab, -100, 2, 0, 0]
    var floats: List[Float64] = [0.0]
    _ = entry_ref_call(entry, SAMBA_PRIM_CE_LOSS, addresses, ints, floats)
    return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=loss)[0])
