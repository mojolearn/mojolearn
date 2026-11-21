# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba training door
(bindings/samba_training_native_door.mojo, door split 2026-11-21): the typed
words of one gradient operation and its three native segments: embedding
(with the dropout mask), the tail (final norm, head and loss, norm backward)
and the embedding backward (mask, tied accumulation). Typed Mojo only, no
Python import (tools/python_free_dirs.sh). The door releases the GIL once
around each segment; nothing here touches the interpreter.

One microbatch's words and a fixed sequence of entry calls, each reading the
previous one's output: no independent units to split (the kernels behind the
entries are parallel inside). The blocks between the segments are the block
door's."""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.neural_descriptor_run import neural_descriptor_record_admit, neural_descriptor_record_call
from core.samba_entry_table import (
    SAMBA_PRIM_EMBEDDING_FORWARD, SAMBA_PRIM_EMBEDDING_BACKWARD,
    SAMBA_PRIM_RMS_NORM_FORWARD, SAMBA_PRIM_RMS_NORM_BACKWARD, SAMBA_PRIM_LINEAR_FORWARD,
    SAMBA_PRIM_LINEAR_BACKWARD, SAMBA_PRIM_NEURAL_RNG, SAMBA_PRIM_HEAD_LOSS, SAMBA_PRIM_CE_LOSS,
    SAMBA_PRIM_ACCUMULATE, SAMBA_PRIM_CHUNKED_HEAD_TRAIN,
)

# The head forms of one gradient operation (`samba_train_head_form`).
comptime SAMBA_HEAD_AFN = 0      # the image's fused training tail (door call; the table does not carry it)
comptime SAMBA_HEAD_CHUNKED = 1  # chunked head + loss + head backward, one entry
comptime SAMBA_HEAD_FUSED = 2    # head + loss + head backward, one entry
comptime SAMBA_HEAD_PLAIN = 3    # linear forward, cross entropy, linear backward


@fieldwise_init
struct SambaTrainWords(Copyable, Movable):
    """One gradient operation's fixed words. `native_id` is the descriptor
    provider's id word (0: none; the stack's primitive entry `prim` is
    called). The door keeps the provider pinned for the operation."""
    var prim: EntryRef
    var native_id: Int
    var rows: Int
    var vocab: Int
    var d: Int
    var eps: Float64
    var items: Int


struct SambaDropoutMetadata(Copyable, Movable):
    # Stable configuration for one gradient operation, not persistent model
    # state. Stream position remains an explicit native operation argument.
    var probability: Float32
    var scale: Float32
    var seed_low: Int
    var seed_high: Int
    var stream: Int

    def __init__(out self, probability: Float64, seed_low: Int, seed_high: Int, stream: Int):
        self.probability = Float32(probability)
        self.scale = Float32(1.0 / (1.0 - Float64(self.probability)))
        self.seed_low = seed_low
        self.seed_high = seed_high
        self.stream = stream


struct SambaTailBuffers(Copyable, Movable):
    """Addresses of one tail segment's float32 buffers (and the int32
    targets `y`). The door owns every buffer and names it after the call.
    `row_denom` is read by the chunked form only, `logits` and `dlogits` by
    the plain form only (0 otherwise)."""
    var loss: Int
    var row_loss: Int
    var dh: Int
    var dnorm: Int
    var dhead: Int
    var hidden: Int
    var norm: Int
    var head: Int
    var y: Int
    var normalized: Int
    var dnormalized: Int
    var row_denom: Int
    var logits: Int
    var dlogits: Int

    def __init__(out self):
        self.loss = 0
        self.row_loss = 0
        self.dh = 0
        self.dnorm = 0
        self.dhead = 0
        self.hidden = 0
        self.norm = 0
        self.head = 0
        self.y = 0
        self.normalized = 0
        self.dnormalized = 0
        self.row_denom = 0
        self.logits = 0
        self.dlogits = 0


def samba_train_shape_check(layer_count: Int, configured: Int) raises:
    if layer_count != configured:
        raise Error("Samba: block registry does not match configured layer count")


def samba_train_head_form(afn: Bool, chunked: Bool, fused: Bool, count: Int, rows: Int, items: Int) -> Int:
    """Which head the operation runs. The chunked entry divides by the
    number of rows, so it serves only an operation whose every target counts
    and whose `num_items` is that count; `chunked` and `fused` are already
    False for an image with the fused training tail."""
    if afn:
        return SAMBA_HEAD_AFN
    if chunked and count == rows and items == count:
        return SAMBA_HEAD_CHUNKED
    if fused:
        return SAMBA_HEAD_FUSED
    return SAMBA_HEAD_PLAIN


def _admit(w: SambaTrainWords, operation: Int) raises:
    if w.native_id != 0:
        neural_descriptor_record_admit(w.native_id, operation)


def samba_train_embed_admit(w: SambaTrainWords) raises:
    """Admit and count the descriptor provider's operations of the embedding
    segment. The counts are written under the GIL (the door calls this
    before its release), as `neural_descriptor_execute_id` wrote them."""
    _admit(w, 0)


def samba_train_tail_admit(w: SambaTrainWords, form: Int) raises:
    """The same for the tail segment, in call order."""
    _admit(w, 2)
    if form == SAMBA_HEAD_PLAIN:
        _admit(w, 4)
        _admit(w, 5)
    _admit(w, 3)


def samba_train_embed_backward_admit(w: SambaTrainWords) raises:
    _admit(w, 1)


@always_inline
def _primitive(w: SambaTrainWords, provider_operation: Int, table_operation: Int, addresses: List[Int],
               ints: List[Int], floats: List[Float64]) raises:
    """One primitive on the route the operation resolved once: the descriptor
    provider's operation by id, else the stack's primitive entry."""
    if w.native_id != 0:
        _ = neural_descriptor_record_call(w.native_id, provider_operation, addresses, ints, floats)
    else:
        _ = entry_ref_call(w.prim, table_operation, addresses, ints, floats)


def samba_dropout_run(prim: EntryRef, dst: Int, src: Int, metadata: SambaDropoutMetadata, size: Int,
                      offset: Int, backward: Bool) raises:
    """The inverted-dropout mask of `size` floats from `src` into `dst`.
    Primitive op 6 (the former `neural_rng` entry's 8 words: 6 ints, 2
    floats); the same mask forward (2) and backward (3)."""
    var ints: List[Int] = [size, offset, metadata.seed_low, metadata.seed_high, metadata.stream, 3 if backward else 2]
    var floats: List[Float64] = [Float64(metadata.probability), Float64(metadata.scale)]
    var addresses: List[Int] = [dst, src]
    _ = entry_ref_call(prim, SAMBA_PRIM_NEURAL_RNG, addresses, ints, floats)


def samba_train_embed_run(w: SambaTrainWords, embedded: Int, weight: Int, ids: Int) raises:
    """Embedding lookup of `rows` ids into `embedded` (rows, d)."""
    var addresses: List[Int] = [embedded, weight, ids]
    var ints: List[Int] = [w.rows, w.vocab, w.d]
    var none = List[Float64]()
    _primitive(w, 0, SAMBA_PRIM_EMBEDDING_FORWARD, addresses, ints, none)


def samba_train_tail_run(w: SambaTrainWords, form: Int, t: SambaTailBuffers) raises:
    """Final RMS norm, the head with its loss and backward in the chosen
    form, and the norm's backward. Writes loss, row_loss, dhead, dnorm, dh."""
    var none = List[Float64]()
    var norm_ints: List[Int] = [w.rows, w.d]
    var norm_floats: List[Float64] = [w.eps]
    var shape_ints: List[Int] = [w.rows, w.vocab, w.d]
    var norm_addresses: List[Int] = [t.normalized, t.hidden, t.norm]
    _primitive(w, 2, SAMBA_PRIM_RMS_NORM_FORWARD, norm_addresses, norm_ints, norm_floats)
    if form == SAMBA_HEAD_CHUNKED:
        var chunked_addresses: List[Int] = [t.loss, t.row_loss, t.row_denom, t.dnormalized, t.dhead, t.normalized,
                                            t.head, t.y]
        _ = entry_ref_call(w.prim, SAMBA_PRIM_CHUNKED_HEAD_TRAIN, chunked_addresses, shape_ints, none)
    elif form == SAMBA_HEAD_FUSED:
        var fused_addresses: List[Int] = [t.loss, t.row_loss, t.dnormalized, t.dhead, t.normalized, t.head, t.y]
        var fused_ints: List[Int] = [w.rows, w.vocab, w.d, -100, 1, w.items]
        var fused_floats: List[Float64] = [0.0]
        _ = entry_ref_call(w.prim, SAMBA_PRIM_HEAD_LOSS, fused_addresses, fused_ints, fused_floats)
    else:
        var forward_addresses: List[Int] = [t.logits, t.normalized, t.head]
        _primitive(w, 4, SAMBA_PRIM_LINEAR_FORWARD, forward_addresses, shape_ints, none)
        # ce_loss words: [n_rows, vocab, ignore_index, reduction, num_items, want_grad], [label_smoothing].
        var loss_addresses: List[Int] = [t.loss, t.row_loss, t.dlogits, t.logits, t.y]
        var loss_ints: List[Int] = [w.rows, w.vocab, -100, 1, w.items, 1]
        var loss_floats: List[Float64] = [0.0]
        _ = entry_ref_call(w.prim, SAMBA_PRIM_CE_LOSS, loss_addresses, loss_ints, loss_floats)
        var backward_addresses: List[Int] = [t.dnormalized, t.dhead, t.dlogits, t.normalized, t.head]
        _primitive(w, 5, SAMBA_PRIM_LINEAR_BACKWARD, backward_addresses, shape_ints, none)
    var norm_backward: List[Int] = [t.dh, t.dnorm, t.dnormalized, t.hidden, t.norm]
    _primitive(w, 3, SAMBA_PRIM_RMS_NORM_BACKWARD, norm_backward, norm_ints, norm_floats)


def samba_train_embed_backward_run(w: SambaTrainWords, dembed: Int, dh: Int, ids: Int, tie: Bool, dhead: Int,
                                   pair: Int) raises:
    """Embedding backward of `dh` into `dembed`; with tied embeddings the
    head's gradient `dhead` is then accumulated into it through the two-row
    scratch `pair` (2, vocab * d), in the order embedding, head."""
    var addresses: List[Int] = [dembed, dh, ids]
    var ints: List[Int] = [w.rows, w.vocab, w.d]
    var none = List[Float64]()
    _primitive(w, 1, SAMBA_PRIM_EMBEDDING_BACKWARD, addresses, ints, none)
    if tie:
        var n = w.vocab * w.d
        var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=pair)
        var embedding_src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=dembed)
        var head_src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=dhead)
        unsafe_memcpy(dest=dst, src=embedding_src, count=n)
        unsafe_memcpy(dest=dst + n, src=head_src, count=n)
        var pair_addresses: List[Int] = [dembed, pair]
        var pair_ints: List[Int] = [n, 2, -1]
        _ = entry_ref_call(w.prim, SAMBA_PRIM_ACCUMULATE, pair_addresses, pair_ints, none)


def samba_loss_read(loss: Int) -> Float64:
    """The float32 loss the tail wrote at `loss`."""
    return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=loss)[0])
