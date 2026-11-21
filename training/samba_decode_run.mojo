# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba decode-session doors
(bindings/samba_decode_native_unpack.mojo and _pack.mojo, door split
2026-11-21): the session's admission checks in the doors' words, the piece
counts of a stack state, the state-to-state copy of the host arm, the
program requests of open and export at their exact size, and the host arm's
per-token column and logits moves. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The program call itself is the typed table call
of the inference door (`samba_program_call`).

Tokens of one session are sequential by definition (token t reads the state
token t - 1 wrote); the copies here are single memory moves."""
from std.memory import unsafe_memcpy
from training.samba_inference_run import _m3_shapes


def samba_decode_layers_check(what: String, held: Int, configured: Int) raises:
    if held != configured:
        raise Error(what + ": the state holds " + String(held) + " layers, the stack " + String(configured))


def samba_decode_rows_check(what: String, b: Int, smax: Int) raises:
    if b < 1 or smax < 1:
        raise Error(what + ": the state holds no rows")


def samba_decode_batch_check(what: String, rows: Int, given: Int) raises:
    if given != rows:
        raise Error(what + ": the session holds " + String(rows) + " rows, inputs has B = " + String(given))


def samba_attention_counts(mut counts: List[Int], b: Int, n_kv_heads: Int, smax: Int, head_dim: Int):
    """The float32 counts of an attention layer's k_cache and v_cache."""
    var n = b * n_kv_heads * smax * head_dim
    counts.append(n)
    counts.append(n)


def samba_m3_counts(mut counts: List[Int], b: Int, heads: Int):
    """The float32 counts of a Mamba-3 layer's ten carried pieces
    (`_m3_shapes`), in their order."""
    var shapes = _m3_shapes(b, heads)
    for k in range(len(shapes)):  # driver-loop: over the fixed Mamba3 continuation buffer ABI
        var n = 1
        for j in range(len(shapes[k])):  # small-loop(piece rank)
            n *= shapes[k][j]
        counts.append(n)


def samba_state_copy_run(dst: List[Int], src: List[Int], counts: List[Int]):
    """Every float32 piece of one stack state into another of the same stack,
    rows and capacity."""
    for i in range(len(counts)):  # driver-loop: over the state's pieces (per layer)
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst[i]),
                      src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=src[i]), count=counts[i])


def samba_decode_request_addresses(first: Int, pieces: List[Int]) -> List[Int]:
    """[first, the state's piece addresses in layer order]: the open request
    (first = the parameter registry) and the export request (first = the
    cursor words out), one allocation of the exact size."""
    var addresses = List[Int](capacity=1 + len(pieces))
    addresses.append(first)
    for i in range(len(pieces)):  # driver-loop: over the state's piece addresses (per layer)
        addresses.append(pieces[i])
    return addresses^


def samba_decode_open_words(opened: List[Int], cursors: List[Int]) -> List[Int]:
    """The program's open words, then the state's cursor words (two per
    layer), one allocation of the exact size."""
    var words = List[Int](capacity=len(opened) + len(cursors))
    for i in range(len(opened)):  # driver-loop: over the program's open words
        words.append(opened[i])
    for i in range(len(cursors)):  # driver-loop: over the state's cursor words (two per layer)
        words.append(cursors[i])
    return words^


def samba_decode_column_run(column: Int, ids: Int, b: Int, n: Int, t: Int):
    """Token `t` of the (b, n) int32 ids into the (b, 1) step column: the
    host arm's per-token input."""
    var src = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=ids)
    var col = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=column)
    for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's id
        col[r] = src[r * n + t]


def samba_decode_logits_run(output: Int, logits: Int, b: Int, n: Int, t: Int, v: Int):
    """The step's (b, v) logits into token `t` of the (b, n, v) output."""
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=output)
    var row = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=logits)
    for r in range(b):  # driver-loop: over the batch rows, a host-arm copy of token t's logits
        unsafe_memcpy(dest=dst + (r * n + t) * v, src=row + r * v, count=v)
