# SPDX-License-Identifier: Apache-2.0
"""RUN half of `providers/python/autograd_program_pyapi.mojo` (wave 8 run
split): the buffer format kind, the builtin class words a list input is read
against, the session shape of an imported value, the typed layout check of a
buffer-protocol input, the typed ProgramInput of a converted sequence, and the
fit's loss code. Typed Mojo only: no Python import."""
from core.cast_in import CastIn, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL
from autograd.rankn import ShapeN
from autograd.program_io import ProgramInput


def _buffer_kind(format: String, itemsize: Int) -> Int:
    """The CAST_KIND_* of a native-order buffer format of one of the
    supported element types, or -1 (the caller then takes the attribute
    path, which spells every refusal)."""
    var code = format
    if code.byte_length() == 2:
        var order = String(code[byte=0])
        # '@' and '=' are native order; '<' is native on every supported
        # (little-endian) host; '>' and '!' fall back (refused there).
        if order != "@" and order != "=" and order != "<":
            return -1
        var element = String(code[byte=1])
        code = element^
    elif code.byte_length() != 1:
        return -1
    if code == "?":
        return CAST_KIND_BOOL if itemsize == 1 else -1
    if code == "b" or code == "h" or code == "i" or code == "l" or code == "q":
        return CAST_KIND_SIGNED
    if code == "B" or code == "H" or code == "I" or code == "L" or code == "Q":
        return CAST_KIND_UNSIGNED
    if (code == "e" and itemsize == 2) or (code == "f" and itemsize == 4) or (code == "d" and itemsize == 8):
        return CAST_KIND_FLOAT
    return -1


@fieldwise_init
struct _SequenceWords(ImplicitlyCopyable, Movable):
    """The exact builtin class words a list input is read against."""
    var list_word: Int
    var tuple_word: Int
    var int_word: Int
    var float_word: Int
    var bool_word: Int


def _imported_session_dims(dims: List[Int]) -> List[Int]:
    # importers/_table.session_shape: a session tensor has at least two axes,
    # a scalar is (1, 1) and a [n] vector (1, n).
    var out = List[Int]()
    for _ in range(2 - len(dims)):  # driver-loop: over rank metadata (at most eight axes); small-loop(2: the padded leading axes): at most two shape entries
        out.append(1)
    for k in range(len(dims)):  # driver-loop: over rank metadata (at most eight axes); small-loop(dims: rank, at most eight): shape metadata, never array elements
        out.append(dims[k])
    return out^


def array_input_layout(kind: Int, address: Int, itemsize: Int, dims_in: List[Int], steps: List[Int], what: String,
                       mut out_src: CastIn, mut out_shape: ShapeN) raises -> Bool:
    """`_array_input_buffer`'s typed half: the session shape and byte strides
    of one buffer read. False leaves the outputs untouched: the caller takes
    the attribute path."""
    var nd = len(dims_in)
    if kind < 0 or nd > 8:
        return False
    var count = 1
    for k in range(nd):  # driver-loop: over rank metadata (at most eight axes); small-loop(nd: input tensor rank): shape metadata, not array elements
        count *= dims_in[k]
    if count == 0:
        return False
    var dims = List[Int]()
    if nd == 0:
        dims.append(1)
        dims.append(1)
    elif nd == 1:
        dims.append(dims_in[0])
        dims.append(1)
    else:
        for k in range(nd):  # driver-loop: over rank metadata (at most eight axes); small-loop(nd: input tensor rank): shape metadata, not array elements
            dims.append(dims_in[k])
    var sh = ShapeN.of(dims)
    var s0 = sh.cols() * itemsize
    var s1 = itemsize
    if nd == 1:
        s0 = steps[0]
    elif nd == 2:
        s0 = steps[0]
        s1 = steps[1]
    elif nd > 2:
        # C-contiguous (NumPy's flag rule: axes of length 1 are skipped)
        var want = itemsize
        for r in range(nd):  # driver-loop: over rank metadata (at most eight axes); small-loop(nd: input tensor rank): stride metadata, not array elements
            var k = nd - 1 - r
            if dims_in[k] != 1 and steps[k] != want:
                return False
            want *= dims_in[k]
    var src = CastIn(address, kind, itemsize, sh.rows(), sh.cols(), s0, s1)
    src.check(what)
    out_src = src
    out_shape = sh
    return True


def program_sequence_input(var dims: List[Int], var values: List[Float64], var ids: List[Int64],
                           integer: Bool) raises -> ProgramInput:
    """`program_input`'s converted sequence as its typed input (the values
    move in; the descriptor addresses them)."""
    if len(dims) == 0:
        dims.append(1)
    if len(dims) == 1:
        dims.append(1)
    var shape = ShapeN.of(dims)
    if integer:
        var id_src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8,
                            shape.rows(), shape.cols(), shape.cols() * 8, 8)
        return ProgramInput(id_src, shape, values^, ids^)
    var src = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 8,
                     shape.rows(), shape.cols(), shape.cols() * 8, 8)
    return ProgramInput(src, shape, values^, ids^)


def imported_sequence_input(dims: List[Int], var values: List[Float64], var ids: List[Int64],
                            integer: Bool) raises -> ProgramInput:
    """`_imported_capture_input`'s converted sequence as its typed input, at
    the session shape of its value."""
    var sh = ShapeN.of(_imported_session_dims(dims))
    if integer:
        var src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 8, sh.rows(), sh.cols(), sh.cols() * 8, 8)
        return ProgramInput(src, sh, values^, ids^)
    var src = CastIn(Int(values.unsafe_ptr()), CAST_KIND_FLOAT, 8, sh.rows(), sh.cols(), sh.cols() * 8, 8)
    return ProgramInput(src, sh, values^, ids^)


def program_fit_loss_kind(name: String) raises -> Int:
    """The fit's loss code for its public name."""
    var kind = -1
    if name == "output":
        kind = 0
    elif name == "mse":
        kind = 1
    elif name == "mae":
        kind = 2
    elif name == "huber":
        kind = 3
    elif name == "bce_with_logits":
        kind = 4
    elif name == "cross_entropy":
        kind = 5
    if kind < 0:
        raise Error("ValueError: unknown loss " + name)
    return kind
