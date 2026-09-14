# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the GPT-2 byte-level BPE tokenizer (`tokenizer/`), the
expose-tokenizer lane, 2026-09-14
(docs/lanes/BRIEF_expose_tokenizer_2026-09-14.md).

HOST ONLY, AND THE ONLY BINDING THIS FAMILY HAS. `tokenizer/` is integer
and table work (a byte buffer, three Unicode range tables, a hash probe and
an argmin over small integers): no float arithmetic, no device kernel, no
GPU binding to route from. So unlike the other host families this one is
not a CPU twin of a GPU entry; it is the door itself, loaded by path through
`_backend.load_host_module` from `python/mojolearn/tokenizer.py`, and the
same binary serves a GPU box and a CPU-only install alike.

WHAT IT COMPUTES. `tokenizer/encoding.mojo::Gpt2Tokenizer.encode_bytes` and
`decode_bytes`: the id sequence tiktoken 0.14.0's `gpt2` encoding assigns to
a byte string, id for id, and the bytes back. `tokenizer/checks/
tokenizer_check.mojo` (`pixi run check-tokenizer`) is where that agreement
is asserted against 43 recorded cases; this file adds no arithmetic and no
policy. `<|endoftext|>` (id 50256) is a token only when the caller passes
`allow_endoftext=True`, otherwise its thirteen characters are ordinary text
(both readings are in the fixture).

THE ADDRESS CONTRACT, mirrored word for word in `python/mojolearn/
tokenizer.py`:

    gpt2_load(ranks_path, unicode_path) -> handle
        parses the two tables once and returns an opaque `_Gpt2Handle`
        every other entry takes first. One handle per GPT2Tokenizer.
    gpt2_n_vocab(handle) -> 50257
    gpt2_max_token_bytes(handle) -> the longest token's byte length, so the
        caller can size a decode output in one call
    gpt2_encode(handle, text_addr, n_bytes, out_addr, out_cap, allow_endoftext)
        reads `n_bytes` uint8 at `text_addr`, writes the ids as int32 at
        `out_addr` and returns how many. An encoding never has more ids than
        bytes (a merge only shortens, and the 13-byte special token is one
        id), so `out_cap = n_bytes` always suffices; a count above `out_cap`
        is refused before anything is written. `n_bytes = 0` reads and
        writes nothing and returns 0.
    gpt2_decode(handle, ids_addr, n_ids, out_addr, out_cap)
        reads `n_ids` int32 at `ids_addr`, writes the bytes at `out_addr`
        and returns how many. An id outside [0, 50257) is refused BY NAME
        AND POSITION (`encoding.mojo`'s own sentence) with nothing written;
        `out_cap = n_ids * gpt2_max_token_bytes` always suffices.

THE SABOTAGE. `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` makes `gpt2_encode`
write the ids in REVERSE order. It is the Python gate's negative control: a
build with it defined must fail `python/mojolearn/tests/
test_tokenizer_surface.py`'s exact-id cases, or that test is not reading
this binary's output. `tokenizer_host_sabotage()` reads it back and
`_backend.load_host_module` refuses such a build outside the gate.
"""
from std.memory import memcpy
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined

from bindings.hostptr import i32_ptr
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from tokenizer.encoding import (
    GPT2_ENDOFTEXT,
    Gpt2Tokenizer,
    load_gpt2_tokenizer_from,
)

comptime TOKENIZER_HOST_SABOTAGE = is_defined["MOJOLEARN_TOKENIZER_HOST_SABOTAGE"]()


struct Gpt2Handle(Movable, Writable):
    """Python-owned tokenizer lifetime: the parsed rank table and Unicode
    classes, loaded once by `gpt2_load` and kept for the handle's life. No
    global slot and no retained host pointer."""

    var tok: Optional[Gpt2Tokenizer]
    var max_token_bytes: Int

    def __init__(out self):
        self.tok = Optional[Gpt2Tokenizer]()
        self.max_token_bytes = 0

    # Both spelled out: `add_type` derives whichever is missing by
    # reflection over the fields, and `Optional[Gpt2Tokenizer]` is not
    # Writable (the byte LM session does the same).
    def write_to(self, mut writer: Some[Writer]):
        writer.write("_Gpt2Handle(loaded=", Bool(self.tok), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_Gpt2Handle(loaded=", Bool(self.tok), ")")


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tokenizer host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _flag(value: PythonObject, name: String) raises -> Bool:
    var type_name = String(py=value.__class__.__name__)
    if type_name != "bool":
        raise Error(
            "tokenizer host: " + name + " must be a bool, got " + type_name
        )
    return Bool(py=value)


def _u8_ptr(addr: Int) raises -> MutPointer[UInt8, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint8 buffer address")
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def _require_loaded(loaded: Bool) raises:
    if not loaded:
        raise Error("tokenizer host: the handle holds no tokenizer; gpt2_load it")


def tokenizer_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def tokenizer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def tokenizer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "tokenizer host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_tokenizer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `tokenizer_host_detected_column` read-back, for the reason
# 8d16ce2f removed it from the forest and byte LM host bindings: the detected
# column folds to the GPU of the machine that ran the build. The comptime
# assert above is the check.


def tokenizer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary writes encode's ids in reverse order on purpose
    (-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(TOKENIZER_HOST_SABOTAGE)


def gpt2_load_binding(
    ranks_path: PythonObject, unicode_path: PythonObject
) raises -> PythonObject:
    """Parse `tokenizer/data/gpt2_ranks.tsv` and `unicode_categories.tsv`
    (or the caller's copies) once into a handle. Every refusal in
    `tokenizer/impl/ranks.mojo::load_rank_table` and
    `unicode_class.mojo::load_unicode_classes` (a rank out of order, an odd
    hex field, an empty class) raises here with the file's own sentence."""
    var rp = String(py=ranks_path)
    var up = String(py=unicode_path)
    var handle = Gpt2Handle()
    with GILReleased(Python()):
        var tok = load_gpt2_tokenizer_from(rp, up)
        var longest = len(String(GPT2_ENDOFTEXT).as_bytes())
        for i in range(tok.ranks.n_tokens()):
            if tok.ranks.length[i] > longest:
                longest = tok.ranks.length[i]
        handle.max_token_bytes = longest
        handle.tok = tok^
    return PythonObject(alloc=handle^)


def gpt2_n_vocab_binding(handle: PythonObject) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[Gpt2Handle]()
    _require_loaded(Bool(owner[].tok))
    return PythonObject(owner[].tok.value().n_vocab())


def gpt2_max_token_bytes_binding(handle: PythonObject) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[Gpt2Handle]()
    _require_loaded(Bool(owner[].tok))
    return PythonObject(owner[].max_token_bytes)


def gpt2_encode_binding(
    handle: PythonObject,
    text_addr: PythonObject,
    n_bytes: PythonObject,
    out_addr: PythonObject,
    out_cap: PythonObject,
    allow_endoftext: PythonObject,
) raises -> PythonObject:
    """`GPT2Tokenizer.encode_bytes` on the host. Returns the id count."""
    var owner = handle.downcast_value_ptr[Gpt2Handle]()
    _require_loaded(Bool(owner[].tok))
    var n = _index(n_bytes)
    var cap = _index(out_cap)
    var allow = _flag(allow_endoftext, "allow_endoftext")
    if n < 0:
        raise Error("gpt2_encode: n_bytes must be >= 0, got " + String(n))
    if cap < 0:
        raise Error("gpt2_encode: out_cap must be >= 0, got " + String(cap))
    if n == 0:
        return PythonObject(0)
    var text_address = _index(text_addr)
    var out_address = _index(out_addr)
    var count = 0
    with GILReleased(Python()):
        var src = _u8_ptr(text_address)
        var text = List[UInt8](length=n, fill=UInt8(0))
        memcpy(dest=text.unsafe_ptr(), src=src, count=n)
        # THE ONE CALL THAT COMPUTES ANYTHING.
        var ids = owner[].tok.value().encode_bytes(text, allow)
        count = len(ids)
        if count > cap:
            raise Error(
                "gpt2_encode: "
                + String(count)
                + " ids do not fit an output of "
                + String(cap)
                + "; nothing written"
            )
        var dst = i32_ptr(out_address)
        for k in range(count):
            comptime if TOKENIZER_HOST_SABOTAGE:
                dst[k] = Int32(ids[count - 1 - k])
            else:
                dst[k] = Int32(ids[k])
    return PythonObject(count)


def gpt2_decode_binding(
    handle: PythonObject,
    ids_addr: PythonObject,
    n_ids: PythonObject,
    out_addr: PythonObject,
    out_cap: PythonObject,
) raises -> PythonObject:
    """`GPT2Tokenizer.decode_bytes` on the host. Returns the byte count."""
    var owner = handle.downcast_value_ptr[Gpt2Handle]()
    _require_loaded(Bool(owner[].tok))
    var n = _index(n_ids)
    var cap = _index(out_cap)
    if n < 0:
        raise Error("gpt2_decode: n_ids must be >= 0, got " + String(n))
    if cap < 0:
        raise Error("gpt2_decode: out_cap must be >= 0, got " + String(cap))
    if n == 0:
        return PythonObject(0)
    var ids_address = _index(ids_addr)
    var out_address = _index(out_addr)
    var count = 0
    with GILReleased(Python()):
        var src = i32_ptr(ids_address)
        var ids = List[Int](capacity=n)
        for k in range(n):
            ids.append(Int(src[k]))
        # THE ONE CALL THAT COMPUTES ANYTHING; an id outside [0, 50257) is
        # refused here by name and position, nothing written.
        var out = owner[].tok.value().decode_bytes(ids)
        count = len(out)
        if count > cap:
            raise Error(
                "gpt2_decode: "
                + String(count)
                + " bytes do not fit an output of "
                + String(cap)
                + "; nothing written"
            )
        var dst = _u8_ptr(out_address)
        if count > 0:
            memcpy(dest=dst, src=out.unsafe_ptr(), count=count)
    return PythonObject(count)


@export
def PyInit__mojolearn_tokenizer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tokenizer_host")
        module.def_function[tokenizer_host_numeric_mode_binding]("tokenizer_host_numeric_mode")
        module.def_function[tokenizer_host_vendor_binding]("tokenizer_host_vendor")
        module.def_function[tokenizer_host_column_binding]("tokenizer_host_column")
        module.def_function[tokenizer_host_sabotage_binding]("tokenizer_host_sabotage")
        _ = module.add_type[Gpt2Handle]("_Gpt2Handle")
        module.def_function[gpt2_load_binding]("gpt2_load")
        module.def_function[gpt2_n_vocab_binding]("gpt2_n_vocab")
        module.def_function[gpt2_max_token_bytes_binding]("gpt2_max_token_bytes")
        module.def_function[gpt2_encode_binding]("gpt2_encode")
        module.def_function[gpt2_decode_binding]("gpt2_decode")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_host: ", error))
