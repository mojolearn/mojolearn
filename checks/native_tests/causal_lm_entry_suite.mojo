# SPDX-License-Identifier: Apache-2.0
"""The CausalLM entry table's four entries, reached natively (lane
calltable-causal-lm, 2026-11-11).

Ordered preconditions first (they share the one process table): a caller
refuses before publication; then the publications `_mojolearn_neural_host`
and `_mojolearn_linalg_host` make at module init (`publish_causal_neural_host`,
`publish_causal_linalg_host`), repeated as a second module init repeats them;
a mismatched tier or vendor key refuses. Then one case per operation the
CausalLM runtime calls through each published entry (primitives: embedding,
RMS norm, linear; linalg: bf16, int8 and int15 round trips and the int15
gemm; block and Mamba state: their arity refusals, which prove the published
pointer is the CPU image's dispatch), each with literal expected values or
the entry's own refusal text. No Python participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column one after another.
Run in a fresh process (the first check needs an unpublished table).
"""
from std.math import sqrt
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import entry_ref_same
from core.host_parallel import host_parallelize
from core.causal_lm_entry_table import (
    CausalEntryKey, causal_entry_call, causal_ref, CAUSAL_TABLE_LINALG, CAUSAL_TABLE_PRIMITIVES, CAUSAL_TABLE_BLOCK,
    CAUSAL_TABLE_MAMBA, CAUSAL_LIN_TO_BF16, CAUSAL_LIN_FROM_BF16, CAUSAL_LIN_QUANTIZE_INT8,
    CAUSAL_LIN_DEQUANTIZE_INT8, CAUSAL_LIN_QUANTIZE_INT15, CAUSAL_LIN_GEMM_INT15, CAUSAL_PRIM_EMBED_FORWARD,
    CAUSAL_PRIM_NORM_FORWARD, CAUSAL_PRIM_LINEAR_FORWARD, CAUSAL_BLOCK_HOST_FORWARD,
    CAUSAL_BLOCK_HOST_FORWARD_INT15, CAUSAL_MAMBA1_FORWARD, CAUSAL_MAMBA2_DECODE,
)
from core.causal_entry_neural_host import publish_causal_neural_host
from gemm.host.causal_entry_host import publish_causal_linalg_host
from core.state_registry import StateHandle
from core.causal_lm_entry_table_sessions import (
    causal_sessions_ref, causal_sessions_call, causal_session_handle_word, causal_session_handle_of, CAUSAL_SESSIONS_TRANSFORMER,
    CAUSAL_SESSIONS_MAMBA, CAUSAL_SESSION_LM_CREATE, CAUSAL_SESSION_PROGRAM_READ,
)

comptime CAUSAL_ENTRY_CASES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> CausalEntryKey:
    return CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _i32(mut l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def _i8(mut l: List[Int8]) -> Int:
    return Int(l.unsafe_ptr())


def _u16(mut l: List[UInt16]) -> Int:
    return Int(l.unsafe_ptr())


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _close(got: Float32, want: Float64, tolerance: Float64) -> Bool:
    return abs(Float64(got) - want) <= tolerance


def precondition_unpublished() raises:
    var none = List[Int]()
    try:
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_PRIMITIVES](_key()), CAUSAL_PRIM_EMBED_FORWARD, none, none, List[Float64]())
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished CausalLM table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: each family resolves once into a ref that
    a connection owner reuses (the same pointer and layout word every time),
    and a mismatched key refuses at resolution, never at a call."""
    require(entry_ref_same(causal_ref[CAUSAL_TABLE_PRIMITIVES](_key()), causal_ref[CAUSAL_TABLE_PRIMITIVES](_key())),
            "CausalLM primitives: a second resolution returned another entry")
    require(entry_ref_same(causal_ref[CAUSAL_TABLE_LINALG](_key()), causal_ref[CAUSAL_TABLE_LINALG](_key())),
            "CausalLM linalg: a second resolution returned another entry")
    try:
        _ = causal_ref[CAUSAL_TABLE_PRIMITIVES](CausalEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("CausalLM: a mismatched key did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))


def preconditions_published() raises:
    ref_resolution()
    var none = List[Int]()
    try:
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 2)),
            CAUSAL_LIN_TO_BF16, none, none, List[Float64]())
        raise Error("an unpublished vendor key did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    try:
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_BLOCK](CausalEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)),
            CAUSAL_BLOCK_HOST_FORWARD, none, none, List[Float64]())
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_MAMBA](CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 4)),
            CAUSAL_MAMBA1_FORWARD, none, none, List[Float64]())
        raise Error("an invalid vendor key did not refuse")
    except problem:
        _refuses(problem, "invalid tier/vendor key")


def case(unit: Int) raises:
    var k = _key()
    var no_floats = List[Float64]()
    if unit == 0:
        # primitives op 0: y[n, width] = w[ids[n], :]
        var y: List[Float32] = [9.0, 9.0, 9.0, 9.0]
        var w: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        var ids: List[Int32] = [2, 0]
        var n = causal_entry_call(causal_ref[CAUSAL_TABLE_PRIMITIVES](k), CAUSAL_PRIM_EMBED_FORWARD,
            [_f32(y), _f32(w), _i32(ids)], [2, 3, 2], no_floats)
        require(n == 4 and y[0] == 4.0 and y[1] == 5.0 and y[2] == 0.0 and y[3] == 1.0, "embedding forward")
        _ = w
        _ = ids
    elif unit == 1:
        # primitives op 2: y = x * w / sqrt(mean(x^2) + eps)
        var y: List[Float32] = [0.0, 0.0]
        var x: List[Float32] = [3.0, 4.0]
        var w: List[Float32] = [1.0, 2.0]
        var eps: List[Float64] = [1.0e-6]
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_PRIMITIVES](k), CAUSAL_PRIM_NORM_FORWARD,
            [_f32(y), _f32(x), _f32(w)], [1, 2], eps)
        var r = sqrt(12.5 + 1.0e-6)
        require(_close(y[0], 3.0 / r, 1.0e-5) and _close(y[1], 8.0 / r, 1.0e-5), "rms norm forward")
        _ = x
        _ = w
    elif unit == 2:
        # primitives op 4: C[m, n] = A[m, k] . W[n, k]^T
        var c: List[Float32] = [0.0, 0.0]
        var a: List[Float32] = [1.0, 2.0]
        var w: List[Float32] = [3.0, 4.0, 5.0, 6.0]
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_PRIMITIVES](k), CAUSAL_PRIM_LINEAR_FORWARD,
            [_f32(c), _f32(a), _f32(w)], [1, 2, 2], no_floats)
        require(c[0] == 11.0 and c[1] == 17.0, "linear forward")
        _ = a
        _ = w
    elif unit == 3:
        # linalg ops 8 and 9: float32 -> bf16 bits -> float32, exact here
        var x: List[Float32] = [1.0, -2.5]
        var bits: List[UInt16] = [0, 0]
        var back: List[Float32] = [0.0, 0.0]
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_TO_BF16, [_u16(bits), _f32(x)], [2], no_floats)
        require(bits[0] == 0x3F80 and bits[1] == 0xC020, "to_bf16 bits")
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_FROM_BF16, [_f32(back), _u16(bits)], [2], no_floats)
        require(back[0] == 1.0 and back[1] == -2.5, "from_bf16")
        _ = x
    elif unit == 4:
        # linalg ops 3 and 4: small integers survive the int8 row codes
        var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var codes: List[Int8] = [0, 0, 0, 0]
        var exponents: List[Int32] = [0]
        var back: List[Float32] = [0.0, 0.0, 0.0, 0.0]
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_QUANTIZE_INT8,
            [_i8(codes), _i32(exponents), _f32(x)], [1, 4], no_floats)
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_DEQUANTIZE_INT8,
            [_f32(back), _i8(codes), _i32(exponents)], [1, 4], no_floats)
        for i in range(4):  # small-loop(four fixture values)
            require(back[i] == x[i], "int8 round trip at " + String(i))
    elif unit == 5:
        # linalg ops 6 and 5: [1, 2] . [3, 4] = 11 under int15
        var a: List[Float32] = [1.0, 2.0]
        var b: List[Float32] = [3.0, 4.0]
        var ahi: List[Int8] = [0, 0]
        var alo: List[Int8] = [0, 0]
        var aex: List[Int32] = [0]
        var bhi: List[Int8] = [0, 0]
        var blo: List[Int8] = [0, 0]
        var bex: List[Int32] = [0]
        var c: List[Float32] = [0.0]
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_QUANTIZE_INT15,
            [_i8(ahi), _i8(alo), _i32(aex), _f32(a)], [1, 2], no_floats)
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_QUANTIZE_INT15,
            [_i8(bhi), _i8(blo), _i32(bex), _f32(b)], [1, 2], no_floats)
        _ = causal_entry_call(causal_ref[CAUSAL_TABLE_LINALG](k), CAUSAL_LIN_GEMM_INT15,
            [_f32(c), _i8(ahi), _i8(alo), _i32(aex), _i8(bhi), _i8(blo), _i32(bex)], [1, 1, 2], no_floats)
        require(c[0] == 11.0, "int15 gemm")
        _ = a
        _ = b
    elif unit == 6:
        # block op 33 reaches the CPU image's Transformer dispatch
        var none = List[Int]()
        try:
            _ = causal_entry_call(causal_ref[CAUSAL_TABLE_BLOCK](k), CAUSAL_BLOCK_HOST_FORWARD, none, none, no_floats)
            raise Error("block forward accepted an empty descriptor")
        except problem:
            _refuses(problem, "CPU Transformer inference: invalid descriptor arity")
    elif unit == 7:
        # block op 34 (fixed15_v1) requires both option tails
        var none = List[Int]()
        try:
            _ = causal_entry_call(causal_ref[CAUSAL_TABLE_BLOCK](k), CAUSAL_BLOCK_HOST_FORWARD_INT15, none, none, no_floats)
            raise Error("fixed15 block forward accepted an empty descriptor")
        except problem:
            _refuses(problem, "CPU Transformer inference: invalid descriptor arity")
    else:
        # Mamba state op 3 reaches the CPU image's Mamba dispatch
        var none = List[Int]()
        try:
            _ = causal_entry_call(causal_ref[CAUSAL_TABLE_MAMBA](k), CAUSAL_MAMBA2_DECODE, none, none, no_floats)
            raise Error("Mamba-2 decode accepted an empty descriptor")
        except problem:
            _refuses(problem, "CPU Mamba state: invalid descriptor arity")


def sessions_table_contract() raises:
    # The session table (core/causal_lm_entry_table_sessions.mojo) on any
    # column: the handle word round trip, the CPU-key refusal (CPU images own
    # no resident session) and an unpublished GPU key's refusal. Its
    # per-operation cases need the GPU images' publications:
    # checks/native_tests/causal_lm_session_gpu_suite.mojo.
    var word = causal_session_handle_word(StateHandle(5, UInt64(3)))
    var back = causal_session_handle_of(word)
    require(word > 0 and back.index == 5 and back.generation == 3, "session handle word round trip")
    try:
        _ = causal_session_handle_of(0)
        raise Error("a zero handle word was admitted")
    except problem:
        _refuses(problem, "no session handle")
    var none = List[Int]()
    try:
        _ = causal_sessions_call(causal_sessions_ref(CAUSAL_SESSIONS_TRANSFORMER, CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)),
            CAUSAL_SESSION_LM_CREATE, none, none, List[Float64]())
        raise Error("a CPU session key was admitted")
    except problem:
        _refuses(problem, "GPU images only")
    comptime if TARGET_COLUMN == COLUMN_CPU:
        try:
            _ = causal_sessions_call(causal_sessions_ref(CAUSAL_SESSIONS_MAMBA, CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 2)),
                CAUSAL_SESSION_PROGRAM_READ, none, none, List[Float64]())
            raise Error("an unpublished session table was admitted")
        except problem:
            _refuses(problem, "provider is not published")


def run_checks() raises:
    precondition_unpublished()
    sessions_table_contract()
    # The publications _mojolearn_neural_host and _mojolearn_linalg_host make
    # at module init; a second init publishes the same pointers.
    publish_causal_neural_host()
    publish_causal_linalg_host()
    publish_causal_neural_host()
    publish_causal_linalg_host()
    preconditions_published()
    var failures = List[Int](length=CAUSAL_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL causal lm entry case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, CAUSAL_ENTRY_CASES)
    else:
        for unit in range(CAUSAL_ENTRY_CASES):  # driver-loop: device cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "causal lm entry cases failed: " + String(failed))
    print("PASS causal lm entry", CAUSAL_ENTRY_CASES)
