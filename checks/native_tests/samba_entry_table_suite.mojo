# SPDX-License-Identifier: Apache-2.0
"""The Samba entry table's published primitive entries, reached natively.

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the publications `_mojolearn_training_host`
and `_mojolearn_neural_host` make at module init, then a tier-key, a vendor
and an operation mismatch refuse. Then one case per operation the Samba
callers convert (`bindings/samba_state_native.mojo::samba_primitive`):
embedding, RMS norm and linear forward on both host images, ce_loss and the
per-call optimizer step on the training host image, with literal expected
values. No Python participates. The device image's publication
(`core/samba_entry_table_training.mojo`) has the same shape; it is reached
on a device column by the installed-wheel Samba tests.

The cases are independent units and run across the native host worker pool.
Run in a fresh process (the first check needs an unpublished table).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.entry_table import entry_ref_call, entry_ref_same
from core.samba_entry_table import (
    SambaEntryKey, samba_table_entry, samba_ref, SAMBA_TABLE_TRAINING, SAMBA_TABLE_NEURAL_HOST,
    SAMBA_PRIM_EMBEDDING_FORWARD, SAMBA_PRIM_RMS_NORM_FORWARD, SAMBA_PRIM_LINEAR_FORWARD,
    SAMBA_PRIM_CE_LOSS, SAMBA_PRIM_OPTIMIZER_STEP,
)
from core.samba_entry_table_training_host import publish_samba_training_host_primitives
from core.samba_entry_table_neural_host import publish_samba_neural_host_primitives

comptime SAMBA_ENTRY_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> SambaEntryKey:
    return SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _call[family: Int](key: SambaEntryKey, op: Int, addresses: List[Int], ints: List[Int],
                       floats: List[Float64]) raises -> Int:
    return entry_ref_call(samba_ref[family](key), op, addresses, ints, floats)


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _i32(mut l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def precondition_unpublished() raises:
    try:
        _ = samba_table_entry[SAMBA_TABLE_TRAINING](_key())
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished samba table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: each family resolves once into a ref that
    a connection owner reuses (the same pointer and layout word every time),
    and a mismatched key refuses at resolution, never at a call."""
    require(entry_ref_same(samba_ref[SAMBA_TABLE_TRAINING](_key()), samba_ref[SAMBA_TABLE_TRAINING](_key())),
            "samba training: a second resolution returned another entry")
    require(entry_ref_same(samba_ref[SAMBA_TABLE_NEURAL_HOST](SambaEntryKey(1, 0)),
                           samba_ref[SAMBA_TABLE_NEURAL_HOST](SambaEntryKey(1, 0))),
            "samba neural host: a second resolution returned another entry")
    try:
        _ = samba_ref[SAMBA_TABLE_TRAINING](SambaEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("samba: a mismatched key did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))


def preconditions_published() raises:
    try:
        _ = samba_table_entry[SAMBA_TABLE_TRAINING](SambaEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = samba_table_entry[SAMBA_TABLE_NEURAL_HOST](SambaEntryKey(Int(GLOBAL_NUMERIC_MODE), 2))
        raise Error("a device key on the host-only family did not refuse")
    except problem:
        _refuses(problem, "host only")
    try:
        var none = List[Int]()
        _ = _call[SAMBA_TABLE_TRAINING](_key(), 7, none, none, List[Float64]())
        raise Error("an unavailable operation did not refuse")
    except problem:
        _refuses(problem, "unavailable operation")
    # A repeated module init publishes the same pointers and is accepted.
    publish_samba_training_host_primitives()
    publish_samba_neural_host_primitives()


def _embedding[family: Int]() raises:
    var w: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var ids: List[Int32] = [1, 0]
    var y: List[Float32] = [0.0, 0.0, 0.0, 0.0]
    _ = _call[family](_key(), SAMBA_PRIM_EMBEDDING_FORWARD, [_f32(y), _f32(w), _i32(ids)], [2, 2, 2], [])
    require(y[0] == 3.0 and y[1] == 4.0 and y[2] == 1.0 and y[3] == 2.0, "embedding_forward")
    _ = w
    _ = ids


def _rms_norm[family: Int]() raises:
    var x: List[Float32] = [2.0, 2.0]
    var w: List[Float32] = [1.0, 3.0]
    var y: List[Float32] = [0.0, 0.0]
    _ = _call[family](_key(), SAMBA_PRIM_RMS_NORM_FORWARD, [_f32(y), _f32(x), _f32(w)], [1, 2], [0.0])
    require(y[0] == 1.0 and y[1] == 3.0, "rms_norm_forward")
    _ = x
    _ = w


def _linear[family: Int]() raises:
    var a: List[Float32] = [1.0, 2.0]
    var w: List[Float32] = [3.0, 4.0, 5.0, 6.0]
    var c: List[Float32] = [0.0, 0.0]
    _ = _call[family](_key(), SAMBA_PRIM_LINEAR_FORWARD, [_f32(c), _f32(a), _f32(w)], [1, 2, 2], [])
    require(c[0] == 11.0 and c[1] == 17.0, "linear_forward")
    _ = a
    _ = w


def case(unit: Int) raises:
    if unit == 0:
        _embedding[SAMBA_TABLE_NEURAL_HOST]()
    elif unit == 1:
        _rms_norm[SAMBA_TABLE_NEURAL_HOST]()
    elif unit == 2:
        _linear[SAMBA_TABLE_NEURAL_HOST]()
    elif unit == 3:
        _embedding[SAMBA_TABLE_TRAINING]()
    elif unit == 4:
        _rms_norm[SAMBA_TABLE_TRAINING]()
    elif unit == 5:
        _linear[SAMBA_TABLE_TRAINING]()
    elif unit == 6:
        var loss: List[Float32] = [0.0]
        var rows: List[Float32] = [0.0]
        var unused: List[Float32] = [0.0]
        var logits: List[Float32] = [0.0, 0.0]
        var targets: List[Int32] = [0]
        var count = _call[SAMBA_TABLE_TRAINING](_key(), SAMBA_PRIM_CE_LOSS,
            [_f32(loss), _f32(rows), _f32(unused), _f32(logits), _i32(targets)], [1, 2, -100, 2, 0, 0], [0.0])
        require(count == 1 and abs(loss[0] - Float32(0.6931472)) < Float32(1e-6), "ce_loss")
        _ = rows
        _ = unused
        _ = logits
        _ = targets
    else:
        # SGD, lr 0.5, no momentum, clip off: 1.0 - 0.5 * 2.0.
        var param: List[Float32] = [1.0]
        var grad: List[Float32] = [2.0]
        var m: List[Float32] = [0.0]
        var v: List[Float32] = [0.0]
        var offsets: List[Int32] = [0, 1]
        var init: List[Int32] = [0]
        var info: List[Float32] = [0.0, 0.0, 0.0]
        var floats: List[Float64] = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        var n = _call[SAMBA_TABLE_TRAINING](_key(), SAMBA_PRIM_OPTIMIZER_STEP,
            [_f32(param), _f32(grad), _f32(m), _f32(v), _i32(offsets), _i32(init), _f32(info)],
            [1, 0, 1, 0, 0], floats)
        require(n == 1 and param[0] == 0.0, "optimizer_step")
        _ = grad
        _ = m
        _ = v
        _ = offsets
        _ = init
        _ = info


def run_checks() raises:
    precondition_unpublished()
    # The publications _mojolearn_training_host and _mojolearn_neural_host make at module init.
    publish_samba_training_host_primitives()
    publish_samba_neural_host_primitives()
    preconditions_published()
    ref_resolution()
    var failures = List[Int](length=SAMBA_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL samba entry table case", unit, String(problem))
    host_parallelize(task, SAMBA_ENTRY_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "samba entry table cases failed: " + String(failed))
    print("PASS samba entry table", SAMBA_ENTRY_CASES)
