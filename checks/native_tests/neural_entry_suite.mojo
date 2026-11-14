# SPDX-License-Identifier: Apache-2.0
"""The neural-blocks entry table's six families, reached natively (lane
calltable-neural-blocks, 2026-11-11; core/neural_entry_table.mojo).

Ordered preconditions first (they share the one process table): a caller
refuses before publication; then the CPU publications the six host images
make at module init (`_mojolearn_neural_host`, `_mojolearn_transformer_host`,
`_mojolearn_mamba_host`, `_mojolearn_x_sequence_host`, `_mojolearn_embedding_host`,
`_mojolearn_embedding_infer_host`), repeated as a second
module init repeats them; a mismatched tier or an invalid vendor key refuses.
Then one case per operation the neural block and SmallMLPTrainer callers
make through each published entry, each checked against the entry's own
refusal text for an inadmissible descriptor (which proves the published
pointer is that image's dispatch and that the words reach it unchanged). No
Python participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column one after another.
Run in a fresh process (the first check needs an unpublished table).
"""
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import entry_ref_same
from core.host_parallel import host_parallelize
from core.neural_entry_table import (
    NeuralEntryKey, neural_entry_call, neural_ref, NEURAL_TABLE_BLOCK_FORWARD, NEURAL_TABLE_BLOCK_TRAIN,
    NEURAL_TABLE_MAMBA_TRAIN, NEURAL_TABLE_MLP_PROGRAM, NEURAL_BLOCK_FORWARD_FRESH,
    NEURAL_BLOCK_SESSION_FORWARD_FRESH, NEURAL_BLOCK_BACKWARD, NEURAL_MAMBA1_BACKWARD,
    NEURAL_MAMBA2_BACKWARD, NEURAL_MAMBA3_BACKWARD, NEURAL_MLP_PROGRAM_OPERATIONS,
    NEURAL_TABLE_EMBEDDING, NEURAL_TABLE_EMBEDDING_INFER, EMB_OP_FORWARD, EMB_OP_BACKWARD,
    EMB_OP_TENSOR_ALLOC,
)
from autograd.small_mlp_program import SMALL_MLP_PROGRAM_OPEN, SMALL_MLP_PROGRAM_CLOSE
from core.neural_entry_neural_host import publish_neural_block_neural_host
from transformer.host.neural_entry_host import publish_neural_transformer_host
from mamba.host.neural_entry_host import publish_neural_mamba_host
from sequence.small_mlp_entry_host import publish_neural_small_mlp_host
# The two embedding images publish from their PUBLISH modules: their entries
# release the GIL around the host kernels, so the publication cannot move to a
# Python-free module without moving that release to the caller. A native suite
# may import a PUBLISH module for this, never a door, unpack, pack or py_*
# module (tools/python_free_dirs.sh, fourth check).
from bindings._mojolearn_embedding_host import publish_neural_embedding_host
from bindings._mojolearn_embedding_infer_host import publish_neural_embedding_infer_host

comptime NEURAL_ENTRY_CASES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> NeuralEntryKey:
    return NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def _expect_refusal[family: Int](operation: Int, addresses: List[Int], ints: List[Int],
    floats: List[Float64], text: String) raises:
    try:
        _ = neural_entry_call(neural_ref[family](_key()), operation, addresses, ints, floats)
    except problem:
        _refuses(problem, text)
        return
    raise Error("an inadmissible descriptor was accepted: " + text)


def precondition_unpublished() raises:
    var none = List[Int]()
    try:
        _ = neural_entry_call(neural_ref[NEURAL_TABLE_BLOCK_TRAIN](_key()), NEURAL_BLOCK_BACKWARD, none, none, List[Float64]())
    except problem:
        _refuses(problem, "provider is not published")
        return
    raise Error("an unpublished neural table did not refuse")


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: each family resolves once into a ref that
    a connection owner reuses (the same pointer and layout word every time),
    and a mismatched key refuses at resolution, never at a call."""
    require(entry_ref_same(neural_ref[NEURAL_TABLE_BLOCK_TRAIN](_key()), neural_ref[NEURAL_TABLE_BLOCK_TRAIN](_key())),
            "neural block train: a second resolution returned another entry")
    require(entry_ref_same(neural_ref[NEURAL_TABLE_MAMBA_TRAIN](_key()), neural_ref[NEURAL_TABLE_MAMBA_TRAIN](_key())),
            "neural mamba train: a second resolution returned another entry")
    try:
        _ = neural_ref[NEURAL_TABLE_BLOCK_TRAIN](NeuralEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("neural: a mismatched key did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))


def preconditions_published() raises:
    ref_resolution()
    var none = List[Int]()
    try:
        _ = neural_entry_call(neural_ref[NEURAL_TABLE_MAMBA_TRAIN](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 2)),
            NEURAL_MAMBA1_BACKWARD, none, none, List[Float64]())
        raise Error("an unpublished vendor key did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    try:
        _ = neural_entry_call(neural_ref[NEURAL_TABLE_BLOCK_FORWARD](NeuralEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)),
            NEURAL_BLOCK_FORWARD_FRESH, none, none, List[Float64]())
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = neural_entry_call(neural_ref[NEURAL_TABLE_MLP_PROGRAM](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 4)),
            SMALL_MLP_PROGRAM_CLOSE, none, none, List[Float64]())
        raise Error("an invalid vendor key did not refuse")
    except problem:
        _refuses(problem, "invalid tier/vendor key")


def case(unit: Int) raises:
    var none = List[Int]()
    var no_floats = List[Float64]()
    if unit == 0:
        # forward image (neural host), op 4: the fresh forward's arity admission
        _expect_refusal[NEURAL_TABLE_BLOCK_FORWARD](NEURAL_BLOCK_FORWARD_FRESH, none, none, no_floats,
            "CPU Transformer inference: invalid descriptor arity")
    elif unit == 1:
        # forward image (neural host): the CPU image serves no session forward
        _expect_refusal[NEURAL_TABLE_BLOCK_FORWARD](NEURAL_BLOCK_SESSION_FORWARD_FRESH, none, none, no_floats,
            "CPU Transformer inference: unavailable operation")
    elif unit == 2:
        # training image (transformer host), op 5: the backward's arity admission
        _expect_refusal[NEURAL_TABLE_BLOCK_TRAIN](NEURAL_BLOCK_BACKWARD, none, [1, 1, 1, 1, 1, 1, 1, 1], no_floats,
            "transformer backward: expected 21 addresses and eight scalars")
    elif unit == 3:
        # training image (transformer host), op 4: the fresh forward's arity admission
        _expect_refusal[NEURAL_TABLE_BLOCK_TRAIN](NEURAL_BLOCK_FORWARD_FRESH, none, none, no_floats,
            "transformer_forward_fresh: invalid descriptor arity")
    elif unit == 4:
        # Mamba training image, op 2 (Mamba1 backward)
        _expect_refusal[NEURAL_TABLE_MAMBA_TRAIN](NEURAL_MAMBA1_BACKWARD, none, [1, 1, 1], no_floats,
            "Mamba host backward: invalid descriptor arity")
    elif unit == 5:
        # Mamba training image, op 3 (Mamba2 backward): 21 null addresses
        # pass the arity admission (with the two dt_limit floats) and reach
        # the address admission
        var zeros = List[Int](length=21, fill=0)
        var limit: List[Float64] = [0.0, 1.0]
        _expect_refusal[NEURAL_TABLE_MAMBA_TRAIN](NEURAL_MAMBA2_BACKWARD, zeros, [1, 1, 1], limit,
            "Mamba host backward: null buffer address")
    elif unit == 6:
        # Mamba training image, op 4 (Mamba3 backward): floats are refused
        var limit: List[Float64] = [0.0, 1.0]
        _expect_refusal[NEURAL_TABLE_MAMBA_TRAIN](NEURAL_MAMBA3_BACKWARD, List[Int](length=21, fill=0),
            [1, 1, 1], limit, "Mamba host backward: invalid descriptor arity")
    elif unit == 7:
        # program image, OPEN: the program's own arity admission
        _expect_refusal[NEURAL_TABLE_MLP_PROGRAM](SMALL_MLP_PROGRAM_OPEN, none, none, no_floats,
            "small MLP program open takes 6 addresses, 1 integer and 5 floats")
    elif unit == 8:
        _program_bounds()
    elif unit == 9:
        # Embedding training host, FORWARD: the forward's arity admission
        _expect_refusal[NEURAL_TABLE_EMBEDDING](EMB_OP_FORWARD, none, [3, 2, 1], no_floats,
            "embedding_forward: addrs must contain 3 addresses")
    elif unit == 10:
        # Embedding training host, BACKWARD: the backward's arity admission;
        # the host image serves no resident tensors
        _expect_refusal[NEURAL_TABLE_EMBEDDING](EMB_OP_BACKWARD, [0, 0, 0], [3, 2], no_floats,
            "embedding_backward: params must contain 5 or 6 values")
        _expect_refusal[NEURAL_TABLE_EMBEDDING](EMB_OP_TENSOR_ALLOC, none, [1, 0], no_floats,
            "embedding host entry: unavailable operation")
    elif unit == 11:
        # Embedding inference host: FORWARD admission; no backward
        _expect_refusal[NEURAL_TABLE_EMBEDDING_INFER](EMB_OP_FORWARD, [0, 0], [3, 2, 1], no_floats,
            "embedding_forward: addrs must contain 3 addresses")
        _expect_refusal[NEURAL_TABLE_EMBEDDING_INFER](EMB_OP_BACKWARD, none, none, no_floats,
            "embedding host entry: unavailable operation")
    else:
        raise Error("unknown neural entry case")


def _program_bounds() raises:
    var none = List[Int]()
    var no_floats = List[Float64]()
    # program image: an operation past the table's count, then CLOSE without
    # a handle
    _expect_refusal[NEURAL_TABLE_MLP_PROGRAM](NEURAL_MLP_PROGRAM_OPERATIONS, none, none, no_floats,
        "small MLP program: unknown operation")
    _expect_refusal[NEURAL_TABLE_MLP_PROGRAM](SMALL_MLP_PROGRAM_CLOSE, none, none, no_floats,
        "small MLP program: the handle comes first")


def run_checks() raises:
    precondition_unpublished()
    # The publications the six CPU images make at module init; a second
    # init publishes the same pointers.
    publish_neural_block_neural_host()
    publish_neural_transformer_host()
    publish_neural_mamba_host()
    publish_neural_small_mlp_host()
    publish_neural_embedding_host()
    publish_neural_embedding_infer_host()
    publish_neural_block_neural_host()
    publish_neural_transformer_host()
    publish_neural_mamba_host()
    publish_neural_small_mlp_host()
    publish_neural_embedding_host()
    publish_neural_embedding_infer_host()
    preconditions_published()
    var failures = List[Int](length=NEURAL_ENTRY_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL neural entry case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, NEURAL_ENTRY_CASES)
    else:
        for unit in range(NEURAL_ENTRY_CASES):  # driver-loop: device cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "neural entry cases failed: " + String(failed))
    print("PASS neural entry", NEURAL_ENTRY_CASES)
