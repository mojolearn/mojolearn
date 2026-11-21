# SPDX-License-Identifier: Apache-2.0
"""The parallel MLP parameter names and shapes, in their published order (RUN)."""
from core.entry_table import EntryRef, entry_ref_call
from core.samba_entry_table import SAMBA_PRIM_ACCUMULATE, SAMBA_PRIM_OPTIMIZER_STEP


def _mlp_names() -> List[String]:
    var result = List[String]()
    result.append("weight1")
    result.append("bias1")
    result.append("weight2")
    result.append("bias2")
    return result^


def _mlp_dims(index: Int) -> List[Int]:
    var result = List[Int]()
    if index == 0:
        result.append(16)
        result.append(8)
    elif index == 1:
        result.append(16)
    elif index == 2:
        result.append(3)
        result.append(16)
    else:
        result.append(3)
    return result^


# ---------------------------------------------------------------- typed RUN
# The parallel MLP owner's rules and kernel sequences over typed values
# (lane dw-parallel-arena, 2026-11-21). `bindings/parallel_mlp_native_door.mojo`
# unpacks the Python state into these arguments and packs the results; no
# Python object reaches this file. The door releases the GIL around the two
# kernel sequences.

comptime MLP_PARAMETERS = 195    # 16*8 + 16 + 3*16 + 3
comptime MLP_FEATURES = 8
comptime MLP_MAX_BATCH = 256
comptime MLP_MAX_STEP = 2147483647
comptime MLP_INDEX_LIMIT = 2147483647


def mlp_schema_admit(schema: String, mode: String, architecture: Bool, order: Bool) raises:
    """`architecture` / `order`: the state's architecture and parameter
    order equal the fixed ones (compared at UNPACK)."""
    if schema != "mojolearn.small-mlp-trainer.v1" or mode != "identical" or not architecture or not order:
        raise Error("SmallMLPTrainer state schema/architecture/mode/order mismatch")


def mlp_config_admit(lr: Float64, beta1: Float64, beta2: Float64, eps: Float64, weight_decay: Float64) raises:
    if lr <= 0 or eps <= 0 or weight_decay < 0 or beta1 < 0 or beta1 >= 1 or beta2 < 0 or beta2 >= 1:
        raise Error("SmallMLPTrainer optimizer configuration outside admitted range")


def mlp_optimizer_kind_admit(kind: String) raises:
    if kind != "AdamW":
        raise Error("SmallMLPTrainer state optimizer mismatch")


def mlp_batch_admit(rank: Int, rows: Int, columns: Int) raises:
    """`rows` / `columns` are read only for a rank-2 batch."""
    if rank != 2 or columns != MLP_FEATURES or rows < 1 or rows > MLP_MAX_BATCH:
        raise Error("SmallMLPTrainer X must have shape (batch, 8), batch in [1,256]")


def mlp_targets_admit(rank: Int, count: Int, rows: Int, format: String) raises -> Bool:
    """Admits a rank-1 native integer target block of `rows` items; returns
    True when its format is a signed one."""
    if rank != 1 or count != rows:
        raise Error("SmallMLPTrainer targets must match batch rows")
    if format == "b" or format == "h" or format == "i" or format == "l" or format == "q":
        return True
    if format == "B" or format == "H" or format == "I" or format == "L" or format == "Q":
        return False
    raise Error("TypeError: SmallMLPTrainer targets must be a native integer array")


def mlp_registry_extent(axes: List[Int], total: Int) raises -> Int:
    """Elements of one gradient tensor of shape `axes`, admitted against the
    kernels' signed 32-bit indexing together with the `total` elements of the
    tensors before it."""
    var n = 1
    for axis in range(len(axes)):  # small-loop(gradient shape dimension words)
        var dimension = axes[axis]
        if dimension < 0 or (dimension != 0 and n > MLP_INDEX_LIMIT // dimension):
            raise Error("gradient registry exceeds signed 32-bit kernel indexing")
        n *= dimension
    if total > MLP_INDEX_LIMIT - n:
        raise Error("gradient registry exceeds signed 32-bit kernel indexing")
    return n


def mlp_ordered_fold(primitives: EntryRef, base: Int, count: Int, shards: Int) raises:
    """The ordered pair fold over one [shards, count] float32 block at
    `base`, in place: step k reads slots k-1 (the running total) and k as one
    [2, count] block and writes the pair's sum over slot k; the folded total
    is slot shards-1. Primitive op 9 (`accumulate`: addresses [out, parts],
    ints [n, a, t_tokens]). Serial by definition: step k reads step k-1's
    sum, in the fixed ascending shard order; each step is one device launch."""
    for shard in range(1, shards):  # driver-loop: over fixed ascending logical gradient shard order
        var addresses: Array[Int, 2] = [base + 4 * shard * count, base + 4 * (shard - 1) * count]
        var ints: Array[Int, 3] = [count, 2, -1]
        var floats: Array[Float64, 0] = []
        _ = entry_ref_call(primitives, SAMBA_PRIM_ACCUMULATE, addresses, ints, floats)


def mlp_step_admit(step: Int, gradients: Int) raises:
    if step == MLP_MAX_STEP:
        raise Error("SmallMLPTrainer step exceeds its supported range")
    if gradients != 4:
        raise Error("SmallMLPTrainer requires exactly four parameter gradients")


def mlp_optimizer_step(primitives: EntryRef, addresses: Array[Int, 7], step: Int, lr: Float64, beta1: Float64,
                       beta2: Float64, eps: Float64, weight_decay: Float64) raises -> Int:
    """One AdamW step over the packed parameters. Primitive op 25
    (`optimizer_step`): addresses [params, grads, m, v, offsets, flags,
    info], ints [J, kind, t, nesterov, maximize], floats [lr, beta1, beta2,
    eps, weight_decay, momentum, dampening, max_norm] (momentum, dampening,
    max_norm and maximize 0); the callee narrows Float64 to Float32. Returns
    the parameter words written."""
    var ints: Array[Int, 5] = [4, 2, step + 1, 0, 0]
    var floats: Array[Float64, 8] = [lr, beta1, beta2, eps, weight_decay, 0.0, 0.0, 0.0]
    return entry_ref_call(primitives, SAMBA_PRIM_OPTIMIZER_STEP, addresses, ints, floats)


def mlp_update_admit(written: Int) raises:
    if written != MLP_PARAMETERS:
        raise Error("RuntimeError: SmallMLPTrainer update returned incomplete state")
