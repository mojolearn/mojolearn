# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba step door (bindings/samba_step_native_door.mojo,
door split 2026-11-21): the step's admission checks in the door's words, the
optimizer plan from typed scalars, the microbatch slices, the stacked
gradient, the aligned accumulation and the step report's words. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). The optimizer entry call
itself is core/samba_optimizer_plan.mojo `samba_optimizer_run` (typed).

The microbatches of one step run one after another: each reads the stack's
blocks and its tape records, which the block door owns one at a time, and
the mask stream position of microbatch k follows k - 1. The copies here are
single memory moves; the kernels behind the entries are parallel inside."""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.neural_descriptor_run import neural_descriptor_record_admit, neural_descriptor_record_call
from core.samba_entry_table import SAMBA_PRIM_ACCUMULATE
from core.samba_optimizer_plan import SambaOptimizerPlan
from training.checks.optimizer_contract import microbatch_split_is_identical
from training.optimizer_public_common import OptHyper


def samba_step_index_check(step: Int) raises:
    if step < 0 or step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit indexing")


def samba_step_plan_run(tensors: Int, step: Int, maximize: Bool, lr: Float64, beta1: Float64, beta2: Float64,
                        eps: Float64, weight_decay: Float64, max_norm: Float64) -> SambaOptimizerPlan:
    """The AdamW plan of optimizer step `step` (already advanced):
    kind 2, no nesterov, no momentum or dampening; `max_norm` <= 0 is no clip."""
    var hyper = OptHyper(2, 0, maximize, lr, beta1, beta2, eps, weight_decay, 0.0, 0.0, max_norm)
    return SambaOptimizerPlan(tensors, step, hyper)


def samba_step_split_check(accumulation: Int, batch: Int, length: Int) raises:
    """The step's batch splits into `accumulation` aligned microbatches."""
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0 or batch % accumulation != 0:
        raise Error("Samba: batch rows must be divisible by power-of-two accumulation_steps")
    if accumulation > 1 and not microbatch_split_is_identical(batch * length, accumulation):
        raise Error("Samba: MISALIGNED microbatch split; optimizer contract clause 9.2")


def samba_registry_count_check(given: Int, names: Int) raises:
    if given != names:
        raise Error("Samba: gradient registry is incomplete")


def samba_microbatch_copy_run(xpart: Int, ypart: Int, ids: Int, y: Int, index: Int, tokens: Int):
    """Microbatch `index`'s `tokens` int32 ids and targets into their own
    C-order buffers."""
    var size = tokens * 4
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=xpart),
                  src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=ids) + index * size, count=size)
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ypart),
                  src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=y) + index * size, count=size)


def samba_gradient_stack_run(stacked: Int, index: Int, n: Int, offset: Int, source: Int, bytes: Int):
    """One tensor's gradient into row `index` of the (accumulation, n)
    stack, at the tensor's registry offset (in floats)."""
    unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=stacked) + (index * n + offset) * 4,
                  src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=source), count=bytes)


def samba_loss_store(losses: Int, index: Int, value: Float64):
    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=losses)[index] = Float32(value)


def samba_float_read(address: Int, index: Int) -> Float64:
    """Float32 word `index` at `address`, widened."""
    return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=address)[index])


def samba_accumulate_admit(native_id: Int) raises:
    """Admit and count the step's two accumulations (gradient, loss) of the
    descriptor provider (operation 9). The door calls this under the GIL,
    before its release."""
    if native_id != 0:
        neural_descriptor_record_admit(native_id, 9)
        neural_descriptor_record_admit(native_id, 9)


def samba_accumulate_run(prim: EntryRef, native_id: Int, gradient: Int, stacked: Int, n: Int, total: Int,
                         losses: Int, parts: Int, tokens: Int) raises:
    """The aligned reduction of a step's microbatches: the (parts, n)
    gradient stack into `gradient` with divisor `tokens`, then the `parts`
    losses into `total`. Primitive op 9 (the former `accumulate` entry) on
    the provider by id, else the stack's primitive entry."""
    var none = List[Float64]()
    var gradient_addresses: List[Int] = [gradient, stacked]
    var gradient_ints: List[Int] = [n, parts, tokens]
    var loss_addresses: List[Int] = [total, losses]
    var loss_ints: List[Int] = [1, parts, -1]
    if native_id != 0:
        _ = neural_descriptor_record_call(native_id, 9, gradient_addresses, gradient_ints, none)
        _ = neural_descriptor_record_call(native_id, 9, loss_addresses, loss_ints, none)
    else:
        _ = entry_ref_call(prim, SAMBA_PRIM_ACCUMULATE, gradient_addresses, gradient_ints, none)
        _ = entry_ref_call(prim, SAMBA_PRIM_ACCUMULATE, loss_addresses, loss_ints, none)


def samba_step_io_device(clip: Bool) -> Int:
    """The resident step's transfer mask with device parameters: step_device
    semantics and the existing upload/download mask."""
    return 10 if clip else 2


def samba_step_io_host(maximize: Bool, maximize_device: Bool) -> Int:
    return 15 if maximize and maximize_device else 0


def samba_step_clipped(info: Int) -> Bool:
    """Whether the step measured a gradient norm: info is [flag, total_norm,
    clip_coef] (float32)."""
    return Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=info)[0] != 0.0


def samba_step_rate_reported(lr: Float64) -> Float64:
    """The rate the step applied: the float64 attribute rounded to float32."""
    return Float64(Float32(lr))


def samba_parameter_format_check(fmt: String) raises:
    if fmt != "f" and fmt != "<f" and fmt != "=f" and fmt != "@f":
        raise Error("TypeError: Samba optimizer parameters must be float32")


def samba_parameter_extent_check(bytes: Int, begin: Int, end: Int) raises:
    if bytes != end - begin:
        raise Error("Samba optimizer parameter size differs from its registry")
