# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba state door (bindings/samba_state_native_door.mojo):
the stack's state as typed words. The ordered parameter registry derived
from a configuration, the saved-state checks in the door's words, the exact
learning rate of one step, the parameter publication and the words of the
per-call optimizer step. Typed Mojo only, no Python import
(tools/python_free_dirs.sh).

The typed words hold addresses of buffers the door's callers own; the door
keeps every owner named past the call that reads its address.

Registry and scalar metadata of one stack: a handful of integers per
tensor, with no independent units worth a worker. The byte copies are one
`memcpy` each.
"""
from std.memory import unsafe_memcpy
from core.device_plan import DevicePlan
from core.weight_epoch import registry_epoch_bump
from sequence.lr_exact import schedule_exact
from training.optimizer_device_plan import optimizer_device_plan
from training.samba_config_run import (
    SambaShape, SAMBA_LAYER_MAMBA3, SAMBA_LAYER_ATTENTION, samba_block_names, samba_block_dims,
)


def samba_optimizer_device_plan() raises -> DevicePlan:
    """The optimizer device plan of one Samba entry: `optimizer_device_plan`,
    read once here and passed down as a value."""
    return optimizer_device_plan()


# ------------------------------------------------------------- the registry


struct SambaRegistry(Movable):
    """The ordered flat parameter registry of one configuration: tensor
    names, shapes and offsets into the flat float32 storage. `offsets` has
    one more entry than `names`; its last entry is the parameter count."""
    var names: List[String]
    var dims: List[List[Int]]
    var offsets: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.dims = List[List[Int]]()
        self.offsets = List[Int]()
        self.offsets.append(0)

    def tensors(self) -> Int:
        return len(self.names)

    def total(self) -> Int:
        return self.offsets[len(self.offsets) - 1]

    def size(self, index: Int) -> Int:
        return self.offsets[index + 1] - self.offsets[index]


def _registry_add(mut registry: SambaRegistry, name: String, dims: List[Int]) raises:
    var n = 1
    for i in range(len(dims)):  # driver-loop: over tensor shape metadata
        var d = dims[i]
        if d < 1 or n > 2147483647 // d:
            raise Error("Samba: invalid tensor shape")
        n *= d
    var offset = registry.total()
    if offset > 2147483647 - n:
        raise Error("Samba: flat parameter registry exceeds signed 32-bit indexing")
    registry.names.append(name)
    registry.dims.append(dims.copy())
    registry.offsets.append(offset + n)


def samba_registry_run(shape: SambaShape, layers: List[Int]) raises -> SambaRegistry:
    """The registry of one configuration: the embedding, each layer's
    tensors in block order, the final norm and (untied) the head. Every
    layer of one kind has the same schema, built once per kind."""
    var registry = SambaRegistry()
    var dm = shape.d_model
    _registry_add(registry, String("embed.weight"), [shape.vocab, dm])
    var mamba_names = samba_block_names(SAMBA_LAYER_MAMBA3)
    var mamba_dims = samba_block_dims(shape, SAMBA_LAYER_MAMBA3)
    var attention_names = samba_block_names(SAMBA_LAYER_ATTENTION)
    var attention_dims = samba_block_dims(shape, SAMBA_LAYER_ATTENTION)
    for i in range(len(layers)):  # driver-loop: over configured model layers
        var prefix = "layers." + String(i) + "."
        if layers[i] == SAMBA_LAYER_MAMBA3:
            for j in range(len(mamba_names)):  # driver-loop: over per-layer tensor registry metadata
                _registry_add(registry, prefix + mamba_names[j], mamba_dims[j])
        else:
            for j in range(len(attention_names)):  # driver-loop: over per-layer tensor registry metadata
                _registry_add(registry, prefix + attention_names[j], attention_dims[j])
    _registry_add(registry, String("norm_f.weight"), [dm])
    if not shape.tie_embeddings:
        _registry_add(registry, String("lm_head.weight"), [shape.vocab, dm])
    return registry^


# ------------------------------------------------------------- the schedule


@fieldwise_init
struct SambaSchedule(Copyable, Movable):
    """A built-in learning-rate schedule as typed words: `kind` 0 constant,
    1 linear, 2 cosine; `total` is -1 when the schedule has no total."""
    var kind: Int
    var peak: Float64
    var low: Float64
    var warmup: Int
    var total: Int


def samba_schedule_kind(kind: String) raises -> Int:
    if kind == "constant":
        return 0
    if kind == "linear":
        return 1
    if kind == "cosine":
        return 2
    raise Error("Samba: unsupported saved learning-rate schedule")


def samba_schedule_rate_run(schedule: SambaSchedule, step: Int) raises -> Float64:
    """The exact float32 learning rate of ONE-BASED `step`
    (sequence/lr_exact.mojo `schedule_exact`, the same host rationals the
    public schedules use), refused in the door's words."""
    var answer = schedule_exact(schedule.kind, schedule.peak, schedule.low, schedule.warmup, schedule.total, step)
    if answer.status != 0:
        raise Error("Samba: native exact learning-rate schedule refused saved configuration, status " + String(answer.status))
    return Float64(answer.value)


# ------------------------------------------------------ saved-state checks


def samba_state_header_check(schema: String, profile: String) raises:
    if schema != "mojolearn.samba-stack-state.v1" or profile != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba: state schema/profile mismatch")


def samba_numeric_profile_check(saved: String) raises:
    if saved != "fp32_v1":
        raise Error("Samba: state numeric profile must be fp32_v1, got " + saved)


def samba_flags_check(rank: Int, length: Int, tensors: Int, format: String) raises:
    """The optimizer's initialized flags: one int32 per registered tensor."""
    if rank != 1 or length != tensors or (format != "i" and format != "<i" and format != "=i"):
        raise Error("Samba: optimizer flags must be one int32 per registered tensor")


def samba_step_check(step: Int) raises:
    if step < 0 or step > 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")


def samba_accumulation_check(accumulation: Int) raises:
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("Samba: accumulation_steps must be a positive power of two")


def samba_optimizer_kind_check(kind: String) raises:
    if kind != "adamw":
        raise Error("Samba: optimizer kind must be adamw")


def samba_counter_check(counter: Int) raises:
    if counter < 0 or counter > 4294967296:
        raise Error("Samba: RNG counter must be in [0, 2^32], including exhaustion")


# ------------------------------------------------------------- publication


def samba_parameters_publish_run(flat: Int, source: Int, count: Int) raises:
    """Copy `count` bytes of saved parameters into the stack's registry at
    `flat` (nothing when the saved copy IS the registry) and stamp the
    registry's weight epoch from the address this write holds."""
    if flat != source:
        var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=flat)
        var src = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=source)
        unsafe_memcpy(dest=dst, src=src, count=count)
    registry_epoch_bump(flat)


def samba_offsets_fill_run(registry: SambaRegistry, address: Int):
    """Write the registry's J+1 offsets as int32 at `address`."""
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(len(registry.offsets)):  # driver-loop: over registry offset metadata
        dst[i] = Int32(registry.offsets[i])


# ---------------------------------------------- the per-call optimizer step


def samba_apply_step_check(step: Int) raises:
    if step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")


def samba_max_norm_check(given: Bool, max_norm: Float64) raises:
    if given and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")


def samba_gradient_place_run(registry: SambaRegistry, index: Int, flat_gradient: Int, source: Int):
    """Copy tensor `index`'s admitted gradient into its registry slot of the
    flat gradient at `flat_gradient`."""
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=flat_gradient)
    var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=source)
    unsafe_memcpy(dest=dst + registry.offsets[index], src=src, count=registry.size(index))


def samba_apply_ints(tensors: Int, step: Int) -> List[Int]:
    """The integer words of the per-call AdamW step (primitive op 25):
    [J, kind, t, nesterov, maximize]."""
    var words: List[Int] = [tensors, 2, step, 0, 0]
    return words^


def samba_apply_floats(lr: Float64, beta1: Float64, beta2: Float64, eps: Float64, weight_decay: Float64,
                       max_norm: Float64) -> List[Float64]:
    """The float words of the per-call AdamW step: [lr, beta1, beta2, eps,
    weight_decay, momentum, dampening, max_norm]."""
    var words: List[Float64] = [lr, beta1, beta2, eps, weight_decay, 0.0, 0.0, max_norm]
    return words^


def samba_apply_written_check(written: Int, count: Int) raises:
    if written != count:
        raise Error("RuntimeError: Samba optimizer returned an incomplete update")
