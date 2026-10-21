# SPDX-License-Identifier: Apache-2.0
"""Typed state of every live SmallMLPTrainer: the native state handle's store.

A trainer holds one integer handle. Its parameters (packed in the registry
order weight1, bias1, weight2, bias2 at offsets 0, 128, 144, 192, 195), its
AdamW moments, flags and step counter, its float32-rounded optimizer settings
and its canonical data schedule live here, in Mojo memory. The Python door
(`bindings/small_mlp_native.mojo`) converts arguments and calls the compiled
training entries with addresses into these buffers; nothing here imports
Python.
"""
from std.ffi import _Global
from std.math import isfinite
from std.memory import memcpy

comptime MLP_TOTAL = 195
comptime MLP_MAX_STEP = 2147483647
# A handle is generation * MLP_HANDLE_SPAN + slot index. The generation makes
# a stale handle (a closed trainer, or a shallow copy of one) refuse instead of
# addressing whichever trainer reuses its slot.
comptime MLP_HANDLE_SPAN = 1 << 20


def mlp_offset(index: Int) -> Int:
    """Start of parameter tensor `index` in the packed registry."""
    if index == 0:
        return 0
    if index == 1:
        return 128
    if index == 2:
        return 144
    if index == 3:
        return 192
    return MLP_TOTAL


def mlp_size(index: Int) -> Int:
    return mlp_offset(index + 1) - mlp_offset(index)


struct MlpState(Copyable, Movable):
    var params: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Int32]
    var step: Int
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var schedule: String

    def __init__(out self):
        self.params = List[Float32](length=MLP_TOTAL, fill=Float32(0))
        self.m = List[Float32](length=MLP_TOTAL, fill=Float32(0))
        self.v = List[Float32](length=MLP_TOTAL, fill=Float32(0))
        self.flags = List[Int32](length=4, fill=Int32(0))
        self.step = 0
        self.lr = 0.0
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.eps = 0.0
        self.weight_decay = 0.0
        self.schedule = String()

    def params_addr(self, index: Int) -> Int:
        return Int(self.params.unsafe_ptr()) + 4 * mlp_offset(index)

    def m_addr(self) -> Int:
        return Int(self.m.unsafe_ptr())

    def v_addr(self) -> Int:
        return Int(self.v.unsafe_ptr())

    def flags_addr(self) -> Int:
        return Int(self.flags.unsafe_ptr())


struct MlpSlot(Copyable, Movable):
    var live: Bool
    var generation: Int
    var state: MlpState
    # The Apple FAST device-resident session (-1: none). While it is open it
    # is the truth for the parameters and moments; `resident_mode` names the
    # numeric mode whose binding opened it and `resident_cap` its minibatch
    # capacity.
    var resident: Int
    var resident_mode: String
    var resident_cap: Int

    def __init__(out self):
        self.live = False
        self.generation = 0
        self.state = MlpState()
        self.resident = -1
        self.resident_mode = String()
        self.resident_cap = 0


struct MlpStore(Defaultable, Movable):
    var slots: List[MlpSlot]
    var next_generation: Int

    def __init__(out self):
        self.slots = List[MlpSlot]()
        self.next_generation = 1


comptime MLP_STORE = _Global[StorageType=MlpStore, name="MojoSmallMlpTrainerStore", init_fn=MlpStore.__init__]


def mlp_store_create(var state: MlpState) raises -> Int:
    var store = MLP_STORE.get_or_create_ptr()
    var index = -1
    for i in range(len(store[].slots)):
        if not store[].slots[i].live:
            index = i
            break
    if index < 0:
        if len(store[].slots) >= MLP_HANDLE_SPAN:
            raise Error("RuntimeError: SmallMLPTrainer has too many live trainers")
        store[].slots.append(MlpSlot())
        index = len(store[].slots) - 1
    var generation = store[].next_generation
    store[].next_generation += 1
    store[].slots[index].live = True
    store[].slots[index].generation = generation
    store[].slots[index].state = state^
    store[].slots[index].resident = -1
    store[].slots[index].resident_mode = String()
    store[].slots[index].resident_cap = 0
    return generation * MLP_HANDLE_SPAN + index


def mlp_store_index(handle: Int) raises -> Int:
    """The live slot of `handle`, or a refusal for a closed or foreign one."""
    var store = MLP_STORE.get_or_create_ptr()
    var index = handle % MLP_HANDLE_SPAN if handle > 0 else -1
    if index < 0 or index >= len(store[].slots) or not store[].slots[index].live or (
            store[].slots[index].generation != handle // MLP_HANDLE_SPAN):
        raise Error("RuntimeError: SmallMLPTrainer state is closed or was never created")
    return index


def mlp_store_release(handle: Int) -> Bool:
    """Free `handle`'s slot; False when it is not live (already released)."""
    var store = MLP_STORE.get_or_create_ptr()
    var index = handle % MLP_HANDLE_SPAN if handle > 0 else -1
    if index < 0 or index >= len(store[].slots) or not store[].slots[index].live or (
            store[].slots[index].generation != handle // MLP_HANDLE_SPAN):
        return False
    store[].slots[index].live = False
    store[].slots[index].state = MlpState()
    store[].slots[index].resident = -1
    store[].slots[index].resident_mode = String()
    store[].slots[index].resident_cap = 0
    return True


def mlp_copy_tensor_out(state: MlpState, index: Int, dst_addr: Int):
    """Parameter tensor `index` into a row-major float32 buffer."""
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    memcpy(dest=dst, src=state.params.unsafe_ptr() + mlp_offset(index), count=mlp_size(index))


def mlp_copy_tensor_in(mut state: MlpState, index: Int, src_addr: Int):
    var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    memcpy(dest=state.params.unsafe_ptr() + mlp_offset(index), src=src, count=mlp_size(index))


def mlp_copy_f32(src_addr: Int, dst_addr: Int, n: Int):
    var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    memcpy(dest=dst, src=src, count=n)


def mlp_copy_i32(src_addr: Int, dst_addr: Int, n: Int):
    var src = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=src_addr)
    var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    memcpy(dest=dst, src=src, count=n)


def mlp_require_finite(addr: Int, n: Int, label: String) raises:
    """ValueError naming `label` when any of `n` float32 values is not finite."""
    var data = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(n):
        if not isfinite(data[i]):
            raise Error("SmallMLPTrainer " + label + " must be finite")


def mlp_require_result_finite(addr: Int, n: Int, label: String) raises:
    """RuntimeError: an operation's output is not finite."""
    var data = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(n):
        if not isfinite(data[i]):
            raise Error("RuntimeError: SmallMLPTrainer " + label + " returned an invalid result")


def mlp_require_moments(state: MlpState) raises:
    """The resumable-state invariants: v nonnegative, flags binary."""
    for i in range(MLP_TOTAL):
        if state.v[i] < 0:
            raise Error("SmallMLPTrainer state requires nonnegative v and binary flags")
    for i in range(4):
        if state.flags[i] < 0 or state.flags[i] > 1:
            raise Error("SmallMLPTrainer state requires nonnegative v and binary flags")


def mlp_require_state(state: MlpState) raises:
    """Every check a state passes before it is published: finite parameters
    and moments, nonnegative v, binary flags."""
    mlp_require_finite(Int(state.params.unsafe_ptr()), MLP_TOTAL, "parameters")
    mlp_require_finite(state.m_addr(), MLP_TOTAL, "m")
    mlp_require_finite(state.v_addr(), MLP_TOTAL, "v")
    mlp_require_moments(state)


def mlp_read_f32(addr: Int) -> Float64:
    """The float32 scalar a training entry wrote at `addr` (a loss), widened."""
    return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=addr)[0])
