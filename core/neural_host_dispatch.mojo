# SPDX-License-Identifier: Apache-2.0
# The CPU inference entries expose the same private operation ABI as the GPU
# entries. Each path calls the original native host numerical owner; provider
# admission and lifetime management remain shared and outside this module.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, PRIMITIVE_ABI_NAME, MAMBA_STATE_ABI_NAME, MAMBA_BLOCK_ABI_NAME, TRANSFORMER_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.mojolearn_neural_host_impl import _run_forward, _transformer_weights_from, _mamba1_run, _mamba2_run, _mamba3_run
from core.hostptr import copy_f32, f32_ptr, read_f32, read_i32
from training.host.samba_ops_oracle import host_samba_embedding_forward, host_samba_rms_norm_forward, host_samba_linear_forward
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.host.transformer_block_host import transformer_host_forward


def neural_host_primitive_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 3 or (op != 0 and op != 2 and op != 4) or len(p) != (2 if op == 2 else 3) or len(f) != (1 if op == 2 else 0):
        raise Error("CPU inference primitive: invalid descriptor or unavailable operation")
    for address in a:  # small-loop(three admitted primitive buffer addresses)
        if address == 0:
            raise Error("CPU inference primitive: null buffer address")
    for value in p:  # small-loop(two or three primitive dimensions)
        if value < 1:
            raise Error("CPU inference primitive: positive shape required")
    if op == 0:
        var y = host_samba_embedding_forward(read_f32(a[1], p[1] * p[2]), read_i32(a[2], p[0]), p[0], p[1], p[2])
        copy_f32(y.unsafe_ptr(), f32_ptr(a[0]), p[0] * p[2])
        return p[0] * p[2]
    if op == 2:
        var y = host_samba_rms_norm_forward(read_f32(a[1], p[0] * p[1]), read_f32(a[2], p[1]), p[0], p[1], Float32(f[0]))
        copy_f32(y.unsafe_ptr(), f32_ptr(a[0]), p[0] * p[1])
        return p[0] * p[1]
    var y = host_samba_linear_forward(read_f32(a[1], p[0] * p[2]), read_f32(a[2], p[1] * p[2]), p[0], p[1], p[2])
    copy_f32(y.unsafe_ptr(), f32_ptr(a[0]), p[0] * p[1])
    return p[0] * p[1]


def neural_host_mamba_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op < 0 or op > 5:
        raise Error("CPU Mamba state: unknown operation")
    var family = op // 2
    var step = (op & 1) != 0
    var want_a = 14 if family == 0 else (16 if family == 1 else 25)
    var want_p = (2 if step else 3) + family
    if len(a) != want_a or len(p) != want_p or len(f) != (2 if family == 1 else 0):
        raise Error("CPU Mamba state: invalid descriptor arity")
    for address in a:  # driver-loop: the existing per-family buffer address vector
        if address == 0:
            raise Error("CPU Mamba state: null buffer address")
    var batch = p[0]
    var length = 1 if step else p[1]
    var width = p[1] if step else p[2]
    var tail = 2 if step else 3
    if family == 0:
        _mamba1_run(a, batch, length, width, 0)
        return 0
    if family == 1:
        return _mamba2_run(a, batch, length, width, p[tail], Float32(f[0]), Float32(f[1]))
    return _mamba3_run(a, batch, length, width, p[tail], p[tail + 1], False)


def neural_host_transformer_operation(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    var fresh = op == 4
    var packed = op == 34
    if not fresh and op != 33 and not packed:
        raise Error("CPU Transformer inference: unavailable operation")
    var base_a = 11 if fresh else 13
    var base_p = 8 if fresh else 10
    if (len(a) != base_a and len(a) != base_a + BLOCK_OPTION_ADDRS) or (len(p) != base_p and len(p) != base_p + BLOCK_OPTION_PARAMS):
        raise Error("CPU Transformer inference: invalid descriptor arity")
    if packed and (len(a) != 13 + BLOCK_OPTION_ADDRS or len(p) != 10 + BLOCK_OPTION_PARAMS):
        raise Error("CPU fixed15 Transformer: both option tails are required")
    var raw = a.copy()
    while len(raw) < base_a + BLOCK_OPTION_ADDRS:  # small-loop(absent optional weight slots)
        raw.append(0)
    for i in range(base_a):  # driver-loop: the existing base buffer vector, with optional gate slot
        if i != 7 and raw[i] == 0:
            raise Error("CPU Transformer inference: null buffer address")
    var opts = BlockOptions.from_params(p, base_p)
    if not fresh:
        return _run_forward(raw, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], opts, packed)
    var expanded = List[Int]()
    for i in range(10):  # small-loop(ten fresh input/weight slots)
        expanded.append(raw[i])
    expanded.append(0); expanded.append(0); expanded.append(raw[10])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(fixed optional tail)
        expanded.append(raw[11 + i])
    if p[0] <= 0 or p[1] <= 0 or p[2] <= 0 or p[3] <= 0 or p[4] <= 0 or p[5] <= 0 or p[6] <= 0 or p[7] < 0:
        raise Error("transformer: invalid shape or window")
    var w = _transformer_weights_from(expanded, p[2], p[3], p[4], p[5], p[6], p[1], opts, String("transformer_forward_fresh"))
    var x = read_f32(expanded[0], p[0] * p[1] * p[2])
    var out = transformer_host_forward(w, x, p[0], p[1], p[1], 0, p[7], List[Float32](), List[Float32](), False)
    copy_f32(out.y.unsafe_ptr(), f32_ptr(expanded[12]), len(out.y))
    return out.cached_tokens


def neural_host_mamba_fresh(a: List[Int], p: List[Int]) raises -> Int:
    if len(a) != 15 or len(p) != 3:
        raise Error("CPU Mamba3 fresh: invalid descriptor arity")
    var expanded = List[Int](length=25, fill=0)
    for i in range(10):  # small-loop(input and weight slots)
        if a[i] == 0:
            raise Error("CPU Mamba3 fresh: null input or weight")
        expanded[i] = a[i]
    for i in range(10, 15):  # driver-loop: five fresh output/report slots
        if a[i] == 0:
            raise Error("CPU Mamba3 fresh: null output or report")
        expanded[i + 10] = a[i]
    return _mamba3_run(expanded, p[0], p[1], p[2], 0, 0, True)


def neural_host_dispatch[FAMILY: Int](op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                                      result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if na < 0 or na > 25 or np < 0 or np > 27 or nf < 0 or nf > 2 or result == 0:
            raise Error("CPU neural descriptor: invalid span or result")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nf != 0 and ff == 0:
            raise Error("CPU neural descriptor: null float span")
        var floats = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # small-loop(at most two floating parameters)
            f.append(floats[i])
        var value = 0
        comptime if FAMILY == 0:
            value = neural_host_primitive_operation(op, a, p, f)
        elif FAMILY == 1:
            value = neural_host_mamba_operation(op, a, p, f)
        elif FAMILY == 2:
            if nf != 0:
                raise Error("CPU Transformer descriptor: unexpected floats")
            value = neural_host_transformer_operation(op, a, p)
        else:
            if op != 5 or nf != 0:
                raise Error("CPU Mamba inference block: unavailable operation")
            value = neural_host_mamba_fresh(a, p)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeHostPrimitiveAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PRIMITIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neural_host_dispatch[0])


comptime NATIVE_HOST_PRIMITIVE_API = _Global[StorageType=NativeHostPrimitiveAPI,
    name="MojoNeuralHostPrimitiveAPIv1", init_fn=NativeHostPrimitiveAPI.__init__]


struct NativeHostMambaStateAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_STATE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neural_host_dispatch[1])


comptime NATIVE_HOST_MAMBA_STATE_API = _Global[StorageType=NativeHostMambaStateAPI,
    name="MojoNeuralHostMambaStateAPIv1", init_fn=NativeHostMambaStateAPI.__init__]


struct NativeHostTransformerAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TRANSFORMER_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neural_host_dispatch[2])


comptime NATIVE_HOST_TRANSFORMER_API = _Global[StorageType=NativeHostTransformerAPI,
    name="MojoNeuralHostTransformerAPIv1", init_fn=NativeHostTransformerAPI.__init__]


struct NativeHostMambaBlockAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(neural_host_dispatch[3])


comptime NATIVE_HOST_MAMBA_BLOCK_API = _Global[StorageType=NativeHostMambaBlockAPI,
    name="MojoNeuralHostMambaBlockAPIv1", init_fn=NativeHostMambaBlockAPI.__init__]
