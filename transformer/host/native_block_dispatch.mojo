# SPDX-License-Identifier: Apache-2.0
# CPU training entry table: same provider transport, original host numerical
# owners. Inference-only and training-host implementations remain distinct.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, TRANSFORMER_BLOCK_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.transformer_host_binding_impl import _run_forward, _backward_run, HostTransformerSession


def transformer_host_block_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(f) != 0:
        raise Error("Transformer host provider: unexpected floating scalars")
    if op == 12 or op == 13:
        if len(p) != 1 or p[0] == 0 or len(a) != 0:
            raise Error("transformer tape: invalid administration descriptor")
        var owner = MutPointer[HostTransformerSession, MutUntrackedOrigin](unsafe_from_address=p[0])
        if owner[].busy or (op == 12 and owner[].closed):
            raise Error("transformer tape: session is busy or closed")
        owner[].invalidate_tape()
        if op == 13:
            owner[].closed = True
        return 0
    if op == 5:
        if len(a) != 21 or len(p) != 8:
            raise Error("transformer backward: expected 21 addresses and eight scalars")
        for address in a:  # driver-loop: fixed input and gradient address vector
            if address == 0:
                raise Error("transformer backward: null buffer address")
        if p[0] <= 0 or p[1] <= 0 or p[2] <= 0:
            raise Error("transformer backward: B, L and d_model must be positive")
        _backward_run(a, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        return 0
    if op != 4:
        raise Error("Transformer host training provider: unavailable operation")
    if (len(a) != 11 and len(a) != 11 + BLOCK_OPTION_ADDRS) or (len(p) != 8 and len(p) != 8 + BLOCK_OPTION_PARAMS):
        raise Error("transformer_forward_fresh: invalid descriptor arity")
    var raw = a.copy()
    while len(raw) < 11 + BLOCK_OPTION_ADDRS:  # small-loop(optional address tail)
        raw.append(0)
    for i in range(11):  # small-loop(original base address admission)
        if i != 7 and raw[i] == 0:
            raise Error("transformer_forward_fresh: null buffer address")
    var expanded = List[Int]()
    for i in range(10):  # small-loop(input plus weight vector)
        expanded.append(raw[i])
    expanded.append(0); expanded.append(0); expanded.append(raw[10])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(optional weight tail)
        expanded.append(raw[11 + i])
    var opts = BlockOptions.from_params(p, 8)
    return _run_forward(expanded, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[1], 0, p[7], False, opts)


def transformer_host_block_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                          result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 36 or na < 0 or na > 25 or np < 0 or np > 28 or nf < 0 or nf > 2 or result == 0:
            raise Error("transformer host provider: invalid descriptor span or operation")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        if nf != 0 and ff == 0:
            raise Error("transformer host provider: null float span")
        var raw = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        var f = List[Float64]()
        for i in range(nf):  # small-loop(at most two admitted floating scalars)
            f.append(raw[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = transformer_host_block_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct TransformerHostBlockAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TRANSFORMER_BLOCK_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(transformer_host_block_v1)


comptime NATIVE_TRANSFORMER_HOST_BLOCK_API = _Global[StorageType=TransformerHostBlockAPI,
    name="MojoTransformerHostBlockAPIv1", init_fn=TransformerHostBlockAPI.__init__]
