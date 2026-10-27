# SPDX-License-Identifier: Apache-2.0
# Native descriptor counterpart of the existing carried-state Mamba entries.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError, MAMBA_STATE_ABI_NAME
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from mamba.impl.mamba_shell_impl import _mamba1_run, _mamba2_run, _mamba3_run


def mamba_state_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    var step = (op & 1) != 0
    var b = p[0]
    var l = 1 if step else p[1]
    var dm = p[1] if step else p[2]
    var tail = 2 if step else 3
    if op < 2:
        var position = 0
        _mamba1_run(a, b, l, dm, step, position)
        return 0
    if op < 4:
        return _mamba2_run(a, b, l, dm, p[tail], Float32(f[0]), Float32(f[1]))
    if op < 6:
        return _mamba3_run(a, b, l, dm, p[tail], p[tail + 1])
    raise Error("Mamba native ABI: unknown operation")


def mamba_state_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= 6:
            raise Error("Mamba native ABI: unknown operation")
        var family = op // 2
        var step = (op & 1) != 0
        var name = "mamba" + String(family + 1) + ("_decode_step" if step else "_forward")
        var counts: List[Int] = [14, 16, 25]
        var descriptions = [
            String(" (x, norm.weight, in_proj.weight, conv1d.weight, conv1d.bias, x_proj.weight, dt_proj.weight, dt_proj.bias, A_log, D, out_proj.weight, conv_window, h, y_out)"),
            String(" (x, norm.weight, in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log, D, gated norm.weight, out_proj.weight, conv_window, h, buffer_xbc, buffer_dtraw, y_out, h_last_out)"),
            String(" (x, block norm.weight, in_proj.weight, dt_bias, B_norm.weight, C_norm.weight, B_bias, C_bias, D, out_proj.weight, theta, h, buffer_qrot, buffer_krot, buffer_v, buffer_dt, buffer_sig, buffer_adt, pending_k, pending_v, y_out, h_last_out, k_last_out, v_last_out, theta_last_out)"),
        ]
        if na != counts[family]:
            raise Error(name + ": addrs must contain " + String(counts[family]) + " addresses" + descriptions[family] + ", got " + String(na))
        var a = native_descriptor_words(aa, na)
        var ni = (2 if step else 3) + family
        var nr = 2 if family == 1 else 0
        if np != ni or nf != nr:
            var params = "B, " + ("" if step else "L, ") + "d_model"
            if family > 0:
                params += ", buf_len"
            if family == 1:
                params += ", dt_lo, dt_hi"
            elif family == 2:
                params += ", pending"
            raise Error(name + ": params must contain " + String(ni + nr) + " values (" + params + "), got " + String(np + nf))
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nr != 0:
            if ff == 0:
                raise Error("Mamba native ABI: null scalar span")
            var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            f.append(fp[0])
            f.append(fp[1])
        if result == 0:
            raise Error("Mamba native ABI: null result")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = mamba_state_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeMambaStateAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MAMBA_STATE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mamba_state_v1)


comptime NATIVE_MAMBA_STATE_API = _Global[StorageType=NativeMambaStateAPI,
    name="MojoMambaStateNativeAPIv1", init_fn=NativeMambaStateAPI.__init__]
