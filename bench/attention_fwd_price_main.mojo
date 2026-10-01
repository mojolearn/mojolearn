"""The fused forward attention at the byte LM board shape (B1 L2048 nh6 hd64,
causal), standalone: seeded pseudo-random operands, timed rounds, an FNV
digest of ctxv / amax / denom. Env: ATTN_L, ATTN_NH, ATTN_ROUNDS."""
from std.os import getenv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext
from transformer.impl.llama.fused_attention import ATTN_ARM_DEFAULT, FUSED_RAN, fused_forward_launch_estash_ran


def _env_int(name: String, default: Int) raises -> Int:
    var v = String(getenv(name))
    if v == "":
        return default
    return Int(v)


def _fnv(h: UnsafePointer[Float32, MutUntrackedOrigin], n: Int) -> UInt64:
    var x = UInt64(0xcbf29ce484222325)
    for i in range(n):
        var w = UInt64(bitcast[DType.uint32](h[i]))
        x = (x ^ w) * UInt64(0x100000001b3)
    return x


def _hex(x: UInt64) -> String:
    var digits = String("0123456789abcdef")
    var out = String("")
    var v = x
    for _ in range(16):
        var d = Int(v & 15)
        out = String(digits[byte=d]) + out
        v = v >> 4
    return out


def main() raises:
    var l = _env_int("ATTN_L", 2048)
    var nh = _env_int("ATTN_NH", 6)
    var rounds = _env_int("ATTN_ROUNDS", 7)
    comptime HD = 64
    var b = 1
    var nkv = nh
    var s = l
    var ctx = DeviceContext()
    var nq = b * l * nh * HD
    var nk = b * nkv * s * HD
    var hq = ctx.enqueue_create_host_buffer[DType.float32](nq)
    var hk = ctx.enqueue_create_host_buffer[DType.float32](nk)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](nk)
    ctx.synchronize()
    # splitmix64 -> uniform in [-0.5, 0.5), scaled: q and k ~ 0.25, v ~ 1
    var st = UInt64(0x9E3779B97F4A7C15)
    for i in range(nq + 2 * nk):
        st += UInt64(0x9E3779B97F4A7C15)
        var z = st
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var u = Float32(Float64(z >> 11) / 9007199254740992.0) - Float32(0.5)
        if i < nq:
            hq[i] = u * Float32(0.5)
        elif i < nq + nk:
            hk[i - nq] = u * Float32(0.5)
        else:
            hv[i - nq - nk] = u * Float32(2.0)
    var q = ctx.enqueue_create_buffer[DType.float32](nq)
    var k = ctx.enqueue_create_buffer[DType.float32](nk)
    var v = ctx.enqueue_create_buffer[DType.float32](nk)
    var ctxv = ctx.enqueue_create_buffer[DType.float32](nq)
    var amax = ctx.enqueue_create_buffer[DType.float32](b * nh * l)
    var denom = ctx.enqueue_create_buffer[DType.float32](b * nh * l)
    var kept = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=q, src_buf=hq)
    ctx.enqueue_copy(dst_buf=k, src_buf=hk)
    ctx.enqueue_copy(dst_buf=v, src_buf=hv)
    ctx.synchronize()
    var scale = Float32(0.125)
    var ran = -1
    var kept_cells = 0
    var walls = List[Float64]()
    for rep in range(rounds + 2):
        var t0 = perf_counter_ns()
        var status = fused_forward_launch_estash_ran(ctx, ctxv, amax, denom, q, k, v, kept,
            b, l, nh, nkv, HD, s, 0, 0, 0, scale, ATTN_ARM_DEFAULT, ran, kept_cells)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        if status != FUSED_RAN:
            raise Error("fused forward did not run: status " + String(status))
        if rep >= 2:
            walls.append(ms)
        print("round", rep, "ms", ms, "ran_arm", ran, "kept_cells", kept_cells)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](nq)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](b * nh * l)
    var hd_ = ctx.enqueue_create_host_buffer[DType.float32](b * nh * l)
    ctx.enqueue_copy(dst_buf=hc, src_buf=ctxv)
    ctx.enqueue_copy(dst_buf=ha, src_buf=amax)
    ctx.enqueue_copy(dst_buf=hd_, src_buf=denom)
    ctx.synchronize()
    # median
    var w = walls.copy()
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if w[j] < w[i]:
                var t = w[i]
                w[i] = w[j]
                w[j] = t
    print("SHAPE B1 L" + String(l) + " nh" + String(nh) + " hd64 causal; median ms", w[len(w) // 2], "min", w[0])
    print("DIGEST ctx " + _hex(_fnv(hc.unsafe_ptr(), nq)) + " amax " + _hex(_fnv(ha.unsafe_ptr(), b * nh * l)) + " denom " + _hex(_fnv(hd_.unsafe_ptr(), b * nh * l)))
    _ = hq^; _ = hk^; _ = hv^; _ = q^; _ = k^; _ = v^; _ = ctxv^; _ = amax^; _ = denom^; _ = kept^; _ = hc^; _ = ha^; _ = hd_^
