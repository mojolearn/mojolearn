# SPDX-License-Identifier: Apache-2.0
# The GEMM routes of DeviceExec (sequence/gemm_tiled.mojo seq_gemm_route,
# lane ag-pass4-gemm): every product is run on every route (one thread a
# cell, the 16 x 16 tile, the thin 16 x 1 and 1 x 16 tiles) and through the
# executor's own dispatch, and the words must be EQUAL BIT FOR BIT across
# routes (each is op_gemm's chain, k ascending, with op_gemm_epi's follower).
# The plain products and the column sums are also checked against a
# float64 oracle within a rounding bound; the host executor's words are
# compared and the differences counted (reported, not required).
# Shapes: the HIGGS MLP's five GEMMs a batch (every epilogue kind) and the
# neighbours M in (1, 17, 300), N in (1, 2, 3, 16, 17, 64, 65), K in (1, 2,
# 27, 28, 29, 255, 256, 257), each in a dense and a transposed layout, and
# the staging boundaries M in (1, 17), N in (1, 16, 33), K in (15, 16, 17,
# 63, 64, 65, 127, 128, 129) (lane ag-pass5-gemm).
# Off NVIDIA/AMD identical (Apple, FAST) every route is the one-thread launch.
#
# Lane ag-pass6-gemm: the 64 x 64 micro-tile (GR_MICRO) is a forced route on
# every shape like the others (the same words required); the split-K order
# (sequence/gemm_split.mojo, a BIT VERSION) is run forced on every shape on
# the device and on the host (OP_GK_SPLIT), and those words must be EQUAL
# device vs host (SPLITDIFF fails the check); where the shape rule selects
# it, the dispatch and the host executor are compared against it instead of
# the one-chain route. The pass-5 shapes keep the old digest ("digest"); the
# pass-6 shapes (micro edges 63..65 x K 15..129, long-K split shapes) add
# "digest6", and every case's split words "split_digest".
from std.memory import bitcast
from sequence.ops import FP, Args, OP_GEMM, OP_GEMM_EPI
from sequence.mlp import EPI_BIAS_ACT, EPI_L2GRAD, EPI_ACT_BWD, ACT_LOGISTIC, ACT_TANH, ACT_RELU, ACT_SOFTMAX
from sequence.gemm_tiled import GR_CELL, GR_TILE, GR_THIN_N, GR_THIN_M, GR_MICRO, GR_TILE16, seq_gemm_route
from sequence.gemm_split import OP_GK_SPLIT, gemm_split_args, gemm_split_on
from sequence.exec_device import DeviceExec
from sequence.exec import HostExec
from autograd.ops import OP_AGK_GEMM_EPI

# GR_CELL, GR_TILE (the tiled family as the dispatch resolves it), GR_THIN_N,
# GR_THIN_M, GR_MICRO and GR_TILE16 forced, then the dispatch's own
comptime N_ROUTES = 7
comptime N_KINDS = 10


def _fill(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):
        var h = (i * 7919 + salt * 104729 + (i // 13) * 31) % 2003
        v[i] = Float32(h - 1001) / Float32(997.0)
    return v^


def _hp(v: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(v.unsafe_ptr()))


def _kind(k: Int) -> Tuple[Int, Int]:
    """(epi, activation) of epilogue kind k: 0 none, 1-4 bias (+ none, ReLU,
    logistic, tanh), 5 L2GRAD, 6-8 act' (ReLU, logistic, tanh), 9 bias +
    tanh again under the column sums' launch (k is only a case index)."""
    if k == 0:
        return (0, 0)
    if k == 1:
        return (EPI_BIAS_ACT, 0)
    if k == 2:
        return (EPI_BIAS_ACT, ACT_RELU)
    if k == 3:
        return (EPI_BIAS_ACT, ACT_LOGISTIC)
    if k == 4 or k == 9:
        return (EPI_BIAS_ACT, ACT_TANH)
    if k == 5:
        return (EPI_L2GRAD, 0)
    if k == 10:
        # bias with the softmax code: the follower leaves z (the softmax is
        # its own launch), as the hand path launches a softmax output layer
        return (EPI_BIAS_ACT, ACT_SOFTMAX)
    if k == 6:
        return (EPI_ACT_BWD, ACT_RELU)
    if k == 7:
        return (EPI_ACT_BWD, ACT_LOGISTIC)
    return (EPI_ACT_BWD, ACT_TANH)


struct GemmTally(Movable):
    var cases: Int
    var launches: Int
    var host_diffs: Int
    var digest: UInt64
    #: lane ag-pass6-gemm: 0 = the pass-5 shapes (digest), 1 = the pass-6
    #: shapes (cases6, digest6); the split words of every case (split_digest)
    var phase: Int
    var cases6: Int
    var digest6: UInt64
    var split_cases: Int
    var split_hits: Int
    var split_diffs: Int
    var split_digest: UInt64

    def __init__(out self):
        self.cases = 0
        self.launches = 0
        self.host_diffs = 0
        self.digest = UInt64(1469598103934665603)
        self.phase = 0
        self.cases6 = 0
        self.digest6 = UInt64(1469598103934665603)
        self.split_cases = 0
        self.split_hits = 0
        self.split_diffs = 0
        self.split_digest = UInt64(1469598103934665603)

    def mix(mut self, v: List[Float32]):
        for i in range(len(v)):
            if self.phase == 0:
                self.digest = (self.digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)
            else:
                self.digest6 = (self.digest6 ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)

    def mix_split(mut self, v: List[Float32]):
        for i in range(len(v)):
            self.split_digest = (self.split_digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)


def _same_bits(name: String, route: Int, got: List[Float32], ref_: List[Float32]) raises:
    for i in range(len(ref_)):
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](ref_[i]):
            print("FAIL", name, "route", route, "index", i, "got", got[i], "route0", ref_[i])
            raise Error(name + ": a GEMM route changed a word")


def _count_diffs(got: List[Float32], ref_: List[Float32]) -> Int:
    var d = 0
    for i in range(len(ref_)):
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](ref_[i]):
            d += 1
    return d


def _run[OP: Int](
    mut dev: DeviceExec, mut host: HostExec, mut tally: GemmTally,
    M: Int, N: Int, K: Int, transposed: Bool, kind: Int, accumulate: Bool,
) raises:
    """One product on every route (OP_GEMM: kind 0, optionally accumulating
    into a C of row stride N + 3; the epilogue opcodes: dense C, kind's
    epilogue, and for OP_AGK_GEMM_EPI with L2GRAD in the dense layout the
    column sums too)."""
    var ek = _kind(kind)
    var epi = ek[0]
    var act = ek[1]
    var ldc = N + 3 if accumulate else N
    var colsum = OP == OP_AGK_GEMM_EPI and epi == EPI_L2GRAD and not transposed
    var name = (
        String("gemm/op") + String(OP) + "_" + String(M) + "x" + String(N) + "x" + String(K)
        + (String("_T") if transposed else String("_D")) + "_kind" + String(kind)
        + (String("_acc") if accumulate else String(""))
    )
    var ha = _fill(M * K, 1 + M + kind)
    var hb = _fill(K * N, 2 + N + kind)
    var hc0 = _fill(M * ldc, 3 + K)
    var h3 = _fill(M * N, 4 + kind)
    if act == ACT_RELU and epi == EPI_ACT_BWD:
        for i in range(len(h3)):
            if h3[i] < Float32(0.0):
                h3[i] = Float32(0.0)
    elif epi == EPI_ACT_BWD:
        for i in range(len(h3)):
            h3[i] = Float32(0.5) + Float32(0.4) * h3[i]
    var a = Args()
    a.i0 = M
    a.i1 = N
    a.i2 = K
    if transposed:
        a.i3 = 1
        a.i4 = M
        a.i5 = 1
        a.i6 = K
    else:
        a.i3 = K
        a.i4 = 1
        a.i5 = N
        a.i6 = 1
    a.i7 = 1 if accumulate else 0
    a.i8 = ldc
    a.i9 = epi
    a.i10 = act
    a.f0 = Float32(0.03)
    a.f1 = Float32(7.0)
    if colsum:
        a.i11 = 1
    # device buffers
    var dA = dev.alloc(M * K)
    var dB = dev.alloc(K * N)
    var dC = dev.alloc(M * ldc)
    var d3 = dev.alloc(M * N)
    var d4 = dev.alloc(N)
    dev.upload(dA, _hp(ha), M * K)
    dev.upload(dB, _hp(hb), K * N)
    dev.upload(d3, _hp(h3), M * N)
    # lane ag-pass6-gemm: the split-K order forced (OP_GK_SPLIT), device
    # words; the host's are compared below. split_on: the dispatch takes it.
    a.p0 = dA
    a.p1 = dB
    a.p2 = dC
    a.p3 = d3
    a.p4 = d4
    var split_on = gemm_split_on[OP](a)
    dev.upload(dC, _hp(hc0), M * ldc)
    dev.launch[OP_GK_SPLIT](gemm_split_args[OP](a), M * N)
    tally.launches += 1
    var split_dev = List[Float32](length=M * ldc, fill=0)
    dev.download(_hp(split_dev), dC, M * ldc)
    # OP_AGK_GEMM_EPI's column sums ride in the split launch (i11)
    var split_cs = List[Float32](length=N, fill=0)
    if colsum:
        dev.download(_hp(split_cs), d4, N)
    var first = List[Float32]()
    var first_cs = List[Float32]()
    for route in range(N_ROUTES):  # driver-loop: over the routes compared
        dev.upload(dC, _hp(hc0), M * ldc)
        var zero = List[Float32](length=N, fill=0)
        dev.upload(d4, _hp(zero), N)
        _ = zero^  # keep the host list alive past the upload
        a.p0 = dA
        a.p1 = dB
        a.p2 = dC
        a.p3 = d3
        a.p4 = d4
        if route < N_ROUTES - 1:
            dev.gemm_launch_route[OP](a, M * N, route)
        else:
            dev.launch[OP](a, M * N)
        tally.launches += 1
        var got = List[Float32](length=M * ldc, fill=0)
        dev.download(_hp(got), dC, M * ldc)
        var got_cs = List[Float32](length=N, fill=0)
        if colsum:
            dev.download(_hp(got_cs), d4, N)
        if route == 0:
            first = got.copy()
            first_cs = got_cs.copy()
        elif route == N_ROUTES - 1 and split_on:
            # the dispatch took the split order: its words are the split's
            _same_bits(name + "/split_dispatch", route, got, split_dev)
            if colsum:
                _same_bits(name + "/colsum", route, got_cs, first_cs)
        else:
            _same_bits(name, route, got, first)
            if colsum:
                _same_bits(name + "/colsum", route, got_cs, first_cs)
    if colsum:
        _same_bits(name + "/split_colsum", N_ROUTES, split_cs, first_cs)
    dev.free(dA)
    dev.free(dB)
    dev.free(dC)
    dev.free(d3)
    dev.free(d4)
    # the float64 oracle: the plain product and the column sums
    if epi == 0:
        for m in range(M):
            for n in range(N):
                var s = Float64(0.0)
                var mag = Float64(0.0)
                if accumulate:
                    s = Float64(hc0[m * ldc + n])
                    mag = abs(s)
                for k in range(K):
                    var t = Float64(ha[m * a.i3 + k * a.i4]) * Float64(hb[k * a.i5 + n * a.i6])
                    s += t
                    mag += abs(t)
                var tol = Float64(K + 2) * Float64(1.2e-7) * mag + Float64(1e-6)
                if not (abs(Float64(first[m * ldc + n]) - s) <= tol):
                    print("FAIL", name, "oracle cell", m, n, "got", first[m * ldc + n], "float64", s)
                    raise Error(name + ": the GEMM is off the float64 oracle")
                if not (abs(Float64(split_dev[m * ldc + n]) - s) <= tol):
                    print("FAIL", name, "oracle split cell", m, n, "got", split_dev[m * ldc + n], "float64", s)
                    raise Error(name + ": the split-K GEMM is off the float64 oracle")
    if colsum:
        for n in range(N):
            var s = Float64(0.0)
            var mag = Float64(0.0)
            for k in range(K):
                var t = Float64(hb[k * N + n])
                s += t
                mag += abs(t)
            var ref_ = s / Float64(a.f1)
            var tol = (Float64(K + 2) * Float64(1.2e-7) * mag + Float64(1e-6)) / Float64(a.f1)
            if not (abs(Float64(first_cs[n]) - ref_) <= tol):
                print("FAIL", name, "oracle colsum", n, "got", first_cs[n], "float64", ref_)
                raise Error(name + ": the column sums are off the float64 oracle")
    # the host executor (reported)
    var hA = host.alloc(M * K)
    var hB = host.alloc(K * N)
    var hC = host.alloc(M * ldc)
    var hp3 = host.alloc(M * N)
    var hp4 = host.alloc(N)
    host.upload(hA, _hp(ha), M * K)
    host.upload(hB, _hp(hb), K * N)
    host.upload(hC, _hp(hc0), M * ldc)
    host.upload(hp3, _hp(h3), M * N)
    a.p0 = hA
    a.p1 = hB
    a.p2 = hC
    a.p3 = hp3
    a.p4 = hp4
    host.launch[OP](a, M * N)
    var hgot = List[Float32](length=M * ldc, fill=0)
    host.download(_hp(hgot), hC, M * ldc)
    var hd: Int
    if split_on:
        hd = _count_diffs(hgot, split_dev)
    else:
        hd = _count_diffs(hgot, first)
    if colsum:
        var hcs = List[Float32](length=N, fill=0)
        host.download(_hp(hcs), hp4, N)
        hd += _count_diffs(hcs, first_cs)
    if hd != 0:
        print("HOSTDIFF", name, "words", hd)
    tally.host_diffs += hd
    # the split order forced on the host: the same words as the device's
    # (required: the bit version's identity, device vs host)
    host.upload(hC, _hp(hc0), M * ldc)
    host.launch[OP_GK_SPLIT](gemm_split_args[OP](a), M * N)
    var hsplit = List[Float32](length=M * ldc, fill=0)
    host.download(_hp(hsplit), hC, M * ldc)
    var sd = _count_diffs(hsplit, split_dev)
    if sd != 0:
        print("SPLITDIFF", name, "words", sd)
    tally.split_diffs += sd
    tally.split_cases += 1
    if split_on:
        tally.split_hits += 1
    tally.mix_split(split_dev)
    host.free(hA)
    host.free(hB)
    host.free(hC)
    host.free(hp3)
    host.free(hp4)
    # Mojo frees a local at its last named use, and _hp only reads the
    # address: without these the host uploads above read freed lists.
    _ = ha^
    _ = hb^
    _ = hc0^
    _ = h3^
    tally.mix(first)
    tally.mix(first_cs)
    if tally.phase == 0:
        tally.cases += 1
    else:
        tally.cases6 += 1


def _shape_all_kinds(mut dev: DeviceExec, mut host: HostExec, mut tally: GemmTally, M: Int, N: Int, K: Int) raises:
    for t in range(2):  # driver-loop: over the two layouts
        var tr = t == 1
        _run[OP_GEMM](dev, host, tally, M, N, K, tr, 0, False)
        _run[OP_GEMM](dev, host, tally, M, N, K, tr, 0, True)
        for k in range(N_KINDS):  # driver-loop: over the epilogue kinds
            _run[OP_AGK_GEMM_EPI](dev, host, tally, M, N, K, tr, k, False)
            _run[OP_GEMM_EPI](dev, host, tally, M, N, K, tr, k, False)


def run_gemm_cases(mut dev: DeviceExec, mut host: HostExec) raises:
    var tally = GemmTally()
    # the HIGGS MLP's five GEMMs a batch (Z1, Z2, dW2, dH, dW1), every kind
    _shape_all_kinds(dev, host, tally, 512, 256, 28)
    _shape_all_kinds(dev, host, tally, 512, 1, 256)
    _shape_all_kinds(dev, host, tally, 256, 1, 512)
    _shape_all_kinds(dev, host, tally, 512, 256, 1)
    _shape_all_kinds(dev, host, tally, 28, 256, 512)
    # the neighbours: each shape on both layouts as a plain and an
    # accumulating OP_GEMM, and one epilogue kind a shape on each epilogue
    # opcode (the kinds rotate over the grid)
    var ms: List[Int] = [1, 17, 300]
    var ns: List[Int] = [1, 2, 3, 16, 17, 64, 65]
    var ks: List[Int] = [1, 2, 27, 28, 29, 255, 256, 257]
    var idx = 0
    var routes_seen = List[Int](length=6, fill=0)
    for i in range(len(ms)):  # driver-loop: over the shape grid
        for j in range(len(ns)):
            for l in range(len(ks)):
                var M = ms[i]
                var N = ns[j]
                var K = ks[l]
                routes_seen[seq_gemm_route(M, N, K, True)] += 1
                _run[OP_GEMM](dev, host, tally, M, N, K, False, 0, False)
                _run[OP_GEMM](dev, host, tally, M, N, K, True, 0, False)
                _run[OP_GEMM](dev, host, tally, M, N, K, False, 0, True)
                _run[OP_AGK_GEMM_EPI](dev, host, tally, M, N, K, False, idx % N_KINDS, False)
                _run[OP_GEMM_EPI](dev, host, tally, M, N, K, True, (idx + 3) % N_KINDS, False)
                idx += 1
    # AMD opbench (Session.linear bias_softmax 65536 x 256 -> 256): the large
    # fused launch on both epilogue opcodes, every route
    _run[OP_AGK_GEMM_EPI](dev, host, tally, 65536, 256, 256, False, 10, False)
    _run[OP_GEMM_EPI](dev, host, tally, 65536, 256, 256, False, 10, False)
    # lane ag-pass5-gemm: the staging boundaries. gemm_dot stages 64 terms a
    # round then guarded rounds of 16 (GEMM_STAGE_V2; 16 and a serial tail
    # under MOJOLEARN_IDN_GEMM_STAGE_OFF), and the 16 x 16 tile's slabs are
    # 128 terms (GT_K): K on both sides of 16, 64 and 128, on the cell
    # (M = 1 or N = 1 bare products), tile16 (17 x 33) and thin routes, every
    # route compared word for word as above
    var bms: List[Int] = [1, 17]
    var bns: List[Int] = [1, 16, 33]
    var bks: List[Int] = [15, 16, 17, 63, 64, 65, 127, 128, 129]
    for i in range(len(bms)):  # driver-loop: over the boundary shape grid
        for j in range(len(bns)):
            for l in range(len(bks)):
                var M = bms[i]
                var N = bns[j]
                var K = bks[l]
                routes_seen[seq_gemm_route(M, N, K, True)] += 1
                _run[OP_GEMM](dev, host, tally, M, N, K, False, 0, False)
                _run[OP_GEMM](dev, host, tally, M, N, K, True, 0, True)
                _run[OP_AGK_GEMM_EPI](dev, host, tally, M, N, K, False, idx % N_KINDS, False)
                _run[OP_GEMM_EPI](dev, host, tally, M, N, K, True, (idx + 3) % N_KINDS, False)
                idx += 1
    # lane ag-pass6-gemm: the pass-6 shapes (digest6). The micro-tile's
    # edges (63, 64, 65, 128 against its 64 x 64 block) and slab boundaries
    # (K on both sides of its 32-term slab and of gemm_dot's 64-term round),
    # every route forced as above; then long-K products on both sides of
    # the split rule's cell bounds (2^15 thin, 2^14 with both edges >= 8)
    # and of its block counts (S = 2, a non-power-of-two S, S = 256 lanes).
    tally.phase = 1
    var mms: List[Int] = [63, 64, 65, 128]
    var mns: List[Int] = [63, 64, 65]
    var mks: List[Int] = [15, 31, 32, 33, 64, 65, 129]
    for i in range(len(mms)):  # driver-loop: over the micro-tile shape grid
        for j in range(len(mns)):
            for l in range(len(mks)):
                var M = mms[i]
                var N = mns[j]
                var K = mks[l]
                routes_seen[seq_gemm_route(M, N, K, True)] += 1
                _run[OP_GEMM](dev, host, tally, M, N, K, False, 0, False)
                _run[OP_GEMM](dev, host, tally, M, N, K, True, 0, True)
                _run[OP_AGK_GEMM_EPI](dev, host, tally, M, N, K, False, idx % N_KINDS, False)
                _run[OP_GEMM_EPI](dev, host, tally, M, N, K, True, (idx + 3) % N_KINDS, False)
                idx += 1
    var sms: List[Int] = [1, 3, 17, 300, 1, 127, 128, 181]
    var sns: List[Int] = [1, 5, 33, 1, 300, 128, 128, 181]
    var sks: List[Int] = [65537, 4097, 16385, 1000, 777, 2048, 129, 300]
    for i in range(len(sms)):  # driver-loop: over the split-K shapes
        var M = sms[i]
        var N = sns[i]
        var K = sks[i]
        routes_seen[seq_gemm_route(M, N, K, True)] += 1
        _run[OP_GEMM](dev, host, tally, M, N, K, False, 0, False)
        _run[OP_GEMM](dev, host, tally, M, N, K, True, 0, True)
        _run[OP_AGK_GEMM_EPI](dev, host, tally, M, N, K, False, 5, False)
        _run[OP_GEMM_EPI](dev, host, tally, M, N, K, False, idx % N_KINDS, False)
        idx += 1
    print(
        "ROUTES grid cell", routes_seen[GR_CELL], "tile", routes_seen[GR_TILE],
        "thinN", routes_seen[GR_THIN_N], "thinM", routes_seen[GR_THIN_M],
    )
    if tally.host_diffs != 0:
        print("HOSTDIFF total words", tally.host_diffs, "(reported, not required)")
    print(
        "SPLIT cases", tally.split_cases, "dispatch_hits", tally.split_hits,
        "device_vs_host_diffs", tally.split_diffs, "split_digest", tally.split_digest,
    )
    if tally.split_diffs != 0:
        raise Error("gemm_cases: the split-K order differs device vs host (" + String(tally.split_diffs) + " words)")
    print("PASS gemm_cases", tally.cases, "launches", tally.launches, "digest", tally.digest)
    print("PASS gemm_cases6", tally.cases6, "digest6", tally.digest6)
