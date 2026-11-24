# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Resident label pairs and a resident X_embedded (lane vdf-xm-wire,
2026-11-24) against the host-input route and the CPU column, byte for byte.

The device route a door takes inside `resident()`: both label vectors are
slots of the ONE store (x_prep/store_slot.mojo), the device union step
(MS_ENCODE_PAIR: concatenate, encode once, codes in two slots) then the
metric entry over the code slots. The host-input route: the host union
encoder (core/metrics_cluster_table.mojo `cluster_union_codes`) and the GPU
image's body over host words. The CPU column: the host image's bodies /
oracles over the same codes. All three must give the same bits.

  0 cluster ops     MI, FM, homogeneity, completeness, v-measure (beta 1.5),
                    the fused three; signed, sparse int32 labels, 70,000 rows;
                    k and the union classes equal the host encoder's
  1 int64 labels    the same values as 8-byte labels: same k, same MI
  2 confusion       MS_CONFUSION over the code slots vs host input vs
                    host_confusion_matrix_i64
  3 prfs + gather   the codes mapped through a permuted class order on the
                    device (MS_GATHER) vs the same map on the host, then
                    MS_PRFS (macro) vs host input vs host_precision_recall_fscore
  4 trustworthiness X and X_embedded both slots, X_embedded only a slot, vs
                    host input vs host_trustworthiness
  5 refusal         a NaN in a resident X_embedded is refused on the device
                    with the door's words
  6 confusion with labels (lane vdf-xm-wire2): the door's label check on the
                    device (y_true's codes through the miss table: a hit for
                    a present selected class, -1 for none), the codes mapped
                    to a labels order with classes outside it (-1) by
                    MS_GATHER, then MS_CONFUSION vs host input vs
                    host_confusion_matrix_i64

What a sabotage changes here: a union step that encodes each vector alone
(different codes, case 0 k check), a code slot copied from the wrong half,
a gather through the wrong table, a k-NN index read from the wrong slot.
"""
from std.testing import TestSuite
from std.memory import bitcast

from x_prep.store_slot import X_PREP_STORE, x_prep_ctx
from metrics.slot_inputs import metric_slot_create
from metrics.slot_entry_device import metric_slot_run
from metrics.slot_ops import (
    MS_ENCODE_PAIR, MS_GATHER, MS_MI, MS_FM, MS_HOMOG, MS_COMPL, MS_V_MEASURE, MS_HCV, MS_CONFUSION, MS_PRFS,
    MS_TRUST,
)
from core.cast_in import CAST_KIND_SIGNED
from core.labels_score_encode import score_encode_signed
from core.metrics_cluster_table import (
    CLUSTER_OP_MUTUAL_INFO, CLUSTER_OP_FOWLKES_MALLOWS, CLUSTER_OP_HOMOGENEITY, CLUSTER_OP_COMPLETENESS,
    CLUSTER_OP_V_MEASURE, CLUSTER_OP_HCV, ClusterFloatPtr, cluster_union_codes,
)
from metrics.cluster_entry_device import metrics_cluster_device_run
from metrics.host.cluster_entry_host import metrics_cluster_host_run
from metrics.host.classification_oracle import (
    host_confusion_matrix_i64, host_precision_recall_fscore, host_trustworthiness,
)
from checks.native_tests.testsuite_cases import require

comptime CASES = 7
comptime N = 70000


def _put_words(addr: Int, words: Int) raises -> Int:
    return X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), addr, words)


def _free(id: Int) raises:
    X_PREP_STORE.get_or_create_ptr()[].free(x_prep_ctx(), id)


def _labels(seed: Int, classes: Int) -> List[Int32]:
    # signed, sparse values (not 0 .. k-1): the encode has work to do
    var out = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # fixture
        out[i] = Int32((((i * 7919 + seed * 104729) // 7) % classes) * 37 - 500)
    return out^


def _same(name: String, got: Float64, want: Float64) raises:
    require(bitcast[DType.uint64](got) == bitcast[DType.uint64](want),
            name + ": " + String(got) + " vs " + String(want) + " (bits differ)")


def _run(op: Int, addrs: List[Int], slots: List[Int], ints: List[Int], floats: List[Float64],
         outs: List[Int]) raises -> Tuple[Int, List[Float64]]:
    var values = List[Float64]()
    var count = metric_slot_run(op, addrs, slots, ints, floats, outs, values)
    return (count, values^)


struct _DevicePair(Movable):
    """The device union step over two resident label slots."""
    var st: Int
    var sp: Int
    var ct: Int
    var cp: Int
    var k: Int

    def __init__(out self, t_addr: Int, p_addr: Int, item: Int, classes_addr: Int) raises:
        var words = N * item // 4
        self.st = _put_words(t_addr, words)
        self.sp = _put_words(p_addr, words)
        self.ct = metric_slot_create(N)
        self.cp = metric_slot_create(N)
        var addrs: List[Int] = [0, 0]
        var slots: List[Int] = [self.st, self.sp]
        var ints: List[Int] = [N, CAST_KIND_SIGNED, item, self.ct, self.cp]
        var outs: List[Int] = [classes_addr]
        self.k = _run(MS_ENCODE_PAIR, addrs, slots, ints, List[Float64](), outs)[0]

    def free(self) raises:
        _free(self.cp)
        _free(self.ct)
        _free(self.sp)
        _free(self.st)


def _cluster_ms(op: Int) -> Int:
    if op == CLUSTER_OP_MUTUAL_INFO:
        return MS_MI
    if op == CLUSTER_OP_FOWLKES_MALLOWS:
        return MS_FM
    if op == CLUSTER_OP_HOMOGENEITY:
        return MS_HOMOG
    if op == CLUSTER_OP_COMPLETENESS:
        return MS_COMPL
    if op == CLUSTER_OP_V_MEASURE:
        return MS_V_MEASURE
    return MS_HCV


def _cluster_case() raises:
    var yt = _labels(1, 13)
    var yp = _labels(2, 9)
    var codes = List[Int32]()
    var k_host = cluster_union_codes(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), N, codes)
    # the union classes of the host encoder, for the class-list check
    var both = yt.copy()
    for i in range(N):  # fixture: the concatenation the union encoder reads
        both.append(yp[i])
    var host_classes = List[Int32](length=2 * N, fill=Int32(0))
    var host_codes = List[Int32](length=2 * N, fill=Int32(0))
    _ = score_encode_signed(Int(both.unsafe_ptr()), 4, 2 * N, Int(host_classes.unsafe_ptr()),
                            Int(host_codes.unsafe_ptr()))
    var dev_classes = List[Int32](length=2 * N, fill=Int32(0))
    var dev = _DevicePair(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, Int(dev_classes.unsafe_ptr()))
    try:
        require(dev.k == k_host, "union k: device " + String(dev.k) + " vs host " + String(k_host))
        for c in range(dev.k):  # driver-loop: the k union classes
            require(dev_classes[c] == host_classes[c], "union class " + String(c) + " differs")
        var ops: List[Int] = [CLUSTER_OP_MUTUAL_INFO, CLUSTER_OP_FOWLKES_MALLOWS, CLUSTER_OP_HOMOGENEITY,
                              CLUSTER_OP_COMPLETENESS, CLUSTER_OP_V_MEASURE, CLUSTER_OP_HCV]
        for j in range(len(ops)):  # driver-loop: the six union-code operations
            var op = ops[j]
            var beta = 1.5 if op == CLUSTER_OP_V_MEASURE or op == CLUSTER_OP_HCV else 1.0
            var floats = List[Float64]()
            if op == CLUSTER_OP_V_MEASURE or op == CLUSTER_OP_HCV:
                floats.append(beta)
            var addrs: List[Int] = [0, 0]
            var slots: List[Int] = [dev.ct, dev.cp]
            var ints: List[Int] = [N, 0, dev.k - 1]
            var device = _run(_cluster_ms(op), addrs, slots, ints, floats, List[Int]())[1].copy()
            var base = Int(codes.unsafe_ptr())
            var host_in = List[Float64](length=3, fill=0.0)
            var cpu = List[Float64](length=3, fill=0.0)
            var count = metrics_cluster_device_run(op, base, base + 4 * N, N, 0, k_host - 1, beta,
                                                   ClusterFloatPtr(unsafe_from_address=Int(host_in.unsafe_ptr())))
            _ = metrics_cluster_host_run(op, base, base + 4 * N, N, 0, k_host - 1, beta,
                                         ClusterFloatPtr(unsafe_from_address=Int(cpu.unsafe_ptr())))
            for v in range(count):  # driver-loop: one or three values
                _same("cluster op " + String(op) + " value " + String(v) + " slot vs host input", device[v],
                      host_in[v])
                _same("cluster op " + String(op) + " value " + String(v) + " slot vs CPU column", device[v], cpu[v])
    finally:
        dev.free()
    _ = len(codes)
    _ = len(yt)
    _ = len(yp)


def _mi_of(dev: _DevicePair) raises -> Float64:
    var addrs: List[Int] = [0, 0]
    var slots: List[Int] = [dev.ct, dev.cp]
    var ints: List[Int] = [N, 0, dev.k - 1]
    return _run(MS_MI, addrs, slots, ints, List[Float64](), List[Int]())[1][0]


def _int64_case() raises:
    var yt = _labels(3, 11)
    var yp = _labels(4, 7)
    var wt = List[Int64](length=N, fill=Int64(0))
    var wp = List[Int64](length=N, fill=Int64(0))
    for i in range(N):  # fixture: the same values at 8 bytes
        wt[i] = Int64(yt[i])
        wp[i] = Int64(yp[i])
    var c4 = List[Int32](length=2 * N, fill=Int32(0))
    var c8 = List[Int64](length=2 * N, fill=Int64(0))
    var d4 = _DevicePair(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, Int(c4.unsafe_ptr()))
    var d8 = _DevicePair(Int(wt.unsafe_ptr()), Int(wp.unsafe_ptr()), 8, Int(c8.unsafe_ptr()))
    try:
        require(d4.k == d8.k, "int64 labels: k differs")
        for c in range(d4.k):  # driver-loop: the k classes at both widths
            require(Int64(c4[c]) == c8[c], "int64 labels: class " + String(c) + " differs")
        _same("int64 vs int32 labels, mutual info", _mi_of(d8), _mi_of(d4))
    finally:
        d8.free()
        d4.free()
    _ = len(yt)
    _ = len(yp)
    _ = len(wt)
    _ = len(wp)


def _codes_lists(yt: List[Int32], yp: List[Int32]) raises -> Tuple[List[Int32], List[Int32], Int]:
    var codes = List[Int32]()
    var k = cluster_union_codes(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), N, codes)
    var t = List[Int32](length=N, fill=Int32(0))
    var p = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # fixture: the two halves of the host union codes
        t[i] = codes[i]
        p[i] = codes[N + i]
    return (t^, p^, k)


def _confusion_case() raises:
    var yt = _labels(5, 6)
    var yp = _labels(6, 6)
    var host = _codes_lists(yt, yp)
    var k = host[2]
    var dev = _DevicePair(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, 0)
    try:
        var from_slots = List[Int64](length=k * k, fill=Int64(-1))
        var from_host = List[Int64](length=k * k, fill=Int64(-2))
        var ints: List[Int] = [N, k, 0]
        var o1: List[Int] = [Int(from_slots.unsafe_ptr())]
        var o2: List[Int] = [Int(from_host.unsafe_ptr())]
        _ = _run(MS_CONFUSION, [0, 0], [dev.ct, dev.cp], ints, List[Float64](), o1)
        _ = _run(MS_CONFUSION, [Int(host[0].unsafe_ptr()), Int(host[1].unsafe_ptr())], [-1, -1], ints,
                 List[Float64](), o2)
        var cpu = host_confusion_matrix_i64(host[0], host[1], N, k, 0)
        for c in range(k * k):  # driver-loop: the k x k cells
            require(from_slots[c] == from_host[c], "confusion cell " + String(c) + ": slot vs host input")
            require(from_slots[c] == cpu[c], "confusion cell " + String(c) + ": slot vs CPU column")
    finally:
        dev.free()
    _ = len(yt)
    _ = len(yp)


def _prfs_case() raises:
    var yt = _labels(7, 5)
    var yp = _labels(8, 5)
    var host = _codes_lists(yt, yp)
    var k = host[2]
    # a permuted class order (all labels reversed): table[c] = k - 1 - c
    var table = List[Int32](length=k, fill=Int32(0))
    for c in range(k):  # fixture: the O(k) order table
        table[c] = Int32(k - 1 - c)
    var mt = List[Int32](length=N, fill=Int32(0))
    var mp = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # the host map the device gather must equal
        mt[i] = table[Int(host[0][i])]
        mp[i] = table[Int(host[1][i])]
    var dev = _DevicePair(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, 0)
    var gt = metric_slot_create(N)
    var gp = metric_slot_create(N)
    try:
        var outs: List[Int] = [Int(table.unsafe_ptr())]
        var miss_t = _run(MS_GATHER, [0], [dev.ct], [N, k, gt], List[Float64](), outs)[0]
        var miss_p = _run(MS_GATHER, [0], [dev.cp], [N, k, gp], List[Float64](), outs)[0]
        require(miss_t == -1 and miss_p == -1, "gather: no row may miss")
        var width = k + 0  # average None: one value per selected class
        var a = List[Float32](length=3 * width + 3, fill=Float32(-1.0))
        var b = List[Float32](length=3 * width + 3, fill=Float32(-2.0))
        var ints: List[Int] = [N, k, 0, 0, 0, k]
        _ = _run(MS_PRFS, [0, 0], [gt, gp], ints, List[Float64](), [Int(a.unsafe_ptr())])
        _ = _run(MS_PRFS, [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())], [-1, -1], ints, List[Float64](),
                 [Int(b.unsafe_ptr())])
        var cpu = host_precision_recall_fscore(mt, mp, N, k, 0, 0, 0, k)
        for i in range(3 * width + 3):  # driver-loop: every output word
            require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), "prfs word " + String(i) + ": slot vs host input")
            require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](cpu[i]), "prfs word " + String(i) + ": slot vs CPU column")
    finally:
        _free(gp)
        _free(gt)
        dev.free()
    _ = len(yt)
    _ = len(yp)


def _confusion_labels_case() raises:
    var yt = _labels(9, 6)
    var yp = _labels(10, 6)
    var host = _codes_lists(yt, yp)
    var ku = host[2]
    # labels = the union classes 3, 0, 5 of the union order (when present), in that order: table[c] is
    # the index of class c in labels, -1 outside it (`_order_table`); miss[c] = -1 for a selected class
    var kl = 0
    var table = List[Int32](length=ku, fill=Int32(-1))
    var miss = List[Int32](length=ku, fill=Int32(0))
    var none = List[Int32](length=ku, fill=Int32(0))
    var picks: List[Int] = [3, 0, 5]
    for j in range(len(picks)):  # fixture: the O(k) labels tables
        if picks[j] < ku:
            table[picks[j]] = Int32(kl)
            miss[picks[j]] = Int32(-1)
            kl += 1
    require(kl >= 2, "fixture: at least two selected classes")
    var mt = List[Int32](length=N, fill=Int32(0))
    var mp = List[Int32](length=N, fill=Int32(0))
    var first = -1
    for i in range(N):  # the host map and the host label check the device must equal
        mt[i] = table[Int(host[0][i])]
        mp[i] = table[Int(host[1][i])]
        if first < 0 and miss[Int(host[0][i])] == -1:
            first = i
    var dev = _DevicePair(Int(yt.unsafe_ptr()), Int(yp.unsafe_ptr()), 4, 0)
    var hit = metric_slot_create(N)
    var gt = metric_slot_create(N)
    var gp = metric_slot_create(N)
    try:
        var answer = _run(MS_GATHER, [0], [dev.ct], [N, ku, hit], List[Float64](), [Int(miss.unsafe_ptr())])[0]
        require(answer == first, "label check: the first selected y_true row " + String(answer) + " vs " + String(first))
        var empty = _run(MS_GATHER, [0], [dev.ct], [N, ku, hit], List[Float64](), [Int(none.unsafe_ptr())])[0]
        require(empty == -1, "label check: no selected class answers -1")
        var outs: List[Int] = [Int(table.unsafe_ptr())]
        _ = _run(MS_GATHER, [0], [dev.ct], [N, ku, gt], List[Float64](), outs)
        _ = _run(MS_GATHER, [0], [dev.cp], [N, ku, gp], List[Float64](), outs)
        var from_slots = List[Int64](length=kl * kl, fill=Int64(-1))
        var from_host = List[Int64](length=kl * kl, fill=Int64(-2))
        var ints: List[Int] = [N, kl, 0]
        _ = _run(MS_CONFUSION, [0, 0], [gt, gp], ints, List[Float64](), [Int(from_slots.unsafe_ptr())])
        _ = _run(MS_CONFUSION, [Int(mt.unsafe_ptr()), Int(mp.unsafe_ptr())], [-1, -1], ints, List[Float64](),
                 [Int(from_host.unsafe_ptr())])
        var cpu = host_confusion_matrix_i64(mt, mp, N, kl, 0)
        for c in range(kl * kl):  # driver-loop: the cells
            require(from_slots[c] == from_host[c], "labels confusion cell " + String(c) + ": slot vs host input")
            require(from_slots[c] == cpu[c], "labels confusion cell " + String(c) + ": slot vs CPU column")
    finally:
        _free(gp)
        _free(gt)
        _free(hit)
        dev.free()
    _ = len(yt)
    _ = len(yp)


comptime TN = 300
comptime TM = 6
comptime TD = 2


def _trust_fixture(dims: Int, seed: Int) -> List[Float32]:
    var out = List[Float32](length=TN * dims, fill=Float32(0.0))
    for i in range(TN * dims):  # fixture: spread values, few exact ties
        out[i] = Float32(((i * 2654435761 + seed * 40503) % 10007)) / Float32(101.0) - Float32(40.0)
    return out^


def _trust_case() raises:
    var x = _trust_fixture(TM, 1)
    var e = _trust_fixture(TD, 2)
    var sx = _put_words(Int(x.unsafe_ptr()), TN * TM)
    var se = _put_words(Int(e.unsafe_ptr()), TN * TD)
    try:
        var ints: List[Int] = [TN, TM, TD, 5, 512, 0]
        var xa = Int(x.unsafe_ptr())
        var ea = Int(e.unsafe_ptr())
        var both = _run(MS_TRUST, [0, 0], [sx, se], ints, List[Float64](), List[Int]())[1][0]
        var emb_only = _run(MS_TRUST, [xa, 0], [-1, se], ints, List[Float64](), List[Int]())[1][0]
        var host_in = _run(MS_TRUST, [xa, ea], [-1, -1], ints, List[Float64](), List[Int]())[1][0]
        var cpu = host_trustworthiness(x, e, TN, TM, TD, 5, 512)
        _same("trustworthiness both slots vs host input", both, host_in)
        _same("trustworthiness X_embedded slot vs host input", emb_only, host_in)
        _same("trustworthiness slot vs CPU column", both, cpu)
    finally:
        _free(se)
        _free(sx)
    _ = len(x)
    _ = len(e)


def _trust_refusal_case() raises:
    var x = _trust_fixture(TM, 3)
    var e = _trust_fixture(TD, 4)
    e[77] = bitcast[DType.float32](UInt32(0x7FC00000))
    var se = _put_words(Int(e.unsafe_ptr()), TN * TD)
    var raised = False
    var words = String("mojolearn trustworthiness: X_embedded contains NaN or infinity; refused here ")
    words += "(metrics/README.md HAND-OFF ask 3)"
    try:
        var ints: List[Int] = [TN, TM, TD, 5, 512, 0]
        _ = _run(MS_TRUST, [Int(x.unsafe_ptr()), 0], [-1, se], ints, List[Float64](), List[Int]())
    except problem:
        raised = True
        require(String(problem) == words, "refusal words: got '" + String(problem) + "'")
    finally:
        _free(se)
    require(raised, "a NaN in a resident X_embedded was not refused")
    _ = len(x)
    _ = len(e)


def _case(which: Int) raises:
    if which == 0:
        _cluster_case()
    elif which == 1:
        _int64_case()
    elif which == 2:
        _confusion_case()
    elif which == 3:
        _prfs_case()
    elif which == 4:
        _trust_case()
    elif which == 5:
        _trust_refusal_case()
    else:
        _confusion_labels_case()


def test_cases() raises:
    var failed = String("")
    for which in range(CASES):  # driver-loop: over the check's cases, one device queue
        try:
            _case(which)
        except problem:
            failed += "\n      metric_label_pairs_resident row " + String(which) + ": " + String(problem)
    if failed.byte_length() > 0:
        raise Error("metric_label_pairs_resident: cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
