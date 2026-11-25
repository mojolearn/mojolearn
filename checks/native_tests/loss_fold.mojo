# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The generic training blocks of lane rnn-parity-c (2026-11-25): EpochMonitor (autograd/epoch_monitor.mojo) and
LossFold (autograd/loss_fold.mojo).

Units (each independent: the CPU column runs them across host cores, each on its own HostExec; the device column
runs each on DeviceExec and requires the host words):
  0  EpochMonitor: scikit-learn's `loss > best - tol` stall rule on a written sequence (tol 0.1, patience 2:
     1.0 0.95 0.5 0.45 0.44 improves every epoch, stalls at epochs 2, 4 and 5, stops after 5); `resumed` from a
     saved best and stall carries on as the uninterrupted monitor (the generic loop's FS_BEST / FS_STALL words);
     `restart` forgets them; a NaN or infinite loss is refused.
  1  LossFold, the 2-vs-20 case on the executor: two parts' gradients (8 words each) folded with the executor's
     scale and accumulate launches, the parts' denominators 2 and 20. (a) The mean form (each gradient the part's
     mean gradient, the total 22 known): weights 2/22 and 20/22; (b) the numerator form with the total unknown
     (the distributed trainer): weight 1 and one scale by 1/22 after the fold; (c) the numerator form with the
     total known (the partitioned trainer): weight 1/22. Each against the float64 oracle (num_A + num_B) / 22
     (tolerance 1e-6 relative), and far from the average of the means; the host fold of the parts' losses,
     add_mean and add_sum, gives (num_A + num_B) / 22 too; a zero total is refused.
"""
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_SCALE
from autograd.ops import OP_AG_ACCUMULATE
from autograd.epoch_monitor import EpochMonitor
from autograd.loss_fold import LossFold
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


def _fail(message: String) raises:
    print("FAIL LOSS_FOLD", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # small-loop(words of one unit)
        if a[i] != b[i]:
            _fail(name + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, got: Float32, want: Float64, tol: Float64) raises:
    var g = Float64(got)
    if abs(g - want) > tol * max(abs(want), Float64(1.0e-6)):
        _fail(name + ": " + String(g) + " differs from the oracle " + String(want))


# ---------------------------------------------------------------- unit 0: EpochMonitor
def run_monitor() raises -> List[Float32]:
    var losses: List[Float64] = [1.0, 0.95, 0.5, 0.45, 0.44]
    var stalls: List[Int] = [0, 1, 0, 1, 2]
    var words = List[Float32]()
    var m = EpochMonitor(Float64(0.1), 2)
    for e in range(len(losses)):  # small-loop(5 written epochs)
        if m.stop():
            _fail("monitor stopped before epoch " + String(e + 1))
        var improved = m.observe(losses[e], e + 1)
        if not improved:
            _fail("epoch " + String(e + 1) + " improves on the best")
        if m.stall != stalls[e]:
            _fail("epoch " + String(e + 1) + ": stall " + String(m.stall) + ", expected " + String(stalls[e]))
        words.append(Float32(m.stall))
    if not m.stop() or m.best_epoch != 5 or m.best != Float64(0.44):
        _fail("after five epochs the monitor stops with best epoch 5 and best 0.44")
    # resumed from the words an earlier fit saved: the same next decision as the monitor that ran on
    var cont = m
    var r = EpochMonitor.resumed(Float64(0.1), 2, Float64(0.44), 1)
    var a = cont.observe(Float64(0.5), 6)
    var b = r.observe(Float64(0.5), 1)
    if a != b or a or cont.stall != 3 or r.stall != 2 or r.best != cont.best or r.best_epoch != 0:
        _fail("a resumed monitor decides as the uninterrupted one (stall counts on, best kept, best epoch per call)")
    r.restart()
    if r.stall != 0 or r.best_epoch != 0 or not r.observe(Float64(5.0), 1):
        _fail("restart forgets the best loss and the stall count")
    var refused = 0
    for k in range(2):  # small-loop(2 non-finite losses)
        var v = Float64(0.0) / Float64(0.0) if k == 0 else Float64(1.0e300) * Float64(1.0e300)
        try:
            _ = r.observe(v, 2)
        except err:
            _ = err
            refused += 1
    if refused != 2:
        _fail("a NaN or infinite monitored loss is refused")
    words.append(Float32(r.stall))
    return words^


# ---------------------------------------------------------------- unit 1: LossFold on the executor
def _part(which: Int, n: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(8 gradient words)
        v[i] = Float32(0.125) * Float32(i + 1) if which == 0 else Float32(-0.0625) * Float32(2 * i + 3)
    return v^


def _fold_on[E: Exec](mut ex: E, fold: LossFold, grads: List[List[Float32]], dens: List[Float32], n: Int
                     ) raises -> List[Float32]:
    """The trainers' fold: each part's gradient scaled by fold.weight and accumulated in part order, then the
    final scale (when not 1); the folded words read back."""
    var totals = ex.alloc(n)
    var incoming = ex.alloc(n)
    var host = List[Float32](length=n, fill=Float32(0))
    ex.upload(totals, FP(unsafe_from_address=Int(host.unsafe_ptr())), n)
    for k in range(len(grads)):  # small-loop(2 parts)
        ex.upload(incoming, FP(unsafe_from_address=Int(grads[k].unsafe_ptr())), n)
        var a = Args()
        a.p0 = incoming
        a.f0 = fold.weight[E](dens[k])
        ex.launch[OP_SCALE](a, n)
        a.p1 = totals
        ex.launch[OP_AG_ACCUMULATE](a, n)
    var tail = fold.final_scale[E]()
    if tail != Float32(1):
        var a = Args()
        a.p0 = totals
        a.f0 = tail
        ex.launch[OP_SCALE](a, n)
    ex.sync()
    var result = List[Float32](length=n, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), totals, n)
    ex.free(incoming)
    ex.free(totals)
    _ = host^
    return result^


def run_fold[E: Exec](mut ex: E) raises -> List[Float32]:
    comptime n = 8
    var dens: List[Float32] = [2.0, 20.0]
    var sums = List[List[Float32]]()      # the parts' numerator gradients
    var means = List[List[Float32]]()     # the parts' mean gradients (numerator / denominator)
    for k in range(2):  # small-loop(2 parts)
        var g = _part(k, n)
        var mg = List[Float32](length=n, fill=Float32(0))
        for i in range(n):  # small-loop(8 words)
            mg[i] = g[i] / dens[k]
        sums.append(g^)
        means.append(mg^)
    var words = List[Float32]()
    var mean_form = _fold_on(ex, LossFold(True, 2, Float32(22)), means, dens, n)
    var tail_form = _fold_on(ex, LossFold(False, 2, Float32(0)), sums, dens, n)
    var known_form = _fold_on(ex, LossFold(False, 2, Float32(22)), sums, dens, n)
    for i in range(n):  # small-loop(8 words)
        var want = (Float64(sums[0][i]) + Float64(sums[1][i])) / Float64(22)
        var mean_of_means = 0.5 * (Float64(sums[0][i]) / 2.0 + Float64(sums[1][i]) / 20.0)
        _close("mean form word " + String(i), mean_form[i], want, 1.0e-6)
        _close("numerator form (final scale) word " + String(i), tail_form[i], want, 1.0e-6)
        _close("numerator form (known total) word " + String(i), known_form[i], want, 1.0e-6)
        if abs(Float64(tail_form[i]) - mean_of_means) <= 1.0e-3 * abs(want):
            _fail("word " + String(i) + " is the average of the means, not sum num / sum den")
        words.append(mean_form[i])
        words.append(tail_form[i])
        words.append(known_form[i])
    # the host fold of the parts' losses (the epoch curve's): mean A 0.5 over 2, mean B 0.25 over 20
    var by_mean = LossFold(True, 0, Float32(0))
    by_mean.add_mean(Float32(0.5), Float32(2))
    by_mean.add_mean(Float32(0.25), Float32(20))
    var by_sum = LossFold(False, 0, Float32(0))
    by_sum.add_sum(Float32(1.0), Float32(2))
    by_sum.add_sum(Float32(5.0), Float32(20))
    _close("add_mean fold", by_mean.mean(), 6.0 / 22.0, 1.0e-6)
    _close("add_sum fold", by_sum.mean(), 6.0 / 22.0, 1.0e-6)
    words.append(by_mean.mean())
    words.append(by_sum.mean())
    var refused = False
    try:
        _ = LossFold(False, 2, Float32(0)).final_scale[E]()
    except err:
        _ = err
        refused = True
    if not refused:
        _fail("a fold whose denominators sum to zero is refused")
    return words^


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    if unit == 0:
        return run_monitor()
    return run_fold(ex)


# ---------------------------------------------------------------- tests
def test_loss_fold() raises:
    comptime U = 2
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL LOSS_FOLD unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, U)
    for u in range(U):  # small-loop(U: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(U):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS LOSS_FOLD 2")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
