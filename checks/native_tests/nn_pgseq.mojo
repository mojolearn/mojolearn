# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Parameter groups of `Sequential.compile` (lane nn-blocks-pgseq, 2026-11-25; autograd/nn_pgseq.mojo, the grouped
branch of autograd/model.mojo model_fit_state).

The model: 3 inputs -> linear(4) -> linear(1), bias on both (tensors 0.weight, 0.bias, 1.weight, 1.bias; 21
parameters), MSE on 8 fixed rows, batch 4, no shuffle unless said.

Units (each independent; the CPU column runs them across host cores on HostExec, the device column each on
DeviceExec in turn):
  0-5  per optimizer (momentum + nesterov, adam, adamw, centered rmsprop with momentum, adagrad with lr_decay and
       an initial accumulator, lamb with its global clip and weight decay): two groups (layer 0, layer 1) and the
       default group, every group with the ungrouped settings: three epochs give, word for word, the ungrouped
       fit's parameters (the groups change nothing but the runs);
  6    the head at 10x lr: SGD without momentum, one step over all 8 rows, layer 1 at 0.5 and layer 0 at 0.05:
       every parameter equals the float64 reference step p - lr_g dL/dp (tolerance 1e-5 relative + 1e-7);
  7    resume: AdamW with the head at its own lr and betas, shuffled: 2 epochs then 2 resumed epochs give the
       words (parameters and both moment buffers) of 4 epochs (each group's beta products rebuilt from the step
       count);
  8    no_decay: AdamW (weight_decay 0.1) with no_decay "bias" gives the words of explicit groups that list the
       biases with weight_decay 0;
  9    refusals and the table: a name the model does not have, a layer without parameters, a tensor in two
       groups, reduce_on_plateau, group words that do not match the model's tensors; and no_decay's twin indices.
"""
from std.memory import bitcast
from std.testing import TestSuite
from autograd.model import FitConfig, LOSS_MSE, TARGET_DENSE, parse_model, model_fit, model_fit_state, IP64, FIT_STATE_WORDS, ModelSpec
from autograd.model_checkpoint import model_layout
from autograd.nn_optim import NN_OPT_LAMB, NN_SCHED_PLATEAU
from autograd.nn_pgseq import PGSEQ_GROUP_WORDS, PgSeqTable, pgseq_admit_schedule, pgseq_resolve, pgseq_table
from sequence.recurrent import OptConfig
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_ADAGRAD
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require

comptime PGSEQ_FIXTURES = 10
comptime N = 8
comptime D = 3
comptime H = 4
comptime NP = D * H + H + H + 1


def _program() -> List[Int]:
    var program: List[Int] = [1, D, 2, 1, H, 1, 0, 1, 1, 1, 0]
    return program^


def _spec() raises -> ModelSpec:
    var table: List[Float32] = [0, 0]
    return parse_model(_program(), table)


def _x() -> List[Float32]:
    var x = List[Float32]()
    for i in range(N * D):
        x.append(Float32(((i * 7) % 11) - 5) * Float32(0.125))
    return x^


def _y() -> List[Float32]:
    var y = List[Float32]()
    for i in range(N):
        y.append(Float32((i % 3) - 1) * Float32(0.5))
    return y^


def _p0() -> List[Float32]:
    var p = List[Float32]()
    for i in range(NP):
        p.append(Float32(((i * 5) % 9) - 4) * Float32(0.0625))
    return p^


def _opt(fixture: Int) -> OptConfig:
    if fixture == 0:
        return OptConfig(OPT_SGD, 1, Float32(0.9), Float32(0), Float32(1e-8), Float32(0.01), Float32(0))
    if fixture == 1:
        return OptConfig(OPT_ADAM, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.01), Float32(0))
    if fixture == 2:
        return OptConfig(OPT_ADAMW, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.01), Float32(0))
    if fixture == 3:
        return OptConfig(OPT_RMSPROP, 1, Float32(0), Float32(0.99), Float32(1e-8), Float32(0.01), Float32(0.5))
    if fixture == 4:
        return OptConfig(OPT_ADAGRAD, 0, Float32(0.01), Float32(0.1), Float32(1e-10), Float32(0.01), Float32(0))
    # lamb: grad averaging, bias correction, the global clip at 0.05 (it binds: the coefficient is above one)
    return OptConfig(NN_OPT_LAMB, 4 | 8 | 16, Float32(0.9), Float32(0.999), Float32(1e-6), Float32(0.01),
                     Float32(0.05))


def _words(c: OptConfig, lr: Float64) -> List[Float64]:
    var w: List[Float64] = [lr, Float64(c.flags), Float64(c.f1), Float64(c.f2), Float64(c.eps), Float64(c.wd),
                            Float64(c.f7), 0.0]
    return w^


def _layout(spec: ModelSpec, mut offsets: List[Int], mut names: List[String]):
    model_layout(spec, offsets, names)


def _cfg(opt: OptConfig, lr: Float32, epochs: Int, batch: Int, shuffle: Bool, offsets: List[Int]) -> FitConfig:
    return FitConfig(LOSS_MSE, TARGET_DENSE, opt, lr, 0, 1, Float32(0.1), batch, epochs, shuffle, UInt64(3), 0, 0,
                     Float32(1), False, List[Float64](), offsets.copy())


def _groups(spec: ModelSpec, kind: Int, layers: List[List[Int]], names: List[List[String]], no_decay: List[String],
            words: List[Float64]) raises -> PgSeqTable:
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var fp = pgseq_resolve(lnames, layers, names, no_decay, words)
    return pgseq_table(fp, 0, kind, offsets, spec.n_params)


def _fit[E: Exec](mut ex: E, spec: ModelSpec, cfg: FitConfig, var params: List[Float32]) raises -> List[Float32]:
    var x = _x()
    var y = _y()
    var curve = List[Float32](length=cfg.epochs, fill=0)
    _ = model_fit(ex, spec, cfg, CastIn.f32_c(Int(x.unsafe_ptr()), N, D), CastIn.f32_c(Int(y.unsafe_ptr()), N, 1),
                  N, f32_ptr(Int(params.unsafe_ptr())), f32_ptr(Int(curve.unsafe_ptr())))
    ex.sync()
    _ = x^
    _ = y^
    _ = curve^
    return params^


def _same(what: String, a: List[Float32], b: List[Float32]) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                what + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _one(v: Int) -> List[Int]:
    var out = List[Int]()
    out.append(v)
    return out^


def _strs(a: String, b: String = "") -> List[String]:
    var out = List[String]()
    out.append(a)
    if b != "":
        out.append(b)
    return out^


def _no_layers(n: Int) -> List[List[Int]]:
    var out = List[List[Int]]()
    for _ in range(n):
        out.append(List[Int]())
    return out^


def _no_names(n: Int) -> List[List[String]]:
    var out = List[List[String]]()
    for _ in range(n):
        out.append(List[String]())
    return out^


def run_equal[E: Exec](mut ex: E, fixture: Int) raises:
    var spec = _spec()
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var opt = _opt(fixture)
    var lr = Float32(0.01)
    var plain = _fit(ex, spec, _cfg(opt, lr, 3, 4, False, offsets), _p0())
    var layers = List[List[Int]]()
    layers.append(_one(0))
    layers.append(_one(1))
    var words = List[Float64]()
    for _ in range(3):
        words.extend(_words(opt, Float64(lr)))
    var cfg = _cfg(opt, lr, 3, 4, False, offsets)
    cfg.groups = _groups(spec, opt.kind, layers, _no_names(2), List[String](), words)
    require(cfg.groups.runs() == 2, "two runs (layer 0's tensors, layer 1's)")
    var grouped = _fit(ex, spec, cfg, _p0())
    _same("groups with the ungrouped settings (optimizer " + String(opt.kind) + ")", grouped, plain)
    var moved = False
    var start = _p0()
    for i in range(NP):
        moved = moved or bitcast[DType.uint32](grouped[i]) != bitcast[DType.uint32](start[i])
    require(moved, "the fit did not move the parameters")


def run_head_lr[E: Exec](mut ex: E) raises:
    var spec = _spec()
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var opt = OptConfig(OPT_SGD, 0, Float32(0), Float32(0), Float32(1e-8), Float32(0), Float32(0))
    var layers = List[List[Int]]()
    layers.append(_one(1))
    var words = _words(opt, 0.5)
    words.extend(_words(opt, 0.05))
    var cfg = _cfg(opt, Float32(0.05), 1, N, False, offsets)
    cfg.groups = _groups(spec, OPT_SGD, layers, _no_names(1), List[String](), words)
    var got = _fit(ex, spec, cfg, _p0())
    # the float64 reference: h = x W0 + b0, yhat = h W1 + b1, L = mean (yhat - y)^2 over the 8 rows
    var x = _x()
    var y = _y()
    var p = _p0()
    var g = List[Float64](length=NP, fill=0.0)
    var w1 = D * H + H
    for r in range(N):
        var h = List[Float64](length=H, fill=0.0)
        for j in range(H):
            var acc = Float64(p[D * H + j])
            for i in range(D):
                acc += Float64(x[r * D + i]) * Float64(p[i * H + j])
            h[j] = acc
        var yhat = Float64(p[w1 + H])
        for j in range(H):
            yhat += h[j] * Float64(p[w1 + j])
        var dy = 2.0 * (yhat - Float64(y[r])) / Float64(N)
        g[w1 + H] += dy
        for j in range(H):
            g[w1 + j] += dy * h[j]
            var dh = dy * Float64(p[w1 + j])
            g[D * H + j] += dh
            for i in range(D):
                g[i * H + j] += dh * Float64(x[r * D + i])
    for k in range(NP):
        var rate = 0.5 if k >= w1 else 0.05
        var want = Float64(p[k]) - rate * g[k]
        var diff = abs(Float64(got[k]) - want)
        require(diff <= 1e-7 + 1e-5 * abs(want), "parameter " + String(k) + ": " + String(got[k])
                + " vs the float64 step " + String(want))


def run_resume[E: Exec](mut ex: E) raises:
    var spec = _spec()
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var opt = OptConfig(OPT_ADAMW, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.01), Float32(0))
    var head = OptConfig(OPT_ADAMW, 0, Float32(0.8), Float32(0.99), Float32(1e-8), Float32(0.0), Float32(0))
    var layers = List[List[Int]]()
    layers.append(_one(1))
    var words = _words(head, 0.02)
    words.extend(_words(opt, 0.005))
    var x = _x()
    var y = _y()
    var X = CastIn.f32_c(Int(x.unsafe_ptr()), N, D)
    var Y = CastIn.f32_c(Int(y.unsafe_ptr()), N, 1)
    # four epochs in one fit
    var cfg4 = _cfg(opt, Float32(0.005), 4, 4, True, offsets)
    cfg4.groups = _groups(spec, OPT_ADAMW, layers, _no_names(1), List[String](), words)
    var pa = _p0()
    var ma = List[Float32](length=NP, fill=0)
    var va = List[Float32](length=NP, fill=0)
    var wa = List[Float32](length=NP, fill=0)  # lane stream-fits: the third buffer (centered RMSprop's momentum)
    var sa = List[Int64](length=FIT_STATE_WORDS, fill=0)
    var ca = List[Float32](length=4, fill=0)
    _ = model_fit_state(ex, spec, cfg4, X, Y, N, f32_ptr(Int(pa.unsafe_ptr())), f32_ptr(Int(ca.unsafe_ptr())),
                        f32_ptr(Int(ma.unsafe_ptr())), f32_ptr(Int(va.unsafe_ptr())),
                        IP64(unsafe_from_address=Int(sa.unsafe_ptr())), f32_ptr(Int(ca.unsafe_ptr())), False,
                        Wio=Int(wa.unsafe_ptr()))
    # two, then two resumed
    var cfg2 = _cfg(opt, Float32(0.005), 2, 4, True, offsets)
    cfg2.groups = _groups(spec, OPT_ADAMW, layers, _no_names(1), List[String](), words)
    var pb = _p0()
    var mb = List[Float32](length=NP, fill=0)
    var vb = List[Float32](length=NP, fill=0)
    var wb = List[Float32](length=NP, fill=0)
    var sb = List[Int64](length=FIT_STATE_WORDS, fill=0)
    var c1 = List[Float32](length=2, fill=0)
    var c2 = List[Float32](length=4, fill=0)
    _ = model_fit_state(ex, spec, cfg2, X, Y, N, f32_ptr(Int(pb.unsafe_ptr())), f32_ptr(Int(c1.unsafe_ptr())),
                        f32_ptr(Int(mb.unsafe_ptr())), f32_ptr(Int(vb.unsafe_ptr())),
                        IP64(unsafe_from_address=Int(sb.unsafe_ptr())), f32_ptr(Int(c1.unsafe_ptr())), False,
                        Wio=Int(wb.unsafe_ptr()))
    _ = model_fit_state(ex, spec, cfg2, X, Y, N, f32_ptr(Int(pb.unsafe_ptr())), f32_ptr(Int(c2.unsafe_ptr())),
                        f32_ptr(Int(mb.unsafe_ptr())), f32_ptr(Int(vb.unsafe_ptr())),
                        IP64(unsafe_from_address=Int(sb.unsafe_ptr())), f32_ptr(Int(c1.unsafe_ptr())), True,
                        Wio=Int(wb.unsafe_ptr()))
    ex.sync()
    _same("grouped fit 2 + 2 resumed epochs vs 4 (parameters)", pb, pa)
    _same("grouped fit 2 + 2 resumed epochs vs 4 (first moments)", mb, ma)
    _same("grouped fit 2 + 2 resumed epochs vs 4 (second moments)", vb, va)
    _same("grouped fit 2 + 2 resumed epochs vs 4 (loss curve)", c2, ca)
    _same("grouped fit 2 + 2 resumed epochs vs 4 (third buffers)", wb, wa)
    _ = wa^
    _ = wb^
    _ = x^
    _ = y^
    _ = sa^
    _ = sb^
    _ = c1^


def run_no_decay[E: Exec](mut ex: E) raises:
    var spec = _spec()
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var opt = OptConfig(OPT_ADAMW, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.1), Float32(0))
    var zero = OptConfig(OPT_ADAMW, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0.0), Float32(0))
    # no_decay "bias": the default group and its twin
    var a = _cfg(opt, Float32(0.01), 3, 4, False, offsets)
    var w = _words(opt, 0.01)
    w.extend(_words(zero, 0.01))
    a.groups = _groups(spec, OPT_ADAMW, _no_layers(0), _no_names(0), _strs("bias"), w)
    var by_suffix = _fit(ex, spec, a, _p0())
    # explicit: the biases in a group with weight_decay 0
    var names = List[List[String]]()
    names.append(_strs("0.bias", "1.bias"))
    var e = _words(zero, 0.01)
    e.extend(_words(opt, 0.01))
    var b = _cfg(opt, Float32(0.01), 3, 4, False, offsets)
    b.groups = _groups(spec, OPT_ADAMW, _no_layers(1), names, List[String](), e)
    var explicit = _fit(ex, spec, b, _p0())
    _same("no_decay='bias' vs the biases in a weight_decay 0 group", by_suffix, explicit)


def _refused(what: String, words: String, err: Error) raises:
    require(words in String(err), what + ": refusal words '" + words + "' not in: " + String(err))


def run_refusals() raises:
    var spec = _spec()
    var offsets = List[Int]()
    var lnames = List[String]()
    _layout(spec, offsets, lnames)
    var opt = _opt(2)
    var w2 = _words(opt, 0.01)
    w2.extend(_words(opt, 0.01))
    var w3 = w2.copy()
    w3.extend(_words(opt, 0.01))
    var names = List[List[String]]()
    names.append(_strs("5.weight"))
    try:
        _ = pgseq_resolve(lnames, _no_layers(1), names, List[String](), w2)
        require(False, "an unknown name was accepted")
    except err:
        _refused("unknown name", "the model has no parameter '5.weight'", err)
    var bad = List[List[Int]]()
    bad.append(_one(5))
    try:
        _ = pgseq_resolve(lnames, bad, _no_names(1), List[String](), w2)
        require(False, "a layer without parameters was accepted")
    except err:
        _refused("layer without parameters", "layer 5 has no parameters", err)
    var twice = List[List[Int]]()
    twice.append(_one(0))
    twice.append(_one(0))
    try:
        _ = pgseq_resolve(lnames, twice, _no_names(2), List[String](), w3)
        require(False, "a tensor in two groups was accepted")
    except err:
        _refused("tensor in two groups", "a parameter belongs to one group", err)
    try:
        pgseq_admit_schedule(NN_SCHED_PLATEAU)
        require(False, "reduce_on_plateau was accepted")
    except err:
        _refused("plateau", "reduce_on_plateau", err)
    var damaged = List[Float64]()
    damaged.append(1.0)
    damaged.append(3.0)
    damaged.append(0.0)
    damaged.extend(_words(opt, 0.01))
    for _ in range(3):
        damaged.append(0.0)
    try:
        _ = pgseq_table(damaged, 0, OPT_ADAMW, offsets, spec.n_params)
        require(False, "words for 3 tensors were accepted for a 4-tensor model")
    except err:
        _refused("mismatch", "do not match", err)
    # no_decay twins: one user group (layer 1), the default (index 1), twins 2 (layer 1's) and 3 (the default's)
    var head = List[List[Int]]()
    head.append(_one(1))
    var w4 = w2.copy()
    w4.extend(w2)
    var fp = pgseq_resolve(lnames, head, _no_names(1), _strs("bias"), w4)
    require(Int(fp[0]) == 4 and Int(fp[1]) == 4 and Int(fp[2]) == 1, "the header [G, T, D]")
    var at = 3 + 4 * PGSEQ_GROUP_WORDS
    require(Int(fp[at]) == 1 and Int(fp[at + 1]) == 3 and Int(fp[at + 2]) == 0 and Int(fp[at + 3]) == 2,
            "tensor groups 0.weight 1, 0.bias 3, 1.weight 0, 1.bias 2")


def run_fixture[E: Exec](mut ex: E, fixture: Int) raises:
    if fixture < 6:
        run_equal(ex, fixture)
    elif fixture == 6:
        run_head_lr(ex)
    elif fixture == 7:
        run_resume(ex)
    elif fixture == 8:
        run_no_decay(ex)
    else:
        run_refusals()


# ---- The host and device columns ----
def test_nn_pgseq() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_fixture(fixture_id: Int) raises:
            var ex = HostExec()
            run_fixture(ex, fixture_id)
        check_cases("nn_pgseq", PGSEQ_FIXTURES, host_fixture)
    else:
        def device_fixture(fixture_id: Int) raises:
            var ex = DeviceExec()
            run_fixture(ex, fixture_id)
        check_cases_serial("nn_pgseq", PGSEQ_FIXTURES, device_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
