# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Every public layer added on 2026-11-22, composed into a small model and FITTED, gives finite answers (lane
mamba2-nan, 2026-11-23).

The shipped recipe `example:seq-mamba2` (python/mojolearn/_verify_recipe.py: (8, 2) -> linear(321) -> softplus ->
ssd_scan(1) -> flatten -> linear(1), mse, sgd lr 0.01, 2 epochs of batch 16 over 32 rows, no shuffle, seed 0)
fitted and then predicted NaN on the CPU column, and `verify --recipe` called the two NaN answers IDENTICAL. The
layer was right (its gradients match finite differences; adam fits the same model): SGD diverged (see _SSD_LR), so
the recipe now has rms_norm after the scan and lr 0.001, and verify reports NaN answers as NOT FINITE. The
layer checks (checks/native_tests/neural_blocks.mojo) run one forward and backward from fixed small parameters, so
they never saw a fit from the model's own init. Here each unit is a model of the kind a user composes, built from
the description words the Sequential door sends (training_sequential_common.SeqModel), and:
  - its seeded init (autograd/model.mojo model_init_params) predicts finite values on the test rows;
  - a fit from that init with the recipe's compile and fit options (mse, sgd 0.01 (0.001 for ssd_scan, see
    _SSD_LR), 2 epochs, batch 16, 32 rows, no shuffle, seed 0) gives a finite loss curve and finite parameters;
  - the fitted model predicts finite values on 16 test rows.
Units: ssd_scan at 1 head (the recipe's model, exactly: softplus -> ssd_scan(1) -> rms_norm, sgd 0.001) and 2
heads, ssd_scan fitted with adam with and without the norm, selective_scan, trapezoid_scan, lm_head,
attention (grouped-query, rotary, causal), rms_norm, gated (multiply of a silu branch), conv1d (Mamba's causal
depthwise conv: padding, groups, narrow), max_pool1d, avg_pool1d, adaptive_avg_pool1d, max_pool2d, avg_pool2d,
batch_norm1d, batch_norm2d, and rnn, lstm, gru with sequence lengths. Each unit prints one line
`SEQ_FINITE <unit> loss <epoch 0> <epoch 1> max|param| <m> max|pred| <m>`.
Data: X standard normal (hashed), y = 0.5 x[0] - x[last] + 0.25 tanh(x[mid]) per sample (the recipe's sequence
generator's target), so the target has the recipe's scale.
Columns: the CPU column runs every unit on HostExec across host cores; the device column runs each unit on DeviceExec
too and requires the host words (loss curve, parameters, predictions).
"""
from std.math import isfinite, tanh
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP
from autograd.model import LOSS_MSE, OUT_RAW, TARGET_DENSE, model_init_params
from checks.native_tests.sequence_port_common import normal
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, L_FLATTEN, L_PERMUTE
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime L_CONV2D = 12
comptime L_MAXPOOL2D = 13
comptime L_AVGPOOL2D = 14
comptime L_CONV1D = 15
comptime L_MAXPOOL1D = 16
comptime L_AVGPOOL1D = 17
comptime L_ADAPTIVE_AVGPOOL1D = 18
comptime L_BRANCH = 19
comptime L_SILU = 32
comptime L_SOFTPLUS = 34
comptime L_RMS_NORM = 35
comptime L_MULTIPLY = 37
comptime L_NARROW = 38
comptime L_ATTENTION = 40
comptime L_RECURRENT = 41
comptime L_SELECTIVE_SCAN = 42
comptime L_SSD_SCAN = 43
comptime L_TRAPEZOID_SCAN = 44
comptime L_LM_HEAD = 45
comptime L_BATCH_NORM = 46

#: the recipe's fit (python/mojolearn/_verify_recipe.py _seq): rows, test rows, epochs, batch, lr
comptime _ROWS = 32
comptime _TEST = 16
comptime _EPOCHS = 2
comptime _BATCH = 16
comptime _LR = 0.01
#: the ssd_scan models' SGD rate. Without a norm the scan's output is large at init (softplus makes every B and C
#: entry positive, so C . B sums 128 positive products): the recipe's first loss was 5744 and full-batch SGD
#: diverged to NaN at lr 0.01, 1e-3 and 1e-4 alike (mamba2-nan-diag.out). With rms_norm after the scan (Mamba-2's
#: norm before its output projection) the head sees 512 features of unit scale, so the mse Hessian's largest
#: eigenvalue is about 2 * 512 and SGD is stable below about 2 / 1024: lr 0.01 still diverged (1.77 -> 49.7 ->
#: 4062), lr 1e-3 falls monotonically (1.77 -> 1.20 over 12 steps).
comptime _SSD_LR = 0.001
comptime _UNITS = 21


def _fail(message: String) raises:
    print("FAIL SEQ_FINITE", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _finite(name: String, what: String, v: List[Float32]) raises -> Float32:
    """Raise when any value is NaN or infinite (naming the first and the count); else the largest |value|."""
    var bad = 0
    var first = -1
    var big = Float32(0)
    for i in range(len(v)):  # small-loop(len(v): check words)
        if not isfinite(v[i]):
            bad += 1
            if first < 0:
                first = i
        elif abs(v[i]) > big:
            big = abs(v[i])
    if bad > 0:
        _fail(name + ": " + String(bad) + " of " + String(len(v)) + " " + what + " are not finite (the first at "
              + String(first) + ": " + String(v[first]) + ")")
    return big


struct _Unit(Movable):
    var name: String
    var model: SeqModel
    var width: Int  # the flat sample width
    var lengths: Bool  # rows carry sequence lengths (the recurrent layers)
    var steps: Int  # the sample's first axis (the lengths' bound)

    def __init__(out self, name: String, var model: SeqModel, width: Int, lengths: Bool = False, steps: Int = 0):
        self.name = name
        self.model = model^
        self.width = width
        self.lengths = lengths
        self.steps = steps


def _seq(sample: List[Int]) raises -> SeqModel:
    var m = SeqModel()
    m.sample = sample.copy()
    m.compile(LOSS_MSE, "sgd", _LR)
    return m^


def _head(mut m: SeqModel):
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, 1, 1)


def _unit(u: Int) raises -> _Unit:
    if u < 2:
        # ssd_scan: (L, 2) -> linear(65 heads + 256) -> softplus (dt > 0) -> ssd_scan(heads) -> rms_norm, sgd 0.001;
        # unit 0 is the shipped recipe example:seq-mamba2 exactly
        var heads = u + 1
        var steps = 8 if u == 0 else 4
        var m = _seq([steps, 2])
        m.layer(L_LINEAR, 65 * heads + 256, 1)
        m.layer(L_SOFTPLUS, 0)
        m.layer(L_SSD_SCAN, heads)
        m.layer(L_RMS_NORM, f=1e-6)
        _head(m)
        m.compile(LOSS_MSE, "sgd", _SSD_LR)
        return _Unit("ssd_scan_h" + String(heads), m^, steps * 2)
    if u == 2:
        # selective_scan: (8, 4) -> linear(2 dim + 2 state) -> softplus (delta > 0) = [u | delta | B | C], dim 3,
        # state 4
        var m = _seq([8, 4])
        m.layer(L_LINEAR, 14, 1)
        m.layer(L_SOFTPLUS, 0)
        m.layer(L_SELECTIVE_SCAN, 4)
        _head(m)
        return _Unit("selective_scan", m^, 32)
    if u == 3:
        # trapezoid_scan: (6, 4) -> linear(heads (P + 2 N + 3)) -> scan(heads 2, state 3), P 2
        var m = _seq([6, 4])
        m.layer(L_LINEAR, 22, 1)
        m.layer(L_TRAPEZOID_SCAN, 2, 3)
        _head(m)
        return _Unit("trapezoid_scan", m^, 24)
    if u == 4:
        var m = _seq([5, 4])
        m.layer(L_LINEAR, 6, 1)
        m.layer(L_LM_HEAD, 11)
        _head(m)
        return _Unit("lm_head", m^, 20)
    if u == 5:
        # attention (heads 2, kv_heads 1, rotary, no window, causal)
        var m = _seq([5, 8])
        m.layer(L_LINEAR, 8, 1)
        m.layer(L_ATTENTION, 2, 1, 1, 0, 1, 0, f=10000.0)
        _head(m)
        return _Unit("attention", m^, 40)
    if u == 6:
        var m = _seq([3, 4])
        m.layer(L_LINEAR, 8, 1)
        m.layer(L_RMS_NORM, f=1e-6)
        _head(m)
        return _Unit("rms_norm", m^, 12)
    if u == 7:
        # gated: 0 linear, 1 linear (gate), 2 silu, 3 branch(0), 4 linear (up), 5 multiply(2)
        var m = _seq([3, 4])
        m.layer(L_LINEAR, 6, 1)
        m.layer(L_LINEAR, 5, 0)
        m.layer(L_SILU)
        m.layer(L_BRANCH, 0)
        m.layer(L_LINEAR, 5, 0)
        m.layer(L_MULTIPLY, 2)
        _head(m)
        return _Unit("gated", m^, 12)
    if u == 8:
        # conv1d: Mamba's causal depthwise conv, (L 6, C 4) -> permute (C, L) -> conv1d k 3, padding 2, groups 4
        # -> the first 6 positions
        var m = _seq([6, 4])
        m.layer(L_LINEAR, 4, 1)
        m.layer(L_PERMUTE, 1, 0, -1, -1, -1, -1)
        m.layer(L_CONV1D, 4, 3, 1, 2, 1, 4)
        m.layer(L_NARROW, 1, 0, 6)
        _head(m)
        return _Unit("conv1d", m^, 24)
    if u < 12:
        # the 1-d pools over (C 4, L 6) after a padded conv1d
        var m = _seq([4, 6])
        m.layer(L_CONV1D, 4, 3, 1, 1, 1, 0)
        var name: String
        if u == 9:
            m.layer(L_MAXPOOL1D, 2, 2, 0)
            name = "max_pool1d"
        elif u == 10:
            m.layer(L_AVGPOOL1D, 3, 1, 1)
            name = "avg_pool1d"
        else:
            m.layer(L_ADAPTIVE_AVGPOOL1D, 3)
            name = "adaptive_avg_pool1d"
        _head(m)
        return _Unit(name, m^, 24)
    if u < 14:
        # the 2-d pools over (C 2, 4, 4) after a padded conv2d
        var m = _seq([2, 4, 4])
        m.layer(L_CONV2D, 2, 3, 1, 1, 1, 0)
        if u == 12:
            m.layer(L_MAXPOOL2D, 2, 2)
        else:
            m.layer(L_AVGPOOL2D, 2, 1)
        _head(m)
        return _Unit("max_pool2d" if u == 12 else "avg_pool2d", m^, 32)
    if u < 16:
        # batch_norm1d over (C 4, L 6) and batch_norm2d over (C 2, 4, 4), momentum 0.1, eps 1e-5
        var momentum = Int(bitcast[DType.uint32](Float32(0.1)))
        if u == 14:
            var m = _seq([4, 6])
            m.layer(L_CONV1D, 4, 3, 1, 1, 1, 0)
            m.layer(L_BATCH_NORM, 4, momentum, 1, f=1e-5)
            _head(m)
            return _Unit("batch_norm1d", m^, 24)
        var m = _seq([2, 4, 4])
        m.layer(L_CONV2D, 2, 3, 1, 1, 1, 0)
        m.layer(L_BATCH_NORM, 2, momentum, 2, f=1e-5)
        _head(m)
        return _Unit("batch_norm2d", m^, 32)
    if u < 19:
        # rnn, lstm, gru (hidden 5) over (T 8, 3) with lengths 1 .. 8 -> the last step -> linear
        var cell = u - 16 if u >= 16 else 0
        var m = _seq([8, 3])
        m.layer(L_RECURRENT, 5, cell)
        m.layer(L_NARROW, 0, 7, 1)
        _head(m)
        var names: List[String] = ["rnn_lengths", "lstm_lengths", "gru_lengths"]
        return _Unit(names[cell], m^, 24, True, 8)
    # u 19, 20: the ssd_scan model fitted with adam (lr 1e-3), without the norm (19) and with it (20)
    var m = _seq([8, 2])
    m.layer(L_LINEAR, 321, 1)
    m.layer(L_SOFTPLUS, 0)
    m.layer(L_SSD_SCAN, 1)
    if u == 20:
        m.layer(L_RMS_NORM, f=1e-6)
    _head(m)
    m.compile(LOSS_MSE, "adam", 1e-3)
    return _Unit("ssd_scan_h1_adam" if u == 19 else "ssd_scan_h1_rms_adam", m^, 16)


def _data(salt: Int, rows: Int, width: Int) -> List[Float32]:
    var x = List[Float32](length=rows * width, fill=0)
    for i in range(rows * width):  # small-loop(rows * width: check data)
        x[i] = Float32(normal(salt, i))
    return x^


def _target(x: List[Float32], rows: Int, width: Int) -> List[Float32]:
    """The recipe's sequence target: 0.5 x[0] - x[last] + 0.25 tanh(x[mid]) per sample."""
    var y = List[Float32](length=rows, fill=0)
    for r in range(rows):  # small-loop(rows: check data)
        var b = r * width
        y[r] = Float32(0.5 * Float64(x[b]) - Float64(x[b + width - 1]) + 0.25 * tanh(Float64(x[b + width // 2])))
    return y^


def _lengths(rows: Int, steps: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=rows, fill=0)
    for r in range(rows):  # small-loop(rows: check data)
        out[r] = Float32(1 + (r * 5 + salt) % steps)
    return out^


def run_unit[E: Exec](mut ex: E, u: Int) raises -> List[Float32]:
    var unit = _unit(u)
    var name = unit.name
    var w = unit.width
    var x = _data(0x5F17 + u, _ROWS, w)
    var y = _target(x, _ROWS, w)
    var xt = _data(0x5F57 + u, _TEST, w)
    if unit.lengths:
        unit.model.lengths = _lengths(_ROWS, unit.steps, 0)
        unit.model.predict_lengths = _lengths(_TEST, unit.steps, 3)
    # the seeded init answers finite values
    var spec = unit.model.spec(w)
    var n = spec.n_params
    var p = ex.alloc(n)
    model_init_params(ex, spec, p, UInt64(0))
    var init = List[Float32](length=n, fill=0)
    ex.sync()
    ex.download(FP(unsafe_from_address=Int(init.unsafe_ptr())), p, n)
    ex.free(p)
    _ = _finite(name, "initial parameters", init)
    unit.model.params = init.copy()
    var init_pred = unit.model.predict(ex, xt, _TEST, w, OUT_RAW)
    _ = _finite(name, "predictions of the initial model", init_pred)
    # the recipe's fit from that init
    var epochs = unit.model.fit(ex, x, _ROWS, w, y, 1, TARGET_DENSE, _EPOCHS, _BATCH, 0, 0)
    if epochs != _EPOCHS:
        _fail(name + ": the fit ran " + String(epochs) + " epochs, " + String(_EPOCHS) + " asked")
    var curve = unit.model.losses()
    _ = _finite(name, "epoch losses", curve)
    var pmax = _finite(name, "fitted parameters", unit.model.params)
    var pred = unit.model.predict(ex, xt, _TEST, w, OUT_RAW)
    var ymax = _finite(name, "predictions of the fitted model", pred)
    print("SEQ_FINITE", name, "loss", curve[0], curve[1], "max|param|", pmax, "max|pred|", ymax)
    var words = curve.copy()
    words.extend(unit.model.params.copy())
    words.extend(pred^)
    _ = x^
    _ = y^
    _ = xt^
    return words^


def test_sequential_finite_fit() raises:
    # Independent units: on the host column each runs on its own HostExec across host cores; the device column
    # then runs them in order on the one device and requires the host words.
    var failed = List[Int](length=_UNITS, fill=0)
    var host_words = List[List[Float32]](length=_UNITS, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = run_unit(host, unit)
        except error:
            print("FAIL SEQ_FINITE unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, _UNITS)
    var bad = 0
    for u in range(_UNITS):  # small-loop(_UNITS: failure flags)
        bad += failed[u]
    if bad > 0:
        _fail(String(bad) + " of " + String(_UNITS) + " units failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(_UNITS):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = run_unit(dev, unit)
            _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS SEQ_FINITE", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
