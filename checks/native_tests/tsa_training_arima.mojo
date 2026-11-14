# SPDX-License-Identifier: Apache-2.0
"""ARIMA on the host fit, forecast and prediction bodies (ported from
python/mojolearn/tests/test_cpu_training_arima.py
`test_arima_fits_on_the_host_when_built`).

The original three-series, 160-point draw (`np.random.default_rng(0)`: a
scaled random walk plus noise) is NumPy PCG64 output, stored as raw float32
words in `fixtures/tsa/arima-host-series.f32` by
`tools/tsa_metrics_native_fixtures.py`. The three lane orders, as the
public `ARIMA(**kw).fit` resolves them (trend None: k = 1 when d + D == 0;
maxiter 1000; no regressors):

    ARIMA_KIND 0   (1, 0, 0)                      k 1
    ARIMA_KIND 1   (0, 1, 1)                      k 0
    ARIMA_KIND 2   (1, 0, 0) x (1, 0, 0, 4), 'c'  k 1

Each order is fitted twice (two independent units). Per unit, through
arima/host/arima_host_entries.mojo with the binding's routing (a prediction
starting at n_obs is the forecast body, an earlier start the in-sample
body):

- forecast(12) equals predict(n_obs, n_obs + 12) word for word;
- the tail of predict(0, n_obs) is the head of predict(n_obs - 4, n_obs + 4),
  whose tail is forecast(4), word for word;
- predict(0, n_obs) is NaN exactly over its first d + s * D steps.

Across the two units of an order the params and forecast(12) words agree.
A second AR coefficient (p = 2) refuses by name. The sabotage-build probe
and the source checks stay in the Python file. Importable; no `main()`.
"""
from std.math import isnan
from std.memory import bitcast
from core.hostptr import f32_ptr
from arima.host.arima_host_entries import _order, _forecast_into, _predict_in_sample_into, arima_host_fit_into
from checks.native_tests.tsa_metrics_fixture_words import TSA_FIXTURE_DIR, fixture_f32
from core.host_parallel import host_parallelize
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime ARIMA_B = 3
comptime ARIMA_N = 160
comptime ARIMA_KINDS = 3
#: two fits per order, then the p = 2 refusal
comptime ARIMA_UNITS = 2 * ARIMA_KINDS + 1
#: words per fit record: params (at most 4 x 3), then forecast(12) (36)
comptime ARIMA_RECORD = 64
comptime ARIMA_H = 12


def arima_series() raises -> List[Float32]:
    return fixture_f32(TSA_FIXTURE_DIR + "arima-host-series.f32", ARIMA_B * ARIMA_N)


@fieldwise_init
struct ArimaKind(Copyable, Movable):
    var p: Int
    var d: Int
    var q: Int
    var P: Int
    var D: Int
    var Q: Int
    var s: Int
    var k: Int

    def complexity(self) -> Int:
        return self.p + self.q + self.P + self.Q + self.k + 1

    def label(self) -> String:
        return ("(" + String(self.p) + "," + String(self.d) + "," + String(self.q) + ")x(" + String(self.P) + ","
                + String(self.D) + "," + String(self.Q) + "," + String(self.s) + ") k=" + String(self.k))


def arima_kind(kind: Int) -> ArimaKind:
    if kind == 0:
        return ArimaKind(1, 0, 0, 0, 0, 0, 0, 1)
    if kind == 1:
        return ArimaKind(0, 1, 1, 0, 0, 0, 0, 0)
    return ArimaKind(1, 0, 0, 1, 0, 0, 4, 1)


struct ArimaFit(Movable):
    var kind: ArimaKind
    var params: List[Float32]

    def __init__(out self, kind: ArimaKind):
        self.kind = kind.copy()
        self.params = List[Float32](length=kind.complexity() * ARIMA_B, fill=Float32(0))


def fit(y: Int, kind: ArimaKind) raises -> ArimaFit:
    var result = ArimaFit(kind)
    var width = kind.complexity() * ARIMA_B
    var x = List[Float32](length=width, fill=Float32(0))
    var x0 = List[Float32](length=width, fill=Float32(0))
    var stats = List[Float32](length=2 * ARIMA_B, fill=Float32(0))
    var flags = List[Int32](length=2 * ARIMA_B, fill=Int32(0))
    var written = arima_host_fit_into(y, 0, f32_ptr(Int(result.params.unsafe_ptr())),
        f32_ptr(Int(x.unsafe_ptr())), f32_ptr(Int(x0.unsafe_ptr())), f32_ptr(Int(stats.unsafe_ptr())),
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(flags.unsafe_ptr())),
        ARIMA_B, ARIMA_N, kind.p, kind.d, kind.q, kind.P, kind.D, kind.Q, kind.s, kind.k, 0, 1000, 0, 0.0, 0.0)
    require(written == width, kind.label() + ": fit result count")
    _ = x^
    _ = x0^
    _ = stats^
    _ = flags^
    return result^


def predict(y: Int, model: ArimaFit, start: Int, end: Int) raises -> List[Float32]:
    """`predict(start, end)` as the binding routes it."""
    var kind = model.kind.copy()
    var order = _order(kind.p, kind.d, kind.q, kind.P, kind.D, kind.Q, kind.s, kind.k, 0)
    var out = List[Float32](length=ARIMA_B * (end - start), fill=Float32(0))
    var op = f32_ptr(Int(out.unsafe_ptr()))
    var count: Int
    if start == ARIMA_N:
        count = _forecast_into(y, 0, 0, Int(model.params.unsafe_ptr()), op, ARIMA_B, ARIMA_N,
                               end - start, order)
    else:
        count = _predict_in_sample_into(y, 0, 0, Int(model.params.unsafe_ptr()), op, ARIMA_B,
                                        ARIMA_N, start, end, order)
    require(count == ARIMA_B * (end - start), kind.label() + ": prediction result count")
    return out^


def forecast(y: Int, model: ArimaFit, steps: Int) raises -> List[Float32]:
    var kind = model.kind.copy()
    var order = _order(kind.p, kind.d, kind.q, kind.P, kind.D, kind.Q, kind.s, kind.k, 0)
    var out = List[Float32](length=ARIMA_B * steps, fill=Float32(0))
    var count = _forecast_into(y, 0, 0, Int(model.params.unsafe_ptr()),
                               f32_ptr(Int(out.unsafe_ptr())), ARIMA_B, ARIMA_N, steps, order)
    require(count == ARIMA_B * steps, kind.label() + ": forecast result count")
    return out^


def same_word(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def refusal_unit(y: Int) raises:
    try:
        _ = fit(y, ArimaKind(2, 0, 0, 0, 0, 0, 0, 1))
    except problem:
        require("no CPU implementation of ARIMA with p=2" in String(problem), "p=2 refusal: " + String(problem))
        return
    raise Error("p=2 did not refuse by name")


def fit_unit(unit: Int, y: Int, record: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """Units 0 .. 5: order unit // 2, fit unit % 2; unit 6: the refusal."""
    if unit == 2 * ARIMA_KINDS:
        refusal_unit(y)
        return
    var kind = arima_kind(unit // 2)
    var label = kind.label()
    var model = fit(y, kind)
    var fc = forecast(y, model, ARIMA_H)
    var pr = predict(y, model, ARIMA_N, ARIMA_N + ARIMA_H)
    for i in range(len(fc)):  # small-loop(36 forecast words)
        require(same_word(fc[i], pr[i]), label + ": forecast(h) and predict(n_obs, n_obs + h) differ")
    var ins = predict(y, model, 0, ARIMA_N)
    var st = predict(y, model, ARIMA_N - 4, ARIMA_N + 4)
    var f4 = forecast(y, model, 4)
    var nan_prefix = kind.d + kind.D * kind.s
    for b in range(ARIMA_B):  # small-loop(three series)
        for j in range(4):  # small-loop(four straddle steps)
            require(same_word(st[b * 8 + j], ins[b * ARIMA_N + ARIMA_N - 4 + j]), label + ": the straddle's in-sample half moved")
            require(same_word(st[b * 8 + 4 + j], f4[b * 4 + j]), label + ": the straddle's forecast half moved")
        for t in range(ARIMA_N):  # small-loop(one series' in-sample steps)
            require(isnan(ins[b * ARIMA_N + t]) == (t < nan_prefix), label + ": in-sample NaN prefix is not d + s*D")
    var at = 0
    for v in model.params:  # small-loop(one unit's record)
        record[at] = v
        at += 1
    for v in fc:  # small-loop(one unit's record)
        record[at] = v
        at += 1
    require(at <= ARIMA_RECORD, "ARIMA record overflow")


def check_refits(records: List[Float32]) raises:
    for kind in range(ARIMA_KINDS):  # small-loop(three orders)
        for i in range(ARIMA_RECORD):  # small-loop(one record pair)
            require(same_word(records[(2 * kind) * ARIMA_RECORD + i], records[(2 * kind + 1) * ARIMA_RECORD + i]),
                    arima_kind(kind).label() + ": two host fits returned different bytes")


# ---- The host column (was tsa_training_arima_cpu_suite.mojo) ----
# ARIMA host fit repeat, forecast/predict agreement and refusal.
#
# The seven units (two fits per order, the refusal) are independent and run
# across host cores, each into its own record; the refit comparison follows.
# The original ran on the host route only (`GATE_BACKENDS = ("cpu",)`), so
# there is no GPU column.
def _run_cpu(mut report: Report) raises:
    var y = arima_series()
    var records = List[Float32](length=ARIMA_UNITS * ARIMA_RECORD, fill=Float32(0))
    var cases = report.cases(ARIMA_UNITS)
    var base = Int(records.unsafe_ptr())
    var y_address = Int(y.unsafe_ptr())
    def task(unit: Int) {imm cases, imm base, imm y_address}:
        try:
            fit_unit(unit, y_address, f32_ptr(base + 4 * unit * ARIMA_RECORD))
        except error:
            cases.fail(unit, error)
    report.parallel(task, ARIMA_UNITS)
    check_refits(records)
    _ = y^


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("tsa_training_arima", cpu=True, gpu=False, serial="", run=run_cases)
