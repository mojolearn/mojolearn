# SPDX-License-Identifier: Apache-2.0
"""The neural trainers' fit loop record (`fit_loop_`; lane alm-neural,
2026-11-24, parent ab-loop-mode), through the descriptor entries the public
doors call, on one executor of the caller's column.

Each fixture fits a tiny model with the iteration cap K and early stopping
off, passing the record words' address the way its door does, and checks
the record: recorded, iterations == K (the epochs run), loop time >= 0.

  0  MLP_FIT (`mlp_fit_impl`): the fifth address; n_iter == K too.
  1  RNN_FIT (`rnn_fit_impl`): the seventh address after a 0 lengths word;
     iterations count epochs (the losses hold K * ceil(N / bs) steps).
  2  The whole-model fit (Sequential, `NativeModelRequest` +
     `model_request_execute`): FIT_LOOP_MARK ends the options and the
     record's address the addresses; both are taken off before the fit, and
     n_iter == K.
  3  CNN_FIT (`cnn_fit_program`): the same mark on the options; the steps
     returned are K * ceil(n / batch).
  4  Without the record (every caller that predates it): MLP_FIT with four
     addresses writes nothing and still runs K epochs.

Suite module (no `main`): `neural_fit_loop.mojo` runs it."""
from core.fit_loop_clock import FitLoop
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import CELL_LSTM, OPT_ADAM
from sequence.ops_impl import rnn_fit_impl, mlp_fit_impl
from sequence.recurrent import Net
from sequence.fit_loop_words import FIT_LOOP_MARK, fit_loop_words, fit_loop_read
from autograd.model_descriptor import MODEL_FIT, NativeModelRequest, model_descriptor_words, model_request_execute
from autograd.model_execution import model_counts_native
from autograd.model import FIT_STATE_WORDS
from sequence.cnn_classifier_program import cnn_fit_program
from x_cnn.classifier_run import (
    CnnClfPlan, CnnClfFitSettings, cnnclf_run_layout, cnnclf_run_fit_addresses, cnnclf_run_fit_ints,
    cnnclf_run_fit_floats, cnnclf_run_optimizer_word, cnnclf_run_fit_steps,
)
from checks.native_tests.testsuite_cases import require

comptime NEURAL_FIT_LOOP_CASES = 5
comptime LOOP_K = 3


def _values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 7 + salt * 5) % 13 - 6) / Float32(8)
    return out^


def _check_record(words: List[Int64], k: Int, what: String) raises:
    var loop = fit_loop_read(words)
    require(loop.recorded, what + ": the fit loop record was not written")
    require(loop.iterations == k, what + ": fit_loop_ iterations " + String(loop.iterations) + ", expected "
            + String(k))
    require(loop.loop_ns >= 0 and loop.milliseconds() >= 0.0, what + ": negative loop time")


def _mlp[E: Exec](mut ex: E, with_record: Bool) raises:
    var n = 24
    var d = 3
    var h = 4
    var x = _values(n * d, 1)
    var y = _values(n, 2)
    var params = _values(d * h + h + h + 1, 3)
    var curve = List[Float32](length=LOOP_K, fill=Float32(0))
    var words = fit_loop_words()
    # [N, D, O, relu, identity out, squared loss, adam, constant rate, no nesterov, batch 8, max_iter K,
    #  shuffle, seed 7, n_iter_no_change huge (no early stop), one hidden layer of h]
    var ip: List[Int] = [n, d, 1, 3, 0, 0, 0, 0, 0, 8, LOOP_K, 1, 7, 1000000, 1, h]
    var fp: List[Float64] = [1e-3, 0.9, 0.999, 1e-8, 0.9, 0.5, 0.0, 0.0]
    var addrs: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(params.unsafe_ptr()),
                            Int(curve.unsafe_ptr())]
    if with_record:
        addrs.append(Int(words.unsafe_ptr()))
    var n_iter = mlp_fit_impl(ex, addrs, ip, fp)
    require(n_iter == LOOP_K, "MLP_FIT: n_iter " + String(n_iter) + ", expected the cap " + String(LOOP_K))
    if with_record:
        _check_record(words, LOOP_K, "MLP_FIT")
    else:
        require(not fit_loop_read(words).recorded, "MLP_FIT without the record address wrote a record")
    _ = x^
    _ = y^
    _ = params^
    _ = curve^


def _rnn[E: Exec](mut ex: E) raises:
    var n = 20
    var t = 4
    var d = 3
    var hidden = 5
    var bs = 8
    var net = Net(CELL_LSTM, d, hidden, 1, 1)
    var x = _values(n * t * d, 4)
    var y = _values(n, 5)
    var params = _values(net.n_params(), 6)
    var steps = LOOP_K * ((n + bs - 1) // bs)
    var losses = List[Float32](length=steps, fill=Float32(0))
    var rates = List[Float32](length=steps, fill=Float32(1e-2))
    var words = fit_loop_words()
    var ints: List[Int] = [CELL_LSTM, d, hidden, 1, 1, 0, n, t, LOOP_K, bs, OPT_ADAM, 0, 1, 11, 0]
    var floats: List[Float64] = [0.9, 0.999, 1e-8, 0.0, 0.0, 0.0]
    _ = rnn_fit_impl(ex, [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(params.unsafe_ptr()),
                          Int(losses.unsafe_ptr()), Int(rates.unsafe_ptr()), 0, Int(words.unsafe_ptr())],
                     ints, floats)
    _check_record(words, LOOP_K, "RNN_FIT")
    _ = x^
    _ = y^
    _ = params^
    _ = losses^
    _ = rates^


def _sequential[E: Exec](mut ex: E) raises:
    var rows = 24
    var features = 3
    # a version-1 description: one linear layer of one unit (code 1, args units 1, bias 1)
    var prog: List[Int] = [1, features, 1, 1, 1, 1, 0]
    var table: List[Float64] = [0.0]
    var n_params = model_counts_native(prog, table)[0]
    var x = _values(rows * features, 7)
    var y = _values(rows, 8)
    var params = List[Float32](length=n_params, fill=Float32(0))
    var m = List[Float32](length=n_params, fill=Float32(0))
    var v = List[Float32](length=n_params, fill=Float32(0))
    var state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    var curve = List[Float32](length=LOOP_K, fill=Float32(0))
    var words = fit_loop_words()
    # [rows, mse, dense target, adam, flags, constant schedule, step 1, batch 8, epochs K, shuffle, seed 0 0,
    #  patience 0 (no early stop), initialize, not resumed], then the record's mark
    var options: List[Int] = [rows, 0, 0, OPT_ADAM, 0, 0, 1, 8, LOOP_K, 1, 0, 0, 0, 1, 0, FIT_LOOP_MARK]
    var fp: List[Float64] = [1e-3, 0.9, 0.999, 1e-8, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0]
    var addrs: List[Int] = [0, 0, Int(params.unsafe_ptr()), Int(curve.unsafe_ptr()), Int(m.unsafe_ptr()),
                            Int(v.unsafe_ptr()), Int(state.unsafe_ptr()), 0, Int(words.unsafe_ptr())]
    var sources: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), rows, features),
                                 CastIn.f32_c(Int(y.unsafe_ptr()), rows, 1)]
    var descriptor = model_descriptor_words(prog, options, sources)
    var request = NativeModelRequest(MODEL_FIT, Int(addrs.unsafe_ptr()), len(addrs), Int(descriptor.unsafe_ptr()),
                                     len(descriptor), Int(fp.unsafe_ptr()), len(fp))
    require(request.loop_words == Int(words.unsafe_ptr()), "model descriptor: the record address was not taken off")
    require(len(request.addresses) == 8 and len(request.options) == 15,
            "model descriptor: the mark and the record address must leave the fit's own words as they were")
    var n_iter = model_request_execute(ex, MODEL_FIT, request)
    ex.sync()
    require(n_iter == LOOP_K, "model fit: n_iter " + String(n_iter) + ", expected the cap " + String(LOOP_K))
    _check_record(words, LOOP_K, "model fit (Sequential)")
    _ = x^
    _ = y^
    _ = params^
    _ = m^
    _ = v^
    _ = state^
    _ = curve^
    _ = addrs^
    _ = descriptor^
    _ = fp^


def cnn_fixture[E: Exec & Movable & Deinitable](var ex: E) raises:
    """Fixture 3: the CNN program takes its executor by value."""
    var n = 16
    var batch = 8
    var classes = 3
    var layout = cnnclf_run_layout(1, 6, 6, 3, 2, [2], classes, 0)
    var plan = CnnClfPlan(n, 1, 6, 6, classes, 3, 2, [2], layout.pools.copy())
    var settings = CnnClfFitSettings(batch, LOOP_K, True, UInt64(7), cnnclf_run_optimizer_word("sgd"), False,
                                     0.01, 0.9, 0.9, 0.999, 1e-8, 0.0, 0.0)
    var x = _values(n * 36, 9)
    var y = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(fixture labels)
        y[i] = Int32(i % classes)
    var total = 0
    for i in range(len(layout.counts)):  # driver-loop: over the layer parameter arrays (model structure)
        total += layout.counts[i]
    # one flat block, one address per parameter array (no nested lists)
    var params = _values(total, 10)
    var param_addresses = List[Int]()
    var offset = 0
    for i in range(len(layout.counts)):  # driver-loop: over the layer parameter arrays (model structure)
        param_addresses.append(Int(params.unsafe_ptr()) + 4 * offset)
        offset += layout.counts[i]
    var curve = List[Float32](length=LOOP_K, fill=Float32(0))
    var steps = List[Float32](length=LOOP_K * cnnclf_run_fit_steps(n, batch), fill=Float32(0))
    var words = fit_loop_words()
    var addresses = cnnclf_run_fit_addresses(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(curve.unsafe_ptr()),
                                             Int(steps.unsafe_ptr()), param_addresses)
    addresses.append(Int(words.unsafe_ptr()))
    var ints = cnnclf_run_fit_ints(plan, settings)
    ints.append(FIT_LOOP_MARK)
    var taken = cnn_fit_program[E](ex^, addresses, ints, cnnclf_run_fit_floats(settings))
    require(taken == LOOP_K * cnnclf_run_fit_steps(n, batch), "CNN_FIT: optimizer steps " + String(taken))
    _check_record(words, LOOP_K, "CNN_FIT")
    _ = x^
    _ = y^
    _ = params^
    _ = curve^
    _ = steps^


def fixture[E: Exec](mut ex: E, unit: Int) raises:
    """Fixtures 0, 1, 2 and 4 on `ex` (fixture 3 is `cnn_fixture`)."""
    if unit == 0:
        _mlp(ex, True)
    elif unit == 1:
        _rnn(ex)
    elif unit == 2:
        _sequential(ex)
    elif unit == 4:
        _mlp(ex, False)
    else:
        raise Error("neural_fit_loop: fixture " + String(unit) + " is not an executor fixture")
