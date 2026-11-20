# SPDX-License-Identifier: Apache-2.0
"""The sequence-expansion neural RUN half (`sequence/xseq_neural_run.mojo`),
reached without Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per
RUN function group: literal words, the estimators' own refusal texts, and the
descriptor calls against a fake entry that checks each envelope and answers
with words the case can read back. The compiled model entries themselves are
not called here.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_call_output import NativeCallWorkspace
from sequence.neural_model_dispatch import (
    RNN_FIT, RNN_PREDICT, MLP_FIT, MLP_PREDICT, MODEL_ARGMAX, MODEL_MIN, MODEL_THRESHOLD, MODEL_UNIFORM,
)
from sequence.xseq_neural_run import (
    XseqRunF32P, XseqRnnFitRequest, XseqRnnPredictRequest, XseqMlpFitRequest, XseqMlpPredictRequest,
    XseqNeuralEntry, xseq_run_shape_text, xseq_run_size, xseq_run_as_f32_check, xseq_neural_run_call, xseq_run_uniform,
    xseq_run_argmax, xseq_run_threshold, xseq_run_curve_min, xseq_run_opt_sgd, xseq_run_opt_adam, xseq_run_opt_nadam,
    xseq_run_opt_lion, xseq_run_opt_rmsprop, xseq_run_opt_adagrad, xseq_run_rnn_cell, xseq_run_rnn_gates,
    xseq_run_rnn_rank_check, xseq_run_rnn_dims_check, xseq_run_rnn_param_count, xseq_run_rnn_bound,
    xseq_run_rnn_schedule, xseq_run_rnn_fit, xseq_run_rnn_layout, xseq_run_rnn_predict, xseq_run_mlp_scalars_check,
    xseq_run_mlp_hidden_check, xseq_run_mlp_out_name, xseq_run_mlp_sizes, xseq_run_mlp_param_count,
    xseq_run_mlp_glorot, xseq_run_mlp_fit, xseq_run_copy_f32, xseq_run_mlp_layers_check, xseq_run_mlp_layer_check,
    xseq_run_mlp_pack, xseq_run_mlp_output_check, xseq_run_mlp_predict,
)

comptime XSEQ_NEURAL_RUN_CASES = 17


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def fake_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """A model descriptor entry stand-in: checks each operation's envelope
    and answers with words the cases read back."""
    try:
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var word = operation * 1000000 + na * 10000 + np * 100 + nf
        if operation == MODEL_UNIFORM:
            require(na == 1 and np == 4 and nf == 2, "uniform envelope")
            # the stream counter lands in the first drawn word
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=a[0])[] = Float32(p[3])
        elif operation == MODEL_ARGMAX:
            require(na == 2 and np == 2 and nf == 0, "argmax envelope")
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[1])[] = Int64(p[0] * 10 + p[1])
        elif operation == MODEL_THRESHOLD:
            require(na == 2 and np == 1 and nf == 0, "threshold envelope")
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[1])[] = Int64(p[0] + 100)
        elif operation == MODEL_MIN:
            require(na == 2 and np == 1 and nf == 0, "minimum envelope")
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[1])[] = 2.5
        elif operation == RNN_FIT:
            require(na == 5 and np == 15 and nf == 6, "rnn fit envelope")
        elif operation == RNN_PREDICT:
            require((na == 4 or na == 7) and np == 10 and nf == 0, "rnn predict envelope")
            word = 0 if p[8] == 0 else p[6] * p[4]   # chunk 0 answers a short count
        elif operation == MLP_FIT:
            require(na == 4 and nf == 8, "mlp fit envelope")
            word = p[10]                             # every epoch ran
        elif operation == MLP_PREDICT:
            require(na == 3 or na == 6, "mlp predict envelope")
            word = p[0] * p[2]
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def _fake() -> XseqNeuralEntry:
    """The fake entry with no retained workspace (a private one per call)."""
    return XseqNeuralEntry(fake_entry, 0)


def run_case(unit: Int) raises:
    if unit == 0:
        require(xseq_run_shape_text(List[Int]()) == "()" and xseq_run_shape_text([5]) == "(5,)"
                and xseq_run_shape_text([2, 3]) == "(2, 3)", "shape text")
        require(xseq_run_size([2, 3, 4]) == 24 and xseq_run_size(List[Int]()) == 1, "size words")
    elif unit == 1:
        xseq_run_as_f32_check([3, 2], 2, "X")
        xseq_run_as_f32_check([4], 0, "y")
        try:
            xseq_run_as_f32_check([4], 2, "X")
            require(False, "rank-one X accepted")
        except problem:
            _refuses(problem, "mojolearn: X must be 2-D, got 1-D shape (4,). Reshape your data either using "
                     + "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it "
                     + "contains a single sample.")
        try:
            xseq_run_as_f32_check([3, 0], 2, "X")
            require(False, "featureless X accepted")
        except problem:
            _refuses(problem, "mojolearn: X is empty, shape (3, 0): Found array with 0 feature(s) (shape=(3, 0)) "
                     + "while a minimum of 1 is required.")
        try:
            xseq_run_as_f32_check([0], 0, "y")
            require(False, "empty y accepted")
        except problem:
            _refuses(problem, "mojolearn: y is empty, shape (0,): Found array with 0 sample(s) (shape=(0,)) "
                     + "while a minimum of 1 is required.")
    elif unit == 2:
        var words = List[Float32](length=4, fill=-1.0)
        var address = Int(words.unsafe_ptr())
        require(xseq_run_uniform(_fake(), address, 3, UInt64(7), 5, -1.0, 1.0) == 8, "uniform counter")
        require(words[0] == 5.0, "uniform draws at the stream counter")
        require(xseq_run_uniform(_fake(), address + 4, 0, UInt64(7), 8, -1.0, 1.0) == 8 and words[1] == -1.0,
                "an empty draw makes no call")
        _ = words
    elif unit == 3:
        var codes = List[Int64](length=1, fill=-1)
        xseq_run_argmax(_fake(), 64, Int(codes.unsafe_ptr()), 0, 3)
        require(codes[0] == -1, "argmax of no rows makes no call")
        xseq_run_argmax(_fake(), 64, Int(codes.unsafe_ptr()), 2, 3)
        require(codes[0] == 23, "argmax words")
        xseq_run_threshold(_fake(), 64, Int(codes.unsafe_ptr()), 4)
        require(codes[0] == 104, "threshold words")
        _ = codes
    elif unit == 4:
        require(xseq_run_curve_min(_fake(), 64, 3) == 2.5, "curve minimum word")
        require(xseq_neural_run_call(_fake(), 3, [1, 2], [3], List[Float64]()) == 3020100, "entry envelope words")
        # A retained provider workspace is leased for the call and free again after it.
        var workspace = List[NativeCallWorkspace]()
        workspace.append(NativeCallWorkspace())
        var held = XseqNeuralEntry(fake_entry, Int(workspace.unsafe_ptr()))
        require(xseq_neural_run_call(held, 3, [1, 2], [3], List[Float64]()) == 3020100, "leased entry words")
        require(xseq_neural_run_call(held, 3, [1, 2], [3], List[Float64]()) == 3020100, "workspace reused")
        require(workspace[0].busy == 0, "workspace lease released")
        _ = workspace
    elif unit == 5:
        var sgd = xseq_run_opt_sgd(1, True, 0.9, 0.01, 0.0)
        require(sgd.kind == 1 and sgd.flags == 1 and len(sgd.floats) == 6 and sgd.floats[0] == 0.9
                and sgd.floats[3] == 0.01 and sgd.floats[4] == 0.0, "sgd slots")
        try:
            _ = xseq_run_opt_sgd(1, True, 0.0, 0.0, 0.0)
            require(False, "nesterov without momentum accepted")
        except problem:
            _refuses(problem, "sgd: nesterov needs momentum > 0 and dampening 0 (torch.optim.SGD)")
        var plain = xseq_run_opt_sgd(1, False, 0.0, 0.0, 0.5)
        require(plain.flags == 0 and plain.floats[4] == 0.5, "plain sgd slots")
    elif unit == 6:
        var adam = xseq_run_opt_adam("adamw", 2, 0.9, 0.999, 1e-8, 0.01)
        require(adam.flags == 0 and adam.floats[1] == 0.999 and adam.floats[2] == 1e-8, "adam slots")
        try:
            _ = xseq_run_opt_adam("adamax", 2, 1.0, 0.999, 1e-8, 0.0)
            require(False, "beta 1 accepted")
        except problem:
            _refuses(problem, "adamax: betas must lie in [0, 1)")
        try:
            _ = xseq_run_opt_nadam(3, 0.9, 0.999, 1e-8, 0.0, -0.1, False)
            require(False, "negative momentum decay accepted")
        except problem:
            _refuses(problem, "nadam: momentum_decay must be >= 0")
        var nadam = xseq_run_opt_nadam(3, 0.9, 0.999, 1e-8, 0.0, 0.004, True)
        require(nadam.flags == 1 and nadam.floats[4] == 0.004, "nadam slots")
        var lion = xseq_run_opt_lion(4, 0.9, 0.99, 0.1)
        require(lion.floats[2] == 0.0 and lion.floats[3] == 0.1, "lion slots")
        var rms = xseq_run_opt_rmsprop(5, True, 0.99, 1e-8, 0.0, 0.5)
        require(rms.flags == 1 and rms.floats[0] == 0.0 and rms.floats[1] == 0.99 and rms.floats[4] == 0.5,
                "rmsprop slots")
        var ada = xseq_run_opt_adagrad(6, 0.1, 1e-10, 0.0, 0.25)
        require(ada.floats[0] == 0.1 and ada.floats[1] == 0.0 and ada.floats[5] == 0.25, "adagrad slots")
    elif unit == 7:
        require(xseq_run_rnn_cell("rnn", True, False) == 0 and xseq_run_rnn_cell("rnn", False, True) == 1
                and xseq_run_rnn_cell("lstm", False, False) == 2 and xseq_run_rnn_cell("gru", False, False) == 3,
                "cell codes")
        try:
            _ = xseq_run_rnn_cell("rnn", False, False)
            require(False, "unknown nonlinearity accepted")
        except problem:
            _refuses(problem, "nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
        require(xseq_run_rnn_gates(2) == 4 and xseq_run_rnn_gates(3) == 3 and xseq_run_rnn_gates(0) == 1, "gates")
        try:
            xseq_run_rnn_rank_check(2)
            require(False, "rank-two X accepted")
        except problem:
            _refuses(problem, "X must be 3-D: (n_samples, n_timesteps, n_features)")
        try:
            xseq_run_rnn_dims_check(2, 0)
            require(False, "no time steps accepted")
        except problem:
            _refuses(problem, "X needs at least one sample and one time step")
    elif unit == 8:
        # one LSTM layer, D 3, H 2, O 1: 4*2*3 + 4*2*2 + 2*4*2, then 1*2 + 1
        require(xseq_run_rnn_param_count(4, 3, 2, 1, 1) == 24 + 16 + 16 + 3, "recurrent parameter count")
        require(xseq_run_rnn_bound(4.0) == 0.5, "recurrent init bound")
        var schedule = xseq_run_rnn_schedule(4, 10, 3)
        require(schedule.batch == 4 and schedule.steps == 9, "minibatch schedule")
        require(xseq_run_rnn_schedule(0, 10, 1).batch == 1 and xseq_run_rnn_schedule(50, 10, 1).steps == 1,
                "batch clipped to [1, n]")
        try:
            _ = xseq_run_rnn_schedule(1, 2147483647, 1)
            require(False, "too many epoch rows accepted")
        except problem:
            _refuses(problem, "max_epochs * n_samples must be below 2^31 - 1, got 1 * 2147483647")
    elif unit == 9:
        var optimizer = xseq_run_opt_adam("adam", 2, 0.9, 0.999, 1e-8, 0.0)
        var request = XseqRnnFitRequest(2, 3, 2, 1, 1, 0, 4, 5, 2, 2, True, UInt64(0x100000002))
        xseq_run_rnn_fit(_fake(), request, optimizer, 64, 72, 80, 88, 96)
    elif unit == 10:
        var ints: List[Int] = [2, 3, 2, 1, 1]
        require(xseq_run_rnn_layout(ints, 59) == 59, "recurrent layout")
        try:
            _ = xseq_run_rnn_layout(ints, 58)
            require(False, "wrong params_ size accepted")
        except problem:
            _refuses(problem, "ValueError: recurrent params_ size differs from the fitted network layout")
        xseq_run_rnn_predict(_fake(), XseqRnnPredictRequest(ints.copy(), 59, 0, 4, 5, 64, False, 0, 0),
                             64, 72, 80, 88)
        xseq_run_rnn_predict(_fake(), XseqRnnPredictRequest(ints.copy(), 59, 0, 4, 5, 64, True, 4096, 7),
                             64, 72, 80, 88)
        try:
            xseq_run_rnn_predict(_fake(), XseqRnnPredictRequest(ints.copy(), 59, 0, 4, 5, 0, False, 0, 0),
                                 64, 72, 80, 88)
            require(False, "short recurrent result accepted")
        except problem:
            _refuses(problem, "RuntimeError: recurrent prediction returned an invalid result count")
    elif unit == 11:
        xseq_run_mlp_scalars_check(0.001, 1.0, 0.0)
        try:
            xseq_run_mlp_scalars_check(0.001, 0.0, 0.0)
            require(False, "max_iter 0 accepted")
        except problem:
            _refuses(problem, "learning_rate_init > 0, max_iter >= 1 and alpha >= 0 are required")
        xseq_run_mlp_hidden_check([3, 1])
        try:
            xseq_run_mlp_hidden_check([3, 0])
            require(False, "empty hidden layer accepted")
        except problem:
            _refuses(problem, "hidden_layer_sizes must be > 0")
        require(xseq_run_mlp_out_name(0) == "identity" and xseq_run_mlp_out_name(1) == "logistic"
                and xseq_run_mlp_out_name(2) == "tanh" and xseq_run_mlp_out_name(3) == "relu"
                and xseq_run_mlp_out_name(4) == "softmax", "output activation names")
        var sizes = xseq_run_mlp_sizes(2, [3], 1)
        require(len(sizes) == 3 and sizes[0] == 2 and sizes[1] == 3 and sizes[2] == 1, "layer widths")
        require(xseq_run_mlp_param_count(sizes) == 13, "packed parameter count")
    elif unit == 12:
        var block = List[Float32](length=13, fill=-1.0)
        var counter = xseq_run_mlp_glorot(_fake(), Int(block.unsafe_ptr()), [2, 3, 1], False, UInt64(9), 0)
        require(counter == 13, "stream counter after the draws")
        require(block[0] == 0.0 and block[6] == 6.0 and block[9] == 9.0 and block[12] == 12.0 and block[1] == -1.0,
                "w then b of each layer, in layout order")
        _ = block
    elif unit == 13:
        var request = XseqMlpFitRequest(4, 2, 1, 3, 0, 0, 0, 0, True, 2, 7, False, 11, 10, [3], 0)
        var floats = List[Float64](length=8, fill=0.0)
        require(xseq_run_mlp_fit(_fake(), request, floats, 64, 72, 80, 88) == 7, "epochs run")
    elif unit == 14:
        var src: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0]
        var dst = List[Float32](length=5, fill=0.0)
        xseq_run_copy_f32(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()) + 4, 2)
        require(dst[0] == 2.0 and dst[1] == 3.0 and dst[2] == 0.0, "word copy")
        var w: List[Float32] = [7.0, 8.0]
        var bias: List[Float32] = [9.0]
        var packed = List[Float32](length=3, fill=0.0)
        xseq_run_mlp_pack(Int(packed.unsafe_ptr()), [Int(w.unsafe_ptr())], [2], [Int(bias.unsafe_ptr())], [1])
        require(packed[0] == 7.0 and packed[1] == 8.0 and packed[2] == 9.0, "w then b packing")
        _ = src
        _ = dst
        _ = w
        _ = bias
        _ = packed
        try:
            xseq_run_mlp_layers_check(2, 1)
            require(False, "unmatched layer lists accepted")
        except problem:
            _refuses(problem, "ValueError: MLP fitted weights require matching nonempty layer lists")
        xseq_run_mlp_layer_check([2, 3], [3], 2)
        try:
            xseq_run_mlp_layer_check([2, 3], [3, 1], 2)
            require(False, "matrix bias accepted")
        except problem:
            _refuses(problem, "ValueError: MLP weights must be matrices and biases must be vectors")
        try:
            xseq_run_mlp_layer_check([2, 3], [3], 4)
            require(False, "broken layout accepted")
        except problem:
            _refuses(problem, "ValueError: MLP fitted parameter shapes do not match the layer layout")
    else:
        xseq_run_mlp_output_check(2, 2)
        try:
            xseq_run_mlp_output_check(2, 3)
            require(False, "output width mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: MLP output weights differ from n_outputs_")
        xseq_run_mlp_predict(_fake(), XseqMlpPredictRequest(5, 2, 3, 3, 4, [4], False, 3, 31, 0, 0), 64, 72, 80)
        xseq_run_mlp_predict(_fake(), XseqMlpPredictRequest(5, 2, 3, 3, 4, [4], False, 3, 31, 4096, 7), 64, 72, 80)
        try:
            xseq_run_mlp_predict(_fake(), XseqMlpPredictRequest(5, 2, 1, 3, 1, [4], True, 2, 31, 0, 0), 64, 72, 80)
            require(False, "one-column probability accepted")
        except problem:
            _refuses(problem, "RuntimeError: mlp_predict: this binding predates the two-column probability; rebuild it")


def run_checks() raises:
    var failures = List[Int](length=XSEQ_NEURAL_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL xseq neural run case", unit, String(problem))
    host_parallelize(task, XSEQ_NEURAL_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "xseq neural run cases failed: " + String(failed))
    print("PASS xseq neural run", XSEQ_NEURAL_RUN_CASES)
