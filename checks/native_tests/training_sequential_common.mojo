# SPDX-License-Identifier: Apache-2.0
"""The public `Sequential` description and fit words, built natively for the
training checks ported from test_autograd_{model,usability,save_resume}.py
(lane tmig-training-2, 2026-11-15).

`SeqModel` holds what the Sequential door holds (bindings/
sequential_description_pack.mojo sequential_compile / sequential_program,
bindings/sequential_execution_pack.mojo sequential_fit) and calls the same
entries it calls: `autograd/model_execution.mojo::model_fit_native` with
the stateful word lists (8 addresses, 15 integers; 13 and 16 with
validation data) and `autograd/model.mojo::model_predict`. The words:

  program v1  [1, features, layers, per layer (code, arg0, arg1, 0)]
  program v2  [2, ndim, d0 .. d7 (unused 1), layers, per layer (code, arg0 .. arg5, 0)]
  ints        [n, loss, target, opt, flags, sched, step_size, batch, epochs,
               shuffle, seed_lo, seed_hi, n_iter_no_change, init, resume (, restore)]
  floats      [lr, f1, f2, eps, weight_decay, 0, tol, gamma, huber_delta, one per layer]

with compile's defaults (adam, lr 1e-3, momentum 0.9, betas 0.9/0.999, eps
1e-8, tol 1e-4, huber 1, constant schedule) and fit's (batch 32, shuffle).
Class labels are passed as int32 codes in float32 words (TARGET_CODES for
cross_entropy, TARGET_CODE_VALUES for bce_with_logits), the form the fit
entry reads when the door's label sort is not involved. No `main()`.
"""
from std.memory import bitcast
from autograd.model import (
    FIT_STATE_WORDS, FS_EPOCH, LOSS_MSE, LOSS_CROSS_ENTROPY, LOSS_BCE_WITH_LOGITS, LOSS_HUBER,
    TARGET_DENSE, TARGET_CODES, TARGET_CODE_VALUES, OUT_RAW, OUT_SOFTMAX, OUT_PROBA2, OUT_ARGMAX,
    OUT_THRESHOLD, model_predict, parse_model, ModelSpec,
)
from autograd.model_execution import model_fit_native, model_float_words
from autograd.model_checkpoint import model_save_checkpoint, model_load_checkpoint, model_checkpoint_info
from core.cast_in import CAST_KIND_SIGNED, CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_SGD, OPT_ADAM, OPT_ADAMW


comptime L_LINEAR = 1
comptime L_RELU = 2
comptime L_SIGMOID = 3
comptime L_TANH = 4
comptime L_SOFTMAX = 5
comptime L_LAYER_NORM = 6
comptime L_DROPOUT = 7
comptime L_RESIDUAL = 8
comptime L_RESHAPE = 9
comptime L_PERMUTE = 10
comptime L_FLATTEN = 11
comptime L_CONV2D = 12
comptime L_MAXPOOL2D = 13

comptime PREDICT_CHUNK = 4096


struct SeqModel(Movable):
    """A Sequential description, its compile options and its fitted state."""
    var codes: List[Int]
    var args: List[Int]       # six per layer
    var floats: List[Float64]  # one per layer
    var sample: List[Int]     # empty: a version-1 description over rows
    var loss: Int
    var opt: Int
    var flags: Int
    var lr: Float64
    var f1: Float64
    var f2: Float64
    var eps: Float64
    var weight_decay: Float64
    var sched: Int
    var step_size: Int
    var gamma: Float64
    var huber_delta: Float64
    var tol: Float64
    var patience: Int
    # fitted state (the door's params_, optimizer_m_, optimizer_v_, _fit_state, _curve)
    var params: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var state: List[Int64]
    var curve: List[Float32]
    var n_iter: Int
    var batch: Int
    var shuffle: Bool
    var fitted: Bool
    var target: Int
    var y_1d: Int
    # the last validated fit's validation curve and (executed, retained) epochs
    var validation_curve: List[Float32]
    var validation_info: List[Int64]

    def __init__(out self):
        self.codes = List[Int]()
        self.args = List[Int]()
        self.floats = List[Float64]()
        self.sample = List[Int]()
        self.loss = LOSS_MSE
        self.opt = OPT_ADAM
        self.flags = 0
        self.lr = 1e-3
        self.f1 = 0.9
        self.f2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 0.0
        self.sched = 0
        self.step_size = 1
        self.gamma = 0.1
        self.huber_delta = 1.0
        self.tol = 1e-4
        self.patience = 0
        self.params = List[Float32]()
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.state = List[Int64]()
        self.curve = List[Float32]()
        self.n_iter = 0
        self.batch = 32
        self.shuffle = True
        self.fitted = False
        self.target = TARGET_DENSE
        self.y_1d = 1
        self.validation_curve = List[Float32]()
        self.validation_info = List[Int64]()

    # ---------------------------------------------------------------- building
    def layer(mut self, code: Int, a0: Int = 0, a1: Int = 0, a2: Int = 0, a3: Int = 0, a4: Int = 0, a5: Int = 0,
              f: Float64 = 0.0):
        self.codes.append(code)
        self.args.append(a0)
        self.args.append(a1)
        self.args.append(a2)
        self.args.append(a3)
        self.args.append(a4)
        self.args.append(a5)
        self.floats.append(f)

    def linear(mut self, units: Int):
        self.layer(L_LINEAR, units, 1)

    def act(mut self, code: Int):
        self.layer(code)

    # ---------------------------------------------------------------- compile
    def compile(mut self, loss: Int, optimizer: String, lr: Float64, weight_decay: Float64 = 0.0, sched: Int = 0,
                step_size: Int = 1, gamma: Float64 = 0.1, tol: Float64 = 1e-4, patience: Int = 0,
                nesterov: Bool = False) raises:
        """`compile(loss, optimizer, lr, ...)` with the public defaults (momentum
        0.9 without Nesterov, betas 0.9/0.999, eps 1e-8, huber_delta 1)."""
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.sched = sched
        self.step_size = step_size
        self.gamma = gamma
        self.tol = tol
        self.patience = patience
        self.flags = 0
        if optimizer == "adam" or optimizer == "adamw":
            self.opt = OPT_ADAM if optimizer == "adam" else OPT_ADAMW
            self.f1 = 0.9
            self.f2 = 0.999
        elif optimizer == "momentum":
            self.opt = OPT_SGD
            self.f1 = 0.9
            self.f2 = 0.0
            self.flags = 1 if nesterov else 0
        elif optimizer == "sgd":
            self.opt = OPT_SGD
            self.f1 = 0.0
            self.f2 = 0.0
        else:
            raise Error("unknown optimizer " + optimizer)

    # ---------------------------------------------------------------- words
    def program(self, features: Int) -> List[Int]:
        var n = len(self.codes)
        var prog = List[Int]()
        if len(self.sample) == 0:
            prog.append(1)
            prog.append(features)
            prog.append(n)
            for i in range(n):  # driver-loop: over layers
                prog.append(self.codes[i])
                prog.append(self.args[6 * i])
                prog.append(self.args[6 * i + 1])
                prog.append(0)
            return prog^
        prog.append(2)
        prog.append(len(self.sample))
        for k in range(8):  # small-loop(8: the description's shape slots)
            prog.append(self.sample[k] if k < len(self.sample) else 1)
        prog.append(n)
        for i in range(n):  # driver-loop: over layers
            prog.append(self.codes[i])
            for k in range(6):  # small-loop(6: layer arguments)
                prog.append(self.args[6 * i + k])
            prog.append(0)
        return prog^

    def table(self) -> List[Float64]:
        return self.floats.copy()

    def spec(self, features: Int) raises -> ModelSpec:
        return parse_model(self.program(features), model_float_words(self.table()))

    # ---------------------------------------------------------------- fit
    def fit[E: Exec](
        mut self, mut ex: E, x: List[Float32], rows: Int, features: Int, y: List[Float32], y_cols: Int,
        target: Int, epochs: Int, batch: Int = -1, shuffle: Int = -1, seed: Int = 0, resume: Bool = False,
    ) raises -> Int:
        """`fit(X, y, epochs=..., batch_size=..., shuffle=..., seed=..., resume=...)`;
        batch/shuffle -1 is the public None. `y`: (rows, y_cols) float32, or
        int32 class codes in float32 words for TARGET_CODES/TARGET_CODE_VALUES."""
        var no_validation = List[Float32]()
        return self._fit(ex, x, rows, features, y, y_cols, target, epochs, batch, shuffle, seed, resume,
                         no_validation, 0, no_validation, 0, False)

    def fit_validated[E: Exec](
        mut self, mut ex: E, x: List[Float32], rows: Int, features: Int, y: List[Float32], y_cols: Int,
        target: Int, epochs: Int, vx: List[Float32], v_rows: Int, vy: List[Float32], restore: Bool,
        batch: Int = -1, shuffle: Int = -1, seed: Int = 0, resume: Bool = False,
    ) raises -> Int:
        """`fit(..., validation_data=(vx, vy), restore_best_weights=restore)`."""
        return self._fit(ex, x, rows, features, y, y_cols, target, epochs, batch, shuffle, seed, resume,
                         vx, v_rows, vy, 1, restore)

    def _fit[E: Exec](
        mut self, mut ex: E, x: List[Float32], rows: Int, features: Int, y: List[Float32], y_cols: Int,
        target: Int, epochs: Int, batch: Int, shuffle: Int, seed: Int, resume: Bool,
        vx: List[Float32], v_rows: Int, vy: List[Float32], validated: Int, restore: Bool,
    ) raises -> Int:
        if resume and not self.fitted:
            raise Error("resume=True continues an earlier fit")
        var batch_size = batch if batch > 0 else (self.batch if resume else 32)
        var shuffled = (shuffle != 0) if shuffle >= 0 else (self.shuffle if resume else True)
        var prog = self.program(features)
        var table = self.table()
        var spec = parse_model(prog, model_float_words(table))
        var n_params = spec.n_params
        var previous = List[Float32]()
        if resume:
            previous = self.curve.copy()
            self.curve = List[Float32](length=self.n_iter + epochs, fill=Float32(0))
        else:
            self.params = List[Float32](length=n_params, fill=Float32(0))
            self.m = List[Float32](length=n_params, fill=Float32(0))
            self.v = List[Float32](length=n_params, fill=Float32(0))
            self.state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
            self.curve = List[Float32](length=epochs, fill=Float32(0))
        var ip: List[Int] = [
            rows, self.loss, target, self.opt, self.flags, self.sched, self.step_size, batch_size, epochs,
            1 if shuffled else 0, seed & 0xFFFFFFFF, seed >> 32, self.patience, 0 if resume else 1,
            1 if resume else 0,
        ]
        var fp: List[Float64] = [
            self.lr, self.f1, self.f2, self.eps, self.weight_decay, 0.0, self.tol, self.gamma, self.huber_delta,
        ]
        for i in range(len(table)):  # driver-loop: over layers
            fp.append(table[i])
        var addrs: List[Int] = [
            0, 0, Int(self.params.unsafe_ptr()), Int(self.curve.unsafe_ptr()), Int(self.m.unsafe_ptr()),
            Int(self.v.unsafe_ptr()), Int(self.state.unsafe_ptr()),
            Int(previous.unsafe_ptr()) if resume and len(previous) > 0 else 0,
        ]
        # the class codes are C-contiguous int32 (model_fit admits them by dtype and
        # shape, autograd/model.mojo "the class codes must be N C-contiguous int32");
        # the float32 words only carry their bits
        var y_src = CastIn.f32_c(Int(y.unsafe_ptr()), rows, y_cols)
        if target == TARGET_CODES or target == TARGET_CODE_VALUES:
            y_src = CastIn(Int(y.unsafe_ptr()), CAST_KIND_SIGNED, 4, rows, y_cols, y_cols * 4, 4)
        var sources: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), rows, features), y_src]
        if validated != 0:
            self.validation_curve = List[Float32](length=epochs, fill=Float32(-1))
            self.validation_info = List[Int64](length=2, fill=Int64(0))
            ip.append(1 if restore else 0)
            addrs.append(0)
            addrs.append(0)
            addrs.append(0)
            addrs.append(Int(self.validation_curve.unsafe_ptr()))
            addrs.append(Int(self.validation_info.unsafe_ptr()))
            sources.append(CastIn.f32_c(Int(vx.unsafe_ptr()), v_rows, features))
            sources.append(CastIn.f32_c(Int(vy.unsafe_ptr()), v_rows, y_cols))
            sources.append(CastIn.f32_c(0, 1, 1))
        self.n_iter = model_fit_native(ex, addrs, sources, prog, ip, fp)
        ex.sync()
        self.batch = batch_size
        self.shuffle = shuffled
        self.fitted = True
        self.target = target
        self.y_1d = 1 if y_cols == 1 else 0
        # every buffer the entry read stays alive past the call
        _ = previous^
        _ = x
        _ = y
        _ = vx
        _ = vy
        return self.n_iter

    def _fit_floats(self) -> List[Float64]:
        var fp: List[Float64] = [
            self.lr, self.f1, self.f2, self.eps, self.weight_decay, 0.0, self.tol, self.gamma, self.huber_delta,
        ]
        for i in range(len(self.floats)):  # driver-loop: over layers
            fp.append(self.floats[i])
        return fp^

    # ---------------------------------------------------------------- checkpoints
    def save(self, file: String, features: Int, export: Bool) raises -> Int:
        """`save_checkpoint(file)` (export False) or `export(file)` through
        autograd/model_checkpoint.mojo::model_save_checkpoint, without
        classes (the codes are the labels here). Returns the byte count."""
        var spec = self.spec(features)
        var ip: List[Int] = [
            1 if export else 0, self.loss, self.target, self.opt, self.flags, self.sched, self.step_size,
            self.patience, self.batch, 1 if self.shuffle else 0, features, spec.out_features(), self.y_1d,
            self.n_iter, 0,
        ]
        var addrs: List[Int] = [
            Int(self.params.unsafe_ptr()), Int(self.m.unsafe_ptr()), Int(self.v.unsafe_ptr()),
            Int(self.state.unsafe_ptr()), Int(self.curve.unsafe_ptr()),
        ]
        return model_save_checkpoint(file, self.program(features), addrs, ip, self._fit_floats(), List[Int]())

    def load(mut self, file: String, features: Int) raises:
        """`Sequential.from_checkpoint(file)` / `Sequential.load(file)` into this
        description: the buffers, counters, batch and shuffle of the file
        (model_checkpoint_info, model_load_checkpoint)."""
        var info = model_checkpoint_info(file)
        var n = len(self.codes)
        if len(info.layer_ints) != 3 * n:
            raise Error("the checkpoint describes another model")
        for i in range(n):  # driver-loop: over layers
            if info.layer_ints[3 * i] != self.codes[i]:
                raise Error("the checkpoint's layer " + String(i) + " differs")
        var n_params = info.ints[len(info.ints) - 1]
        var n_run = info.ints[12]
        self.params = List[Float32](length=n_params, fill=Float32(0))
        self.m = List[Float32](length=n_params, fill=Float32(0))
        self.v = List[Float32](length=n_params, fill=Float32(0))
        self.state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
        self.curve = List[Float32](length=max(n_run, 1), fill=Float32(0))
        var addrs: List[Int] = [
            Int(self.params.unsafe_ptr()), Int(self.m.unsafe_ptr()), Int(self.v.unsafe_ptr()),
            Int(self.state.unsafe_ptr()), Int(self.curve.unsafe_ptr()), 0,
        ]
        _ = model_load_checkpoint(file, addrs)
        self.n_iter = n_run
        self.batch = info.ints[7]
        self.shuffle = info.ints[8] != 0
        self.target = info.ints[1]
        self.y_1d = info.ints[11]
        self.fitted = not info.is_export

    def losses(self) -> List[Float32]:
        """`loss_curve_`: the first n_iter epoch losses."""
        var out = List[Float32]()
        for i in range(min(self.n_iter, len(self.curve))):  # small-loop(epochs): fitted counters
            out.append(self.curve[i])
        return out^

    def epoch(self) -> Int:
        return Int(self.state[FS_EPOCH])

    # ---------------------------------------------------------------- predict
    def predict[E: Exec](
        self, mut ex: E, x: List[Float32], rows: Int, features: Int, transform: Int
    ) raises -> List[Float32]:
        """The model's (rows, out) output through `transform` (OUT_RAW,
        OUT_SOFTMAX, OUT_PROBA2, OUT_ARGMAX or OUT_THRESHOLD)."""
        var spec = self.spec(features)
        var width = spec.out_features()
        if transform == OUT_PROBA2:
            width = 2
        elif transform == OUT_ARGMAX or transform == OUT_THRESHOLD:
            width = 1
        var out = List[Float32](length=rows * width, fill=Float32(-7.0))
        model_predict(ex, spec, CastIn.f32_c(Int(x.unsafe_ptr()), rows, features), rows,
                      FP(unsafe_from_address=Int(self.params.unsafe_ptr())), FP(unsafe_from_address=Int(out.unsafe_ptr())),
                      PREDICT_CHUNK, transform)
        ex.sync()
        _ = x
        return out^


def codes_as_words(codes: List[Int]) -> List[Float32]:
    """int32 class codes in float32 words (the fit entry's code targets)."""
    var out = List[Float32](length=len(codes), fill=Float32(0))
    for i in range(len(codes)):  # small-loop(fixture labels): fixture-sized, no product work
        out[i] = bitcast[DType.float32](Int32(codes[i]))
    return out^
