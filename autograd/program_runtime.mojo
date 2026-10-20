# SPDX-License-Identifier: Apache-2.0
# Native execution and minibatch training of captured operation graphs.
# All graph/epoch/batch walks, random streams and optimizer state live here.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_COPY, OP_SEQ_IOTA
from autograd.ops_api import OpsSession, OPS_DROPOUT
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.rankn import ShapeN
from autograd.tape import Tape, Tensor
from autograd.cast_ops import upload_cast, upload_cast_i32
from core.cast_in import CastIn
from autograd.data import permutation, gather_rows
from autograd.optim import step_lr
from autograd.losses import mse, mae, huber, binary_cross_entropy_with_logits, cross_entropy
from autograd.import_api import ImportRules
from checks.numerics import identical_div


def parameter_buffers[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram,
) raises -> List[FP]:
    var out = List[FP]()
    for j in range(len(p.params)):  # driver-loop: over program parameters
        var slot: Int
        try:
            slot = s._slot(p.params[j])
        except:
            raise Error("program parameter was freed or changed shape")
        if s.owned[slot] == 0 or not same_shape(s.shapes[slot], p.param_shapes[j]):
            raise Error("program parameter was freed or changed shape")
        out.append(FP(unsafe_from_address=s.owned[slot]))
    return out^


def replay_values[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, inputs: List[FP], shapes: List[ShapeN],
    params: List[FP], trainable: List[Bool], mut temps: List[Int], mut phs: List[Int],
    step: Int = -1,
) raises -> List[Int]:
    var vals = List[Int]()
    for j in range(len(inputs)):  # driver-loop: over program inputs
        var h = s._leaf(inputs[j], shapes[j])
        temps.append(h)
        vals.append(h)
    for j in range(len(params)):  # driver-loop: over program parameters
        var t = s.tape.variable_nd(s.ex, params[j], p.param_shapes[j], trainable[j])
        var h = s._add(t, 0, trainable[j], 0)
        temps.append(h)
        phs.append(h)
    for k in range(p.n_nodes()):  # driver-loop: over captured operations
        var hs = List[Int]()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over node arguments
            var rf = p.args[j]
            hs.append(phs[-rf - 1] if rf < 0 else vals[rf])
        var ints = List[Int]()
        for j in range(p.i_at[k], p.i_at[k + 1]):  # driver-loop: over node metadata
            ints.append(p.ints[j])
        var fs = List[Float32]()
        for j in range(p.f_at[k], p.f_at[k + 1]):  # driver-loop: over node metadata
            fs.append(p.fs[j])
        if step >= 0 and p.code[k] == OPS_DROPOUT:
            # Preserve the caller's stream key; advance its counter per
            # optimizer step and node so distinct layers never share masks.
            ints[len(ints) - 1] += step * p.n_nodes() + k
        var h = s._program_op(k, p.code[k], hs, ints, fs)
        temps.append(h)
        vals.append(h)
    var outputs = List[Int]()
    for j in range(len(p.outs)):  # driver-loop: over program outputs
        var rf = p.outs[j]
        outputs.append(phs[-rf - 1] if rf < 0 else vals[rf])
    return outputs^


def program_loss[E: Exec](
    mut tape: Tape, mut ex: E, output: Tensor, target: Tensor, kind: Int, delta: Float32,
) raises -> Tensor:
    if kind == 0:
        if tape.rows(output) * tape.cols(output) != 1:
            raise Error("loss='output' requires one scalar program output")
        return output
    if kind == 1:
        return mse(tape, ex, output, target)
    if kind == 2:
        return mae(tape, ex, output, target)
    if kind == 3:
        return huber(tape, ex, output, target, delta)
    if kind == 4:
        return binary_cross_entropy_with_logits(tape, ex, output, target)
    if kind == 5:
        return cross_entropy(tape, ex, output, target)
    raise Error("unknown program loss")


def release_training_buffers[E: Exec](
    mut ex: E, full: List[FP], batches: List[FP], ty: FP, yb: FP,
    indices: FP, batch_losses: FP, losses: FP,
) raises:
    for k in range(len(full)):  # driver-loop: over input buffer owners
        ex.free(full[k])
        ex.free(batches[k])
    ex.free(ty)
    ex.free(yb)
    ex.free(indices)
    ex.free(batch_losses)
    ex.free(losses)


def train_program[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], pid: Int, srcs: List[CastIn], shapes: List[ShapeN],
    target: CastIn, loss_kind: Int, delta: Float32, optimizer: Int, parameters: List[Int],
    epochs: Int, batch: Int, shuffle_rows: Bool, seed: UInt64, curve: FP,
) raises:
    var p = s.program(pid)
    if s.tape.is_recording():
        raise Error("Program.fit uses a captured inference session")
    if epochs < 1 or batch < 1 or len(srcs) != p.n_in or len(parameters) < 1 or len(p.outs) != 1:
        raise Error("Program.fit needs inputs, trainable parameters, one output, and positive epochs/batch_size")
    var oi = s._opt_index(optimizer)
    var n = shapes[0].d[0]
    var bs = min(n, batch)
    var n_batches = (n + bs - 1) // bs
    var params = parameter_buffers(s, p)
    var trainable = List[Bool](length=len(p.params), fill=False)
    var states = List[Int](length=len(p.params), fill=-1)
    for i in range(len(parameters)):  # driver-loop: over chosen trainable parameters
        var found = False
        for j in range(len(p.params)):  # driver-loop: over captured parameters
            if parameters[i] == p.params[j]:
                if trainable[j]:
                    raise Error("Program.fit lists a parameter twice")
                var slot = s._slot(p.params[j])
                if s.int_words[slot]:
                    raise Error("Program.fit cannot train integer ids")
                trainable[j] = True
                found = True
        if not found:
            raise Error("Program.fit parameter is not a parameter of this program")
    var full = List[FP]()
    var batches = List[FP]()
    for k in range(len(srcs)):  # driver-loop: over input shape contracts
        if shapes[k].ndim < 2 or shapes[k].d[0] != n or shapes[k].ndim != p.in_shapes[k].ndim:
            raise Error("Program.fit inputs must have equal sample counts and their captured ranks")
    if loss_kind != 0 and target.rows != n:
        raise Error("Program.fit targets must match the input sample count")
    for k in range(len(srcs)):  # driver-loop: over program input arrays
        var buf = s.ex.alloc(srcs[k].count())
        if p.in_kinds[k] == PROGRAM_IN_IDS:
            upload_cast_i32(s.ex, buf, srcs[k], "program ids")
        else:
            upload_cast(s.ex, buf, srcs[k])
        full.append(buf)
        batches.append(s.ex.alloc(bs * (shapes[k].count() // n)))
    var ty = s.ex.alloc(max(1, target.count()))
    var yb = s.ex.alloc(max(1, bs * target.cols))
    if loss_kind != 0:
        upload_cast(s.ex, ty, target)
    for j in range(len(params)):  # driver-loop: over trainable parameter state
        if trainable[j]:
            states[j] = s._state_of(oi, p.params[j], p.param_shapes[j].count())
            s.wants_grad[s._slot(p.params[j])] = True
    var indices = s.ex.alloc(n)
    var batch_losses = s.ex.alloc(n_batches)
    var losses = s.ex.alloc(epochs)
    var held = Tape()
    swap(held, s.prog_tape)
    swap(held, s.tape)
    var temps = List[Int]()
    try:
        s.tape.record()
        for e in range(epochs):  # driver-loop: over epochs
            var epoch = s.opts[oi].epoch
            if shuffle_rows:
                permutation(s.ex, seed, epoch, n, indices)
            else:
                var order = Args()
                order.p0 = indices
                s.ex.launch[OP_SEQ_IOTA](order, n)
            for bi in range(n_batches):  # driver-loop: over minibatches
                var start = bi * bs
                var count = min(bs, n - start)
                s._drop_temps(temps)
                temps.clear()
                s.tape.reset()
                var batch_shapes = List[ShapeN]()
                for k in range(len(full)):  # driver-loop: over program inputs
                    var sh = shapes[k]
                    var width = sh.count() // n
                    gather_rows(s.ex, full[k], indices, start, count, width, batches[k])
                    sh.d[0] = count
                    batch_shapes.append(sh)
                if loss_kind != 0:
                    gather_rows(s.ex, ty, indices, start, count, target.cols, yb)
                var phs = List[Int]()
                var step = 0
                for j in range(len(states)):  # driver-loop: over optimizer states
                    if states[j] >= 0:
                        step = max(step, s.opt_states[states[j]].t)
                var outs = replay_values(s, p, batches, batch_shapes, params, trainable, temps, phs, step)
                var out = s.tensor(outs[0])
                var y = s.tape.variable(s.ex, yb, count, max(1, target.cols), False)
                var loss = program_loss(s.tape, s.ex, out, y, loss_kind, delta)
                s.tape.backward_scalar_with_rules(s.ex, loss, ImportRules())
                var weighted = s.tape.scale(s.ex, loss, Float32(count))
                var a = Args()
                a.p0 = s.tape.value(weighted)
                a.p1 = batch_losses + bi
                s.ex.launch[OP_COPY](a, 1)
                for j in range(len(params)):  # driver-loop: over optimizer parameter updates
                    if not trainable[j]:
                        continue
                    var si = states[j]
                    var st = s._state_view(si)
                    var t = s.opt_states[si].t + 1
                    var gradient = s.tape.gradient(s.tensor(phs[j]))
                    step_lr(s.ex, s.opts[oi].cfg, st, params[j], gradient, p.param_shapes[j].count(),
                            t, s.learning_rate(optimizer))
                    s.opt_states[si].t = t
                    s.opt_states[si].pw1 = st.pw1
                    s.opt_states[si].pw2 = st.pw2
                    s.opt_states[si].mu_prod = st.mu_prod
            var terms = s.tape.variable(s.ex, batch_losses, 1, n_batches, False)
            var average = s.tape.scale(s.ex, s.tape.sum(s.ex, terms), identical_div(Float32(1), Float32(n)))
            var a = Args()
            a.p0 = s.tape.value(average)
            a.p1 = losses + e
            s.ex.launch[OP_COPY](a, 1)
            s.scheduler_step(optimizer)
        s.ex.sync()
        s.ex.download(curve, losses, epochs)
    except err:
        s._drop_temps(temps)
        s.tape.no_grad()
        swap(held, s.tape)
        swap(held, s.prog_tape)
        release_training_buffers(s.ex, full, batches, ty, yb, indices, batch_losses, losses)
        raise err
    s._drop_temps(temps)
    s.tape.no_grad()
    swap(held, s.tape)
    swap(held, s.prog_tape)
    release_training_buffers(s.ex, full, batches, ty, yb, indices, batch_losses, losses)
