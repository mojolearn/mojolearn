# SPDX-License-Identifier: Apache-2.0
# Native execution and minibatch training of captured operation graphs.
# All graph/epoch/batch walks, random streams and optimizer state live here.
from sequence.exec_trait import Exec
from std.memory import bitcast
from sequence.ops import FP, Args, OP_COPY, OP_SEQ_IOTA
from autograd.ops_api import OpsSession, OPS_DROPOUT
from autograd.program import OpsProgram, PROGRAM_IN_IDS, same_shape
from autograd.program_choice import ProgramChoice, conditional_program
from autograd.program_replay_metadata import PreparedProgramReplay
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
    random_stride: Int = 0, random_offset: Int = 0,
) raises -> List[Int]:
    # Existing one-shot callers retain their interface and operation-local
    # ownership. Training below prepares once and reuses across minibatches.
    var prepared = PreparedProgramReplay(p)
    return replay_values_prepared(s, p, prepared, inputs, shapes, params, trainable,
                                  temps, phs, step, random_stride, random_offset)


def replay_values_prepared[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], p: OpsProgram, mut prepared: PreparedProgramReplay,
    inputs: List[FP], shapes: List[ShapeN], params: List[FP], trainable: List[Bool],
    mut temps: List[Int], mut phs: List[Int], step: Int = -1,
    random_stride: Int = 0, random_offset: Int = 0,
) raises -> List[Int]:
    ref vals = prepared.values
    vals.clear()
    for j in range(len(inputs)):  # driver-loop: over program inputs
        var h = s._leaf(inputs[j], shapes[j], p.in_kinds[j] == PROGRAM_IN_IDS)
        temps.append(h)
        vals.append(h)
    for j in range(len(params)):  # driver-loop: over program parameters
        var t = s.tape.variable_nd(s.ex, params[j], p.param_shapes[j], trainable[j])
        var h = s._add(t, 0, trainable[j], 0)
        s.int_words[s._slot(h)] = s.int_words[s._slot(p.params[j])]
        temps.append(h)
        phs.append(h)
    # Handles change with each tape/minibatch and are resolved in the original
    # node order. Integer/float arguments already belong to this native replay
    # inventory: no per-minibatch metadata copy or new scratch allocation.
    ref hs = prepared.handles
    var stride = random_stride if random_stride > 0 else p.n_nodes()
    for k in range(p.n_nodes()):  # driver-loop: over captured operations
        hs.clear()
        for j in range(p.a_at[k], p.a_at[k + 1]):  # driver-loop: over node arguments
            var rf = p.args[j]
            hs.append(phs[-rf - 1] if rf < 0 else vals[rf])
        ref node = prepared.nodes[k]
        if p.code[k] == OPS_DROPOUT:
            if step >= 0:
                # Advance from the original captured key, never from the prior
                # replay's updated counter. Preserve branch/node separation.
                node.ints[len(node.ints) - 1] = node.original_last_int + (step * stride + random_offset + k)
            elif len(node.ints) != 0:
                # A later inference/predicate replay must see the original key.
                node.ints[len(node.ints) - 1] = node.original_last_int
        var h = s._program_op(k, p.code[k], hs, node.ints, node.fs)
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
    mut ex: E, full: List[FP], batches: List[FP], ty: Optional[FP], yb: Optional[FP],
    indices: Optional[FP], batch_losses: Optional[FP], losses: Optional[FP],
) raises:
    for k in range(len(full)):  # driver-loop: over acquired full input buffers
        ex.free(full[k])
    for k in range(len(batches)):  # driver-loop: over acquired minibatch buffers
        ex.free(batches[k])
    for buffer in [ty, yb, indices, batch_losses, losses]:  # driver-loop: over nullable setup buffer owners
        if buffer:
            ex.free(buffer.value())


def train_program_control[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], pid: Int, srcs: List[CastIn], shapes: List[ShapeN],
    target: CastIn, loss_kind: Int, delta: Float32, optimizer: Int, parameters: List[Int],
    epochs: Int, batch: Int, shuffle_rows: Bool, seed: UInt64, curve: FP,
    condition_id: Int, otherwise_id: Int,
) raises -> Tuple[Int, Int]:
    var p = s.program(pid)
    var conditional = condition_id > 0 and otherwise_id > 0
    if (condition_id > 0) != (otherwise_id > 0):
        raise Error("ValueError: conditional training needs condition and otherwise Programs")
    var choice = ProgramChoice()
    if conditional:
        choice = conditional_program(p, s.program(condition_id), s.program(otherwise_id))
        p = choice.body.copy()
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
                if conditional and not choice.body_parameters[j] and not choice.other_parameters[j]:
                    raise Error("ValueError: conditional training cannot update a predicate-only parameter")
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
    var ty = Optional[FP]()
    var yb = Optional[FP]()
    var indices = Optional[FP]()
    var batch_losses = Optional[FP]()
    var losses = Optional[FP]()
    var held = Tape()
    var swapped = False
    var temps = List[Int]()
    var true_batches = 0
    var false_batches = 0
    # Program graphs, feature widths and trainable membership belong to this
    # admitted fit. Only the minibatch row count and conditional selection vary.
    # Reuse their native containers throughout the epoch/batch loop.
    var batch_shapes = shapes.copy()
    var widths = List[Int]()
    for k in range(len(shapes)):  # driver-loop: over admitted input shape metadata once per fit
        widths.append(shapes[k].count() // n)
    var selected = trainable.copy()
    var no_grad = List[Bool](length=len(params) if conditional else 0, fill=False)
    var phs = List[Int]()
    var pred_params = List[Int]()
    var branch_stride = p.n_nodes() + choice.otherwise.n_nodes() if conditional else 0
    # Graphs are immutable snapshots for this fit. Prepare their node metadata
    # once; neither predicate selection nor row counts alter those arguments.
    var body_replay = PreparedProgramReplay(p)
    var predicate_replay = PreparedProgramReplay()
    var otherwise_replay = PreparedProgramReplay()
    if conditional:
        predicate_replay = PreparedProgramReplay(choice.condition)
        otherwise_replay = PreparedProgramReplay(choice.otherwise)
    var cfg = s.opts[oi].cfg
    try:
        for k in range(len(srcs)):  # driver-loop: over program input arrays
            var buf = s.ex.alloc(srcs[k].count())
            # Register ownership before upload: a failed copy must not leave
            # an executor allocation unreachable until session destruction.
            full.append(buf)
            if p.in_kinds[k] == PROGRAM_IN_IDS:
                upload_cast_i32(s.ex, buf, srcs[k], "program ids")
            else:
                upload_cast(s.ex, buf, srcs[k])
            batches.append(s.ex.alloc(bs * (shapes[k].count() // n)))
        ty = s.ex.alloc(max(1, target.count()))
        yb = s.ex.alloc(max(1, bs * target.cols))
        if loss_kind != 0:
            upload_cast(s.ex, ty.value(), target)
        for j in range(len(params)):  # driver-loop: over trainable parameter state
            if trainable[j]:
                states[j] = s._state_of(oi, p.params[j], p.param_shapes[j].count())
                s.wants_grad[s._slot(p.params[j])] = True
        indices = s.ex.alloc(n)
        batch_losses = s.ex.alloc(n_batches)
        losses = s.ex.alloc(epochs)
        swap(held, s.prog_tape)
        swap(held, s.tape)
        swapped = True
        s.tape.record()
        for e in range(epochs):  # driver-loop: over epochs
            var epoch = s.opts[oi].epoch
            # Scheduler advancement occurs only at the end of this epoch.
            var lr = s.learning_rate(optimizer)
            if shuffle_rows:
                permutation(s.ex, seed, epoch, n, indices.value())
            else:
                var order = Args()
                order.p0 = indices.value()
                s.ex.launch[OP_SEQ_IOTA](order, n)
            for bi in range(n_batches):  # driver-loop: over minibatches
                var start = bi * bs
                var count = min(bs, n - start)
                s._drop_temps(temps)
                temps.clear()
                s.tape.reset()
                for k in range(len(full)):  # driver-loop: over program inputs
                    gather_rows(s.ex, full[k], indices.value(), start, count, widths[k], batches[k])
                    batch_shapes[k].d[0] = count
                if loss_kind != 0:
                    gather_rows(s.ex, ty.value(), indices.value(), start, count, target.cols, yb.value())
                phs.clear()
                var step = 0
                for j in range(len(states)):  # driver-loop: over optimizer states
                    if states[j] >= 0:
                        step = max(step, s.opt_states[states[j]].t)
                var take_body = True
                var random_stride = 0
                var random_offset = 0
                if conditional:
                    # The predicate sees this minibatch and current shared
                    # parameters, but contributes no derivative. No Python
                    # callback, unselected branch replay or gradient update.
                    s.tape.no_grad()
                    pred_params.clear()
                    var decision = replay_values_prepared(s, choice.condition, predicate_replay,
                                                 batches, batch_shapes, params, no_grad, temps, pred_params)
                    if s.shape_nd(decision[0]).count() != 1:
                        raise Error("ValueError: conditional training predicate must return one scalar")
                    var scalar = s.scalar(decision[0])
                    if (bitcast[DType.uint32](scalar) & UInt32(0x7fffffff)) >= UInt32(0x7f800000):
                        raise Error("ValueError: conditional training predicate must be finite")
                    take_body = scalar != 0
                    if take_body:
                        true_batches += 1
                    else:
                        false_batches += 1
                    for j in range(len(selected)):  # driver-loop: over chosen branch trainable parameters
                        selected[j] = trainable[j] and (choice.body_parameters[j] if take_body else choice.other_parameters[j])
                    s._drop_temps(temps)
                    temps.clear()
                    s.tape.reset()
                    s.tape.record()
                    # Epoch plus minibatch is a persistent native step key.
                    # Distinct branch node spans cannot reuse dropout keys.
                    step = epoch * n_batches + bi
                    random_stride = branch_stride
                    random_offset = 0 if take_body else p.n_nodes()
                # Borrow the admitted immutable graph. Copying it per batch
                # duplicated every node, argument, shape and output table.
                var outs = replay_values_prepared(s, p, body_replay, batches, batch_shapes, params, selected,
                                         temps, phs, step, random_stride, random_offset) if take_body else replay_values_prepared(
                                         s, choice.otherwise, otherwise_replay, batches, batch_shapes, params, selected,
                                         temps, phs, step, random_stride, random_offset)
                var out = s.tensor(outs[0])
                var y = s.tape.variable(s.ex, yb.value(), count, max(1, target.cols), False)
                var loss = program_loss(s.tape, s.ex, out, y, loss_kind, delta)
                var differentiable = s.tape.requires_grad(loss)
                if differentiable or not conditional:
                    s.tape.backward_scalar_with_rules(s.ex, loss, ImportRules())
                var weighted = s.tape.scale(s.ex, loss, Float32(count))
                var a = Args()
                a.p0 = s.tape.value(weighted)
                a.p1 = batch_losses.value() + bi
                s.ex.launch[OP_COPY](a, 1)
                for j in range(len(params)):  # driver-loop: over optimizer parameter updates
                    if not selected[j] or not differentiable:
                        continue
                    var si = states[j]
                    var st = s._state_view(si)
                    var t = s.opt_states[si].t + 1
                    var gradient = s.tape.gradient(s.tensor(phs[j]))
                    step_lr(s.ex, cfg, st, params[j], gradient, p.param_shapes[j].count(), t, lr)
                    s.opt_states[si].t = t
                    s.opt_states[si].pw1 = st.pw1
                    s.opt_states[si].pw2 = st.pw2
                    s.opt_states[si].mu_prod = st.mu_prod
            var terms = s.tape.variable(s.ex, batch_losses.value(), 1, n_batches, False)
            var average = s.tape.scale(s.ex, s.tape.sum(s.ex, terms), identical_div(Float32(1), Float32(n)))
            var a = Args()
            a.p0 = s.tape.value(average)
            a.p1 = losses.value() + e
            s.ex.launch[OP_COPY](a, 1)
            s.scheduler_step(optimizer)
        s.ex.sync()
        s.ex.download(curve, losses.value(), epochs)
    except err:
        if swapped:
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
    return (true_batches, false_batches)


# TOMBSTONE: the ops_* per-op session doors and lamb_step_py (dead native entries: reached only through their deleted exports) deleted 2026-10-26 by e4236afa3; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_sequence.patch

