# SPDX-License-Identifier: Apache-2.0
from autograd import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import Args, FP, OP_COPY
from autograd.ops import ADD, TANH
from autograd.checks.support import check_values


def run_lifecycle_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var x = tape.tensor(ex, [Float32(2), Float32(-3)], 1, 2, True)
    var xx = tape.multiply(ex, x, x)
    var branch = tape.add(ex, xx, x)
    var loss = tape.sum(ex, branch)
    tape.backward(ex, loss)
    check_values("diamond", tape.gradients(ex, x), [Float32(5), Float32(-5)])
    tape.backward(ex, loss, True)
    check_values("leaf_accumulation", tape.gradients(ex, x), [Float32(10), Float32(-10)])
    tape.backward(ex, loss)
    check_values("gradient_reset", tape.gradients(ex, x), [Float32(5), Float32(-5)])

    var detached = tape.detach(ex, x)
    var product = tape.multiply(ex, x, detached)
    var detached_loss = tape.sum(ex, product)
    tape.backward(ex, detached_loss)
    check_values("detach", tape.gradients(ex, x), [Float32(2), Float32(-3)])
    var unused_negative = tape.scalar(ex, -1, True)
    var unused_log = tape.log(ex, unused_negative)
    tape.backward(ex, loss)
    check_values("unreachable_branch", tape.gradients(ex, unused_negative), [Float32(0)])

    var refusals = 0
    try:
        tape.backward(ex, branch)
    except:
        refusals += 1
    var seed_bad = tape.scalar(ex, 1)
    try:
        tape.backward_from(ex, branch, seed_bad)
    except:
        refusals += 1
    var other = Tape()
    var foreign = other.scalar(ex, 2, True)
    try:
        var invalid = tape.add(ex, x, foreign)
    except:
        refusals += 1
    try:
        var invalid = tape.tensor(ex, [Float32(1)], 2, 2)
    except:
        refusals += 1
    try:
        var invalid = tape.reshape(ex, x, 0, 2)
    except:
        refusals += 1
    try:
        var invalid = tape.sum(ex, x, 2)
    except:
        refusals += 1
    try:
        tape.backward(ex, detached)
    except:
        refusals += 1
    var constant_scalar = tape.scalar(ex, 3)
    try:
        tape.backward(ex, constant_scalar)
    except:
        refusals += 1
    # An unknown custom VJP must refuse before changing established grads.
    var custom = tape.custom(ex, [x], tape.value(x), 1, 2, 123, Args())
    var custom_loss = tape.sum(ex, custom)
    try:
        tape.backward(ex, custom_loss)
    except:
        refusals += 1
    check_values("missing_rule_preserves_grads", tape.gradients(ex, x), [Float32(5), Float32(-5)])
    tape.reset()
    try:
        var invalid = tape.value(x)
    except:
        refusals += 1
    if refusals != 10:
        raise Error("autograd did not reject every invalid graph operation")

    # A repeated native training graph must not retain one set of buffers
    # per iteration. Once shapes stabilize both slots and capacities do too.
    var slots = 0
    var capacity = 0
    for iteration in range(30):
        tape.reset()
        var p = tape.tensor(ex, [Float32(2), Float32(3)], 1, 2, True)
        var squared = tape.multiply(ex, p, p)
        var objective = tape.mean(ex, squared)
        tape.backward(ex, objective)
        var total = 0
        for n in tape._capacities:
            total += n
        if iteration == 1:
            slots = len(tape._buffers)
            capacity = total
        elif iteration > 1 and (slots != len(tape._buffers) or capacity != total):
            raise Error("repeated graph grows tape storage")
    print("PASS lifecycle refusals=10 repeated_graphs=30")


def _upload[E: Exec](mut ex: E, dst: FP, mut values: List[Float32]) raises:
    ex.upload(dst, FP(unsafe_from_address=Int(values.unsafe_ptr())), len(values))


@fieldwise_init
struct _PassRules(BackwardRules):
    """Tag 7: an identity op, its upstream copied to its input's gradient.
    Tag 8: a pass-through recorded with custom_into_shared (its gradient
    is its input's), so it writes nothing."""
    def supports(self, tag: Int) -> Bool:
        return tag == 7 or tag == 8

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == 7:
            var a = Args()
            a.p0 = upstream
            a.p1 = gradients[0]
            ex.launch[OP_COPY](a, rows[0] * cols[0])
        elif Int(gradients[0]) != Int(upstream):
            raise Error("pass-through: the shared gradient is not one buffer")


def run_alias_checks[E: Exec](mut ex: E) raises:
    """Aliasing and mutation refusals (docs/plans/NATIVE_AUTOGRAD.md,
    "Aliasing and mutation"): overlapping caller storage, a seed over
    another gradient, backward after mark_mutated or a failed
    assert_unchanged are refused; the seed-in-place case is accepted."""
    var tape = Tape()
    var free_a = ex.alloc(2)
    var free_b = ex.alloc(2)
    var free_c = ex.alloc(2)
    var free_d = ex.alloc(4)
    var refusals = 0
    var x = tape.tensor(ex, [Float32(2), Float32(-3)], 1, 2, True)
    var w = tape.tensor(ex, [Float32(1), Float32(0), Float32(0), Float32(1)], 2, 2)
    # 1-3: variable_into's gradient destination over a recorded value, over
    # a recorded gradient, and over its own data
    try:
        var bad = tape.variable_into(ex, free_a, tape.value(x), 1, 2)
    except:
        refusals += 1
    try:
        var bad = tape.variable_into(ex, free_a, tape._nodes[x._index].grad, 1, 2)
    except:
        refusals += 1
    try:
        var bad = tape.variable_into(ex, free_a, free_a, 1, 2)
    except:
        refusals += 1
    # 4-5: linear_into / unary_into outputs over a recorded value
    try:
        var bad = tape.linear_into(ex, x, w, None, 0, tape.value(x))
    except:
        refusals += 1
    try:
        var bad = tape.unary_into(ex, x, TANH, tape.value(w))
    except:
        refusals += 1
    # 6: custom_into's output and gradient destination are one range
    try:
        var bad = tape.custom_into(ex, [x], free_a, free_a, 1, 2, 7, Args())
    except:
        refusals += 1
    # 7: binary_recorded's output over a recorded gradient
    try:
        var bad = tape.binary_recorded(ex, x, x, ADD, tape._nodes[x._index].grad)
    except:
        refusals += 1
    if len(tape._nodes) != 2:
        raise Error("a refused caller-storage record changed the graph")

    # Seed in place: binary_recorded's output holds a + b (computed here as
    # literals), its gradient lives in free_c and the seed is free_c itself.
    tape.reset()
    var a = tape.tensor(ex, [Float32(1), Float32(2)], 1, 2, True)
    var b = tape.tensor(ex, [Float32(3), Float32(4)], 1, 2, True)
    var sum_values: List[Float32] = [Float32(4), Float32(6)]
    _upload(ex, free_b, sum_values)
    var seed_values: List[Float32] = [Float32(1), Float32(-2)]
    _upload(ex, free_c, seed_values)
    var s = tape.binary_recorded(ex, a, b, ADD, free_b, free_c, True)
    var seed = tape.variable(ex, free_c, 1, 2, False)
    tape.backward_from(ex, s, seed)
    check_values("seed_in_place/a", tape.gradients(ex, a), [Float32(1), Float32(-2)])
    check_values("seed_in_place/b", tape.gradients(ex, b), [Float32(1), Float32(-2)])
    # 8: a seed over another tensor's gradient (a's)
    var over = tape.variable(ex, tape.gradient(a), 1, 2, False)
    try:
        tape.backward_from(ex, s, over)
    except:
        refusals += 1
    check_values("refused_seed_preserves_grads", tape.gradients(ex, a), [Float32(1), Float32(-2)])
    # A leaf root seeded in place: accepted alone, refused with accumulate (9).
    tape.reset()
    var leaf_values: List[Float32] = [Float32(5), Float32(7)]
    _upload(ex, free_a, leaf_values)
    _upload(ex, free_c, seed_values)
    var leaf = tape.variable_into(ex, free_a, free_c, 1, 2)
    var leaf_seed = tape.variable(ex, free_c, 1, 2, False)
    tape.backward_from(ex, leaf, leaf_seed)
    check_values("leaf_seed_in_place", tape.gradients(ex, leaf), [Float32(1), Float32(-2)])
    try:
        tape.backward_from(ex, leaf, leaf_seed, True)
    except:
        refusals += 1

    # A declared shared gradient (custom_into_shared) is accepted, with the
    # seed in that one range; a partial overlap of it stays refused (13).
    tape.reset()
    var src = tape.tensor(ex, [Float32(5), Float32(7)], 1, 2, True)
    _upload(ex, free_b, seed_values)
    var ident = tape.custom_into(ex, [src], free_a, free_b, 1, 2, 7, Args())
    var through = tape.custom_into_shared(ex, [ident], free_c, ident, 1, 2, 8, Args())
    var shared_seed = tape.variable(ex, free_b, 1, 2, False)
    tape.backward_with_rules(ex, through, shared_seed, _PassRules())
    check_values("shared_gradient", tape.gradients(ex, src), [Float32(1), Float32(-2)])
    try:
        var bad = tape.custom_into(ex, [src], free_d, free_b + 1, 1, 2, 7, Args())
    except:
        refusals += 1

    # mark_mutated: an output recorded before the mark refuses (10); one
    # recorded after it differentiates.
    tape.reset()
    var p = tape.tensor(ex, [Float32(2), Float32(3)], 1, 2, True)
    var old_square = tape.multiply(ex, p, p)
    var old_loss = tape.sum(ex, old_square)
    tape.mark_mutated(p)
    try:
        tape.backward(ex, old_loss)
    except:
        refusals += 1
    var new_square = tape.multiply(ex, p, p)
    var new_loss = tape.sum(ex, new_square)
    tape.backward(ex, new_loss)
    check_values("after_mark_mutated", tape.gradients(ex, p), [Float32(4), Float32(6)])

    # watch / assert_unchanged: unchanged passes; storage rewritten outside
    # the tape is caught (11) and the old consumers then refuse (12).
    tape.reset()
    var start_values: List[Float32] = [Float32(1), Float32(2), Float32(3), Float32(4)]
    _upload(ex, free_d, start_values)
    var v = tape.variable(ex, free_d, 2, 2)
    var v_square = tape.multiply(ex, v, v)
    var v_loss = tape.sum(ex, v_square)
    tape.watch(ex, v)
    tape.assert_unchanged(ex, v)
    var changed_values: List[Float32] = [Float32(1), Float32(2), Float32(3), Float32(5)]
    _upload(ex, free_d, changed_values)
    try:
        tape.assert_unchanged(ex, v)
    except:
        refusals += 1
    try:
        tape.backward(ex, v_loss)
    except:
        refusals += 1
    tape.reset()
    if refusals != 13:
        print("FAIL alias refusals", refusals, "of 13")
        raise Error("autograd did not refuse every aliasing or mutation violation")
    tape.release(ex)
    ex.free(free_a)
    ex.free(free_b)
    ex.free(free_c)
    ex.free(free_d)
    print("PASS alias refusals=13 seed_in_place=2 shared=1")
