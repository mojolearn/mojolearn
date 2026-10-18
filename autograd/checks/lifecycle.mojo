# SPDX-License-Identifier: Apache-2.0
from autograd import Tape, Tensor
from sequence.exec_trait import Exec
from sequence.ops import Args
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
