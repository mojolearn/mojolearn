"""MultiClass end to end: train, predict, and make the two agree.

    pixi run check-multiclass-train

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`.

WHAT IS UNDER TEST that the kernel-level checks cannot reach: the WIRING.
`check-multilogit`, `check-multiclass-oracle` and `check-multiclass-score`
each gate one piece against libm or against algebra. None of them can see a
plane-major buffer read as bin-major, a model whose `dim` disagrees with its
leaf count, or a predict path that drops the pinned class -- because each of
those is a defect in how the pieces are JOINED.

THE GATE THAT FINDS THOSE IS THE CROSS-CHECK BETWEEN THE TWO APPLY PATHS.
Training updates the cursor through `add_model_value_kernel`, which walks
PARTITIONS -- each leaf's rows are already gathered, so it never evaluates a
split. Prediction updates it through `compute_bins_and_add_kernel`, which
walks ROWS and re-derives each row's leaf from the compressed index. They
are different kernels reading different buffers in different orders, and on
the learn set they must produce the same cursor. So:

    the loss `fit` reports, computed on the device from the TRAINING cursor

must equal

    the loss this file computes on the HOST from `predict_multi_floats`

to the precision float32 allows. A layout confusion between the two paths
moves one and not the other. That is the same property `boosting_check`
pins for the single-dimensional path, extended to the dimension that has
two layouts in play (`leaf_values` bin-major, cursor plane-major).

THE OTHER GATES, all analytic:

1. **Probabilities sum to one**, every row. The softmax is over
   `numClasses`, of which only `numClasses - 1` are stored; a port that
   forgot the pinned class's `exp(-maxApprox)` term fails this.
2. **The model's `dim` is `numClasses - 1`**, not `numClasses`. The leaf
   carries `numClasses` inside the WALKER and `numClasses - 1` once
   `MakeEstimationResult` has projected it, and the model stores the
   projected one.
3. **A LEARNABLE label is learned and a RANDOM one is not.** On a label
   that is a deterministic function of one feature, argmax accuracy must
   reach far above chance; on a shuffled label of the same distribution it
   must stay near chance. The second half is the control that makes the
   first half mean something -- a predictor that always returned class 0
   would pass an accuracy floor on a skewed fixture and fail this.
4. **The loss falls.**

SABOTAGES:

    T1  the host loss computed with the class planes ROTATED
    T2  the host loss computed with the pinned class taken as class 0
        rather than as the last

Both perturb the CHECK's expectation, not the trainer, and both must break
the two-path agreement -- which is what proves that agreement is reading
the class assignment and not merely a magnitude.
"""

from max.gpu.host import DeviceContext
from std.math import log

from gbdt.models.cuda.evaluator import pack_model_for_evaluator
from gbdt.models.model_text import load_model_text, model_text
from gbdt.train import (
    multiclass_probabilities,
    predict_multi_floats,
    train,
)

comptime N_ROWS = 4096
comptime N_FEATURES = 5


def splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def build_x() -> List[Float32]:
    var x = List[Float32]()
    for c in range(N_FEATURES):
        for r in range(N_ROWS):
            var h = splitmix(
                UInt64(0x51) ^ UInt64(r * 2654435761 + c * 40503)
            )
            x.append(Float32(Int(h % UInt64(1000))) / Float32(1000.0))
    return x^


def learnable_labels(num_classes: Int) -> List[Float32]:
    """The class is a deterministic function of FEATURE 0's bucket, so a
    tree that finds feature 0 can be exactly right."""
    var y = List[Float32]()
    for r in range(N_ROWS):
        var h = splitmix(UInt64(0x51) ^ UInt64(r * 2654435761))
        var b = Int(h % UInt64(1000))
        y.append(Float32(b * num_classes // 1000))
    return y^


def random_labels(num_classes: Int) -> List[Float32]:
    """THE CONTROL: the same class distribution, independent of every
    feature. Accuracy here must stay near chance."""
    var y = List[Float32]()
    for r in range(N_ROWS):
        var h = splitmix(UInt64(0x9E) ^ UInt64(r * 7919))
        y.append(Float32(Int(h % UInt64(UInt(num_classes)))))
    return y^


def host_multiclass_loss(
    probs: List[Float32],
    labels: List[Float32],
    num_classes: Int,
    sabotage: Int,
) -> Float64:
    """`-mean(log p[label])`, which is what their `functionValue` is.

    Their kernel accumulates `w * (classApprox - log(sumExpApprox))`
    (`multilogit.cu:87-88`), which is `w * log p[targetClass]`; `fit`
    negates it and divides by the row count. So this and the reported loss
    are the same quantity computed by two different routes over two
    different cursors.
    """
    var acc = Float64(0.0)
    for r in range(N_ROWS):
        var lab = Int(labels[r])
        # T1: rotate the class planes. T2: read the pinned class as 0.
        if sabotage == 1:
            lab = (lab + 1) % num_classes
        elif sabotage == 2:
            lab = (num_classes - 1) - lab
        var p = Float64(probs[r * num_classes + lab])
        if p < 1e-30:
            p = 1e-30
        acc += log(p)
    return -acc / Float64(N_ROWS)


def accuracy(
    probs: List[Float32], labels: List[Float32], num_classes: Int
) -> Float64:
    var correct = 0
    for r in range(N_ROWS):
        var best = 0
        for k in range(1, num_classes):
            if probs[r * num_classes + k] > probs[r * num_classes + best]:
                best = k
        if best == Int(labels[r]):
            correct += 1
    return Float64(correct) / Float64(N_ROWS)


def check_one_vs_all_train(ctx: DeviceContext) raises -> Int:
    """`MultiClassOneVsAll` end to end, and what it does NOT share.

    The gates that carry over: the loss falls, a learnable label is
    learned while a random control is not, the two apply paths agree, and
    the save/load round trip is bit-identical.

    The gates that DO NOT apply, and saying so is the point:

      * `dim` is `numClasses`, not `numClasses - 1`
        (`multiclass_targets.h:129-134`). There is no pinned class.
      * There is therefore NO gauge to fix: `MakeEstimationResult` is the
        identity, and adding a constant to a leaf's approxes CHANGES the
        prediction rather than leaving it alone. A check that asserted
        shift-invariance here would be asserting something false.
      * The probabilities are `numClasses` INDEPENDENT sigmoids and do NOT
        sum to one. `multiclass_probabilities`' softmax is MultiClass's
        and must not be applied to this model.
    """
    var bad = 0
    var x = build_x()
    var nc = 5
    var y = learnable_labels(nc)
    var tm = train(
        ctx, x, y, N_ROWS, N_FEATURES,
        border_count=32, n_estimators=20, max_depth=4,
        loss="MultiClassOneVsAll", learning_rate=Float32(0.3),
    )
    var dim = tm.model.weak_models[0].dim
    if dim != nc:
        print("  FAIL one-vs-all dim is", dim, "and should be", nc)
        bad += 1
    var first = tm.losses[0]
    var last = tm.losses[len(tm.losses) - 1]
    if not (last < first):
        print("  FAIL one-vs-all loss did not fall:", first, "->", last)
        bad += 1

    var ap = predict_multi_floats(ctx, tm, x, N_ROWS)
    # THE ARGMAX IS OVER THE RAW APPROXES, because each plane's sigmoid is
    # monotone in its own approx and the classes are independent -- there
    # is no shared denominator to normalise by.
    var correct = 0
    for r in range(N_ROWS):
        var best = 0
        for k in range(1, nc):
            if ap[r * nc + k] > ap[r * nc + best]:
                best = k
        if best == Int(y[r]):
            correct += 1
    var acc = Float64(correct) / Float64(N_ROWS)
    if acc < 1.0 / Float64(nc) + 0.25:
        print("  FAIL one-vs-all accuracy", acc, "is not above chance")
        bad += 1

    var yr = random_labels(nc)
    var tmr = train(
        ctx, x, yr, N_ROWS, N_FEATURES,
        border_count=32, n_estimators=20, max_depth=4,
        loss="MultiClassOneVsAll", learning_rate=Float32(0.3),
    )
    var apr = predict_multi_floats(ctx, tmr, x, N_ROWS)
    var rc = 0
    for r in range(N_ROWS):
        var best = 0
        for k in range(1, nc):
            if apr[r * nc + k] > apr[r * nc + best]:
                best = k
        if best == Int(yr[r]):
            rc += 1
    var accr = Float64(rc) / Float64(N_ROWS)
    if accr > acc - 0.2:
        print(
            "  FAIL the one-vs-all RANDOM control reached", accr,
            "against the learnable", acc,
        )
        bad += 1

    var text = model_text(tm)
    var reloaded = load_model_text(text)
    var ap2 = predict_multi_floats(ctx, reloaded, x, N_ROWS)
    var moved = 0
    for i in range(len(ap)):
        if ap[i] != ap2[i]:
            moved += 1
    if moved != 0:
        print("  FAIL one-vs-all save/load moved", moved, "approxes")
        bad += 1

    if bad == 0:
        print(
            "  ok   one-vs-all: dim", dim, ", loss", first, "->", last,
            ", accuracy", acc, "vs random", accr,
            ", save/load bit-identical",
        )
    return bad


def check_multiclass_train(ctx: DeviceContext) raises:
    var failures = 0
    var x = build_x()

    for nc in [2, 3, 7]:
        print("-- numClasses", nc, "--")
        var y = learnable_labels(nc)
        var tm = train(
            ctx, x, y, N_ROWS, N_FEATURES,
            border_count=32, n_estimators=20, max_depth=4,
            loss="MultiClass", learning_rate=Float32(0.3),
        )

        # GATE 2: the model's dimension
        var dim = tm.model.weak_models[0].dim
        if dim != nc - 1:
            print(
                "  FAIL model dim is", dim, "and should be numClasses - 1 =",
                nc - 1,
            )
            failures += 1
        var n_leaves = 1 << 4
        if len(tm.model.weak_models[0].leaf_values) != n_leaves * dim:
            print(
                "  FAIL leaf_values holds",
                len(tm.model.weak_models[0].leaf_values),
                "for", n_leaves, "leaves x", dim, "dims",
            )
            failures += 1

        # GATE 4: the loss falls
        var first = tm.losses[0]
        var last = tm.losses[len(tm.losses) - 1]
        if not (last < first):
            print("  FAIL loss did not fall:", first, "->", last)
            failures += 1

        var ap = predict_multi_floats(ctx, tm, x, N_ROWS)
        var pr = multiclass_probabilities(ap, N_ROWS, nc)

        # GATE 1: probabilities sum to one, every row
        var worst_sum = Float64(0.0)
        for r in range(N_ROWS):
            var s = Float64(0.0)
            for k in range(nc):
                s += Float64(pr[r * nc + k])
                if pr[r * nc + k] < Float32(0.0):
                    failures += 1
            var e = s - 1.0
            if e < 0.0:
                e = -e
            if e > worst_sum:
                worst_sum = e
        if worst_sum > 1e-5:
            print("  FAIL probabilities sum off by", worst_sum)
            failures += 1

        # THE CROSS-CHECK: the device's training loss against the host's
        # loss over the tree-wise apply
        var host_loss = host_multiclass_loss(pr, y, nc, 0)
        var d = host_loss - Float64(last)
        if d < 0.0:
            d = -d
        var scale = Float64(last)
        if scale < 1e-3:
            scale = 1e-3
        if d / scale > 5e-3:
            print(
                "  FAIL the two apply paths disagree: device", last,
                "host", host_loss,
            )
            failures += 1
        else:
            print(
                "  ok   loss", first, "->", last,
                "; host apply agrees to", d / scale,
            )

        # GATE 3: learnable is learned
        var acc = accuracy(pr, y, nc)
        var chance = 1.0 / Float64(nc)
        if acc < chance + 0.25:
            print("  FAIL accuracy", acc, "is not above chance", chance)
            failures += 1
        else:
            print("  ok   accuracy", acc, "against chance", chance)

        # GATE 3's CONTROL: random is not learned
        var yr = random_labels(nc)
        var tmr = train(
            ctx, x, yr, N_ROWS, N_FEATURES,
            border_count=32, n_estimators=20, max_depth=4,
            loss="MultiClass", learning_rate=Float32(0.3),
        )
        var apr = predict_multi_floats(ctx, tmr, x, N_ROWS)
        var prr = multiclass_probabilities(apr, N_ROWS, nc)
        var accr = accuracy(prr, yr, nc)
        # 20 trees at depth 4 can memorise some of 4,096 rows, so the bar
        # is "not close to the learnable case" rather than "at chance"
        if accr > acc - 0.2:
            print(
                "  FAIL the RANDOM control reached", accr,
                "against the learnable", acc,
                "-- the fixture is not discriminating",
            )
            failures += 1
        else:
            print(
                "  ok   random control", accr, "well under learnable", acc,
            )

        # THE ROUND TRIP. `check-model-io` gates the one-dimensional
        # format bit for bit; nothing gated `dim > 1`, and the writer
        # emitted `n_leaves` values per tree rather than
        # `n_leaves * dim` until this check existed -- a MultiClass model
        # would have lost every dimension but the first, silently, on save.
        var text = model_text(tm)
        var reloaded = load_model_text(text)
        var ap2 = predict_multi_floats(ctx, reloaded, x, N_ROWS)
        var moved = 0
        for i in range(len(ap)):
            if ap[i] != ap2[i]:
                moved += 1
        if moved != 0:
            print(
                "  FAIL save/load moved", moved, "of", len(ap),
                "approxes -- the multi-dimensional round trip is lossy",
            )
            failures += 1
        else:
            print(
                "  ok   save/load is BIT-IDENTICAL over", len(ap),
                "approxes",
            )

        # THE DEVICE EVALUATOR MUST REFUSE THIS MODEL, and refusing is the
        # PORTED behaviour rather than a limitation of ours: their own
        # `libs/model/cuda/evaluator.cpp:28` is
        #
        #     CB_ENSURE(ModelTrees->GetDimensionsCount() == 1,
        #               "Model is not one-dimensional, GPU evaluation is
        #                not supported yet");
        #
        # and their kernel agrees structurally -- `EvalObliviousTrees`
        # advances `leafValues += (1 << curTreeDepth)` per tree
        # (`evaluator.cu:222`), one value per leaf with no dimension
        # stride. A model whose `leaf_values` is `n_leaves * dim` fed to
        # that walk would silently predict the FIRST class's approxes.
        #
        # PORTING_RULES 8: both sides of the switch, by a named check per
        # side. `check-model-io` and `check-catboost-apply` exercise the
        # one-dimensional side; this is the other one.
        # BOTH SIDES, and numClasses = 2 is what supplies the second one:
        # a two-class MultiClass model has `dim == numClasses - 1 == 1`,
        # so it IS one-dimensional and the evaluator must PACK it. Only
        # `dim > 1` may be refused. The first version of this gate
        # expected a refusal for every MultiClass model and failed at
        # nc=2 -- correctly, because a binary MultiClass fit is an
        # ordinary one-dimensional model and refusing it would be a
        # regression rather than fidelity.
        var refused = False
        try:
            var _packed = pack_model_for_evaluator(ctx, tm.model)
        except e:
            refused = True
        if dim == 1:
            if refused:
                print(
                    "  FAIL the device evaluator refused a dim-1 model,"
                    " which is one-dimensional and must pack",
                )
                failures += 1
            else:
                print("  ok   dim 1 packs, as a one-dimensional model must")
        else:
            if not refused:
                print(
                    "  FAIL the device evaluator PACKED a dim-", dim,
                    "model; theirs refuses it and ours must",
                )
                failures += 1
            else:
                print("  ok   the device evaluator refuses dim", dim)

        # SABOTAGES, on the host expectation only
        if nc >= 3:
            for sab in [1, 2]:
                var bad_loss = host_multiclass_loss(pr, y, nc, sab)
                var bd = bad_loss - Float64(last)
                if bd < 0.0:
                    bd = -bd
                if bd / scale <= 5e-3:
                    print(
                        "  FAIL T" + String(sab),
                        "changed nothing: the cross-check is not reading"
                        " the class assignment",
                    )
                    failures += 1
                else:
                    print(
                        "  ok   T" + String(sab), "moves the host loss to",
                        bad_loss,
                    )

    print()
    print("-- MultiClassOneVsAll: no pinned class, no gauge --")
    failures += check_one_vs_all_train(ctx)

    if failures != 0:
        raise Error(
            "multiclass train check: " + String(failures) + " failures"
        )
    print()
    print("multiclass train check: PASS")


def main() raises:
    var ctx = DeviceContext()
    check_multiclass_train(ctx)
