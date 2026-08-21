"""`fit` actually runs CatBoost's single-target symmetric learner.

The last wiring step. `gbdt/methods/doc_parallel_boosting.fit` takes
`use_pointwise_searcher`, and this gate fits the SAME data both ways and
requires the two models to agree.

## What the two arms share and what they do not

    shared     the compressed index, the target, the weak-target
               computation, the bootstrap, leaf estimation, the apply,
               the loss
    different  the STRUCTURE SEARCHER, and nothing else

`check-pointwise-vs-greedy` already showed the two searchers pick identical
splits from one weak target. This gate is the consequence at the level a
user sees: if the splits agree at every level of every tree, and every other
stage is the same code, then the per-iteration losses must agree too.

**AND THAT IS WHY IT IS A GATE AND NOT A DEMO.** A boosting loop amplifies:
one different split at iteration 3 changes the residuals every later
iteration fits, so the loss curves diverge and keep diverging. Two curves
that agree to the bit for twenty iterations is a much stronger statement
than one tree matching.

## Gates

  W1  the pointwise arm actually LEARNS -- the loss falls and beats
      predicting the mean. A wired-but-broken arm that returns a constant
      model would otherwise pass W2 trivially against itself.
  W2a the FIRST tree is identical -- splits, split types and every leaf
      value -- checked before the boosting loop can amplify anything, so a
      failure names the tree rather than the curve.
  W2  the two arms' loss curves are IDENTICAL, iteration for iteration.
  W3  the arms are actually DIFFERENT CODE: this is asserted by
      construction rather than measured, and W2 is what would catch a
      `use_pointwise_searcher` that was silently ignored -- because if it
      were ignored, W2 would pass for the wrong reason. So W3 checks the
      searcher directly: it fits ONE tree each way at depth 1 and requires
      the pointwise arm to have consulted its own searcher, by asserting
      the two arms disagree when the pointwise arm is given a DIFFERENT
      score function. Same data, same everything else, one knob the greedy
      arm cannot see.
"""

from max.gpu.host import DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
)

comptime N_ROWS = 3000
comptime N_ITERS = 20
comptime MAX_DEPTH = 4


def main() raises:
    var ctx = DeviceContext()
    # SIX one-byte features: 4 fit in one cindex word, so features 8 and 9
    # live in the SECOND column of their policy. The signal is put there on
    # purpose -- see the note in pointwise_vs_greedy_check.
    var folds: List[Int] = [1, 12, 9, 20, 32, 48, 100, 64, 127, 96]
    var n_features = len(folds)
    var lay = build_layout(folds)

    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        N_ROWS * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    var host_bin = List[List[Int]]()
    for f in range(n_features):
        ref cf = lay.features[f]
        var col = List[Int]()
        for r in range(N_ROWS):
            var x = UInt32(r * 2654435761 + f * 40503 + 0x2545F491)
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            var v = Int(x % UInt32(folds[f]))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(N_ROWS), cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    var targets = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var weights = ctx.enqueue_create_buffer[DType.float32](N_ROWS)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](N_ROWS)
    var mean = Float64(0.0)
    for r in range(N_ROWS):
        var y = (
            Float32(host_bin[9][r]) * 0.4
            - Float32(host_bin[1][r]) * 1.5
            + Float32(host_bin[8][r]) * 0.6
        )
        if host_bin[9][r] > 45 and host_bin[1][r] > 6:
            y += 5.0
        ht.unsafe_ptr().unsafe_store(r, y)
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
        mean += Float64(y)
    mean /= Float64(N_ROWS)
    var variance = Float64(0.0)
    for r in range(N_ROWS):
        var d = Float64(ht.unsafe_ptr().unsafe_load(r)) - mean
        variance += d * d
    variance /= Float64(N_ROWS)
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var failures = 0

    # ---- arm A: the greedy-subsets searcher (the default) ------------
    var m_greedy = TAdditiveModel()
    var l_greedy = fit(
        m_greedy, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
        False, N_ITERS, Float32(0.3), Float32(3.0), True,
    )

    # ---- arm B: CatBoost's single-target symmetric learner -----------
    var m_pw = TAdditiveModel()
    var l_pw = fit(
        m_pw, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
        False, N_ITERS, Float32(0.3), Float32(3.0), True,
        use_pointwise_searcher=True,
    )

    # tree 0, both halves of the weak model, before the loop amplifies
    ref g0 = m_greedy.weak_models[0].structure.splits
    ref p0 = m_pw.weak_models[0].structure.splits
    ref gv = m_greedy.weak_models[0].leaf_values
    ref pv = m_pw.weak_models[0].leaf_values
    var t0 = 0
    if len(g0) != len(p0) or len(gv) != len(pv):
        print("FAIL W2: tree 0 differs in SHAPE:", len(g0), "splits /",
              len(gv), "leaves against", len(p0), "/", len(pv))
        t0 += 1
    else:
        for i in range(len(g0)):
            if (g0[i].feature_id != p0[i].feature_id
                or g0[i].bin_idx != p0[i].bin_idx
                or g0[i].split_type != p0[i].split_type):
                print("FAIL W2: tree 0 split", i, "differs")
                t0 += 1
        for i in range(len(gv)):
            if gv[i] != pv[i]:
                if t0 < 4:
                    print("FAIL W2: tree 0 leaf", i, ":", gv[i], "vs",
                          pv[i])
                t0 += 1
    if t0 != 0:
        failures += 1
    else:
        print("  ok   W2a -- tree 0 identical:", len(g0),
              "splits and all", len(gv), "leaf values")

    print("iter   greedy-subsets        pointwise")
    for i in range(len(l_pw)):
        print("  ", i + 1, "  ", l_greedy[i], "   ", l_pw[i])

    # ---------------------------------------------------------------- W1
    if len(l_pw) != N_ITERS:
        print("FAIL W1: the pointwise arm ran", len(l_pw), "iterations")
        failures += 1
    elif l_pw[len(l_pw) - 1] >= l_pw[0]:
        print("FAIL W1: the pointwise arm did not reduce the loss at all")
        failures += 1
    elif Float64(l_pw[len(l_pw) - 1]) >= variance:
        print(
            "FAIL W1: the pointwise arm did not beat predicting the mean:",
            l_pw[len(l_pw) - 1], "against", variance,
        )
        failures += 1
    else:
        print(
            "  ok   W1 -- the pointwise arm learns:", l_pw[0], "->",
            l_pw[len(l_pw) - 1], "against a mean of", variance,
        )

    # ---------------------------------------------------------------- W2
    var differ = 0
    for i in range(len(l_pw)):
        if l_greedy[i] != l_pw[i]:
            if differ < 3:
                print(
                    "   W2 iteration", i + 1, ": greedy", l_greedy[i],
                    "pointwise", l_pw[i],
                )
            differ += 1
    if differ != 0:
        print(
            "FAIL W2:", differ, "of", len(l_pw),
            "iterations differ. The two arms share every stage but the"
            " structure searcher, and check-pointwise-vs-greedy says the"
            " searchers agree -- so a divergence here is in the WIRING,"
            " not in either searcher.",
        )
        failures += 1
    else:
        print(
            "  ok   W2 --", len(l_pw),
            "iterations identical to the bit, across a boosting loop that"
            " amplifies any single differing split",
        )

    # ---------------------------------------------------------------- W3
    # the knob must actually reach the searcher: change something only the
    # pointwise arm's scorer can see and require the model to move
    var m_l2 = TAdditiveModel()
    var l_l2 = fit(
        m_l2, ctx, N_ROWS, folds, MAX_DEPTH, cindex, targets, weights,
        False, N_ITERS, Float32(0.3), Float32(3.0), True,
        score_function=SCORE_FUNCTION_L2,
        use_pointwise_searcher=True,
    )
    var moved = 0
    for i in range(len(l_l2)):
        if l_l2[i] != l_pw[i]:
            moved += 1
    if moved == 0:
        print(
            "FAIL W3: changing the score function changed NOTHING in the"
            " pointwise arm, so its searcher is not being consulted --"
            " which would also make W2 pass for the wrong reason.",
        )
        failures += 1
    else:
        print(
            "  ok   W3 -- the score function reaches the pointwise"
            " searcher:", moved, "of", len(l_l2), "iterations move",
        )

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print("fit runs CatBoost's single-target symmetric learner: W1-W3 pass")
