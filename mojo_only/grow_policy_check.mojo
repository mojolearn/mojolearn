# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`grow_policy` on `train()`: the three policies are three different models.

    pixi run check-grow-policy

NO CATBOOST COUNTERPART: a gate, so `mojo_only/`. DEVIATION 259 put
`grow_policy` / `max_leaves` / `min_data_in_leaf` on `train()` and the
Python surface; `check-depthwise` and `check-lossguide` gate the SEARCHER,
this gates the BOOSTING LOOP's dispatch on the policy, the non-symmetric
model's apply, and its text round trip. Reach is per branch
(`reached-but-inert`), and the claims are contrasts rather than values
because the expected values would be our own tally:

1. REACH, BY CONTRAST. One fit per policy on the same data, same seed,
   depth 6 (so Lossguide's default 31 leaves binds): SymmetricTree,
   Depthwise and Lossguide must produce THREE DIFFERENT prediction
   vectors. A policy that fell through to the symmetric loop would tie
   with it here. At depth 4 with max_leaves 16 the two non-symmetric
   policies were MEASURED to coincide on this fixture (printed, not
   asserted): every leaf improves at every level, so "split every
   improving leaf" and "split the best leaf until the budget" close on
   the same full tree. Both non-symmetric models must
   also LEARN (last loss well under the first) -- a tree that dispatches
   and then applies zeros would pass the contrast and fail this.
2. THE MODEL SHAPE. The Depthwise and Lossguide ensembles are
   non-oblivious (`is_oblivious()` false, every weak model a
   `TNonSymmetricTree`), and the symmetric one is oblivious; the
   non-symmetric trees are RAGGED -- at least one tree has a leaf count
   that is not `1 << depth` -- which is the property an oblivious tree
   cannot have.
3. LOSSGUIDE'S KNOBS MOVE THE MODEL. `max_leaves` 4 vs 31 differ;
   `min_data_in_leaf` 1 vs 500 differ under Lossguide.
4. TRAIN/PREDICT CONSISTENCY on the non-symmetric shape: `predict_floats`
   on the training rows reproduces the fit's final loss, for both
   policies. This is the apply -- `add_non_symmetric_tree_doc_parallel` --
   agreeing with the partition-wise cursor update the fit used.
5. TEXT ROUND TRIP: `model_text` -> `load_model_text` -> `predict_floats`
   is bit-identical to the in-memory model's predictions, for both
   non-symmetric policies, and the text carries `ntree` records (not
   `tree`). A round trip that silently re-read a non-symmetric model as
   oblivious would fail the prediction identity.
6. THE REFUSALS, BY NAME: `use_pointwise_searcher` under a non-symmetric
   policy, `min_data_in_leaf != 1` under SymmetricTree, `max_leaves` not
   `1 << depth` under Depthwise, `grow_policy="Region"`, and MultiClass
   under Lossguide (their trainer registry has no such entry) all raise.
7. RUN-TO-RUN CONTROL AT A SIZE THAT REACHES THE 8-BIT HISTOGRAM ARM:
   20,000 rows x 24 hashed features at 128 borders, two trees, fitted
   twice per non-symmetric policy, UNTRACED, must give bit-identical
   predictions. This is the control that found DEVIATION 261 (one staging
   pair reused for three id lists per level in the non-symmetric driver;
   the shared-Int32 hist_2 arms disagreed with themselves at exactly this
   shape and at no smaller one) and the one the traced cards cannot be,
   because a trace drains at every record.
"""

from max.gpu.host import DeviceContext

from gbdt.models.model_text import load_model_text, model_text
from gbdt.train import TrainedModel, predict_floats, train


comptime N_ROWS = 4096
comptime N_FEATURES = 8


def build_x() -> List[Float32]:
    """`train_api_check`'s fixture: hashed pseudo-uniform columns and a
    3-category code in column 7."""
    var x = List[Float32]()
    for feat in range(N_FEATURES):
        for r in range(N_ROWS):
            var v: Float32
            if feat == 7:
                v = Float32((r * 7 + feat) % 3)
            else:
                var h = (r * 2654435761 + feat * 97003) % 10000
                v = Float32(h) / Float32(10000.0) - Float32(0.5)
            x.append(v)
    return x^


def build_y(x: List[Float32]) -> List[Float32]:
    var y = List[Float32]()
    for r in range(N_ROWS):
        var f0 = x[0 * N_ROWS + r]
        var f3 = x[3 * N_ROWS + r]
        var f5 = x[5 * N_ROWS + r]
        var c = x[7 * N_ROWS + r]
        # a NON-ADDITIVE target, so a ragged tree has something an
        # oblivious one does not capture at the same depth
        var target = Float32(3.0) * f0 - Float32(2.0) * f3 + (
            Float32(4.0) * f5 if f0 > Float32(0.0) else Float32(0.0)
        ) + (Float32(1.0) if c == Float32(1.0) else Float32(0.0))
        y.append(target)
    return y^


def fit_policy(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    policy: String,
    max_leaves: Int = -1,
    min_data_in_leaf: Int = 1,
    depth: Int = 4,
) raises -> TrainedModel:
    var one_hot = List[Bool]()
    for feat in range(N_FEATURES):
        one_hot.append(feat == 7)
    return train(
        ctx, x, y, N_ROWS, N_FEATURES,
        border_count=32,
        n_estimators=12,
        max_depth=depth,
        learning_rate=Float32(0.3),
        one_hot=one_hot,
        grow_policy=policy,
        max_leaves=max_leaves,
        min_data_in_leaf=min_data_in_leaf,
    )


def differ(a: List[Float32], b: List[Float32]) raises -> Bool:
    if len(a) != len(b):
        raise Error("prediction lengths differ")
    for i in range(len(a)):
        if a[i] != b[i]:
            return True
    return False


def mse(p: List[Float32], y: List[Float32]) -> Float64:
    var se = Float64(0.0)
    for r in range(len(y)):
        var d = Float64(p[r]) - Float64(y[r])
        se += d * d
    return se / Float64(len(y))


def expect_raise(what: String, ok: Bool, msg: String) raises:
    if ok:
        raise Error("expected a refusal for " + what + " and got none")
    var head = msg
    if msg.byte_length() > 72:
        head = String(msg[byte=0:72])
    print("  refused by name:", what, "--", head)


def main() raises:
    print("grow_policy on train(): three policies, one loop")
    var ctx = DeviceContext()
    var x = build_x()
    var y = build_y(x)

    # depth 6: Depthwise may reach 64 leaves, Lossguide stops at its
    # default 31, so the two CANNOT coincide; see the measurement below
    # for what happens at a depth the leaf budget does not bind
    var sym = fit_policy(ctx, x, y, "SymmetricTree", depth=6)
    var dw = fit_policy(ctx, x, y, "Depthwise", depth=6)
    var lg = fit_policy(ctx, x, y, "Lossguide", depth=6)

    # ---- claim 1: reach by contrast, and both learn
    var p_sym = predict_floats(ctx, sym, x, N_ROWS)
    var p_dw = predict_floats(ctx, dw, x, N_ROWS)
    var p_lg = predict_floats(ctx, lg, x, N_ROWS)
    if not differ(p_sym, p_dw):
        raise Error("Depthwise predictions equal SymmetricTree's")
    if not differ(p_sym, p_lg):
        raise Error("Lossguide predictions equal SymmetricTree's")
    if not differ(p_dw, p_lg):
        raise Error("Lossguide predictions equal Depthwise's")
    for name_model in range(3):
        var first: Float64
        var last: Float64
        if name_model == 0:
            first = sym.losses[0]
            last = sym.losses[len(sym.losses) - 1]
        elif name_model == 1:
            first = dw.losses[0]
            last = dw.losses[len(dw.losses) - 1]
        else:
            first = lg.losses[0]
            last = lg.losses[len(lg.losses) - 1]
        if not (last < first / 4.0):
            raise Error(
                "policy " + String(name_model) + " did not learn: "
                + String(first) + " -> " + String(last)
            )
    print(
        "  claim 1 OK: three distinct prediction vectors; final losses"
        " sym", sym.losses[len(sym.losses) - 1],
        "depthwise", dw.losses[len(dw.losses) - 1],
        "lossguide", lg.losses[len(lg.losses) - 1],
    )

    # ---- claim 2: the shape
    if not sym.model.is_oblivious():
        raise Error("the SymmetricTree ensemble is not oblivious")
    if dw.model.is_oblivious() or lg.model.is_oblivious():
        raise Error("a non-symmetric ensemble reads as oblivious")
    if dw.model.size() != 12 or lg.model.size() != 12:
        raise Error("non-symmetric ensembles do not hold 12 trees")
    var ragged = False
    for t in range(dw.model.size()):
        if dw.model.non_symmetric_models[t].bin_count() != (1 << 6):
            ragged = True
    if not ragged:
        raise Error(
            "every Depthwise tree has exactly 64 leaves at depth 6; on this"
            " fixture the frontier should be ragged somewhere"
        )
    var lg_leaves_ok = True
    for t in range(lg.model.size()):
        if lg.model.non_symmetric_models[t].bin_count() > 31:
            lg_leaves_ok = False
    if not lg_leaves_ok:
        raise Error("a Lossguide tree exceeds the default max_leaves 31")
    print("  claim 2 OK: non-oblivious, ragged, within max_leaves")

    # ---- claim 3: Lossguide's knobs move the model
    # THE MEASUREMENT the brief asked for: Depthwise and Lossguide at a
    # depth the leaf budget does not bind (depth 4, max_leaves 16 >= 2^4).
    # Their SelectLeavesToSplit differ -- every improving leaf vs the one
    # best leaf, no sign test under Lossguide (LOSSGUIDE.md 1) -- so the
    # two trees coincide only when every leaf at every level improves and
    # the depth bound closes both. Printed, not asserted: it is a property
    # of the fixture, not of the port.
    var dw4 = fit_policy(ctx, x, y, "Depthwise", depth=4)
    var lg16 = fit_policy(ctx, x, y, "Lossguide", depth=4, max_leaves=16)
    var p_dw4 = predict_floats(ctx, dw4, x, N_ROWS)
    var p_lg16 = predict_floats(ctx, lg16, x, N_ROWS)
    print(
        "  measured: Depthwise(depth 4) vs Lossguide(depth 4, max_leaves 16)"
        " predictions",
        "DIFFER" if differ(p_dw4, p_lg16) else "COINCIDE",
        "on this fixture",
    )

    var lg4 = fit_policy(ctx, x, y, "Lossguide", max_leaves=4, depth=6)
    var p_lg4 = predict_floats(ctx, lg4, x, N_ROWS)
    if not differ(p_lg, p_lg4):
        raise Error("max_leaves 4 and 31 give the same Lossguide model")
    for t in range(lg4.model.size()):
        if lg4.model.non_symmetric_models[t].bin_count() > 4:
            raise Error("a max_leaves=4 Lossguide tree has more than 4 leaves")
    var lg500 = fit_policy(
        ctx, x, y, "Lossguide", min_data_in_leaf=500, depth=6
    )
    var p_lg500 = predict_floats(ctx, lg500, x, N_ROWS)
    if not differ(p_lg, p_lg500):
        raise Error(
            "min_data_in_leaf 1 and 500 give the same Lossguide model"
        )
    var dw500 = fit_policy(
        ctx, x, y, "Depthwise", min_data_in_leaf=500, depth=6
    )
    var p_dw500 = predict_floats(ctx, dw500, x, N_ROWS)
    if not differ(p_dw, p_dw500):
        raise Error(
            "min_data_in_leaf 1 and 500 give the same Depthwise model"
        )
    print("  claim 3 OK: max_leaves and min_data_in_leaf move the models")

    # ---- claim 4: train/predict consistency on the non-symmetric shape
    for which in range(2):
        var last = (
            dw.losses[len(dw.losses) - 1] if which == 0
            else lg.losses[len(lg.losses) - 1]
        )
        var pm = mse(p_dw if which == 0 else p_lg, y)
        var drift = pm - last
        if drift < 0:
            drift = -drift
        if drift > 1e-9 + 1e-5 * last:
            raise Error(
                "predict_floats does not reproduce the "
                + String("Depthwise" if which == 0 else "Lossguide")
                + " fit: " + String(pm) + " vs " + String(last)
            )
    print("  claim 4 OK: predict reproduces both fits' final loss")

    # ---- claim 5: text round trip
    for which in range(2):
        var text = model_text(dw if which == 0 else lg)
        if text.find("\nntree ") < 0 or text.find("\ntree ") >= 0:
            raise Error("the non-symmetric model text lacks ntree records")
        var back = load_model_text(text)
        if back.model.is_oblivious():
            raise Error("the loaded non-symmetric model reads as oblivious")
        var pb = predict_floats(ctx, back, x, N_ROWS)
        if differ(pb, p_dw if which == 0 else p_lg):
            raise Error(
                "the round-tripped "
                + String("Depthwise" if which == 0 else "Lossguide")
                + " model predicts differently"
            )
        if model_text(back) != text:
            raise Error("the model text does not re-serialize identically")
    print("  claim 5 OK: ntree text round trip is bit-identical")

    # ---- claim 6: the refusals
    var ok = True
    var msg = String("")
    try:
        _ = train(
            ctx, x, y, N_ROWS, N_FEATURES, border_count=32, n_estimators=1,
            max_depth=3, grow_policy="Depthwise", use_pointwise_searcher=True,
        )
    except e:
        ok = False
        msg = String(e)
    expect_raise("pointwise searcher + Depthwise", ok, msg)
    ok = True
    msg = String("")
    try:
        _ = train(
            ctx, x, y, N_ROWS, N_FEATURES, border_count=32, n_estimators=1,
            max_depth=3, grow_policy="SymmetricTree", min_data_in_leaf=5,
        )
    except e:
        ok = False
        msg = String(e)
    expect_raise("min_data_in_leaf under SymmetricTree", ok, msg)
    ok = True
    msg = String("")
    try:
        _ = train(
            ctx, x, y, N_ROWS, N_FEATURES, border_count=32, n_estimators=1,
            max_depth=3, grow_policy="Depthwise", max_leaves=5,
        )
    except e:
        ok = False
        msg = String(e)
    expect_raise("max_leaves 5 under Depthwise", ok, msg)
    ok = True
    msg = String("")
    try:
        _ = train(
            ctx, x, y, N_ROWS, N_FEATURES, border_count=32, n_estimators=1,
            max_depth=3, grow_policy="Region",
        )
    except e:
        ok = False
        msg = String(e)
    expect_raise("grow_policy Region", ok, msg)
    ok = True
    msg = String("")
    try:
        var ymc = List[Float32]()
        for r in range(N_ROWS):
            ymc.append(Float32(r % 3))
        _ = train(
            ctx, x, ymc, N_ROWS, N_FEATURES, border_count=32,
            n_estimators=1, max_depth=3, grow_policy="Lossguide",
            loss="MultiClass",
        )
    except e:
        ok = False
        msg = String(e)
    expect_raise("MultiClass under Lossguide", ok, msg)
    print("  claim 6 OK: five refusals, each by name")

    # ---- claim 7: run-to-run control at the 8-bit histogram shape
    comptime CR = 20000
    comptime CF = 24
    var cx = List[Float32]()
    var cy = List[Float32]()
    var acc = List[Float64]()
    for _ in range(CR):
        acc.append(0.0)
    for f in range(CF):
        for r in range(CR):
            var h = UInt32(r * 2654435761 + f * 40503 + 0x9E3779B9)
            h ^= h << 13
            h ^= h >> 17
            h ^= h << 5
            var v = Float32(Int(h % 1024)) / Float32(1024.0)
            cx.append(v)
            acc[r] += Float64(v) * Float64((f * 7) % 11)
    for r in range(CR):
        cy.append(Float32(acc[r]))
    for which in range(2):
        var pol = String("Depthwise") if which == 0 else String("Lossguide")
        var p_a = List[Float32]()
        for rep in range(2):
            var tm = train(
                ctx, cx, cy, CR, CF, border_count=128, n_estimators=2,
                max_depth=6, learning_rate=Float32(0.3),
                random_seed=UInt64(7), grow_policy=pol,
            )
            var p = predict_floats(ctx, tm, cx, CR)
            if rep == 0:
                p_a = p^
            elif differ(p_a, p):
                raise Error(
                    pol + " at 20000 x 24 x 128 borders disagrees with"
                    " itself run to run (DEVIATION 261's shape)"
                )
    print("  claim 7 OK: both non-symmetric policies are run-to-run"
          " identical at the 8-bit histogram shape")
    print("grow_policy check: PASS")
