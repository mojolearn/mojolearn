# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the gbdt-ordered-rmse, gbdt-feature-freq,
gbdt-pointwise-l2-bayesian-eval and gbdt-categorical-ctr lanes, through the
public estimators where the gbdt host binding is built and the package took
the CPU-only path (skipped, and SAID to be skipped, otherwise): a small
OrderedRMSE fit survives a save and load, and the CTR model's predictions
repeat and score its eval curve. The two-fit repeats and the refusals by
name are the native check `gbdt_public_fits`. It is a plumbing check; the
bit claim against the GPU columns is the release verification's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_gbdt_ordered
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import os
import sys
import tempfile

import mojolearn
from mojolearn import _backend


def test_ordered_save_load_and_ctr_predictions_on_the_host_when_built():
    """The boundary arms of the host check: the OrderedRMSE archive round
    trip and the CTR model's predictions (served by the host CTR door). The
    two-fit repeats, the weighted and unweighted refusals, the one-hot and
    CTR model texts, the CTR eval curve and the best-model tree count are the
    native check `gbdt_public_fits_cpu` (lane tmig-trees, 2026-11-14)."""
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built():
        print("SKIP: the gbdt host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gbdt_host")
    assert not bool(module.gbdt_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1200, 6)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1]).astype(np.float32)
    perm = rng.permutation(1200)
    m = mojolearn.OrderedRMSE(n_estimators=3, max_depth=3).fit(x, y, permutation=perm)
    expected = np.asarray(m.predict(x)).tobytes()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ordered.npz")
        m.save(path)
        back = mojolearn.OrderedRMSE.load(path)
        assert np.asarray(back.predict(x)).tobytes() == expected

    coded = np.column_stack([
        (x[:, 2] > 0).astype(np.float32),
        ((x[:, 3] > 0).astype(np.int32) * 2 + (x[:, 4] > 0)).astype(np.float32),
        x[:, :4],
    ]).astype(np.float32)
    yc = (x[:, 0] > 0).astype(np.int32)
    det = dict(bootstrap_type="No", random_strength=0.0)
    preds = [np.asarray(mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="Logloss",
                                                   cat_features=[1], **det).fit(coded, yc).predict(coded)).tobytes()
             for _ in range(2)]
    assert preds[0] == preds[1], "two CTR host fits predicted different bytes"
    # The evaluation cursor must use final CTR tables, as public predict
    # does, rather than the ordered learn-only training columns.
    ex_ctr = coded[:300].copy()
    ex_ctr[::7, 1] = 31
    ey_ctr = 1 - yc[:300]
    heldout = mojolearn.GradientBoosting(
        n_estimators=2, max_depth=3, loss="Logloss", cat_features=[1],
        use_best_model=False, **det,
    ).fit(coded, yc, eval_set=(ex_ctr, ey_ctr))
    curve = np.asarray(heldout.test_loss_curve_)
    margins = np.asarray(heldout.predict(ex_ctr), dtype=np.float64)
    expected_loss = np.mean(np.logaddexp(0.0, margins) - ey_ctr * margins)
    np.testing.assert_allclose(curve[-1], expected_loss, rtol=2e-6, atol=1e-7)


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
