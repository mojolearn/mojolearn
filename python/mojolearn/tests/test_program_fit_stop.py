# SPDX-License-Identifier: Apache-2.0
"""Program.fit's stopping criterion runs on the device: a fit with ``tol``
stops after ``n_iter_`` epochs and leaves the parameters, the loss curve and
the optimizer (its step count and schedule) exactly as a fit of that many
epochs without ``tol``, although the loop enqueued epochs past the stop
between two status reads. Tiny data."""
import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_session_call entry (rebuild it)")


def _data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3)).astype(np.float32)
    y = (x @ np.array([[1.0], [-2.0], [0.5]], np.float32)).astype(np.float32)
    return x, y


def _fit(epochs, **stop):
    from mojolearn import ops
    x, y = _data()
    with ops.Session() as s:
        a, w = s.tensor(x), s.tensor(np.zeros((3, 1), np.float32))
        model = s.capture([a], [s.matmul(a, w)])
        opt = s.optimizer("adam", lr=0.05)
        # 40 rows in batches of 8: 5 steps an epoch, so several epochs
        # are enqueued between two reads of the loop's status.
        model.fit(x, targets=y, optimizer=opt, epochs=epochs, batch_size=8, seed=3, **stop)
        ran, curve, after = model.n_iter_, np.array(model.loss_curve_), s.numpy(w)
        # The optimizer continues from the stopping epoch's state.
        model.fit(x, targets=y, optimizer=opt, epochs=1, batch_size=8, seed=3)
        return ran, curve, after, s.numpy(w)


def test_tol_stops_on_the_device_with_the_stopping_epochs_state():
    # tol 1e3: every epoch after the first fails to improve by tol, so two
    # stalled epochs stop the fit after epoch 3.
    ran, curve, after, continued = _fit(12, tol=1e3, n_iter_no_change=2)
    assert ran == 3 and curve.shape == (3,)
    ref_ran, ref_curve, ref_after, ref_continued = _fit(3)
    assert ref_ran == 3
    np.testing.assert_array_equal(curve, ref_curve)
    np.testing.assert_array_equal(after, ref_after)
    np.testing.assert_array_equal(continued, ref_continued)


def test_no_tol_runs_every_epoch():
    ran, curve, _, _ = _fit(4)
    assert ran == 4 and curve.shape == (4,)
    ran, curve, _, _ = _fit(4, tol=-1e30, n_iter_no_change=1)
    assert ran == 4 and curve.shape == (4,)
