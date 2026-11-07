"""Lane ne3-c-blocks-embedding: the embedding, cross-entropy and residual-dropout
entries keep their answers and refusals after the per-call copies left them.

Rows N118, N158, N160, N164 (and N117 where a device binding is present). Each
case compares the new route with an independent NumPy statement of the same
answer, or with the host-array route of the same entry.
"""
import numpy as np
import pytest

from mojolearn import _backend
from mojolearn import _training_impl as T


def _embedding_binding():
    try:
        return _backend.binding("_mojolearn_embedding", "identical")
    except Exception as error:  # noqa: BLE001 - no compiled binding on this install
        pytest.skip(f"no embedding binding: {error}")


def _addr(a):
    return a.__array_interface__["data"][0]


def test_embedding_forward_gathers_through_caller_buffers():
    ext = _embedding_binding()
    rng = np.random.default_rng(3)
    w = rng.standard_normal((37, 5)).astype("f4")
    ids = rng.integers(0, 37, size=29).astype("i4")
    y = np.empty((29, 5), dtype="f4")
    n = ext.embedding_forward([_addr(w), _addr(ids), _addr(y)], [37, 5, 29])
    assert n == 29 * 5
    np.testing.assert_array_equal(y.view("u4"), w[ids].view("u4"))


@pytest.mark.parametrize("bad_at", [0, 7, 8, 28])
def test_embedding_refuses_first_bad_id_by_position(bad_at):
    ext = _embedding_binding()
    w = np.ones((11, 3), dtype="f4")
    ids = np.zeros(29, dtype="i4")
    ids[bad_at] = 11
    ids[-1] = -1 if bad_at != 28 else 11
    y = np.empty((29, 3), dtype="f4")
    with pytest.raises(Exception, match=f"id 11 at position {bad_at} is outside"):
        ext.embedding_forward([_addr(w), _addr(ids), _addr(y)], [11, 3, 29])


def test_embedding_refuses_nonfinite_table_at_first_index():
    ext = _embedding_binding()
    w = np.ones((9, 4), dtype="f4")
    w[5, 2] = np.inf
    w[7, 0] = np.nan
    ids = np.zeros(3, dtype="i4")
    y = np.empty((3, 4), dtype="f4")
    with pytest.raises(Exception, match="infinity in W at flat index 22"):
        ext.embedding_forward([_addr(w), _addr(ids), _addr(y)], [9, 4, 3])


@pytest.mark.parametrize("accumulate", [0, 1])
@pytest.mark.parametrize("plan", [0, 1])
def test_embedding_backward_folds_into_caller_dw(accumulate, plan):
    ext = _embedding_binding()
    rng = np.random.default_rng(5)
    v, d, t = 13, 6, 40
    ids = rng.integers(0, v, size=t).astype("i4")
    dy = rng.standard_normal((t, d)).astype("f4")
    dw = rng.standard_normal((v, d)).astype("f4") if accumulate else np.full((v, d), 7.0, dtype="f4")
    start = dw.copy()
    ext.embedding_backward([_addr(dy), _addr(ids), _addr(dw)], [v, d, t, 2, accumulate, plan])
    # The profile's fold: ascending t per row, float32 at every step; the
    # padding row stored +0.0.
    want = start.copy() if accumulate else np.zeros((v, d), dtype="f4")
    for pos in range(t):
        row = int(ids[pos])
        if row != 2:
            want[row] = (want[row] + dy[pos]).astype("f4")
    want[2] = 0.0
    np.testing.assert_array_equal(dw, want)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_scalar_reductions_without_row_buffer(reduction):
    x = np.array([[.2, .7, -.3], [.9, -.2, .5], [-.7, .8, .1], [.3, .1, .2]], dtype="f4")
    y = np.array([2, 0, -100, 1], dtype="i4")
    rows = T.cross_entropy(x, y, reduction="none", numeric_mode="identical")
    total = T.cross_entropy(x, y, reduction=reduction, numeric_mode="identical")
    assert isinstance(total, float)
    assert np.asarray(rows).shape == (4,)
    folded = float(np.asarray(rows, dtype="f8").sum())
    if reduction == "mean":
        folded /= 3
    assert total == pytest.approx(folded, rel=1e-6)
    loss, grad = T.cross_entropy(x, y, reduction=reduction, return_grad=True, numeric_mode="identical")
    assert loss == total
    assert np.asarray(grad).shape == (4, 3)


def test_residual_dropout_device_arrays_match_host_arrays():
    from mojolearn._residual_dropout_impl import residual_dropout, residual_dropout_backward
    to_device = getattr(T, "to_device", None)
    if to_device is None:
        pytest.skip("no training device arrays on this install")
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 33)).astype("f4")
    r = rng.standard_normal((4, 33)).astype("f4")
    try:
        dx, dr = to_device(x), to_device(r)
    except Exception as error:  # noqa: BLE001 - CPU-only install
        pytest.skip(f"no device arrays: {error}")
    host = residual_dropout(x, r, p=0.25, seed=9, stream=1, offset=3)
    dev = residual_dropout(dx, dr, p=0.25, seed=9, stream=1, offset=3)
    np.testing.assert_array_equal(np.asarray(dev.to_host()).view("u4"), np.asarray(host).view("u4"))
    g = rng.standard_normal((4, 33)).astype("f4")
    hv, hr = residual_dropout_backward(g, p=0.25, seed=9, stream=1, offset=3)
    dv, drr = residual_dropout_backward(to_device(g), p=0.25, seed=9, stream=1, offset=3)
    np.testing.assert_array_equal(np.asarray(dv.to_host()).view("u4"), np.asarray(hv).view("u4"))
    np.testing.assert_array_equal(np.asarray(drr.to_host()).view("u4"), np.asarray(hr).view("u4"))
