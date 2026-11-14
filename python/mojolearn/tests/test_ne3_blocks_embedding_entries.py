"""Lane ne3-c-blocks-embedding: the embedding, cross-entropy and residual-dropout
entries keep their answers and refusals after the per-call copies left them.

Rows N118, N158, N160, N164. The embedding gather and fold answers and the
residual-dropout device/host agreement are checked natively in
checks/native_tests/embedding_ne3_entries_gpu_suite.mojo (2026-11-14); this
file keeps the Python entries' refusals and the cross-entropy reductions'
Python results.
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
