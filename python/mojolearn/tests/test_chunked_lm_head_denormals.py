"""Public chunked-head folds obey identical's signed-zero denormal policy."""
import numpy as np
import pytest


def _flush_bits(a):
    out = a.copy()
    bits = out.view(np.uint32)
    subnormal = ((bits & 0x7f800000) == 0) & ((bits & 0x7fffff) != 0)
    bits[subnormal] &= np.uint32(0x80000000)
    return out


@pytest.mark.parametrize("rows,vocab,width", [(3, 11, 7), (9, 37, 17), (17, 67, 31)])
def test_subnormal_operands_equal_signed_zero_with_unchanged_targets(rows, vocab, width):
    from mojolearn import training

    rng = np.random.default_rng(704)
    hidden = rng.uniform(-1, 1, (rows, width)).astype(np.float32)
    weight = rng.uniform(-0.5, 0.5, (vocab, width)).astype(np.float32)
    # Whole feature columns include both signs and tiny/near-normal mantissas.
    # The remaining ordinary columns keep this a nonconstant softmax problem.
    words = np.array([1, 0x80000001, 0x007fffff, 0x807fffff], dtype=np.uint32)
    hidden[:, :2] = np.resize(words, (rows, 2)).view(np.float32)
    weight[:, :2] = np.resize(words[::-1], (vocab, 2)).view(np.float32)
    targets = rng.integers(0, vocab, rows, dtype=np.int32)
    flushed_h, flushed_w = _flush_bits(hidden), _flush_bits(weight)
    assert hidden.tobytes() != flushed_h.tobytes()
    assert weight.tobytes() != flushed_w.tobytes()
    actual = training.chunked_lm_head_loss(
        hidden, weight, targets, return_grad=True, numeric_mode="identical")
    expected = training.chunked_lm_head_loss(
        flushed_h, flushed_w, targets, return_grad=True, numeric_mode="identical")
    alone = training.chunked_lm_head_loss(hidden, weight, targets, numeric_mode="identical")
    assert np.float32(alone).tobytes() == np.float32(actual[0]).tobytes()
    for name, left, right in zip(("loss", "d_hidden", "d_weight"), actual, expected):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        assert np.all(np.isfinite(left)), name
        assert left.shape == right.shape, name
        assert left.tobytes() == right.tobytes(), name
