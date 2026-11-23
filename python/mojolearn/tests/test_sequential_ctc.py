# SPDX-License-Identifier: Apache-2.0
"""Public boundary of lane nn-losses-2 (2026-11-23): `Sequential.log_softmax(dim)`, nll and kl_div on a final
log_softmax, and `compile(loss="ctc", blank=, zero_infinity=)` with `fit(target_lengths=)`. Admission, refusals in
clear words, finite small fits, and the save_checkpoint / from_checkpoint and export / load round trips. The
numerics (float64 oracles, alignment enumeration, finite differences, host versus device words) are checked
natively (checks/native_tests/nn_losses_2.mojo)."""
import numpy as np
import pytest

from mojolearn import Sequential


def frames(n=24, steps=8, seed=0):
    """(n, steps, 4) frame sequences of 1 or 2 labels of {1, 2, 3} (blank 0), their labels (n, 2), target lengths
    and input lengths."""
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.standard_normal((n, steps, 4))).astype(np.float32)
    y = np.ones((n, 2), dtype=np.int64)
    tl = 1 + np.arange(n) % 2
    il = np.zeros(n, dtype=np.int64)
    for i in range(n):
        labels = rng.integers(1, 4, tl[i])
        y[i, :tl[i]] = labels
        seq = [0] + [int(v) for label in labels for v in (label, label, 0)]
        for t, f in enumerate(seq):
            x[i, t, f] += 1.0
        il[i] = len(seq)
    return x, y, tl, il


def ctc_model(**kw):
    return Sequential(numeric_mode="identical", input_shape=(8, 4)).gru(8).linear(4).log_softmax().compile(
        loss="ctc", optimizer="adam", lr=0.01, **kw)


def test_log_softmax_is_admitted_and_named():
    m = Sequential(numeric_mode="identical", input_shape=(3, 5)).linear(4).log_softmax(1)
    assert [layer[0] for layer in m.layers][-1] == "log_softmax"
    m.add("log_softmax")
    assert [layer[0] for layer in m.layers][-1] == "log_softmax"


@pytest.mark.parametrize("dim, words", [(0, "batch axis"), (3, "not an axis"), (1.5, "integer")])
def test_log_softmax_refuses_a_bad_dim(dim, words):
    x = np.zeros((2, 3, 5), np.float32)
    m = Sequential(numeric_mode="identical", input_shape=(3, 5))
    with pytest.raises(Exception, match=words):
        m.log_softmax(dim).flatten().linear(1).compile().fit(x, np.zeros(2, np.float32), epochs=1)


def test_log_softmax_outputs_are_log_probabilities():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 6)).astype(np.float32)
    labels = rng.integers(0, 3, 16)
    m = Sequential(numeric_mode="identical").linear(3).log_softmax().compile(loss="nll", lr=0.05)
    m = m.fit(x, labels, epochs=2, seed=1)
    p = np.asarray(m.predict_proba(x), dtype=np.float64)
    assert np.all(np.isfinite(m.loss_curve_)) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert m.predict(x).shape == (16,)


def test_kl_div_on_log_softmax_fits():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((32, 6)).astype(np.float32)
    p = rng.uniform(0.05, 1.0, (32, 3)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    m = Sequential(numeric_mode="identical").linear(3).log_softmax()
    m = m.compile(loss="kl_div", reduction="batchmean", lr=0.05).fit(x, p, epochs=3, seed=2)
    assert np.all(np.isfinite(m.loss_curve_)) and m.loss_curve_[-1] < m.loss_curve_[0]


def test_ctc_fits_and_predicts_log_probabilities():
    x, y, tl, il = frames()
    m = ctc_model().fit(x, y, epochs=3, seed=3, lengths=il, target_lengths=tl)
    curve = np.asarray(m.loss_curve_)
    assert len(curve) == 3 and np.all(np.isfinite(curve))
    out = np.asarray(m.predict(x, lengths=il))
    assert out.shape == (len(x), 8 * 4) and np.all(np.isfinite(out))
    # each step's 4 values are log-probabilities
    assert np.allclose(np.exp(out.reshape(-1, 4).astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)


def test_ctc_zero_infinity_and_sum():
    x, y, tl, _ = frames()
    y[0] = [2, 2]          # two equal labels need 3 steps;
    tl[0] = 2
    il = np.full(len(x), 8)
    il[0] = 2              # sample 0 has 2: no alignment
    a = ctc_model(zero_infinity=True).fit(x, y, epochs=1, seed=4, lengths=il, target_lengths=tl)
    assert np.all(np.isfinite(a.loss_curve_)) and np.all(np.isfinite(a.params_))
    b = ctc_model(reduction="sum").fit(x, y, epochs=1, seed=4, lengths=il, target_lengths=tl)
    assert np.isinf(b.loss_curve_[0])


@pytest.mark.parametrize("change, words", [
    ("no_target_lengths", "target_lengths"),
    ("blank_label", "blank"),
    ("long_target", "target_lengths must be whole numbers"),
    ("bad_blank", "blank"),
    ("batchmean", "'mean' or 'sum'"),
    ("validation", "validation_data"),
])
def test_ctc_refusals_in_words(change, words):
    x, y, tl, il = frames()
    kw = dict(lengths=il, target_lengths=tl)
    model = ctc_model
    if change == "no_target_lengths":
        kw.pop("target_lengths")
    elif change == "blank_label":
        y[3, 0] = 0
    elif change == "long_target":
        tl = tl.copy()
        tl[2] = 3
        kw["target_lengths"] = tl
    elif change == "bad_blank":
        model = lambda: ctc_model(blank=4)
    elif change == "batchmean":
        model = lambda: ctc_model(reduction="batchmean")
    elif change == "validation":
        kw = dict(target_lengths=tl, validation_data=(x, y))
    with pytest.raises(Exception, match=words):
        model().fit(x, y, epochs=1, **kw)


def test_target_lengths_refused_without_ctc():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 4)).astype(np.float32)
    m = Sequential(numeric_mode="identical").linear(1).compile()
    with pytest.raises(Exception, match="ctc"):
        m.fit(x, np.zeros(8, np.float32), epochs=1, target_lengths=np.ones(8, np.int64))


def test_a_sequence_of_log_probabilities_takes_ctc():
    x, _, _, _ = frames()
    m = Sequential(numeric_mode="identical", input_shape=(8, 4)).linear(4).log_softmax().compile(loss="mse")
    with pytest.raises(Exception, match="ctc"):
        m.fit(x, np.zeros((len(x), 32), np.float32), epochs=1)


def test_ctc_checkpoint_and_export_round_trip(tmp_path):
    x, y, tl, il = frames()
    m = ctc_model(blank=0, zero_infinity=True).fit(x, y, epochs=2, seed=6, lengths=il, target_lengths=tl)
    want = np.asarray(m.predict(x, lengths=il))
    m.save_checkpoint(str(tmp_path / "m.ckpt"))
    back = Sequential.from_checkpoint(str(tmp_path / "m.ckpt"))
    assert [layer[0] for layer in back.layers] == [layer[0] for layer in m.layers]
    assert np.asarray(back.predict(x, lengths=il)).tobytes() == want.tobytes()
    # resuming the loaded model continues with the saved loss, blank and zero_infinity: the same words
    a = m.fit(x, y, epochs=1, resume=True, lengths=il, target_lengths=tl)
    b = back.fit(x, y, epochs=1, resume=True, lengths=il, target_lengths=tl)
    np.testing.assert_array_equal(a.params_, b.params_)
    np.testing.assert_array_equal(np.asarray(a.loss_curve_), np.asarray(b.loss_curve_))
    a.export(str(tmp_path / "m.mlx"))
    loaded = Sequential.load(str(tmp_path / "m.mlx"))
    assert np.asarray(loaded.predict(x, lengths=il)).tobytes() == np.asarray(a.predict(x, lengths=il)).tobytes()
