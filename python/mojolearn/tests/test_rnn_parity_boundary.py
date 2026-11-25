# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of the recurrent estimators' options (lane rnn-parity, Andrew 2026-11-25; tests by lane
rnn-parity-D): their refusals in words, the parameter names get_params reports, the state_dict names
(PyTorch's `_reverse` suffix, no recurrent biases with bias=False), and PyTorch itself as an external oracle of
the bidirectional two-layer GRU. The numbers are checked natively (checks/native_tests/rnn_parity.mojo) and by
the verify lanes sequence-rnn-weighted-steps and sequence-rnn-bidi-options (tools/identity_lanes/rnn_parity.py)."""
import numpy as np
import pytest

import mojolearn

N, T, D, H = 12, 5, 2, 4


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, T, D)).astype(np.float32)
    return x, x[:, -1, 0].copy(), (x[:, :, 0] > 0).astype(np.int64)


def _small(cls, **kw):
    return cls(hidden_size=H, max_epochs=1, batch_size=6, **kw)


# ---------------------------------------------------------------- refusals
def test_refuses_an_unknown_output_word():
    x, y, _ = _data()
    with pytest.raises(ValueError, match="output must be 'final' or 'sequence'"):
        _small(mojolearn.GRURegressor, output="every").fit(x, y)


@pytest.mark.parametrize("dropout", [1.0, 1.5, -0.1])
def test_refuses_dropout_outside_zero_one(dropout):
    x, y, _ = _data()
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        _small(mojolearn.GRURegressor, num_layers=2, dropout=dropout).fit(x, y)


def test_refuses_class_weight_on_a_regressor():
    x, y, _ = _data()
    with pytest.raises(ValueError, match="class_weight is for classifiers"):
        _small(mojolearn.LSTMRegressor, class_weight={0: 2.0}).fit(x, y)


@pytest.mark.parametrize("output, shape", [("final", (N + 1,)), ("final", (N, T)), ("sequence", (N, T + 1))])
def test_refuses_a_sample_weight_of_another_shape(output, shape):
    x, y, ys = _data()
    target = ys if output == "sequence" else (y > 0).astype(np.int64)
    with pytest.raises(ValueError, match="sample_weight must have shape"):
        _small(mojolearn.GRUClassifier, output=output).fit(x, target, sample_weight=np.ones(shape, np.float32))


@pytest.mark.parametrize("kw, slots", [({}, 1), ({"num_layers": 2}, 2), ({"bidirectional": True}, 2)])
def test_refuses_an_initial_state_of_another_shape(kw, slots):
    x, y, _ = _data()
    wrong = np.zeros((slots + 1, N, H), np.float32)
    with pytest.raises(ValueError, match=r"h_0 must have shape \(num_layers \* num_directions, n_samples, "
                                         r"hidden_size\) = \(" + str(slots) + ", " + str(N) + ", " + str(H) + r"\)"):
        _small(mojolearn.GRURegressor, **kw).fit(x, y, h_0=wrong)
    m = _small(mojolearn.GRURegressor, **kw).fit(x, y)
    with pytest.raises(ValueError, match="h_0 must have shape"):
        m.predict(x, h_0=np.zeros((slots, N + 1, H), np.float32))


@pytest.mark.parametrize("output", ["final", "sequence"])
def test_refuses_c_0_without_h_0(output):
    # PyTorch takes (h_0, c_0) together; a cell state alone is refused, never silently dropped
    x, y, _ = _data()
    target = np.repeat(y[:, None], T, axis=1) if output == "sequence" else y
    c0 = np.zeros((1, N, H), np.float32)
    with pytest.raises(ValueError, match="c_0 needs h_0"):
        _small(mojolearn.LSTMRegressor, output=output).fit(x, target, c_0=c0)


def test_refuses_validation_lengths_on_one_side_only():
    x, y, _ = _data()
    xv, yv, _ = _data(1)
    lengths = np.full(N, T, np.int64)
    with pytest.raises(ValueError, match="give lengths for both the training and the validation sequences"):
        _small(mojolearn.GRURegressor).fit(x, y, lengths=lengths, validation_data=(xv, yv))


def test_refuses_validation_data_of_another_arity():
    x, y, _ = _data()
    with pytest.raises(ValueError, match=r"validation_data must be \(X_val, y_val\) or \(X_val, y_val, lengths_val\)"):
        _small(mojolearn.GRURegressor).fit(x, y, validation_data=(x,))


@pytest.mark.parametrize("state", [np.zeros((1, N, H), np.float32), [np.zeros((1, N, H), np.float32)],
                                   (np.zeros((1, N, H), np.float32),)])
def test_step_refuses_a_state_that_is_not_the_pair_step_returned(state):
    x, y, _ = _data()
    m = _small(mojolearn.GRURegressor).fit(x, y)
    with pytest.raises(TypeError, match=r"state must be the \(h, c\) pair step returned, or None"):
        m.step(x[:, :1], state)


@pytest.mark.parametrize("state", [False, True])
def test_step_refuses_a_bidirectional_model(state):
    # a reverse direction reads the steps that come later: no piece of a sequence can be scored before it ends
    x, y, _ = _data()
    m = _small(mojolearn.GRURegressor, bidirectional=True).fit(x, y)
    carried = (np.zeros((2, N, H), np.float32), None) if state else None
    with pytest.raises(ValueError, match="step: a bidirectional model cannot stream"):
        m.step(x[:, :1], carried)


# ---------------------------------------------------------------- names
_OPTIONS = ("output", "bidirectional", "dropout", "bias", "clip_grad_norm", "class_weight", "early_stopping",
            "validation_fraction", "n_iter_no_change", "tol", "warm_start")


@pytest.mark.parametrize("cls", [mojolearn.GRURegressor, mojolearn.GRUClassifier, mojolearn.LSTMRegressor,
                                 mojolearn.LSTMClassifier, mojolearn.RNNRegressor, mojolearn.RNNClassifier])
def test_get_params_reports_every_option_with_its_default(cls):
    params = cls().get_params()
    defaults = dict(output="final", bidirectional=False, dropout=0.0, bias=True, clip_grad_norm=None,
                    class_weight=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
                    warm_start=False)
    for name in _OPTIONS:  # glue: test fixture, the option names
        assert name in params, name
        assert params[name] == defaults[name], name


def test_state_dict_names_reverse_directions_and_no_recurrent_biases():
    x, y, _ = _data()
    m = _small(mojolearn.LSTMRegressor, num_layers=2, bidirectional=True, bias=False).fit(x, y)
    state = m.state_dict()
    expected = {f"weight_{kind}_l{layer}{suffix}" for kind in ("ih", "hh") for layer in (0, 1)
                for suffix in ("", "_reverse")} | {"head.weight", "head.bias"}
    assert set(state) == expected
    assert np.asarray(state["weight_ih_l1"]).shape == (4 * H, 2 * H)
    assert np.asarray(state["head.weight"]).shape == (1, 2 * H)


def test_state_dict_with_biases_names_them_per_direction():
    x, y, _ = _data()
    state = _small(mojolearn.GRURegressor, bidirectional=True).fit(x, y).state_dict()
    for name in ("bias_ih_l0", "bias_hh_l0", "bias_ih_l0_reverse", "bias_hh_l0_reverse"):  # glue: names
        assert np.asarray(state[name]).shape == (3 * H,)


# ---------------------------------------------------------------- PyTorch as the external oracle
def test_bidirectional_gru_state_dict_loads_into_torch_and_matches():
    torch = pytest.importorskip("torch")
    x, y, _ = _data()
    m = mojolearn.GRURegressor(hidden_size=H, num_layers=2, bidirectional=True, max_epochs=2, batch_size=6,
                               random_state=3).fit(x, y)
    state = m.state_dict()
    gru = torch.nn.GRU(D, H, num_layers=2, bidirectional=True, batch_first=True)
    gru.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in state.items() if not k.startswith("head.")})
    head = torch.nn.Linear(2 * H, 1)
    head.load_state_dict({"weight": torch.from_numpy(np.array(state["head.weight"])),
                          "bias": torch.from_numpy(np.array(state["head.bias"]))})
    with torch.no_grad():
        out, h_n = gru(torch.from_numpy(x))
        pred = head(torch.cat([h_n[-2], h_n[-1]], dim=1))[:, 0]
    np.testing.assert_allclose(np.asarray(m.hidden_sequence(x)), out.numpy(), rtol=1e-5, atol=1e-5)
    # the top layer's h_n from the output: forward at the last step, reverse at the first (step refuses a
    # bidirectional model: it cannot stream)
    seq = np.asarray(m.hidden_sequence(x))
    np.testing.assert_allclose(seq[:, -1, :H], h_n.numpy()[-2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(seq[:, 0, H:], h_n.numpy()[-1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(m.predict(x)), pred.numpy(), rtol=1e-5, atol=1e-5)
    # and back: torch's tensors load into another fitted estimator, which then predicts the same words
    other = mojolearn.GRURegressor(hidden_size=H, num_layers=2, bidirectional=True, max_epochs=1, batch_size=6,
                                   random_state=4).fit(x, y)
    back = {k: v.numpy() for k, v in gru.state_dict().items()}
    back.update({"head.weight": head.weight.detach().numpy(), "head.bias": head.bias.detach().numpy()})
    other.load_state_dict(back)
    np.testing.assert_array_equal(np.asarray(other.predict(x)), np.asarray(m.predict(x)))


# ---------------------------------------------------------------- lane rnn-review-fixes (two review findings)
# On a GPU installation each case runs on both sides, device="cpu" and device="gpu"; a CPU-only installation has
# one side (the device default).
SIDES = [None] if mojolearn.vendor() == "cpu" else ["cpu", "gpu"]


def _side(cls, device, **kw):
    return _small(cls, **kw) if device is None else _small(cls, device=device, **kw)


@pytest.mark.parametrize("device", SIDES)
@pytest.mark.parametrize("bad", [np.r_[np.full(N - 1, T), T + 1], np.r_[np.full(N - 1, T), 0],
                                 np.r_[2.5, np.full(N - 1, T)], np.r_[-1, np.full(N - 1, T)]])
def test_balanced_class_weight_refuses_a_bad_length_before_reading_it(device, bad):
    """P1-a: the lengths are checked once, in admission, before the balanced class histogram uses each as a loop
    bound over its sequence's labels (an oversized last length read past the target buffer)."""
    x, _, ys = _data()
    with pytest.raises(ValueError, match="fit: every length must be a whole number from 1 to the sequence "
                                         "length " + str(T)):
        _side(mojolearn.GRUClassifier, device, output="sequence", class_weight="balanced").fit(x, ys, lengths=bad)


def test_a_bad_length_is_refused_with_its_own_words_where_the_histogram_would_miss_a_class():
    """P1-a: a length of 0 on the only sequence with class 1 is a bad length, not a missing class."""
    x, _, _ = _data()
    y = np.zeros((N, T), np.int64)
    y[3] = 1
    bad = np.full(N, T)
    bad[3] = 0
    with pytest.raises(ValueError, match="every length must be a whole number"):
        _small(mojolearn.GRUClassifier, output="sequence", class_weight="balanced").fit(x, y, lengths=bad)


def _padding_fixture():
    x, _, ys = _data()
    lengths = np.r_[np.full(N // 2, T), np.arange(N - N // 2) % T + 1]
    padded = np.arange(T)[None, :] >= lengths[:, None]
    return x, ys, lengths, padded


@pytest.mark.parametrize("device", SIDES)
@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_the_padding_label_changes_nothing(device, class_weight):
    """P1-b: classes_ and the head's width come from the labels of valid steps; a label used only as padding is
    no class (with 'balanced' it used to be a class with no example, and fit raised)."""
    x, ys, lengths, padded = _padding_fixture()
    fits = []
    for pad in (7, 0, -3):
        m = _side(mojolearn.GRUClassifier, device, output="sequence", class_weight=class_weight, random_state=3)
        fits.append(m.fit(x, np.where(padded, pad, ys), lengths=lengths))
    first = fits[0]
    np.testing.assert_array_equal(first.classes_, [0, 1])
    assert first.n_outputs_ == 2
    proba = np.asarray(first.predict_proba(x, lengths=lengths))
    assert proba.shape == (N, T, 2)
    labels = np.asarray(first.predict(x, lengths=lengths))
    valid = ~padded
    for other in fits[1:]:
        np.testing.assert_array_equal(other.classes_, first.classes_)
        assert other.n_outputs_ == first.n_outputs_
        assert other.digest_ == first.digest_
        np.testing.assert_array_equal(np.asarray(other.params_), np.asarray(first.params_))
        np.testing.assert_array_equal(np.asarray(other.predict(x, lengths=lengths))[valid], labels[valid])
        np.testing.assert_array_equal(np.asarray(other.predict_proba(x, lengths=lengths)), proba)


@pytest.mark.parametrize("device", SIDES)
def test_a_padding_label_first_seen_in_validation_or_partial_fit_is_not_refused(device):
    """P1-b: validation_data and partial_fit map labels against the fitted classes at valid steps only."""
    x, ys, lengths, padded = _padding_fixture()
    m = _side(mojolearn.GRUClassifier, device, output="sequence", random_state=3)
    m.fit(x, np.where(padded, 0, ys), lengths=lengths, validation_data=(x, np.where(padded, 9, ys), lengths))
    assert len(m.validation_loss_) == 1
    m.partial_fit(x, np.where(padded, 11, ys), lengths=lengths)
    np.testing.assert_array_equal(m.classes_, [0, 1])
    # a label at a VALID step that is no fitted class is still refused
    unknown = np.where(padded, 0, ys)
    unknown[0, 0] = 5
    with pytest.raises(ValueError, match="which is not one of the fitted classes"):
        m.partial_fit(x, unknown, lengths=lengths)
