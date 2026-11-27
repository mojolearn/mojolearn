# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the recurrent options of Sequential.lstm / gru / rnn (lane nn-recurrent, 2026-11-23):
num_layers, dropout, bidirectional and output. The layers each call writes, the refusals at add, and a save /
load round trip. The numerics (the reverse direction and the final state against a float64 oracle, every
gradient against central differences, padded versus alone, the bidirectional GRU baseline, host versus device
words) are the native check checks/native_tests/nn_recurrent.mojo."""
import numpy as np
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def _tracks(n=24, steps=6):
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, steps + 1, n).astype(np.int64)
    x = rng.standard_normal((n, steps, 2)).astype(np.float32)
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, y, lengths


def test_defaults_write_the_one_layer_they_always_wrote():
    for kind in ("lstm", "gru", "rnn"):  # glue: test fixture, the three public names
        a = getattr(mojolearn.Sequential(input_shape=(6, 2)), kind)(4)
        b = getattr(mojolearn.Sequential(input_shape=(6, 2)), kind)(4, num_layers=1, bidirectional=False,
                                                                     dropout=0.0, output="sequence")
        assert a.layers == b.layers
        assert len(a.layers) == 1


def test_bidirectional_two_layers_are_flattened_in_pytorch_order():
    m = mojolearn.Sequential(input_shape=(6, 2)).gru(4, num_layers=2, bidirectional=True, dropout=0.25)
    kinds = _kinds(m)
    assert kinds[1:4] == ["branch", kinds[0], "concat"]
    assert kinds[4] == "dropout" and kinds[5:9] == [kinds[0], "branch", kinds[0], "concat"]
    assert len(kinds) == 9
    # the reverse directions read the layer input (the model input, then the dropout's output)
    assert m.layers[1][1][0] == -1 and m.layers[6][1][0] == 4
    # the joins: the two directions on the features (axis 1 of the (length, hidden) sample)
    assert m.layers[3][1][:2] == (0, 2) and m.layers[8][1][:2] == (5, 7)


def test_stacked_unidirectional_equals_layers_added_one_by_one():
    a = mojolearn.Sequential(input_shape=(6, 2)).lstm(4, num_layers=3)
    b = mojolearn.Sequential(input_shape=(6, 2)).lstm(4).lstm(4).lstm(4)
    assert a.layers == b.layers


def test_final_output_shapes():
    x, y, lengths = _tracks()
    for both in (False, True):  # glue: test fixture, the two directions
        m = (mojolearn.Sequential(input_shape=(6, 2))
             .lstm(5, num_layers=2, bidirectional=both, output="final").flatten().linear(2)
             .compile(loss="cross_entropy"))
        m.fit(x, y, epochs=1, batch_size=8, lengths=lengths)
        proba = m.predict_proba(x, lengths=lengths)
        assert proba.shape == (24, 2) and np.isfinite(proba).all()
    seq = (mojolearn.Sequential(input_shape=(6, 2)).gru(5, bidirectional=True).flatten().linear(1)
           .compile(loss="mse"))
    seq.fit(x, y.astype(np.float32), epochs=1, batch_size=8, lengths=lengths)
    # (length 6, 2 directions * hidden 5) into the linear layer: 60 weights per output, plus the bias
    weights = sum(np.asarray(v).size for k, v in seq.state_dict().items() if k.startswith("5."))
    assert weights == 61


@pytest.mark.parametrize("kwargs,message", [
    (dict(num_layers=0), "num_layers"),
    (dict(num_layers=1.5), "num_layers"),
    (dict(bidirectional=1), "bidirectional must be True or False"),
    (dict(dropout=1.0), r"dropout must be in \[0, 1\)"),
    (dict(dropout=-0.1), r"dropout must be in \[0, 1\)"),
    (dict(output="last"), "output must be 'sequence' or 'final'"),
    (dict(output=None), "output must be 'sequence' or 'final'"),
])
def test_refusals(kwargs, message):
    with pytest.raises(Exception, match=message):
        mojolearn.Sequential(input_shape=(6, 2)).lstm(4, **kwargs)


def test_unknown_option_is_refused():
    with pytest.raises(TypeError):
        mojolearn.Sequential(input_shape=(6, 2)).gru(4, proj_size=2)


def test_save_load_round_trip(tmp_path):
    x, y, lengths = _tracks()
    m = (mojolearn.Sequential(input_shape=(6, 2))
         .gru(4, num_layers=2, bidirectional=True, dropout=0.1, output="final").flatten().linear(2)
         .compile(loss="cross_entropy"))
    m.fit(x, y, epochs=2, batch_size=8, seed=3, lengths=lengths)
    want = m.predict_proba(x, lengths=lengths)
    for save in ("save_checkpoint", "export"):  # glue: test fixture, the two file forms
        path = str(tmp_path / (save + ".ckpt"))
        getattr(m, save)(path)
        again = mojolearn.Sequential.load(path)
        assert again.layers == m.layers
        np.testing.assert_array_equal(np.asarray(again.params_), np.asarray(m.params_))
        np.testing.assert_array_equal(again.predict_proba(x, lengths=lengths), want)
    state = m.state_dict()
    fresh = (mojolearn.Sequential(input_shape=(6, 2))
             .gru(4, num_layers=2, bidirectional=True, dropout=0.1, output="final").flatten().linear(2)
             .compile(loss="cross_entropy"))
    fresh.load_state_dict(state)
    np.testing.assert_array_equal(np.asarray(fresh.params_), np.asarray(m.params_))


def test_projected_stack_widths_and_checkpoint(tmp_path):
    x, y, lengths = _tracks(n=6, steps=3)
    m = (mojolearn.Sequential(input_shape=(3, 2))
         .lstm(4, num_layers=2, bidirectional=True, proj_size=2, output="final")
         .flatten().linear(2).compile(loss="cross_entropy"))
    m.fit(x, y, epochs=1, batch_size=6, seed=5, lengths=lengths)
    state = m.state_dict()

    def shapes(suffix):
        return sorted(np.shape(v) for k, v in state.items() if k.endswith(suffix))

    assert shapes(".weight_ih") == [(16, 2), (16, 2), (16, 4), (16, 4)]
    assert shapes(".weight_hh") == [(16, 2)] * 4
    assert shapes(".weight_hr") == [(2, 4)] * 4
    path = str(tmp_path / "projected.ckpt")
    m.save_checkpoint(path)
    restored = mojolearn.Sequential.load(path)
    assert restored.layers == m.layers
    np.testing.assert_array_equal(restored.params_, m.params_)
    np.testing.assert_array_equal(restored.predict_proba(x, lengths=lengths), m.predict_proba(x, lengths=lengths))


@pytest.mark.parametrize("width", [-1, 4, 5, 1.25])
def test_projected_lstm_rejects_invalid_width(width):
    with pytest.raises(Exception, match="proj_size"):
        mojolearn.Sequential(input_shape=(3, 2)).lstm(4, proj_size=width)


def _projection_weights(s):
    return [s.tensor(np.full(shape, (i + 1) / 16, np.float32))
            for i, shape in enumerate(((12, 2), (12, 2), (1, 12), (1, 12), (2, 3)))]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1, 0, 1.5, 4])
def test_projected_lstm_rejects_invalid_length(bad):
    with mojolearn.ops.Session() as s:
        x = s.tensor(np.zeros((3, 2, 2), np.float32))
        wi, wh, bi, bh, hr = _projection_weights(s)
        lens = s.tensor(np.array([3, bad], np.float32))
        with pytest.raises(Exception, match="lengths"):
            s.lstm(x, wi, wh, bi, bh, lengths=lens, weight_hr=hr)


@pytest.mark.parametrize("reverse", [False, True])
def test_projected_lstm_masks_padding_and_carries_state(reverse):
    xs = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 16
    h0 = np.array([[0.125, -0.25], [0.5, 0.25]], np.float32)
    c0 = np.full((2, 3), 0.125, np.float32)

    def evaluate(padding):
        data = xs.copy()
        data[1:, 1] = padding
        with mojolearn.ops.Session(record=True) as s:
            x = s.tensor(data, requires_grad=True)
            wi, wh, bi, bh, hr = _projection_weights(s)
            y, h, c = s.lstm(x, wi, wh, bi, bh, h_0=s.tensor(h0), c_0=s.tensor(c0),
                              lengths=s.tensor(np.array([3, 1], np.float32)), weight_hr=hr,
                              return_state=True, reverse=reverse)
            s.backward(s.add(s.sum(y), s.sum(c)))
            return tuple(np.asarray(s.numpy(t)) for t in (y, h, c)) + (np.asarray(s.grad(x)),)

    clean = evaluate(0.0)
    polluted = evaluate(np.nan)
    for a, b in zip(clean, polluted):
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    y, h, c, grad = clean
    assert y.shape == (3, 2, 2) and c.shape == (2, 3)
    np.testing.assert_array_equal(h, y[0 if reverse else -1])
    carried = h0[1] if reverse else y[0, 1]
    np.testing.assert_array_equal(y[1:, 1], np.broadcast_to(carried, (2, 2)))
    np.testing.assert_array_equal(grad[1:, 1], np.zeros((2, 2), np.float32))


def test_saved_projected_program_resizes_all_three_outputs(tmp_path):
    def inputs(steps, batch):
        return (np.arange(steps * batch * 2, dtype=np.float32).reshape(steps, batch, 2) / 32,
                np.full((batch, 2), 0.25, np.float32), np.full((batch, 3), -0.125, np.float32))

    path = str(tmp_path / "dynamic.mlp")
    with mojolearn.ops.Session() as s:
        wi, wh, bi, bh, hr = _projection_weights(s)
        x, h, c = [s.tensor(a) for a in inputs(3, 2)]
        outputs = s.lstm(x, wi, wh, bi, bh, h_0=h, c_0=c, weight_hr=hr, return_state=True)
        s.capture(inputs=[x, h, c], outputs=list(outputs)).save(path)
    with mojolearn.ops.Session() as s:
        program = s.load_program(path)
        names = [node["op"] for node in program.inspect()["nodes"]]
        assert names.count("projected_lstm") == 1
        assert names.count("projected_lstm_output") == 3
        for dims in ((2, 3), (4, 2)):
            arrays = inputs(*dims)
            actual = program.run(*arrays, dynamic_shapes=True)
            with mojolearn.ops.Session() as eager:
                x, h, c = [eager.tensor(a) for a in arrays]
                wi, wh, bi, bh, hr = _projection_weights(eager)
                outputs = eager.lstm(x, wi, wh, bi, bh, h_0=h, c_0=c, weight_hr=hr, return_state=True)
                for result, handle in zip(actual, outputs):
                    np.testing.assert_array_equal(result, eager.numpy(handle))
