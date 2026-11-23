# SPDX-License-Identifier: Apache-2.0
"""Public boundary of classic attention in Sequential (lane nn-attention, 2026-11-23):
multihead_attention, positional_encoding, transformer_encoder_layer / transformer_decoder_layer and the
transformer_encoder / transformer_decoder stacks: the flat description each call writes, the refusals in
words, state_dict names and shapes, save and load round trips, and the predictions against a NumPy float64
reference (an independent oracle) written from the model's own state_dict. The numerics (every gradient,
host versus device words, the copy task) are the native check checks/native_tests/nn_attention.mojo."""
import numpy as np
import pytest

import mojolearn

S = mojolearn.Sequential


def _kinds(model):
    return [layer[0] for layer in model.layers]


def _x(rows, shape, seed=0):
    return np.random.default_rng(seed).standard_normal((rows,) + tuple(shape)).astype(np.float32)


def _fitted(model, shape, rows=16, seed=0):
    x = _x(rows, shape, seed)
    y = _x(rows, (1,), seed + 1).reshape(rows, 1)
    return model.compile(loss="mse", optimizer="adam", lr=0.01).fit(
        x.reshape(rows, -1), y, epochs=1, batch_size=8, shuffle=False, seed=0), x


# ---------------------------------------------------------------- NumPy float64 reference
def _tlin(x, w, b):
    y = x @ w.astype(np.float64).T
    return y if b is None else y + b.astype(np.float64)


def _mha_ref(xq, xkv, sd, name, heads, causal=False, lengths=None):
    e = xq.shape[-1]
    w = sd[name + ".in_proj_weight"].astype(np.float64).reshape(-1)
    kd = xkv.shape[-1]
    wq = w[:e * e].reshape(e, e)
    wk = w[e * e:e * e + e * kd].reshape(e, kd)
    wv = w[e * e + e * kd:].reshape(e, kd)
    bq, bk, bv = np.split(sd[name + ".in_proj_bias"].astype(np.float64), 3)
    q, k, v = _tlin(xq, wq, bq), _tlin(xkv, wk, bk), _tlin(xkv, wv, bv)
    n, t, _ = q.shape
    s = xkv.shape[1]
    hd = e // heads
    q = q.reshape(n, t, heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(n, s, heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(n, s, heads, hd).transpose(0, 2, 1, 3)
    sc = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    mask = np.zeros((n, 1, t, s), dtype=bool)
    if causal:
        mask |= np.triu(np.ones((t, s), dtype=bool), 1)[None, None]
    if lengths is not None:
        mask |= (np.arange(s)[None, :] >= np.asarray(lengths)[:, None])[:, None, None, :]
    sc = np.where(mask, -np.inf, sc)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ctx = (p @ v).transpose(0, 2, 1, 3).reshape(n, t, e)
    return _tlin(ctx, sd[name + ".out_proj.weight"], sd[name + ".out_proj.bias"])


def _ln_ref(x, sd, name, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * sd[name + ".weight"] + sd[name + ".bias"]


def _lin_ref(x, sd, name):
    return _tlin(x, sd[name + ".weight"], sd[name + ".bias"])


def _pe_ref(t, d):
    pos = np.arange(t, dtype=np.float64)[:, None]
    div = 10000.0 ** (np.arange(0, d, 2, dtype=np.float64) / d)
    pe = np.zeros((t, d))
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe


# ---------------------------------------------------------------- descriptions
def test_multihead_attention_describes_its_arguments():
    m = (S(input_shape=(5, 4)).linear(8).multihead_attention(8, 2, dropout=0.25, is_causal=True)
         .multihead_attention(8, 4, bias=False, kdim=4, vdim=4, source=-1, key_padding_mask=True)
         .flatten().linear(1))
    assert _kinds(m) == ["linear", "multihead_attention", "multihead_attention", "flatten", "linear"]
    # (num_heads, flags (1 bias, 2 causal, 4 key padding), source (-2 self), embed_dim, kdim (0 = embed_dim), 0)
    assert m.layers[1][1] == (2, 3, -2, 8, 0, 0) and m.layers[1][2] == 0.25
    assert m.layers[2][1] == (4, 4, -1, 8, 4, 0)


def test_attn_mask_causal_is_is_causal():
    a = S(input_shape=(5, 8)).multihead_attention(8, 2, attn_mask="causal")
    b = S(input_shape=(5, 8)).multihead_attention(8, 2, is_causal=True)
    assert a.layers == b.layers


def test_positional_encoding_describes_its_arguments():
    m = S(input_shape=(6,)).embedding(10, 8).positional_encoding(8, dropout=0.0, max_len=64)
    assert m.layers[1] == ("positional_encoding", (64, 8, 0, 0, 0, 0), 0.0)


def test_encoder_layer_post_norm_is_pytorch_module_order():
    m = S(input_shape=(6, 8)).transformer_encoder_layer(8, 2, 16, dropout=0.1)
    assert _kinds(m) == ["multihead_attention", "dropout", "residual", "layer_norm", "linear", "relu", "dropout",
                         "linear", "dropout", "residual", "layer_norm"]
    assert m.layers[2][1] == -1 and m.layers[9][1] == 3
    assert m.layers[0][1] == (2, 1, -2, 8, 0, 0) and m.layers[0][2] == pytest.approx(0.1)


def test_encoder_layer_pre_norm_and_gelu():
    m = S(input_shape=(6, 8)).linear(8).transformer_encoder_layer(8, 2, 16, activation="gelu", norm_first=True)
    assert _kinds(m) == ["linear", "layer_norm", "multihead_attention", "dropout", "residual", "layer_norm",
                         "linear", "gelu", "dropout", "linear", "dropout", "residual"]
    assert m.layers[4][1] == 0 and m.layers[11][1] == 4


def test_decoder_layer_reads_its_memory():
    m = S(input_shape=(6, 8)).linear(8).transformer_decoder_layer(8, 2, 16, memory=0, tgt_is_causal=True)
    kinds = _kinds(m)
    assert kinds.count("multihead_attention") == 2 and kinds.count("layer_norm") == 3
    attn = [layer for layer in m.layers if layer[0] == "multihead_attention"]
    assert attn[0][1][1:3] == (3, -2)      # causal self-attention
    assert attn[1][1][1:3] == (1, 0)       # attention to layer 0


def test_stacks_copy_the_layer_and_set_memory():
    enc = S().transformer_encoder_layer(8, 2, 16, dropout=0.0)
    m = S(input_shape=(6, 8)).linear(8).transformer_encoder(enc, 3, norm=S().layer_norm())
    assert len(m.layers) == 1 + 3 * 11 + 1 and len(enc.layers) == 11
    assert m.layers[3][1] == 0 and m.layers[14][1] == 11      # each copy's first residual reads its input
    memory = len(m.layers) - 1
    dec = S().transformer_decoder_layer(8, 2, 16, dropout=0.0, tgt_is_causal=True)
    m = m.transformer_decoder(dec, 2, memory)
    cross = [layer for layer in m.layers[memory + 1:] if layer[0] == "multihead_attention" and layer[1][2] != -2]
    assert len(cross) == 2 and all(layer[1][2] == memory for layer in cross)
    assert [layer for layer in dec.layers if layer[0] == "multihead_attention"][1][1][2] == -3   # template unchanged


@pytest.mark.parametrize("call, message", [
    (lambda m: m.multihead_attention(8, 3), "divisible"),
    (lambda m: m.multihead_attention(8, 2, batch_first=False), "batch_first must be True"),
    (lambda m: m.multihead_attention(8, 2, add_bias_kv=True), "add_bias_kv"),
    (lambda m: m.multihead_attention(8, 2, kdim=4, vdim=5, source=-1), "kdim and vdim must be equal"),
    (lambda m: m.multihead_attention(8, 2, kdim=4), "self-attention takes kdim"),
    (lambda m: m.multihead_attention(8, 2, attn_mask=np.zeros((5, 5))), "attn_mask takes None or 'causal'"),
    (lambda m: m.multihead_attention(8, 2, dropout=1.0), "dropout must be in"),
    (lambda m: m.positional_encoding(7), "even"),
    (lambda m: m.positional_encoding(8, max_len=10000), "max_len"),
    (lambda m: m.transformer_encoder_layer(8, 2, activation="tanh"), "activation"),
    (lambda m: m.transformer_encoder_layer(8, 2, bias=False), "bias=False"),
    (lambda m: m.transformer_encoder(S(), 2), "at least one layer"),
    (lambda m: m.transformer_decoder(S().transformer_encoder_layer(8, 2), 2, -1), "transformer_decoder_layer"),
    (lambda m: m.transformer_decoder(S().transformer_decoder_layer(8, 2, memory=0), 2, -1), "without memory"),
])
def test_refusals_at_add(call, message):
    with pytest.raises((ValueError, TypeError), match=message):
        call(S(input_shape=(5, 8)))


@pytest.mark.parametrize("build, message", [
    (lambda m: m.multihead_attention(6, 2), "embed_dim is 6"),
    (lambda m: m.multihead_attention(8, 2, source=5), "key/value source"),
    (lambda m: m.linear(4).multihead_attention(8, 2, source=-1, kdim=4, vdim=4), "kdim is 4"),
    (lambda m: m.narrow(0, 0, 3).multihead_attention(8, 2, source=-1, is_causal=True), "causal mask"),
    (lambda m: m.transformer_decoder_layer(8, 2, 16), "memory is not set"),
    (lambda m: m.positional_encoding(8, max_len=4), "longer than max_len"),
])
def test_refusals_at_fit(build, message):
    model = build(S(input_shape=(5, 8))).flatten().linear(1)
    with pytest.raises((ValueError, RuntimeError), match=message):
        _fitted(model, (5, 8))


# ---------------------------------------------------------------- numerics against NumPy
def test_multihead_attention_matches_numpy():
    model, _ = _fitted(S(input_shape=(5, 4)).linear(8).multihead_attention(8, 2, is_causal=True)
                       .flatten().linear(1), (5, 4))
    sd = model.state_dict()
    assert sd["1.in_proj_weight"].shape == (24, 8) and sd["1.out_proj.weight"].shape == (8, 8)
    assert sd["1.in_proj_bias"].shape == (24,) and sd["1.out_proj.bias"].shape == (8,)
    x = _x(6, (5, 4), 3).astype(np.float64)
    h = _lin_ref(x, sd, "0")
    h = _mha_ref(h, h, sd, "1", 2, causal=True)
    want = _lin_ref(h.reshape(6, -1), sd, "3")
    got = model.predict(x.astype(np.float32).reshape(6, -1))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_cross_attention_and_positional_encoding_match_numpy():
    model, _ = _fitted(S(input_shape=(5, 4)).linear(8).positional_encoding(8, dropout=0.0)
                       .multihead_attention(8, 2, kdim=4, vdim=4, source=-1).flatten().linear(1), (5, 4))
    sd = model.state_dict()
    assert sd["2.in_proj_weight"].shape == (8 * (8 + 2 * 4),)
    x = _x(4, (5, 4), 5).astype(np.float64)
    h = _lin_ref(x, sd, "0") + _pe_ref(5, 8)
    h = _mha_ref(h, x, sd, "2", 2)
    want = _lin_ref(h.reshape(4, -1), sd, "4")
    np.testing.assert_allclose(model.predict(x.astype(np.float32).reshape(4, -1)), want, rtol=1e-4, atol=1e-4)


def test_encoder_layer_matches_numpy():
    model, _ = _fitted(S(input_shape=(5, 8)).transformer_encoder_layer(8, 2, 16, dropout=0.1)
                       .flatten().linear(1), (5, 8))
    sd = model.state_dict()
    x = _x(4, (5, 8), 7).astype(np.float64)
    h = _ln_ref(x + _mha_ref(x, x, sd, "0", 2), sd, "3")
    f = _lin_ref(np.maximum(_lin_ref(h, sd, "4"), 0.0), sd, "7")
    h = _ln_ref(h + f, sd, "10")
    want = _lin_ref(h.reshape(4, -1), sd, "12")
    np.testing.assert_allclose(model.predict(x.astype(np.float32).reshape(4, -1)), want, rtol=1e-4, atol=1e-4)


def test_key_padding_mask_ignores_padded_positions():
    model = (S(input_shape=(6, 4)).linear(8).multihead_attention(8, 2, key_padding_mask=True)
             .narrow(0, 0, 1).flatten().linear(1))
    x = _x(16, (6, 4), 9).reshape(16, -1)
    lengths = np.array([6, 3, 1, 4] * 4, dtype=np.float32)
    model = model.compile(loss="mse", optimizer="adam", lr=0.01).fit(
        x, _x(16, (1,), 10).reshape(16, 1), epochs=1, batch_size=8, shuffle=False, seed=0, lengths=lengths)
    noisy = x.reshape(16, 6, 4).copy()
    for r, n in enumerate(lengths.astype(int)):
        noisy[r, n:] = 1e3
    a = model.predict(x, lengths=lengths)
    b = model.predict(noisy.reshape(16, -1), lengths=lengths)
    np.testing.assert_array_equal(a, b)


def test_state_dict_and_checkpoint_round_trips(tmp_path):
    enc = S().transformer_encoder_layer(8, 2, 16, dropout=0.0)
    model, x = _fitted(S(input_shape=(5, 8)).positional_encoding(8).transformer_encoder(enc, 2)
                       .multihead_attention(8, 2, bias=False).flatten().linear(1), (5, 8))
    flat = x.reshape(len(x), -1)
    want = model.predict(flat)
    sd = model.state_dict()
    attn = [name for name in sd if name.endswith("in_proj_bias")]
    assert len(attn) == 3
    np.testing.assert_array_equal(sd[attn[-1]], 0.0)    # bias=False: the biases stay zero
    other = S(input_shape=(5, 8)).positional_encoding(8).transformer_encoder(enc, 2)
    other = other.multihead_attention(8, 2, bias=False).flatten().linear(1).load_state_dict(sd)
    np.testing.assert_array_equal(other.predict(flat), want)
    path = tmp_path / "attention.ckpt"
    model.save_checkpoint(str(path))
    loaded = S.from_checkpoint(str(path))
    np.testing.assert_array_equal(loaded.predict(flat), want)
    assert loaded.layers == model.layers and loaded.digest_ == model.digest_
