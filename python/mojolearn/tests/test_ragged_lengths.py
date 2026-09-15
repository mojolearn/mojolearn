# SPDX-License-Identifier: Apache-2.0
"""`lengths=`, the ragged right-padded batch (python/mojolearn/_ragged.py, 2026-09-15).

Two groups. The helper tests need nothing compiled: they hold the byte
copies, the refusals and the last-real-position gather to their contract
with a stand-in call. The surface tests run each model that takes
`lengths` against the same rows run alone, on whatever bindings this
install has, and skip BY NAME when the binding is absent (the CPU-only
install has the host byte LM and nothing on a GPU): every real position must
equal the row alone byte for byte, and every padding output must be +0.0,
with NaN or out-of-vocabulary padding in the input. Cross-vendor standing is
not this file's claim; tools/identity_break.py --ragged is.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _ragged

LENS = (16, 1, 7, 9, 16, 3, 12, 15)


# ---------------------------------------------------------------- helpers, nothing compiled

def test_padded_copy_keeps_real_positions_and_zeros_padding():
    x = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) + 1
    out = np.asarray(_ragged.padded_copy(x, (4, 2), "<f4", "t"))
    assert out[0].tobytes() == x[0].tobytes()
    assert out[1, :2].tobytes() == x[1, :2].tobytes()
    assert out[1, 2:].tobytes() == bytes(out[1, 2:].nbytes)
    assert x[1, 2:].min() > 0                       # the source is untouched


def test_zero_padding_writes_positive_zero_in_place():
    y = np.full((3, 5, 2), -1.0, dtype=np.float32)
    back = _ragged.zero_padding(y, (5, 1, 3), "t")
    assert back is y
    for i, n in enumerate((5, 1, 3)):
        assert (y[i, :n] == -1.0).all()
        assert y[i, n:].tobytes() == bytes(y[i, n:].nbytes)   # +0.0, never -0.0


def test_int32_ids_are_copied_by_position():
    ids = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.int32)
    out = np.asarray(_ragged.padded_copy(ids, (1, 3), "<i4", "t"))
    assert out.tolist() == [[5, 0, 0], [8, 9, 10]]


def test_last_real_rows_reads_each_rows_last_real_position():
    logits = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    out = np.asarray(_ragged.last_real_rows(logits, (2, 3), "t"))
    assert out.shape == (2, 1, 4)
    assert out[0, 0].tolist() == logits[0, 1].tolist()
    assert out[1, 0].tolist() == logits[1, 2].tolist()


@pytest.mark.parametrize("bad, kind, text", [
    ((0, 3), ValueError, "outside [1, L = 3]"),
    ((4, 3), ValueError, "outside [1, L = 3]"),
    ((3,), ValueError, "lengths has 1 entries but the batch has B = 2"),
    ((3.0, 3), TypeError, "is not an integer"),
    ((True, 3), TypeError, "is a bool"),
    (7, TypeError, "must be a sequence"),
])
def test_lengths_refused_by_name(bad, kind, text):
    with pytest.raises(kind) as exc:
        _ragged.lengths_for(bad, 2, 3, "Thing.forward")
    assert text in str(exc.value) and "Thing.forward" in str(exc.value)


def test_numpy_integers_are_admitted():
    assert _ragged.lengths_for(np.array([3, 1], dtype=np.int64), 2, 3, "t") == (3, 1)


def test_ragged_forward_refuses_a_state_before_calling():
    called = []
    with pytest.raises(ValueError, match="cannot carry a state"):
        _ragged.ragged_forward(called.append, np.zeros((2, 3, 1), np.float32), object(), (3, 3), "<f4", "t")
    assert called == []


def test_ragged_forward_hands_the_call_zeroed_padding_and_zeros_its_output():
    x = np.full((2, 3, 2), np.nan, dtype=np.float32)
    x[0, :1] = 1.0
    x[1] = 2.0
    seen = []

    def run(padded):
        seen.append(np.array(np.asarray(padded), copy=True))
        return ml.Array.from_list(np.full((2, 3, 2), 9.0, np.float32).tolist(), "<f4")
    out, lens = _ragged.ragged_forward(run, x, None, (1, 3), "<f4", "t")
    assert lens == (1, 3)
    assert np.isfinite(seen[0]).all() and (seen[0][0, 1:] == 0).all()
    out = np.asarray(out)
    assert (out[0, :1] == 9).all() and out[0, 1:].tobytes() == bytes(out[0, 1:].nbytes) and (out[1] == 9).all()


# ---------------------------------------------------------------- the surfaces, on this install's bindings

def _hw(shape, seed):
    return np.random.default_rng(seed).uniform(-0.125, 0.125, size=shape).astype(np.float32)


def _weights(shapes, ones, seed):
    return {n: (np.ones(s, np.float32) if n in ones else _hw(s, seed + i)) for i, (n, s) in enumerate(shapes.items())}


def _blocks():
    dm, di = 32, 64
    yield "transformer", lambda: ml.TransformerBlock(_weights({
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,), "q_proj.weight": (32, dm),
        "k_proj.weight": (16, dm), "v_proj.weight": (16, dm), "o_proj.weight": (dm, 32),
        "gate_proj.weight": (64, dm), "up_proj.weight": (64, dm), "down_proj.weight": (dm, 64)},
        ("input_layernorm.weight", "post_attention_layernorm.weight"), 1), n_heads=2, n_kv_heads=1)
    yield "mamba1", lambda: ml.Mamba1Block(_weights({
        "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4), "conv1d.bias": (di,),
        "x_proj.weight": (34, di), "dt_proj.weight": (di, 2), "dt_proj.bias": (di,), "A_log": (di, 16),
        "D": (di,), "out_proj.weight": (dm, di)}, ("norm.weight",), 20))
    yield "mamba2", lambda: ml.Mamba2Block(_weights({
        "block_norm.weight": (dm,), "in_proj.weight": (2 * di + 257, dm), "conv1d.weight": (di + 256, 1, 4),
        "conv1d.bias": (di + 256,), "dt_bias": (1,), "A_log": (1,), "D": (1,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)}, ("block_norm.weight", "norm.weight"), 40))
    yield "mamba3", lambda: ml.Mamba3Block(_weights({
        "block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + 3 + 32, dm), "dt_bias": (1,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (1, 128), "C_bias": (1, 128), "D": (1,),
        "out_proj.weight": (dm, di)}, ("block_norm.weight", "B_norm.weight", "C_norm.weight"), 60))


def _build_or_skip(name, make):
    try:
        return make()
    except (ImportError, RuntimeError, OSError) as exc:
        pytest.skip(f"{name}: no usable binding on this install ({type(exc).__name__}: {str(exc)[:120]})")


def _call_or_skip(name, fn):
    try:
        return fn()
    except (ImportError, OSError, NotImplementedError) as exc:
        pytest.skip(f"{name}: no usable binding on this install ({type(exc).__name__}: {str(exc)[:120]})")


def _assert_ragged(padded, alone, lens):
    out = np.asarray(padded)
    for i, n in enumerate(lens):
        a = np.ascontiguousarray(np.asarray(alone(i, n)))
        assert np.ascontiguousarray(out[i:i + 1, :n]).tobytes() == a.tobytes(), f"row {i} real positions moved"
        assert np.ascontiguousarray(out[i, n:]).tobytes() == bytes(out[i, n:].nbytes), f"row {i} padding not +0.0"


@pytest.mark.parametrize("name, make", list(_blocks()))
def test_block_forward_lengths_equals_rows_alone(name, make):
    blk = _build_or_skip(name, make)
    x = np.random.default_rng(3).standard_normal((len(LENS), 16, 32)).astype(np.float32)
    for i, n in enumerate(LENS):
        x[i, n:] = np.nan
    padded = _call_or_skip(name, lambda: blk.forward(x, lengths=LENS))
    _assert_ragged(padded, lambda i, n: blk.forward(np.ascontiguousarray(x[i:i + 1, :n])), LENS)
    with pytest.raises(ValueError, match="cannot carry a state"):
        blk.forward(x, blk.allocate_state(len(LENS), 16) if name == "transformer" else blk.allocate_state(len(LENS)),
                    lengths=LENS)


def _byte_lm_flat(shape):
    rng = np.random.default_rng(5)
    parts = []
    for name, s in zip(shape.parameter_names, shape.parameter_shapes):
        parts.append(np.ones(s, np.float32) if name.endswith(("norm1_w", "norm2_w"))
                     else rng.uniform(-0.125, 0.125, size=s).astype(np.float32))
    return np.ascontiguousarray(np.concatenate([p.reshape(-1) for p in parts]))


def _ragged_ids(lens, length, vocab):
    ids = np.random.default_rng(9).integers(0, vocab, size=(len(lens), length)).astype(np.int32)
    for i, n in enumerate(lens):
        ids[i, n:] = vocab + 7
    return ids


def test_host_language_model_logits_and_next_bytes_lengths():
    shape = ml.ByteLanguageModelConfig()
    m = _build_or_skip("LanguageModelInference", lambda: ml.LanguageModelInference(_byte_lm_flat(shape), shape=shape,
                                                                                   threaded=False))
    lens = (32, 1, 7, 9, 31, 3)
    ids = _ragged_ids(lens, shape.length, shape.vocab_size)
    padded = _call_or_skip("LanguageModelInference", lambda: m.logits(ids, lengths=lens))
    _assert_ragged(padded, lambda i, n: m.logits(np.ascontiguousarray(ids[i:i + 1, :n])), lens)
    assert m.next_bytes(ids, lengths=lens) == [m.next_bytes(np.ascontiguousarray(ids[i:i + 1, :n]))[0]
                                                for i, n in enumerate(lens)]
    with pytest.raises(ValueError, match="outside"):
        m.logits(ids, lengths=(33,) * len(lens))


def test_trainer_logits_lengths():
    shape = ml.ByteLanguageModelConfig()
    m = _build_or_skip("SmallByteLanguageModelTrainer", lambda: ml.SmallByteLanguageModelTrainer(
        _byte_lm_flat(shape), data_schedule={"dataset": "test", "order": "sequential"}, shape=shape))
    lens = (32, 1, 7, 9, 31, 3)
    ids = _ragged_ids(lens, shape.length, shape.vocab_size)
    padded = _call_or_skip("SmallByteLanguageModelTrainer", lambda: m.logits(ids, lengths=lens))
    _assert_ragged(padded, lambda i, n: m.logits(np.ascontiguousarray(ids[i:i + 1, :n])), lens)
    assert m.next_bytes(ids, lengths=lens) == [m.next_bytes(np.ascontiguousarray(ids[i:i + 1, :n]))[0]
                                                for i, n in enumerate(lens)]


def test_samba_forward_lengths():
    cfg = ml.SambaConfig(vocab=64, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    st = _build_or_skip("SambaStack", lambda: ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3))
    ids = _ragged_ids(LENS, 16, 64)
    padded = _call_or_skip("SambaStack", lambda: st.forward(ids, lengths=LENS))
    _assert_ragged(padded, lambda i, n: st.forward(np.ascontiguousarray(ids[i:i + 1, :n])), LENS)
