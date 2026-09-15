# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public CPU neural inference (lane/inference-tokenizer-neural, 2026-09-15):
`MLPInference` and `TransformerBlockInference` over the shipped
`_mojolearn_neural_host` binding.

What this file holds, on the host it runs on:
  - both classes read back a CPU, IDENTICAL, CPU-column binding;
  - MLP logits from a saved checkpoint equal the four-weight constructor's,
    each row equals the row alone, and every checkpoint refusal is by name;
  - the Transformer stateless forward equals itself split by batch rows and
    by ragged lengths, and every training entry is refused by name;
  - when the source reference bindings are also present (a CPU identity
    column, never a wheel), both equal `SmallMLPTrainer.predict_logits` and
    `TransformerBlock.forward` byte for byte.
Cross-vendor identity is not asserted here: `tools/identity_break.py`'s
mlp, transformer and transformer-window lanes diff these classes' held-out
and batch cells against the committed Apple, NVIDIA and AMD columns.

    cd python && python3 -m pytest -q mojolearn/tests/test_neural_inference.py

Without the binding built (`bindings/build_neural_host.sh`) pytest skips with
the load error; a skip is not a pass. A binding built with
`-D MOJOLEARN_HOST_SABOTAGE=1` is refused without
MOJOLEARN_HOST_ALLOW_SABOTAGE=1, and with it the reference comparisons fail.
"""
import json
import os
import tempfile

import pytest

np = pytest.importorskip("numpy")

import mojolearn as ml
from mojolearn import _backend

_SHAPES = ((16, 8), (16,), (3, 16), (3,))
_T_NAMES = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_proj.weight",
            "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight",
            "up_proj.weight", "down_proj.weight")


@pytest.fixture(scope="module")
def binding():
    try:
        return _backend.load_host_module("_mojolearn_neural_host")
    except ImportError as exc:
        pytest.skip(f"neural host binding not built: {exc}")


def _mlp_weights():
    return [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
            for s in _SHAPES]


def _x(rows, cols=8, seed=0):
    return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)


def _block_weights(dm=32, nh=2, nkv=1, it=64, seed=1):
    hd = dm // nh
    shapes = ((dm,), (dm,), (nh * hd, dm), (nkv * hd, dm), (nkv * hd, dm), (dm, nh * hd), (it, dm), (it, dm), (dm, it))
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.1).astype(np.float32) for n, s in zip(_T_NAMES, shapes)}


def _reference_present():
    try:
        _backend.load_host_module("_mojolearn_training_host")
        _backend.load_host_module("_mojolearn_transformer_host")
    except ImportError:
        return False
    return _backend._CPU_ONLY is not None


def test_binding_reads_back_cpu_identical(binding):
    assert str(binding.neural_host_vendor()) == "cpu"
    assert int(binding.neural_host_numeric_mode()) == 1
    assert str(binding.neural_host_column()) == "cpu"
    assert bool(binding.neural_host_sabotage()) is False or os.environ.get("MOJOLEARN_HOST_ALLOW_SABOTAGE") == "1"


def test_binding_exports_no_training_entry(binding):
    names = {n for n in dir(binding) if not n.startswith("_")}
    assert names == {"neural_host_numeric_mode", "neural_host_vendor", "neural_host_column",
                     "neural_host_sabotage", "mlp_forward_logits", "transformer_forward_fresh"}, names


def test_mlp_logits_shape_rows_alone_and_checkpoint(binding):
    inf = ml.MLPInference(*_mlp_weights())
    X = _x(64)
    got = np.asarray(inf.predict_logits(X))
    assert got.shape == (64, 3) and got.dtype == np.float32
    for i in (0, 1, 7, 63):
        assert np.asarray(inf.predict_logits(X[i:i + 1])).tobytes() == got[i:i + 1].tobytes(), i
    assert np.asarray(inf.predict_logits(X[1:8])).tobytes() == got[1:8].tobytes()
    assert set(inf.weights_) == {"weight1", "bias1", "weight2", "bias2"}


def _checkpoint_from_weights(tmp):
    """A trainer checkpoint written by the trainer's own encoder, without
    running the trainer (no binding needed to write it)."""
    from mojolearn import _mlp_impl as M
    from mojolearn._buffer import zeros
    w = [M._array(v, s, n) for v, s, n in zip(_mlp_weights(), _SHAPES, M._NAMES)]
    state = dict(schema=M._STATE_SCHEMA, architecture=[8, 16, 3], numeric_mode="identical",
                 parameter_order=list(M._NAMES), weights=dict(zip(M._NAMES, w)),
                 optimizer=dict(kind="AdamW", step=0, m=zeros((M._TOTAL,), "<f4"),
                                v=zeros((M._TOTAL,), "<f4"), flags=zeros((4,), "<i4")),
                 config=dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01),
                 data_schedule={"dataset": "test"})
    payload = M._encode_state(state)
    import hashlib
    env = dict(schema=M._FILE_SCHEMA, payload=payload,
               payload_sha256=hashlib.sha256(M._canonical(payload)).hexdigest())
    path = os.path.join(tmp, "mlp.json")
    with open(path, "wb") as fh:
        fh.write(M._canonical(env) + b"\n")
    return path


def test_mlp_from_checkpoint_equals_constructor_and_refuses_by_name(binding):
    tmp = tempfile.mkdtemp()
    path = _checkpoint_from_weights(tmp)
    X = _x(32, seed=3)
    a = np.asarray(ml.MLPInference.from_checkpoint(path).predict_logits(X))
    b = np.asarray(ml.MLPInference(*_mlp_weights()).predict_logits(X))
    assert a.tobytes() == b.tobytes()
    env = json.load(open(path))
    env["payload"]["data_schedule"] = {"dataset": "tampered"}
    bad = os.path.join(tmp, "bad.json")
    json.dump(env, open(bad, "w"))
    with pytest.raises(ValueError, match="integrity mismatch"):
        ml.MLPInference.from_checkpoint(bad)
    env["schema"] = "other"
    json.dump(env, open(bad, "w"))
    with pytest.raises(ValueError, match="schema mismatch"):
        ml.MLPInference.from_checkpoint(bad)
    with open(bad, "wb") as fh:
        fh.write(b"{" * 40000)
    with pytest.raises(ValueError, match="size bound"):
        ml.MLPInference.from_checkpoint(bad)


def test_mlp_refusals(binding):
    w = _mlp_weights()
    with pytest.raises(ValueError, match="shape"):
        ml.MLPInference(w[0].T.copy(), *w[1:])
    inf = ml.MLPInference(*w)
    with pytest.raises(ValueError, match=r"shape \(batch, 8\)"):
        inf.predict_logits(_x(4, cols=7))
    with pytest.raises(ValueError, match=r"\[1, 256\]"):
        inf.predict_logits(_x(257))
    bad = w[1].copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ml.MLPInference(w[0], bad, w[2], w[3])


def test_transformer_forward_rows_ragged_and_refusals(binding):
    tw = _block_weights()
    for window in (0, 8):
        blk = ml.TransformerBlockInference(tw, n_heads=2, n_kv_heads=1, window=window)
        x = np.random.default_rng(5).standard_normal((4, 16, 32)).astype(np.float32)
        y = np.asarray(blk.forward(x))
        assert y.shape == (4, 16, 32)
        for i in range(4):
            assert np.asarray(blk.forward(x[i:i + 1])).tobytes() == y[i:i + 1].tobytes(), (window, i)
        # causal: a prefix is the prefix of the whole
        assert np.asarray(blk.forward(np.ascontiguousarray(x[:, :7]))).tobytes() == np.ascontiguousarray(y[:, :7]).tobytes()
        lengths = [16, 7, 1, 12]
        yr = np.asarray(blk.forward(x, lengths=lengths))
        for i, n in enumerate(lengths):
            alone = np.asarray(blk.forward(np.ascontiguousarray(x[i:i + 1, :n])))
            assert yr[i:i + 1, :n].tobytes() == alone.tobytes(), (window, i)
            assert not yr[i, n:].any()
    blk = ml.TransformerBlockInference(tw, n_heads=2, n_kv_heads=1)
    with pytest.raises(ValueError, match="carried state is not supported"):
        blk.forward(np.zeros((1, 2, 32), np.float32), state=object())
    for call in (lambda: blk.step(np.zeros((1, 1, 32), np.float32), None),
                 lambda: blk.allocate_state(1, 4),
                 lambda: blk.backward(np.zeros((1, 2, 32), np.float32), np.zeros((1, 2, 32), np.float32))):
        with pytest.raises(NotImplementedError, match="stateless forward only"):
            call()
    with pytest.raises(ValueError, match="d_model must equal n_heads"):
        ml.TransformerBlockInference(tw, n_heads=3)


def test_equals_the_reference_path_when_present(binding):
    """A CPU identity column only: the source reference bindings are there."""
    if not _reference_present():
        pytest.skip("the source reference training and transformer host bindings are not built here (a wheel has neither)")
    from mojolearn._cpu_reference import reference_training
    X = _x(64, seed=9)
    t = np.random.default_rng(9).integers(0, 3, 64).astype(np.int32)
    with reference_training():
        tr = ml.SmallMLPTrainer(*_mlp_weights(), data_schedule={"dataset": "test"})
        tr.train_step(X, t)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "trained.json")
    tr.save_checkpoint(path)
    assert (np.asarray(ml.MLPInference.from_checkpoint(path).predict_logits(X)).tobytes()
            == np.asarray(tr.predict_logits(X)).tobytes())
    tw = _block_weights(seed=11)
    x = np.random.default_rng(11).standard_normal((2, 16, 32)).astype(np.float32)
    for window in (0, 8):
        a = np.asarray(ml.TransformerBlock(tw, n_heads=2, n_kv_heads=1, window=window).forward(x))
        b = np.asarray(ml.TransformerBlockInference(tw, n_heads=2, n_kv_heads=1, window=window).forward(x))
        assert a.tobytes() == b.tobytes(), window
