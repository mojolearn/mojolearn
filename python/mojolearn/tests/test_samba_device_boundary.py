# SPDX-License-Identifier: Apache-2.0
"""`device=` on `SambaStack`, at the Python boundary. Same rule and same words as every other class.

The device is a constructor parameter and a keyword of `from_checkpoint`, checked where it is given, before
any field is written or any file is read, and resolved once into the stack's one native route (training
entries, primitives, blocks, optimizer and decode program together); no operation reads it again. A stack
runs on the device of its installation: on a GPU installation `device="cpu"` is refused in words, because
the blocks' backward has no CPU images loaded beside the GPU set.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import SambaConfig, SambaStack, training

TWO_DEVICES = mojolearn.vendor() != "cpu"


def config():
    return SambaConfig(13, 8, ["attention"], n_heads=2, intermediate=11)


def stack(**options):
    return SambaStack(config(), generator=training.Generator(7), **options)


def ids():
    return (np.arange(12, dtype=np.int32) % 13).reshape(2, 6)


def test_default_device_is_none_and_runs():
    model = stack()
    assert model.device is None
    assert np.asarray(model.forward(ids())).shape == (2, 6, 13)
    assert np.isfinite(model.train_step(ids(), ids())["loss"])


@pytest.mark.parametrize("value", ["CPU", "cuda", 0, True, b"cpu"])
def test_unknown_device_is_refused_at_construction(value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        stack(device=value)


def test_installation_device_by_name():
    name = "gpu" if TWO_DEVICES else "cpu"
    model = stack(device=name)
    assert model.device == name
    reference = stack()
    assert np.array_equal(np.asarray(model.forward(ids())), np.asarray(reference.forward(ids())))
    assert model.train_step(ids(), ids())["loss"] == reference.train_step(ids(), ids())["loss"]


def test_other_device_is_refused_at_construction():
    if TWO_DEVICES:
        with pytest.raises(ValueError, match="device='cpu' is not available for SambaStack"):
            stack(device="cpu")
    else:
        with pytest.raises(ValueError, match="device='gpu' needs a GPU build"):
            stack(device="gpu")


def test_refused_constructor_leaves_a_live_stack_untouched():
    model = stack()
    before = np.asarray(model.forward(ids())).copy()
    with pytest.raises(ValueError):
        model.__init__(config(), generator=training.Generator(9), device="metal")
    assert model.device is None
    assert np.array_equal(np.asarray(model.forward(ids())), before)


def test_checkpoint_restore_takes_and_refuses_device(tmp_path):
    model = stack()
    model.train_step(ids(), ids())
    path = tmp_path / "samba.ckpt"
    model.save_checkpoint(path)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        SambaStack.from_checkpoint(path, device="metal")
    other = "cpu" if TWO_DEVICES else "gpu"
    with pytest.raises(ValueError):
        SambaStack.from_checkpoint(path, device=other)
    name = "gpu" if TWO_DEVICES else "cpu"
    restored = SambaStack.from_checkpoint(path, device=name)
    assert restored.device == name
    assert SambaStack.from_checkpoint(path).device is None
    assert np.array_equal(np.asarray(restored.forward(ids())), np.asarray(model.forward(ids())))


def test_changed_numeric_mode_is_admitted_again():
    model = stack()
    model.forward(ids())
    model.numeric_mode = "no-such-mode"
    with pytest.raises(ValueError):
        model.forward(ids())


def test_decode_session_and_state_dict_use_the_stack_route():
    model = stack()
    state = model.allocate_state(2, 8)
    session = model.decode_session(state)
    assert session.is_open
    session.close()
    saved = model.state_dict()
    model.load_state_dict(saved)
    assert np.array_equal(np.asarray(model.state_dict()["parameters"]), np.asarray(saved["parameters"]))


# moved from python/mojolearn/tests/test_neural_inference.py
# SambaInference at the public boundary: rows alone, ragged lengths, returned
# shape/dtype and the refusals in words (vocabulary, state type, forward only).
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


@pytest.fixture(scope="module")
def ninfs_binding():
    try:
        return _backend.load_host_module("_mojolearn_neural_host")
    except ImportError as exc:
        pytest.skip(f"neural host binding not built: {exc}")


def _ninfs_samba_config(tied=True):
    return ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64,
                          tie_embeddings=tied)


def _ninfs_samba_weights(cfg, seed=5):
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.1).astype(np.float32) for n, s in cfg.registry()}


@pytest.mark.parametrize("tied", (True, False))
def test_samba_forward_rows_ragged_and_refusals(ninfs_binding, tied):
    cfg = _ninfs_samba_config(tied)
    inf = ml.SambaInference(cfg, _ninfs_samba_weights(cfg))
    ids = np.random.default_rng(7).integers(0, 256, (4, 16)).astype(np.int32)
    y = np.asarray(inf.forward(ids))
    assert y.shape == (4, 16, 256) and y.dtype == np.float32
    for i in range(4):
        assert np.asarray(inf.logits(ids[i:i + 1])).tobytes() == y[i:i + 1].tobytes(), i
    lengths = [16, 3, 1, 9]
    junk = ids.copy()
    for i, n in enumerate(lengths):
        junk[i, n:] = 999
    yr = np.asarray(inf.forward(junk, lengths=lengths))
    for i, n in enumerate(lengths):
        assert yr[i:i + 1, :n].tobytes() == np.asarray(inf.forward(np.ascontiguousarray(ids[i:i + 1, :n]))).tobytes()
        assert not yr[i, n:].any()
    with pytest.raises(ValueError, match=r"\[0, vocab\)"):
        inf.forward(np.full((1, 2), 256, np.int32))
    # the native admission refuses a non-native argument first (samba_admission)
    with pytest.raises(TypeError, match="state must be a SambaState|native input types"):
        inf.forward(ids, state=object())
    for name in ("loss", "train_step"):
        with pytest.raises(NotImplementedError, match="inference is forward only"):
            getattr(inf, name)(None, None)
    with pytest.raises(ValueError, match="state is required"):
        inf.step(ids[:, 0], None)
    w = _ninfs_samba_weights(cfg)
    w.pop("norm_f.weight")
    with pytest.raises(ValueError, match="weight dict mismatch"):
        ml.SambaInference(cfg, w)


# moved from python/mojolearn/tests/test_samba_checkpoint_native.py
# SambaStack.from_checkpoint refuses a missing path and a Python path callback, natively.
import pytest

from mojolearn import SambaStack
from mojolearn._verify_native_runtime import audit_native_call


def _sckp_assert_refusal(result, error):
    assert isinstance(result.exception, error), result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not result.report()[name], result.report()


def test_checkpoint_missing_path_and_python_path_callback_refusals(tmp_path):
    _sckp_assert_refusal(audit_native_call(SambaStack.from_checkpoint, tmp_path / "missing"), FileNotFoundError)

    class CallbackPath:
        def __fspath__(self):
            pytest.fail("Python path callback executed")

    _sckp_assert_refusal(audit_native_call(SambaStack.from_checkpoint, CallbackPath()), TypeError)


# moved from python/mojolearn/tests/test_samba_cpu_inference_native.py
# CPU SambaInference refuses training (loss, train_step) natively with NotImplementedError.
import numpy as np
import pytest

from mojolearn import SambaConfig, SambaInference, lowbit
from mojolearn._verify_native_runtime import audit_native_call


def _scpu_packed():
    config = SambaConfig(19, 8, ["attention"], n_heads=2, n_kv_heads=1, intermediate=11, tie_embeddings=False)
    rng = np.random.default_rng(412)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4") for name, shape in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    return config, lowbit.pack(weights, "float32")


@pytest.mark.parametrize("method", ["loss", "train_step"])
def test_cpu_samba_training_refusal_is_native(method):
    config, packed = _scpu_packed()
    model = SambaInference(config, packed)
    ids = np.array([[1, 2]], dtype="i4")
    result = audit_native_call(getattr(model, method), ids, ids)
    assert isinstance(result.exception, NotImplementedError), result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not result.report()[name], result.report()


# moved from python/mojolearn/tests/test_samba_inference_native.py
# SambaStack inference refusals (bad state, empty batch, unknown token) are native; the read-only forward admission
# still refuses a Python hook in a field it reads.
import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack
from mojolearn._verify_native_runtime import audit_native_call


def _sinf_model(layers):
    dm = 32 if "mamba3" in layers else 8
    config = SambaConfig(19, dm, layers, n_heads=2, n_kv_heads=1, intermediate=48 if dm == 32 else 11,
                         tie_embeddings=True, dropout=0.0)
    rng = np.random.default_rng(428)
    weights = {name: rng.uniform(-0.04, 0.04, shape).astype("f4") for name, shape in config.registry()}
    for name in weights:
        if "norm" in name or name.endswith(".D"):
            weights[name].fill(1)
    return SambaStack(config, weights=weights)


@pytest.mark.parametrize("operation,args", [("step", (np.array([1], dtype="i4"), None)),
                                           ("allocate_state", (0, 2)),
                                           ("forward", (np.array([[999]], dtype="i4"),))])
def test_public_inference_refusals_are_native(operation, args):
    model = _sinf_model(["attention"])
    result = audit_native_call(getattr(model, operation), *args)
    assert isinstance(result.exception, ValueError), result.report()
    report = result.report()
    for name in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads"):
        assert not report[name], report


class _SinfHookedInt(int):
    def __index__(self):  # pragma: no cover - must never run
        raise AssertionError("Python hook ran inside a native Samba forward")


def test_read_only_forward_still_refuses_hooks_in_fields_it_reads():
    # The read-only admission walks only the fields forward reads; each of them
    # keeps the full walk's refusal.
    model = _sinf_model(["attention"])
    ids = np.array([[1, 2, 3]], dtype="i4")
    audit_native_call(model.forward, ids).assert_native()
    vars(model)["n_total"] = _SinfHookedInt(model.n_total)
    with pytest.raises(TypeError):
        model.forward(ids)
