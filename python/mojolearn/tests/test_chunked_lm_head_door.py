# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door to the chunked LM head v2 (lane/exposure-leftovers,
2026-09-23): `mojolearn.training.chunked_lm_head_loss`.

Both training bindings register `chunked_lm_head_v2_loss` and
`chunked_lm_head_v2_train`, the GPU binding over the device kernels and the
training host binding over training/checks/chunked_lm_head_oracle.mojo. Before
this lane only the GPU binding registered them and nothing in Python called
either, so every test below fails at that commit: the source tests because the
host registration, the manifest exports and the public name were absent, the
runtime tests because `training.chunked_lm_head_loss` did not exist.

SOURCE CHECKS (run anywhere, nothing built, `mojolearn` never imported):
the two bindings register the same two names with the same address and param
counts; the manifest exports them and ships the oracle as a host module; the
public name is in `training.__all__`; the laneless declaration names what the
missing identity lane owes.

RUNTIME CHECKS (run only where the training host binding is built and the
package took the CPU-only route; they print SKIP and return otherwise): the
loss agrees with a float64 reference within float32 rounding on a vocabulary
that crosses the 256 chunk and ends in a one-token tail; the loss is the same
bits with and without the gradients (the loss stage is the forward the train
stage runs first); two calls are the same bytes; the gradients agree with the
float64 reference; each refusal fires by name before any output. The bit
claim against a GPU column is the identity lane's, which is owed.

    cd python && python3 -m pytest mojolearn/tests/test_chunked_lm_head_door.py -q
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOST = "bindings/_mojolearn_training_host.mojo"
GPU = "bindings/_mojolearn_training.mojo"
NAMES = {"chunked_lm_head_v2_loss": 6, "chunked_lm_head_v2_train": 8}


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _registered(text):
    return set(re.findall(r'def_function\[\w+\]\("(\w+)"\)', text))


def _addr_count(text, name):
    m = re.search(r'_addrs\(addresses, (\d+), "%s"\)' % name, text)
    return int(m.group(1)) if m else None


def _surface():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_clh_surface", ROOT / "python" / "mojolearn" / "host_surface.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ source checks


def test_both_bindings_register_the_same_two_entries():
    host, gpu = _read(HOST), _read(GPU)
    for name, n_addr in NAMES.items():
        assert name in _registered(gpu), name
        assert name in _registered(host), f"{HOST} does not register {name}"
        assert _addr_count(gpu, name) == n_addr, name
        assert _addr_count(host, name) == n_addr, name
        assert f'_params(params, 3, "{name}")' in host, name
    assert "from training.checks.chunked_lm_head_oracle import" in host
    # the host arm is the oracle, never the device module
    assert not re.search(r"^\s*from training\.chunked_lm_head_v2 import", host, re.M)


def test_manifest_exports_the_entries_and_ships_the_oracle():
    fam = _surface().family("training")
    for name in NAMES:
        assert name in fam["exports"], name
    assert "training/checks/chunked_lm_head_oracle.mojo" in fam["host_modules"]
    assert "chunked_lm_head_loss" in fam["classes"]
    oracle = _read("training/checks/chunked_lm_head_oracle.mojo")
    assert not re.search(r"^\s*from (std\.gpu|max\.gpu)", oracle, re.M)
    assert 'is_defined["MOJOLEARN_HOST_SABOTAGE"]' in oracle


def test_the_door_is_public_and_its_debt_is_declared():
    training = _read("python/mojolearn/training.py")
    assert re.search(r"__all__ = \[.*'chunked_lm_head_loss'", training, re.S)
    impl = _read("python/mojolearn/_training_impl.py")
    assert "def chunked_lm_head_loss(" in impl
    for name in NAMES:
        assert f"binding.{name}(" in impl, name
    accounting = _read("tools/lane_accounting.py")
    assert '"training.chunked_lm_head_loss": (' in accounting


# ----------------------------------------------------------- runtime checks


def _cpu_only_with_training_host():
    try:
        from mojolearn import _backend
    except ImportError as exc:
        print(f"SKIP: mojolearn does not import here ({exc})")
        return False
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if "_mojolearn_training_host" not in _backend.host_families_built():
        print("SKIP: _mojolearn_training_host is not built")
        return False
    return True


def _inputs(rows=5, vocab=513, width=8):
    import numpy as np
    rng = np.random.default_rng(23)
    h = rng.uniform(-1.0, 1.0, (rows, width)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, (vocab, width)).astype(np.float32)
    # one target in each chunk and one on the one-token tail
    t = np.array([0, 255, 256, 511, 512][:rows], dtype=np.int64)
    return h, w, t


def _reference(h, w, t):
    import numpy as np
    h64, w64 = h.astype(np.float64), w.astype(np.float64)
    logits = h64 @ w64.T
    shift = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shift)
    p /= p.sum(axis=1, keepdims=True)
    rows = h.shape[0]
    loss = -np.mean(np.log(p[np.arange(rows), t]))
    dlogit = p.copy()
    dlogit[np.arange(rows), t] -= 1.0
    dlogit /= rows
    return loss, dlogit @ w64, dlogit.T @ h64


def test_loss_and_gradients_match_a_float64_reference_when_built():
    if not _cpu_only_with_training_host():
        return
    import numpy as np
    from mojolearn import training
    h, w, t = _inputs()
    loss = training.chunked_lm_head_loss(h, w, t)
    loss2, dh, dw = training.chunked_lm_head_loss(h, w, t, return_grad=True)
    ref_loss, ref_dh, ref_dw = _reference(h, w, t)
    assert isinstance(loss, float)
    np.testing.assert_allclose(loss, ref_loss, rtol=2e-5)
    assert np.float32(loss).tobytes() == np.float32(loss2).tobytes(), \
        "the loss stage and the train stage disagree on the loss"
    assert np.asarray(dh).shape == h.shape and np.asarray(dw).shape == w.shape
    np.testing.assert_allclose(np.asarray(dh), ref_dh, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dw), ref_dw, rtol=1e-4, atol=1e-6)


def test_two_calls_are_the_same_bytes_when_built():
    if not _cpu_only_with_training_host():
        return
    import numpy as np
    from mojolearn import training
    h, w, t = _inputs()
    a = training.chunked_lm_head_loss(h, w, t, return_grad=True)
    b = training.chunked_lm_head_loss(h, w, t, return_grad=True)
    assert np.float32(a[0]).tobytes() == np.float32(b[0]).tobytes()
    assert np.asarray(a[1]).tobytes() == np.asarray(b[1]).tobytes()
    assert np.asarray(a[2]).tobytes() == np.asarray(b[2]).tobytes()


def test_refusals_fire_by_name_when_built():
    if not _cpu_only_with_training_host():
        return
    import numpy as np
    import pytest
    from mojolearn import training
    h, w, t = _inputs()
    with pytest.raises(ValueError, match="vocab must be at least 2"):
        training.chunked_lm_head_loss(h, w[:1], np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError, match="width"):
        training.chunked_lm_head_loss(h, w[:, :4], t)
    with pytest.raises(ValueError, match="targets for"):
        training.chunked_lm_head_loss(h, w, t[:4])
    with pytest.raises(TypeError, match="float32"):
        training.chunked_lm_head_loss(h.astype(np.float64), w, t)
    bad_t = t.copy()
    bad_t[2] = 513
    with pytest.raises(Exception, match="target outside"):
        training.chunked_lm_head_loss(h, w, bad_t, return_grad=True)
    bad_h = h.copy()
    bad_h[1, 3] = np.nan
    with pytest.raises(Exception, match="non-finite|nonfinite|finite"):
        training.chunked_lm_head_loss(bad_h, w, t)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
