# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.training` functions at the Python boundary: the refusals. The losses and gradients are the
fn-training verify lanes."""
import numpy as np
import pytest


# ---------------------------------------------------------------- moved from test_chunked_lm_head_door.py
# Runtime checks (run only where the training host binding is built and the package took the CPU-only route; they
# print SKIP and return otherwise): each refusal fires by name before any output.

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


def _chunked_inputs(rows=5, vocab=513, width=8):
    rng = np.random.default_rng(23)
    h = rng.uniform(-1.0, 1.0, (rows, width)).astype(np.float32)
    w = rng.uniform(-0.5, 0.5, (vocab, width)).astype(np.float32)
    # one target in each chunk and one on the one-token tail
    t = np.array([0, 255, 256, 511, 512][:rows], dtype=np.int64)
    return h, w, t


def test_refusals_fire_by_name_when_built():
    if not _cpu_only_with_training_host():
        return
    from mojolearn import training
    h, w, t = _chunked_inputs()
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
    with pytest.raises(Exception, match="NaN in chunked lm head hidden at flat index 11"):
        training.chunked_lm_head_loss(bad_h, w, t)
