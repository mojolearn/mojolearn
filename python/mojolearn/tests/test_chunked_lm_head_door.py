# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door to the chunked LM head v2: `mojolearn.training.chunked_lm_head_loss`.

Runtime checks (run only where the training host binding is built and the
package took the CPU-only route; they print SKIP and return otherwise): each
refusal fires by name before any output. The Float64-reference agreement, the
loss-stage/train-stage agreement and the repeat words are the native check
checks/native_tests/training_chunked_lm_head_door.mojo
(pixi run check-native-training-chunked-lm-head-door-cpu).

    cd python && python3 -m pytest mojolearn/tests/test_chunked_lm_head_door.py -q
"""


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
    with pytest.raises(Exception, match="NaN in chunked lm head hidden at flat index 11"):
        training.chunked_lm_head_loss(bad_h, w, t)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
