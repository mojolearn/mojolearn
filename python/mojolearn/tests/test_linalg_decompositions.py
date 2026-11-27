# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public decomposition exports, result objects and argument binding.

The CPU and GPU numerical tests live in linalg_numpy_cases.mojo and
linalg_numerical.mojo and use stored NumPy float64 reference values.
"""
import numpy as np
import pytest

from mojolearn import host_surface, linalg


def _host_built():
    from pathlib import Path
    from mojolearn import _backend
    return Path(_backend.host_module_path("_mojolearn_linalg_host")).exists()


needs_host = pytest.mark.skipif(
    not _host_built(), reason="_mojolearn_linalg_host.so is not built here"
)


def test_the_three_doors_ship_in_the_linalg_family():
    """A door nobody declared is a door the CPU column will not check: the
    lane harness REFUSES a lane whose family does not claim it."""
    fam = host_surface.family("linalg")
    for name in ("qr_r", "eigh", "svdvals"):
        assert name in fam["exports"], name
    for lane in ("linalg-qr", "linalg-eigh", "linalg-svdvals"):
        assert lane in fam["training_lanes"], lane
    assert "decomposition/host/linalg_public.mojo" in fam["host_modules"]


@needs_host
def test_eigh_triangle_argument():
    a = np.array([[2, 7], [1, 3]], dtype="f4")
    default = linalg.eigh(a)
    lower = linalg.eigh(a, UPLO="L")
    assert len(default) == len(lower) == 2
    for actual, expected in zip(default, lower):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    with pytest.raises(ValueError):
        linalg.eigh(a, UPLO="X")


@needs_host
@pytest.mark.parametrize("rows,cols", [(16, 4), (48, 5), (5, 9), (6, 6)])
def test_qr_and_svd_return_objects(rows, cols):
    a = np.eye(rows, cols, dtype="f4")
    k = min(rows, cols)
    qr = linalg.qr(a, mode="reduced")
    q, r = qr
    assert q.shape == qr.Q.shape == (rows, k)
    assert r.shape == qr.R.shape == (k, cols)
    assert np.asarray(q).tobytes() == np.asarray(qr.Q).tobytes()
    with pytest.raises(ValueError, match="unrecognized mode"):
        linalg.qr(a, mode="economic")
    for full in (False, True):
        u, s, vh = linalg.svd(a, full_matrices=full)
        assert u.shape == (rows, rows if full else k)
        assert s.shape == (k,)
        assert vh.shape == (cols if full else k, cols)
    values = linalg.svd(a, compute_uv=False)
    assert np.asarray(values).tobytes() == np.asarray(linalg.svdvals(a)).tobytes()
