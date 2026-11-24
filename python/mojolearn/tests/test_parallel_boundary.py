# SPDX-License-Identifier: Apache-2.0
"""Parallel drivers public boundary: the refusals of a route a driver does not have, in words.

The sharded answers held to the plain fits are verify lanes (psd-*); what stays here is what a digest cannot see.
"""
import mojolearn
from mojolearn import _backend


# ---------------------------------------------------------------- moved from test_cpu_training_par_classical.py (vdv-pytest-a g2)

def test_cooperative_drivers_refuse_the_cpu_route_by_name():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    import numpy as np
    from mojolearn.parallel_classical import fit_logistic
    X = np.zeros((4, 2), np.float32)
    try:
        fit_logistic(mojolearn.LogisticRegression(), X, np.array([0, 1, 0, 1]), devices=(0,))
    except NotImplementedError as exc:
        assert "no CPU implementation of the cooperative multi-GPU driver glm_fit" in str(exc), str(exc)
    else:
        raise AssertionError("a cooperative driver was admitted on a CPU-only install")
