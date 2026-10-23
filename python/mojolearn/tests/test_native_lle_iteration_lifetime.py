# SPDX-License-Identifier: Apache-2.0
"""Public iterative LLE retains every raw-address input through the solve."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_lle_auto_iteration_keeps_factor_and_permutation_owners():
    # Start with allocator poisoning active before importing native modules.
    # The earlier 64-row regression used dense SVD and never crossed the
    # >200-row threshold for the shift-invert call with these three inputs.
    env = dict(os.environ, PYTHONMALLOC="debug", MOJOLEARN_NUMERIC_MODE="identical")
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    body = textwrap.dedent("""
        import gc
        import numpy as np
        from mojolearn import LocallyLinearEmbedding

        # Same finite input prefix and public settings as the accepted
        # x-decomp-manifold identity lane; keep auto and its default limit.
        x = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
        before = x.copy()
        first = None
        first_error = None
        for _ in range(2):
            model = LocallyLinearEmbedding(
                n_neighbors=10, n_components=2, random_state=0,
                numeric_mode="identical",
            ).fit(x)
            result = np.asarray(model.embedding_).copy()
            assert result.shape == (len(x), 2)
            assert np.isfinite(result).all()
            assert np.isfinite(model.reconstruction_error_)
            # The public operation already enforces convergence and unit
            # norms; also require repeated fitting to preserve all bits.
            if first is None:
                first = result.copy()
                first_error = model.reconstruction_error_
            else:
                np.testing.assert_array_equal(result.view(np.uint32), first.view(np.uint32))
                assert model.reconstruction_error_ == first_error
            del model
            gc.collect()
        np.testing.assert_array_equal(x, before)
    """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", body], env=env,
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
