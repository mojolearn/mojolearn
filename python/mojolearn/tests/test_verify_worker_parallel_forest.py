# SPDX-License-Identifier: Apache-2.0
"""fit_forest through the verifier's own worker path completes.

`verify --lanes ...,par-forest` stopped at cell/par-forest/base on the L4:
the verify worker sat in `_parallel_pool.DevicePool._receive` inside
`parallel_ensemble.fit_forest` until the 120 s cell limit, while the same
fit_forest finishes in seconds from pytest. This runs that exact cell the
way the verifier does (a fresh `_verify_worker` process in its own session,
the CLI's thread-budget environment, the harness's base fixture), with the
verifier's own cell limit. A timeout reports the stacks of the worker and of
its parallel driver workers (`_verify_worker._stacks_on_timeout`).
"""
import os

import pytest

from mojolearn import _backend


BUDGET = ("MOJOLEARN_CPU_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
          "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
          "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS",
          "OMP_MAX_ACTIVE_LEVELS", "OMP_DYNAMIC", "MOJOLEARN_VERIFY_CPU_THREADS")


# Three environments, so one run separates the causes: the CLI's whole
# thread budget (what `verify` runs under), only mojolearn's own host thread
# ceiling, and none (the verifier's process layout alone: own session, file
# stdout/stderr, a fresh worker process).
@pytest.mark.parametrize("budget", ["cli", "mojolearn-threads", "none"])
def test_par_forest_cell_completes_inside_the_verify_worker(monkeypatch, budget):
    if _backend.vendor() in (None, "cpu"):
        pytest.skip("the par-forest cell's sharded workers need a GPU device")
    from mojolearn import _verify_all as verifier
    from mojolearn import _verify_worker as worker
    from mojolearn._verify_resources import budget_environment
    try:
        path, _how = verifier.harness_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    harness = verifier.load_harness(path)
    # The CLI re-executes `verify` under its thread budget before any native
    # import; its worker (and the worker's driver children) inherit it.
    _effective, env = budget_environment(1)
    for key in BUDGET:
        if budget == "cli" or (budget == "mojolearn-threads" and key == "MOJOLEARN_CPU_THREADS"):
            monkeypatch.setenv(key, env[key])
        else:
            monkeypatch.delenv(key, raising=False)
    X, yc, yr = harness.fixture("base")
    held = harness.heldout("base")
    request = dict(action="cell", harness=os.path.abspath(path), table="", lane="par-forest",
                   fixture="base", repeats=1, extra_parts=[],
                   fixture_hashes=dict(X=harness._h(X), y_clf=harness._h(yc),
                                       y_reg=harness._h(yr), held=harness._h(held)))
    result = worker.run(request, timeout=120)
    parts = result["value"]
    value, error = parts["train"]
    assert error is None, error
    assert value
