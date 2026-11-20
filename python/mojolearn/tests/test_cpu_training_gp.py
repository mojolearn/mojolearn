# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the gp, gp-matern12, gp-matern32 and gp-matern52-ard
lanes: a runtime check that runs only where the gp host binding is built and
the package took the CPU-only path.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): the four lane kernels fit and predict through
the host binding at one and at four CPU row tasks and return the same bytes.
It is a plumbing check. The bit claim against the GPU columns is the CPU
identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_gp
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import os
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend


def _run_with_threads(threads, code):
    """Run `code` in a fresh interpreter with MOJOLEARN_CPU_THREADS set to
    `threads` (None: unset) and return its stdout lines. The host thread
    policy is read once per process (core/host_predict_threads.mojo), so a
    count set inside this process after the first host call changes
    nothing; only a new process honours it. The child imports the same
    mojolearn package this process holds."""
    import subprocess
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if threads is None:
        env.pop("MOJOLEARN_CPU_THREADS", None)
    else:
        env["MOJOLEARN_CPU_THREADS"] = threads
    pkg_parent = str(Path(mojolearn.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv.pop(1)); exec(sys.argv.pop(1))",
         pkg_parent, code],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (threads, proc.stdout[-2000:], proc.stderr[-4000:])
    return proc.stdout.strip().splitlines()


#: The GP thread-count check, run once per thread count in its own process:
#: the test's draw and kernels, one sha256 line per kernel over L_, alpha_,
#: the log marginal likelihood, the predicted mean and std.
_GP_THREADS_CODE = """
import hashlib
import numpy as np
import mojolearn
rng = np.random.default_rng(0)
x = rng.standard_normal((96, 4)).astype(np.float32)
y = (x[:, 0] - 0.5 * x[:, 2]).astype(np.float32)
xs = rng.standard_normal((16, 4)).astype(np.float32)
kernels = (
    lambda: mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0) + mojolearn.WhiteKernel(0.1),
    lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern(1.0, nu=0.5) + mojolearn.WhiteKernel(0.1),
    lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern(1.0, nu=1.5) + mojolearn.WhiteKernel(0.1),
    lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern([1.0, 2.0, 0.5, 4.0], nu=2.5)
    + mojolearn.WhiteKernel(0.1),
)
for make in kernels:
    m = mojolearn.GaussianProcessRegressor(kernel=make()).fit(x, y)
    assert m.info_ == 0, m.info_
    assert np.isfinite(m.log_marginal_likelihood_value_)
    mean, std = m.predict(xs, return_std=True)
    h = hashlib.sha256()
    for part in (np.asarray(m.L_).tobytes(), np.asarray(m.alpha_).tobytes(),
                 np.float64(m.log_marginal_likelihood_value_).tobytes(),
                 np.asarray(mean).tobytes(), np.asarray(std).tobytes()):
        h.update(part)
    print(h.hexdigest())
"""


def test_gaussian_process_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gp_host" not in _backend.host_families_built():
        print("SKIP: the gp host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gp_host")
    assert not bool(module.gp_host_sabotage()), "a sabotage build loaded outside the gate"
    # One and several row tasks must be the same numerical path; each count
    # runs in its own process (the thread policy is read once per process).
    outs = [_run_with_threads(threads, _GP_THREADS_CODE) for threads in ("1", "4")]
    assert len(outs[0]) == 4, outs[0]
    assert outs[0] == outs[1], "GP bytes moved with CPU row task count"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
