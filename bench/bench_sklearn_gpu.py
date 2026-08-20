#!/usr/bin/env python3
"""Does scikit-learn's Array API GPU path actually reach the Apple GPU?

WHY THIS FILE EXISTS
--------------------
`bench_sklearn.py` races our GPU against scikit-learn's CPU and says so in
its docstring: "cuML and cuVS cannot run on Apple silicon at all, so there is
no GPU arm on the other side to compare against."

That sentence was never checked. scikit-learn 1.9 has Array API dispatch, and
several estimators declare `array_api_support`, which means a user CAN hand
them a torch tensor on the `mps` device and reach the same Apple GPU we do.
If that works, our PCA and OLS ratios are measured against a baseline running
with one hand tied, and the headline is wrong in our favor.

So this file asks the question the harness was assuming the answer to. It is
scikit-learn against scikit-learn: the SAME estimator, the SAME data, CPU
arm and MPS arm, so the ratio isolates the device and nothing else. Our Mojo
numbers do not appear here at all.

WHAT WAS FOUND, 2026-08-20 (see bench/results/ for the recorded run)
-------------------------------------------------------------------
Most of the path does not exist on this hardware. Probed on sklearn 1.9.0 +
pytorch 2.13.0, MPS available:

    PCA(svd_solver="auto")             REFUSED  aten::_linalg_eigh
                                                unimplemented for MPS
    PCA(svd_solver="covariance_eigh")  REFUSED  same op
    PCA(svd_solver="randomized")       REFUSED  no LU under Array API, falls
                                                back to QR, asks for 149 GiB
    PCA(svd_solver="full")             RUNS, but torch warns that
                                       aten::linalg_svd is unsupported on MPS
                                       and silently runs it on the CPU
    Ridge(solver="cholesky")           REFUSED  Array API dispatch supports
                                                only svd/lsqr-class solvers
    Ridge(solver="auto")               RUNS, but silently becomes "svd"
    Ridge(solver="svd")                RUNS on mps
    LinearRegression                   REFUSED  no Array API support at all

Read the consequences carefully, because two of them protect our headline and
one of them does not:

1. scikit-learn's DEFAULT PCA cannot run on the Apple GPU. `auto` resolves to
   `covariance_eigh` at our shape and that op is missing on MPS.
2. `Ridge(alpha=0, solver="cholesky")` -- the algorithm-matched OLS
   denominator, the honest one -- cannot run on the Apple GPU either.
3. `LinearRegression` cannot, which is what makes the default-arm OLS number
   a claim about the only thing a user on this machine can actually run.
4. But `PCA(full)` and `Ridge(svd)` DO run, and they are the arms this file
   times. `PCA(full)` is a GPU arm in name only, since the decomposition
   itself falls back to the CPU; timing it is how that stops being a warning
   string and starts being a number.

REFUSALS ARE OUTPUT, NOT SILENCE. Every estimator above is attempted on
every run and the failures are printed with their exception. If a future
torch implements `linalg_eigh` on MPS, this file starts emitting an arm that
did not exist before, and the PCA comparison has to be re-read. A harness
that swallowed those exceptions would keep printing the old headline.

TIMING NOTES
------------
* **MPS is asynchronous.** `torch.mps.synchronize()` is called before every
  `perf_counter()` stop. Without it these arms would time dispatch, not work,
  and would look absurdly fast.
* **Transfer is outside the timed region**, matching `bench_main.mojo`, whose
  `enqueue_copy` calls all happen before its repeat loop. Both sides time
  compute on resident data. `bench_sklearn.py` documents this asymmetry as a
  caveat for one-shot `fit(numpy_array)` workloads; it applies here too.
* **float32 both arms.** MPS has no float64, and neither does Metal, so this
  is the only dtype the comparison can be made in.
* The data is `bench_sklearn.u01`, the same splitmix64 stream the Mojo
  benchmark uses, imported rather than copied so the two files cannot drift.
"""

import os

# BEFORE importing sklearn or scipy. `set_config(array_api_dispatch=True)`
# raises RuntimeError if scipy's own support was not enabled at import time,
# so setting this after the import is too late and fails with a message that
# does not obviously point here.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sklearn
from bench_sklearn import REPEATS, emit, u01
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

try:
    import torch
except ImportError:
    print("REFUSED all-arms torch-not-installed "
          "(pixi run -e skgpu ...)", file=sys.stderr)
    raise SystemExit(1)


def mps_ready():
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def time_cpu(fn, x, y=None):
    t = time.perf_counter()
    fn(x, y) if y is not None else fn(x)
    return time.perf_counter() - t


def time_mps(fn, x, y=None):
    """Time on MPS with an explicit sync, because the queue is async."""
    torch.mps.synchronize()
    t = time.perf_counter()
    fn(x, y) if y is not None else fn(x)
    torch.mps.synchronize()
    return time.perf_counter() - t


def attempt(name, thunk):
    """Run one arm. A refusal is reported, never swallowed."""
    try:
        return thunk()
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        msg = str(exc).replace("\n", " ")[:120]
        print(f"REFUSED {name} {type(exc).__name__}: {msg}")
        return None


def main():
    if not mps_ready():
        print("REFUSED all-arms mps-unavailable "
              f"built={torch.backends.mps.is_built()}")
        raise SystemExit(1)

    sklearn.set_config(array_api_dispatch=True)

    # Same shapes as bench_sklearn.py's pca/ols arms, same salts, so the
    # numbers sit beside that file's without a shape caveat.
    pca_rows, pca_cols, pca_comp = 4000000, 32, 8
    ols_rows, ols_cols = 4000000, 32

    pc_np = np.ascontiguousarray(u01(pca_rows, pca_cols, 3) * 4.0,
                                 dtype=np.float32)
    ol_np = np.ascontiguousarray(u01(ols_rows, ols_cols, 6) - 0.5,
                                 dtype=np.float32)
    ol_w = 1.0 + 0.1 * np.arange(ols_cols)
    ob_np = np.ascontiguousarray(ol_np @ ol_w, dtype=np.float32)

    # Resident before the timer starts, on both devices.
    pc_gpu = torch.from_numpy(pc_np).to("mps")
    ol_gpu = torch.from_numpy(ol_np).to("mps")
    ob_gpu = torch.from_numpy(ob_np).to("mps")
    torch.mps.synchronize()

    # THE THIRD ARM, AND THE REASON IT EXISTS.
    #
    # The first run of this file compared a NUMPY array against a TORCH tensor
    # on mps and read the 2.6x gap as the device. It is not: those two arms
    # differ in the library as well as the chip. A numpy input sends sklearn
    # through scipy and LAPACK; a torch input sends it through torch's own
    # kernels, and for `svd` torch has no MPS implementation and falls back to
    # its CPU one. So the "gpu" arm may be running the expensive step on the
    # CPU too, and the gap would then be torch's CPU LAPACK against scipy's.
    #
    # That is the identical confound this repository already refuses for OLS,
    # where `LinearRegression` (gelsd) versus our normal equations was an
    # ALGORITHM difference reported as a hardware one. Same trap, same fix:
    # hold everything constant but the thing being measured.
    #
    # Three arms, and each adjacent pair isolates one variable:
    #   *_cpu       numpy  -> scipy/LAPACK  on CPU
    #   *_torchcpu  torch  -> torch kernels on CPU   (library changed)
    #   *_gpu       torch  -> torch kernels on mps   (device changed)
    pc_tcpu = torch.from_numpy(pc_np)
    ol_tcpu = torch.from_numpy(ol_np)
    ob_tcpu = torch.from_numpy(ob_np)

    # The arms that the probe found DO reach the device, plus their CPU
    # counterparts so each ratio isolates the device.
    def pca_full(x, _=None):
        return PCA(n_components=pca_comp, svd_solver="full").fit(x)

    def ridge_svd(x, y):
        return Ridge(alpha=0.0, solver="svd", fit_intercept=False).fit(x, y)

    # The arms the probe found are REFUSED. Attempted every run anyway: if
    # torch ever implements them, this harness must notice rather than keep
    # printing a headline that assumed they were impossible.
    def pca_auto(x, _=None):
        return PCA(n_components=pca_comp).fit(x)

    def pca_cov_eigh(x, _=None):
        return PCA(n_components=pca_comp, svd_solver="covariance_eigh").fit(x)

    def ridge_cholesky(x, y):
        return Ridge(alpha=0.0, solver="cholesky",
                     fit_intercept=False).fit(x, y)

    def linreg(x, y):
        return LinearRegression(fit_intercept=False).fit(x, y)

    for rep in range(REPEATS):
        for nm, fn in (("pca_full", pca_full),):
            s = attempt(f"{nm}_cpu", lambda: time_cpu(fn, pc_np))
            if s is not None:
                emit(f"{nm}_cpu", s)
            s = attempt(f"{nm}_torchcpu", lambda: time_cpu(fn, pc_tcpu))
            if s is not None:
                emit(f"{nm}_torchcpu", s)
            s = attempt(f"{nm}_gpu", lambda: time_mps(fn, pc_gpu))
            if s is not None:
                emit(f"{nm}_gpu", s)

        for nm, fn in (("ols_svd", ridge_svd),):
            s = attempt(f"{nm}_cpu", lambda: time_cpu(fn, ol_np, ob_np))
            if s is not None:
                emit(f"{nm}_cpu", s)
            s = attempt(f"{nm}_torchcpu",
                        lambda: time_cpu(fn, ol_tcpu, ob_tcpu))
            if s is not None:
                emit(f"{nm}_torchcpu", s)
            s = attempt(f"{nm}_gpu", lambda: time_mps(fn, ol_gpu, ob_gpu))
            if s is not None:
                emit(f"{nm}_gpu", s)

        # Refusal probes. Only on the first rep: they raise, and raising
        # four times per rep buys nothing but noise.
        if rep == 0:
            attempt("pca_auto_gpu", lambda: time_mps(pca_auto, pc_gpu))
            attempt("pca_cov_eigh_gpu",
                    lambda: time_mps(pca_cov_eigh, pc_gpu))
            attempt("ols_normal_eq_gpu",
                    lambda: time_mps(ridge_cholesky, ol_gpu, ob_gpu))
            attempt("ols_default_gpu", lambda: time_mps(linreg, ol_gpu, ob_gpu))


if __name__ == "__main__":
    main()
