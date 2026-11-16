# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E, CPU training for the pca, tsvd, ols, ridge and dbscan
host fits, run only where the estimators host binding is built and the
package took the CPU-only path (skipped, and SAID to be skipped, otherwise):
the five entries run through the host binding on a small draw, twice, and
return the same bytes; the binding reads back sabotage False; an oversized
n_components is refused by name. It is a plumbing check. The bit claim
against the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def test_host_fits_run_and_repeat_on_a_cpu_only_install():
    """Runtime, only where it can run. Skipping is stated, never silent."""
    if _backend._CPU_ONLY is None:
        print("  skip: a GPU set loaded; the host fits are served on a CPU-only install only")
        return
    if "_mojolearn_estimators_host" not in _backend.host_families_built():
        print("  skip: _mojolearn_estimators_host is not built under", _backend.host_dir())
        return
    try:
        import numpy as np
    except ImportError:
        print("  skip: numpy is not installed")
        return
    m = _backend.load_host_module("_mojolearn_estimators_host")
    assert bool(m.estimators_host_sabotage()) is False
    rng = np.random.default_rng(7)
    n, d, k = 512, 5, 2
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    def addr(a):
        return a.ctypes.data

    def run():
        comp = np.empty((k, d), np.float32)
        mu = np.empty(d, np.float32)
        ev = np.empty(k, np.float32)
        ratio = np.empty(k, np.float32)
        sv = np.empty(k, np.float32)
        noise = float(m.pca_fit(addr(x), addr(comp), addr(mu), addr(ev), addr(ratio), addr(sv), [n, d, k]))
        tcomp = np.empty((k, d), np.float32)
        tsv = np.empty(k, np.float32)
        m.tsvd_fit(addr(x), addr(tcomp), addr(tsv), [n, d, k])
        w = np.empty(d, np.float32)
        m.ols_fit(addr(x), addr(y), addr(w), [n, d])
        r = np.empty(d, np.float32)
        m.ridge_fit(addr(x), addr(y), addr(r), [n, d, 0.5])
        labels = np.empty(n, np.int32)
        passes = int(m.dbscan_fit(addr(x), addr(labels), 0, [n, d, 0.9, 5, 0, 0, 1, 0]))
        assert passes >= 1 and labels.min() >= -1
        return comp.tobytes() + mu.tobytes() + ev.tobytes() + ratio.tobytes() + sv.tobytes() + \
            repr(noise).encode() + tcomp.tobytes() + tsv.tobytes() + w.tobytes() + r.tobytes() + \
            labels.tobytes()

    first, second = run(), run()
    assert first == second, "a host fit returned different bytes on two runs"
    assert np.isfinite(np.frombuffer(first[: k * d * 4], np.float32)).all()
    # The shape refusals are pca_validate's and reach the caller by name.
    try:
        m.pca_fit(addr(x), addr(np.empty((k, d), np.float32)), addr(np.empty(d, np.float32)),
                  addr(np.empty(k, np.float32)), addr(np.empty(k, np.float32)),
                  addr(np.empty(k, np.float32)), [n, d, d + 1])
    except Exception as exc:  # noqa: BLE001
        assert "n_components cannot exceed n_cols" in str(exc), str(exc)
    else:
        raise AssertionError("pca_fit accepted n_components > n_cols")


TESTS = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main(argv=None):
    failures = []
    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    for name, why in failures:
        print(f"FAIL {name}: {why}")
    if failures:
        print(f"test_cpu_training_e: RED. {len(failures)} of {len(TESTS)} checks failed.")
        return 1
    print(f"test_cpu_training_e: GREEN. {len(TESTS)} checks (vendor {mojolearn.vendor()}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
