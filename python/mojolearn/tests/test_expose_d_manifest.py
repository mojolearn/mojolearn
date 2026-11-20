# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream D's public surface: the kernel-methods, mixture, HDBSCAN,
resample, IVF and Embedding classes and modules are exported by the package
and the training functions by `mojolearn.training`.

    cd python && python3 -m mojolearn.tests.test_expose_d_manifest
"""
import sys

import mojolearn

CLASSES = ("Cholesky", "KernelRidge", "Nystroem", "RBFSampler", "GaussianMixture", "HDBSCAN", "IVFIndex", "Embedding")


def test_classes_exported():
    for name in CLASSES:
        assert name in mojolearn.__all__, name
        assert hasattr(mojolearn, name), name
    for mod in ("kernel_methods", "mixture", "hdbscan", "resample", "embedding"):
        assert mod in mojolearn.__all__, mod
    assert "Cholesky" in mojolearn.linalg.__all__
    for fn in ("embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward", "linear_forward", "linear_backward"):
        assert fn in mojolearn.training.__all__, fn


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
        print(f"test_expose_d_manifest: RED. {len(failures)} of {len(TESTS)} checks failed.")
        return 1
    print(f"test_expose_d_manifest: GREEN. {len(TESTS)} checks; no binding was loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
