# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream D's place in the tree (2026-09-14), checked from SOURCE so it
runs on a box with nothing built: the six new bindings (the four workstream
D families, then IVF and Embedding when they left `_NOT_YET`) are registered
in both places `_backend` needs (DEVIATION 869), the five packaging lists
agree, every binding source exports exactly the names its Python half
calls, the build scripts exist and name their binding, the classes are
exported, the alpha API and the support matrix name them, the lane bodies
parse, `_NOT_YET` is empty, and the embedding sabotage script names every
arm `embedding_identical.mojo` defines.

    cd python && python3 -m mojolearn.tests.test_expose_d_manifest
"""
import ast
import os
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend

ROOT = Path(__file__).resolve().parents[3]

NEW = {
    "_mojolearn_kernel_methods": ("build_kernel_methods.sh", "python/mojolearn/kernel_methods.py"),
    "_mojolearn_mixture": ("build_mixture.sh", "python/mojolearn/mixture.py"),
    "_mojolearn_hdbscan": ("build_hdbscan.sh", "python/mojolearn/hdbscan.py"),
    "_mojolearn_resample": ("build_resample.sh", "python/mojolearn/resample.py"),
    "_mojolearn_ivf": ("build_ivf.sh", "python/mojolearn/_ivf_impl.py"),
    "_mojolearn_embedding": ("build_embedding.sh", "python/mojolearn/embedding.py"),
}
#: the availability flag each workstream D binding exports, as
#: python/mojolearn/_parallel_worker.py checks it before its driver runs
PARALLEL_FLAGS = {
    "kernel_methods": "kernel_methods_rows_parallel_available",
    "mixture": "gmm_parallel_available",
    "hdbscan": "hdbscan_rows_parallel_available",
    "resample": "resample_ranges_parallel_available",
    "ivf": None,
    "embedding": None,
}
CLASSES = ("Cholesky", "KernelRidge", "Nystroem", "RBFSampler", "GaussianMixture", "HDBSCAN", "IVFIndex", "Embedding")


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _exports(binding_src):
    return set(re.findall(r'm\.def_function\[\w+\]\(\s*"(\w+)"', _read(binding_src)))


def _calls(py_src, prefix_re):
    return set(re.findall(prefix_re, _read(py_src)))


def test_backend_registers_both_places():
    for name, (script, _) in NEW.items():
        assert name in _backend._MODULES, name
        assert _backend._build_script(name) == script, (name, _backend._build_script(name))
        assert name in _backend._IDENTICAL_ONLY, name + " must be identical only"
    assert mojolearn._NOT_YET == {}, sorted(mojolearn._NOT_YET)


def test_packaging_lists_agree():
    sys.path.insert(0, str(ROOT / "packaging"))
    import check_ext_lists as c
    want = set(_backend._MODULES) - {"_mojolearn_byte_lm"}
    for path, how, var, ident in c.SOURCES:
        got = how(path, var)
        if ident:
            got = got | how(path, ident)
        assert got == want, (path, sorted(want ^ got))
    smoke = _read("packaging/linux/smoke.py")
    for name in NEW:
        assert name in smoke, name


def test_binding_exports_match_python_calls():
    pairs = {
        "bindings/_mojolearn_kernel_methods.mojo": ("python/mojolearn/kernel_methods.py", r"_extension\(\)\.(\w+)\("),
        "bindings/_mojolearn_mixture.mojo": ("python/mojolearn/mixture.py", r"_extension\(\)\.(\w+)\("),
        "bindings/_mojolearn_hdbscan.mojo": ("python/mojolearn/hdbscan.py", r"_extension\(\)\.(\w+)\("),
        "bindings/_mojolearn_resample.mojo": ("python/mojolearn/resample.py", r"_extension\(numeric_mode\)\.(\w+)\("),
        "bindings/_mojolearn_ivf.mojo": ("python/mojolearn/_ivf_impl.py", r'_entry\(self\._extension\(\), "(\w+)"\)\('),
        "bindings/_mojolearn_embedding.mojo": ("python/mojolearn/embedding.py", r"_extension\(\)\.(\w+)\("),
    }
    for binding, (py, pat) in pairs.items():
        exported, called = _exports(binding), _calls(py, pat)
        assert called, (py, "no calls found")
        assert called <= exported, (binding, sorted(called - exported))
        stem = binding.rsplit("/", 1)[1][len("_mojolearn_"):-len(".mojo")]
        for suffix in ("_vendor", "_numeric_mode"):
            assert stem + suffix in exported, (binding, stem + suffix)
        # A multi-GPU availability flag must name a driver that reads it. The
        # generic `<stem>_parallel_available` names returned 1 with no driver
        # behind them and were removed on 2026-09-15; the flags the ordered
        # drivers check are the ones below, and IVF and Embedding have none.
        assert stem + "_parallel_available" not in exported, (binding, "stale flag", stem + "_parallel_available")
        flag = PARALLEL_FLAGS[stem]
        if flag is None:
            assert not any(e.endswith("_parallel_available") for e in exported), (binding, sorted(exported))
        else:
            assert flag in exported, (binding, flag)
            assert f"'{flag}'" in _read("python/mojolearn/_parallel_worker.py"), (flag, "no driver reads it")
    gp = _exports("bindings/_mojolearn_gp.mojo")
    chol = _calls("python/mojolearn/_cholesky_impl.py", r"_extension\(\)\.(\w+)\(")
    assert chol and chol <= gp, sorted(chol - gp)


def test_build_scripts_exist_and_name_their_binding():
    for name, (script, _) in NEW.items():
        p = ROOT / "bindings" / script
        assert p.exists() and os.access(p, os.X_OK), script
        text = p.read_text()
        assert f"bindings/{name}.mojo" in text and f"{name}.so" in text, script
        assert "MOJOLEARN_NUMERIC_MODE=identical only" in text or "identical only" in text, script


def test_classes_exported():
    for name in CLASSES:
        assert name in mojolearn.__all__, name
        assert hasattr(mojolearn, name), name
    for mod in ("kernel_methods", "mixture", "hdbscan", "resample", "embedding"):
        assert mod in mojolearn.__all__, mod
    assert "Cholesky" in mojolearn.linalg.__all__
    for fn in ("embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward", "linear_forward", "linear_backward"):
        assert fn in mojolearn.training.__all__, fn


def test_docs_name_the_surfaces():
    alpha = _read("python/mojolearn/ALPHA_API.md")
    matrix = _read("SUPPORT_MATRIX.md")
    for name in CLASSES + ("bootstrap", "permutation_test", "monte_carlo_integrate", "oversampling_factor", "embedding_forward"):
        assert name in alpha, ("ALPHA_API.md", name)
        assert name in matrix, ("SUPPORT_MATRIX.md", name)
    assert "faster" not in matrix.split("Workstream D")[1][:6000].lower(), "no speed word in the workstream D rows"


def test_every_staged_lane_body_reached_the_harness():
    """The eight `docs/lanes/LANE_BODY_*.py` staging files are gone (2026-09-19),
    and this is what replaced the test that parsed them.

    They were lane definitions held outside `tools/identity_break.py` while
    they were written. Every one had already been merged -- checked lane by
    lane before the delete -- so the files were duplicates of harness code,
    and a test asserting a duplicate still parses was checking the copy, not
    the thing that runs. What is worth holding is the property that made
    deleting them safe: each staged lane is IN the harness.
    """
    src = _read("tools/identity_break.py")
    for lane in ("par-cholesky", "par-kernel-ridge", "par-gmm", "par-hdbscan",
                 "par-resample", "par-ivf", "par-scaler", "par-kmeans"):
        assert f'@lane("{lane}")' in src, (
            f"{lane} was staged in a LANE_BODY_ file that has been deleted and is not in the harness")


def test_embedding_sabotage_script_names_every_arm():
    src = _read("embedding/checks/embedding_identical.mojo")
    arms = set(re.findall(r'is_defined\[\s*"MOJOLEARN_EMB_SABOTAGE_(\w+)"', src))
    assert len(arms) >= 16, sorted(arms)
    script = _read("tools/embedding_sabotage_arm.sh")
    listed = set(re.search(r'ALL_ARMS="([^"]*)"', script).group(1).split())
    assert listed == arms, sorted(listed ^ arms)
    assert os.access(ROOT / "tools/embedding_sabotage_arm.sh", os.X_OK)
    pixi = _read("pixi.toml")
    assert "check-embedding-sabotage" in pixi and "check-embedding =" in pixi


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
    print(f"test_expose_d_manifest: GREEN. {len(TESTS)} source checks; no binding was loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
