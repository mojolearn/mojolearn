# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream D's place in the tree (2026-09-14), checked from SOURCE so it
runs on a box with nothing built: the four new bindings are registered in
both places `_backend` needs (DEVIATION 869), the five packaging lists
agree, every binding source exports exactly the names its Python half
calls, the build scripts exist and name their binding, the classes are
exported, the alpha API and the support matrix name them, the lane bodies
parse, IVF stays unexposed, and the embedding sabotage script names every
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
}
CLASSES = ("Cholesky", "KernelRidge", "Nystroem", "RBFSampler", "GaussianMixture", "HDBSCAN")
LANE_BODIES = ("cholesky", "kernel_methods", "mixture", "hdbscan", "resample", "ivf", "training_primitives", "kmeans")


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
    assert "_mojolearn_ivf" not in _backend._MODULES, "IVF is prepared, not exposed"


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
        "bindings/_mojolearn_ivf.mojo": ("python/mojolearn/_ivf_impl.py", r"_extension\(\)\.(\w+)\("),
    }
    for binding, (py, pat) in pairs.items():
        exported, called = _exports(binding), _calls(py, pat)
        assert called, (py, "no calls found")
        assert called <= exported, (binding, sorted(called - exported))
        stem = binding.rsplit("/", 1)[1][len("_mojolearn_"):-len(".mojo")]
        for suffix in ("_vendor", "_numeric_mode", "_parallel_available"):
            assert stem + suffix in exported, (binding, stem + suffix)
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
    assert (ROOT / "bindings/build_ivf.sh").exists()


def test_classes_exported():
    for name in CLASSES:
        assert name in mojolearn.__all__, name
        assert hasattr(mojolearn, name), name
    for mod in ("kernel_methods", "mixture", "hdbscan", "resample"):
        assert mod in mojolearn.__all__, mod
    assert "Cholesky" in mojolearn.linalg.__all__
    for fn in ("embedding_forward", "embedding_backward", "rms_norm_forward", "rms_norm_backward", "linear_forward", "linear_backward"):
        assert fn in mojolearn.training.__all__, fn
    assert "IVFFlat" not in mojolearn.__all__


def test_docs_name_the_surfaces():
    alpha = _read("python/mojolearn/ALPHA_API.md")
    matrix = _read("SUPPORT_MATRIX.md")
    for name in CLASSES + ("bootstrap", "permutation_test", "monte_carlo_integrate", "oversampling_factor", "embedding_forward"):
        assert name in alpha, ("ALPHA_API.md", name)
        assert name in matrix, ("SUPPORT_MATRIX.md", name)
    assert "faster" not in matrix.split("Workstream D")[1][:6000].lower(), "no speed word in the workstream D rows"


def test_lane_bodies_parse_and_name_lanes():
    for stem in LANE_BODIES:
        p = ROOT / "docs/lanes" / f"LANE_BODY_{stem}.py"
        assert p.exists(), p
        tree = ast.parse(p.read_text(), filename=str(p))
        lanes = [d.args[0].value for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 for d in node.decorator_list if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "lane"]
        assert lanes, (p, "no @lane")
        assert all("Xh=None" in ast.get_source_segment(p.read_text(), node) for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.decorator_list), (p, "the harness signature is (ml, X, yc, yr, Xh=None)")


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
