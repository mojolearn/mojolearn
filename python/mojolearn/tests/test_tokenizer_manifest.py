# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tokenizer's place in the tree (the expose-tokenizer lane, 2026-09-14),
checked from SOURCE so it runs on a box with nothing built: the manifest
declares the family, the package exports the class, the alpha API document
names it, the build shim execs the one builder, the CPU identity gate builds
the binding it will read back, and the table lookup refuses by name before
any binding is loaded.

    cd python && python3 -m mojolearn.tests.test_tokenizer_manifest
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import mojolearn
from mojolearn import host_surface
from mojolearn import tokenizer as tokenizer_module

ROOT = Path(__file__).resolve().parents[3]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_family():
    f = host_surface.family("tokenizer")
    assert f["binding"] == "_mojolearn_tokenizer_host"
    assert f["routes"] is None, "the tokenizer has no GPU binding to route from"
    assert f["classes"] == ("GPT2Tokenizer",)
    assert f["sabotage_define"] == "MOJOLEARN_TOKENIZER_HOST_SABOTAGE"
    assert f["training_lanes"] == () and f["inference_lanes"] == ()
    assert "tokenizer/encoding.mojo" in f["host_modules"]
    assert "_mojolearn_tokenizer_host" not in host_surface.routed_modules().values()
    assert "_mojolearn_tokenizer_host" in host_surface.bindings()


def test_package_exports_the_class():
    assert "GPT2Tokenizer" in mojolearn.__all__
    assert "tokenizer" in mojolearn.__all__
    assert mojolearn.GPT2Tokenizer is tokenizer_module.GPT2Tokenizer
    assert "GPT2Tokenizer" in tokenizer_module.__all__


def test_alpha_api_names_it():
    text = _read("python/mojolearn/ALPHA_API.md")
    assert "mojolearn.tokenizer.GPT2Tokenizer" in text
    assert "_mojolearn_tokenizer_host" in text


def test_shim_execs_the_one_builder():
    lines = [l for l in _read("bindings/build_tokenizer_host.sh").splitlines() if l.strip() and not l.startswith("#")]
    assert lines == ['exec sh "$(dirname -- "$0")/build_host_family.sh" tokenizer "$@"']


def test_cpu_identity_gate_builds_the_binding():
    """The gate reads back EVERY declared binding, so a declared family the
    workflow does not build is a gate that fails on the first runner."""
    text = _read(".github/workflows/cpu-identity-gate.yml")
    # Since 2026-09-15 the gate builds every family the manifest declares in
    # one loop (`for family in $BUILD_FAMILIES`, from --families or
    # --wheel-families), and cpu_identity_gate_check.py build-list fails
    # the manifest step when that list leaves one out.
    assert 'sh "bindings/build_${family}_host.sh"' in text
    assert "--wheel-families" in text and "build-list" in text
    assert "tokenizer" in host_surface.wheel_families()


def test_binding_source_reads_the_sabotage_define():
    src = _read("bindings/_mojolearn_tokenizer_host.mojo")
    assert 'is_defined["MOJOLEARN_TOKENIZER_HOST_SABOTAGE"]' in src
    names = re.findall(r'module\.def_function\[[A-Za-z0-9_]+\]\("([A-Za-z0-9_]+)"\)', src)
    assert sorted(names) == sorted(host_surface.family("tokenizer")["exports"])


def test_data_dir_resolves_the_checkout_tables():
    d = tokenizer_module.data_dir()
    assert os.path.isfile(os.path.join(d, "gpt2_ranks.tsv"))
    assert os.path.isfile(os.path.join(d, "unicode_categories.tsv"))


def test_missing_tables_refused_by_name_before_any_load():
    empty = tempfile.mkdtemp()
    try:
        mojolearn.GPT2Tokenizer(data_directory=empty)
    except FileNotFoundError as exc:
        assert "gpt2_ranks.tsv does not exist" in str(exc)
    else:
        raise AssertionError("an empty data directory was not refused")
    saved = os.environ.get("MOJOLEARN_TOKENIZER_DATA")
    os.environ["MOJOLEARN_TOKENIZER_DATA"] = empty
    try:
        # The override comes first; the checkout's tables come after, so
        # data_dir still resolves. What must show is the override in the
        # search list of the refusal for a directory tree with no tables.
        looked = [d for _, d in tokenizer_module._candidates()]
        assert looked[0] == os.path.abspath(empty)
    finally:
        if saved is None:
            del os.environ["MOJOLEARN_TOKENIZER_DATA"]
        else:
            os.environ["MOJOLEARN_TOKENIZER_DATA"] = saved


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
        print(f"test_tokenizer_manifest: RED. {len(failures)} of {len(TESTS)} checks failed.")
        return 1
    print(f"test_tokenizer_manifest: GREEN. {len(TESTS)} source checks; no binding was loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
