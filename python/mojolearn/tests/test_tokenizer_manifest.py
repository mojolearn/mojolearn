# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tokenizer's place in the tree (the expose-tokenizer lane, 2026-09-14),
checked from SOURCE so it runs on a box with nothing built: the manifest
declares the family, the package exports the class, the alpha API document
names it, the build shim execs the one builder, the CPU identity gate builds
the binding it will read back, no vocabulary or table file is tracked, and a
missing vocabulary is refused by name before any binding is loaded.

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
    # A covered TRAINING lane since lane/cpu-verifier-gaps-7 (11c5f2192,
    # 2026-09-15): the gate builds this family into its sabotage host set with
    # the family's own define, which reverses the ids of gpt2_encode and of
    # every document of gpt2_encode_batch, so the lane's train, infer and batch
    # parts all move. It was () when this test was written (2026-09-14), before
    # that gate existed. No INFERENCE lane, because the family has no saved
    # model to predict from.
    assert f["training_lanes"] == ("tokenizer",) and f["inference_lanes"] == ()
    assert "tokenizer" in host_surface.covered_lanes()
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


def test_no_vocabulary_is_tracked_or_shipped():
    """mojolearn ships no vocabulary and tracks no tokenizer data file
    (2026-09-15): the tables are gone from the tree, the Unicode classes are
    generated at build time, and the generated module is ignored."""
    assert not (ROOT / "tokenizer" / "data").exists()
    assert not (ROOT / "tokenizer" / "checks" / "fixtures").exists()
    assert "tokenizer/impl/unicode_table_generated.mojo" in _read(".gitignore")
    assert "sh tokenizer/tools/gen_unicode_table.sh" in _read("bindings/build_host_family.sh")
    pin = _read("tokenizer/impl/unicode_class.mojo")
    assert re.search(r'^comptime UNICODE_TABLE_SHA256_PINNED = "[0-9a-f]{64}"$', pin, re.M)


def test_missing_vocabulary_refused_by_name_before_any_load():
    try:
        mojolearn.GPT2Tokenizer()
    except ValueError as exc:
        assert "needs a vocabulary, and mojolearn ships none" in str(exc)
    else:
        raise AssertionError("a tokenizer with no vocabulary was not refused")
    missing = os.path.join(tempfile.mkdtemp(), "ranks.tsv")
    try:
        mojolearn.GPT2Tokenizer.from_ranks_file(missing)
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("a missing rank file was not refused")


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
