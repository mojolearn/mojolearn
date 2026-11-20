# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tokenizer's public surface: the package exports the class, the old
name is a deprecated alias, the installed package ships no vocabulary, and a
missing vocabulary is refused by name before any binding is loaded.

    cd python && python3 -m mojolearn.tests.test_tokenizer_manifest
"""
import os
import sys
import tempfile

import mojolearn
from mojolearn import tokenizer as tokenizer_module


def test_package_exports_the_class():
    assert "BpeTokenizer" in mojolearn.__all__
    assert "tokenizer" in mojolearn.__all__
    assert mojolearn.BpeTokenizer is tokenizer_module.BpeTokenizer
    assert "BpeTokenizer" in tokenizer_module.__all__


def test_gpt2_tokenizer_is_a_deprecated_alias():
    """Renamed 2026-09-18: `GPT2Tokenizer` shipped in 0.8.x and stays in
    mojolearn.__all__ as the SAME class, with a DeprecationWarning."""
    import warnings
    assert "GPT2Tokenizer" in mojolearn.__all__
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        a = mojolearn.GPT2Tokenizer
        b = tokenizer_module.GPT2Tokenizer
        from mojolearn import GPT2Tokenizer as c
    assert a is b is c is tokenizer_module.BpeTokenizer
    msgs = [str(w.message) for w in seen if issubclass(w.category, DeprecationWarning)]
    assert len(msgs) >= 3 and all("renamed BpeTokenizer" in m for m in msgs), msgs


#: File names a byte-pair vocabulary or rank table ships under.
_VOCAB_NAMES = ("vocab.json", "merges.txt", "tokenizer.json", "ranks.tsv", "encoder.json", "vocab.bpe")


def test_installed_package_ships_no_vocabulary():
    """mojolearn ships no vocabulary (2026-09-15): the installed
    distribution's file list holds no rank table or vocabulary file; the
    caller supplies one."""
    from importlib.metadata import PackageNotFoundError, files
    try:
        listed = files("mojolearn")
    except PackageNotFoundError:
        listed = None
    if not listed:
        print("SKIP: mojolearn is not installed as a distribution here")
        return
    shipped = [str(f) for f in listed
               if f.name in _VOCAB_NAMES or f.suffix in (".tiktoken", ".bpe")]
    assert shipped == [], shipped


def test_missing_vocabulary_refused_by_name_before_any_load():
    try:
        mojolearn.BpeTokenizer()
    except ValueError as exc:
        assert "needs a vocabulary; mojolearn ships none" in str(exc)
    else:
        raise AssertionError("a tokenizer with no vocabulary was not refused")
    missing = os.path.join(tempfile.mkdtemp(), "ranks.tsv")
    try:
        mojolearn.BpeTokenizer.from_ranks_file(missing)
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
    print(f"test_tokenizer_manifest: GREEN. {len(TESTS)} checks; no binding was loaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
