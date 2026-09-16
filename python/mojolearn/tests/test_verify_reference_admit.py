# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`_verify_reference.admit` and the DIRECTORY a column happens to sit in.

    cd python && python3 -m mojolearn.tests.test_verify_reference_admit

THE TRAP (found 2026-09-16, lane/sabotage-evidence). `admit` matched every
token of `_EXCLUDED_NAME_TOKENS` against the WHOLE lowercased path, so a
perfectly clean column was refused because of the name of a directory above
it. A negative control is recorded BESIDE the clean column it is a control
for, in one record directory, and naming that directory after the thing it
records is the obvious thing to do. Two committed clean columns were being
silently discarded for their neighbor's sin:

    2026-09-15_metrics-sabotage-coverage/cpu-prod.json
    2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json

A silent refusal is the bad failure here. The column is not reported as
rejected; it simply stops feeding the shipped reference table, and the next
person to put a clean column in a sensibly named directory loses it the same
way with no error to read.

WHY THE FIX IS NARROW. `partial`, `probe`, `unfixed` and `post-merge-smoke`
are MEANINGFUL as directory markers: `2026-09-14_kmeans-sqrt-fix/unfixed/`
holds columns taken with the bug still present, and admitting those would feed
known-wrong hashes into the table. Only `sabotage` needs to match the file
name alone, because a sabotage build ALSO says so in its own metadata
(`host.families[*].sabotage`, and the `*_sabotage` harness flags), which is
the stronger check and the one that still refuses it. Measured over the 495
committed columns: 220 admitted before, 222 after, nothing newly refused, and
no column carrying a sabotage signal admitted.

The last three tests are the blast-radius guard. They are the reason this fix
is narrow rather than "match the basename", which would have admitted eight
`unfixed/` and `probe/` columns.
"""
import sys

import pytest

from mojolearn import _verify_reference as vref

RECORDS = "bench/results/identity_break"


def clean_column():
    """The smallest column `admit` accepts: identical mode, a real commit, one
    device, default fixtures, no sabotage signal of any kind."""
    return dict(
        cells={"ols/base": {"verdict": "STABLE", "hashes": ["a" * 16]}},
        mode="identical",
        commit="1334beeedd92",
        vendor="cpu-amd-epyc-9754-128-core-processor",
        package={},
        host={"families": {"_mojolearn_estimators_host": {"column": "cpu", "sabotage": False}}},
    )


# ------------------------------------------------- the trap this test exists for

def test_clean_column_in_a_sabotage_named_directory_is_admitted():
    """FAILS before the fix. This is the whole point of the file."""
    path = f"{RECORDS}/2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json"
    assert vref.admit(clean_column(), path) is None, (
        "a clean column was refused for the name of a directory above it"
    )


def test_clean_column_in_a_sabotage_coverage_directory_is_admitted():
    path = f"{RECORDS}/2026-09-15_metrics-sabotage-coverage/cpu-prod.json"
    assert vref.admit(clean_column(), path) is None


# ------------------------------------------- a sabotage column is STILL refused

def test_sabotage_column_refused_by_its_own_file_name():
    path = f"{RECORDS}/2026-09-15_kmeans-transform/cpu-apple-m4.host-sabotage.json"
    assert vref.admit(clean_column(), path) is not None


def test_sabotage_column_refused_by_its_binding_read_back():
    """The metadata path, which does not care what anything is named. This is
    what `forest_host_sabotage()` was failing to report for the CTR arm."""
    j = clean_column()
    j["host"]["families"]["_mojolearn_forest_host"] = {"column": "cpu", "sabotage": True}
    assert vref.admit(j, f"{RECORDS}/2026-09-16_negative-controls/cpu.json") is not None


def test_sabotage_column_refused_by_its_harness_flag():
    j = clean_column()
    j["batch_sabotage"] = True
    assert vref.admit(j, f"{RECORDS}/2026-09-16_negative-controls/cpu.json") is not None


# --------------------------------------------------------- the blast-radius guard

def test_unfixed_directory_is_still_refused():
    """`unfixed/` holds columns taken with the bug present. Admitting them
    would feed known-wrong hashes into the shipped table."""
    path = f"{RECORDS}/2026-09-14_kmeans-sqrt-fix/unfixed/apple-m4.json"
    assert vref.admit(clean_column(), path) is not None


def test_probe_directory_is_still_refused():
    path = "bench/results/e1g/2026-09-14_142050-amd-probe/remote/mamba2_probe/identity_42.json"
    assert vref.admit(clean_column(), path) is not None


def test_partial_and_smoke_directories_are_still_refused():
    for token in ("partial", "post-merge-smoke"):
        path = f"{RECORDS}/2026-09-14_{token}-run/apple-m4.json"
        assert vref.admit(clean_column(), path) is not None, token


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
