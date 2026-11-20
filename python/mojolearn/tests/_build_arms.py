# SPDX-License-Identifier: Apache-2.0
"""Test-side helpers for switches that are build defines, not environment reads.

A product route is chosen by a `-D MOJOLEARN_<NAME>` define at build time
(docs/plans/ENV_GATES_2026-11-15.md), so one installed wheel holds one arm.
A test of a non-default arm runs only against a wheel built with that arm,
and a comparison between two arms crosses two runs (two wheels):

    MOJOLEARN_TEST_BUILD_DEFINES="MOJOLEARN_A MOJOLEARN_B"
        the defines the installed wheel was built with (set by the controller
        that built it; empty = the default build)
    MOJOLEARN_TWO_BUILD_RECORD=<dir>      write this build's words per check
    MOJOLEARN_TWO_BUILD_REFERENCE=<dir>   compare them with the other build's

These variables are read by the TEST controller only. No product code reads
them; they never choose a route.
"""
import json
import os
from pathlib import Path

import pytest


def build_defines():
    """The `-D` names the installed wheel was built with, as the controller
    reported them."""
    return set(os.environ.get("MOJOLEARN_TEST_BUILD_DEFINES", "").split())


def built_with(define):
    return define in build_defines()


def requires_build(*defines):
    """Skip unless the installed wheel was built with every one of `defines`."""
    missing = [d for d in defines if not built_with(d)]
    return pytest.mark.skipif(bool(missing), reason="needs a wheel built with " + " ".join("-D " + d for d in missing))


def default_build_only(*defines):
    """Skip on a wheel built with any of `defines` (the check is the default arm's)."""
    present = [d for d in defines if built_with(d)]
    return pytest.mark.skipif(bool(present), reason="default-arm check; this wheel has " + " ".join("-D " + d for d in present))


def two_build_compare(name, words):
    """Record `words` (JSON-able) for this build under `name`, and compare
    them with the other build's record when a reference is given. Skips when
    neither directory is set, or after recording without a reference."""
    record = os.environ.get("MOJOLEARN_TWO_BUILD_RECORD", "")
    reference = os.environ.get("MOJOLEARN_TWO_BUILD_REFERENCE", "")
    if not record and not reference:
        pytest.skip("set MOJOLEARN_TWO_BUILD_RECORD / _REFERENCE to compare two builds")
    if record:
        Path(record).mkdir(parents=True, exist_ok=True)
        (Path(record) / (name + ".json")).write_text(json.dumps(words, sort_keys=True))
    if not reference:
        pytest.skip("recorded this build's words; run the other build with MOJOLEARN_TWO_BUILD_REFERENCE")
    want = json.loads((Path(reference) / (name + ".json")).read_text())
    assert words == want, name
