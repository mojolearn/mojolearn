#!/usr/bin/env python3
"""The capture harness's runtime witness, against a synthetic witness.

WHY THIS FILE EXISTS. The capture runs only where a GPU is, so every check
inside it is first executed on a rented box. Two leases were spent on failures
of that class: a missing native symbol, and then this check refusing the run it
was asked to make. Neither needed a GPU to find. Anything in that harness that
is a decision about a dict rather than about arithmetic belongs in a function
and belongs here.

The regression this pins is narrow and worth stating plainly. `native_profile`
is the BINARY's identity and never varies with the shape, because one binary
runs every shape. The run's identity is `profile`. Comparing the first against
the run's shape passes at the default and refuses every other shape, which is
exactly the shape a second certificate needs.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f'_{name}', ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shape_module = _load('byte_lm_shape')
capture = _load('byte_lm_real_text_capture')

TWO = shape_module.Shape()
FOUR = shape_module.Shape([4, 32, 32, 4, 2, 8, 64, 2, 256])


def witness(shape, **overrides):
    """What the surface reports on a healthy box running `shape`."""
    base = dict(native_vendor='cuda',
                native_profile=shape_module.DEFAULT_PROFILE,
                profile=shape.profile,
                native_numeric_mode=1)
    base.update(overrides)
    return base


def test_a_healthy_witness_is_admitted_at_both_shapes():
    assert capture.admit_runtime(witness(TWO), 'cuda', TWO)
    assert capture.admit_runtime(witness(FOUR), 'cuda', FOUR)


def test_the_binary_identity_does_not_vary_with_the_shape():
    """The regression that cost a lease. At the four-row shape the binary still
    reports the default profile, and that must be admitted; a binary reporting
    the run's shape instead is the anomaly."""
    assert capture.admit_runtime(witness(FOUR), 'cuda', FOUR)
    assert not capture.admit_runtime(
        witness(FOUR, native_profile=FOUR.profile), 'cuda', FOUR)


def test_a_run_at_the_wrong_shape_is_refused():
    """A witness saying it ran the certified shape, while this capture believes
    it is writing a four-row tree, must not be admitted: the tree would carry
    one shape's bytes under another shape's name."""
    assert not capture.admit_runtime(witness(TWO), 'cuda', FOUR)
    assert not capture.admit_runtime(witness(FOUR), 'cuda', TWO)


@pytest.mark.parametrize('bad', [
    dict(native_vendor='hip'),
    dict(native_vendor='metal'),
    dict(native_vendor=None),
    dict(native_numeric_mode=0),
    dict(native_numeric_mode=None),
    dict(native_profile='mojolearn.byte-lm.something-else.fp32.v1'),
    dict(profile='mojolearn.byte-lm.something-else.fp32.v1'),
])
def test_a_spoiled_witness_is_refused(bad):
    assert not capture.admit_runtime(witness(FOUR, **bad), 'cuda', FOUR)


def test_a_witness_missing_a_field_is_refused_rather_than_defaulted():
    """An absent key must never read as agreement."""
    for key in ('native_vendor', 'native_profile', 'profile', 'native_numeric_mode'):
        spoiled = witness(FOUR)
        del spoiled[key]
        assert not capture.admit_runtime(spoiled, 'cuda', FOUR)
    assert not capture.admit_runtime({}, 'cuda', FOUR)
