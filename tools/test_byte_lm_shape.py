#!/usr/bin/env python3
"""The shape helper against the literals the certified run was admitted with.

DEVIATION 2682. The helper replaced five sets of shape literals with one
derivation, and a derivation is only safe here if it provably reproduces what
the b2-l32 capture was produced and gated against. These tests are that proof,
and they are deliberately written as literals rather than as expressions, so a
change to the derivation cannot make its own test agree with it.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f'_{name}', ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shape_module = _load('byte_lm_shape')
Shape = shape_module.Shape


def test_the_default_reproduces_the_certified_literals():
    """Every one of these is copied from what the capture harness and the
    corpus manifest said before the helper existed."""
    s = Shape()
    assert s.profile == 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
    assert s.schedule == ('step s zero-based: row b reads bytes[(s*64+b*32) % 65504 '
                          ': start+33]; targets shifted one byte')
    assert s.validation_starts == [65536, 65600, 65664, 65728, 65792, 65856, 65920, 65984]
    assert s.n_total == 34944
    assert s.n_ids == 66
    assert s.n_tensors == 20
    assert s.validation_batches == 8
    assert s.counts() == {
        'initial_p': 34944, 'initial_m': 34944, 'initial_v': 34944,
        'post_p': 34944, 'post_m': 34944, 'post_v': 34944, 'grad': 34944,
        'initial_flags': 20, 'post_flags': 20, 'loss': 1, 'ids': 66}


def test_the_default_registry_matches_the_capture_harness_table():
    """The 20 tensors, their shapes and their offsets, as the harness spelled
    them out and as every retained capture.json records them."""
    entries = Shape().registry()
    assert [e['name'] for e in entries] == [
        'embed',
        'block0.norm1_w', 'block0.w_q', 'block0.w_k', 'block0.w_v', 'block0.w_o',
        'block0.norm2_w', 'block0.w_gate', 'block0.w_up', 'block0.w_down',
        'block1.norm1_w', 'block1.w_q', 'block1.w_k', 'block1.w_v', 'block1.w_o',
        'block1.norm2_w', 'block1.w_gate', 'block1.w_up', 'block1.w_down',
        'lm_head']
    assert entries[0] == dict(name='embed', shape=[256, 32], offset=0, count=8192)
    assert entries[2] == dict(name='block0.w_q', shape=[32, 32], offset=8224, count=1024)
    assert entries[-1]['offset'] + entries[-1]['count'] == 34944
    # The control localized its first wrong element to block0.w_q index 0, which
    # is flat element 8224. If that arithmetic ever moves, that diagnosis moves.
    assert entries[2]['offset'] == 8224


def test_the_default_train_schedule_reads_the_same_bytes():
    s = Shape()
    assert s.tokens_per_step == 64
    assert [s.train_start(0, b) for b in range(2)] == [0, 32]
    assert [s.train_start(1, b) for b in range(2)] == [64, 96]
    assert [s.train_start(127, b) for b in range(2)] == [8128, 8160]


def test_a_capture_without_a_recorded_shape_is_the_default():
    """Every retained b2-l32 tree predates the model_shape key, so reading one
    must give the default rather than a refusal."""
    assert shape_module.from_capture({'profile': Shape().profile}) == Shape()
    assert shape_module.from_capture({}) == Shape()


def test_a_recorded_shape_is_read_back_and_must_agree_with_its_profile():
    four = Shape([4, 32, 32, 4, 2, 8, 64, 2, 256])
    config = dict(profile=four.profile, model_shape=four.to_json())
    assert shape_module.from_capture(config) == four
    with pytest.raises(ValueError, match='disagrees'):
        shape_module.from_capture(dict(profile=Shape().profile, model_shape=four.to_json()))


def test_the_four_row_shape_derives_what_its_manifest_commits():
    """The committed manifest and the derivation must agree, or the capture
    harness refuses at the door and the disagreement is found on a rented box."""
    four = Shape([4, 32, 32, 4, 2, 8, 64, 2, 256])
    manifest = json.loads((ROOT / 'training/corpus/tinyshakespeare/manifest-b4-l32.json').read_text())
    fields = four.manifest_fields(corpus_sha=manifest['sha256'], corpus_bytes=manifest['bytes'])
    for key, value in fields.items():
        assert manifest[key] == value, key
    assert four.manifest_name == 'manifest-b4-l32.json'
    assert four.profile.startswith('mojolearn.byte-lm.b4-l32-')
    # Same parameters, twice the tokens. That is the whole point of the second
    # shape: the arrays are the same size and the sums behind them are not.
    assert four.n_total == Shape().n_total == 34944
    assert four.tokens_per_step == 2 * Shape().tokens_per_step
    assert four.n_ids == 132


def test_both_shapes_hold_out_the_same_number_of_target_bytes():
    two, four = Shape(), Shape([4, 32, 32, 4, 2, 8, 64, 2, 256])
    for s in (two, four):
        assert s.validation_batches * s.tokens_per_step == shape_module.VALIDATION_TARGETS
    assert four.validation_starts == [65536, 65664, 65792, 65920]
    # The last byte either schedule reads, which has to stay inside the pinned
    # validation range or the two are not reading the same text.
    def last_byte(s):
        return s.validation_starts[-1] + (s.batch - 1) * s.length + s.length + 1
    assert last_byte(two) == last_byte(four) == 66049
    assert last_byte(four) <= shape_module.VALIDATION_RANGE[1]


@pytest.mark.parametrize('bad', [
    [2, 32, 32, 4, 2, 9, 64, 2, 256],     # odd head dim
    [2, 32, 31, 4, 2, 8, 64, 2, 256],     # d_model is not heads times head dim
    [2, 32, 32, 4, 3, 8, 64, 2, 256],     # heads not divisible by kv
    [2, 9000, 32, 4, 2, 8, 64, 2, 256],   # length above the admitted bound
    [0, 32, 32, 4, 2, 8, 64, 2, 256],     # a zero dimension
    [True, 32, 32, 4, 2, 8, 64, 2, 256],  # a bool is not a dimension
])
def test_illegal_shapes_are_refused(bad):
    with pytest.raises(ValueError):
        Shape(bad)


def test_parse_accepts_the_spellings_the_tools_pass():
    assert shape_module.parse(None) == Shape()
    assert shape_module.parse('default') == Shape()
    assert shape_module.parse('4,32') == Shape([4, 32, 32, 4, 2, 8, 64, 2, 256])
    assert shape_module.parse('2,32,32,4,2,8,64,2,256') == Shape()
    for bad in ('', 'four,32', '-4,32', '4,'):
        with pytest.raises(ValueError):
            shape_module.parse(bad)


def test_a_shape_that_cannot_hold_out_evenly_is_refused_rather_than_rounded():
    """512 targets must divide into whole batches, or two shapes' held-out
    losses would be means over different amounts of text."""
    odd = Shape([3, 32, 32, 4, 2, 8, 64, 2, 256])
    with pytest.raises(ValueError, match='evenly'):
        odd.validation_starts
