# SPDX-License-Identifier: Apache-2.0
"""tools/experiment_guards.py: the per-family guard layout and the permanent-record check of the split."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_guards as eg  # noqa: E402

HEAD = 'from std.sys.compile import is_defined\n\n\ndef _check_configuration() -> Bool:\n'
TAIL = '    return True\n\n\ncomptime {c} = _check_configuration()\n'


def module(body, const='SIX_LANE_CONFIGURATION_OK'):
    return HEAD + ''.join('    ' + l + '\n' for l in body) + TAIL.format(c=const)


BEFORE = module(['# Lane x (2026-10-08): header of a batch',
                 'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A"',
                 'comptime assert not is_defined["MOJOLEARN_B"](), "removed: B"',
                 '# TOMBSTONE: MOJOLEARN_C (LOSER) deleted 2026-10-09 by abc; code recoverable at abc^.',
                 'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"'])


def test_header_carried_to_two_modules_is_a_faithful_move():
    after = {eg.SHARED: module(['# Lane x (2026-10-08): header of a batch',
                                'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A"']),
             'core/experiment_guards_neural.mojo': module(
                 ['# Lane x (2026-10-08): header of a batch',
                  'comptime assert not is_defined["MOJOLEARN_B"](), "removed: B"',
                  '# TOMBSTONE: MOJOLEARN_C (LOSER) deleted 2026-10-09 by abc; code recoverable at abc^.',
                  'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"',
                  'comptime assert not is_defined["MOJOLEARN_NEW"](), "added after the split: not judged"'],
                 'NEURAL_GUARDS_OK')}
    assert eg.record_problems(BEFORE, after) == []


@pytest.mark.parametrize('body, why', [
    (['# Lane x (2026-10-08): header of a batch', 'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A"',
      '# TOMBSTONE: MOJOLEARN_C (LOSER) deleted 2026-10-09 by abc; code recoverable at abc^.',
      'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"'], 'dropped'),
    (['# Lane x (2026-10-08): header of a batch', 'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A"',
      'comptime assert not is_defined["MOJOLEARN_B"](), "removed: B"',
      'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"'], 'comment line dropped'),
    (['# Lane x (2026-10-08): header of a batch', 'comptime assert not is_defined["MOJOLEARN_B"](), "removed: B"',
      'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A"',
      '# TOMBSTONE: MOJOLEARN_C (LOSER) deleted 2026-10-09 by abc; code recoverable at abc^.',
      'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"'], 'out of its original order'),
    (['# Lane x (2026-10-08): header of a batch', 'comptime assert not is_defined["MOJOLEARN_A"](), "removed: A (reworded)"',
      'comptime assert not is_defined["MOJOLEARN_B"](), "removed: B"',
      '# TOMBSTONE: MOJOLEARN_C (LOSER) deleted 2026-10-09 by abc; code recoverable at abc^.',
      'comptime assert not is_defined["MOJOLEARN_C"](), "removed: C"'], 'dropped'),
])
def test_record_problems_catch_drops_reorders_and_rewrites(body, why):
    problems = eg.record_problems(BEFORE, {eg.SHARED: module(body)})
    assert any(why in p for p in problems), problems


def test_duplicated_code_line_is_refused():
    body = eg.body_lines(BEFORE)
    after = {eg.SHARED: BEFORE, 'core/experiment_guards_trees.mojo': module([body[1].strip()], 'TREES_GUARDS_OK')}
    assert any('kept 2 times' in p for p in eg.record_problems(BEFORE, after))


def test_is_guard_file_and_family_of():
    assert eg.is_guard_file(eg.SHARED) and eg.family_of(eg.SHARED) == 'shared'
    assert eg.is_guard_file('core/experiment_guards_neural.mojo')
    assert eg.family_of('core/experiment_guards_neural.mojo') == 'neural'
    assert not eg.is_guard_file('core/experiment_guards_x/y.mojo')
    assert not eg.is_guard_file('core/device_plan.mojo') and eg.family_of('core/device_plan.mojo') is None


def test_layout_problems_on_a_synthetic_tree(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'bindings').mkdir()
    (tmp_path / eg.SHARED).write_text(module(['comptime assert True, "x"']))
    (tmp_path / 'core/experiment_guards_neural.mojo').write_text(
        'from core.device_plan import X\n' + module(['comptime assert True, "x"'], 'NEURAL_GUARDS_OK'))
    good = ('from core.experiment_guards_neural import NEURAL_GUARDS_OK\n\n'
            'def build__mojolearn_a() -> PythonObject:\n'
            '    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"\n'
            '    try:\n        pass\n')
    (tmp_path / 'bindings/_mojolearn_a.mojo').write_text(good)
    (tmp_path / 'bindings/_mojolearn_b.mojo').write_text(
        'def build__mojolearn_b() -> PythonObject:\n    try:\n        pass\n')
    (tmp_path / 'bindings/_mojolearn_c.mojo').write_text(good.replace('_mojolearn_a', '_mojolearn_c').replace(
        '    comptime assert NEURAL', '    var x = 1\n    comptime assert NEURAL'))
    problems = eg.layout_problems(tmp_path, ['_mojolearn_a', '_mojolearn_b', '_mojolearn_c'])
    assert any('core/experiment_guards_neural.mojo imports' in p for p in problems)
    assert any('_mojolearn_b.mojo imports 0 family' in p for p in problems)
    assert any('_mojolearn_c.mojo: build__mojolearn_c() does not assert' in p for p in problems)
    assert not any('_mojolearn_a.mojo' in p for p in problems)


def test_repository_layout():
    assert eg.layout_problems() == []


def test_repository_split_kept_the_permanent_record():
    before = eg.base_text()
    if before is None:
        pytest.skip('pre-split guard file not in this clone')
    assert eg.record_problems(before, dict(eg.guard_texts())) == []


def test_grid_guards_read_every_module():
    import six_lane_grid as G
    g = G.Guards.of_files()
    assert g.assert_count >= 348
    assert g.problems([]) == []
    assert g.problems(['MOJOLEARN_IDN_NEURAL_NN01=1'])           # a neural module record
    assert g.problems(['MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3'])  # a shared arm set
    assert [str(p.relative_to(G.ROOT)) for p in G.GUARD_FILES] == eg.guard_files()
    for rel, text in eg.guard_texts():   # each module stands alone: every comptime name it uses is its own
        assert G.Guards(text).problems([]) == [], rel
