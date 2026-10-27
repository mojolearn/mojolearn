# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Replay retained startup readbacks; no fit, GPU, or benchmark execution."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import bench_board_algos as A
import bench_board_params as BP

RECORDS = json.loads((Path(__file__).parent / 'param_readbacks/smoke0862_refused_lanes.json').read_text())


@pytest.mark.parametrize('key', sorted(RECORDS))
def test_explicit_recipe_repair_and_negative_control(key):
    race = RECORDS[key]
    lane = race['lane']
    arms = copy.deepcopy(race['arms'])
    assert BP.check('algos/' + lane, arms, family='algos')['verdict'] == 'REFUSED'
    changed = ('sample_interval',) if lane == 'additive-chi2' else ('cache_size', 'shrinking')
    spec = A.LANES[lane]
    for name in changed:
        for arm, rec in arms.items():
            recipe = spec['params'] if arm == 'ours' else spec.get('sk_params', spec['params'])
            rec['params'][name] = recipe[name]
    result = BP.check('algos/' + lane, arms, family='algos')
    assert result['verdict'] == 'MATCHED', result['problems']
    for name in changed:
        bad = copy.deepcopy(arms)
        v = bad['sklearn-cpu']['params'][name]
        bad['sklearn-cpu']['params'][name] = not v if isinstance(v, bool) else v + 1
        assert BP.check('algos/' + lane, bad, family='algos')['verdict'] == 'REFUSED'
