"""Prepared manifests must retain exact full recipe coverage and owner gates."""
import json
from pathlib import Path

import bench_board as B

PLAN = Path(__file__).resolve().parents[1] / 'experiments/neural29_refactor_run_20261027'


def test_plan_is_exact_current_neural_recipe_set_with_full_settings():
    manifest = json.loads((PLAN / 'manifest.json').read_text())
    recipes = json.loads((PLAN / 'recipes.json').read_text())
    assert not recipes['measured'] and recipes['shape'] == 'full'
    assert recipes['scored_rounds'] == recipes['warmups'] == 1
    for vendor in ('amd', 'nvidia'):
        current = B.plan_races(vendor, ['identical'], families=['neural'], neural_shape='full')
        assert len(manifest[vendor]) == len(set(manifest[vendor])) == len(current) == 29
        assert set(manifest[vendor]) == {r['id'] for r in current}
        assert {r['id'] for r in recipes['recipes'][vendor]} == set(manifest[vendor])
        for race in recipes['recipes'][vendor]:
            assert race['dimensions'] == B.NEURAL.shape_record(race['lane'], 'full')
            assert race['settings'] == B.race_settings(dict(rounds=1, vendor=vendor), race)


def test_prepared_run_is_not_launched_and_has_no_extra_gpu_validation():
    launch = json.loads((PLAN / 'launch-plan.json').read_text())
    assert not launch['launched'] and not launch['old_evidence_mutated']
    assert launch['separate_gpu_validation_campaign'] is False
    assert 'internal CPython dispatch' in launch['prerequisite'][0]
    for vendor in launch['vendors'].values():
        assert vendor['expected'] == 29 and vendor['require_existing_box']
        assert not vendor['neural_smoke'] and not vendor['release_after_archive']
