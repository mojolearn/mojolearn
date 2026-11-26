"""--classic, --neural and --all select the verification profiles; default/quick are classic."""
from types import SimpleNamespace

import pytest

from mojolearn import __main__ as cli
from mojolearn import _verification_profiles as profiles
from mojolearn import _verify_all as suite


def inventory():
    families = [dict(family='core', training_lanes=['knn', 'kmeans']),
                dict(family='x_linear', training_lanes=['x-sgd-clf']),
                dict(family='training', training_lanes=['mlp']),
                dict(family='neural', training_lanes=['hf-causal-lm']),
                dict(family='x_sequence', training_lanes=['sequence-lstm', 'sequence-stl'])]
    surface = SimpleNamespace(FAMILIES=families)
    harness = SimpleNamespace(LANES={name: None for f in families for name in f['training_lanes']})
    return harness, surface


def test_every_declared_classical_lane_and_inference_route_survives_default():
    harness, surface = inventory()
    lanes, report = profiles.select(harness, surface, list(harness.LANES))
    assert set(lanes) == {'knn', 'kmeans', 'x-sgd-clf', 'sequence-stl', 'hf-causal-lm'}
    assert report['excluded_neural'] == ['mlp', 'sequence-lstm']
    assert report['name'] == 'classic'
    assert report['neural_command'] == 'python -m mojolearn verify --neural'


def test_neural_is_the_complementary_scope():
    harness, surface = inventory()
    lanes, report = profiles.select(harness, surface, list(harness.LANES), profile='neural')
    assert set(lanes) == {'mlp', 'sequence-lstm'}
    assert report['name'] == 'neural' and report['excluded_neural'] == []
    with pytest.raises(ValueError, match='use --neural or --all'):
        profiles.select(harness, surface, ['mlp'], asked=['mlp'])
    with pytest.raises(ValueError, match='use --classic or --all'):
        profiles.select(harness, surface, ['knn'], profile='neural', asked=['knn'])


def test_all_is_both_profiles_and_excludes_nothing():
    harness, surface = inventory()
    classic, _ = profiles.select(harness, surface, list(harness.LANES), profile='classic')
    neural, _ = profiles.select(harness, surface, list(harness.LANES), profile='neural')
    lanes, report = profiles.select(harness, surface, list(harness.LANES), profile='all',
                                    asked=['mlp', 'knn'])
    assert lanes == list(harness.LANES) and set(lanes) == set(classic) | set(neural)
    assert not set(classic) & set(neural)
    assert report['name'] == 'all' and report['excluded_neural'] == []
    with pytest.raises(ValueError, match='unknown verification profile'):
        profiles.select(harness, surface, list(harness.LANES), profile='routine')


@pytest.mark.parametrize('flags,quick,profile', [([], False, None), (['--quick'], True, None),
                                                 (['--classic'], False, 'classic'),
                                                 (['--neural'], False, 'neural'),
                                                 (['--all'], False, 'all')])
def test_public_cli_dispatch_selects_the_profile(monkeypatch, flags, quick, profile):
    seen = []
    monkeypatch.setattr(suite, 'cmd_verify_all', lambda args: seen.append(args) or 0)
    assert cli._verify_dispatch(cli.build_parser().parse_args(['verify', *flags])) == 0
    assert seen[0].routine
    assert seen[0].quick is quick and seen[0].profile == profile
    assert suite._profile(seen[0]) == (profile or 'classic')
    # the bundled saved models are classic: --neural alone skips them, --all runs them
    assert seen[0].no_models is (profile == 'neural')
    assert suite._depth(seen[0]) == ('quick' if quick else 'full' if profile else 'base')


@pytest.mark.parametrize('flag', ['--neural-training', '--full', '--training', '--inference'])
def test_retired_scope_flags_are_usage_errors(flag):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['verify', flag])


@pytest.mark.parametrize('flags', [['--classic', '--neural'], ['--all', '--neural'], ['--all', '--models-only']])
def test_profile_flags_are_exclusive(flags):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['verify', *flags])


def test_profile_counts_split_one_run_by_profile():
    rows = [dict(lane='knn', state='IDENTICAL'), dict(lane='mlp', state='DIVERGENT'),
            dict(lane='portable:rf', state='IDENTICAL')]
    both = suite.profile_counts(rows, {'mlp'}, 'all')
    assert set(both) == {'classic', 'neural'}
    assert both['classic']['lanes'] == 1 and both['classic']['counts']['IDENTICAL'] == 2
    assert both['neural']['lanes'] == 1 and both['neural']['counts']['DIVERGENT'] == 1
    assert set(suite.profile_counts(rows, {'mlp'}, 'neural')) == {'neural'}


def test_public_forest_algorithms_cannot_disappear_from_cross_check():
    harness = suite.load_harness()
    selected, every, families = suite.cross_check_lanes(harness, 'all')
    assert set(suite.CROSS_CHECK_FOREST_LANES) <= set(selected)
    assert selected == every
    assert families['forest'] == 'rf-clf'
