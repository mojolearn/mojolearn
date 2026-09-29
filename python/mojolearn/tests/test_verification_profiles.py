"""Default/quick bundle routine work; neural training cannot sneak in."""
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
    assert report['excluded_neural_training'] == ['mlp', 'sequence-lstm']
    assert report['name'] == 'routine'


def test_neural_training_is_explicit_and_has_the_complementary_scope():
    harness, surface = inventory()
    lanes, report = profiles.select(harness, surface, list(harness.LANES), neural=True)
    assert set(lanes) == {'mlp', 'sequence-lstm'}
    assert report['name'] == 'neural-training'
    with pytest.raises(ValueError, match='--neural-training explicitly'):
        profiles.select(harness, surface, ['mlp'], asked=['mlp'])


@pytest.mark.parametrize('flags,quick,neural', [([], False, False), (['--quick'], True, False),
                                            (['--neural-training'], False, True)])
def test_public_cli_dispatch_bundles_inference_and_classical(monkeypatch, flags, quick, neural):
    seen = []
    monkeypatch.setattr(suite, 'cmd_verify_all', lambda args: seen.append(args) or 0)
    assert cli._verify_dispatch(cli.build_parser().parse_args(['verify', *flags])) == 0
    assert seen[0].routine
    assert seen[0].quick is quick and seen[0].neural_training is neural
    assert seen[0].no_models is neural


def test_public_forest_algorithms_cannot_disappear_from_cross_check():
    harness = suite.load_harness()
    selected, every, families = suite.cross_check_lanes(harness, 'all')
    assert set(suite.CROSS_CHECK_FOREST_LANES) <= set(selected)
    assert selected == every
    assert families['forest'] == 'rf-clf'
