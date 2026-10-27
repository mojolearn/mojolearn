"""Public constructors and mode readbacks used before measurement; no fits."""
import sys
from pathlib import Path

import numpy as np
import pytest
import mojolearn as ml


@pytest.mark.parametrize('name', ['DecisionTreeClassifier', 'DecisionTreeRegressor'])
@pytest.mark.parametrize('mode', [None, 'identical'])
def test_decision_tree_accepts_documented_mode_and_reads_it_back(name, mode):
    model = getattr(ml, name)(numeric_mode=mode, max_depth=16, n_bins=128)
    assert model.get_params()['numeric_mode'] == mode
    assert model.numeric_mode_used() == 'identical'
    assert model.vendor_used() in ('metal', 'cuda', 'hip')


def test_svc_measurement_probe_uses_the_class_native_binding():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
    import classical_two_datasets as C
    data = dict(X=np.zeros((4, 2), dtype=np.float32),
                y=np.array([0, 1, 0, 1]), Xq=np.zeros((2, 2), dtype=np.float32))
    runner = C.OursSVC(data, {'svc': {'gamma': 0.5}})
    assert '_BINDING' not in vars(runner.params_obj)
    assert runner.info['numeric_mode_used'] == 'identical'
    assert runner.info['vendor_used'] in ('metal', 'cuda', 'hip')


@pytest.mark.parametrize('lane', ['decision-tree-clf', 'decision-tree-reg'])
def test_tree_measurement_reads_compiled_configuration_without_fit(lane):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
    import bench_board_algos as A
    data = dict(X=np.zeros((4, 2), dtype=np.float32),
                Xq=np.zeros((2, 2), dtype=np.float32),
                y=np.array([0, 1, 0, 1]), yq=np.array([0, 1]))
    # build constructs the runner and its parameter record; fit is a separate callback.
    runner = A.build(lane, 'ours', data)
    assert runner is not None
