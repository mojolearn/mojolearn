# SPDX-License-Identifier: Apache-2.0
"""Run after rebuilding ET: invalid devices must fail before null data pointers.

PYTHONPATH=python python checks/extratrees_gpu_only_binding_smoke.py
No training or GPU kernels are launched by this refusal check.
"""
from mojolearn import ExtraTreesClassifier, ExtraTreesRegressor
from mojolearn.extratrees import _fit_params
from mojolearn._backend import binding

for mode in ('fast', 'deterministic', 'identical'):
    native = binding('_mojolearn_trees', mode=mode)
    for cls, name, classes in (
        (ExtraTreesClassifier, 'et_classifier_fit', 2),
        (ExtraTreesRegressor, 'et_regressor_fit', 0),
    ):
        model = cls()
        params = _fit_params(2, 1, classes, model._cfg, 'gpu', model._criterion_code)
        for invalid in (0, -1, 2, 1.5, float("nan"), float("inf")):
            params[20] = invalid
            try:
                getattr(native, name)(0, 0, params)
            except Exception as exc:
                assert 'GPU-only' in str(exc), (mode, name, invalid, str(exc))
            else:
                raise AssertionError((mode, name, invalid))
    print(mode, 'PASS: CPU/invalid device refused before pointer access')
