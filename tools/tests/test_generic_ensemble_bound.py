"""Tool tests for the trees group's ensemble bound in the generic lanes (lane slow-lanes-1, 2026-11-26).

tools/identity_lanes/generic.py and checks_generic.py fit every class of the trees group at n_estimators 8 when its
own value is unset or larger; a fit that refuses the bound runs again at the class's own size, so no fitted class
becomes an n/a part. The fragments are executed in a namespace of stubs, as tools/identity_break.py executes them
in its own (`_load_lane_fragments`); no mojolearn and no device are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _fragment(name: str) -> dict:
    ns = {"np": np, "lane": lambda _name: (lambda fn: fn), "_batch_decl": lambda *a, **k: None,
          "_h": lambda *a: "h", "_fit": lambda parts: parts, "_lane_side": lambda: "cpu"}
    exec(compile((ROOT / "tools" / "identity_lanes" / name).read_text(), name, "exec"), ns)
    return ns


class _Ensemble:
    """A stand-in with get_params / set_params, its fit refusing sizes in `refuse`."""

    _estimator_type = "regressor"

    def __init__(self, n_estimators=None, refuse=()):
        self.n_estimators = n_estimators
        self.refuse = tuple(refuse)
        self.fitted_at = []

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "refuse": self.refuse}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self.fitted_at.append(self.n_estimators)
        if self.n_estimators in self.refuse:
            raise ValueError("n_estimators must be at least 10")
        return self


def test_the_bound_sets_unset_and_large_sizes_in_the_trees_group_only():
    for frag, bound in (("generic.py", "_generic_bound_ensemble"), ("checks_generic.py",
                                                                    "_checks_generic_bound_ensemble")):
        fn = _fragment(frag)[bound]
        for own in (None, 100, 1000, np.int64(50)):
            est = _Ensemble(own)
            assert fn(est, "trees") == {"n_estimators": own}
            assert est.n_estimators == 8
        for own in (1, 8):
            est = _Ensemble(own)
            assert fn(est, "trees") == {} and est.n_estimators == own
        est = _Ensemble(100)
        assert fn(est, "linear") == {} and est.n_estimators == 100


def test_a_refused_bound_fits_at_the_class_own_size_never_n_a():
    ns = _fragment("generic.py")
    fixture = ns["_generic_fixture"](np.ones((64, 4), np.float32), np.zeros(64, np.int64), np.ones(64, np.float32))
    failed = []

    class Refusing(_Ensemble):
        def __init__(self, n_estimators=100, refuse=(8,)):
            super().__init__(n_estimators, refuse)

    made = []
    ml = type("ML", (), {"Refusing": staticmethod(lambda **kw: made.append(Refusing(**kw)) or made[-1])})()
    est, args, why = ns["_generic_prepare"](ml, fixture, "", "Refusing", Refusing, True, "Refusing", "cpu", failed,
                                            "trees")
    assert failed == [] and why is None and args is not None
    assert est.fitted_at == [8, 100] and est.n_estimators == 100


def test_the_params_lanes_and_other_groups_fit_at_the_class_own_size():
    ns = _fragment("generic.py")
    fixture = ns["_generic_fixture"](np.ones((64, 4), np.float32), np.zeros(64, np.int64), np.ones(64, np.float32))
    made = []
    ml = type("ML", (), {"E": staticmethod(lambda **kw: made.append(_Ensemble(**kw)) or made[-1])})()
    est, _args, _why = ns["_generic_prepare"](ml, fixture, "", "E", _Ensemble, True, "E", "cpu", [], "linear")
    assert est.fitted_at == [None]
