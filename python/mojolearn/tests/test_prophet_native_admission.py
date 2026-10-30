"""Public lazy import must preserve Prophet's captured native class layout."""
import subprocess
import sys

import pytest


PROGRAM = r'''
import sys
import mojolearn as ml

order, changed = sys.argv[1:]
if order == 'public_first':
    public = ml.ProphetForecaster
import mojolearn._x_sequence_prophet as module
cls = module.ProphetForecaster
context = module._XSEQ_CONTEXT

def assert_layout():
    mismatches = [(owner.__name__, key)
                  for owner, hooks in context['xseq_prophet_layouts']
                  for key, expected in hooks if vars(owner).get(key) is not expected]
    assert not mismatches, mismatches

assert_layout()
if changed == 'none':
    direct = cls(n_changepoints=3, max_iter=4)
    public = ml.ProphetForecaster
    assert public is cls
    assert_layout()
    model = public(n_changepoints=3, max_iter=4)
    assert model.growth == direct.growth == 'linear'
    assert model.n_changepoints == 3
    assert model.add_seasonality('custom', period=5., fourier_order=2) is model
    assert model.seasonalities['custom']['fourier_order'] == 2
    assert model.get_params()['n_changepoints'] == 3
    assert model.set_params(n_changepoints=2) is model
    assert model.n_changepoints == 2
else:
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(changed)
        raise AssertionError('changed Python class hook executed')
    missing = object()
    original = vars(cls).get(changed, missing)
    setattr(cls, changed, forbidden)
    try:
        # A later public lookup must not refresh the captured lifecycle
        # around an application mutation and thereby bless that mutation.
        public = ml.ProphetForecaster
        assert public is cls
        try:
            public()
        except TypeError as error:
            assert 'methods or storage hooks changed' in str(error), str(error)
        else:
            raise AssertionError('changed class hook was admitted')
        assert not calls, calls
    finally:
        if original is missing:
            delattr(cls, changed)
        else:
            setattr(cls, changed, original)
    assert_layout()
    assert cls().growth == 'linear'
print('passed', order, changed)
'''


@pytest.mark.parametrize("order", ["direct_first", "public_first"])
@pytest.mark.parametrize("changed", ["none", "__getattribute__", "fit"])
def test_prophet_public_layout_and_mutation_refusal(order, changed):
    result = subprocess.run([sys.executable, "-c", PROGRAM, order, changed],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
