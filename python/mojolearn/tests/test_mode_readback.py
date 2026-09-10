"""Compiled mode takes precedence over a native module's internal name."""
from types import SimpleNamespace
import pytest
from mojolearn._mode import NumericModeMixin


@pytest.mark.parametrize('code,expected', [(0, 'fast'), (1, 'identical'), (2, 'deterministic')])
@pytest.mark.parametrize('getter', ['metrics_numeric_mode', 'umap_numeric_mode'])
def test_metrics_compiled_mode_without_tier_in_name(code, expected, getter):
    module = SimpleNamespace(__name__='mojolearn._mojolearn_metrics',
                             **{getter: lambda: code})
    model = NumericModeMixin()
    model._bind = lambda: module
    assert model.numeric_mode_used() == expected


def test_readback_wins_over_misleading_module_path():
    model = NumericModeMixin()
    model._bind = lambda: SimpleNamespace(
        __name__='mojolearn.identical._mojolearn_rf', rf_numeric_mode=lambda: 0)
    assert model.numeric_mode_used() == 'fast'


def test_unknown_compiled_mode_is_not_inferred_from_path():
    model = NumericModeMixin()
    model._bind = lambda: SimpleNamespace(
        __name__='mojolearn.identical._mojolearn_rf', rf_numeric_mode=lambda: 99)
    with pytest.raises(RuntimeError, match='unknown numeric mode'):
        model.numeric_mode_used()


def test_legacy_artifact_keeps_path_hint():
    model = NumericModeMixin()
    model._bind = lambda: SimpleNamespace(__name__='mojolearn.identical._mojolearn_gp')
    assert model.numeric_mode_used() == 'identical'
