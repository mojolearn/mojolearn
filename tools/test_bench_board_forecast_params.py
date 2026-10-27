import bench_board_forecast_params as F
import bench_board_params as BP


def test_var_fixed_order_only():
    p = dict(ic=None, maxlags=2)
    assert F.semantic_exception('algos/var', 'ic', 'statsmodels-cpu', p, p, 'statsmodels')
    assert not F.semantic_exception('algos/var', 'ic', 'statsmodels-cpu', p, dict(p, ic='aic'), 'statsmodels')
    assert not F.semantic_exception('algos/var', 'ic', 'statsmodels-cpu', p, {}, 'statsmodels')


def test_theta_estimation_only():
    for key in ('initial_smoothed', 'alpha', 'theta'):
        assert F.semantic_exception('algos/theta', key, 'statsforecast-cpu', {key: None}, {}, 'statsforecast')
        assert not F.semantic_exception('algos/theta', key, 'statsforecast-cpu', {key: 0.1}, {}, 'statsforecast')
        assert not F.semantic_exception('algos/damped-ets', key, 'statsforecast-cpu', {key: None}, {}, 'statsforecast')
    assert not F.semantic_exception('algos/autoarima', 'method', 'statsforecast-cpu', {'method': 'ml'}, {'method': 'CSS-ML'}, 'statsforecast')


def test_controls_do_not_hide_active_intervals_or_distribution():
    for key, value in [('prediction_intervals', object()), ('distribution', 'student')]:
        assert not F.semantic_exception('algos/theta', key, 'statsforecast-cpu', {}, {key: value}, 'statsforecast')


def test_record_retains_inherited_and_unknown_attributes():
    class Model:
        def __init__(self, season_length=24):
            self.season_length = season_length
            self.model = 'STM'
            self.new_control = 7
    record = F.statsforecast_record(Model(), BP)
    assert record['params'] == dict(season_length=24, model='STM', new_control=7)
    assert record['implementation']['name'].endswith('Model')


def test_croston_interval_method_only_when_intervals_disabled():
    theirs = dict(only_conformal_intervals=True, prediction_intervals=None)
    assert F.semantic_exception('algos/croston', 'only_conformal_intervals', 'statsforecast-cpu', {}, theirs, 'statsforecast')
    assert not F.semantic_exception('algos/croston', 'only_conformal_intervals', 'statsforecast-cpu', {}, dict(theirs, prediction_intervals={'n_windows': 2}), 'statsforecast')
