"""The installed endpoint verifier consumes the selected program's training and outputs."""
from contextlib import nullcontext

from mojolearn import _verify_api_faults as faults
from mojolearn import _verify_api_runner as api


def test_imported_program_fault_moves_each_endpoint_and_restores_its_bytes():
    endpoints = faults.fault_endpoints('program-product')
    clean = api.capture(endpoints=endpoints, side='cpu')
    retained = {}
    result = faults.check_faults(clean, retain=retained.__setitem__)['program-product']
    assert result['status'] == 'CAUGHT_AND_RESTORED', result
    assert set(result['differences']) == {'output differs: ' + endpoint for endpoint in endpoints}
    assert set(retained) == {'program-product-faulty', 'program-product-restored'}
    restored = retained['program-product-restored']
    assert restored['endpoints'] == clean['endpoints']
    assert retained['program-product-faulty']['side'] == restored['side'] == 'cpu'


def test_unchanged_vjp_derivative_fails_even_when_forward_changes(monkeypatch):
    endpoint = faults.FAULTS['program-product'].rsplit('.', 1)[0] + '.vjp_if'

    def capture(output, derivative):
        return dict(expected=[endpoint], pending=[], endpoints={
            endpoint: dict(status='captured', kind='numerical', output={
                'outputs': output, 'input_gradients': derivative, 'parameter_gradient': derivative,
            }),
        })

    clean = capture(1, 2)
    replays = iter((capture(3, 2), clean))
    monkeypatch.setattr(faults, 'inject', lambda name: nullcontext())
    monkeypatch.setattr(api, 'capture', lambda **kwargs: next(replays))
    result = faults.check_faults(clean)['program-product']
    assert result['status'] == 'FAILED'
    assert result['differences'] == ['output differs: ' + endpoint]
    assert result['observation_errors'] == [
        endpoint + ' did not detect the fault in input_gradients',
        endpoint + ' did not detect the fault in parameter_gradient',
    ]
