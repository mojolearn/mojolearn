"""Independent MLP calculus and complete native snapshot admission tests."""
import numpy as np
import pytest

from mojolearn import SmallMLPTrainer, _backend
from mojolearn._parallel_mlp import _native_parallel_mlp as native
from mojolearn._verify_native_runtime import audit_native_call


def model_fixture():
    w1 = np.zeros((16, 8), np.float32)
    w2 = np.zeros((3, 16), np.float32)
    b1 = np.zeros(16, np.float32)
    b2 = np.zeros(3, np.float32)
    w1[0, 0], b1[0] = 1., .5
    w2[:, 0] = [.25, -.25, .125]
    return SmallMLPTrainer(w1, b1, w2, b2, data_schedule={'dataset': 'independent-small-mlp'},
                           lr=.0625, betas=(.5, .75), eps=1e-8, weight_decay=.25)


@pytest.fixture(autouse=True)
def identical_mode():
    if _backend.default_mode() != 'identical':
        pytest.skip('ParallelNeural preserves its existing identical-only admission')


def checked(operation, *args):
    return audit_native_call(native, operation, args).assert_native()


def independent_gradient(state, x, y):
    w1, b1, w2, b2 = (np.asarray(state['weights'][name], dtype=np.float64)
                       for name in ('weight1', 'bias1', 'weight2', 'bias2'))
    z = np.asarray(x, dtype=np.float64) @ w1.T + b1
    h = np.maximum(z, 0)
    logits = h @ w2.T + b2
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(probabilities[np.arange(len(y)), y]).mean()
    dl = probabilities.copy()
    dl[np.arange(len(y)), y] -= 1
    dl /= len(y)
    dh = (dl @ w2) * (z > 0)
    return loss, [dh.T @ x, dh.sum(axis=0), dl.T @ h, dl.sum(axis=0)]


@pytest.mark.parametrize('fused', ['0', '1'])
@pytest.mark.parametrize('integer_dtype', [np.int16, np.int32, np.int64, np.uint64])
def test_gradient_matches_independent_calculus_and_preserves_input(monkeypatch, fused, integer_dtype):
    monkeypatch.setenv('MOJOLEARN_MLP_FUSED', fused)
    model = model_fixture()
    state = checked('snapshot', model)
    x = np.zeros((3, 8), np.float32)
    x[:, 0] = [1., -.25, .5]
    y = np.array([0, 1, 2], integer_dtype)
    expected_loss, expected_gradients = independent_gradient(state, x, y.astype(np.intp))
    original = {key: np.asarray(value).tobytes() for key, value in state['weights'].items()}
    loss, gradients = checked('gradient', state, np.asfortranarray(x), y)
    assert loss == pytest.approx(expected_loss, rel=2e-6, abs=2e-7)
    for observed, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(np.asarray(observed), expected, rtol=3e-6, atol=3e-7)
    assert {key: np.asarray(value).tobytes() for key, value in state['weights'].items()} == original


def test_snapshot_reconstruction_is_owned_and_usable_by_existing_model_api():
    model = model_fixture()
    state = checked('snapshot', model)
    rebuilt = checked('from_state', state)
    result = checked('snapshot', rebuilt)
    # Existing public model methods run outside the native audit as a
    # compatibility check; they are not used as the numerical oracle.
    ordinary = rebuilt.state_dict()
    for key in ('schema', 'architecture', 'numeric_mode', 'parameter_order', 'config', 'data_schedule'):
        assert result[key] == ordinary[key] == state[key]
    for key in state['weights']:
        np.testing.assert_array_equal(np.asarray(result['weights'][key]), np.asarray(state['weights'][key]))
    before = np.asarray(result['weights']['weight1']).copy()
    np.asarray(state['weights']['weight1'])[:] = 99
    np.testing.assert_array_equal(np.asarray(checked('snapshot', rebuilt)['weights']['weight1']), before)


def test_invalid_unsigned_target_refused_without_python_callbacks():
    state = checked('snapshot', model_fixture())
    observed = audit_native_call(native, 'gradient', (state, np.zeros((1, 8), np.float32),
                                                     np.array([2**64 - 1], np.uint64)))
    assert isinstance(observed.exception, ValueError)
    assert 'classes 0..2' in str(observed.exception)
    report = observed.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not report[key], report


def test_rejected_snapshot_does_not_partially_publish():
    model = model_fixture()
    before = checked('snapshot', model)
    invalid = checked('snapshot', model)
    np.asarray(invalid['optimizer']['v'])[0] = -1
    observed = audit_native_call(native, 'publish', (model, invalid))
    assert isinstance(observed.exception, ValueError)
    after = checked('snapshot', model)
    for key in before['weights']:
        assert np.asarray(before['weights'][key]).tobytes() == np.asarray(after['weights'][key]).tobytes()
