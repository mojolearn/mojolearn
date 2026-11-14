"""Complete native snapshot admission tests of the parallel small-MLP owner.

The independent calculus of a shard gradient and the first two AdamW apply
steps moved to checks/native_tests/training_small_mlp_{cpu,gpu}.mojo, the
ordered left-associated gradient fold to training_primitives_{cpu,gpu}.mojo
(lane tmig-training, 2026-11-14). Snapshot publication, reconstruction and
target admission stay here.
"""
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
