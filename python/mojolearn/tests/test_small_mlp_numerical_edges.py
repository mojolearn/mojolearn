# SPDX-License-Identifier: Apache-2.0
"""Root-only remote native-edge and AdamW numerical qualification.

Authored tests, not qualification evidence. Opt in with
MOJOLEARN_RUN_SMALL_MLP_GPU=1 on remote Linux CUDA/HIP in IDENTICAL mode.
No Apple execution. CPU float64 equations below are independent quality
oracles, never performance baselines or cross-library bitwise certificates.
Helper bit-pattern expectations are closed-form IEEE edge cases; optimizer
updates use the predeclared tolerances below because FP64 equations and
pinned FP32 operations have different rounding schedules.
"""
import os
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MOJOLEARN_CPU_THREADS'):
    os.environ[_key] = '2'

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('MOJOLEARN_RUN_SMALL_MLP_GPU') != '1',
    reason='root-only opt-in remote Linux CUDA/HIP numerical tests')

# Fixed before any execution. These are tolerance gates, not equality gates.
UPDATE_RTOL = 2e-5
UPDATE_ATOL = 2e-7


@pytest.fixture
def native():
    if sys.platform != 'linux':
        pytest.fail('SmallMLP numerical edges refuse Apple/non-Linux execution')
    from mojolearn import _backend, _mlp_impl
    if _backend.default_mode() != 'identical' or _backend.numeric_mode() != 'identical':
        pytest.fail('Select IDENTICAL before importing the model package')
    binding = _mlp_impl.SmallMLPTrainer._binding()
    if _backend.read_vendor(binding) not in ('cuda', 'hip'):
        pytest.fail('SmallMLP numerical edges require native CUDA or HIP')
    if int(binding.training_numeric_mode()) != 1:
        pytest.fail('Native training binding is not IDENTICAL')
    return binding


def address(array):
    assert array.dtype == np.float32 and array.flags.c_contiguous
    return int(array.ctypes.data)


def bits(values):
    return np.asarray(values, dtype=np.uint32).view(np.float32)


def test_bias_ftz_and_relu_signed_zero_closed_form(native):
    # FTZ keeps the sign; adding -0 to -0 keeps -0. Smallest normal values
    # remain normal. ReLU must canonicalize every nonpositive result to +0.
    x = bits([0x80000000, 0, 1, 0x80000001, 0x00800000, 0x80800000]).reshape(1, 6)
    bias = bits([0x80000000] * 6)
    output = np.empty_like(x)
    expected = np.asarray([0x80000000, 0, 0, 0x80000000, 0x00800000, 0x80800000], np.uint32)
    before = x.tobytes(), bias.tobytes()
    assert native.mlp_bias_activation(address(x), address(bias), address(output), [1, 6, 0]) == 6
    np.testing.assert_array_equal(output.view(np.uint32).ravel(), expected)
    assert native.mlp_bias_activation(address(x), address(bias), address(output), [1, 6, 1]) == 6
    np.testing.assert_array_equal(output.view(np.uint32).ravel(), [0, 0, 0, 0, 0x00800000, 0])
    assert before == (x.tobytes(), bias.tobytes())


def test_relu_backward_zero_derivative_and_ftz_seam(native):
    activation = bits([0, 0x80000000, 0xbf800000, 0x3f800000,
                       0x3f800000, 0x3f800000, 0x3f800000]).reshape(1, 7)
    incoming = bits([0x3f800000, 0xbf800000, 0x3f800000, 0x80000000,
                     1, 0x80000001, 0xbf000000]).reshape(1, 7)
    output = np.empty_like(activation)
    before = activation.tobytes(), incoming.tobytes()
    assert native.mlp_relu_backward(address(activation), address(incoming), address(output), [1, 7]) == 7
    np.testing.assert_array_equal(output.view(np.uint32).ravel(),
                                  [0, 0, 0, 0x80000000, 0, 0x80000000, 0xbf000000])
    assert before == (activation.tobytes(), incoming.tobytes())


def test_sum_rows_exercises_ordered_cancellation(native):
    # Both columns have exact real sum 2. At 2**24, adding 1 ties and rounds
    # back to 2**24 in FP32. Thus these ascending schedules yield 1 and 2.
    # An exact/FP64 reduction or reordered device reduction fails column 0.
    values = np.asarray([[2**24, 2**24], [1, -(2**24)], [-(2**24), 1], [1, 1]], np.float32)
    output = np.empty(2, np.float32)
    assert native.mlp_sum_rows(address(values), address(output), [4, 2]) == 2
    np.testing.assert_array_equal(output.view(np.uint32), [0x3f800000, 0x40000000])
    np.testing.assert_array_equal(values.astype(np.float64).sum(axis=0), [2., 2.])


@pytest.mark.parametrize('operation', ['bias', 'sum'])
def test_finite_inputs_with_overflow_are_refused(native, operation):
    maximum = np.finfo(np.float32).max
    if operation == 'bias':
        values = np.asarray([[maximum]], np.float32)
        bias = np.asarray([maximum], np.float32)
        output = np.empty_like(values)
        with pytest.raises((RuntimeError, ValueError), match='nonfinite'):
            native.mlp_bias_activation(address(values), address(bias), address(output), [1, 1, 0])
    else:
        values = np.asarray([[maximum], [maximum]], np.float32)
        output = np.empty(1, np.float32)
        with pytest.raises((RuntimeError, ValueError), match='nonfinite'):
            native.mlp_sum_rows(address(values), address(output), [2, 1])


def flatten(weights, names):
    return np.concatenate([weights[name].ravel() for name in names]).astype(np.float64)


def adamw_reference(before, gradients, *, reset_moments=False, ascent=False):
    """Independent FP64 AdamW equations, without pinned-op transcription."""
    names = before['parameter_order']
    parameters = flatten(before['weights'], names)
    gradient = flatten(gradients, names)
    state, config = before['optimizer'], before['config']
    first = np.zeros_like(parameters) if reset_moments else state['m'].astype(np.float64)
    second = np.zeros_like(parameters) if reset_moments else state['v'].astype(np.float64)
    beta1, beta2 = config['beta1'], config['beta2']
    t = state['step'] + 1
    first = beta1 * first + (1 - beta1) * gradient
    second = beta2 * second + (1 - beta2) * np.square(gradient)
    corrected_first = first / (1 - beta1**t)
    corrected_second = second / (1 - beta2**t)
    update = config['lr'] * corrected_first / (np.sqrt(corrected_second) + config['eps'])
    decayed = parameters * (1 - config['lr'] * config['weight_decay'])
    return decayed + update if ascent else decayed - update, first, second


def test_public_adamw_complete_state_matches_independent_equations(native):
    from mojolearn import SmallMLPTrainer
    shapes = ((16, 8), (16,), (3, 16), (3,))
    weights = [((np.arange(np.prod(shape), dtype=np.float32) % 7 - 3) / 32).reshape(shape)
               for shape in shapes]
    model = SmallMLPTrainer(*weights, data_schedule={'dataset': 'dyadic-edges.v1', 'next_batch': 4},
                            lr=.01, betas=(.75, .875), eps=1e-6, weight_decay=.125)
    planted = model.state_dict()
    # Nonzero carried state and a noninitial step make dropped moments and
    # reset bias-correction counters observable. These are valid public state.
    planted['optimizer']['step'] = 4
    planted['optimizer']['m'][:] = np.linspace(.01, .03, 195, dtype=np.float32)
    planted['optimizer']['v'][:] = np.linspace(.02, .04, 195, dtype=np.float32)
    model.load_state_dict(planted)
    for batch_index in (0, 1):
        before = model.state_dict()
        x = ((np.arange(7 * 8, dtype=np.float32) % 13 - 6 + batch_index) / 8).reshape(7, 8)
        y = (np.arange(7, dtype=np.int32) + batch_index) % 3
        result = model.train_step(x, y, return_input_grad=True)
        after = model.state_dict()
        expected_parameters, expected_m, expected_v = adamw_reference(before, result['gradients'])
        actual_parameters = flatten(after['weights'], before['parameter_order'])
        for actual, expected in ((actual_parameters, expected_parameters),
                                 (after['optimizer']['m'], expected_m),
                                 (after['optimizer']['v'], expected_v)):
            np.testing.assert_allclose(actual, expected, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        assert after['optimizer']['step'] == before['optimizer']['step'] + 1 == result['step']
        np.testing.assert_array_equal(after['optimizer']['flags'], before['optimizer']['flags'])
        for key in ('schema', 'architecture', 'numeric_mode', 'parameter_order', 'config', 'data_schedule'):
            assert after[key] == before[key]
        # Negative controls establish that this tolerance sees missing carried
        # moments and an optimizer sign reversal; neither requires a second
        # GPU update or a change to the production implementation.
        reset_parameters, reset_m, reset_v = adamw_reference(before, result['gradients'], reset_moments=True)
        ascent_parameters, _, _ = adamw_reference(before, result['gradients'], ascent=True)
        assert not np.allclose(expected_parameters, reset_parameters, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        assert not np.allclose(expected_m, reset_m, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        assert not np.allclose(expected_v, reset_v, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
        assert not np.allclose(expected_parameters, ascent_parameters, rtol=UPDATE_RTOL, atol=UPDATE_ATOL)
