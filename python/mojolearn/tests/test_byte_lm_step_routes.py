# SPDX-License-Identifier: Apache-2.0
"""The byte LM's two training routes give the same loss words on a tiny model.

The default build trains with the restored hand step (training/byte_lm.mojo,
the 0.8.55 route); a build with -D MOJOLEARN_BYTE_LM_PROGRAM_STEP trains with
lane x7's program step (one Program.fit per call). One installed wheel holds
one route, so the comparison crosses two runs of this file:

    MOJOLEARN_BYTE_LM_ROUTE_RECORD=<path>     write this build's loss words
    MOJOLEARN_BYTE_LM_ROUTE_REFERENCE=<path>  compare them with the other
                                              build's recorded words

Without the reference the cross-route test skips; the within-build checks
(evaluate before a step equals that step's loss; train_steps equals the same
train_step calls word for word) run on either build.

Within one build every column runs the same route (the switch lives in
training/byte_lm_config.mojo): the CPU column's composed host step by default,
its host program under the define. `test_host_and_device_give_the_same_words`
trains the same tiny model on the GPU trainer and on the CPU column's
`LanguageModelHostTrainer` in one process and requires the same loss, gradient,
parameter and moment words.
"""
import json
import os

import numpy as np
import pytest

from mojolearn import SmallByteLanguageModelTrainer as Trainer
from mojolearn import ByteLanguageModelConfig as Shape

N_STEPS = 3


def initial():
    return (np.random.default_rng(41).standard_normal(34944) * .02).astype(np.float32)


def batches():
    rng = np.random.default_rng(43)
    return rng.integers(0, 256, (N_STEPS, 2, 33), dtype=np.int32)


def model():
    return Trainer(initial(), data_schedule={'dataset': 'route-test'}, resident=True)


def words(values):
    return [int(w) for w in np.asarray(values, dtype=np.float32).view(np.uint32)]


def route_words():
    """Loss words of one tiny model: the evaluation before training, then
    N_STEPS single steps, then the evaluation after them."""
    data = batches()
    trainer = model()
    try:
        before = trainer.evaluate(data[0])
        steps = [trainer.train_step(data[k])['loss'] for k in range(N_STEPS)]
        after = trainer.evaluate(data[0])
    finally:
        trainer.close()
    return {'evaluate_before': words([before]), 'steps': words(steps), 'evaluate_after': words([after])}


def test_evaluate_before_a_step_is_that_steps_loss():
    data = batches()
    trainer = model()
    try:
        before = trainer.evaluate(data[0])
        loss = trainer.train_step(data[0])['loss']
    finally:
        trainer.close()
    assert np.isfinite(before)
    assert words([before]) == words([loss])


def test_train_steps_equals_the_same_train_step_calls():
    data = batches()
    single = model()
    multi = model()
    try:
        one_by_one = [single.train_step(data[k])['loss'] for k in range(N_STEPS)]
        result = multi.train_steps(data)
        assert words(result['losses']) == words(one_by_one)
        assert result['step'] == N_STEPS and multi.step_ == single.step_ == N_STEPS
        assert (np.asarray(multi.state_dict()['parameters']).tobytes()
                == np.asarray(single.state_dict()['parameters']).tobytes())
    finally:
        single.close()
        multi.close()


def test_both_routes_give_the_same_loss_words():
    record = os.environ.get('MOJOLEARN_BYTE_LM_ROUTE_RECORD', '')
    reference = os.environ.get('MOJOLEARN_BYTE_LM_ROUTE_REFERENCE', '')
    if not record and not reference:
        pytest.skip('set MOJOLEARN_BYTE_LM_ROUTE_RECORD / _REFERENCE to compare the hand and program builds')
    got = route_words()
    if record:
        with open(record, 'w') as f:
            json.dump(got, f, sort_keys=True)
    if not reference:
        pytest.skip('recorded this build\'s words; run the other build with MOJOLEARN_BYTE_LM_ROUTE_REFERENCE')
    with open(reference) as f:
        want = json.load(f)
    assert got == want


def _host_trainer():
    """The CPU column's trainer at the same initialization and optimizer, or
    a skip naming why the host binding is not loadable in this install."""
    try:
        from mojolearn import LanguageModelHostTrainer
        return LanguageModelHostTrainer(initial(), shape=Shape(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                                        weight_decay=.01)
    except (ImportError, OSError) as error:
        pytest.skip(f'the CPU byte LM binding is not loadable here: {error}')


def _bytes(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).tobytes()


def test_host_and_device_give_the_same_words():
    data = batches()
    host = _host_trainer()
    device = Trainer(initial(), data_schedule={'dataset': 'route-test'}, lr=1e-3, betas=(.9, .999), eps=1e-8,
                     weight_decay=.01, resident=True)
    try:
        assert words([host.loss(data[0])]) == words([device.evaluate(data[0])])
        host_losses = [int(host.train_step(data[k])) for k in range(N_STEPS)]
        device_losses = words([device.train_step(data[k])['loss'] for k in range(N_STEPS)])
        assert host_losses == device_losses
        state = device.state_dict()
        assert _bytes(host.parameters_) == _bytes(state['parameters'])
        assert _bytes(host.m_) == _bytes(state['m'])
        assert _bytes(host.v_) == _bytes(state['v'])
        assert words([host.loss(data[0])]) == words([device.evaluate(data[0])])
    finally:
        device.close()
