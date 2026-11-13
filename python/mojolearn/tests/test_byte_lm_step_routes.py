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


# ---------------------------------------------------------------------------
# Forward-only bisect of the CPU column (lane byte-lm-host-step-words,
# 2026-11-13). The Mac run of this file (verify-20261112/fix20261113) passed
# the first assertion of test_host_and_device_give_the_same_words and failed
# on the step losses (host 1085368213, device 1085370318 at step 0). The
# bisect found every host forward equal and the cause in the door:
# `LanguageModelHostTrainer.loss` ran a whole training step and published it
# (bindings/byte_lm_host_trainer_native_door.mojo), so the `train_step` after
# it reported the loss AFTER an update. `loss` is now the forward alone.
# Every test below runs on the CPU column alone and compares one pair of host
# forwards, so a failing pair names the stage:
#   logits reference vs threaded     the block oracles + gemm_oracle head
#                                    against byte_lm_host_kernels.mojo
#   loss reference vs threaded       ce_forward_oracle against
#                                    ce_causal_mean_loss_fast on each path's logits
#   step (rows) vs step (oracle)     loss_host_rows.ce_host_rows against
#                                    ce_forward_oracle inside the step
#   step (oracle) vs loss reference  gemm_host_rows head + ce_forward_oracle
#                                    against gemm_oracle head + ce_forward_oracle
#   step vs trainer loss             the composed step against the threaded loss
# The last pair is the one the Mac run showed failing.

STEP_ROWS_ENV = 'MOJOLEARN_BYTE_LM_HOST_STEP_ROWS'


def _host_inference(threaded):
    """The CPU column's inference forward at the same initialization, or a
    skip naming why the host binding is not loadable in this install."""
    try:
        from mojolearn import LanguageModelInference
        return LanguageModelInference(initial(), shape=Shape(), threaded=threaded)
    except (ImportError, OSError) as error:
        pytest.skip(f'the CPU byte LM binding is not loadable here: {error}')


def _inputs(ids):
    """The (batch, length) input rows of a (batch, length + 1) training batch."""
    return np.ascontiguousarray(ids[:, :-1])


def test_host_logits_reference_equals_threaded():
    ids = batches()[0]
    reference = _host_inference(threaded=False)
    threaded = _host_inference(threaded=True)
    assert _bytes(reference.logits(_inputs(ids))) == _bytes(threaded.logits(_inputs(ids)))


def test_host_loss_reference_equals_threaded():
    ids = batches()[0]
    reference = _host_inference(threaded=False)
    threaded = _host_inference(threaded=True)
    assert int(reference.loss_bits(ids)) == int(threaded.loss_bits(ids))


def test_host_trainer_loss_equals_threaded_inference_loss():
    ids = batches()[0]
    host = _host_trainer()
    threaded = _host_inference(threaded=True)
    assert words([host.loss(ids)]) == [int(threaded.loss_bits(ids))]


def test_host_step_loss_equals_host_loss():
    """The composed step's forward against the threaded loss on the same
    parameters: the pair the Mac run showed failing."""
    ids = batches()[0]
    host = _host_trainer()
    before = words([host.loss(ids)])
    assert [int(host.train_step(ids))] == before


def test_host_step_rows_equals_step_oracle(monkeypatch):
    """`ce_host_rows` and `adam_host_rows` (the default) against the serial
    oracle calls (MOJOLEARN_BYTE_LM_HOST_STEP_ROWS=0): loss, gradient,
    parameters and both moments, every byte."""
    ids = batches()[0]
    monkeypatch.delenv(STEP_ROWS_ENV, raising=False)
    rows = _host_trainer()
    rows_loss = int(rows.train_step(ids))
    monkeypatch.setenv(STEP_ROWS_ENV, '0')
    oracle = _host_trainer()
    oracle_loss = int(oracle.train_step(ids))
    assert rows_loss == oracle_loss
    assert _bytes(rows.gradient_) == _bytes(oracle.gradient_)
    assert _bytes(rows.parameters_) == _bytes(oracle.parameters_)
    assert _bytes(rows.m_) == _bytes(oracle.m_)
    assert _bytes(rows.v_) == _bytes(oracle.v_)


def test_host_step_oracle_loss_equals_reference_inference_loss(monkeypatch):
    """The composed step with the serial oracle loss (gemm_host_rows head,
    ce_forward_oracle) against the reference inference forward (gemm_oracle
    head, ce_forward_oracle): the two differ only in the head GEMM call."""
    ids = batches()[0]
    monkeypatch.setenv(STEP_ROWS_ENV, '0')
    host = _host_trainer()
    reference = _host_inference(threaded=False)
    assert int(host.train_step(ids)) == int(reference.loss_bits(ids))


def test_host_and_device_give_the_same_loss_words():
    """Forward only: the CPU column's loss against the device's evaluation
    before any step (the first assertion of
    test_host_and_device_give_the_same_words on its own)."""
    ids = batches()[0]
    host = _host_trainer()
    device = Trainer(initial(), data_schedule={'dataset': 'route-test'}, resident=True)
    try:
        assert words([host.loss(ids)]) == words([device.evaluate(ids)])
    finally:
        device.close()
