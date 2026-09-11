#!/usr/bin/env python3
"""Native session lifetime, export and failure controls; small correctness fixtures only.

DEVIATION 2514 (docs/lanes/DESIGN_lm_device_owned_step_2026-09-11.md,
section 7) at fixture A (B2/L7/DM24, 3 layers, V513):

  G1  eight resident `step_result='lean'` steps against the stateless arm:
      loss bits, exported gradient bytes and exported state after every
      step; one `evaluate()` between steps 4 and 5 with `export_state()`
      before and after; a second resident run with NO intermediate exports
      ends in the same state; `close()` at step 3 on the first run and
      continue.
  G2  `export_state()` at step 3 into a NEW trainer (`load_state_dict`, and
      `from_checkpoint_bytes`), continued to 8: equal to the stateless arm
      at every step including the gradient of step 4.
  G3  exported arrays and the ids array mutated after the call: the next
      step equals the stateless arm; `load_state_dict` of the mutated
      export is refused (negative v) and the live session is untouched.
  G4  failed-update controls: the Python post-check fault always; the five
      native faults only when the binding reports
      `byte_lm_fault_inject_available()` (a gate build with
      `-D MOJOLEARN_BYTE_LM_FAULT_INJECT=1`). Each: export before, one
      faulted step raising the named message, export after equal to
      before (flags and completed_steps included), `export_gradients()`
      refused, then the fault off and one good step equal to the arm.
  plus the lifecycle checks this tool carried before: shape-changing
      restore and evaluation; signed-zero optimizer admission (the session
      is now KEPT after an admission refusal, nothing was written).

The stateless arm creates one context per call and the resident runs
create one each, in one process: the DEVIATION 2494 shape. If the
second-context hang is open on the box, run with
MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 (DEVIATION 2513) and say so in the record.
"""
import argparse
import hashlib
import json
import os
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    args = parser.parse_args()
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'identical'
    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    from mojolearn import _byte_lm_impl as impl

    shape = Shape(2, 7, 24, 3, 1, 8, 40, 3, 513)
    rng = np.random.default_rng(67261)
    parameters = rng.normal(0, .05, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            parameters[entry['offset']:entry['offset'] + entry['size']] += np.float32(1)
    schedule = {'fixture': 'lifecycle'}
    n_gate = 8
    n_control = n_gate + 6  # the six G4 controls each end in one good step
    batches = [rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
               for _ in range(n_control)]
    eval_ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
    checks = []

    def f32(x):
        return np.float32(x).tobytes()

    def same_state(a, b):
        for key in ('parameters', 'm', 'v', 'flags'):
            assert a[key].tobytes() == b[key].tobytes(), key
        assert a['completed_steps'] == b['completed_steps']

    def sha(a):
        return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

    def resident_trainer():
        return Trainer(parameters, shape=shape, resident=True, step_result='lean', data_schedule=schedule)

    # The stateless reference arm, run once; every resident run compares
    # against these captures.
    control = Trainer(parameters, shape=shape, data_schedule=schedule)
    control_results, control_states = [], []
    control_eval = None
    for k, batch in enumerate(batches):
        if k == 4:
            control_eval = control.evaluate(eval_ids)
        control_results.append(control.train_step(batch))
        control_states.append(control.state_dict())

    def gradient_of(trainer, result):
        if 'flat_gradients' in result:
            return result['flat_gradients']
        return trainer.export_gradients()['flat_gradients']

    def step_and_compare(trainer, k, exports=True, mutate_ids=False):
        tokens = batches[k].copy() if mutate_ids else batches[k]
        result = trainer.train_step(tokens)
        if mutate_ids:
            tokens[:] = 0
        ref = control_results[k]
        assert result['step'] == result['completed_steps'] == ref['step'] == k + 1
        assert f32(result['loss']) == f32(ref['loss']), 'loss bits differ at step %d' % (k + 1)
        if exports:
            assert gradient_of(trainer, result).tobytes() == ref['flat_gradients'].tobytes(), 'gradient differs at step %d' % (k + 1)
            same_state(trainer.export_state(), control_states[k])
        return result

    # G1, run A: exports after every step, eval between 4 and 5, close at 3.
    first = resident_trainer()
    lean = step_and_compare(first, 0)
    assert set(lean) == {'loss', 'step', 'completed_steps', 'next_batch_index', 'flags'}
    assert all(first._state[key] is None for key in ('parameters', 'm', 'v'))
    owner = first._native_session
    step_and_compare(first, 1)
    assert first._native_session is owner
    step_and_compare(first, 2)
    state_at_3 = first.export_state()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'step3.json')
        first.export_checkpoint(path)
        with open(path, 'rb') as stream:
            checkpoint_at_3 = stream.read()
    first.close()
    first.close()
    assert first._native_session is None
    assert all(first._state[key] is not None for key in ('parameters', 'm', 'v'))
    same_state(first.state_dict(), control_states[2])
    step_and_compare(first, 3)
    assert first._native_session is not None and first._native_session is not owner
    before_eval = first.export_state()
    loss = first.evaluate(eval_ids)
    after_eval = first.export_state()
    same_state(before_eval, after_eval)
    assert f32(loss) == f32(control_eval)
    checks.append('G1: evaluate on the resident state exports equal before and after')
    step_and_compare(first, 4)
    # G3 at step 6: mutate every export and the ids array after the call.
    mutated = first.export_state()
    np.asarray(mutated['parameters'])[0] += np.float32(.25)
    np.asarray(mutated['v'])[0] = np.float32(-1)
    np.asarray(mutated['flags'])[0] = 1
    grads = first.export_gradients()
    np.asarray(grads['flat_gradients'])[0] = 7
    np.asarray(grads['gradients'][shape.parameter_names[1]]).flat[0] = 7
    step_and_compare(first, 5, mutate_ids=True)
    try:
        first.load_state_dict(mutated)
    except ValueError as error:
        assert 'nonnegative' in str(error), str(error)
    else:
        raise AssertionError('mutated export was admitted')
    assert first._native_session is not None
    step_and_compare(first, 6)
    checks.append('G3: export and ids mutation cannot reach the device; mutated import refused, session untouched')
    step_and_compare(first, 7)
    final_a = first.export_state()
    witnesses = {key: sha(final_a[key]) for key in ('parameters', 'm', 'v')}
    witnesses['loss_bits_step8'] = f32(control_results[7]['loss']).hex()
    witnesses['gradient_step8'] = sha(control_results[7]['flat_gradients'])
    checks.append('G1: eight lean steps equal the stateless arm with exports after every step')

    # G1, run B: no intermediate exports; the end state must be the same.
    second = resident_trainer()
    for k in range(n_gate):
        step_and_compare(second, k, exports=False)
    same_state(second.export_state(), control_states[7])
    assert second.export_gradients()['flat_gradients'].tobytes() == control_results[7]['flat_gradients'].tobytes()
    checks.append('G1: a resident run without intermediate exports ends in the same state')

    # G2: continue from the step-3 export in NEW trainers.
    continued = resident_trainer().load_state_dict(state_at_3)
    for k in range(3, n_gate):
        step_and_compare(continued, k)
    from_bytes = Trainer.from_checkpoint_bytes(checkpoint_at_3, resident=True)
    for k in range(3, n_gate):
        step_and_compare(from_bytes, k)
    checks.append('G2: load_state_dict and from_checkpoint_bytes continuations equal the arm through step 8')

    # G4: failed-update controls at step 8, each followed by one good step.
    binding = impl._load(shape)
    faults = [('python_post', 'post-check sabotage')]
    native_faults = [('loss_nonfinite', 'byte LM: nonfinite loss at 0'),
                     ('grad_nonfinite', 'byte LM: nonfinite gradients at 0'),
                     ('opt_refuse', 'optimizer: NaN in exp_avg at flat index 5'),
                     ('after_nonfinite', 'byte LM: nonfinite second moments at 3'),
                     ('after_negative', 'byte LM: negative second moment')]
    fault_inject = bool(callable(getattr(binding, 'byte_lm_fault_inject_available', None))
                        and binding.byte_lm_fault_inject_available())
    if fault_inject:
        faults += native_faults
    original_array = impl._array

    def reject_flags(value, shape_, name, dtype='<f4'):
        if name == 'flags':
            raise RuntimeError('post-check sabotage')
        return original_array(value, shape_, name, dtype)

    k = n_gate
    for name, expected in faults:
        before = second.export_state()
        owner = second._native_session
        if name == 'python_post':
            impl._array = reject_flags
        else:
            os.environ['MOJOLEARN_BYTE_LM_FAULT'] = name
        try:
            try:
                second.train_step(batches[k])
            # The binding raises the native message as a plain Exception (as
            # the stateless path always has); the Python layer's own
            # refusals are RuntimeError. Both are the expected failure here.
            except (RuntimeError, Exception) as error:
                assert expected in str(error), '%s: %s' % (name, error)
            else:
                raise AssertionError('%s: the faulted step did not raise' % name)
        finally:
            impl._array = original_array
            os.environ.pop('MOJOLEARN_BYTE_LM_FAULT', None)
        assert second._native_session is owner, name
        after = second.export_state()
        same_state(before, after)
        try:
            second.export_gradients()
        except RuntimeError as error:
            assert 'no gradient' in str(error), str(error)
        else:
            raise AssertionError('%s: gradient export was not refused after the faulted step' % name)
        step_and_compare(second, k)
        k += 1
        checks.append('G4 %s: rolled back on the device, session kept, next step equals the arm' % name)
    if not fault_inject:
        checks.append('G4 native faults SKIPPED: binding built without MOJOLEARN_BYTE_LM_FAULT_INJECT')

    # Changed registry/profile after restore must discard old device buffers.
    other_shape = Shape(1, 5, 16, 2, 1, 8, 24, 1, 257)
    fresh = Trainer(np.zeros(other_shape.n_total, np.float32), shape=other_shape,
                    data_schedule={'fixture': 'shape-change'})
    first.load_state_dict(fresh.state_dict())
    assert first._native_session is None
    other_ids = np.zeros((1, 6), np.int32)
    assert f32(first.evaluate(other_ids)) == f32(fresh.evaluate(other_ids))
    checks.append('shape-changing restore and evaluation')
    edge = Trainer(parameters, shape=shape, resident=True, betas=(0., .999),
                   data_schedule={'fixture': 'signed-zero-config'})
    edge.evaluate(batches[0])
    owner = edge._native_session
    edge._state['config']['beta1'] = -0.0
    try:
        edge.evaluate(batches[0])
    except Exception as error:
        assert 'optimizer mismatch' in str(error), str(error)
    else:
        raise AssertionError('signed-zero optimizer mismatch was ignored')
    assert edge._native_session is owner
    checks.append('optimizer admission distinguishes signed zero; nothing written, session kept')
    for trainer in (edge, first, second, continued, from_bytes, control, fresh):
        trainer.close()
    print(json.dumps({'passed': True, 'checks': checks, 'witnesses_fixture_a': witnesses,
                      'fault_inject_build': fault_inject,
                      'keep_context': os.environ.get('MOJOLEARN_BYTE_LM_KEEP_CONTEXT') == '1',
                      'qualification': 'local lifecycle and exact regression at fixture A; not timing'}, indent=2))


if __name__ == '__main__':
    main()
