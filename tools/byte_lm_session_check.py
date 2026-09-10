#!/usr/bin/env python3
"""Native session lifetime and failure controls; small correctness fixtures only."""
import argparse
import json
import os


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
    resident = Trainer(parameters, shape=shape, resident=True, data_schedule={'fixture': 'lifecycle'})
    control = Trainer(parameters, shape=shape, data_schedule={'fixture': 'lifecycle'})
    ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
    checks = []

    def same_state(a, b):
        for key in ('parameters', 'm', 'v', 'flags'):
            assert a[key].tobytes() == b[key].tobytes(), key
        assert a['completed_steps'] == b['completed_steps']

    def pair(label):
        left, right = resident.train_step(ids), control.train_step(ids)
        assert left['flat_gradients'].tobytes() == right['flat_gradients'].tobytes()
        assert np.float32(left['loss']).tobytes() == np.float32(right['loss']).tobytes()
        same_state(resident.state_dict(), control.state_dict())
        checks.append(label)

    pair('initial step matches reconstruction')
    saved = resident.state_dict()
    owner = resident._native_session
    resident.evaluate(ids)
    assert resident._native_session is owner
    pair('repeated call reuses owner and matches reconstruction')
    resident.close()
    resident.close()
    assert resident._native_session is None
    pair('close and lazy reconstruction resume exactly')
    assert resident._native_session is not owner
    resident.load_state_dict(saved)
    control.load_state_dict(saved)
    pair('restore invalidates resident state and resumes exactly')

    # Sabotage the supplied host mirror. A stateless implementation would
    # silently train the altered model; resident admission must detect it.
    saved = resident.state_dict()
    resident._state['parameters'][0] += np.float32(.25)
    try:
        resident.train_step(ids)
    except Exception as error:
        assert 'state' in str(error), str(error)
    else:
        raise AssertionError('resident-state mismatch control did not fire')
    assert resident._native_session is None
    resident.load_state_dict(saved)
    checks.append('resident mismatch control detected; failed session discarded')

    # Fail Python validation AFTER a real native update. The advanced device
    # state must be discarded and the last successful host state retained.
    original = impl._array
    def reject_gradient(value, shape, name, dtype=np.float32):
        if name == 'pre-update gradients':
            raise RuntimeError('post-native validation sabotage')
        return original(value, shape, name, dtype)
    impl._array = reject_gradient
    try:
        try:
            resident.train_step(ids)
        except RuntimeError as error:
            assert 'sabotage' in str(error)
        else:
            raise AssertionError('post-native validation control did not fire')
    finally:
        impl._array = original
    assert resident._native_session is None
    same_state(saved, resident.state_dict())
    pair('post-native failure recovers from last committed state')

    # Changed registry/profile after restore must discard old device buffers.
    other_shape = Shape(1, 5, 16, 2, 1, 8, 24, 1, 257)
    fresh = Trainer(np.zeros(other_shape.n_total, np.float32), shape=other_shape,
                    data_schedule={'fixture': 'shape-change'})
    resident.load_state_dict(fresh.state_dict())
    assert resident._native_session is None
    other_ids = np.zeros((1, 6), np.int32)
    assert resident.evaluate(other_ids) == fresh.evaluate(other_ids)
    checks.append('shape-changing restore and evaluation')
    edge = Trainer(parameters, shape=shape, resident=True, betas=(0., .999),
                   data_schedule={'fixture': 'signed-zero-config'})
    edge.evaluate(ids)
    edge._state['config']['beta1'] = -0.0
    try:
        edge.evaluate(ids)
    except Exception as error:
        assert 'optimizer mismatch' in str(error), str(error)
    else:
        raise AssertionError('signed-zero optimizer mismatch was ignored')
    assert edge._native_session is None
    checks.append('optimizer admission distinguishes signed zero')
    edge.close()
    resident.close()
    control.close()
    fresh.close()
    print(json.dumps({'passed': True, 'checks': checks,
                      'qualification': 'local lifecycle and exact regression; not timing'}, indent=2))


if __name__ == '__main__':
    main()
