"""Fixed short byte-language-model task and recovery fixtures; test code only."""
from support import binding_check, consumed

def train_case(args, *, resident=True, vocabulary=257, logits=False):
    import numpy as np
    from mojolearn import LanguageModelConfig as Shape, LanguageModelTrainer as Trainer
    from mojolearn import _mojolearn_byte_lm as binding
    shape = Shape(2, 7, 24, 3, 1, 8, 40, 2, vocabulary)
    rng = np.random.default_rng(738)
    parameters = rng.normal(0,.03,shape.n_total).astype('float32')
    for entry in Trainer.parameter_registry(shape):
        if 'norm' in entry['name']:
            parameters[entry['offset']:entry['offset']+entry['size']] += np.float32(1)
    # A fixed next-token progression produces a learnable task, including tails.
    batches = [np.asarray([[((step * 14 + row * 7 + t) * 3) % vocabulary for t in range(8)]
                          for row in range(2)],dtype='int32') for step in range(12)]
    heldout = batches[0].copy()
    trainer = Trainer(parameters, shape=shape, resident=resident,
                      step_result='lean' if resident else 'full', data_schedule={'fixture':'frozen-12-step'})
    initial = float(trainer.evaluate(heldout))
    curve, times = [], []
    for step, batch in enumerate(batches):
        result, elapsed = consumed(lambda: float(trainer.train_step(batch)['loss']))
        curve.append(float(result)); times.append(elapsed)
        assert trainer.state_dict()['completed_steps'] == step+1
        if step == 3:
            saved = trainer.export_state()
            trainer.load_state_dict(saved)  # swap handles; next call must refresh views
        if step == 6:
            before = trainer.state_dict()['completed_steps']
            bad = batch.copy(); bad[0,0] = vocabulary
            try: trainer.train_step(bad)
            except (ValueError, RuntimeError): pass
            else: raise AssertionError('invalid IDs accepted')
            assert trainer.state_dict()['completed_steps'] == before
    final = float(trainer.evaluate(heldout))
    if logits:
        output, logits_ms = consumed(lambda: trainer.logits(heldout[:,:-1]))
        assert np.asarray(output).shape == (shape.batch, shape.length, vocabulary)
    else: logits_ms = None
    resumed = Trainer(parameters, shape=shape, resident=resident,
                      step_result='lean' if resident else 'full', data_schedule={'fixture':'frozen-12-step'})
    resumed.load_state_dict(trainer.export_state())
    resumed_loss = float(resumed.evaluate(heldout))
    # This check admits normal FAST rounding but catches stale parameter views.
    assert abs(resumed_loss-final) <= max(1e-5,abs(final)*1e-4)
    trainer.close(); resumed.close()
    return binding_check(binding,'byte_lm'), dict(contract=dict(shape=[2,7,24,3,1,8,40,2,vocabulary], steps=12, seed=738),
        metrics=dict(heldout_loss=dict(value=final,rtol=1e-3,atol=1e-5),
                     resume_error=dict(value=abs(resumed_loss-final),rtol=.1,atol=1e-5)),
        initial_loss=initial, learning_curve=curve, train_step_ms=times, logits_ms=logits_ms,
        resident=resident, checkpoints=1, rejected_update=1)
