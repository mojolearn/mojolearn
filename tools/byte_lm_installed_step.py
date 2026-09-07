#!/usr/bin/env python3
"""Remote installed-wheel one-step functional witness; invoked by the serial qualifier."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from mojolearn.language_model import SmallByteLanguageModelTrainer


def main():
    out = Path(os.environ['MOJOLEARN_INSTALLED_RECORD']).parent
    parameters = {}
    for row in SmallByteLanguageModelTrainer.parameter_registry():
        values = ((np.arange(row['size'], dtype=np.float32) % 17) - 8) * np.float32(0.002)
        if 'norm' in row['name'] and row['name'].endswith('_w'):
            values.fill(1)
        parameters[row['name']] = values.reshape(row['shape'])
    trainer = SmallByteLanguageModelTrainer(parameters, data_schedule={'fixture': 'installed-one-step-v1'})
    metadata = trainer.run_metadata()
    assert metadata['native_vendor'] == os.environ['MOJOLEARN_EXPECT_VENDOR']
    assert metadata['native_numeric_mode'] == 1
    ids = np.arange(66, dtype=np.int32).reshape(2, 33)
    before = out / 'byte-lm-before.json'
    after = out / 'byte-lm-after.json'
    restored = out / 'byte-lm-restored.json'
    trainer.save_checkpoint(before)
    result = trainer.train_step(ids)
    trainer.save_checkpoint(after)
    clone = SmallByteLanguageModelTrainer.from_checkpoint(after)
    evaluation = clone.evaluate(ids)
    clone.save_checkpoint(restored)
    assert after.read_bytes() == restored.read_bytes(), 'Restore/evaluation changed full state'
    report = dict(schema='mojolearn.installed-byte-lm-step.v1', status='PASS',
        metadata=metadata, loss=result['loss'], evaluation_loss=evaluation,
        completed_steps=result['completed_steps'],
        gradients_hex=np.asarray(result['flat_gradients'], dtype='<f4').tobytes().hex(),
        checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (before, after, restored)},
        scope='One installed IDENTICAL forward/backward AdamW step and checkpoint restore; no 128-step or cross-vendor claim')
    (out / 'byte-lm-identical.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
