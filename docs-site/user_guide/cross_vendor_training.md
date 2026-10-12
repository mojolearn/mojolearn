# Moving training between vendors

In `identical` mode, a training step gives the same bits on NVIDIA and AMD GPUs for the same release,
input bytes, model state, settings and supported configuration. Apple GPU and CPU results are recorded
beside them. So you can stop a language model run on one vendor's GPU and resume it from its checkpoint,
optimizer state included, on another vendor's GPU. See [Numeric modes](numeric_modes.md).

## Checkpoint handoff

A binary checkpoint holds the parameters, both AdamW moments, the per-tensor flags, the step count and
your data schedule. Write it on one machine and restore it on another:

```python
import mojolearn

# On the first machine (for example, an NVIDIA GPU)
sha = trainer.export_checkpoint_binary("step-1000.blm")   # returns the file's SHA-256

# On the second machine (for example, an AMD GPU), with the same mojolearn release
trainer = mojolearn.LanguageModelTrainer.from_checkpoint_binary("step-1000.blm")
result = trainer.train_step(next_ids)                     # the batch for step 1,001
```

The trainer does not fetch data. Feed the same batches in the same order after the handoff, and record
how in `data_schedule`, which is stored in the checkpoint.

Bits may change between releases, so resume a run with the release that wrote its checkpoint.
