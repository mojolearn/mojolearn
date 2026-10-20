# Moving training between vendors

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. That includes every training step, for the same release, input bytes, model state, settings
and supported configuration. So you can stop a language model run on one vendor's GPU and resume it from its
checkpoint, optimizer state included, on another vendor's GPU. See [Numeric modes](numeric_modes.md).

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

## Sequential models

A `Sequential` checkpoint holds the layers, the loss, optimizer and schedule chosen with `compile`, the
trained parameters, the optimizer state, the loss curve and the counters of the row order and the
optimizer. Save it on one machine, load it on another and continue with `resume=True`: the row orders,
the learning-rate schedule and the optimizer steps carry on, so two epochs here and two more there give
the same parameters as four epochs on either machine.

```python
import mojolearn

# On the first machine (for example, an NVIDIA GPU)
model = mojolearn.Sequential().linear(16).relu().linear(1)
model = model.compile(loss="mse", optimizer="adam", lr=0.01)
model = model.fit(X, y, epochs=2, batch_size=32, seed=0)
model.save_checkpoint("model.ckpt")

# On the second machine (for example, an Apple GPU, or a CPU-only install)
model = mojolearn.Sequential.from_checkpoint("model.ckpt")
model = model.fit(X, y, epochs=2, resume=True)   # the same parameters as fit(X, y, epochs=4)
```

Pass the same training data after the handoff. A custom loop built on `ops.Session` saves its
parameters, its optimizer state and its own seed, epoch and step counters with `Session.save_state`
and restores them with `Session.load_state`.

Bits may change between releases, so resume a run with the release that wrote its checkpoint.
