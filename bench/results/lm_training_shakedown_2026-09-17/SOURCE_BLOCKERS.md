# LM training shakedown: source-level blockers found before any GPU run

Lane `lane/lm-training-shakedown`, 2026-09-17. Read at `origin/main` 372609f8e.
These are READ FROM THE CODE, not measured. The measured arms (batch sweep,
10,000 steps, resume at 162M) are separate files in this directory.

Target shape under consideration: batch 1, length 2048, d_model 768, 12 layers,
12 heads, head_dim 64, intermediate 2048, vocab 50257, 162,147,840 parameters.

## A. The step counter is capped at 999,999, in every layer

`completed_steps` is refused at or above 1,000,000 by the host, the binding and
the Mojo trainer alike. It is not one guard that could be relaxed in one place:

| file | line | guard |
|---|---|---|
| `python/mojolearn/_byte_lm_impl.py` | 219 | `if not 0 <= value <= 999999:` in `_step`, called twice by `_validate_state` |
| `python/mojolearn/_byte_lm_impl.py` | 765 | stateless path: `if train and working['completed_steps'] >= 999999:` |
| `python/mojolearn/_byte_lm_impl.py` | 843 | resident path: `if train and self._state['completed_steps'] >= 999999:` |
| `bindings/_mojolearn_byte_lm.mojo` | 340 | `completed >= 1000000 or (action == 1 and completed >= 999999)` |
| `bindings/_mojolearn_byte_lm.mojo` | 592 | `completed < 0 or completed >= 1000000` |
| `bindings/_mojolearn_byte_lm.mojo` | 807 | session step: `if completed >= 999999:` |
| `bindings/_mojolearn_byte_lm.mojo` | 1188 | `claimed < 0 or claimed >= 1000000` |
| `training/byte_lm.mojo` | 230, 261, 653 | `raise Error("byte LM: completed step must be in [0,1000000)")` |

What that ceiling buys at L2048:

| batch | tokens/step | max tokens in 999,999 steps |
|---|---|---|
| 1 | 2,048 | 2.048 B |
| 2 | 4,096 | 4.096 B |
| 4 | 8,192 | 8.192 B |
| 5 | 10,240 | 10.24 B |
| 13 | 26,624 | 26.6 B |

So FineWeb-Edu 10BT needs batch >= 5 and a 25B-token run needs batch >= 13,
PURELY to stay under the step counter. Batch 1 cannot reach either target no
matter how long the box is rented.

## B. The checkpoint file path refuses this model by 1,855x

`_byte_lm_impl.py:70` sets `_CHECKPOINT_LIMIT = 2 * 1024 * 1024`, and
`export_checkpoint` refuses before it even copies state:

    if state_shape(self._state).n_total * 24 + state_shape(self._state).n_tensors * 8 > _CHECKPOINT_LIMIT:
        raise ValueError('Byte-LM checkpoint exceeds 2 MiB; export state_dict arrays')

24 bytes per parameter is the JSON/hex envelope, so the largest model
`save_checkpoint` / `export_checkpoint` / `from_checkpoint` will ever accept is
**87,381 parameters**. The 34,944-parameter model that passed cross-vendor
resume fits; 162,147,840 parameters is 1,855x over the bound. The docstring
names the intended replacement ("at larger shapes `export_state()` arrays are
the checkpoint"), but no harness in the repository does that today, and the
`from_checkpoint` restore path is not reachable at this size.

## C. `export_state()` costs about 16 s of per-element Python at this size

`_validate_state` (`_byte_lm_impl.py:328`) ends with

    if any(x < 0 for x in flat_view(v, 'f')) or any(x not in (0, 1) for x in flat_view(flags, 'i')):

`flat_view` returns a memoryview (`_bufcheck.py:163`), so the first `any` is a
Python-level loop over all 162,147,840 second moments. Measured on one core of
the M4 by extrapolating a synthetic `array.array` scan (linear by construction):
0.1971 s at 2,000,000 elements and 0.8061 s at 8,000,000, which is 15.98 s and
16.34 s at 162,147,840. `_validate_state` runs on every `export_state()`, every
`load_state_dict()` and every stateless `train_step`. Against a 0.2326 s step
that is a 70x per-step tax on the stateless path and a fixed ~16 s per
checkpoint on the resident path. `all_finite` is native and is not part of this.

## D. Gradient accumulation exists, but only through a second trainer class

CORRECTED 2026-09-17, after this file first claimed there was none.
`SmallByteLanguageModelTrainer.train_step` is forward + backward + AdamW update
in one call and has no accumulation of its own, which is what the first reading
saw. But `ParallelByteLanguageModelTrainer`
(`python/mojolearn/parallel_training.py:89`) takes `logical_shards`
microbatches per `train_step`, sums their gradients in a fixed order and
advances the optimizer exactly ONCE:

    self._state['completed_steps'] = before + 1

`logical_shards` is admitted in [1, 1024] and `devices=(0,)` replays every
shard on ONE GPU. `training/byte_lm_parallel.mojo:117` builds one `ByteTrainer`
per DEVICE, not per shard, plus two `n_total` float32 accumulator buffers, so
on one device the extra cost over the plain path should be about 1.30 GB and
nothing that scales with K.

That is the way past blocker A. At batch 1, length 2048:

| logical shards | tokens per optimizer step | steps for 10B | steps for 25B |
|---|---|---|---|
| 1 | 2,048 | 4,882,813 (over the cap) | 12,207,032 (over the cap) |
| 4 | 8,192 | 1,220,704 (over the cap) | 3,051,758 (over the cap) |
| 8 | 16,384 | 610,352 | 1,525,879 (over the cap) |
| 16 | 32,768 | 305,176 | 762,940 |
| 64 | 131,072 | 76,294 | 190,735 |

So logical_shards >= 8 clears the step cap for 10BT and >= 16 clears it for
25B, without needing a batch that fits in device memory.

THREE THINGS ARE UNMEASURED AND ONE IS A SEMANTIC CHANGE:

  * this class has only ever run at toy shapes (the checks in
    `tools/byte_lm_parallel_check.py` and its siblings use
    `Shape(batch=2, length=7, d_model=24, ..., n_layers=3, vocab_size=256)`).
    Nothing has run it at 162M;
  * whether the device peak really stays flat in K on one device is a
    prediction from reading the allocation, not a measurement;
  * whether K microbatches per optimizer step BEATS K separate steps on
    throughput is unmeasured. It should, because the AdamW update over
    162,147,840 parameters is a fixed per-step cost that K shards amortize,
    but that is an argument, not a number;
  * the reduction is an ordered SUM of per-shard mean cross-entropies, not a
    mean over the effective batch (`parallel_training.py:5`). The gradient is
    therefore K times a single shard's, which is a learning-rate decision for
    a real run. It does not affect determinism.

`tools/lm_shards_probe.py` in this lane measures the first three.

## E. Signed 32-bit span check caps batch at 20 at this shape

`_byte_lm_config.py:41` refuses any shape whose largest span exceeds
2,147,483,647. At L2048 V50257 the binding span is `b * l * vocab`
= b * 102,926,336, so batch 20 is the last admitted batch, and `b * h * l * l`
= b * 50,331,648 admits 42. This is not the binding constraint (device memory
is), but batch 13 from blocker A is inside it and batch 21 is not.

## F. The data pipeline reads the whole corpus into host RAM and has no tokenizer

`tools/lm_step_memory_probe.py:229` `CorpusBatches`:

  * requires ONE file plus a `manifest.json` beside it with schema
    `mojolearn.byte-lm.corpus.v1`, its sha256 and byte length;
  * does `raw = self.path.read_bytes()` -- the entire corpus resident in host
    memory, then `np.frombuffer`;
  * slices `bytes[(k*batch*length + b*length) % (n - length - 1) : +length+1]`
    and casts to int32. Raw BYTES are the token ids. There is no tokenizer in
    the path at all.

Consequences at the target shape:

  * FineWeb-Edu 10BT is in R2 as 14 parquet shards, 28,518,193,415 bytes
    (`tools/dataset_store.sh:53,98`), described there as "the corpus for a real
    LM run". Nothing in the repository reads parquet into the trainer, and no
    `manifest.json` of this schema exists for it.
  * With byte ids, only rows 0-255 of the 50,257-row embedding and lm_head ever
    receive a gradient. 99.5% of the two largest tensors (77.2M of the
    162.1M parameters) would be trained on nothing. A 50,257-vocab run needs a
    real tokenizer path; `python/mojolearn/_bpe_trainer.py` and `tokenizer/`
    exist and ship no vocabulary, so the vocabulary is itself an owed artifact.
  * enwik8 is 100,000,000 bytes and pile_github 97,124,565. A 2.048B-token
    batch-1 run at the step cap wraps enwik8 about 20 times.

## G. What has actually been run at this shape

Census of every `result.json` under `bench/results/` carrying an LM `shape`:

  * two shapes only: `(1, 2048, 384, 8, 8192)` 4 records and
    `(1, 2048, 768, 12, 50257)` 413 records. **Every LM run on record is
    batch 1.** Batch 2 and above has never been executed at any transformer LM
    shape, so it is untested, not merely unmeasured.
  * the largest `steps_completed` at the 162M target shape is **4**
    (`bench/results/e1g/2026-09-12_133013-amd-mi300x-hotaisle-step-glue/` and
    `2026-09-11_215139-nvidia-h100-80gb-hbm3-step-glue/`). The 128-step
    continuous runs in `docs/LM_TRAINING_CLAIM_PLAN.md` are the 34,944-parameter
    two-block model, a different profile.

## H. The batch-1 memory and speed numbers we already hold

From `tools/lm_step_memory_probe.py` result.json files, all H100, all target
shape, all batch 1:

| leg | mode | median step s | device peak | host RSS |
|---|---|---|---|---|
| `2026-09-10_230820-nvidia/lm-step-memory/target` | stateless, full | 44.975 | 12.14 GB | 21.93 GB |
| `2026-09-10_233303-nvidia/lm-step-memory/target` | stateless, full | 38.930 | 12.13 GB proc / 12.14 GB wide | 22.34 GB |
| `2026-09-11_004220-nvidia/lm-step-memory/target` | resident, full | 8.589 | 14.29 GB wide | 8.86 GB |
| `2026-09-11_004220-nvidia/lm-step-memory/target-lean` | resident, lean | 0.5631 | 14.29 GB wide | 8.17 GB |
| `2026-09-12_133007-nvidia-h100-owed-rest/attention-step/...-enwik8` | resident, lean | **0.2326** | **16.96 GB** proc / 16.97 GB wide | 8.31 GB |

The 0.2326 s cell quoted in `bench/OPPONENT_REFERENCE.md` is that last row: the
same probe, resident + lean, enwik8, and only THREE steps
(`steady_step_seconds = [0.23427, 0.23099]`). Device peak at batch 1 is
16.96 GB, not the 12.14 GB of the older stateless run: the faster attention arms
stash more. The `bench/results/lm_capacity_2026-09-10/*.json` files are labelled
"host arithmetic only; not measured memory or throughput" and their
`fit_admitted: false` is an analytic upper bound, not an OOM observation.

Subtracting the fixed 2.594 GB of parameters + gradient + m + v leaves about
14.4 GB that scales with batch at the current arm. Straight-line extrapolation
predicts batch 2 near 31 GB, batch 4 near 60 GB and batch 8 near 117 GB, so an
80 GB H100 would cap somewhere around batch 4 or 5. **That is a prediction from
one point, not a measurement**, and it is exactly what the batch sweep in this
lane exists to replace.
