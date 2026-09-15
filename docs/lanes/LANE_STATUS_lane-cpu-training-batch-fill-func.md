# LANE STATUS: lane/cpu-training-batch-fill-func

Goal: fill the `batch` part of tools/identity_break.py for the function,
tokenizer, optimizer and byte LM trainer lanes wherever the public API has an
axis whose elements may not read their neighbors, and name the missing method
where it does not.

## Filled
- bootstrap: replicate r of `distribution` through resample's `r_first` handle,
  all six statistics, in windows of two (one replicate refuses by name,
  resample/checks/intervals.mojo:355).
- permutation-test: permutation r of `null_distribution` through `r_first`.
- par-resample: the same calls through parallel_classical on `_par_devices()`
  (the mean and quantile arms and the permutation null). Metal only on the M4;
  the lane has no CPU driver and refuses at the lane on CPU, as before.
- metrics-classification: silhouette_samples at chunksize equal to the window
  rows over the whole sample (the axis is cuML's chunk, not a row subset).
- cross-val: three explicit folds passed as cv index pairs, each alone.
- optim-sgd, optim-adam-clip: rows of one (64, 8) parameter tensor through
  fresh SGD, Adam and AdamW step_accumulated, no clip; parameters and both
  moment buffers compared.

## Still N/A (reason in the declaration)
metrics (scalar reductions), monte-carlo (folded scalars only), tokenizer (no
list entry), byte-lm-host-train (mean-reduction step; batch-invariance-2's
batchgrad records the same reason), par-byte-lm, par-byte-lm-model-pool,
par-byte-lm-offload (train_step, state_dict, export_gradients, checkpoint only).

## Evidence (M4, one core, shared machine)
Metal: the 23-binding identical set from the batch2 worktree (the merge changed
only comments in resample, training, metrics, gbdt and core). CPU: resample,
training, metrics, gbdt and core host bindings built from the tree with -j 1.
Lanes: metrics, metrics-classification, cross-val, bootstrap, permutation-test,
monte-carlo, par-resample, optim-sgd, optim-adam-clip.

- Branch commit, fixtures base, ties, denormal, dupes, odd, `--repeats 2`:
  Metal 35 batch hashes, CPU the same 30 (par-resample refuses on CPU).
  `--diff` Metal CPU: summary (batch) IDENTICAL=30, N/A=10, ONE-COLUMN=5.
  With the 166-lane record's three columns: summary IDENTICAL=72, ONE-COLUMN=5;
  summary (batch) IDENTICAL=30, N/A=42, ONE-COLUMN=5. Sabotage
  (`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, `--repeats 1`): 35 of 35 Metal and
  30 of 30 CPU batch cells BATCH_MOVED.
- Merged head (8d4084437), fixtures base, denormal, odd, host bindings
  rebuilt: every batch hash equal to the branch commit's; `--diff` Metal CPU:
  summary IDENTICAL=24, ONE-COLUMN=3, summary (batch) IDENTICAL=18, N/A=6,
  ONE-COLUMN=3; with the record's three columns: summary IDENTICAL=72,
  ONE-COLUMN=3, summary (batch) IDENTICAL=18, N/A=54, ONE-COLUMN=3. Sabotage:
  21 of 21 Metal and 18 of 18 CPU BATCH_MOVED.
- Second merge head (f591d0341, origin/main with KMeans.predict, which
  changed cluster.py and the core host binding), fixture base, core rebuilt:
  the six Metal and six CPU batch hashes equal the earlier ones (par-resample
  Metal equal too); record diff summary IDENTICAL=72, ONE-COLUMN=1, summary
  (batch) IDENTICAL=6, N/A=66, ONE-COLUMN=1; sabotage 7 of 7 Metal and 6 of 6
  CPU BATCH_MOVED.
- Two more merges of origin/main before the push. The first (small-gaps,
  metrics.fowlkes_mallows_score) conflicted in the declaration list; the
  resolution keeps this branch's declarations and gives
  metrics-fowlkes-mallows main's own reason. The second (par-wave2, which
  changed the core host binding and _parallel_pool.py) merged cleanly. At
  246e9851b, base fixture, metrics and core host rebuilt, CPU copy resynced:
  179 lanes, one batch declaration each; the 7 Metal and 6 CPU batch hashes
  equal every earlier head's; record diff summary IDENTICAL=72, ONE-COLUMN=1,
  summary (batch) IDENTICAL=6, N/A=66, ONE-COLUMN=1, no DIVERGENT row;
  sabotage 7 of 7 Metal and 6 of 6 CPU BATCH_MOVED.
- docs_facts --check and wheel_ci.py pins . pass. The gate's one pinned batch
  summary diffs the committed GPU JSONs, which this lane does not change.

## Owed
- NVIDIA and AMD batch cells for the filled lanes, and par-resample on two
  devices: the next release record. No box was rented.
