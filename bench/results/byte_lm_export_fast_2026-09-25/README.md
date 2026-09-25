# Byte-LM state and gradient export into the caller's memory, 2026-09-25

Branch `lane/byte-lm-export-fast`, DEVIATION 3120. The GPT-3 Small runner
reads back the whole state (parameters, m, v, flags: 1.95 GB at
162,147,840 parameters) and the summed gradient (648.6 MB) after every
optimizer step to hash them. `bench/results/lm_segment_hash_overlap_2026-09-25`
(on `lane/lm-segment-hash-overlap`) measured that read-back as most of the
line's `hash_seconds` on an H100 host.

## What changed

- **Binding** (`bindings/_mojolearn_byte_lm.mojo`,
  `training/checks/train_loop.mojo`, `training/byte_lm_parallel.mojo`).
  `byte_lm_parallel_export` (state and gradient), `byte_lm_parallel_fold_export`
  and `byte_lm_parallel_shard_gradient` used `download_f32` (a fresh pinned
  host buffer, a device-to-host copy into it, an element loop appending into
  a Mojo `List`) and then `copy_f32` of that List into the caller's array.
  Each array now takes one device-to-host transfer straight into the
  caller's memory (`download_f32_into`, the same `enqueue_copy` to a caller
  pointer that `cluster/estimator.mojo` uses under DEVIATION 2672 on every
  vendor). The pooled m and v land at each owner's `optimizer_first`, which
  is where the old List concatenation put them; the binding checks that the
  owners' ranges are contiguous and cover the model before it copies
  anything. The flags keep their small host loop. On a raise, the caller's
  arrays hold an unspecified partial export.
- **Package** (`python/mojolearn/parallel_training.py`). `export_raw(into=)`,
  `export_gradients(into=)` and `fold_export(into=)` write caller-owned
  buffers in place and return them. Nothing is allocated. Each buffer must be
  writable, C-contiguous and one-dimensional, with the exact dtype
  (`<f4`, or `<i4` for flags) and exactly `n_total` (or `n_tensors`)
  elements. Anything else is refused before the binding is called. The
  no-argument forms are unchanged.
- **Runner** (`tools/lm_segment.py run`). `ExportBuffers` allocates the five
  read-back arrays once and passes them back through `into=` every step. A
  package without `into=` (every published wheel up to 0.8.18) gets the old
  calls.

## Measured (one H100, same box, before and after)

RunPod pod `ko6m6qc11l8ngl`: NVIDIA H100 80GB HBM3, driver 580.126.09,
Intel Xeon Platinum 8480+ (224 CPUs, SHA-NI), Python from the checkout's
pixi environment. Both bindings were built on the box from source
(`bindings/build_byte_lm.sh`, `MOJOLEARN_GPU_ARCHS=sm_90a`,
`MOJOLEARN_NUMERIC_MODE=identical`, column nvidia). OLD is this commit with
`tools/byte_lm_export_fast/change.patch` reversed. Its five files' sha256
equal origin/main's (`h100/files.old.sha256`; for example the binding
`e969c5f1455e...` and the runner `7f649b2ddadf...`). NEW is commit
`c29820729`. The timer `tools/byte_lm_export_fast/box_export.py` takes step
101 from `runs/t3/2026-09-22/A/1/ckpt_00000100.blm` (sha256
`80cd2126a89ba6d8...`) and then times each export three times on that state
(`h100/export_old.json`, `h100/export_new.json`). The values below are
medians.

| seconds | OLD | NEW |
|---|---|---|
| state, `export_raw()` (allocates four arrays) | 5.89 | 1.26 |
| state, binding call into buffers allocated once | 4.17 | 0.152 |
| state, `export_raw(into=...)` | (no `into=`) | **0.144** |
| gradient, `export_gradients()` (allocates) | 1.43 | 0.408 |
| gradient, binding call into a buffer allocated once | 0.900 | 0.052 |
| gradient, `export_gradients(into=...)` | (no `into=`) | **0.052** |
| allocating the four state arrays alone | 1.13 | 1.11 |

A reusing caller now pays 0.144 s for 1.95 GB (about 13.5 GB/s) and 0.052 s
for 648.6 MB (about 12.5 GB/s), where it paid about 4.1 s and 0.90 s. The
no-argument forms now cost mostly the allocation of fresh zero-filled arrays.

**The runner's `hash_seconds`** comes from the replays below. OLD is
origin/main's runner with the OLD binding (fresh arrays every step). NEW is
this branch's runner with the NEW binding (reused buffers):

| step | OLD `hash_seconds` | NEW `hash_seconds` | step seconds OLD / NEW |
|---|---|---|---|
| 101 | 7.148 | 1.821 | 39.37 / 39.87 |
| 102 | 5.932 | 0.580 | 30.66 / 30.65 |
| 103 | 7.163 | 0.949 | 30.68 / 30.65 |

Step 101 in each run includes first-call warmup and, for NEW, the one
allocation of the reused buffers. Steps 102 and 103 are the steady state:
about 6.5 s of the line's cost became about 0.6 to 0.95 s. That remainder is
the read-back (about 0.2 s) plus the digests the runner on main takes
serially (about 0.34 s measured by the hash-overlap lane). The step itself
is unchanged: 30.65 s both ways.

## Digest equality

- **Every exported array, OLD against NEW binding, same state** (after step
  101, each process started from the same checkpoint). The plain sha256 of
  each array is equal between the two bindings. Within each run, the
  fresh, direct and `into=` forms also agree on all three repeats
  (`forms_agree: true`):

  | array | sha256 (both bindings) |
  |---|---|
  | parameters | `189a49eee383450f...` |
  | m | `ceaca18311d713f0...` |
  | v | `ef5df7ac3c40c184...` |
  | flags | `360d579dbd14759b...` |
  | gradient | `5caaccdbf79eec9b...` |

  Full digests are in `h100/export_{old,new}.json`. The chain digests
  (`sliced-sha256-8.v2`) of those exports are state `abc8b816b5c3fb15...` and
  gradient `25830bfc2016dc14...` from both bindings. Both equal route A
  segment 1's line for step 101 (`equals_expected: true`).
- **Replay 101 to 103** (`tools/lm_segment.py run --no-checkpoints
  --expect-chain` against the read-only witness
  `~/mojolearn-evidence/gpt3-run/witness/A-1.chain.partial.jsonl`, sha256
  in `h100/expect_chain.sha256`): **PASS** with the NEW binding and runner,
  and PASS with the OLD ones. Both give state 101 `abc8b816...`,
  102 `a9421f91...`, 103 `fcdb48b8...` and gradients `25830bfc...`,
  `94c40a6e...`, `19a43804...`, plus the same losses and learning-rate bits
  (`h100/replay-{old,new}/`).
- **`--control split`, one step, NEW**: PASS against step 101
  (`h100/split-new/`). This path uses `byte_lm_parallel_shard_gradient`,
  `byte_lm_parallel_fold_export` and the non-pooled export (m and v in one
  piece), all three changed here.

## Unit tests (CPU, this Mac)

`tools/tests/test_parallel_export_into.py` (26 tests, a stand-in binding
that writes known bytes at the addresses it receives). They cover:

- the `into=` forms give the same bytes as the no-argument forms, for NumPy
  arrays and the package's own `Array`;
- the same objects come back, and a reused buffer is overwritten by the
  next export;
- an `into=` export allocates nothing;
- refusals before the binding is called: wrong dtype (float64, int32 for
  float, float32 or int64 for flags, bytearray), one element short or long,
  flags sized like parameters, two-dimensional, strided, read-only, `bytes`,
  a non-buffer, a list, a missing or extra key;
- one buffer passed for two arrays reaches the overlap refusal;
- the runner's `ExportBuffers` reuses its buffers and gives the same v1 and
  v2 digests as fresh exports, and falls back when the package has no
  `into=`.

The existing lm_segment, controls, witness and driver tests pass. Their
NumPy stand-in trainer has no `into=`, so they exercise the fallback.

## Owed before a release

- **AMD.** `bindings/_mojolearn_byte_lm.mojo` and
  `training/checks/train_loop.mojo` are shared by every column. The AMD
  (gfx942) binding therefore runs the new copy too, and the replay proof on
  an AMD box (101 to 103 held to the chain, and a two-device pooled export,
  where m and v come from two contexts) is owed before a release. It was not
  run here: this lane rents no Hot Aisle or DigitalOcean box while T3 runs.
- **Apple.** The Metal binding compiles the same source. It was not built on
  this Mac (no Mojo builds here). The same `enqueue_copy` to a caller
  pointer already ships on Metal in `cluster/estimator.mojo`
  (DEVIATION 2672), but this binding's Apple compile and its export digests
  are owed.
- **Two devices.** The pooled multi-device export (each owner's range copied
  from its own context) has not run on hardware. The one-device H100 run
  covers only a single owner (pooled, one device) and the non-pooled split
  path.
- `lane/lm-segment-hash-overlap`'s `Snapshot` calls the binding directly
  into reused buffers, so it gets the binding change without the runner
  change. Its merge will conflict with `ExportBuffers` in the synchronous
  loop, which it replaces.
- The model-pool and offload exports (`byte_lm_model_pool_export`,
  `byte_lm_offload_export`) still stage through Lists. They are not on the
  T3 path.

## Cost

Two RunPod H100 80GB HBM3 legs at about $3.49 an hour:

- Leg 1, pod `m4xan0j62h7wud`, 11:45:02 to about 11:48:45 ET (about
  4 minutes). The body found no `change.patch` because the leg archive drops
  every `results/` directory. It stopped before any build
  (`h100/leg1_status.txt`). The scripts moved to `tools/byte_lm_export_fast/`.
- Leg 2, pod `ko6m6qc11l8ngl`, 11:49:35 to about 12:08:40 ET (about
  19 minutes).

Both pods were deleted and confirmed gone (HTTP 404; `h100/teardown.txt`,
`h100/leg1_teardown.txt`). The total is about 23 minutes, or about $1.35.

## Files

`tools/byte_lm_export_fast/leg.sh` (the box body), `box_export.py` (the
timer) and `change.patch` (this lane's change to the five files, reversed on
the box to make OLD). Under `h100/`:

- `status.txt` (the box's timeline)
- `export_{old,new}.json`
- `replay-{old,new}/` and `split-new/` (chain, segment.json, log)
- `files.{old,new,new_again}.sha256` (the trees)
- `byte_lm.{old,new}.so.sha256`
- the input digests
- `host.txt`, `gpu.txt`, `mojo_version.txt`, `leg.txt`

The full leg directories stay in `~/mojolearn-evidence/byte-lm-export-fast/`.
