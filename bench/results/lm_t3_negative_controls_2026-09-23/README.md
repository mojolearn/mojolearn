# T3 negative controls at the real shape, 2026-09-23

Plan section 6 item 7 (`docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md`): controls that
must be seen to fail, at the real shape, once. Every run started from route
A's `ckpt_00000100.blm` of the live T3 run (R2 `runs/t3/2026-09-22/A/1/`,
1,945,780,168 bytes, sha256 `80cd2126a89ba6d8...`, read only), ran two
optimizer steps (global 101 and 102) with `tools/lm_segment.py run
--no-checkpoints --expect-chain`, and was held to route A segment 1's own
chain lines 100 to 102 (`expect_chain.jsonl`, taken from the running box's
chain, hash scheme `sliced-sha256-8.v2`). So lm_segment itself read every
verdict. Every run reported that the checkpoint's own state hash equals the
chain's step-100 line (`2b37e85edfd701eb...`) before anything touched it.

Box: one NVIDIA H100 80GB HBM3 on RunPod (`gpu.txt`, driver 580.126.09),
one device, the PUBLISHED wheel mojolearn 0.8.15 in a Python 3.11 venv
(`wheel.sha256` 281677583838358d..., the same file the run installs), tools
from commit 38d7d828d (branch `lane/lm-negative-controls`). The body is
`tools/lm_controls_body.sh`, rendered by `python3 tools/lm_segment_leg.py
controls`; it uploads nothing. Route A's chain was written on a 2-GPU H100
pod; these replays ran on one GPU.

| control | flag | expected | observed | first differing step | fields that differed |
|---|---|---|---|---|---|
| positive (plain replay) | none | PASS | **PASS**, steps 101 and 102 | none | none: state, gradient, 64 losses and lr bits equal at both steps |
| harness check | `--control none` | PASS | **PASS**, steps 101 and 102 | none | none (every line stamped `control: none`) |
| zeroed moments | `--zero-moments` | FAIL | **FAIL** | 101 | `state_sha256` only |
| K=63 | `--control shards=63` | FAIL | **FAIL** | 101 | `state_sha256`, `gradient_sha256`, `losses_f32_hex` |
| two shards swapped | `--control swap=5,40` | FAIL | **FAIL** | 101 | `state_sha256`, `gradient_sha256`, `losses_f32_hex` |
| first two shards swapped | `--control swap=0,1` | losses only | **FAIL on losses only** | 101 | `losses_f32_hex` only |
| split step, no edit (the ulp harness) | `--control split` | PASS | **NOT RUN**: crashed before step 101 | none | see below |
| one ulp in one shard's gradient | `--control ulp=63,auto` | FAIL | **NOT RUN**: crashed before step 101 | none | see below |

Digests at step 101 (expected, from route A's chain: state
`abc8b816b5c3fb15...`, gradient `25830bfc2016dc14...`):

| control | state_sha256 | gradient_sha256 | losses |
|---|---|---|---|
| positive, none | `abc8b816b5c3fb15...` | `25830bfc2016dc14...` | all 64 equal |
| zero-moments | `7bcaf6fc113b58a9...` | `25830bfc2016dc14...` (equal) | all 64 equal |
| shards=63 | `1a77ebc67e783e4f...` | `75d90aeed2ad7a4b...` | 63 values, each equal to the chain's value in the same position |
| swap=5,40 | `c34ac3e30fb0adb9...` | `69052888b6e623d4...` | positions 5 and 40 exchanged, the other 62 equal |
| swap=0,1 | `abc8b816b5c3fb15...` (equal) | `25830bfc2016dc14...` (equal) | positions 0 and 1 exchanged |

Full digests are in each control's `segment.json` (`disagreements`) and
`chain.jsonl`. Step 102 for the passing runs: state `a9421f91b947f82c...`,
gradient `94c40a6e3d5ec5aa...`, equal to the chain.

## What the fields say

- **Zeroed moments** change only the state. The gradient at step 101 depends
  on the parameters alone, which the control leaves as they were, so the
  summed gradient and every loss agree, and the AdamW update from zeroed m
  and v moves the state.
- **K=63** keeps the first 63 of the step's 64 shard batches unchanged, so
  the 63 losses agree position by position. The summed gradient no longer
  contains shard 63, so the gradient and the state differ, and the loss
  vector's length differs.
- **The swap.** The fold is an ordered left fold on the device (total =
  g0, then total = total + g_k in shard order), so its result depends on
  order. Swapping positions 5 and 40 moves the two losses and changes the
  rounding of the sum, so the gradient and the state differ. Swapping
  positions 0 and 1 changes only the losses: g0 + g1 and g1 + g0 are the same
  float32 bits (addition commutes; only the association of the rest of the
  fold matters, and that is unchanged), so the gradient and the state agree
  bit for bit. A swap control has to avoid the first two positions to test
  the fold. The comparison still catches it, on the losses.

## Not run: split and one ulp, and the defect they found

`split` and `ulp` use the live worker's per-shard calls (`shard_gradient_fold`,
`fold_export`, `shard_gradient`, `fold_reset`, `fold_add`, `apply_gradient`).
This is the one place the published trainer lets a single shard's gradient be
edited before the fold without changing the binding. Both crashed at their
first `fold_export` (`split/run.log`, `ulp/run.log`):

    File ".../site-packages/mojolearn/parallel_training.py", line 283, in fold_export
        return memoryview(out).cast('B').tobytes()
    TypeError: memoryview: a bytes-like object is required, not 'Array'

In mojolearn 0.8.15, `ParallelByteLanguageModelTrainer.fold_export` builds
the package's own `Array` and then wraps it in `memoryview`. That `Array`
has no buffer protocol on Python 3.10 and 3.11 (DEVIATION 2305), and the
rented image is Python 3.11. The same pattern is in the 0.8.15
`cross_vendor.Worker.run` (`memoryview(state["parameters"])` over
`export_raw()`). T2's live segment passed because it ran bindings built
from source under the pixi interpreter, not this wheel in a 3.11 venv.
The live segment of any run that installs this wheel on a Python 3.11 box
goes through both calls. Python 3.12 is not affected.

The tool no longer depends on that method. `tools/lm_segment.py`
`fold_export()` makes the same binding call into the same buffer and reads
it through `_bytes_of` (commit 255487df6, unit-tested). The one-rental limit
had been spent, so `split` and `ulp` were not rerun. A rerun is the same
body with `--controls "split:split ulp:ulp=63,auto"` on one H100, about 15
minutes. The coordinator's refusal of a one-ulp shard gradient was already
seen on the M4 (plan section 6 item 7).

Not attempted here: a binding with one arithmetic change in the attention
tail (plan item 7's fifth control) needs a rebuilt binding, and this leg
installs the published wheel and builds nothing.

## Cost

One pod, `0gdinh13iz7z2y`, created 19:33:28 ET, terminated and verified
gone (HTTP 404) at 19:47:44 ET (`leg_teardown.txt`). That is 14.3 minutes at
$3.49 an hour, about $0.83, under a 120-minute lease and a $12 cap. Each
two-step replay took about 118 s; a failing control stops at step 101 after
about 71 s. Token fetch 51 s (12.4 GB, ranged), checkpoint fetch 10 s.

## Files

`status.txt` (the box's timeline), `gpu.txt`, `uname.txt`, `binding.txt`,
`wheel.sha256`, `ckpt.sha256`, `expect_chain.jsonl` and its sha256, and per
control `segment.json`, `chain.jsonl` (two lines for a pass, one for a
fail), `log.txt` and `run.log` (stdout and stderr). The full leg directory
stays in `~/mojolearn-evidence/gpt3-run/t3-controls/`.

## The same controls with the published 0.8.17 wheel (2026-09-24, `wheel-0.8.17/`)

0.8.17 is the Python-only release that fixes the `fold_export` and
cross-vendor `memoryview(Array)` crash on Python 3.10 and 3.11 found above;
every GPU and host binding in it is the 0.8.16 wheel's bytes, which are the
0.8.15 wheel's. One H100 (RunPod, Python 3.11 venv, wheel sha256
`dd3899f0e28d...`), from route A's `ckpt_00000100.blm`, two optimizer steps
each, held to the real chain's lines 101 and 102 with `--expect-chain`:

| control | expected | observed | first differing step | fields |
|---|---|---|---|---|
| positive (plain replay) | PASS | PASS at 101 and 102 | none | none |
| `--control none` | PASS | PASS | none | none |
| `--control split` (the device fold export path the live worker takes) | PASS | PASS | none | none; the path that raised `TypeError` in 0.8.15 runs |
| `--control ulp=63,auto` | FAIL | FAIL | 101 | state `d3eefb6d...` against `abc8b816...`, gradient `f386db39...` against `25830bfc...`; the 64 losses equal |

So the published 0.8.17 reproduces the running segment's steps bit for bit
on Python 3.11 through the fold export, and one ulp in one element of one
shard's gradient is caught at the first step. The T3 spec names wheel 0.8.17
from segment A/3 on (the driver re-reads the spec per segment); segments A/1
and A/2 ran 0.8.15, whose bindings are the same bytes. Pod
`ge6s83dbffmzae`, 02:52 to 03:03 UTC, deleted and confirmed gone (HTTP 404,
`wheel-0.8.17/leg_teardown.txt`), about $0.65.

## The published 0.8.18 wheel on both vendors (2026-09-25, `wheel-0.8.18/`)

0.8.18 rebuilds every binding: the AMD step-time work
(`bench/results/amd_step_time_2026-09-24/`, MI300X 141.0 to 32.3 s an
optimizer step, DigitalOcean MI325X 135.6 to 70.1 s) and the GEMM launch
bound that reaches NVIDIA binaries too. Before any AMD segment resumed on it,
the same replay ran from the published wheel (sha256 `c160fb6d5101...`) on
one H100 (RunPod, Python 3.11) and on one MI325X (DigitalOcean, Python
3.12), two steps each from route A's `ckpt_00000100.blm` held to the live
chain's lines 101 and 102:

| control | H100 | MI325X |
|---|---|---|
| positive (plain replay) | PASS | PASS |
| `--control none` | PASS | PASS |
| `--control split` (fold export path) | PASS | not run (a one-box worker path) |
| `--control ulp=63,auto` | FAIL at 101 on state and gradient | FAIL at 101 on state and gradient |

The release's own columns on the same wheel read 618 cells IDENTICAL and
none DIVERGENT against the CPU column on a 4090 and on an MI325X
(`bench/results/release_verification/2026-09-25_pypi_0818/`). Both check
boxes were deleted and confirmed gone (`wheel-0.8.18/*/leg_teardown.txt`);
about $1 and $3. Segment A/4 (AMD) started on 0.8.18 at 02:07 UTC; route B
stays held on Andrew's instruction while the AMD and NVIDIA step-time lanes
continue.
