# T3 CPU witness: route A segment 1 replayed on a CPU, 2026-09-23

The CPU witness column of `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` (section 6,
items 5 and 6), run once against the live T3 run's first segment (route A,
2-GPU H100, `runs/t3/2026-09-22/A/1/` in R2). A rented RunPod CPU pod loaded
the run's checkpoints and recomputed recorded GPU arithmetic with the HOST
byte LM binding `_mojolearn_byte_lm_host.so` from the PUBLISHED wheel
mojolearn 0.8.15 (manylinux, sha256 `281677583838358deb50f3a86c2505c695c1ab4ee1f42cfa3476a17ad1d9a8bd`,
binding sha256 `3aa40d5679c60e15...`, nothing built), through
`tools/lm_cpu_witness.py`. Shape: 162,147,840 parameters, batch 4, length
2048, vocabulary 50,257, K=64 shards a step, hash scheme `sliced-sha256-8.v2`,
recipe sha256 `9f7f695b9a0bae17...`.

Every check first loaded the checkpoint, hashed its state under the recipe's
scheme and held it to the chain line of that step (both matched:
checkpoint 100 `2b37e85edfd701eb...`, checkpoint 700 `983700bf6fedbe79...`),
held the checkpoint's data schedule and the token manifest to the recipe, and
fetched only the token byte ranges it needed from the stream's parts in R2.

## Results

| check | from | compared to | want | got | verdict | seconds |
|---|---|---|---|---|---|---|
| loss, shard 0, threaded path | ckpt 100 | chain line 101 `losses_f32_hex[0]` | `40de1f6a` | `40de1f6a` | **PASS** | 45.6 |
| loss, shard 0, reference path (one thread) | ckpt 100 | chain line 101 `losses_f32_hex[0]` | `40de1f6a` | `40de1f6a` | **PASS** | 3,268 |
| loss, shard 63, threaded | ckpt 100 | chain line 101 `losses_f32_hex[63]` | `40def7ee` | `40def7ee` | **PASS** | 60.3 |
| loss, shard 0, threaded | ckpt 700 | chain line 701 `losses_f32_hex[0]` | `40814591` | `40814591` | **PASS** | 45.0 |
| CONTROL: one token id + 1 (row 0, position 0, 44 to 45) | ckpt 100 | chain line 101 shard 0 | `40de1f6a` | `40de1fe6` | **FAIL, expected** | 44.4 |
| CONTROL: low bit of parameter 38,597,376 (block0.norm1_w[0], `3f802920` to `3f802921`) | ckpt 100 | chain line 101 shard 0 | `40de1f6a` | `40de1f69` | **FAIL, expected** | 47.5 |
| held-out batch 0 (shard 013) | ckpt 100 | recorded for other columns | | `40da1f68` (6.816334) | RECORDED | 45.3 |
| held-out batch 0 (shard 013) | ckpt 700 | recorded for other columns | | `40864a82` (4.196595) | RECORDED | 45.4 |
| loss, **all 64 shards**, threaded, four at a time | ckpt 100 | chain line 101 `losses_f32_hex[0..63]` | 64 values | 64 equal | **64 PASS** | 124 median |
| loss, **all 64 shards**, threaded, four at a time | ckpt 700 | chain line 701 `losses_f32_hex[0..63]` | 64 values | 64 equal | **64 PASS** | (same run) |
| gradient: one host training step of shard 0 | ckpt 100 | chain line 101 `gradient_sha256` (needs all 64) | `25830bfc2016dc14...` | none | **NOT COMPARED**: shard 0 did not finish in 8,850 s | over 8,850 |

The first eight rows are leg 1 (`leg1/`, AMD EPYC 4564P, 16 vCPU); the two
all-shard rows are leg 3 (`leg3_all_shards/`, the same CPU model, a second
pod; every row in `losses.tsv`); the gradient row is leg 2
(`leg2_gradient/`, AMD Ryzen Threadripper 7960X, 16 vCPU, 128 GB).

The CPU loss is the H100's loss bit for bit for every one of the 128
shards of global steps 101 and 701, on the threaded path, and for shard 0 on
the one-thread reference path too, and one changed token or one changed parameter
bit moves it off: one ulp in one norm weight moved the loss by one ulp.

## The held-out batch, defined

The recipe names no held-out batch. The plan's "shard 013" is FineWeb-Edu
sample-10BT parquet shard 013, which the token stream holds as its
`validation_range` [2,926,502,182, 3,110,556,447)
(`bench/results/fineweb_tokens_2026-09-22`). Held-out batch h is the training
schedule's own rule applied to that range: row b reads ids
`[vlo + (h*B*L + b*L) % (vhi - vlo - L - 1) : + L + 1]`. Batch 0 is the
first 4 x 2048 + 1 ids of shard 013 (ids sha256 `e266419932ca83aa...`).
Every vendor at every boundary can be held to these bits
(`lm_cpu_witness.py heldout --expect-f32-hex`).

## What the gradient witness can and cannot compare

**Not compared.** The plan's item 5 has the witness "replay one shard's
gradient ... and match the recorded shard-gradient hash". Two things stand
in the way, and both are findings, not retries:

1. **The chain records no per-shard gradient hash.** A chain line holds the
   64 shard losses and the hash of the SUMMED gradient only;
   `lm_segment._window_witness` (record windows) adds per-array state hashes
   and the ids' hashes but no shard gradient. So one shard's gradient has
   nothing recorded to be compared to, and the only gradient comparison
   available is the whole fold of all 64 shards against `gradient_sha256`.
   `lm_cpu_witness.py gradient` and `fold` do exactly that (per-shard
   gradients saved in parallel, the ordered left fold continued across
   owners, a test that swaps two shards and sees FAIL); a record window
   that also wrote each shard's gradient hash would make a one-shard
   gradient witness possible.
2. **The host training step is too slow and too large at this shape.**
   `byte_lm_host_train_step` has the reference path only (one thread, by
   design: the backward's cross-row folds have no threaded twin). One shard
   of 8,192 tokens did not finish in 8,850 s on a Threadripper 7960X, with a
   peak resident set of 54.6 GiB (`leg2_gradient/rss.tsv`); on leg 1's
   64 GB pod it was killed by the memory limit. A whole step is therefore at
   least 64 x 2.5 = 157 CPU hours at about 55 GiB per concurrent shard, far
   beyond the three-hour budget, so the summed gradient of step 101 was not
   compared. For scale, the forward alone takes 3,268 s on the reference
   path and 45 s on the threaded path, so a threaded backward is what would
   bring a whole-step CPU gradient witness within reach.

What was run in its place is the largest check of the recorded step that
fits: every one of the 64 shard losses of steps 101 and 701, each a full
forward of 8,192 tokens through all 162M parameters from the boundary
checkpoint, all 128 bit-equal to the H100.

## Cost

| leg | pod | CPU | billed | spend |
|---|---|---|---|---|
| 1: losses, controls, held-out, first gradient attempt | 3dzd3n2w5w88kk, cpu5g 16 vCPU | AMD EPYC 4564P | 3,828 s | $0.78 |
| 2: gradient alone | 9gml0ret9c7uxx, cpu5m 16 vCPU 128 GB | AMD Ryzen Threadripper 7960X | 9,014 s | $2.60 |
| 3: all 128 shard losses | obrolmxw4auf1f, cpu5g 16 vCPU | AMD EPYC 4564P | 4,383 s | $0.90 |
| total | | | 4.8 h | **$4.28** |

Every pod was deleted by `tools/runpod_cpu_leg.sh` and verified gone (HTTP
404, `*/teardown.txt`). Checkpoints (1.95 GB each) were fetched by the pods
from R2 in 20 to 40 s; token ids came as byte ranges of the stream's parts,
never the whole 12.4 GB.

## Files

`chain_excerpt.jsonl` (chain lines 100, 101, 700, 701 of route A segment 1,
from the partial chain sha256 `8a6fc036...`), `witness.cmd`,
`gradient.cmd` and `losses.cmd` (the three pod bodies, presigned URLs as placeholders), and per leg
`records/*.json` (one record per check: digests, bits, token ranges, box,
binding), the check logs, `gate.txt`, `lscpu.txt`, `wheel.sha256`,
`checkpoints.sha256`, `inputs.sha256`, `pip_freeze.txt`, `timings.tsv` and
`teardown.txt` (the verified delete). Rerun: `python3 tools/lm_cpu_witness.py
--help`; tests `tools/tests/test_lm_cpu_witness.py`.
