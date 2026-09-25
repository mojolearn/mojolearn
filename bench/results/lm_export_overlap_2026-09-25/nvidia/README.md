# Export and hash overlap merged, two H100s, 2026-09-25

Branch `lane/lm-export-overlap-nvidia`, off `lane/lm-export-overlap`
(the merge of `lane/byte-lm-export-fast` and `lane/lm-segment-hash-overlap`).
A light proof, everything run once: only this branch's byte LM binding was
built, and it ran on two devices with the pooled optimizer, so m and v were
read back from two device contexts. Neither lane had run that before, and the
two lanes had never run together on a GPU.

## The box

RunPod pod `onxornly141vbg`, 2x NVIDIA H100 80GB HBM3 (driver 580.126.09),
created 17:23:36Z, deleted 17:40:36Z and confirmed gone (HTTP 404,
`teardown.txt`). About 17 minutes at $6.98 an hour, about $2.00. One earlier
create attempt got "no instances currently available" and created nothing.
The body is `tools/byte_lm_export_fast/leg_2gpu_nvidia.sh`, run through
`tools/gemm_remote_leg.sh nvidia` with `MOJOLEARN_GEMM_LEG_GPU_COUNT=2`,
`MOJOLEARN_GEMM_LEG_EXTRA`, `MOJOLEARN_GPU_ARCHS=sm_90a`,
`MOJOLEARN_NUMERIC_MODE=identical`, column nvidia. The binding was built on
the box from source (`byte_lm.new.so.sha256`). The runner's own gemm card
was identical to the Apple card.

## 1. Two-device export: PASS

`box_export.py --devices 0,1` from `runs/t3/2026-09-22/A/1/ckpt_00000100.blm`
(sha256 `80cd2126a89ba6d8...`), one step, then every export form
(`export_new_2gpu.json`). Every array's plain sha256 equals the OLD binding's
one-H100 export of the same state
(`bench/results/byte_lm_export_fast_2026-09-25/h100/export_old.json`) and the
NEW one-H100 export, and all forms agree within the run:

| array | sha256 (2 devices NEW = 1 device OLD = 1 device NEW) |
|---|---|
| parameters | `189a49eee383450f...` |
| m | `ceaca18311d713f0...` |
| v | `ef5df7ac3c40c184...` |
| flags | `360d579dbd14759b...` |
| gradient | `5caaccdbf79eec9b...` |

Chain digests state `abc8b816b5c3fb15...` and gradient `25830bfc2016dc14...`
equal route A segment 1's line for step 101.

Export seconds (medians of three; the OLD column is the one-H100 run of the
export lane, OLD was not rebuilt here):

| seconds | OLD, 1 H100 | NEW, 2 H100 |
|---|---|---|
| state, `export_raw()` | 5.89 | 1.24 |
| state, binding call into reused buffers | 4.17 | 0.136 |
| state, `export_raw(into=...)` | (none) | 0.135 |
| gradient, `export_gradients()` | 1.43 | 0.412 |
| gradient, binding call into a reused buffer | 0.900 | 0.045 |
| gradient, `export_gradients(into=...)` | (none) | 0.045 |

## 2. Overlapped replay on two devices with checkpoints: PASS

One run of `tools/lm_segment.py run` on devices 0,1, the default overlapped
loop, from checkpoint 100 for nine steps (101 to 109), held to A-1's witness
chain (`--expect-chain`, sha256 in `expect_chain.sha256`), under the recipe
with `checkpoint_every` 2 (`recipe_every2.sha256`), with presigned PUTs to the
scratch prefix `scratch/progress-proof-nvidia-2026-09-25/A/5/` (nothing under
`runs/t3/`). All nine steps agree with the expected chain. Field by field
against the witness (batch_index, gradient_sha256, hash_scheme,
losses_f32_hex, lr_f32_hex, route, schema, state_sha256, step) every line is
equal; only seconds, hash_seconds, prev, label and segment differ.

| step | state | gradient | step s | hash_seconds |
|---|---|---|---|---|
| 101 | `abc8b816b5c3fb15` | `25830bfc2016dc14` | 27.12 | 1.73 |
| 102 | `a9421f91b947f82c` | `94c40a6e3d5ec5aa` | 15.44 | 1.78 |
| 103 | `fcdb48b8ab51f2ef` | `19a43804ef4065de` | 15.45 | 5.11 |
| 104 | `64ca50a4cb8adaa7` | `88dc6074614bae71` | 15.45 | 0.50 |
| 105 | `d1e5bb8778c0b2a8` | `ea62cef5ad7d41bb` | 15.45 | 5.48 |
| 106 | `575ba19c183c5d9a` | `ebbcef53fe391824` | 15.46 | 0.18 |
| 107 | `9fd0a0e722a228a6` | `d6ff20b520a780e3` | 15.46 | 4.86 |
| 108 | `5b64448d53f3514b` | `c9c54884c49e587d` | 15.46 | 0.19 |
| 109 | `cbac80b8dc7dcff8` | `2a8ec4697838552e` | 15.45 | 4.85 |

A step that follows no checkpoint costs 0.17 to 0.50 s of `hash_seconds`. A
step after a checkpoint step reads about 5 s, which is the wait for the
previous step's save (about 4.7 s, taken under the interpreter lock), as the
hash overlap lane found on one device.

Checkpoints (`run_checkpoints.sha256`): `ckpt_00000102.blm`
`b043a0015be6e0d9...` (1,945,780,168 bytes) is the same file the hash overlap
lane wrote on ONE H100 with both runners. The others are 104
`517998d5a64e0132...`, 106 `a669e32aa410c024...`, 108 `46001e6929a50558...`,
109 `9c9256f60b5903a7...`. The `--sync-hash` replay, the OLD replay and the
split control were dropped from the scope before the pod ran; they are not
here.

## 3. Progress PUTs and a resume from R2 alone: PASS

The upload thread PUT `ckpt_00000102.blm` (130.9 s), then
`manifest.progress.tsv` and `chain.progress.jsonl` to step 103, then
`ckpt_00000104.blm` (96.4 s), then both progress files to step 105. Training
ran ahead of the uploads, so all nine steps were computed by then. The body
saw the step 105 progress line and sent SIGKILL (exit 137), while checkpoint
106 was the next upload, so no `chain.jsonl`, `manifest.tsv`,
`segment.json` or later checkpoint reached R2
(`resume/r2_scratch_objects.txt`: the two progress files, 102 and 104, and
nothing else).

On the Mac, with a copy of the T3 ledger in a scratch directory, a spec whose
`run` is the scratch prefix and whose route A has segment 5 from step 100 to
109 (`resume/spec.json`), and no results directory,
`lm_run_driver.record_partial` read the chain and manifest from R2, held both
checkpoints to R2's object sizes, and returned **`resume_from` 104** with
checkpoints 102 and 104 at the sha256 the box computed, arrival PASS,
`from_r2` true, and nothing refused (`resume/record_partial.txt`,
`resume/ledger_after.json`). The partial chain it saved
(`resume/partial.chain.jsonl`, steps 101 to 105) is byte for byte the first
five lines of the chain the box wrote. The real ledger was not written.

## Files

`status.txt` (the box's timeline), `export_new_2gpu.json`, `run/` (the chain,
manifest and log the box wrote), `run.clean.log`, `run_checkpoints.sha256`,
`upload_keys.txt` (key names only), the input digests, `files.new.sha256`,
`byte_lm.new.so.sha256`, `builds/`, `host.txt`, `gpu.txt`, `gpu_topo.txt`,
`mojo_version.txt`, `leg.txt`, `teardown.txt`, and `resume/`. The full leg
directory stays in `~/mojolearn-evidence/lm-export-overlap-nvidia/`.
