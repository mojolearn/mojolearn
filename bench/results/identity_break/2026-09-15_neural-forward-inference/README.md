# Public CPU inference for Mamba, Samba and the byte LM (2026-09-15)

Lane `lane/inference-neural-forward`, commit 0b459d9e8. Andrew, Sep 15: train on a GPU,
infer anywhere. The new public classes, over the shipped `_mojolearn_neural_host`:

| family | public call | lanes |
|---|---|---|
| Mamba | `Mamba1BlockInference(weights).forward(x, lengths=None)`, `Mamba2BlockInference(weights, dt_limit=...)`, `Mamba3BlockInference(weights)`, each from a zero state | mamba1, mamba2, mamba2-dtlimit, mamba3 |
| Samba | `SambaInference.from_checkpoint(path).forward(ids, lengths=None)` (or `SambaInference(config, weights)`) | samba, samba-untied-dropout-accum |
| byte LM | `LanguageModelInference.from_checkpoint(path)` (already public; no API change) | byte-lm, byte-lm-resident |

A carried state, `step`, `allocate_state` and `backward` (and `loss`, `train_step` on
`SambaInference`) refuse by name. On a CPU column, `tools/identity_break.py` asks each of
these lanes' infer, reload, batch, batchscale and ragged cells through the public class:
the Mamba classes built from the fitted block's weights, `SambaInference` from the stack's
saved checkpoint, and `LanguageModelInference` from the trainer's exported checkpoint.
The same routing now covers the transformer and transformer-window lanes' batchscale and
ragged cells. Train and batchgrad cells keep the fitted model.

## Where it ran

One RunPod CPU pod, `tools/runpod_cpu_leg.sh`, AMD EPYC 4564P, 16 vCPU, pod 56kjdb5tpviw22,
2,879 s billed, deleted (`DELETE -> 204`, then `GET -> 404`). All bindings were built on the
box from 0b459d9e8, with `neural` and `byte_lm` also built `-D MOJOLEARN_HOST_SABOTAGE=1`.
The body is `leg_body.sh`. No GPU was rented.

Two earlier pods on this lane were deleted and verified gone. zg8rcwgwivr3lm (8 vCPU,
commit 7a60ba0f7) ran the columns, but it handed batchgrad the inference class, which
refuses backward, so those cells read REFUSED. It also loaded the byte LM sabotage under
a module name the clean binding already held, so that arm moved nothing. 9j5nnzm9lz7ipf
stopped in its first seconds on a shell variable name. Neither result is used here.

## Identity against the committed GPU columns (`diff.record.txt`)

Columns: `2026-09-15_batch2`'s `apple-m4.json`, `nvidia-h100-sm_90a.json` and
`amd-mi325x-gfx942.json`, plus `cpu-amd-epyc-4564p.json`. Ten lanes, nine fixtures, two
repeats, `--batch-grad --batch-scale --ragged`, `--require-columns 4 --owed-json`.
Exit 0:

    summary: IDENTICAL=90
    summary (infer/model): IDENTICAL=126, N/A=54
    summary (batch): IDENTICAL=90
    summary (batchgrad): IDENTICAL=72, N/A=18
    summary (batchscale): IDENTICAL=90
    summary (ragged): IDENTICAL=90
    summary (owed): OWED=0

Every cell is IDENTICAL x4. The 54 N/A are the Mamba and Transformer blocks' model column
(no save), and the 18 batchgrad N/A are the byte LM trainers (mean reduction). rlpair is
NOT-COMPARED because the batch2 record carries none. No cell is owed.

## Host sabotage, seen to fail (`diff.host-sabotage.txt`)

The same ten lanes, one repeat. Every reference binding was clean; `neural` and `byte_lm`
were the sabotage builds (`MOJOLEARN_HOST_DIR` set to the mixed directory):

    summary: DIVERGENT=18, IDENTICAL=72
    summary (infer/model): DIVERGENT=107, IDENTICAL=19, N/A=54
    summary (batch): DIVERGENT=90
    summary (batchscale): DIVERGENT=90
    summary (ragged): DIVERGENT=90

- **Held-out cells:** every batch, batchscale and ragged cell of every lane moved. Every
  infer cell moved except mamba1 `negative`; that fixture's batch and ragged cells did move.
- **Model cells:** the 18 IDENTICAL are the Samba checkpoint files, which no inference
  arithmetic writes.
- **Train cells:** the Mamba, Samba and Transformer train cells did not move, because
  their training stays on the clean reference bindings. The 18 DIVERGENT train cells are
  byte-lm and byte-lm-resident. Their CPU trainer steps through the same `byte_lm` binding
  `LanguageModelInference` uses, so for the byte LM this arm cannot keep training clean.
- **rlpair:** the byte LM rlpair cells read RLPAIR_MOVED for the same reason.

## Batch sabotage, seen to fail (`diff.batch-sabotage.txt`, `batch-sabotage/`)

`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1`, base fixture, one repeat, one JSON per lane. The
harness refuses to merge sabotage runs. Each lane is diffed against the production column,
and on all ten lanes the batch, batchscale and ragged parts read BATCH_MOVED=1. The other
eight rows are ONE-COLUMN, the fixtures the sabotage run did not ask.

## No training symbol in the shipped binding (`nm.txt`)

`nm --defined-only`, matching `backward|optimizer|adamw|decode|clip_grad|ce_loss|train_step`:

| binary | bytes | defined symbols | matches |
|---|---|---|---|
| `_mojolearn_neural_host.so` (ships) | 490,504 | 508 | 0 |
| `_mojolearn_training_host.so` (reference) | 420,416 | 550 | 20 |
| `_mojolearn_mamba_host.so` (reference) | 1,256,872 | 734 | 71 |
| `_mojolearn_transformer_host.so` (reference) | 341,528 | 419 | 12 |

The first pod read `nm -D`, whose dynamic table holds only `PyInit` in every one of these
binaries, so its zero could not fail. The controls above show this check can.

## The installed test wheel (`calls.compare.txt`)

- **Build:** a wheel of the ten shipped families was built with the pkg env's `build`.
  The pod's envs carry no pip, so it was installed by extracting the archive into
  `/tmp/target`, after checking each of its 97 RECORD sha256 values.
- **Byte comparison:** `wheel_calls.py` hashed 16 public calls twice, once from the source
  tree and once from the installed wheel (`MOJOLEARN_HOST_DIR` unset):
  - the three Mamba classes, with and without `dt_limit`, whole and ragged, L = 70 across
    the Mamba-3 chunk;
  - `SambaInference`, tied and untied, whole and ragged;
  - `LanguageModelInference` on both paths, whole and ragged.
  All 16 are EQUAL.
- **Tests:** `test_neural_inference.py`, run against the installed wheel alone: 12 passed,
  2 skipped by name (the reference comparisons, which need reference bindings no wheel has).

## Wheel size (Linux x86-64, this lane's local test wheel)

| | origin/main d5265152b | this lane |
|---|---|---|
| wheel | 1,891,155 bytes | 1,972,140 bytes (+80,985) |
| `_mojolearn_neural_host.so` | 269,368 bytes, 76,001 compressed | 490,504 bytes, 154,316 compressed |

## Tests on the pod

- `test_neural_inference`: 14 passed.
- `test_cpu_inference_boundary`: 10 passed.
- `test_host_surface`: 121 passed, 2 failed. The two failures name record files the pod
  does not carry (`bench/results` is not shipped to a pod). On the Mac, with the files
  present, `test_host_surface` and `test_cpu_inference_boundary` pass (133).

## Owed

- **mamba2-pretrained:** `tools/mamba2_pretrained_identity.py` has not run on any GPU
  (its brief says so), so there are no pretrained mamba2-130m hashes for a CPU run of
  `Mamba2BlockInference` to diff against.
- **Other CPUs:** this is one x86 CPU column. The Apple M4 CPU column, on the Mac, is not
  part of this record.
