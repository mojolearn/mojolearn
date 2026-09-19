# Three laneless public surfaces, and the negative controls for them

lane/laneless-public-classes, 2026-09-19, Apple M4, ONE CORE, `nice -n 19`,
CPU host bindings only (no Metal job ran). Every column here is
`tools/identity_break.py --repeats 2` over all nine fixtures.

`tools/verification_matrix.py --json` reported 35 public API entries with no
identity lane at all. Fourteen are `models.*` and belong to
lane/models-namespace-lanes. Of the remaining 21, three lanes cover twelve:

| lane | covers |
| --- | --- |
| `saved-model-host-infer` | `HostForest`, `HostGBDT`, `host_predict`, `host_predict_proba` |
| `lowbit-conversions` | `lowbit.pack_one`, `lowbit.materialize_one`, `lowbit.widen_bf16`, `lowbit.format_of`, `lowbit.is_packed`, `linalg.from_bf16` |
| `grad-accumulation` | `training.accumulate_grads`, `training.accumulation_is_aligned` |

## The columns

| file | host set | what it is |
| --- | --- | --- |
| `clean.json` | the prebuilt clean host set | 27 cells, every one STABLE; 18 batch cells STABLE, 9 declared `n/a` |
| `sabotage-linalg-lowbit-convert.json` | linalg built `-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1` | `lowbit-conversions` moves 9/9, and so do the eight other linalg and low-bit lanes in the same column |
| `clean-linalg-rebuilt.json` | a clean rebuild of the EDITED oracle | its pair: the same nine lanes x nine fixtures, clean |
| `sabotage-linalg-generic-only.json` | linalg built `-D MOJOLEARN_HOST_SABOTAGE=1` ALONE | `lowbit-conversions` moves **0/9** |
| `sabotage-forest-own-define.json` | forest built with `-D MOJOLEARN_FOREST_HOST_SABOTAGE=1` (+ the CTR arm) | `saved-model-host-infer` moves 9/9 |
| `sabotage-forest-generic-only.json` | forest built `-D MOJOLEARN_HOST_SABOTAGE=1` ALONE | `saved-model-host-infer` moves **0/9** |
| `sabotage-training.json` | training built `-D MOJOLEARN_HOST_SABOTAGE=1` | `grad-accumulation` moves 9/9 |
| `regression-prebuilt-linalg.json` | the prebuilt linalg binding, built BEFORE the oracle edit | 16 cells, base fixture |
| `regression-rebuilt-linalg.json` | a clean rebuild of the EDITED oracle | the same 16 cells |

## What the pairs say, in the order they matter

**The low-bit conversion seams had no negative control at all.** `widen_bf16`,
`narrow_bf16`, `quantize_rows_int8` and `dequantize_rows_int8` in
`gemm/host/gemm_lowbit_oracle.mojo` are the whole of contract clauses L-1
through L-6 and they are what `mojolearn.lowbit` and the four public
`mojolearn.linalg` conversion entries compute. `MOJOLEARN_HOST_SABOTAGE`
reaches `gemm_oracle`'s leaf and `gemm_int8_oracle`'s dequantized cell and
stops: `sabotage-linalg-generic-only.json` is bit for bit
`clean.json` on this lane, all nine fixtures, every part. This is the same
defect shape as `linalg-eigh` (no arm) and `bpe-trainer` (an arm the lane
never reaches). The four seams were given their own arm,
`MOJOLEARN_LOWBIT_CONVERT_SABOTAGE`, and
`host_surface.GATE_SABOTAGE_OWN_DEFINES` now names it for the linalg family
so the CPU identity gate's sabotage set carries it. `base`
`0ff2da2cf430c48a -> d58be6e94de78728`.

**A clean build of the edited oracle moves no shipping bit.** The two
`regression-*.json` columns are the same 16 cells (`gemm-bf16`, `gemm-int8`,
`gemm-pinned`, `gemm-transposed`, `cholesky`, the ten `*-bf16w`/`*-int8w`
lanes and `lowbit-conversions`) hashed against the pre-edit prebuilt binding
and against a fresh build of the edited source: all 16 identical.

**The forest family's arm is not the generic one.**
`sabotage-forest-generic-only.json` reads the clean hash on all nine
fixtures; `-D MOJOLEARN_FOREST_HOST_SABOTAGE=1` moves all nine
(`base` `3918f891959b852a -> 4f407d8f1968fd09`). A column built with the
generic define alone would have recorded this lane as passing.

**Gradient accumulation's arm would have been inert on two fixtures.**
`host_samba_accumulate`'s arm WRITES 0.0 into the first combined element.
`X[0, 0]` is exactly 0.0 on `denormal_ftz` and a subnormal that flushes to
0.0 on `denormal`, so microbatches cut straight from the fixture would have
combined to 0.0 there and the arm would have changed nothing on two of nine
fixtures. The lane adds `+ 1.0`; measured, 9/9 move
(`base` `7914bf0b244d9142 -> c18d7abf3e2a978e`).

## Parts that do not move, and why each is kept

| lane | part | why no build arm can reach it |
| --- | --- | --- |
| `saved-model-host-infer` | `sha` | it hashes the ARCHIVE the estimator wrote; the host binding only reads it |
| `lowbit-conversions` | `f32_copy` | `pack_one(x, "float32")` is a copy that touches no binding |
| `grad-accumulation` | `single` | A = 1 returns from an EARLY RETURN above the arm |
| `grad-accumulation` | `aligned`, `flags` | integers out of a predicate and a set of refusals, no float fold near them |
| `lowbit-conversions` | `flags` bits 2, 3, 22 | designed 0s: `is_packed` of an unpacked tensor twice, and the python-fallback tripwire |

## The batch part was seen to fail

`MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` turns both declared batch parts
BATCH_MOVED at row 0 (`HostForest.predict`, `pack_one + materialize_one`).
`grad-accumulation` declares `n/a:accumulation-step`: the A microbatches ARE
the arithmetic the lane hashes.

## Not covered, and not faked

Nine of the 21 are not covered by any lane here, for reasons in
`docs/lanes/LANE_STATUS_laneless-public-classes.md`: the two decode sessions
have no CPU route at all (a test holds their refusal instead), and the seven
`parallel_*` entries are multi-device drivers whose worker operations are
absent from `_parallel_pool.CPU_OPERATIONS`, so every one of them refuses BY
NAME on a CPU column.
