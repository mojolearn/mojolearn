# lowbit-int15: the GEMM of the fifteen-bit profile

Lane `lowbit-int15`, branch `lane/lowbit-int15`, worktree
`~/mojolearn-wt/lowbit-int15`, forked from `lane/lowbit-flag`. Brief
`~/mojolearn-evidence/lowbit-units/brief.md` (section Lane C and the updates
after it). Lane files `~/mojolearn-evidence/lowbit-int15/`. Nothing here
merges to main and no default moves.

Profile: `mojolearn.identical.gemm.int15i64.v1`. Contract:
`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6, clauses W-1 to W-11.

## What exists

| Piece | File |
|---|---|
| Seams (row scale, code, pieces, recombination, Int64 to float32, dequantization) | `checks/numerics_int15.mojo` |
| Host oracle, the same sum through the pieces, the transposing quantizer | `gemm/host/gemm_int15_oracle.mojo` |
| Device: the conversions in two schedules (rows, parallel), the split; FLAT, PIECES and MMA plans; two entry points | `gemm/checks/gemm_int15.mojo` |
| Device, Apple: the float matrix unit in exact chunks, forms TWO and FOUR (not dispatched) | `gemm/checks/gemm_int15_apple.mojo` |
| Gates | `gemm/checks/gemm_int15_check.mojo`, `pixi run check-gemm-int15[-force-flat, -unstated-loads, -sabotage, -host-sabotage, -piece-sabotage, -quant-sabotage, -apple-chunk-sabotage]` |
| Cross-check against the quality lane's simulation, forward and backward | `bench/lowbit_quality/int15_export.py`, `gemm/checks/gemm_int15_sim_check.mojo`, vectors `gemm/checks/vectors/` |
| The quality lane's arithmetic, byte for byte, with its pin | `bench/lowbit_quality/arith.py`, `bench/lowbit_quality/ARITH_PIN` |
| Python | `mojolearn.linalg.matmul_int15`, `quantize_int15`, `dequantize_int15` (`python/mojolearn/_linalg_impl.py`), both bindings |
| The repo's harness | lane `gemm-int15` in `tools/identity_break.py`, `tools/identity_lanes/int15.core` and `int15.checks`, four patches in `gemm/checks/sabotage/int15_*.patch` |
| Timing | `bench/gemm_int15_price_main.mojo`, `tools/lowbit_int15/table.py` |
| Job scripts | `tools/lowbit_int15/{box,gate,int8_gate,sim,price,harness}_job.sh`, `digests.py`, `identity_table.py`, `fetch_steward.py`, `sync_h100.sh` |

An operand crosses the Python boundary as its two int8 planes and its row
exponents (the package's Array has no int16).

## The bound on k

`INT15_MAX_K = 65536`. A piece reaches -128, so one piece product reaches
`128 * 128` and `HH` alone admits `k <= 131071`; `HL + LH` share one Int32
and reach 32512 per term, which admits `k <= 66052`. The smallest bound is
66052 and the profile stops at the power of two below it. Contract 6.2 has
the seven derivations, each with its planted case.

## Identity: what ran

| Box | Job | Commit | What | Verdict |
|---|---|---|---|---|
| H100 (nvc3) | nvc3-0012 | e3d0fa24e | gate | DID NOT COMPILE, Failures 1 |
| H100 | nvc3-0013 | 4b5675de4 | gate, force-flat, three arms | GREEN |
| H100 | nvc3-0014 | 06df635bc | export, sim check, three arms | GREEN: host oracle and device equal the simulation on 120240 codes and 1172 cells (12 NaN, 85 infinite) |
| M3 Ultra | 1790652213352, 1790652490744 | 4b5675de4, 06df635bc | gate; gate and sim | GREEN |
| M2 Pro | 1790652483764 | 06df635bc | gate and sim | GREEN |
| MI325X | 1790652488034 | 06df635bc | gate and sim | GREEN |
| H100 | nvc3-0021 | 42bb85725 | gate (with the parallel and transposing quantizers, the stated loads and the unstated-loads arm), the int8 gate, sim | GREEN, every arm seen failing; stated and unstated loads printed the same 981 digests |
| MI325X | 1790654387236 | 42bb85725 | the same three | GREEN, the same 981 digests |
| M3 Ultra, MI325X | 1790653519274, 1790653523354 | 77598eeb7 | gate with the quantizer's defect arm | GREEN |
| M2 Pro | 1790653526005 | 77598eeb7 | gate | GREEN (its timing phase failed, Failures 2) |

Step 1 table (commit 06df635bc, four boxes): 178 cases, every plan's digest
is the host oracle's, the host oracles agree across two x86-64 and two arm64
hosts, every arm failed the gates it must
(`~/mojolearn-evidence/lowbit-int15/digests/IDENTITY_TABLE_step1.md`).

## Timing

Runs of record: one job alone on its box; build, an untimed run of every
arm, then the timed run of the same binary; 5 timed calls per arm, median.
`over` is the fifteen-bit time over fp32.v1's at the same row in the same
run; above 1 it took longer. Tables:
`~/mojolearn-evidence/lowbit-int15/tables/`.

| Box | Run | Job | Commit | Complete inference call at the training rows, over fp32.v1 | Three products of one layer, added, over fp32.v1 |
|---|---|---|---|---|---|
| H100 | 1, loads with no alignment stated | nvc3-0017 | 77598eeb7 | 2.80 to 3.42 | 3.03 to 3.22 |
| H100 | 2, alignment stated | nvc3-0023 | 42bb85725 | 1.21 to 1.45 | 1.39 to 1.42 |
| MI325X | 1, loads with no alignment stated | 1790653523354 | 77598eeb7 | 0.66 to 0.78 | 0.76 to 1.09 |
| MI325X | after the fix | | | not run yet | not run yet |
| M3 Ultra | 1, flat and pieces kernels | 1790653519274 | 77598eeb7 | 11.8 to 13.3 | 12.7 to 13.7 |
| M3 Ultra | the float-unit plans | | | not run yet | not run yet |
| M2 Pro | 1 | 1790653526005 | 77598eeb7 | FAILED, Failures 2 | |

On the H100 four unit products do not take less time than one fp32.v1
product at the training rows, before or after the fix. At three decode rows
the complete call takes less (the head at t1 and t8, mlp_up at t8).

The head's input gradient contracts over the vocabulary, `k = 128256`,
above `INT15_MAX_K`: the profile refuses it by name, so the head has no
three-product line. The bound comes from the Int32 piece accumulators only;
a plan that carries them into Int64 every 65536 steps would admit it. Not
written; it needs its own clause.

READ WITH CARE. In runs 1 and 2 on the H100 the column "product alone"
reads above the complete call that contains it at the large rows. The unit
product was timed directly after the pieces kernel, a launch of 300 ms. The
alternation now times the two one-thread-per-cell kernels last.

## Failures

1. nvc3-0012: `gemm_int15_check.mojo` did not parse (a kernel argument named
   `out`, which Mojo reserves). All five phases exited non-zero, so the
   sabotage phases read `held=yes` although nothing had run; the `reach:`
   lines caught it. Fixed at the root: a sabotage phase holds only when its
   log carries the program's own verdict line with a non-zero count.
2. M2 Pro, 1790653526005, timing: `POISON SURVIVED at cell 1723968` at
   `mlp_down.t512` (512 x 4096 x 14336), arm `training.planes.rowquant`, in
   the warm run and in the timed run alike (both stopped after 192 digest
   lines). The pieces kernel, one thread per cell, did not write every cell
   of that product. The same kernel at the same row wrote every cell in
   three earlier arms of the same run. At that row one launch takes seconds
   on the M2 Pro (its rate at mlp_down.t8 is 7 G MAC/s). CAUSE NOT
   ESTABLISHED. Owed: the row alone on that box, repeated.
3. The first spelling of `mma_operands_aligned` took pointers with origin
   parameters and did not compile; it takes addresses.
4. The flag branch: `origin/lane/lowbit-flag` at 078fab2c8 held a rename only.
   Reported; fixed by the orchestrator in 819f43f16; merged at db7eb7842.

## The arith.py pin

Moved from blob 17337a5c to c60a455c (lane/lowbit-quality at 8a6589cb3). The
diff adds `DROPPED_NOTE`, `is_dropped_kind`, `Spec.dropped` and one field of
`Spec.describe`; no line of `ftz`, `pow2_f32`, `row_exponent`,
`quantize_rows` or `product_prepared` moved. The forward vectors carry the
old blob in their header until the next export job.

## Held in reserve: a rule for the attention product P.V

Decided by the orchestrator (brief, "THE ATTENTION PRODUCT P.V"): in the
profile's first version P.V stays on fp32.v1, the pinned ascending chain the
block runs today. The profile covers the projections, the head and Q.K^T.

Why: P.V contracts over the keys. Under clause W-9 the right operand would
be V^T, one scale per feature taken over all keys of the span. At prefill
that absmax reaches keys after the query; at decode step `i` the span is
`0..i`, so the exponent and the codes of the same V values differ from
prefill's, and decode == prefill fails by construction.

The rule held in reserve, as a later version under its own name:
- V is quantized per KEY over the features, exponent `ev[j]`, as K is, so
  its codes are made once and cached.
- The key's scale is folded into the left operand, exactly:
  `P'[i, j] = P[i, j] * 2^ev[j]`.
- P' is quantized per QUERY over the keys, exponent `ep[i]`.
- Cell `(i, d) = 2^ep[i] * sum_j code(P'[i, j]) * code(V[j, d])`: a per-cell
  scale and an exact integer sum, on the same kernels (right operand the V
  codes read column-wise, its exponents zero).
- A masked key has `P = 0`, so a row's absmax, exponent and codes are the
  same at prefill and at decode.
It is another arithmetic than the one whose perplexity was measured for
`attn_pv`, so quality would be measured again under it.

## Owed

- The repo's harness verdict for `gemm-int15` (step 2), four boxes.
- The Apple float-unit plans: gate on both Apple generations, timing on the
  M3 Ultra, both forms.
- Timing after the fix on the MI325X; a second H100 run after the fix
  (nvc3-0024, the orchestrator's).
- Lane D's tuned unit kernel, once its gates of record are green, as an arm.
- The backward vectors of the quality lane (step 50 committed; a later step
  is being exported), host oracle and devices held to them.
- The forward vectors exported again under the moved pin.
- Failures 2, reproduced alone.
- The blocks (step 3): not started; waits on the orchestrator's word.
