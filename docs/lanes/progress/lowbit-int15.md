# lowbit-int15: the GEMM of the fifteen-bit profile

Lane `lowbit-int15`, branch `lane/lowbit-int15`, worktree
`~/mojolearn-wt/lowbit-int15`, forked from `lane/lowbit-flag`. Brief
`~/mojolearn-evidence/lowbit-units/brief.md` (section Lane C and the updates
after it). Lane files `~/mojolearn-evidence/lowbit-int15/`. Nothing here
merges to main and no default moves.

Profile: `mojolearn.identical.gemm.int15i64.v1`. Contract:
`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6.

## What exists

| Piece | File |
|---|---|
| Seams (row scale, code, pieces, recombination, Int64 to float32, dequantization) | `checks/numerics_int15.mojo` |
| Host oracle, and the same sum through the pieces | `gemm/host/gemm_int15_oracle.mojo` |
| Device: quantize, dequantize, split; FLAT, PIECES and MMA plans; two entry points | `gemm/checks/gemm_int15.mojo` |
| Gates | `gemm/checks/gemm_int15_check.mojo`, `pixi run check-gemm-int15[-force-flat|-sabotage|-host-sabotage|-piece-sabotage]` |
| Cross-check against the quality lane's simulation | `bench/lowbit_quality/int15_export.py`, `gemm/checks/gemm_int15_sim_check.mojo`, vectors `gemm/checks/vectors/int15_sim_vectors.q15` |
| The quality lane's arithmetic, byte for byte, with its pin | `bench/lowbit_quality/arith.py`, `bench/lowbit_quality/ARITH_PIN` |
| Job scripts | `tools/lowbit_int15/{box,gate,sim}_job.sh`, `tools/lowbit_int15/digests.py` |

## The bound on k

`INT15_MAX_K = 65536`. A piece reaches -128, so one piece product reaches
`128 * 128` and `HH` alone admits `k <= 131071`; `HL + LH` share one Int32
and reach 32512 per term, which admits `k <= 66052`. The smallest bound is
66052 and the profile stops at the power of two below it. Contract 6.2 has
the seven derivations, each with its planted case.

## What ran

| Box | Job | Commit | What | Verdict |
|---|---|---|---|---|
| H100 (nvc3) | nvc3-0012 | e3d0fa24e | gate | DID NOT COMPILE, see Failures 1 |
| H100 | nvc3-0013 | 4b5675de4 | gate, force-flat, three arms | GREEN: 12 gates pass on both dispatches; each arm failed the gates it must |
| H100 | nvc3-0014 | (sim commit) | export, sim check, three arms | GREEN: host oracle and device equal the simulation on 120240 codes and 1172 cells (12 NaN, 85 infinite) |
| M3 Ultra (m3ultra-b) | 1790652213352 | 4b5675de4 | gate, force-flat, three arms | GREEN: 12 gates pass (flat and pieces; the MMA plan does not run on Apple) |

Digests, H100 against M3 Ultra (`tools/lowbit_int15/digests.py`): 171 cases,
every plan of both boxes and both host oracles printed the same digest.

## Failures

1. nvc3-0012: `gemm_int15_check.mojo` did not parse (a kernel argument named
   `out`, which Mojo reserves as an argument convention). All five phases
   exited non-zero, so the three sabotage phases read `held=yes` in
   `status.tsv` although nothing had run. The `reach:` lines caught it and
   the verdict was RED. Fixed at the root: a sabotage phase now holds only
   when its log carries the program's own verdict line with a non-zero
   count.

## Owed

See the end of this file; it is rewritten at every checkpoint.

- The gate and the sim check on `do-amd` and on `m2pro`; the sim check on `m3ultra-b`.
- The timing, two tables (inference, training), H100, Apple and AMD.
- The `PROFILES` row and the `mojolearn.linalg` entry.
