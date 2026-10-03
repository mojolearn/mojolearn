# lane/apple-fast-ptimpute: power-transformer, simple-imputer (cut from main 8897404da, 2026-10-03)

Binding `x_prep`; board family algos; datasets istella (1,000,000 x 220, the regression) first, then
taxi (1,000,000 x 11). Every switch is a `-D MOJOLEARN_<NAME>` define, default OFF, compiled only under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`; IDENTICAL compiles main's code
unchanged. Profile and the cause of the istella regression: docs/apple-fast/notes/ptimpute.md. Request
lines: docs/apple-fast/ab/ptimpute.txt (one tag per lane x dataset x define; the last line isolates
PT_SPEC against PT_COLBATCH, the only pair that is not independent).

Quality rule: FAST never degrades quality. Every switch here changes the fold ORDER only (Welford
per thread, Chan's merge over chunks, a tree in the block), never the function: the same
log-likelihood, the same golden-section points and decisions, the same transform. Welford + Chan is
never less accurate than the serial two-pass fold it replaces, so `lambdas_`, `mean_`, `scale_` and
the imputer's statistics stay within FAST run-to-run spread (bench/x_prep_quality.py pairs them with
scikit-learn). The board's quality column (transform reconstruction / impute error) must not move.

| define | binding bit | lane | mechanism | expected | risk |
|---|---|---|---|---|---|
| `MOJOLEARN_PT_FOLD_NOX` | 16 | power-transformer | `x_prep/fastred.mojo` pt_fold_fast_kernel: after K = 0 the fold reads T alone (T is NaN exactly where X is), as `pt_fold_unit` does. Main's kernel reads X at stride d for every row at every one of the 50 evaluations, only to test NaN: at d = 220 that is one cache line per word, 220M line fetches per evaluation. | istella: the 50 strided X passes vanish; the fold becomes two coalesced passes over T. The largest single term of the 2,957 ms. taxi: small (the strided stream is small at d = 11). | Lowest. Three lines inside the kernel's loop. |
| `MOJOLEARN_PT_COLBATCH` | 1 | power-transformer | `x_prep/fastpt.mojo` pt_tile_kernel + pt_tile_finish_kernel, dispatched from `x_prep/device.mojo` at the `pt_fold` stage (the `pt_map` stage before it is skipped, `pt_log` is not staged). Block = (row chunk, 32-column group), a thread owns one column over the chunk's rows, so a simdgroup reads consecutive words of one row; the lambda-free log and the transform are computed in registers; count / mean / M2 by Welford into (chunk, column) partials; the finish kernel (a block a column) merges by Chan on a tree and takes the golden step (`pt_finish`). Python (`_expansion_prep.py` PowerTransformer.fit) shrinks T and LG to one word when the binding exports `x_prep_ptimpute_flags`. Tile set {64, 256} rows picked by d. | Per evaluation ONE coalesced read of X (880 MB at istella) instead of pt_map's X + LG read + T write plus the fold's X (strided) + 2 x T: roughly 7 GB of traffic per evaluation down to under 1 GB, and two n*d blocks fewer in the arena (3.5 GB -> 1.8 GB of device words). istella and taxi both. | The register InlineArray of 7 candidates is runtime-indexed (M = 1 here); the finish kernel folds ~3,900 chunks a column on 256 threads. |
| `MOJOLEARN_PT_SPEC` | 2 (+1) | power-transformer | COLBATCH's kernel over the IDENTICAL search's speculated points: Python stages the `pt_spts` / `pt_smap` / `pt_sfold` / `pt_sres` program with S = 3 (7 candidates a round, fixed, no env read); the device skips `pt_smap` and runs `pt_tile_kernel` with M candidates a thread (each its own Welford pair in registers), then pt_stile_finish_kernel writes VALS; `pt_sres` (a unit) walks the tree exactly as `pt_finish` would. No candidates' buffer, so `_pt_spec_depth`'s 2^28-word cap (which forced S = 0 at istella) does not apply. Turns COLBATCH on. | 18 passes over X instead of 50 and 18 x 3 launches instead of 100; each pass does 7 transforms per element (exp + a few ops), so it trades memory for compute. Compare with COLBATCH (last request line): a win if the M3 is bandwidth bound on the single-candidate pass, a loss if the 7 exps per element dominate. | More registers per thread (7 x (lambda, mean, M2)); if occupancy drops the pass gets slower than 7x the compute. Correctness: the same points and decisions as the IDENTICAL speculated search, which is on main. |
| `MOJOLEARN_PT_FUSED_TRANSFORM` | 4 | power-transformer | The standardize tail `pt_apply` (n*d, writes TX) + `col_stats` (TX, two strided passes) as `cs_tile_kernel` with LAM >= 0: the transform computed in registers from X, Welford + min / max / maxabs partials per (chunk, column), `cs_tile_finish_kernel` writes the six stats rows that `std_params` reads. Detected by the stage pattern (`fused_tail_pair`: pt_apply with no standardisation whose output is the next col_stats' input); Python shrinks TX to a word. | Removes an 880 MB write, two strided passes over it (2 x 220M line fetches at istella) and one n*d block of the arena. Expected tens of ms at istella, a few at taxi. | Low; one pass, same shape as SI_ONEPASS's kernel. Box-Cox of x <= 0 yields a NaN transform: skipped as the unit skips the NaN word pt_apply would have written (the fit then raises as before). |
| `MOJOLEARN_SI_ONEPASS` | 8 | simple-imputer (also power-transformer's opening col_stats) | Every `col_stats` stage as `cs_tile_kernel` + `cs_tile_finish_kernel` (LAM = -1): one row-tiled coalesced pass with the NaN mask fused instead of `col_stats_fast_kernel`'s two passes at stride d (one block a column). | simple-imputer istella: the strided 2 x 220M line fetches become one 880 MB coalesced pass; the radix sort stays (lane/apple-fast-prep2's `MOJOLEARN_X_PREP_FAST_QSELECT` already replaces it for the median; not duplicated here). taxi: small. power-transformer: its first col_stats, same saving. | Low. |
| `MOJOLEARN_PTIMPUTE_ALL` | 31 | both | All of the above (NOX is moot under COLBATCH: the fast fold kernel no longer runs). | The combined power-transformer istella number; simple-imputer = SI_ONEPASS alone. | The union. |

Reading the results: for power-transformer istella the order of expected gain is COLBATCH (or SPEC)
> NOX > FUSED_TRANSFORM ~ SI_ONEPASS. If NOX alone already brings FAST well under IDENTICAL's 433 ms
and COLBATCH does not beat it, keep NOX (three lines). SPEC is only worth keeping if it beats
COLBATCH head to head (the last request line). For simple-imputer only SI_ONEPASS applies.

Risky compile sites (no Mojo toolchain run on the GPU here; the FAST builds compiled on the M4):
`x_prep/fastpt.mojo` InlineArray[Float32, 7] indexed by a runtime candidate index inside the row
loop (pt_tile_kernel); the 2-D `grid_dim=(chunks, cgroups)` with a runtime `block_dim=tpb`
(preprocessing/minmax.mojo's idiom); `pt_finish` called from a new kernel (fastred.mojo already does);
`x_prep/device.mojo` `continue` inside `comptime if` blocks (the RR_EIGH idiom at device.mojo:330).

## Compile status (M4, compile only, nothing run; 2026-10-03)
All builds are `bindings/build_x_prep.sh`. The per-define and IDENTICAL builds are at head ed8e8ee7e:
- FAST `-D MOJOLEARN_PT_FOLD_NOX`: rc=0
- FAST `-D MOJOLEARN_PT_COLBATCH`: rc=0
- FAST `-D MOJOLEARN_PT_SPEC` (also turns on COLBATCH): rc=0
- FAST `-D MOJOLEARN_PT_FUSED_TRANSFORM`: rc=0
- FAST `-D MOJOLEARN_SI_ONEPASS`: rc=0
- FAST, all defines off: rc=0
- IDENTICAL (default mode): rc=0
- FAST `-D MOJOLEARN_PTIMPUTE_ALL`: rc=0 on a build that started at 06:34:57, two minutes before
  ed8e8ee7e (fastpt.mojo launch helpers take `mut ctx`). The rebuild at head was stopped when Andrew
  halted lane compiles. Compile owed: peer.

Brief candidates folded into other defines: PT_TILE220 is the row-tiled column-group layout of
COLBATCH / FUSED_TRANSFORM / SI_ONEPASS (`pt_tile_kernel`, `cs_tile_kernel`), and SI_LIVEBUF is not a
separate define. The imputer's launch and buffer savings are in SI_ONEPASS.
