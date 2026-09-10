# Mamba3 state-increment operand tiles — September 10, 2026

The independently tested tile reduces the large public shapes by 7.2% and 3.2% in the first comparison while preserving complete output bits. The final guarded-default rerun measures 65.87/120.40 ms, 8.7%/4.5% below the same-pod baseline. A forced tile on the tiny call loses about 40 microseconds, so the default keeps its original implementation; the final tiny median is 0.984050 ms.

| Public shape | Baseline median ms | Forced tile median ms | Final default ms |
| --- | ---: | ---: | ---: |
| B2 L4 D32 | 0.989344 | 1.029426 | 0.984050 |
| B8 L4096 D512 | 72.118117 | 66.936124 | 65.868374 |
| B8 L1024 D2048 | 126.097558 | 122.015735 | 120.395977 |

Five timed rounds per arm, same H100 80GB HBM3/driver 580.126.09. Both public arms use image `/usr/bin/python3` 3.11.10, NumPy 1.26.3 and the matched archived seed-7 fixture harness. Mojo builds use activated Pixi Mojo 1.0.0. No opponent was measured; historical opponent evidence remains in `bench/OPPONENT_REFERENCE.md`. This artifact primarily claims same-pod implementation improvement. Reusing the admitted archived H100/driver reference-scan medians (16.396241/19.383267 ms) puts the final large ratios at 4.02×/6.21×. That historical opponent was not retimed on this physical card; our public forward includes transfers and reports while the reference measures its device scan. The requested 1–1.5× target remains unmet.

## Execution and admission

The kernel stages eight P rows of independently rounded decayed V and 32 N columns of K for all Q tokens in 10,240 bytes of shared memory. Each of 256 threads owns one increment and executes its complete original ascending Q-term FMA fold, including padded zero terms. The previous shared-V implementation is the performance baseline; a third scalar implementation supplies a separate direct numerical comparison. No accumulator crosses a thread boundary. State scan, reports, weight mutation visibility, refusal order, and arithmetic seams remain unchanged.

The new default is IDENTICAL/NVIDIA only, requires the column to support 256 threads and 10 KiB of shared memory, and requires `B * chunks * heads >= 128` (at least 1,048,576 increment outputs). This is a structural occupancy guard; intermediate workloads have not been tuned. `MOJOLEARN_MAMBA3_TILED_INCREMENT=1` forces the tile when capacities permit, including Apple validation; `MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE=1` retains the old default. Apple remains on its prior default because it has correctness evidence but no tile performance evidence.

## Recorded checks

- Direct scalar/shared-V/tiled gate: 2,752,512 complete increment cells equal on NVIDIA and Apple, including Q32/Q64, eight ragged lengths, B2/H3, signed zeros, subnormals, cancellation and large finite operands.
- NVIDIA baseline and forced-tile native default and B2/L65/D64 traces compare byte-for-byte. Decode-cross, continuation and refusal gates pass in both arms.
- Both NVIDIA arms pass the 39,087,232-cell fresh-vs-explicit-state check and mutable-weight refusals, plus all 102 public surface checks.
- Three complete public output files, including both 64 MiB large outputs, have equal SHA256 values. Binaries are omitted; witnesses are in `h100/m3-increment-tile/full-output-identity.json`.
- Apple forced-tile native default/long traces match baseline, decode-cross/continuation/refusal pass, and the forced binding passes 146,560 fresh/report cells plus 102 surface checks. NVIDIA final default passes all 39,087,232 fresh/report cells, mutable refusals, 102 surface checks, and complete SHA comparison against the original baseline. See `h100/m3-increment-final/`.

## Reproduction and setup provenance

Base main2419895f; initial kernel041bc46a; final guarded dispatch eded801f. `tools/mamba3_increment_tile_leg.sh` runs the baseline and forced tile in an isolated checkout; it requires the three committed corpus smoke cases, `mamba/corpus/gen_corpus.py`, and the matched archived public driver/helper pair. Specify `MOJOLEARN_MAMBA3_PYTHON=/usr/bin/python3` on the rental image. It activates Pixi for compilation.

Initial setup attempts stopped before any valid timing: the archive omitted the empty extension directory and corpus generator; Pixi lacked torch needed by the fixture module; the main-tree helper predates the archived driver's `py_rows` API. The final script checks required corpus inputs, creates the extension directory, selects an explicit Python runtime, and installs the matched archived pair. Completed same-source baseline binaries were reused; successful evidence comes from resumed leg r5. Its exact commands are retained in `h100/m3-increment-tile/resumed-leg.sh`. No failed attempt is counted as a numerical pass or timing sample.
