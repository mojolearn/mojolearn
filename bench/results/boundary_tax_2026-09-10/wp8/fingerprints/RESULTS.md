# Final forest identity-break matrix

**PASS: 108/108 cells STABLE, 0 MOVED, 0 REFUSED; 216 completed fits.**

FAST, DETERMINISTIC and IDENTICAL each ran four public learners (RF/ET classifier/regressor), all nine fixtures, and two repeats: 36 cells and 72 fits per mode. Fixtures: base, ties, hashed, wide, denormal, denormal_ftz, dupes, odd and negative. Each RF/ET binding's compiled numeric mode and Metal vendor were read back before execution.

The unmodified `tools/identity_break.py` hashes complete public predictions and classifier probabilities; `_h` accepts `Array` through `np.asarray`. All expected cells and repeat counts were checked independently from the saved JSON. Logs and JSON retain per-cell hashes. `summary.json` records counts; `provenance.json` records commands and binding/harness hashes.

FAST ran with explicit `--allow-fast`; its observed repeatability is not a numerical guarantee. These local repeat checks neither compare modes to one another nor establish cross-vendor IDENTICAL qualification. No expected hashes or assertions were relaxed, and no product changes were made for this run.
