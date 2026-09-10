# RF column-tile evidence

`oracle.log` is the per-cell host oracle and direct negative-control gate.
`timing_*json` contains parsed samples and model hashes. Native executables
are retained locally under `build/rf-column-tiles-timing/`; their SHA256 hashes
are in `timing_binary_sha256.json`, and sources/commands are in the bench tool.
The exploratory four-class run predates explicit compiled tile readback;
its nominal 3% median improvement is below its 6.2% canary variation.
Both 262145-row binary runs are invalid for speed claims: FAST canary
spread1.184 and IDENTICAL1.699. The final validity threshold is1.1.
Earlier raw driver JSON printed the provisional1.5 threshold; the separate
canonical parsed timing JSON applies the final1.1 threshold. All recorded
candidate/reference full-model fingerprints agree within each fixture.
No binary is intended for version control. RF defaults remain unchanged.

`summary.txt` and per-build launch logs record all99 native complete-model /
prediction checks plus route/fallback assertions. `public-fast-columns4.log`
records public classifier/regressor AOT smoke and88 tiled launches. The initial
public smoke loader failure is saved separately; the corrected wrapper sets
DYLD_LIBRARY_PATH to the development runtime before public imports.
