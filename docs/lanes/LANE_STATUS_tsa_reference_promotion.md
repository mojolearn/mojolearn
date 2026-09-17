# CPU TSA reference admission and default promotion

2026-09-17; resumed from `docs/VERIFICATION_RESTART_PLAN.md` at `1752ab7c4`.

`select-d` and additive `holtwinters` now carry strict historical references
for all nine fixtures and join default public CPU verification. The generator
admits 18 cells and 72 parts from the retained Linux clean column at
`73a5af1734e24b872274c8830918c155e9534bd6`. Every admitted part has two matching
samples and the required input/protocol witnesses. Neither lane has a conflict.
The same record's native sabotage controls cover training on all nine fixtures
for each lane. No new native implementation or native-control claim is made.

The scoped admission preserves all 1,638 existing cells, their record indices,
and their hashes. Whole-table regeneration would change 356 reference values
and remove 121 parts, so it was reviewed as staging only. The shipped table
retains its legacy admission label; this batch does not upgrade old evidence.
`bench/results/tsa-reference-promotion-probe/2026-09-17/admit.py` reproduces the
addition from the immutable baseline using the strict generator, with no typed
reference hashes.

Installed-wheel replay exposed a concurrent packaging omission: importing
`mojolearn` imports the new `models` subpackage, but neither wheel builder
included it. The setuptools list now declares `mojolearn.models`; Linux derives
its Python payload from that same explicit list. A packer regression checks the
model modules and exclusion of tests. This closes importability, not qualification
of the new model loader on real checkpoints or new native bindings.

Validation: 266 focused tests plus two subtests, the generated matrix check,
and installed CPU development-wheel replay outside the checkout with no host,
harness or model overrides. Both lanes run all nine fixtures twice: 54 IDENTICAL,
36 explicit N/A, zero divergent, refused or owed parts. The OLS comparator
self-test passes. Coverage retains the legacy label, has a current harness
snapshot, exposes all 18 historical native controls, and includes both lanes
in default CPU selection. CPU availability is 128, with 51 withheld and 50
parallel exclusions; the harness still registers 229 lanes and maps the 246
appendix entries.

The first isolated-venv attempt exposed missing runtime dylibs in the reused
development-wheel staging. That failed report is retained; the rerun stages and
statically verifies the complete runtime closure with the existing release
helper. The successful development wheel reuses Mac native bindings and is not
a fresh native release build. Its receipt records the wheel and binding digests.
A broad default CPU drift test was stopped and rerun scoped to these two lanes;
there was no Metal run or cloud rental. Local numerical work was single-threaded.

Remaining: strict regeneration/replay of the whole table and optional properties,
51 withheld CPU lanes (including unwatched samba and gbdt-yeti-rank), native
controls for the remaining appendix entries, the eleven parallel lanes without
historical pairs, and final Linux/macOS artifact qualification. No PyPI release.

Before integration, origin/main advanced to `ac49b36c5` with concurrent kNN
and forest-groves work. It was merged and all 266 focused tests plus two
subtests and matrix consistency passed again. The installed-wheel receipt
qualifies the scoped development artifact at `032325dc5`, not those concurrent
native changes. Their hardware qualification remains with their own records.
