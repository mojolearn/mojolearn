# WP3 resident inference boundary gate

**PASS**: six native matrix cells on Apple Metal: FAST, DETERMINISTIC and IDENTICAL, each with separate-array and packed-sibling model layouts. Every binary reports the expected compiled numeric mode, vendor `metal`, and layout flag; `provenance.json` records these witnesses and source/binary hashes.

Each cell checks RF and ET semantics at output widths 1/2/3/5/8/9, comparing complete Float32 output bits from the retained List path with both borrowed-pointer paths (temporary and reused I/O). Ragged 33-tree forests, root leaves, nonzero child IDs, row/grove tails, subnormal features and poisoned internal leaf slots are included. The packed layout's mutated-device-leaf negative control must change the output.

Workspace checks exercise repeated shape, changing input, resize, empty input and close against an independent split oracle. Existing release/stale/new-handle, invalid shape/graph, nonfinite input/output and destruction checks also pass.

Command: `nice -n 19 tools/with_build_lock.sh bash tools/check_forest_resident_layouts.sh bench/results/boundary_tax_2026-09-10/wp3`.

Build/run logs are retained beside this report. Executables were hashed and moved to `/tmp/mojolearn-boundary-wp3-binaries/`; none are tracked. This is small-fixture correctness evidence, not a speed measurement or cross-vendor qualification. The test validates the existing default path; it does not establish a new default change.
