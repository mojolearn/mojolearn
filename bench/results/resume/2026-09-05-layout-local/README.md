# Local k-NN layout continuation

All four IDENTICAL public correctness arms passed on Apple M4 at dirty
`323b7301`: baseline, selector-only, transpose-only and both. Each arm's
133,348 dispatch pairs and 10,280 additional metric pairs matched every
distance bit and neighbor index. The existing strict four-arm parser checked
activation flags, fixture order, completeness and completion markers.

`results.json` records source hashes and output hashes. Compressed build logs and plain check logs preserve raw output. `source.patch.gz` plus the two extracted fixture
files retain the relevant changes relative to the named commit. The fixture
extraction was also checked against HEAD: only entry-point names changed.
Seven existing Python comparator tests passed, shell syntax passed for the
affected controllers and benchmark scripts, and `git diff --check` passed.

Each correctness binary was built with `pixi run mojo build -j 2 -I .
-D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_layout_dispatch_check.mojo`, adding
`-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1` for selector/both and
`-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1` for transpose/both.
Builds and executions ran serially under `tools/with_build_lock.sh`.

This is local source correctness evidence. It does not qualify remote
hardware, installed wheels, FAST/DETERMINISTIC regressions or performance,
and does not justify changing the production dispatch default.

All four timing drivers also compiled successfully with the same flags and
`bench/knn_layout_dispatch_price.mojo` as the entry point. Their compressed
build logs are retained; no timing driver was executed in this continuation.
