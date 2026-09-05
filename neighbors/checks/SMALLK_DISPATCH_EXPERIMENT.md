# Opt-in small-k selection inside exact k-NN

The production default is unchanged. Compile an IDENTICAL build with:

```sh
tools/with_identical_mode.sh pixi run mojo build -I . \
  -D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1 \
  bench/knn_smallk_dispatch_check.mojo -o /tmp/knn-smallk-identity
```

Omit the define for the baseline. Setting it to `0` still enables a
presence-based `is_defined` flag. Setting an environment variable alone does
not pass this define to Mojo. Existing binding build scripts do not forward
this experimental define; a main-lane experimental extension build must add
the flag explicitly to its compiler command and retain that command/hash.
No release build should add it before end-to-end qualification.
The driver calls public `knn_search` for K8/K10/K16 and fallback K4/K15,
with 1/257/1000 queries and dyadic/duplicate-heavy data. Its second call uses
a different query tile and checks every output bit. Build it again without
the define and compare all 133,348 `DISPATCH_CELL` records, requiring both
completion markers and opposite `SMALLK_DISPATCH experimental` headers.
The older `knn_identity_check.mojo` uses K2/K4 and checks fallback behavior;
it cannot prove this specialization was reached.

The experiment changes only the selector inside `tiled_brute_force_knn`:
IDENTICAL mode, non-vendor-top-k path, `k` exactly 8, 10 or 16, and
`k <= n_index <= INT32_MAX`. It directly launches the corresponding fixed-K
candidate with the existing `q*k` output offsets and a 256-thread block.
Other K values, short index rows, FAST, DETERMINISTIC, explicit vendor top-k
and fused paths retain their prior routes. The outer loop still advances by
the actual query-tile row count. Norms, distances, sqrt and composite-key
ordering are untouched. Candidate results still require byte qualification;
the launch change is not itself a certificate.

Main-lane qualification should compare full public k-NN distances and indices
against an unarmed build, including K8/10/16, fallback K3/9/15/17, ties,
different metrics, and query counts 1/32/128/255/256/257/1000. Vary query tile
size so the second tile, nonzero output offsets and final partial tile run.
Preserve refusal and explicit alternate-method tests. Compare all three
vendors before promoting any IDENTICAL dispatch; run FAST/DETERMINISTIC
regressions to verify the experiment cannot reach those modes.

Price the actual request, with input transfers and outputs included, on
representative, duplicate-heavy and tightly clustered data. Keep the legacy
scratch allocations in this first experiment: removing them changes another
part of the request. `neighbors/estimator.mojo` allocates radix scratch using
`buf_len=max(n_index//8,k)`; the isolated selector benchmark uses a larger
buffer. Its scratch figures and component speedups are not public-API results.
No workspace reduction or end-to-end speedup is claimed by this patch.
