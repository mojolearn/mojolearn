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

## Four-arm layout follow-up

`bench/knn_layout_dispatch_check.mojo` and
`bench/knn_layout_dispatch_price.mojo` combine the selector flag with
`-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1`. Build with neither
flag, selector only, transpose only, and both. The public correctness driver
adds explicit L2, rooted L2, L1 and cosine cases with transpose tails; its
143,628 selected distance/index pairs must agree across all four arms.
The shared fixtures live in `bench/knn_smallk_dispatch_fixture.mojo` and
`bench/knn_smallk_price_fixture.mojo`, and are part of build provenance.

The full Apple M4 / NVIDIA RTX 4090 campaign at `9fe07a33` passed strict
correctness and pricing-output comparison. Each vendor ran nine rotating
rounds per arm at 32, 128 and 1,000 queries. NVIDIA's combined arm reduced
the largest request's median from 17.241 ms selector-only to 9.273 ms;
Apple did not reproduce that gain and regressed on the smallest request.
See the [full comparison](../../bench/results/resume/2026-09-05-layout-apple-price/README.md)
and its raw samples. These results do not justify changing all-vendor defaults.
AMD, broader data distributions and installed-wheel qualification remain open.

The four-arm parser accepts ordinary runtime diagnostics before the first
benchmark record, while requiring `LAYOUT_FLAGS` to be the first protocol
record, complete ordered cells and final completion markers. It compares
every distance bit and index before admitting timings. An empty controller
console is valid when child logs contain all output and the payload succeeded;
a missing console or failed/missing payload status remains a refusal.
