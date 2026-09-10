# WP6 / WP7 host transfers — September 10, 2026

Scope: non-tree WP6 and WP7 (DEVIATIONS 2486–2487). Tree bindings,
GBDT, isolation forest and WP8 belong to the other lane.

## Implementation

`bindings/hostptr.mojo` is the shared typed-pointer and byte-copy module.
The non-tree GP, byte-LM, Mamba, SVM, metrics, preprocessing, estimator,
linalg, ARIMA, TSA, solver, training, Transformer and base bindings delegate
their local pointer helpers to it. Each binding migration has its own commit.
Pointer-only migrations do not claim to remove a matrix copy.

GP prediction bulk-copies the quadratic Cholesky factor and removes its unused
prediction-only zero target list. Byte-LM bulk-copies three state inputs and
four outputs, and moves the captured parameter, optimizer and gradient lists
instead of copying them. Moved capture fields are reset to empty lists so the
capture remains destructible. Validation, publication and ownership checks
remain in place. Mamba uses the shared bulk reader for its formerly scalar
arms; SVM support matrices, metric labels, scaler results and KDE inputs and
outputs use the shared copies.

The listed non-tree estimator upload helpers now copy into their existing
pinned allocations with the shared SIMD-8 body. Existing DMA, synchronization
and source lifetimes are preserved. ARIMA's batched-fit helper and spectral's
check-side device I/O already used list copies plus DMA; those are unchanged.

WP7 removes duplicate immutable kernel-method and GP uploads by borrowing the
input buffers and making non-owning sub-buffer views for legacy callees.
Mixture prediction, scoring and responsibilities reuse one precision upload;
the fit's distinct inverse remains distinct. Spectral removes the pinned copy
used only for host normalization. HDBSCAN removes the pre-download sync and
retains the copy-completion sync.

kNN classification retains the search's device index allocation. Host sorting
still runs and sets a flag if it shifts any element. Only that case uploads
the corrected order; classification otherwise reuses the original allocation.
Weighted arithmetic remains in its existing host order. This does not claim
to eliminate the weighted arm's distance download or weight upload.

## Qualification

The shared helper passed 54,944 host bit checks on Apple M4, including signed
zeros, infinities, subnormals, NaN payloads, integer labels, misalignment,
tails and boundary sentinels. Five named pointer/length refusals and same-span
copies passed. The gate lives at `checks/hostptr_check.mojo`; a directory named
`bindings/checks/` shadows the root package during binding compilation. Main's
8d87cc92 fixed that build-path problem; d330a49d is the refreshed GPU baseline.
The earlier `gp-candidate-UNQUALIFIED.patch` is a historical draft, superseded
by the actual source changes on this branch; do not apply it again.

NVIDIA qualification used expiring RTX 4090 rentals and bounded builds/checks,
complete exported-array captures and estimator cards. Classical surfaces run
in all three numeric tiers; neural surfaces run only in IDENTICAL, matching
main's current supported split. Test repairs apply equally to both versions:
Mamba's Torch oracle needs contiguous input for negative-stride fixtures, and
Transformer's tests must export `Array` before NumPy dtype/view operations and
restore cache buffers through supported assignment. Reference tolerances are
unchanged. Missing corpus files and the `einops` oracle dependency were supplied.

The unchanged baseline times out when a process creates a new trainer context
after a stateless Byte-LM call. Stack diagnostics locate it inside the next
`byte_lm_run_configured`/`byte_lm_session_run`; it is not a measured copy-sweep
regression. A resident-only process passed two training steps and evaluation.
Both that sequence and separate training/evaluation calls for each lifetime
mode are captured. The failed mixed/stateless sequences are retained separately
from passing captures. The cause of the context-recreation timeout remains open.

The integrated source passed 88 host tests. All 13 integration output captures
also match the controlled candidate's captures. Across all supported modes,
21 complete output captures (956 recorded arrays), 36 primary stage traces
and two additional cards match before/after. The full IDENTICAL suites pass.

The deterministic and FAST kernel-method suites retain one pre-existing
`POLY_VIA_POW moved NO bit` sabotage failure. Both versions reach their card
checks and emit matching stage fingerprints before the same failure. The two
failure markers and original logs are retained; those sabotage suites are not
reported as passing. No reference tolerance or product arithmetic was loosened.

One early ARIMA trace also contained a nested classical capture's records.
Its ARIMA prefix already matched the candidate exactly; the combined original
and the extra suffix are retained, and the split is recorded in trace-repair.json.
The dedicated classical output captures are compared separately.

## Large measurements

These are native complete calls on RTX 4090, with warmup and five alternating
old/new timed pairs, and complete selected-output bit comparisons.

GP mean prediction from a synthetic preconstructed 20,000-row / one-feature model,
four queries, `return_std=False`, carries a 1.6 GB identity factor buffer. The qualified repeat
used three warmup pairs and took 48 seconds total: median **4.073 s → 1.514 s**
(**62.8% less time**); minimum **3.924 s → 1.423 s** (2.76× speedup).
Baseline spread was 5.1%. The first window's 29.8% spread disqualified it and
is retained. The repeat uses the same two admitted native binaries on a new
RTX 4090 allocation, with its own paired baseline; absolute timings from the
two allocations are not combined. A failed clean-node launch lacked the MAX
runtime libraries; the successful repeat staged their lockfile-pinned ELF closure.

kNN classification used 400k index rows, 4k queries and 32 features; 16 queries
were exact training matches. All 48 native calls passed complete probability
and class-order comparisons. Medians:

| k | weights | before (ms) | after (ms) | less time |
|---|---|---:|---:|---:|
| 10 | uniform | 43.118 | 42.156 | 2.2% |
| 10 | distance | 45.506 | 43.705 | 4.0% |
| 15 | uniform | 46.661 | 44.548 | 4.5% |
| 15 | distance | 48.896 | 46.537 | 4.8% |

Every timed pair favored the candidate. Baseline spreads were 0.35–3.25%.
Minimum times, all samples, source/binary hashes and full evidence are in
[the qualification record](../../bench/results/wp6_wp7_2026-09-10/README.md).
These classification measurements do not revise the cached bare-search kNN
opponent ratio. No opponent was timed, so the opponent cache is unchanged.
Large byte-LM speed and Apple/AMD estimator execution remain RUN OWED; the
host helper test is not a substitute. All scoped rentals were terminated and
the provider confirmed their absence.
