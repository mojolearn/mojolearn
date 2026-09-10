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

## Qualification in progress

The shared helper passed 54,944 host bit checks on Apple M4, including signed
zeros, infinities, subnormals, NaN payloads, integer labels, misalignment,
tails and boundary sentinels. Five named pointer/length refusals and same-span
copies passed. The gate lives at `checks/hostptr_check.mojo`; a directory named
`bindings/checks/` shadows the root package during binding compilation. Main's
8d87cc92 fixed that build-path problem; d330a49d is the refreshed GPU baseline.
The earlier `gp-candidate-UNQUALIFIED.patch` is a historical draft, superseded
by the actual source changes on this branch; do not apply it again.

NVIDIA qualification uses an expiring RTX 4090 rental, bounded builds/checks,
complete exported-array captures and estimator cards. Classical surfaces run
in all three numeric tiers; neural surfaces run only in IDENTICAL, matching
main's current supported split. Test repairs apply equally to both versions:
Mamba's Torch oracle needs contiguous input for negative-stride fixtures, and
Transformer's tests must export `Array` before NumPy dtype/view operations and
restore cache buffers through supported assignment. Reference tolerances are
unchanged. Missing corpus files and the `einops` oracle dependency were supplied.

The unchanged baseline times out on a second Byte-LM native call on this
rental, including a stateless-only process. Stack diagnostics locate it inside
`byte_lm_run_configured`/`byte_lm_session_run`; it is not a measured copy-sweep
regression. One training or evaluation call per fresh process is captured for
both lifetime modes. This does not qualify repeated-call training on NVIDIA.
The failed multi-call runs are retained separately from passing captures.

Large timing gates are native complete calls, not kernel-only measurements:
GP prediction from a preconstructed 20,000-row factor (1.6 GB), and kNN
classification at 400k rows / 4k queries / 32 features, k=10/15 and both
weight policies. Each uses warmup plus five alternating old/new pairs and
complete result-bit comparisons. A baseline range above 20% disqualifies the
timing window. No opponent runs or new opponent ratios are part of this sweep.
Final timing and comparison results remain pending. Estimator execution on
Apple and AMD remains RUN OWED; the host helper test is not a substitute.
