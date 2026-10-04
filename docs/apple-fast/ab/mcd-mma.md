# MCD main-MMA arithmetic repair

Branch lane/apple-fast-mcd-mma, base main d4bb2b795. Follows process in
EXPERIMENT_PROCESS.md. Prior scalar COMPAT failed capped quality at
ab4265c9a, tag gap26-mcdrepair-small-ready; failures remain documented.

Candidate define MOJOLEARN_MCD_BATCH_MMA, opt-in Apple FAST only, binding
x_decomp. It requires main's DECOMP_FAST_GEMM_MMA to be enabled; adding the
existing MMA_OFF define disables this experiment instead of changing baseline
arithmetic. Wide d>64 and d<2 still use the existing main path.

The code retains batched support compaction, column means, round-robin
Jacobi eigensolves, candidate stopping/ranking and support selection. It
replaces the scalar candidate covariance, weighted-eigenvector Gram and
Mahalanobis products with calls to the SAME _launch_gemm_mma used by main
DKit.mm. Each candidate keeps its own dimensions and K-split dispatch; calls
are queued without intermediate host reads. Centering, reciprocal scaling,
publication and distance row sums remain parallel GPU kernels. Inactive
candidate state is preserved by guarded publication; inactive MMA inputs
are zeroed. No host computation of model state or CPU fallback is added.

This conservative version enqueues one GEMM per candidate rather than inventing
a new batched GEMM reduction order. It may sacrifice some of COMPAT's speed
while eliminating per-candidate synchronization. Speed is UNKNOWN. Matrix
scratch adds approximately8*nc*r*d+12*nc*d*d bytes per phase. For the full
n100000 taxi phase B(nc about3330,r1500,d11), this is about440MB; concurrent
live phases may increase peak memory. Main's split-K atomics may still vary
last bits; no bit equality or quality pass is claimed before M3 evidence.

Actual saved cap3000 artifacts: d11,h1506, raw ranks both10 but final pinvh
ranks7(A) vs6(B). A's retained eigenvalue.000444677 disappears in B; precision
norm2249.28 vs58.83. Both satisfy their own C*P*C residual below5e-7. Raw
support differs338 entries; B final support1961 is a subset of A2083.
This is selection/model drift, not an isolated final precision inconsistency.

## Manager validation plan

Compile two FAST x_decomp arms with empty defines and
-D MOJOLEARN_MCD_BATCH_MMA. Verify binary manifests and exact checkout.
No local compile or GPU test was run by the lane.

Use a NEW quality tag with tools/mcd_compat_ab.sh, MCD_DEFINE=MOJOLEARN_MCD_BATCH_MMA,
MCD_A_SO and MCD_B_SO pointing to verified prebuilt arms. Start taxi cap3000,
then EllipticEnvelope on the same cap only if MinCovDet passes. Saved baseline
A.npz may be reused via MCD_BASELINE_NPZ after manager review of relevant source
changes; comparator still enforces dataset/lane/shape/data_sha equality.

MCD_DEFINE=MOJOLEARN_MCD_BATCH_MMA MCD_A_SO=... MCD_B_SO=... \
  bash tools/mcd_compat_ab.sh NEW_TAG taxi min-cov-det 3000

Existing gates remain: fitted location/covariance/precision/raw quantities/
query and training distances relative difference<=.01, flags/support/raw
support Jaccard>=.99, equal raw rank. Both flag masks being all true is
insufficient. Keep full timing gated until capped fitted state passes; run
only one full-size scored fit per arm, and do not rerace opponents. If quality
still diverges, instrument the first candidate/step covariance/determinant/
reciprocal mask before revising kernels again. This candidate changes no
thresholds, seed streams, tie rules or selection/stopping semantics.

## Serial queue progression after capped MCD PASS

Measured kernel77520069e compiled both arms and passed gap26-mcd-mma-small:
location/raw covariance exact, covariance_rel4.39e-8, precision_rel1.09e-7,
distances about1.5e-7, support/raw support/flags exact. This establishes capped
MinCovDet quality only; no full-size speed claim follows.

New helper tools/mcd_mma_serial.sh is intended ONLY for serial M3 CMD jobs.
It verifies prerequisite artifact lane/size and re-runs the artifact comparison
(no fit), verifies the existing77520069e binary manifests/source scope, stages
arms within that queued job, then runs one fit per missing arm. It refuses
existing output artifacts and writes PASS only after the comparison succeeds.
All post-fit quality computation and NPZ compression remain inside that same
serial job. It does not launch cloud work or edit the queue itself.

The cap3000MCD artifact paths below are confirmed from local queue receipt
sync/queue-mcd-mma-0912.json (inner wrapper tag matches outer queue tag).
Use new unique tags after checking they have not already produced artifacts.

1. Capped EllipticEnvelope, after existing MCD cap3000 artifacts pass:

   bash tools/mcd_mma_serial.sh ee-small gap26-mcd-mma-ee-small \
     "$HOME/afc-def/gap26-mcd-mma-small/A.npz" \
     "$HOME/afc-def/gap26-mcd-mma-small/B.npz"

2. Full MinCovDet, only after the previous EE cap3000 pair passes:

   bash tools/mcd_mma_serial.sh mcd-full gap26-mcd-mma-full \
     "$HOME/afc-def/gap26-mcd-mma-ee-small/A.npz" \
     "$HOME/afc-def/gap26-mcd-mma-ee-small/B.npz" \
     "$HOME/afc-def/gap26-mcdcompat-taxi/A.npz"

3. Full EllipticEnvelope, only after the full MinCovDet pair passes:

   bash tools/mcd_mma_serial.sh ee-full gap26-mcd-mma-ee-full \
     "$HOME/afc-def/gap26-mcd-mma-full/A.npz" \
     "$HOME/afc-def/gap26-mcd-mma-full/B.npz"

Each is a distinct serial CMD on this lane branch after root syncs the remote
ref. A failed prerequisite exits before staging or fitting. There is no second
timing pass: fit_ms and fitted-state quality come from the same single fit.
EE additionally checks actual offset_ and decision_function using the existing
1% numeric bound; all previous support/flag/rank gates remain unchanged.

### Existing baseline reuse review

Source review through current main12fdd6697: since88a94a86f, x_decomp,
_mojolearn_x_decomp, _expansion_decomp and bench_board_algos have only four
comment lines changed in mcd_fast. Other main changes are ARIMA, label prep,
HDBSCAN/KDE/comments and management/docs; none changes numeric MCD fitting.
Therefore the saved full main A at gap26-mcdcompat-taxi is eligible for QUALITY
reuse conditional on its original binary/source provenance and exact fixture
hash. New verify-baseline preflight loads current fixture and checks dataset,
lane,shape,data_sha BEFORE running B; its mismatch aborts with no new fit.
The capped MCD A cannot substitute for full data, and MinCovDet A cannot serve
as EllipticEnvelope A. No old EE baseline is assumed.

The old70.878s A timing is not automatically certified isolated. Root must
review its run window against the maintenance audit/provenance before using
it for a speed verdict. If isolation remains uncertain, keep speed
HOLD-measurement while retaining the quality comparison and new B timing;
do not silently rerun that already-measured A. A fresh scored A replacement
requires applicable user authorization. One pair does not estimate noise.

Existing compiled77520069e arms remain reusable: revisions after that SHA
modify tools/docs only. verify_arms.py enforces the numerical-source equality.
No new compilation, scored work or remote scheduling was performed by this
preparation task.
