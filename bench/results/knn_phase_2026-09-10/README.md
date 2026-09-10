# Small Apple kNN phase diagnostic

Root command: `MOJOLEARN_KNN_PHASE_SMOKE=1 nice -n 19
tools/with_build_lock.sh pixi run bash tools/knn_residual_phase_probe.sh apple
/tmp/mojolearn-knn-phase-smoke-20260910`. Both builds use2 compiler jobs;
OMP/OPENBLAS threads2. Exit0; two reversed-order full-output comparisons pass.
Inputs:65537 index rows,129 queries,17 features,k10. Controls retain exact
rounding repair. This is not the400k/4000-query target price or an opponent run.

Raw phase rows include warmups and both request/device calls. Summary medians
mix those diagnostic calls explicitly; they are not production timing estimates.
Default/control distance medians were8.28/10.01ms in pass0,4.69/4.17ms in pass1:
large cross-pass drift and a reversed ranking prevent a performance conclusion.
The proposed per-vector exponent-minimum cache is unimplemented; no default
changed. Binary hashes retained; executables remain in/tmp, not committed.
The script now serializes itself and exposes the explicit small-smoke option.
