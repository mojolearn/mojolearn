# CPU proof follow-up

Dedicated worktree: ~/mojolearn-wt/cpu-proof-followup.
Branch: lane/cpu-proof-followup. Continues CPU_IDENTITY_AND_MULTI_GPU_PLAN.md.

Completed kernel work was merged to main bd0533295 after all 31 runtime tests
passed; its finished worktree was removed. Native evidence remains outside
Git, and committed records retain CPU/Apple and saved-model runtime evidence.

Current main inventory: 234 routes = 161 default public CPU + 23 ordinary
pending + 50 parallel. Pending is the previous 17 plus six new kernel routes.
The 18 logical CPU parallel implementations are not physical multi-GPU proof.
No target to implement the other 32 parallel routes on CPU. Whole loaded-LM
composition evidence, independent vendor qualification, and exact-wheel replay
remain outstanding; this ledger is not an exhaustive parameter-space proof.

Release run 35350125464: ARM64 clean sweep succeeded, classical UMAP gate and
full sabotage sweep failed. New fixes: byte-LM loader sabotage opt-in in the
workflow; preserve actual mismatching reference-sharded neighbor outputs as
NumericalMismatch instead of discarding them as a generic refusal. Clean lane
hashes and genuine exception behavior stay unchanged. Regression tests cover
both classifier and regressor, each compared output, and worker failure.

The frozen release is NOT updated by this branch. The workflow-only fix can
be cherry-picked independently (41588b491). Changing identity_break.py in the
release changes its qualification tool snapshot: redo required GPU source
qualification before release; do not silently reuse incompatible witnesses.
Current native controls must be rerun after the fixes. Historical ARM64 output
is diagnosis, not a claim that either patch has passed native certification.

Next: finish CPU certification on all architectures; collect fresh independent
NVIDIA/AMD UMAP and pending property evidence after the active CPU matrix ends;
resolve retained failures without weakening refusal handling; then repeat exact
release qualification. Multi-GPU gate/feature work has not landed in this
follow-up: physical UUID/ownership/actual-execution gate precedes claims for
forecasting/CV, loaded-LM and IVF capacity, GPC splitting, and distributed GEMM.

Validation: 85 reporting/orchestration tests passed in 1.80 seconds. These
include all nine new neighbor reporting cases; no native execution is claimed
from this mocked regression suite.

Native targeted rerun PASSED on Apple M4 at 49b747897: all three clean ties
cells were STABLE twice. Fresh native core sabotage gave repeated DIVERGENT
hashes for both sharded-neighbor lanes; fresh native byte-LM sabotage loaded
with the explicit opt-in and triggered RLPAIR_MOVED. The strict sabotage
column checker accepted the recorded controls with zero failures. Raw records,
logs, binding digests, recipe and receipt are committed. No full-fixture or
Linux release recertification is claimed. Session 69681 finished successfully;
all native artifacts remain externally retained after worktree cleanup.

## Multi-GPU continuation checkpoint

The finished CPU worktrees were removed; retained records and external native
artifacts remain. A dedicated lane/multigpu-cv branch now implements bounded
GPU fold scheduling and driver UUID/PCI worker inventory. It remains OFF main
pending real NVIDIA/AMD validation. Simulated worker/driver tests pass (66),
with 10 optional sklearn comparison tests skipped. No physical execution,
capacity improvement or throughput improvement is claimed from those tests.

One independently tested fix is landing on main: DevicePool rejects duplicate
visible-device tokens and invalid later indices before its first child starts.
Its 15 isolated mask/ordering/HIP-filter regression tests pass. This fixes
logical device selection; physical UUID/PCI admission remains in the feature
branch. Preserve that active worktree until GPU qualification is finished.

The frozen release has selective proof-orchestration repairs at 54fd2188c,
not new algorithms. Its tool-source qualification must be refreshed before
publication. Original-source CPU certification is still running; new rentals
remain deferred until that matrix ends.
