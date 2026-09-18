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
