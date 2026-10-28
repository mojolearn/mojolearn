# Lane: integrator (provider migration READY queue -> main)

Read ~/mojolearn-evidence/briefs-20261028/COMMON.md first. You are the ONLY agent allowed to mutate main.
Work in a NEW worktree: `git -C /Users/andrewhendel/CascadeProjects/mojolearn worktree add ~/mojolearn-wt/integration-provider-queue-20261028 -b integration/provider-queue-20261028 main`
(if disk is tight use tools/lean_worktree.sh). Evidence dir: ~/mojolearn-evidence/integration-provider-queue-20261028/.

Merge these GREEN, compiled, pushed heads IN THIS ORDER (each has a compile receipt named in the root handoff;
verify with `git merge-base --is-ancestor` / `git cherry` that the content is not already in main before merging;
the CTR+ranking GBDT sets ARE already in main as cherry-picks, skip those):
 1. aefc4545aa  (lane/peer1-native-next-20261027; includes 33a63ba6ad). NOT the branch tip 4ad607eff5: the 7 later
    descriptor commits are unqualified. Merge exactly aefc4545aa.
 2. 0456ee3e6a  cluster refit lifetime (lane/cluster-refit-lifetime-20261027)
 3. 0f0b336d87  archive label/kernel admission (lane/archive-label-kernel-admission-20261028). Contains BOTH
    68f5140002 (archive schema) and 5e7c530779 (combined GP). Merging it covers queue items 3 and 9.
 4. a3a8150111  linalg helper (lane/linalg-buffer-boundary-20261028) -- must precede 5.
 5. 32276a868c then cca559b8a0 (row resample), then 6be526212e, then 74c1e58ebe (metrics) -- all on
    lane/peer6-metrics-20261027; merge 74c1e58ebe (it contains the earlier ones). NOT the tip de7fea9506.
 6. f160cd1283 ARIMA execution, then e456212ca7 VAR/STL, then 4b9999bee4 companion (lane/peer6-timeseries-20261027
    tip = 4b9999bee4; merging the tip covers all three; e456 must never land without 4b99).
 7. 654531acce Program owned distributed (lane/program-distributed-boundary-20261027)
 8. 7cacec39ba SVM state/coef (origin/lane/svm-state-coef-20261028)
 9. 8e67e1bce1 Peer4 Array.from_buffer (lane/peer4-public-boundaries-20261028 tip; includes backend 4ce6e5c2e7 and
    numeric-profile 82fde80801 equivalents fcee3d4bfc+ff81abe394; if those children conflict, keep the parent's version).
10. Peer2 batch 8: c7694b3277, 9a0f043190, 36a66a6885, 2f3e60c2ca (on origin/lane/peer2-preprocessing-decomposition-manifold-20261027).
    Cherry-pick exactly these four; EXCLUDE e8a10* (prep) and 808a23* (temporal helper) and 9608dee4e5 and the rest of that branch.
11. 77ed135648 GBDTAdapter params (lane/peer5-tree-aux-20261027 head; the worktree's 7 dirty files are NOT part of it).
HOLD (do not merge): Prophet b00f153308; anything from lane/peer3-*; lane/peer4-mode-boundaries-20261028; prep 9608dee4e5.

Conflict policy: these lanes share bindings/_native_public_attributes.py, public_attributes_native.mojo,
classical archive/schema files, native_codec.mojo, _mode.py. Resolve by PRESERVING BOTH sides' additions (the handoff
says every owner was notified of disjoint additive schemas); never drop one lane's slots/entries. Never replace a whole
shared file with one side's version. When a conflict is not clearly additive, read both lanes' HANDOFF receipts under
experiments/ before choosing, and record the decision in the evidence dir.

Then compile ONCE, on a rented RunPod CPU box (32 vCPU class, `bash tools/runpod_cpu_leg.sh --lane integ-queue ... --rent`,
--worktree your worktree; dry-run first). Build the affected bindings: collect the entry points named in every merged
head's compile receipt (experiments/*/compile*.json, HANDOFF.txt), dedupe, and build each at O0 identical CPU
(see COMMON.md command), plus any binding whose Mojo sources changed vs main (`git diff --name-only main...HEAD -- '*.mojo' bindings/`)
and is not covered. If a full CPU host binding build (tools/release_host_build.py / tools/merged_bindings.py; read their
usage) fits in the lease, run it too and record it; otherwise record exactly what was and was not compiled.
Fix compile errors in the integration branch with minimal coherent edits (commit each fix separately with the error as the
message body); rebuild the failing entry. Retain logs + binaries + source hashes on the Mac, then reap the pod.

When every built entry is green: fast-forward or merge integration -> main in the main checkout
(/Users/andrewhendel/CascadeProjects/mojolearn, which must still equal origin/main; `git fetch` first; if origin/main moved,
merge it into your branch and recompile only the entries whose inputs changed), run the normal push (`git push origin main`;
if the hook names a wide rebuild, `MOJOLEARN_WIDE_REBUILD_OK=1 git push origin main > push.log 2>&1` is the documented
acknowledgement, not a bypass). Record the push log and exit code. Append a receipt to PROVIDER_MIGRATION_HANDOFF_2026-10-28.txt
("INTEGRATION 2026-10-28 (Claude integrator)": merged heads, compile receipts, main sha) and commit+push it.

If something cannot be made green, leave it OUT of main (revert that merge on the integration branch) and report it
exactly; ship everything else. Write ~/mojolearn-evidence/integration-provider-queue-20261028/HANDOFF.md as you go.
