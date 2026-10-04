# STATE 2026-10-04 ~22:25Z — Claude orchestrator (IDENTICAL speed, CPU-out-of-GPU, PTX). READ FIRST TO RESUME

## Branches (all on origin)
- **Integration:** `integration/identical-all-20261004` @ 0d4e49c8f (or newer). Worktree `~/mojolearn-wt/identical-all`.
  Contains: main ffb7f89ff (135 behind newer main), PTX work, 9 idn speed lanes, fam/fam2 passes, no-bench-tuning 1+2,
  fix-* (10), review-fixes, compile-fix (up to its merge time), cpu2-l1..l12, nr-gemm/mamba/attn/small,
  ml-k3-amd-fix/gbdt/prep-nb/cluster-nbrs. AMD portable path REVERTED (parked on lane/amd-portable).
- **Not yet merged into integration (merge these next):**
  - `lane/compile-fix` (head 45617e193+): compile fixes after the merge; still running import/GPU smoke on the 4090.
  - `lane/cpu3-{gbdt-a,gbdt-b,bindings,python,trees,neighbors,seq,core}`: round-3 CPU removal (owed checker findings).
    cpu3-bindings DONE (a15cf85eb). Others were running; each told to commit+push WIP.
- Every lane brief: `~/mojolearn-evidence/briefs-2026-10-04/*.md`. Lane reports: `~/mojolearn-evidence/fam/*.md`,
  `~/mojolearn-evidence/candidate-audit/additions-*.tsv` (recipe/inventory rows to add to docs/identical JSON).

## Boxes (all HELD, all billing; harvester PIDs 98987 + 71989 on this Mac copy /root/lq every 30 s, renew leases)
- Blackwell RTX PRO 6000 `ssh -p 10652 root@64.119.209.250` (RunPod 54zn9tbvoqw70h, ~$1.69/h): wave-ops identity rerun r1.
- AMD MI325X `ssh root@162.243.193.137` (DO droplet 606074135, ~$3.80/h): wave-ops identity rerun r1.
- RTX 4090 `ssh -p 14625 root@213.173.109.16` (RunPod k1tnp0gtira6bq, ~$0.77/h): compile-fix builds, /root/lq/compile-fix-18e813f2bef5.
- If stopping for a long time: release boxes (owner decision; boxes self-renew via the harvesters). Approved but NOT done:
  extra 4090 + extra MI325X for the compile/ID batch (`~/mojolearn-evidence/PENDING-rentals-2026-10-04.md`).
- Box watcher: `bash ~/mojolearn-evidence/box-watch.sh` (run via Monitor).

## Wave (frozen source a4d01a130, /root/lq/medium-wave-a4d-qualified-v2 on both boxes)
- Quality: PASS on both (repaired supplements). DART adjudicated PASS; predictions byte-identical NV vs AMD.
- Identity attempt 2: INCOMPLETE_OR_FAILED on both (282 steps). wave-ops running rerun `r1` of affected cases.
  wave-ops was asked to write `~/mojolearn-evidence/wave-ops-state.md`.
- Timing: NOT run. Rule: one warmup + one measured per arm, ON vs MOJOLEARN_IDN_ALL_OFF, our build only, opponents from store.

## Compile
- compile-fix on the 4090 over head 18e813f2b: 255/285 green first pass; ivf/solver/x_ann/x_cluster fixed (reserved name `out`,
  comment marker). Results `~/mojolearn-evidence/compile-fix/`. Everything merged after 18e813f2b is UNCOMPILED.

## CPU in GPU paths
- Checker extended (tools/hooks/no_host_routes.py: `mojo-host-loop`, `py-native-host`, `--owed`). On 0d4e49c8f: 1,077 owed
  findings (`~/mojolearn-evidence/consolidate-1/owed.tsv`, per-lane split in `owed/`), baseline debt 1,187 rows.
- Not done (follow-ups): estimators taking device-resident input (CV still downloads per fold); spectral affinity
  round trip; sparse precomputed graph lists; NearestCentroid priors; tokenizer encode on device (no GPU tokenizer binding).
- Decided CPU-only (orchestrator, owner-delegated): tokenizer BPE training. PCA `mle` rank -> device soft-f64 (in cpu3-python).

## Reviews done (read before more work)
- `~/mojolearn-evidence/review-bits-2026-10-04.md` (fixed by review-fixes), `neural-roadmap-review-2026-10-04.md`,
  `ml-trees-roadmap-review-2026-10-04.md`, `cpu-reaudit-2026-10-04.md`, `candidate-audit-2026-10-04.md`,
  `identity-conflicts-2026-10-04.md`, `no-bench-tuning/inventory*.md`.
- Owner decisions PENDING: L14 (defer), K7 (approve w/ recall gate), X14 (reject), L8 (reject) — reviewer recs, owner not answered.
- Test debt: sabotage patches seam_5540, x_cnn 5701/5704/5711/5712, e2e_host_output_bit no longer apply (tests can't fail);
  target-encoding sabotage patches old code; no seam for MLP device epoch. Assigned to cpu3-seq / cpu3-python.

## Next steps, in order
1. Merge lane/compile-fix + finished cpu3-* into integration; resolve conflicts; push.
2. Re-run `python3 tools/hooks/no_host_routes.py --owed` for the remaining count.
3. Compile batch of the merged head (4090 + approved extra boxes), fix errors.
4. Same-bits ID checks NV/AMD (+Apple later), then timing A/B, keep/drop, board update of kept rows (L40S for full-size).
5. Freeze, final PTX batch (4090/H100/A100). Publishing out of scope.

## Rules (memory has them): no Apple machines; grep logs; same bits across vendors within a version; no CPU in GPU paths;
no benchmark-shape tuning; wait for Mojo/Modular rather than workarounds; Opus for subagents; max 20 concurrent subagents.

## Update 22:40Z
- Done and pushed (not yet merged into integration): lane/cpu3-bindings a15cf85eb, lane/cpu3-seq 2dcea21c6, lane/cpu3-python 6871dbf16, lane/compile-fix 45617e193+.
- cpu3-python left 34 py-native-host findings with reasons (~/mojolearn-evidence/cpu3-python/); cpu3-seq left AutoARIMA resident series.
- Orchestrator decision: keep cpu3-python removing the Apple FAST kfeat LAZYW host draw (device draw instead) under the no-CPU-in-GPU-path rule.
- Still running at 22:40Z: cpu3-gbdt-a, cpu3-gbdt-b, cpu3-trees, cpu3-neighbors, cpu3-core, compile-fix (smoke), wave-ops (identity r1).
- lane/cpu3-gbdt-b 19d215832 done (17 left: combination-CTR pipeline is host lists end to end; REAL CPU route remains: gbdt/train.mojo calls host compute_simple_ctrs in the fit for counter_calc_method=Full or IDENTICAL >=2^24 rows -> needs resident device CTR columns redesign).
