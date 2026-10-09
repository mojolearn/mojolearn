# Host-route fence vs the Oct 6 archive pushes (lane H1, 2026-10-09)

Base: origin/main 03b648834. Checker: tools/hooks/no_host_routes.py at that commit, report mode only
(`--tree <sha>` and `--tree <sha> --branch`). Evidence: ~/mojolearn-evidence/host-route-fence-20261009/
(out/*.log per branch and mode, rc.txt, findings.tsv = every finding, unique.tsv = 54 distinct findings,
main-owed.tsv, main-py-late-findings.tsv, analyze.py, scope_path.py).

## Which branches

The refused pushes are ~/mojolearn-evidence/nomatrix-push-20261008.log: 25 local-only branches (none on
origin), rewritten without experiments/six_lane_integration/matrix.json into refs/nomatrix/* and kept in
unpushed-branches-{original,nomatrix}-20261008.bundle (25 heads, all objects still in the local store).
The refs/archive/branch-prune-20261006/* (378) and refs/archive/worktree-cleanup-20261006/* (66) refs are
outside refs/heads/, which the fence never judged; they are not the 49.

## Why the push was judged on the whole tree

The shared hooks the push ran (.git/hooks/pre-push and .git/hooks/no_host_routes.py in the main checkout) are
stale: they predate the 2026-10-07 `--branch` mode (no `--branch` anywhere in the installed checker), so every
non-main push was judged on its whole tree. Rerun `sh tools/hooks/install.sh`. It would not have saved these
pushes: 20 of the 25 are refused in `--branch` mode too, because their merge-base with main (74967f711,
Oct 6) predates bd4c36dbe ("Publish preserved full A/B snapshot", Oct 6), the commit through which the same
code reached main.

## Per branch (findings: whole tree / --branch)

| branch | tree | branch |
|---|---|---|
| campaign/apple-opponent-quality-20261006, campaign/full-classification-qn-recipes-20261006, fix/apple-opponent-selector-20261006, fix/lars-direct-20261006, integration/six-lane-ab-20261006, integration/six-lane-comparison-scope-fix-20261006, integration/six-lane-full-tsvd-20261006, integration/six-lane-state-capture-20261006, lane/apple-classification-admission, lane/apple-full-classification-prep, lane/apple-six-admission, lane/full-recipe-glue, lane/mlp-full-registration, lane/public-linear-state-capture-20261006, lane/var-native-fittedvalues, repair/amd-full-classification-registration, repair/apple-fast-huber-fold-20261006, repair/cnn-native-orchestration-20261006 (18) | 49 | 49 (the same 49) |
| repair/prep-native-sparse-20261006 | 57 | 57 (the 49 + 8 in python/mojolearn/_sparse_input.py) |
| ideas/trees-identical-20261006 | 18 | 18 (5 mojo-host-loop + the 13 host-lanes/SHAP below) |
| lane/neural-identical-ideas-20261006 | 1 | 1 (training/dev_tensors.mojo `for j in range(len(pool[].n))`, gone from main) |
| lane/idle-retention-60m-20261007, lane/targeted-benchmark-metadata-20261007, lane/targeted-cpu-build-20261007 | 15 | 0 (pass in --branch mode) |
| lane/host-route-source-repair-20261006 | 0 | 0 |
| origin/main (whole tree) | 0 | - |

## The 49, by class (line numbers are the branch's; main's in brackets)

A. 19 mojo-host-loop (owed rule), LIVE ON MAIN, on main's owed list (`--owed origin/main`, 127 rows):
   bindings/_mojolearn_mamba.mojo:559,813 [559,813]; bindings/_mojolearn_training.mojo:1429 [1439];
   core/forest_inference.mojo:212,1170,1236 [262,1230,1296]; ensemble/importance_device.mojo:101 [101];
   ensemble/randomforest.mojo:3438 [3428]; gbdt/methods/leaves_estimation/tree_t26_device.mojo:97 [97];
   mamba/impl/ops/neural_mamba_scan.mojo:195 [195]; mamba/impl/ops/neural_scan_profile.mojo:203 [213];
   training/byte_lm.mojo:2034 [2034]; training/neural_ab_lifetime.mojo:50 [51];
   training/neural_ab_optimizer.mojo:70 [75]; training/neural_attention_owner.mojo:125,131,141,161 [same];
   training/neural_session_mlp.mojo:59 [59].
   Owed rules never enter the baseline: this is fix-lane work (move the loops into kernels or prove they walk
   handles/configs). Several walk metadata (leases, owners, configs, ranges), not rows.
B. 9 Python glue-only findings, LIVE ON MAIN, in NO ledger: main's baseline has no py-* row, so the late-rule
   path grandfathers whatever origin/main already carries, and `--tree origin/main` passes (rc 0) without
   reporting them. Main carries 14 such findings (main-py-late-findings.tsv); these 9 of them:
   python/mojolearn/_training_impl.py:2774,2775,2777,2781 py-data-loop and :2778 py-reduce `sum(rows)`
   [2789,2790,2792,2796,2793]; _byte_lm_config.py:59; _expansion_metrics.py:3184; _mamba_impl.py:927,1807.
   They are argument/shape glue (lists of buffer addresses, session shapes, dataclass fields) without a
   reviewed `# glue: <3+ words>` note, plus one Python `sum()` over per-session row counts. Fix lane, not
   baseline (the baseline never grows): add the glue notes, and have the Mojo mlp_sessions binding size the
   output (or return total rows) so `sum(rows)` leaves Python. The other 5 on main (_expansion_prep.py:6212,
   _mamba_impl.py:1814, _transformer_impl.py:1344,1380,1381) carry `# glue:` notes of only two words.
C. 13 host-threads/host-threshold in core/host_lanes.mojo:28,378,393,394,404,425,434,436,
   core/host_predict_threads.mojo:46,81,82, xtrees/shap_host.mojo:11,59. The code is on main but host-only
   there. The branch puts it on a GPU binding's import graph through two edges (scope_path.py):
   transformer/experiments/summary_model.mojo `from core.host_lanes import host_f32_uninit` (unused, reached
   from bindings/_mojolearn_transformer.mojo) and xtrees/api.mojo `from xtrees import shap_host as shap_cpu`
   (bindings/_mojolearn_x_trees.mojo; the call sits in the `comptime if XTREES_DEVICE_OPS ... else` host
   branch, so the GPU build never ran it). Both edges were on main from bd4c36dbe (Oct 6) to 64cc80087 and
   34bd84cf8 (Oct 7), which cut them. DEAD-BRANCH ONLY now.
D. 8 GONE FROM MAIN (dead-branch only): x_cluster/device_ops.mojo:1458 mojo-host-loop and :2616
   serial-launch `_c42_active` grid_dim=1 (both on main from bd4c36dbe, since removed);
   training/neural_ab_optimizer.mojo:186 one-block-n (main now launches fold_blocks blocks);
   python/mojolearn/_expansion_prep.py:6153, _mamba_impl.py:1814, _transformer_impl.py:1344,1380,1381
   (main has rewritten lines; their main versions are in B's "other 5").

## Verdict

None of the 49 is a GPU-path CPU route we shipped in a release: the last release tag (0.8.36, 2026-10-03)
predates bd4c36dbe, and the only finding that moved data work to the host on a GPU install while on main
(C, the TreeSHAP host import) was compile-gated away from the GPU build and was cut on Oct 7. 21 of the 49
are dead-branch only (C + D), so archiving those branches loses nothing. 28 are the same code main
carries today: 19 owed mojo-host-loop findings already on main's owed list (fix lanes) and 9 Python glue
findings that main carries unledgered (fix lane above). Those 28 must be fixed on main whether or not the
branches are archived, and release 0.8.37 will carry them unless that lane lands first.
The fence is right to stay on lane/* and main. Archived refs never merge, so they go under
refs/heads/archive/<date>/<old name>, which this lane exempts (hook change below), instead of R2 bundles.

## Hook change (lane/host-route-fence-archive)

- tools/hooks/pre-push: refs/heads/archive/*, refs/tags/archive-*, refs/tags/archive/* skip the host-route
  fence with one stderr line ("host-route fence skipped for <ref>: it is an archive ref, and archive refs are
  never merged (the size fence still applies)"); main stays whole-tree, every other refs/heads/* stays
  --branch, salvage/* stays silently skipped, other refs stay unjudged. The size fence runs for every push.
- tools/hooks/test_no_host_routes.py: test_pre_push_skips_archive_refs and
  test_pre_push_judges_every_other_ref_as_before (written, not run).
- CONTRIBUTING.md (Repository size: hooks) ref table; tools/hooks/install.sh rerun note; checker docstring.
- After merge, rerun `sh tools/hooks/install.sh` in the main checkout, or the stale installed hook keeps
  judging every branch on its whole tree and never learns the archive namespace.
