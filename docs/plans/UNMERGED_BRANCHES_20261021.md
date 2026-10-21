# Unmerged remote branches: triage (2026-10-21, lane triage-old-lanes)

Read-only audit. Nothing was deleted, merged, built or tested.

Base: origin/main `5158fd89d`. Set: `git branch -r --no-merged origin/main | grep -E 'origin/(lane|fix|feature|integ|board|finish|archive)/'`,
which gives 330 branches (not ~40). `integ/20261021` is not in the set: it is merged.

## Method

For each branch:

1. **Commits:** "ahead" is `git rev-list --count origin/main..<branch>`. The figure in brackets counts commits with no patch-equal
   commit on main (`git cherry origin/main <branch>`, "+" lines).
2. **Files:** the files touched by those unique commits (or by the whole branch diff when every commit is patch-equal), with
   the top directories shown.
3. **Content check:** the added code lines (.mojo/.py/.sh, at least 25 characters, trimmed) from the unique commits, looked up
   in the set of every line in main's tree. At 90% or more the branch counts as merged by content.
4. **Defines:** the `MOJOLEARN_*` defines each branch adds, checked for presence in main's code and for verdict rows in
   `docs/apple-fast/EXPERIMENTS.md`, `docs/TOMBSTONES.md`, `docs/apple-fast/*` and `docs/plans/*`.
5. **Prior audits:** a search of the docs for the branch name. Three earlier audits cover most branches:
   `docs/identical/candidate-audit/candidate-audit-2026-10-04.md` (CA), `docs/plans/host-route-fence-2026-10-09.md`
   (the archive/20261006 pushes) and `docs/plans/FINISH_LEFTOVERS_20261021.md` (the finish/* lanes).
   There is no `docs/apple-fast/LEDGER*.md` on main; the Apple FAST ledger lives only in the evidence directory.

Verdicts: **a** merged by content (patch-equal, lines present, or the define/feature is on main); **b** documented loser,
superseded experiment, finished measurement harness or abandoned tooling; **a/b** an Apple FAST batch lane whose winners were
ported and whose losers are documented in TOMBSTONES/EXPERIMENTS; **c** unfinished work main does not have; **d** unknown.

Counts: a=136 a/b=25 b=164 c=1 d=4 (330 total).

Recommendations: **retire** = safe to delete. **keep** = the branch carries code that exists nowhere else and a doc cites
its sha. Before retiring any branch whose sha EXPERIMENTS.md or TOMBSTONES.md cites as "recoverable at", push an
`archive/<branch>` tag at its head. Many Apple FAST rows cite lane shas, not patches.

## (c) branches: work main does not have

- `lane/apple-fast-bpe` (2026-10-03, 9 ahead): Apple FAST device BPE training and encoding (`tokenizer/fast/bpe_device.mojo`,
  `bindings/_mojolearn_tokenizer_fast.mojo`, `python/mojolearn/tokenizer.py` routing) behind `BPE_TRAIN_DEVICE`,
  `BPE_ENCODE_DEVICE`, `BPE_LIVEBUF`, `BPE_GROUP_FILTER`, `BPE_MERGE_BATCH` and `BPE_ALL`. EXPERIMENTS.md:34 lists it under
  Untried candidates as "NOT on main", never measured. Porting it means merging main into the branch (it is 7358 commits
  behind), compiling on the M2, then running the bpe-train and bpe-encode enwik8 A/Bs on the M3. The same code also sits inside
  the aggregate lanes `lane/apple-fast-batch` and `lane/apple-fast-af-all`. Port from apple-fast-bpe only.

The (d) branches may also hold work main lacks, but no doc gives a verdict:

- `lane/lowbit-apple-tuned`, `lane/lowbit-e2e`, `lane/lowbit-quality` (2026-09-29): int15 Apple tuned GEMM checks and price
  bench, low-bit end-to-end probes, and a low-bit quality bench (`gemm/checks/gemm_int15_apple_tuned.mojo`,
  `bench/lowbit_quality/*`, `tools/lowbit_e2e/*`). None of these files is on main. Research and measurement code, not library
  runtime. Resolved below: ported as tooling.
- `lane/mamba3-statepass` (2026-09-09, 23 ahead): Mamba3 SISO direct transfers and tiled QKS/YSTATE state passing with H100
  evidence. Its defines are absent and its file `mamba/impl/mamba_ssm/ops/mamba3_siso.mojo` is gone from main (Mamba3 now
  lives in `autograd/kernels/mamba3.mojo`). Resolved below: it is on main (16b650a67).

## Notes

- **finish/\*:** all 16 are (a) per FINISH_LEFTOVERS_20261021.md, which already asks to retire them.
- **archive/20261006/\*:** 23 of the 25 are (a): for the integration-era ones, host-route-fence-2026-10-09.md:21-22 says their code reached
  main through bd4c36dbe, and the three late tooling branches (idle-retention, targeted-*) are patch-equal on main. host-route-source-repair and neural-identical-ideas are (b).
- **Measurement harnesses:** the `*-measure`, `stage-diag`, `merged-check` and similar branches are (b). They are one-off
  `pr*_job.sh` / `cell_ab_job*.sh` runners that `lq` replaced, and CA classes them "do not merge".
- **Review chains:** in the `-r1..-r5` Apple FAST chains (callpath, catalog-gemm, shared-downstream, mcd-ordered, arima-scalar-k3),
  each earlier link is a strict prefix of the next, so only the tip that EXPERIMENTS.md cites needs keeping.
- **fp-contract-off-study:** its study README (contract=off moves 16 of 627 cells) is not on main. Port the README if the result
  should stay findable.

## Resolved

Lane `port-lowbit-lanes`, 2026-10-21. Each (d) branch below was read commit by commit against today's main.

- `lane/lowbit-apple-tuned` @ 4d265d747: **(1) checks and a price bench, ported.** Tuned Apple float-unit forms of
  int15i64.v1 (FOUR, THREE, TWO), their gate and clock; nothing was dispatched. Main's `_numeric_profile.MEASURED`
  already cites its job 1 results. Ported to `gemm/checks/gemm_int15_apple_tuned{,_check,_dev,_threads_probe}.mojo`,
  `gemm/checks/gemm_apple_devload_probe{1,2}.mojo`, `bench/gemm_int15_apple_tuned_price_main.mojo`,
  `tools/lowbit_apple_tuned/{job.sh,table.py}`, tables, digests and job notes under `bench/results/lowbit_apple_tuned/`
  (no logs or speed dumps), progress file `docs/studies/lowbit/APPLE_TUNED_INT15_2026-09-29.md`. Adapted: the profile
  bound, `OP_*`, the host sabotage switch and the value flip now come from `gemm/contract.mojo`, and both tuned stores
  carry main's exponent arm (`MOJOLEARN_INT15_EXPONENT_SABOTAGE`), as `gemm_int15_apple.mojo` does. Uncompiled on main.
  Owed: the tuned check does not call `check_int15_row_scales`, so the exponent arm has no gate line in `job.sh` yet.
- `lane/lowbit-e2e` @ b53c7d168: **(1) a public-call probe, ported.** `tools/lowbit_e2e/{job.sh,public_probe.py,model_probe.py}`
  (fixed15_v1 against fp32_v1 through `mojolearn.linalg`, cross-vendor hashes); H100 and MI325X summary JSON under
  `bench/results/lowbit_e2e/2026-09-29/`; progress and handoff under `docs/studies/lowbit/E2E*_2026-09-29.md`.
  `model_probe.py` was never run. The steward/`nvidia_central` commands in the handoff are historical; today a run goes
  through `lq add <box> CMD`.
- `lane/lowbit-quality` @ b967b0647: **(1) a quality bench, ported.** The rest of `bench/lowbit_quality/` (the Llama
  simulation, inference evaluation, byte-LM training simulation, backward export, quantizer check, table builders, job
  scripts). Main's `arith.py` (same arithmetic, newer pin, neutral comments) is kept. Ported the inference and training
  tables and top-level summary JSON under `bench/results/lowbit_quality/` (the 112 MB of per-run records and npz stay on
  the branch), and the progress file `docs/studies/lowbit/QUALITY_2026-09-29.md`. Its verdicts are the source of
  `_numeric_profile.REJECTED` (int8) and the fixed15 quality numbers.
- `lane/mamba3-statepass` @ b42f1c857: **(a) on main, not (d).** Squash-merged on 2026-09-09 as 16b650a67 ("Remove
  Mamba3 transfer overhead and tile state computations without changing folds"). The files were moved by 48c4092ff,
  not deleted: `mamba/impl/ops/mamba3_siso.mojo`, `mamba/impl/modules/mamba3_transfer.mojo`,
  `mamba/host/gen/mamba3_refusal.mojo`; `mamba/checks/mamba3_refusal_check.mojo`, `mamba3_statepass_price.mojo` and
  `tools/mamba3_statepass_leg.sh` are on main, and the `MOJOLEARN_MAMBA3_LEGACY_*` defines are in `mamba3_siso.mojo`.
  The handoff and evidence README were removed by the cleanups 2e7d346a5 and cbe5a8a63. The one part never merged is
  the rejected hardware-FTZ fold (`MOJOLEARN_MAMBA3_HARDWARE_FOLD`, `mamba3_fold.mojo`, `mamba3_fold_check.mojo`): on
  the H100 it returns zero for 0x3f7fffff * 0x00800000 + 0, and the lane withdrew it. It never reached main, so it
  needs no patch, and the code can be recovered at that sha. Retire.
- `lane/fp-contract-off-study` @ c53393787: **(1) a study, ported.** The README is now
  `docs/studies/FP_CONTRACT_OFF_2026-09-26.md`. contract=off moves 16 of 627 cells on Metal, the RTX 4090 and the CPU.
  It also splits the CPU from both GPUs on the Poisson loss (6 cells). The recommendation is to keep the default and
  write those expressions with explicit `fma`. The hook (`MOJOLEARN_MOJO_BUILD_FLAGS`) is on main. Retire the branch.

## Table

| Branch | Date | Ahead (unique) | Files (unique commits): top dirs | Verdict | Evidence | Recommendation |
|---|---|---|---|---|---|---|
| `archive/20261006/campaign/apple-opponent-quality-20261006` | 2026-10-06 | 47 (39) | 815: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/campaign/full-classification-qn-recipes-20261006` | 2026-10-06 | 91 (83) | 1116: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/fix/apple-opponent-selector-20261006` | 2026-10-06 | 57 (49) | 884: experiments/classical_identical_ideas, experiments/six_lane_integration, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/fix/lars-direct-20261006` | 2026-10-06 | 46 (38) | 814: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/ideas/trees-identical-20261006` | 2026-10-06 | 1 (1) | 142: experiments/trees_identical_20261006, gbdt/methods, extratrees/impl | a | 92% of unique added code lines present on main | retire |
| `archive/20261006/integration/six-lane-ab-20261006` | 2026-10-06 | 41 (34) | 808: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/integration/six-lane-comparison-scope-fix-20261006` | 2026-10-06 | 52 (44) | 822: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/integration/six-lane-full-tsvd-20261006` | 2026-10-06 | 60 (52) | 890: experiments/classical_identical_ideas, experiments/six_lane_integration, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/integration/six-lane-state-capture-20261006` | 2026-10-06 | 47 (39) | 819: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/lane/apple-classification-admission` | 2026-10-06 | 70 (62) | 947: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/lane/apple-full-classification-prep` | 2026-10-06 | 65 (57) | 941: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/lane/apple-six-admission` | 2026-10-06 | 44 (36) | 812: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.81 | retire |
| `archive/20261006/lane/full-recipe-glue` | 2026-10-06 | 49 (41) | 815: experiments/classical_identical_ideas, experiments/trees_identical_20261006, experiments/performance_ideas | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/20261006/lane/host-route-source-repair-20261006` | 2026-10-06 | 2 (2) | 35: cluster/host, glm/checks, glm/host | b | host-route-fence-2026-10-09.md class C/D: the GPU-import edges were cut on main in 64cc80087/34bd84cf8; x_cluster serial launch removed | retire |
| `archive/20261006/lane/idle-retention-60m-20261007` | 2026-10-07 | 1 (0) | 5: docs/MEASUREMENT_USAGE_OWNERS.md, tools/do_amd_usage_owner.py, tools/runpod_usage_lease.py | a | git cherry: all 1 commits patch-equal on main | retire |
| `archive/20261006/lane/mlp-full-registration` | 2026-10-06 | 95 (87) | 1122: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/lane/neural-identical-ideas-20261006` | 2026-10-06 | 1 (1) | 57: experiments/neural_identical_20261006, mamba/impl, embedding/checks | b | idea catalog + untested A/B source; fence doc :32; ideas carried in NEURAL_AB_EXPERIMENT_INVENTORY_2026-10-06.md | retire |
| `archive/20261006/lane/public-linear-state-capture-20261006` | 2026-10-06 | 88 (80) | 1082: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/lane/targeted-benchmark-metadata-20261007` | 2026-10-07 | 1 (0) | 3: experiments/six_lane_integration | a | git cherry: all 1 commits patch-equal on main | retire |
| `archive/20261006/lane/targeted-cpu-build-20261007` | 2026-10-07 | 1 (0) | 2: tools/six_lane_ab.py, tools/test_six_lane_cpu_target.py | a | git cherry: all 1 commits patch-equal on main | retire |
| `archive/20261006/lane/var-native-fittedvalues` | 2026-10-06 | 86 (78) | 1085: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/repair/amd-full-classification-registration` | 2026-10-06 | 92 (84) | 1116: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/repair/apple-fast-huber-fold-20261006` | 2026-10-06 | 79 (71) | 993: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/repair/cnn-native-orchestration-20261006` | 2026-10-06 | 87 (79) | 1087: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.83 | retire |
| `archive/20261006/repair/prep-native-sparse-20261006` | 2026-10-06 | 85 (77) | 1027: experiments/six_lane_integration, experiments/classical_identical_ideas, experiments/trees_identical_20261006 | a | docs/plans/host-route-fence-2026-10-09.md:21-29 (same code reached main via bd4c36dbe); unique-line presence 0.82 | retire |
| `archive/worktree-cleanup-20260919-release-087-build` | 2026-09-16 | 8 (6) | 10: packaging/linux, packaging/macos, docs/lanes | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `archive/worktree-cleanup-20260919-release-087-macos` | 2026-09-16 | 8 (6) | 10: packaging/linux, packaging/macos, docs/lanes | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `board/m3-0834-readback` | 2026-10-02 | 1 (1) | 1: tools/bench_board_algos.py | b | one-line readback for the 0.8.34 M3 board; board superseded by later releases | retire |
| `finish/cross-vendor-native-only-20261020` | 2026-10-20 | 2 (0) | 14: python/mojolearn, bindings/byte_checkpoint_io.mojo, bindings/byte_checkpoint_native.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/distributed-20261020` | 2026-10-20 | 45 (6) | 19: python/mojolearn, autograd/distributed_pyapi.mojo, autograd/partition_checkpoint.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/generation-20261020` | 2026-10-20 | 83 (1) | 2: tools/hooks | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/generation-graph-repair-20261020` | 2026-10-20 | 56 (7) | 46: packaging/native_only, python/mojolearn, bindings/_mojolearn_tokenizer_host.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/graph-20261020` | 2026-10-20 | 38 (13) | 33: python/mojolearn, docs/plans, autograd/control_checkpoint.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/graph-hoist-20261020` | 2026-10-20 | 10 (0) | 27: python/mojolearn, autograd/checks, autograd/distributed_program.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/graph-process-20261020` | 2026-10-20 | 1 (0) | 2: bindings/native_process_pool.mojo, python/mojolearn | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/installed-native-audit-20261020` | 2026-10-20 | 7 (0) | 13: packaging/native_only, docs/plans, packaging/portable_math | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/native-autograd-conv-smoke-20261020` | 2026-10-20 | 1 (0) | 6: packaging/native_only | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/native-classical-public-20261020` | 2026-10-20 | 4 (0) | 12: python/mojolearn, bindings/classical_archive.mojo, bindings/classical_archive_read.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/native-optimizer-smoke-20261020` | 2026-10-20 | 2 (0) | 7: packaging/native_only, bindings/native_json.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/native-svm-smoke-20261020` | 2026-10-20 | 1 (1) | 6: packaging/native_only | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/parallel-causal-20261020` | 2026-10-20 | 13 (0) | 22: python/mojolearn, bindings/causal_lm_construct.mojo, bindings/causal_lm_public.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/parallel-neural-admission-20261020` | 2026-10-20 | 16 (0) | 38: python/mojolearn, docs/plans, autograd/layers.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/pca-native-owner-20261020` | 2026-10-20 | 5 (0) | 9: python/mojolearn, bindings/_mojolearn_tokenizer_host.mojo, bindings/pca_archive.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `finish/samba-public-20261020` | 2026-10-20 | 16 (0) | 26: python/mojolearn, ivf/checks, bindings/samba_admission.mojo | a | docs/plans/FINISH_LEFTOVERS_20261021.md table: (a) | retire |
| `fix/amd-merge-ordering` | 2026-09-16 | 3 (3) | 6: docs/lanes, core/device_mutex.mojo, core/device_mutex_check.mojo | b | mutex handoff probe (Sep 16); sibling rf-mutex-claim-acquire head says the fix landed on main as 36ec47446 | retire |
| `fix/classical-mds-20260930` | 2026-09-30 | 3 (3) | 6: python/mojolearn, tools/bench_board.py, tools/bench_board_resume.py | b | board-runner bounds + ClassicalMDS top-k (Sep 30); tools/bench_board_resume.py and classical_mds_check.py were never adopted; board runner replaced by lq + six_lane_grid | retire |
| `fix/identity-collection-completeness` | 2026-09-20 | 1 (0) | 2: python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `fix/macos-smoke-identical-only` | 2026-09-15 | 22 (5) | 22: holtwinters/impl, packaging/linux, packaging/macos | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `fix/metal-parallel-causal-lm` | 2026-09-20 | 1 (0) | 2: python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `fix/parallel-model-save` | 2026-09-20 | 2 (0) | 3: python/mojolearn, tools/test_parallel_saved_models_runtime.py | a | git cherry: all 2 commits patch-equal on main | retire |
| `fix/release-archive-tokenizer-generator` | 2026-09-15 | 21 (5) | 22: holtwinters/impl, packaging/linux, packaging/macos | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `fix/release-build-preflight-import` | 2026-09-15 | 2 (1) | 4: CHANGELOG.md, CITATION.cff, python/mojolearn | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `fix/release-portable-math-transport` | 2026-09-20 | 2 (0) | 5: docs/lanes, hdbscan/host, python/mojolearn | a | git cherry: all 2 commits patch-equal on main | retire |
| `fix/release-post-record-allowlist` | 2026-09-15 | 58 (40) | 24: holtwinters/impl, packaging/linux, packaging/macos | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `fix/release-verifier-bindings` | 2026-09-15 | 3 (1) | 4: CHANGELOG.md, CITATION.cff, python/mojolearn | b | 0.8.6/0.8.7 release tooling (Sep 15-16); candidate-audit-2026-10-04.md "Release / docs / fix tooling (old releases): abandoned"; main ships 0.8.55 with a rebuilt release chain | retire |
| `lane/136-lane-record` | 2026-09-14 | 1 (0) | 14: bench/results, .github/workflows, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/ab-measure` | 2026-10-01 | 3 (3) | 2: tools/ab_job.sh, tools/nv_ab_pr59.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/algos-ann` | 2026-09-27 | 1 (1) | 18: x_ann/checks, python/mojolearn, tools/identity_lanes | b | WIP "UNVERIFIED" option parity; main has the index save/load (python/mojolearn/_expansion_ann.py:157,197) and the t-SNE sklearn params | retire |
| `lane/algos-ann-speed` | 2026-09-27 | 6 (6) | 18: x_ann/checks, x_ann/cagra_core.mojo, x_ann/cagra_device.mojo | b | candidate-audit-2026-10-04.md:36 "abandoned / reference" (CAGRA chunked queries, t-SNE tiling; patches conflict) | retire |
| `lane/algos-trees` | 2026-09-28 | 1 (1) | 1: docs/lanes | b | progress-note only (docs/lanes/progress/*.md); superseded by later handoffs | retire |
| `lane/ann-apple-base` | 2026-09-28 | 1 (1) | 1: x_ann/stage_timer.mojo | a | candidate-audit-2026-10-04.md:21 "content-absorbed" | retire |
| `lane/apple-fast-af-all` | 2026-10-03 | 74 (29) | 72: docs/apple-fast, glm/impl, gbdt/methods | b | aggregation lane (16 lanes); members judged individually in EXPERIMENTS.md; OPTICS2 defines DROPPED (TOMBSTONES OPTICS2_ALL); the open BPE part lives on lane/apple-fast-bpe | retire |
| `lane/apple-fast-arima-k1-diagnostics` | 2026-10-04 | 5 (5) | 7: docs/apple-fast, tools/arima_assoc_scan_oracle.py, tools/arima_gaussian_scan_oracle.py | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-arima-k1-oracle` | 2026-10-04 | 3 (3) | 5: docs/apple-fast, tools/arima_assoc_scan_oracle.py, tools/arima_gaussian_scan_oracle.py | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-arima-k1-review` | 2026-10-04 | 4 (4) | 6: docs/apple-fast, tools/arima_assoc_scan_oracle.py, tools/arima_gaussian_scan_oracle.py | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-arima-k3-df-gpu` | 2026-10-04 | 1 (1) | 1: bindings/_mojolearn_arima.mojo | a | 100% of unique added code lines present on main | retire |
| `lane/apple-fast-arima-k3-df-reference` | 2026-10-04 | 6 (6) | 10: docs/apple-fast, arima/impl, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-k3-tail-probe` | 2026-10-04 | 5 (5) | 8: arima/impl, docs/apple-fast, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-scalar-k3` | 2026-10-04 | 1 (1) | 7: arima/impl, docs/apple-fast, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-scalar-k3-quality` | 2026-10-04 | 2 (2) | 8: arima/impl, docs/apple-fast, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-scalar-k3-r1` | 2026-10-04 | 3 (3) | 8: arima/impl, docs/apple-fast, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-scalar-k3-r2` | 2026-10-04 | 4 (4) | 8: arima/impl, docs/apple-fast, bindings/_mojolearn_arima.mojo | b | EXPERIMENTS.md:991 ARIMA_FAST_SCALAR_LL: M3 kernel quality HOLD across 13 groups; r2 is the cited sha; ARIMA_K3_DF_REFERENCE_PLAN.txt on main | keep r2 only (cited sha); retire the rest |
| `lane/apple-fast-arima-scan-design` | 2026-10-04 | 1 (1) | 2: docs/apple-fast | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-arima-scan-oracle` | 2026-10-04 | 1 (1) | 2: docs/apple-fast, tools/arima_assoc_scan_oracle.py | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-arima-scan-oracle-review` | 2026-10-04 | 2 (2) | 4: docs/apple-fast, tools/arima_assoc_scan_oracle.py | b | EXPERIMENTS.md:925 rank-one/K1 temporal scan: numerical HOLD; oracle tools only, no runtime code | retire |
| `lane/apple-fast-batch` | 2026-10-03 | 177 (88) | 130: docs/apple-fast, python/mojolearn, glm/impl | b | aggregation lane; TOMBSTONES/EXPERIMENTS rows per define (DROPPED-noise 43, KEPT 7 ported); open BPE part lives on lane/apple-fast-bpe | retire |
| `lane/apple-fast-bpe` | 2026-10-03 | 9 (9) | 8: docs/apple-fast, tokenizer/fast, bindings/_mojolearn_tokenizer_fast.mojo | c | EXPERIMENTS.md:34 Untried candidates: BPE_TRAIN_DEVICE/ENCODE_DEVICE/LIVEBUF/GROUP_FILTER/MERGE_BATCH/ALL "NOT on main", never measured | keep |
| `lane/apple-fast-bsplit-readiness` | 2026-10-04 | 1 (1) | 3: docs/apple-fast, tools/svgp_fast_pair.py, tools/svgp_fast_quality.py | a | EXPERIMENTS.md Untried candidates: SVGP_FAST_BSPLIT "on main opt-in"; readiness note only | retire |
| `lane/apple-fast-callpath-probe` | 2026-10-04 | 2 (2) | 12: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bindings/_mojolearn_callpath_probe.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-probe-r1` | 2026-10-04 | 3 (3) | 13: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bench/apple_callpath_quality_main.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-probe-r2` | 2026-10-04 | 4 (4) | 13: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bench/apple_callpath_quality_main.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-probe-r3` | 2026-10-04 | 5 (5) | 13: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bench/apple_callpath_quality_main.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-probe-r4` | 2026-10-04 | 6 (6) | 14: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bench/apple_callpath_quality_main.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-probe-r5` | 2026-10-04 | 7 (7) | 14: experiments/apple_callpath, bench/apple_callpath_quality.mojo, bench/apple_callpath_quality_main.mojo | b | EXPERIMENTS.md:1065 r5 3491a4d4 PASS bits/lifecycle, "not production"; cited sha | keep (cited sha) |
| `lane/apple-fast-callpath-quality` | 2026-10-04 | 1 (1) | 8: experiments/apple_callpath, bench/apple_callpath_quality.mojo, experiments/__init__.mojo | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-callpath-reach-audit` | 2026-10-04 | 2 (2) | 1: docs/apple-fast | b | EXPERIMENTS.md:1036,1065 callpath r5 PASS bits/lifecycle, "not production", no timing admitted; r0-r4 superseded by r5 | retire |
| `lane/apple-fast-catalog-gemm-all` | 2026-10-04 | 6 (6) | 10: experiments/apple_fast, ab/catalog-gemm-all.md, ab/catalog-gemm-probe.md | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-gemm-probe` | 2026-10-04 | 3 (3) | 9: experiments/apple_fast, ab/catalog-gemm-probe.md, bindings/_mojolearn_gemm_probe.mojo | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-gemm-probe-r1` | 2026-10-04 | 4 (4) | 9: experiments/apple_fast, ab/catalog-gemm-probe.md, bindings/_mojolearn_gemm_probe.mojo | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-gemm-probe-r2` | 2026-10-04 | 5 (5) | 9: experiments/apple_fast, ab/catalog-gemm-probe.md, bindings/_mojolearn_gemm_probe.mojo | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-matrix-timing` | 2026-10-04 | 7 (7) | 12: experiments/apple_fast, ab/catalog-gemm-all.md, ab/catalog-gemm-probe.md | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-matrix-timing-r1` | 2026-10-04 | 8 (8) | 12: experiments/apple_fast, ab/catalog-gemm-all.md, ab/catalog-gemm-probe.md | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-catalog-resident-probe` | 2026-10-04 | 1 (1) | 10: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn_resident_gemm_probe.mojo | b | EXPERIMENTS.md:989 catalog G1/G5 standalone HOLD for unrestricted use; RESIDENT_GEMM_RESULTS_2026-10-04.md; probe bindings only | retire |
| `lane/apple-fast-chol-20261004` | 2026-10-04 | 4 (4) | 2: cholesky/checks, tools/apple_fast_chol_quality.py | b | docs/apple-fast/EXPERIMENTS.md:805 [DROPPED-slower1] | retire |
| `lane/apple-fast-classical-ideas-20261006` | 2026-10-06 | 3 (3) | 143: experiments/performance_ideas, experiments/apple_fast_classical_20261006, gbdt/methods | a | 98% of unique added code lines present on main | retire |
| `lane/apple-fast-dbscantaxi` | 2026-10-03 | 1 (1) | 3: dbscan/impl, docs/apple-fast | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2394 [ported4 DROPPED-slower3 RECORD2 READY-AB2] | retire |
| `lane/apple-fast-decomp-linalg` | 2026-10-03 | 19 (14) | 12: docs/apple-fast, python/mojolearn, bindings/_mojolearn_x_decomp.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2012 [DROPPED-slower13 ported13 READY-AB8 SUPERSEDED5 DEFAULT3 DROPPED2] | retire |
| `lane/apple-fast-decomp-mma-k16` | 2026-10-04 | 2 (2) | 5: ab/decomp-mma-k16.md, docs/apple-fast, tools/decomp_mma_k16.py | b | docs/apple-fast/EXPERIMENTS.md:938 [PASS1 HOLD-speed1] | retire |
| `lane/apple-fast-decomp-sdk-control` | 2026-10-04 | 1 (1) | 5: docs/apple-fast, experiments/apple_fast, x_decomp/device.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:444 [PARKED1 ported1] | retire |
| `lane/apple-fast-decomp-sparse` | 2026-10-03 | 31 (19) | 15: docs/apple-fast, python/mojolearn, bindings/_mojolearn_x_decomp.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1943 [ported3 DROPPED-semantics2 SUPERSEDED2 READY-AB1] | retire |
| `lane/apple-fast-dwcurrent` | 2026-10-04 | 3 (3) | 6: gbdt/methods, docs/apple-fast, tools/aft_verified_dw.py | b | EXPERIMENTS.md: GBDT_DW_BRIDGE_SCAN DROPPED-slower (+5.34%, gap26-dwcurrent-taxi) | retire |
| `lane/apple-fast-eigh-cache` | 2026-10-04 | 7 (7) | 6: docs/apple-fast, tools/apple_fast_eigh_quality.py, tools/eigh_cache_verified.py | b | docs/TOMBSTONES.md:1949 [PASS2 rejected1] | retire |
| `lane/apple-fast-eigh-panel-df` | 2026-10-04 | 3 (3) | 5: docs/apple-fast, tools/eigh_panel_df_pair.py, tools/eigh_w4_quality.py | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:1217 [ported1 HOLD1 DROPPED-slower1 READY-AB1] | retire |
| `lane/apple-fast-eigh-panels` | 2026-10-04 | 7 (6) | 6: ab/eigh-panels-review.md, docs/apple-fast, tools/eigh_w4_pair.py | a | EIGH_FAST_TRIDIAG(_PANELS) on main (default with _OFF); 74% of unique lines present | retire |
| `lane/apple-fast-fa` | 2026-10-03 | 4 (4) | 6: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2057 [ported10 DROPPED-slower10 HOLD-quality8 READY-AB6] | retire |
| `lane/apple-fast-fastonly2` | 2026-10-03 | 1 (1) | 2: ivf/impl, x_ann/kpp_seed_device.mojo | a | 98% of unique added code lines present on main | retire |
| `lane/apple-fast-gap-kapprox2` | 2026-10-03 | 15 (12) | 15: python/mojolearn, bindings/_mojolearn_x_neighbors.mojo, bindings/_mojolearn_x_neighbors_host.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1664 [merged5 DROPPED4 KEPT2] | retire |
| `lane/apple-fast-gap-linalg2` | 2026-10-03 | 4 (1) | 1: docs/apple-fast | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-gap-linalg2-kpca` | 2026-10-03 | 2 (2) | 1: tools/kpca_quality_ab.sh | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-gap-linalg2-pca` | 2026-10-03 | 3 (3) | 6: decomposition/estimator.mojo, decomposition/impl, docs/apple-fast | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-gap-linalg2-svd` | 2026-10-03 | 1 (1) | 2: x_decomp/svd_aw.mojo, x_decomp/tsqr_device.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:1218 [PARKED2 ported1] | retire |
| `lane/apple-fast-gap-misc` | 2026-10-03 | 6 (4) | 5: gbdt/methods | b | docs/TOMBSTONES.md:595 [DROPPED-noise6 MERGED1] | retire |
| `lane/apple-fast-gap-tsa` | 2026-10-03 | 1 (1) | 1: docs/apple-fast | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2467 [DROPPED-noise10 KEPT5 DROPPED-slower2 merged1] | retire |
| `lane/apple-fast-general-speed` | 2026-10-04 | 5 (4) | 3: docs/apple-fast, svm/impl, x_linear/bayes.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1088 [merged5 DROPPED-quality3 KEPT1] | retire |
| `lane/apple-fast-graph` | 2026-10-02 | 3 (3) | 4: docs/apple-fast, python/mojolearn, x_neighbors/iter_device.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:494 [ported3 KEPT1 DEFAULT1 READY-AB1] | retire |
| `lane/apple-fast-isotonic-knn` | 2026-10-03 | 11 (7) | 6: docs/apple-fast, bindings/_mojolearn_x_linear.mojo, python/mojolearn | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1456 [DROPPED-slower4 merged4 DROPPED-noise2 loser2 HELD1 ported1] | retire |
| `lane/apple-fast-kapprox` | 2026-10-03 | 3 (3) | 4: docs/apple-fast, tools/hooks, tools/kpca_quality_check.py | b | docs/TOMBSTONES.md:1556 [DROPPED-quality10 KEPT1] | retire |
| `lane/apple-fast-knn-lean-gpu` | 2026-10-04 | 1 (1) | 6: bindings/_mojolearn_x_neighbors.mojo, docs/apple-fast, python/mojolearn | a | EXPERIMENTS.md: XN_FAST_NAN_FIT_LEAN_GPU DEFAULT, on main | retire |
| `lane/apple-fast-knn-lean-gpu-harness-r1` | 2026-10-04 | 2 (2) | 6: bindings/_mojolearn_x_neighbors.mojo, docs/apple-fast, python/mojolearn | a | 90% of unique added code lines present on main | retire |
| `lane/apple-fast-layoutfix` | 2026-10-03 | 2 (2) | 7: python/mojolearn, tools/layoutfix_smoke.py | b | NoCpuEntry ImportError fix; main _backend.py has its own NoCpuEntry handling (2 hits) after the backend rewrite | retire |
| `lane/apple-fast-linalg-20261004` | 2026-10-04 | 4 (4) | 3: tools/apple_fast_eigh_quality.py, x_decomp/device.mojo, x_decomp/eigh_tangent.mojo | b | docs/apple-fast/EXPERIMENTS.md:739 [HOLD-speed2] | retire |
| `lane/apple-fast-linsvr` | 2026-10-03 | 7 (7) | 8: glm/impl, docs/apple-fast | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1181 [DROPPED-noise8 merged8 DROPPED-slower4 KEEP2 KEPT1] | retire |
| `lane/apple-fast-lle` | 2026-10-02 | 12 (7) | 10: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, bindings/_mojolearn_x_linear.mojo | b | docs/TOMBSTONES.md:1462 [DROPPED-slower2] | retire |
| `lane/apple-fast-lost-candidates` | 2026-10-04 | 1 (1) | 1: docs/apple-fast | b | audit doc; superseded by MANAGER_STATE_2026-10-04.md:7 "all lost candidates ported" and EXPERIMENTS.md Untried candidates (2026-10-05) | retire |
| `lane/apple-fast-lu-mma-dbuf` | 2026-10-04 | 2 (2) | 5: ab/lu-mma-dbuf.md, docs/apple-fast, tools/lu_mma_dbuf.py | b | docs/apple-fast/EXPERIMENTS.md:924 [PASS1 HOLD-speed1] | retire |
| `lane/apple-fast-lu-pivot-shuffle` | 2026-10-04 | 2 (2) | 6: ab/lu-pivot-shuffle.md, bindings/_mojolearn_x_decomp.mojo, docs/apple-fast | a | LU_FAST_PIVOT_SHUFFLE on main (rollback _OFF); EXPERIMENTS.md w2-lu-pivot-shuffle rows | retire |
| `lane/apple-fast-m2b1` | 2026-10-03 | 11 (10) | 16: glm/impl, docs/apple-fast, bindings/_mojolearn_x_linear.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1457 [KEEP5 DROPPED-noise4 merged4 loser2 DROPPED-slower2 HELD1] | retire |
| `lane/apple-fast-manager-remote` | 2026-10-04 | 1 (1) | 1: tools/apple_fast_job_policy.py | a | 100% of unique added code lines present on main | retire |
| `lane/apple-fast-mcd-g1-quality` | 2026-10-04 | 1 (1) | 1: tools/mcd_g1_quality.py | a | 93% of unique added code lines present on main | retire |
| `lane/apple-fast-mcd-ordered-cov` | 2026-10-04 | 1 (1) | 7: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a | MCD_ORDERED_COV on main (opt-in, NEUTRAL: EXPERIMENTS.md rab3-mcdord); superseded by -r3 | retire |
| `lane/apple-fast-mcd-ordered-quality` | 2026-10-04 | 2 (2) | 10: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a | MCD_ORDERED_COV on main (opt-in, NEUTRAL: EXPERIMENTS.md rab3-mcdord); superseded by -r3 | retire |
| `lane/apple-fast-mcd-ordered-quality-r1` | 2026-10-04 | 3 (3) | 10: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a | MCD_ORDERED_COV on main (opt-in, NEUTRAL: EXPERIMENTS.md rab3-mcdord); superseded by -r3 | retire |
| `lane/apple-fast-mcd-ordered-quality-r2` | 2026-10-04 | 4 (4) | 10: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a | MCD_ORDERED_COV on main (opt-in, NEUTRAL: EXPERIMENTS.md rab3-mcdord); superseded by -r3 | retire |
| `lane/apple-fast-mcd-ordered-quality-r3` | 2026-10-04 | 5 (5) | 10: docs/apple-fast, bindings/_mojolearn_x_decomp.mojo, python/mojolearn | a | EXPERIMENTS.md:1216 ported (lane/apple-fast-rec-misc); RECORD NEUTRAL, stays off | retire |
| `lane/apple-fast-mcd-quality` | 2026-10-03 | 2 (2) | 2: tools/mcd_quality_ab.py, tools/mcd_quality_ab.sh | b | quality tool for MCD_DEVICE_CSTEPS, DROPPED-quality (TOMBSTONES, patch experiments/removed/MOJOLEARN_MCD_DEVICE_CSTEPS.patch) | retire |
| `lane/apple-fast-mcd-saved-oracle` | 2026-10-04 | 3 (3) | 6: docs/apple-fast, tools/mcd_compat_ab.sh, tools/mcd_profile.sh | a | MCD_SKIP_PINVH KEEP and DEFLATE HOLD both on main (EXPERIMENTS.md w4-mcd rows); oracle tools only on branch | retire |
| `lane/apple-fast-mcd-saved-oracle-r1` | 2026-10-04 | 4 (4) | 6: docs/apple-fast, tools/mcd_compat_ab.sh, tools/mcd_profile.sh | a | MCD_SKIP_PINVH KEEP and DEFLATE HOLD both on main (EXPERIMENTS.md w4-mcd rows); oracle tools only on branch | retire |
| `lane/apple-fast-old-linalg-audit` | 2026-10-04 | 1 (1) | 2: docs/apple-fast, python/mojolearn | b | doc note on invalid old QR/SVD comparisons; superseded by EXPERIMENTS reconciliation 2026-10-05 | retire |
| `lane/apple-fast-opp` | 2026-10-03 | 7 (5) | 3: bench/speed, tools/bench_board_algos.py, tools/opp_only_board.py | b | opponents-only board tooling (tools/opp_only_board.py, never on main); opponents are scored once and stored (CLAUDE.md) | retire |
| `lane/apple-fast-optics2` | 2026-10-03 | 5 (5) | 8: docs/apple-fast, x_cluster/device_ops.mojo, x_cluster/host | b | TOMBSTONES OPTICS2_ALL / OPTICS_STEP_BATCH DROPPED (54-84x slower), "never merged" | retire |
| `lane/apple-fast-opv` | 2026-10-03 | 8 (7) | 11: docs/apple-fast, tools/opv_build_check.sh, tools/opv_quality.py | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2303 [merged4] | retire |
| `lane/apple-fast-pca-eig` | 2026-10-02 | 3 (3) | 4: docs/apple-fast, decomposition/estimator.mojo, decomposition/impl | b | docs/TOMBSTONES.md:1979 [DROPPED-noise10] | retire |
| `lane/apple-fast-prep2` | 2026-10-03 | 2 (1) | 2: x_prep/device.mojo, x_prep/fastprep2.mojo | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-prep3` | 2026-10-02 | 9 (4) | 8: docs/apple-fast, bindings/_mojolearn_x_prep.mojo, python/mojolearn | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1723 [DROPPED-noise4 merged4 KEEP1] | retire |
| `lane/apple-fast-pt-queue` | 2026-10-04 | 1 (1) | 2: docs/apple-fast, tools/pt_score_verified.py | a | 97% of unique added code lines present on main | retire |
| `lane/apple-fast-rec-ab11` | 2026-10-04 | 5 (4) | 3: docs/apple-fast, svm/impl, x_linear/bayes.mojo | b | ARD_EQ_ONEPASS DROPPED-quality (TOMBSTONES); LDA/SVM legacy defines on main | retire |
| `lane/apple-fast-remaining-audit-20261004` | 2026-10-04 | 1 (0) | 1: docs/apple-fast | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/apple-fast-resample` | 2026-10-02 | 6 (5) | 7: docs/apple-fast, python/mojolearn, bindings/_mojolearn_resample.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1777 [DROPPED-slower9 KEPT4 ported4 DROPPED-semantics2 DEFAULT2 READY-AB2] | retire |
| `lane/apple-fast-results` | 2026-10-04 | 166 (166) | 4: docs/apple-fast | b | M3 queue/results sync files (166 commits, docs/apple-fast/m3/*); results reconciled into EXPERIMENTS.md 2026-10-05 | retire |
| `lane/apple-fast-robust` | 2026-10-02 | 8 (6) | 6: docs/apple-fast, x_decomp/kit_device.mojo, x_decomp/mcd_fast.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:2099 [DROPPED-quality3 DEFAULT2 ported2 READY-AB1 merged1] | retire |
| `lane/apple-fast-scoped-pca-caller` | 2026-10-04 | 2 (2) | 2: tools/apple_fast_job_policy.py, tools/scoped_pca_fit.py | b | GEMM_INLINE_OUTCOMES.md: scoped PCA-fit HOLD (quality report invalid); SCOPED_GEMM defines on main | retire |
| `lane/apple-fast-sgdoc-parallel` | 2026-10-03 | 7 (7) | 7: docs/apple-fast, tools/bench_board_algos.py, tools/sgdoc_check.py | b | docs/apple-fast/EXPERIMENTS.md:734 [HOLD2] | retire |
| `lane/apple-fast-shap` | 2026-10-02 | 5 (4) | 7: docs/apple-fast, python/mojolearn, xtrees/api.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:180 [SUPERSEDED3 ported3 KEEP2 KEPT2 DEFAULT2 READY-AB1] | retire |
| `lane/apple-fast-shared-downstream-g1` | 2026-10-04 | 4 (4) | 22: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-downstream-g1-r1` | 2026-10-04 | 5 (5) | 23: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-downstream-g1-r2` | 2026-10-04 | 6 (6) | 25: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: "Shared G1/G5 code is preserved on historical candidate branches; it was not restored to main" | keep (record of the shared G1/G5 code) |
| `lane/apple-fast-shared-downstream-g5` | 2026-10-04 | 5 (5) | 22: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-downstream-g5-r1` | 2026-10-04 | 6 (6) | 23: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-downstream-g5-r2` | 2026-10-04 | 7 (7) | 25: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: "Shared G1/G5 code is preserved on historical candidate branches; it was not restored to main" | keep (record of the shared G1/G5 code) |
| `lane/apple-fast-shared-downstream-quality` | 2026-10-04 | 3 (3) | 19: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-gemm-g1g5` | 2026-10-04 | 1 (1) | 12: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn_shared_gemm_probe.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-gemm-g1g5-r1` | 2026-10-04 | 2 (2) | 12: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn_shared_gemm_probe.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-scoped-timing` | 2026-10-04 | 7 (7) | 26: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: shared G1/G5 narrow caller leads, inverse slower, "no broad default/board admission"; superseded by the -r2/-g5 tips | retire |
| `lane/apple-fast-shared-scoped-timing-g5` | 2026-10-04 | 8 (8) | 26: experiments/apple_fast, docs/apple-fast, bindings/_mojolearn.mojo | b | GEMM_INLINE_OUTCOMES.md: "Shared G1/G5 code is preserved on historical candidate branches; it was not restored to main" | keep (record of the shared G1/G5 code) |
| `lane/apple-fast-sym-ctr` | 2026-10-03 | 7 (2) | 3: gbdt/ctrs, gbdt/methods, gbdt/train.mojo | a/b | winners ported, rest documented: docs/TOMBSTONES.md:804 [ported10 KEPT6 DEFAULT6 READY-AB5 DROPPED-noise2 DROPPED-speed2] | retire |
| `lane/apple-fast-sym-est` | 2026-10-03 | 4 (1) | 2: gbdt/methods | a/b | winners ported, rest documented: docs/TOMBSTONES.md:967 [ported7 KEPT4 DEFAULT4 DROPPED-BUG3 DROPPED-inconclusive3 READY-AB3] | retire |
| `lane/apple-fast-sym-feat` | 2026-10-03 | 14 (1) | 6: bindings/_mojolearn_gbdt.mojo, gbdt/estimator.mojo, gbdt/gpu_data | a | 96% of unique added code lines present on main | retire |
| `lane/apple-fast-sym-hist` | 2026-10-03 | 3 (0) | 9: gbdt/methods, docs/apple-fast | a | git cherry: all 3 commits patch-equal on main | retire |
| `lane/apple-fast-sym-iter` | 2026-10-03 | 5 (1) | 3: gbdt/methods | a/b | winners ported, rest documented: docs/TOMBSTONES.md:1016 [DROPPED-noise18 ported5 READY-AB2] | retire |
| `lane/apple-fast-sym-multi` | 2026-10-03 | 6 (1) | 2: gbdt/methods, gbdt/targets | a | 97% of unique added code lines present on main | retire |
| `lane/apple-fast-target-current` | 2026-10-04 | 5 (4) | 6: docs/apple-fast, bindings/_mojolearn_x_prep.mojo, python/mojolearn | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-target-scratch` | 2026-10-04 | 3 (3) | 4: bindings/_mojolearn_x_prep.mojo, docs/apple-fast, python/mojolearn | a | TARGET_SCRATCH on main; 83% of unique lines present; sibling target-current KEEP merged | retire |
| `lane/apple-fast-trees-io` | 2026-10-02 | 6 (3) | 9: docs/apple-fast, isolation_forest/impl, bindings/_mojolearn_svm.mojo | b | docs/TOMBSTONES.md:714 [DROPPED-noise4] | retire |
| `lane/apple-fast-trees-scan` | 2026-10-02 | 9 (8) | 6: gbdt/gpu_util, docs/apple-fast, ensemble/randomforest.mojo | b | docs/TOMBSTONES.md:757 [DROPPED-noise3 DROPPED-semantics2] | retire |
| `lane/apple-fast-trees-symmetric` | 2026-10-02 | 6 (4) | 8: gbdt/methods, docs/apple-fast | b | docs/TOMBSTONES.md:811 [DROPPED-noise11 DROPPED-slower2] | retire |
| `lane/apple-fast-w2-shap` | 2026-10-04 | 3 (3) | 7: python/mojolearn, docs/apple-fast, tools/shap_pipe_pair.py | b | docs/TOMBSTONES.md:1789 [PASS2 rejected1] | retire |
| `lane/apple-fast-w2-ts` | 2026-10-04 | 4 (4) | 12: python/mojolearn, arima/impl, bindings/_mojolearn_arima.mojo | b | docs/apple-fast/EXPERIMENTS.md:891 [PASS1 HOLD1] | retire |
| `lane/apple-fast-w3-arima` | 2026-10-04 | 1 (1) | 12: arima/impl, docs/apple-fast, tools/arima_fast_fit_quality.py | b | docs/apple-fast/EXPERIMENTS.md:906 [PASS1 HOLD1] | retire |
| `lane/apple-fast-w3-dw` | 2026-10-04 | 2 (2) | 5: gbdt/methods, docs/apple-fast, tools/aft_w3dw.py | b | docs/TOMBSTONES.md:607 [PASS2] | retire |
| `lane/apple-fast-w3-prep` | 2026-10-04 | 4 (4) | 7: bindings/_mojolearn_x_prep.mojo, python/mojolearn, tools/xprep_pinned_pair.py | b | docs/TOMBSTONES.md:1839 [] | retire |
| `lane/apple-fast-w4-decomp` | 2026-10-04 | 6 (6) | 8: bindings/_mojolearn_estimators.mojo, bindings/_mojolearn_x_decomp.mojo, decomposition/estimator.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:914 [PASS3 DEFAULT1 merged1] | retire |
| `lane/apple-fast-w4-decomp-harness-r1` | 2026-10-04 | 7 (7) | 8: bindings/_mojolearn_estimators.mojo, bindings/_mojolearn_x_decomp.mojo, decomposition/estimator.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:1176 [ported3 HOLD-quality2 DEFAULT1] | retire |
| `lane/apple-fast-w4-eigh` | 2026-10-04 | 4 (4) | 4: tools/eigh_w4_pair.py, tools/eigh_w4_quality.py, x_decomp/device.mojo | a | EIGH_FAST_TRIDIAG(_PANELS) on main (default with _OFF); 74% of unique lines present | retire |
| `lane/apple-fast-w4-linalg` | 2026-10-04 | 3 (3) | 7: docs/apple-fast, tools/chol_fast_nb512_pair.py, tools/chol_fast_tall_quality.py | b | docs/apple-fast/EXPERIMENTS.md:916 [PASS1 HOLD-speed1] | retire |
| `lane/apple-fast-w4-mcd` | 2026-10-04 | 2 (2) | 5: docs/apple-fast, tools/mcd_compat_ab.sh, tools/mcd_profile.sh | a | EXPERIMENTS KEPT/DEFAULT, defines on main (e.g. KPCA lanczos_dev.mojo, fastprep2.mojo) | retire |
| `lane/apple-fast-w4-shap` | 2026-10-04 | 3 (3) | 7: docs/apple-fast, python/mojolearn, tools/pshap_delta_pair.py | a | 93% of unique added code lines present on main | retire |
| `lane/apple-fast-w4-small` | 2026-10-04 | 6 (6) | 14: python/mojolearn, bindings/_mojolearn_resample.mojo, bindings/_mojolearn_x_neighbors.mojo | a/b | winners ported, rest documented: docs/apple-fast/EXPERIMENTS.md:909 [PASS4 HOLD2 PARKED1 DEFAULT1 merged1] | retire |
| `lane/apple-fast-w4-tslu` | 2026-10-04 | 3 (3) | 7: docs/apple-fast, tools/chol_fast_nb512_pair.py, tools/chol_fast_tall_quality.py | b | docs/TOMBSTONES.md:1973 [PASS2] | retire |
| `lane/apple-identical-steps` | 2026-09-26 | 2 (2) | 1: ensemble/decisiontree | a | candidate-audit-2026-10-04.md:21 "content-absorbed" | retire |
| `lane/apple-merged-check` | 2026-09-28 | 1 (1) | 17: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple-merged-check2` | 2026-09-28 | 1 (1) | 17: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple-merged-check3` | 2026-09-28 | 1 (1) | 17: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple-merged-owed-check` | 2026-09-28 | 1 (1) | 17: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple-verify-20261001` | 2026-09-30 | 1 (1) | 1: tools/apple_verify_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple2-merged-check` | 2026-09-28 | 1 (1) | 3: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple2-merged-check2` | 2026-09-28 | 1 (1) | 3: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple2-merged-check3` | 2026-09-28 | 1 (1) | 3: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple2-merged-check4` | 2026-09-28 | 1 (1) | 3: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple2-merged-check5` | 2026-09-28 | 1 (1) | 3: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/apple3-merged-check` | 2026-09-28 | 2 (1) | 5: tools/merged_check | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/bench-board-m3ultra` | 2026-09-29 | 37 (30) | 20: bench/results, docs/BENCH_BOARD.md, tools/bench_board.py | b | Sep 29-30 board tooling; main board tools (tools/bench_board*.py) moved on; boards now via six_lane_grid + lq | retire |
| `lane/bf16-units` | 2026-09-28 | 1 (0) | 1: tools/dev_pod.sh | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/board-resume-20260930` | 2026-09-30 | 12 (12) | 4560: bench/results, tools/bench_board.py, tools/bench_board_resume.py | b | Sep 29-30 board tooling; main board tools (tools/bench_board*.py) moved on; boards now via six_lane_grid + lq | retire |
| `lane/catboost-parity-on-lm-attention-fallback` | 2026-09-19 | 7 (5) | 41: python/mojolearn, gbdt/host, gbdt/methods | a | 89% of unique lines on main (ensemble.py, gbdt ordered boosting) | retire |
| `lane/categorical-feature-cpu` | 2026-09-19 | 1 (0) | 2: python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/classical-pass-measure-20260930` | 2026-09-30 | 22 (15) | 117: bench/results, EXPERIMENTS.md, bindings/_mojolearn_mamba.mojo | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/cluster-apple-prof` | 2026-09-28 | 7 (3) | 6: hdbscan/impl, bench/dbscan_m4_probe.py, core/stage_prof.mojo | b | one-off diagnostic/profiling probes (STAGE_PROF etc., absent on main) | retire |
| `lane/cluster-speed-base` | 2026-09-27 | 1 (1) | 1: bench/x_cluster_speed.py | a | 98% of unique added code lines present on main | retire |
| `lane/cpu-par-fallback` | 2026-09-28 | 1 (1) | 6: python/mojolearn, tools/cpu_path_audit.py | b | WIP plain CPU fallback for par-* drivers; CPU routes are not a GPU fallback (CLAUDE.md GPU rules); cpu_path_audit.py on main | retire |
| `lane/cpu-training-e3` | 2026-09-14 | 5 (0) | 24: python/mojolearn, cluster/host, metrics/host | a | git cherry: all 5 commits patch-equal on main | retire |
| `lane/cpu-training-gate-speed` | 2026-09-15 | 1 (1) | 3: .github/workflows, tools/cpu_identity_gate_check.py, tools/identity_break.py | b | Sep 15 identity-gate tooling/records; identity now recorded by tools/record_identity_column.sh (REPEATS=1) | retire |
| `lane/data-runtime-million-cpu` | 2026-09-20 | 1 (0) | 4: bench/bench_entropy_host.py, bench/evidence, bindings/_mojolearn_metrics_host.mojo | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/distance-pairwise-cpu` | 2026-09-19 | 1 (0) | 4: bindings/_mojolearn_core_host.mojo, core/knn_host_predict.mojo, neighbors/checks | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/ensemble-meta-cpu` | 2026-09-19 | 1 (0) | 3: bindings/_mojolearn_forest_host.mojo, core/gbdt_host_predict.mojo, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/expose-d-fixes` | 2026-09-14 | 6 (0) | 51: python/mojolearn, docs/lanes, packaging/linux | a | git cherry: all 6 commits patch-equal on main | retire |
| `lane/forecast-timeseries-cpu` | 2026-09-19 | 1 (0) | 3: bindings/holtwinters_host_predict.mojo, holtwinters/checks, holtwinters/host | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/forest-train-speed` | 2026-09-17 | 1 (1) | 3: extratrees/impl, docs/lanes | b | candidate-audit-2026-10-04.md:37 "NOT YET BUILT", abandoned | retire |
| `lane/forest-training-cpu-20260919` | 2026-09-19 | 1 (0) | 1: ensemble/host | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/fp-contract-off-exp` | 2026-09-25 | 1 (0) | 24: bindings/build.sh, bindings/build_arima.sh, bindings/build_byte_lm.sh | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/fp-contract-off-study` | 2026-09-25 | 2 (1) | 1: bench/results | b | study result only (bench/results/fp_contract_off_2026-09-26 README: contract=off moves 16 of 627 cells); the build-flag hook landed via lane/fp-contract-off-exp (patch-equal on main); the README is not on main | retire (resolved: see Resolved) |
| `lane/gbdt-depthwise-fast` | 2026-09-20 | 1 (0) | 1: gbdt/methods | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/gbdt-rest` | 2026-09-16 | 4 (4) | 24: python/mojolearn, bench/results, gbdt/host | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/gemm-p1-fold-bypass` | 2026-09-19 | 4 (4) | 2: gemm/checks, tools/gemm_p1_hotaisle_body.sh | a | candidate-audit-2026-10-04.md:21 "content-absorbed" | retire |
| `lane/gram-prep-million-cpu` | 2026-09-20 | 1 (0) | 3: bench/bench_gram_splitk.py, bench/evidence, decomposition/host | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/host-token-split-2` | 2026-10-01 | 3 (2) | 1: training/byte_lm_host.mojo | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/host-token-split-2-measure` | 2026-09-30 | 2 (1) | 1: training/byte_lm_host.mojo | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/host-token-split-measure` | 2026-09-30 | 2 (1) | 1: tools/hostpr_both.sh | a | 100% of unique added code lines present on main | retire |
| `lane/hr2-kpca-seq` | 2026-10-02 | 3 (3) | 6: bindings/_mojolearn_x_sequence.mojo, sequence/exec_device.mojo, sequence/fold32.mojo | a | candidate-audit-2026-10-04.md:29 owed GARCH item; SEQ_WARP_FIT team fit now on main (sequence/garch.mojo:66) | retire |
| `lane/identical-body-tiles-20261005` | 2026-10-05 | 2 (1) | 1: experiments/identical_speed | a | branch only selects a profile in experiments/identical_speed/selection.json; the defines are on main; EXPERIMENTS.md:1445-1449 rows (OPEN, unmeasured; GROUP_TILES_BODY later DROPPED-noise in TOMBSTONES) | retire |
| `lane/identical-group-slack2-20261005` | 2026-10-05 | 2 (1) | 1: experiments/identical_speed | a | branch only selects a profile in experiments/identical_speed/selection.json; the defines are on main; EXPERIMENTS.md:1445-1449 rows (OPEN, unmeasured; GROUP_TILES_BODY later DROPPED-noise in TOMBSTONES) | retire |
| `lane/identical-group-slack8-20261005` | 2026-10-05 | 2 (1) | 1: experiments/identical_speed | a | branch only selects a profile in experiments/identical_speed/selection.json; the defines are on main; EXPERIMENTS.md:1445-1449 rows (OPEN, unmeasured; GROUP_TILES_BODY later DROPPED-noise in TOMBSTONES) | retire |
| `lane/identical-one-page-20261005` | 2026-10-05 | 2 (1) | 1: experiments/identical_speed | a | branch only selects a profile in experiments/identical_speed/selection.json; the defines are on main; EXPERIMENTS.md:1445-1449 rows (OPEN, unmeasured; GROUP_TILES_BODY later DROPPED-noise in TOMBSTONES) | retire |
| `lane/identical-packed64-20261005` | 2026-10-05 | 2 (1) | 1: experiments/identical_speed | a | branch only selects a profile in experiments/identical_speed/selection.json; the defines are on main; EXPERIMENTS.md:1445-1449 rows (OPEN, unmeasured; GROUP_TILES_BODY later DROPPED-noise in TOMBSTONES) | retire |
| `lane/identity-record-next` | 2026-09-15 | 3 (3) | 19: bench/results, python/mojolearn, docs/lanes | b | Sep 15 identity-gate tooling/records; identity now recorded by tools/record_identity_column.sh (REPEATS=1) | retire |
| `lane/ivf-filter-fix-measure` | 2026-10-01 | 1 (1) | 2: tools/ivf48_ab.sh, tools/ivf48_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/kernel-methods-cpu` | 2026-09-19 | 1 (0) | 2: gaussian_process/host, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/kmeans-mojo-1-2` | 2026-09-26 | 7 (7) | 442: gbdt/methods, neighbors/checks, gbdt/gpu_util | b | candidate-audit-2026-10-04.md:44 Mojo 1.2 migration trial, "parked"; main has since moved toolchains | retire |
| `lane/lane-map-import-graph` | 2026-09-23 | 4 (0) | 7: docs/VERIFY.md, tools/binding_stamps.py, tools/lane_map_import_graph.py | a | git cherry: all 4 commits patch-equal on main | retire |
| `lane/large-data-classical-cpu` | 2026-09-20 | 1 (0) | 3: bindings/_mojolearn_metrics_host.mojo, metrics/host, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/large-metrics-shared-cpu` | 2026-09-20 | 1 (0) | 3: bindings/_mojolearn_metrics_host.mojo, metrics/host, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/linear-apple-base` | 2026-09-28 | 1 (0) | 1: bench/x_linear_speed.py | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/linfit-speed` | 2026-09-29 | 45 (45) | 27: bench/results, x_linear/checks, tools/linfit_speed.py | b | candidate-audit-2026-10-04.md:35 SGD_IN_FIT_KERNEL / X_LINEAR_GLM_TEAM absent, 19 of 20 patches conflict: "abandoned unless re-derived" | retire |
| `lane/lm-attention-fallback` | 2026-10-05 | 3 (2) | 9: docs/lanes, CITATION.cff | b | candidate-audit-2026-10-04.md: "Docs/tooling without Mojo", the old dirty default-checkout branch (deletes handoff docs, edits CITATION.cff) | retire |
| `lane/lm-head-v2-dweight-reuse` | 2026-09-20 | 1 (0) | 39: python/mojolearn, bench/results, bench/evidence | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/loss-metrics-objectives-cpu` | 2026-09-19 | 1 (0) | 3: checks/sample_quantile_unweighted_check.mojo, gbdt/metrics, pixi.toml | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/lowbit-amd-tuned-lb` | 2026-09-29 | 1 (1) | 1: gemm/checks | b | one-off diagnostic/profiling probes (STAGE_PROF etc., absent on main) | retire |
| `lane/lowbit-apple-now` | 2026-09-29 | 1 (1) | 1: gemm/checks | b | one-off diagnostic/profiling probes (STAGE_PROF etc., absent on main) | retire |
| `lane/lowbit-apple-tuned` | 2026-09-29 | 21 (20) | 57: bench/results, gemm/checks, tools/lowbit_apple_tuned | d | Sep 29 low-bit (int15 Apple tuned GEMM, e2e probes, quality bench) not on main (gemm_int15_apple_tuned.mojo, bench/lowbit_quality absent); no verdict found in docs | retire (resolved: see Resolved) |
| `lane/lowbit-blocks` | 2026-09-29 | 1 (1) | 1: docs/lanes | b | progress-note only (docs/lanes/progress/*.md); superseded by later handoffs | retire |
| `lane/lowbit-e2e` | 2026-09-29 | 5 (5) | 11: bench/results, tools/lowbit_e2e, docs/HANDOFF_LOWBIT_E2E_2026-09-29.md | d | Sep 29 low-bit (int15 Apple tuned GEMM, e2e probes, quality bench) not on main (gemm_int15_apple_tuned.mojo, bench/lowbit_quality absent); no verdict found in docs | retire (resolved: see Resolved) |
| `lane/lowbit-quality` | 2026-09-29 | 25 (25) | 299: bench/results, bench/lowbit_quality, docs/lanes | d | Sep 29 low-bit (int15 Apple tuned GEMM, e2e probes, quality bench) not on main (gemm_int15_apple_tuned.mojo, bench/lowbit_quality absent); no verdict found in docs | retire (resolved: see Resolved) |
| `lane/m3-sweep` | 2026-09-28 | 2 (2) | 2: docs/lanes, tools/apple_sweep_run.py | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/mamba-opp` | 2026-09-29 | 6 (5) | 8: tools/bench_board.py, tools/bench_board_neural.py, tools/bench_board_params.py | b | Sep 29-30 board tooling; main board tools (tools/bench_board*.py) moved on; boards now via six_lane_grid + lq | retire |
| `lane/mamba3-amd-regs2h-measure` | 2026-10-01 | 2 (1) | 1: tools/pr50_amd.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/mamba3-padding-perf` | 2026-09-19 | 2 (0) | 2: docs/lanes, mamba/checks | a | git cherry: all 2 commits patch-equal on main | retire |
| `lane/mamba3-statepass` | 2026-09-09 | 23 (23) | 188: bench/results, mamba/impl, mamba/checks | d | Sep 9 Mamba3 state-pass (direct transfers, tiled QKS/YSTATE); defines absent, mamba/impl/mamba_ssm/ops/mamba3_siso.mojo gone from main (rewritten as autograd/kernels/mamba3.mojo); likely superseded, unverified | retire (resolved: see Resolved) |
| `lane/manifold-spectral-cpu` | 2026-09-19 | 1 (0) | 2: bench/spectral_host_eig_price.mojo, spectral/checks | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/merged` | 2026-10-02 | 40 (0) | 238: python/mojolearn, gbdt/methods, x_decomp/checks | a | git cherry: all 40 commits patch-equal on main | retire |
| `lane/mixture-anomaly-cpu` | 2026-09-19 | 2 (0) | 4: bench/gmm_host_scoring_price.py, bench/iforest_host_score_price.py, isolation_forest/checks | a | git cherry: all 2 commits patch-equal on main | retire |
| `lane/model-load-shared-cpu` | 2026-09-19 | 1 (0) | 4: python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/model-selection-cpu` | 2026-09-19 | 1 (0) | 2: checks/model_selection_kfold_perf.py, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/mojo-1-2-trial` | 2026-09-26 | 2 (2) | 13: bench/results | b | candidate-audit-2026-10-04.md:44 Mojo 1.2 migration trial, "parked"; main has since moved toolchains | retire |
| `lane/neighbors-rest` | 2026-09-16 | 6 (5) | 212: bench/results, python/mojolearn, neighbors/checks | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-apple-base` | 2026-09-28 | 1 (1) | 1: tools/apple_speed_neural | a | 93% of unique added code lines present on main | retire |
| `lane/neural-apple3-measure` | 2026-10-01 | 2 (2) | 3: tools/pr44_40_mac.sh, tools/pr44_gpu.sh, tools/pr44_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-apple4` | 2026-10-01 | 2 (1) | 3: mamba/host, mamba/impl, tools/neural_experiments.py | a | candidate-audit-2026-10-04.md:42 regs2h AMD S16 merged as PR #50 | retire |
| `lane/neural-experiment-measure-20260930` | 2026-09-30 | 12 (11) | 110: bench/results, bindings/_mojolearn_mamba.mojo, bindings/_mojolearn_transformer.mojo | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass102` | 2026-10-01 | 1 (1) | 1: python/mojolearn | b | host-algorithm switches (X_LINEAR_HOST_ALGOS, SVGP_JITTER_TRIES) in Python glue; absent on main; host routes are debt (CLAUDE.md) | retire |
| `lane/neural-pass13` | 2026-10-02 | 3 (2) | 1: gemm/host | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-pass13-measure` | 2026-09-30 | 3 (2) | 2: gemm/host, tools/pr19_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass144` | 2026-10-02 | 3 (1) | 9: bindings/_mojolearn_linalg.mojo, bindings/_mojolearn_linalg_host.mojo, decomposition/checks | b | candidate-audit-2026-10-04.md:41 superseded (round-robin eigh on every column) | retire |
| `lane/neural-pass17-measure` | 2026-10-01 | 2 (1) | 3: tools/cell_ab_job2.sh, tools/pr23_mac.sh, tools/pr23_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass19` | 2026-09-30 | 1 (1) | 1: gemm/host | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-pass26-measure` | 2026-10-01 | 5 (1) | 2: tools/pr31_gpu.sh, tools/pr31_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass30-measure` | 2026-10-01 | 2 (1) | 2: tools/cell_ab_job2.sh, tools/pr35_36_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass31-measure` | 2026-10-01 | 2 (1) | 1: tools/cell_ab_job2.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass32-measure` | 2026-10-01 | 2 (1) | 2: tools/cell_ab_job2.sh, tools/pr37_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass34-measure` | 2026-10-01 | 2 (1) | 3: tools/cell_ab_job2.sh, tools/pr38_nv.sh, tools/pr38_opt.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass35-measure` | 2026-10-01 | 3 (2) | 2: tools/ivf_ab.sh, tools/ivf_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass36` | 2026-10-01 | 4 (4) | 1: x_decomp/device.mojo | b | candidate-audit-2026-10-04.md:40 DROPPED: LU fused 46% slower; QR tiles lost on L40S | retire |
| `lane/neural-pass36-measure` | 2026-10-01 | 10 (7) | 5: tools/cell_ab_job2.sh, tools/pr40_nv.sh, tools/pr40b_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass37-measure` | 2026-10-01 | 2 (1) | 2: tools/cell_ab_job2.sh, tools/pr41_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass38-measure` | 2026-10-01 | 2 (1) | 2: tools/cell_ab_job2.sh, tools/pr43_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass39-measure` | 2026-10-01 | 1 (1) | 1: tools/cell_ab_job2.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass40-measure` | 2026-10-01 | 1 (1) | 2: tools/cell_ab_job2.sh, tools/pr45_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass41` | 2026-10-01 | 8 (6) | 1: x_decomp/device.mojo | b | candidate-audit-2026-10-04.md:40 DROPPED: LU fused 46% slower; QR tiles lost on L40S | retire |
| `lane/neural-pass41-measure` | 2026-10-01 | 15 (9) | 3: tools/cell_ab_job2.sh, tools/pr46_nv.sh, x_decomp/device.mojo | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass42-measure` | 2026-10-01 | 1 (1) | 2: tools/ivf47_ab.sh, tools/ivf47_nv.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass44-measure` | 2026-10-01 | 1 (0) | 8: bindings/_mojolearn_byte_lm.mojo, core/device_arena.mojo, tools/byte_lm_launch_probe.py | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/neural-pass45-measure` | 2026-10-01 | 6 (3) | 2: tools/pr52_amd.sh, tools/pr52b_amd.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass46-measure` | 2026-10-01 | 4 (2) | 2: tools/nv_pr54.sh, tools/pr54_gpu_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass47-measure` | 2026-10-01 | 5 (4) | 3: tools/neural_pass_job.sh, tools/nv_extra.sh, tools/nv_pass_wrapper.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass48` | 2026-10-01 | 1 (1) | 4: transformer/impl, core/device_scan.mojo, core/scratch_pool.mojo | a | SCRATCH_POOL on main (core/scratch_pool, 8 files; TOMBSTONES/EXPERIMENTS rows) | retire |
| `lane/neural-pass48-measure` | 2026-10-01 | 2 (2) | 5: transformer/impl, core/device_scan.mojo, core/scratch_pool.mojo | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass49-measure` | 2026-10-01 | 3 (2) | 2: tools/nv_pr57.sh, tools/pr57_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass49-on-61` | 2026-10-01 | 1 (0) | 2: training/byte_lm_logits.mojo, training/checks | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/neural-pass5` | 2026-10-01 | 3 (2) | 3: mamba/host, mamba/impl, tools/neural_experiments.py | a | S17 tail on main as the default (autograd/kernels/mamba3.mojo:28, MAMBA3_S17_TAIL_NAIVE reverts) | retire |
| `lane/neural-pass50-measure` | 2026-10-01 | 1 (1) | 2: tools/nv_pr58.sh, tools/pr58_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass53-on-56` | 2026-10-01 | 1 (0) | 1: transformer/impl | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/neural-pass54` | 2026-10-01 | 1 (0) | 1: gemm/checks | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/neural-pass54-measure` | 2026-10-01 | 3 (1) | 2: tools/nv_pr62.sh, tools/pr62_job.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/neural-pass57` | 2026-10-01 | 1 (1) | 1: transformer/impl | a | MOJOLEARN_ATTN_DKDV_BJ16 on main, transformer/impl/llama/fused_attention.mojo:4482 cites PR #64 | retire |
| `lane/neural-pass59` | 2026-10-01 | 1 (1) | 1: transformer/impl | a | 100% of unique added code lines present on main | retire |
| `lane/neural-pass62-on-64` | 2026-10-01 | 1 (0) | 1: gemm/checks | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/neural-pass67` | 2026-10-01 | 6 (1) | 1: transformer/impl | a | MOJOLEARN_ATTN_CACHE_WAIT on main, transformer/impl/llama/modeling_llama.mojo:3899-3903 cites lane/neural-pass67 087231ba3 | retire |
| `lane/neural-pass76` | 2026-10-01 | 1 (1) | 1: python/mojolearn | b | host-algorithm switches (X_LINEAR_HOST_ALGOS, SVGP_JITTER_TRIES) in Python glue; absent on main; host routes are debt (CLAUDE.md) | retire |
| `lane/neural-pass77` | 2026-10-01 | 2 (2) | 2: python/mojolearn, x_cluster/host | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-pass78` | 2026-10-01 | 5 (4) | 3: python/mojolearn, x_decomp/host.mojo, x_decomp/host_simd.mojo | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-pass83` | 2026-10-01 | 3 (3) | 2: python/mojolearn, x_linear/ops.mojo | b | candidate-audit-2026-10-04.md:19 "CPU/host-route work ... CPU routes (undo list) or superseded" | retire |
| `lane/neural-toggle-sweep-20260930` | 2026-09-30 | 18 (14) | 114: bench/results, EXPERIMENTS.md, bindings/_mojolearn_mamba.mojo | b | Sep 30 toggle-sweep tooling; superseded by six_lane_grid + grid_controls | retire |
| `lane/neural-xlw` | 2026-10-02 | 10 (1) | 2: x_linear/device.mojo, x_linear/moments_grid.mojo | b | merge lane of neural-pass132 (SGD minibatch partials); x_linear/moments_grid.mojo on main; SGD batch rule settled Oct 17 (auto batch kept) | retire |
| `lane/ordered-speed-diag2` | 2026-09-29 | 1 (1) | 1: gbdt/methods | b | one-off diagnostic/profiling probes (STAGE_PROF etc., absent on main) | retire |
| `lane/ordered-speed-m2diag` | 2026-09-29 | 1 (1) | 1: checks/kernel_matrix.mojo | b | one-off diagnostic/profiling probes (STAGE_PROF etc., absent on main) | retire |
| `lane/ordered-speed-prespeed` | 2026-09-29 | 1 (1) | 1: bench/speed | a | 100% of unique added code lines present on main | retire |
| `lane/par-harness` | 2026-09-28 | 3 (3) | 6: tools/par_harness, docs/lanes, tools/algos_lane_check.py | a | 99% of unique added code lines present on main | retire |
| `lane/pass4-measure-20260930` | 2026-09-30 | 1 (1) | 1: tools/pass4_amd_job.sh | a | 93% of unique added code lines present on main | retire |
| `lane/probabilistic-million-cpu` | 2026-09-20 | 1 (0) | 4: bench/bench_mutual_info_host.py, bench/evidence, bindings/_mojolearn_metrics_host.mojo | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/probability-calibration-cpu` | 2026-09-19 | 1 (0) | 2: bindings/host_helpers.mojo, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/python-310-311` | 2026-09-23 | 9 (0) | 120: bench/results, python/mojolearn, tools/attention_v2_oracle.py | a | git cherry: all 9 commits patch-equal on main | retire |
| `lane/release-gpu-columns` | 2026-09-22 | 11 (8) | 14: docs/RELEASE_CHECKLIST.md, tools/do_release061_leg.sh, tools/gemm_remote_leg.sh | b | Sep 22-23 release/provider tooling never adopted (tools/release_gpu_columns.py, tools/vultr_leg.sh absent); boxes are RunPod/DO, columns via record_identity_column.sh | retire |
| `lane/release-reuse-bindings` | 2026-09-23 | 6 (0) | 9: packaging/linux, packaging/macos, docs/RELEASE_CHECKLIST.md | a | git cherry: all 6 commits patch-equal on main | retire |
| `lane/rf-mutex-claim-acquire` | 2026-09-16 | 5 (5) | 8: tools/rf_nondeterminism, docs/lanes, ensemble/decisiontree | a | head commit a22fa27be: "the fix landed on main as 36ec47446, in a better shape" | retire |
| `lane/rf-score-weighted-nondeterminism` | 2026-09-16 | 17 (17) | 26: tools/rf_nondeterminism, docs/lanes, ensemble/checks | a | 89% of unique lines on main; candidate-audit-2026-10-04.md:38 diagnostic probe, superseded by the RF lifetime fix | retire |
| `lane/s17-measure-20260930` | 2026-09-30 | 4 (4) | 4: mamba/host, mamba/impl, tools/neural_experiments.py | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/sabotage-no-host-routes` | 2026-10-02 | 1 (1) | 1: x_linear/device.mojo | b | candidate-audit-2026-10-04.md:16 hook sabotage test, "do not merge" | retire |
| `lane/stage-diag-20261001` | 2026-10-01 | 3 (2) | 3: tools/amd_toggle_sweep.sh, tools/stage_diag_amd.sh, tools/stage_diag_mac.sh | b | measurement/job harness (docs/identical/candidate-audit/candidate-audit-2026-10-04.md: "Measurement / diagnostic / tooling jobs ... do not merge"); lq replaced pr*_job scripts | retire |
| `lane/tokenizer-text-cpu-20260919` | 2026-09-19 | 1 (0) | 2: bench/speed, tokenizer/impl | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/tsa-classical-utils-cpu` | 2026-09-19 | 1 (0) | 2: holtwinters/host, python/mojolearn | a | git cherry: all 1 commits patch-equal on main | retire |
| `lane/vultr-leg` | 2026-09-23 | 3 (3) | 4: docs/HANDOFF_gpt3_small_run_2026-09-23.md, tools/lm_run_driver.py, tools/tests | b | Sep 22-23 release/provider tooling never adopted (tools/release_gpu_columns.py, tools/vultr_leg.sh absent); boxes are RunPod/DO, columns via record_identity_column.sh | retire |
| `lane/wheel-smoke-do-fallback` | 2026-09-23 | 4 (0) | 4: docs/RELEASE_CHECKLIST.md, tools/release.py, tools/release_wheel_smoke.sh | a | git cherry: all 4 commits patch-equal on main | retire |
