# Claude handoff: IDENTICAL integration, NVIDIA/AMD speed, and PTX

Checkpoint date: 2026-10-04. This file supersedes older operating rules where they conflict with the latest user instructions below. Read AGENTS.md and CLAUDE.md in the integration worktree first; latest explicit user instructions override older conflicting guidance. Do not resume the unrelated RTL/FPGA handoff pasted into this conversation.

## 0. Start here

**Work in one integration branch:** `integration/identical-all-20261004`, worktree `/Users/andrewhendel/mojolearn-wt/identical-all`, remote `origin` = `https://github.com/mojolearn/mojolearn.git`.

The default project directory `/Users/andrewhendel/CascadeProjects/mojolearn` is a different, dirty `lane/lm-attention-fallback` checkout. Do not edit/reset it or switch its branch. Frozen remote build trees are intentional snapshots of the integration branch, not competing development lanes.

**Bottom line of actual progress:** the planned 50-library ON/OFF builds finished on both rented NVIDIA Blackwell and AMD MI325X. The first 25-gate-per-arm quality batch finished on both: **39 PASS, 7 FAIL, 4 not applicable**. Broad 47-case identity and speed measurement have NOT run. No speed winner from this integration wave has been promoted to main. Current work fixes the failed harness/dependency gates without redoing successful work, then resumes identity and one controlled speed batch.

Read these first:

1. `docs/identical/optimization-ledger.json` — rules, decisions and proof links.
2. `docs/identical/original-handoff-coverage.json` — original candidate accounting and gaps.
3. `bench/results/identical-integration-20261004/manifest.json` and individual receipts — retained findings, not a final release qualification.
4. `tools/identical_wave_plan.json` — medium plan, now corrected to 52 builders.
5. The live job/status files in sections 2–4 before starting anything.

Original background: `~/mojolearn-evidence/HANDOFF_2026-10-04_identical-speed-and-ptx.md`. Its old Apple ban/permission history, delete-OFF-toggle advice, and exact verification order have been superseded as described below.

## 1. Latest user instructions — preserve these

- This session owns **all IDENTICAL and PTX** coordination. Use subagents for real fixes. **Do not contact peer sessions.** Earlier peer coordination has ended.
- **NO Apple checks now:** no local/M2/M3 Apple GPU or CPU verification or compile matrix. The proposed M2 move was cancelled before any workload or setup was launched there. Existing unrelated Apple jobs were not touched. Local lightweight orchestration, Git, SSH and result harvesting remain in use.
- Focus iteration on rented **NVIDIA and AMD**. Exhaustive combined-source verification is deferred until the combined work is on main, before a future PyPI release. Do not claim Apple is qualified from current limited proofs.
- Bitwise identity is **between vendors within a candidate version/arm**, not against old-version bits or necessarily ON versus OFF. Quality must remain acceptable within noise, and speed must improve before a candidate is a winner.
- No CPU numerical fallback in GPU algorithm execution. Separate host verification and retained-output statistics are allowed on the Linux servers; do not time host as a GPU arm.
- **Our unpublished source builds only. Opponents come from stored results; never execute opponents or install PyPI mojolearn.** Source-built packaging artifacts are allowed.
- One measured invocation per arm. Existing chosen board contract is one untimed warmup plus one measured sample. Correctness diagnostics are untimed and separately labeled. No full race reruns disguised as validation.
- Keep explicit optimization ON/OFF controls, including rollback after promotion. Keep comments explaining changes. Required correctness fixes stay in both arms; OFF must not reintroduce a broken Python initializer or invalid NaN boundary.
- Rent fast suitable boxes, hold them, use R2, and **start verified result harvesting BEFORE workloads**. Existing boxes/healthy jobs should not be disrupted or duplicated.
- Full logs to files; inspect exit codes, structured summaries and bounded `rg`/`grep`. Tell every subagent the same. Do not dump large logs or transcripts.
- Commit and push work on the integration branch. Do not force-push. Publishing/PyPI upload is OUT OF SCOPE.

## 2. Machines, holds, and watchers

All three boxes are held. Do not accidentally rent duplicates. Rates below are previously reported rates, not refreshed quotes.

| Box | SSH | ID / role |
|---|---|---|
| NVIDIA Blackwell RTX PRO 6000 Workstation 96 GB | `ssh -p 10652 root@64.119.209.250` | RunPod `54zn9tbvoqw70h`, $1.69/h. Main medium wave. `sm_120`. Provider caps power at 500 W; supported 600-W request was denied, do not keep retrying. |
| AMD MI325X VF | `ssh root@162.243.193.137` | DigitalOcean droplet `606074135`, $3.80/h. `gfx942`. |
| NVIDIA RTX 4090 | `ssh -p 14625 root@213.173.109.16` | RunPod `k1tnp0gtira6bq`, account burn previously about $0.77/h. `sm_89`. Current-source PTX probe completed here; currently spare/held. |

Toolchain on all: `/root/mojolearn/.pixi/envs/default`, Mojo `1.0.0(ed45d567)`. Use explicit architectures; both native and wave builders have architecture flags. Do not assume old sm_89 defaults on Blackwell. Shared remote compile semaphore: `/root/mojolearn-evidence/compile_slot.sh`; per compiler `-j 1`, max four shared slots. Blackwell has many CPU cores but its EPYC7663 per-core compilation is slower than the 4090 host's Ryzen7950X; fastest GPU does not imply fastest compiler.

**Collectors/lease renewers are independent live background processes on the local Mac:**

- PID `98987`: `~/mojolearn-evidence/identical-all/remote_supervisor.py`, config `supervisor-config.json` there. Covers 4090 and AMD.
- PID `71989`: `~/mojolearn-evidence/identical-blackwell/remote_supervisor.py`, config `supervisor-config.json` there. Covers Blackwell.
- Read each `supervisor-status.json`, `supervisor.log`, per-box `copy.log`, `renew.log`, `r2.log`. At the handoff check, both were ready, fresh, copy_rc=0; AMD R2 verification also reported rc0. These collection/lease processes should remain running after handoff.
- They copy `/root/lq/` about every30s, renew leases/deadmen about every30min, and upload verified R2 snapshots periodically. They exclude source/virtualenv/Git trees. Keep logs OUTSIDE directories named `source/`.
- Local mirrors: `identical-all/nv/lq`, `identical-all/amd/lq`, `identical-blackwell/nv/lq` under `~/mojolearn-evidence/`.
- These collectors do NOT automatically fix failures. Separate phase watchers existed and have stopped on quality failures. Do not mistake a healthy collector for a running validation job.

Lease state: `~/mojolearn-evidence/devpods/identical-nv/`, `.../devpods/identical-blackwell/`, and `~/mojolearn-evidence/identical-all/amd-box/`. Existing renewal scripts/configs are authoritative. Do not restart an already-live supervisor or replace its state blindly.

Credentials exist in local `~/.mojolearn_runpod_key`, `~/.mojolearn_do_token`, `~/.mojolearn_r2`, `~/.mojolearn_hotaisle_key`. Never print or commit contents. Hot Aisle was checked but no box rented. Older balance snapshot is `identical-all/provider-balances.json`; do not interpret it as a live balance.

## 3. Frozen source and completed wave state

- Numerical source for the current medium wave: **`a4d01a130c8d05ff0208a39989d36b86b69aa0ed`**. Includes RF lifetime fix, LU NaN fix, linear initializer fix, and Blackwell SGD launch-bound metadata.
- Original external prepare harness: **`fb2a9692692509a0223e5c286846839b25e3450b`**.
- Both servers: **`/root/lq/medium-wave-a4d-fb2`**. `prepare.json` PASS; original `on/source`, `off/source`, per-arm `build-products.json`, exact plan and hashes retained.
- Builds reused six proven modules per arm, then compiled remaining44. No need to repeat these50 libraries. Provenance helper verifies source, architecture, arm, compiler/lock, complete artifact hashes and explicit dependency-equivalent donor graphs.
- Original plan47cases/50builders/25quality gates. Mamba pair replaced an incorrectly listed ByteLM pair. The plan is now52builders because linalg + linalg_host were missing.
- Original `quality.json`: **ON21PASS/4FAIL; OFF18PASS/3FAIL/4N/A**, on BOTH vendors. All failed logs and original receipts must stay unchanged.
- Broad `identity.json` absent; no `timing.json` run. Do not mark a wave PASS merely from targeted earlier tests.
- Blackwell old stage watcher PID34220 has STOPPED with `unexpected quality gate failure`. AMD old quality runner187242 and watcher189218 finished; watcher blocked as required. Do not restart the same failed output directory or rerun the whole quality batch.

### Exact quality failures and repair ownership

| Failure | Cause / intended action |
|---|---|
| ON `dart_reference` | Old one-row classification tolerance: ON accuracy .76824 versus OFF .76834 and old minimum .76833. ON logloss .5290970868 versus OFF .5290595562. Regression R2 .9254969788 versus .9254970149. Saved predictions now permit paired noise analysis, no refit. Do not silently change old receipts or assume failure-to-reject proves equivalence. |
| ON `iforest_lifetime` | Plan omitted required `--out`. Fixed by `bdc511fff`; invoke corrected gate with a fresh output directory. |
| Both `resident_qr_memory` | Missing `_mojolearn_linalg.so` and host counterpart. Future plan fixed `ad970dfca`; build only missing pair, then run affected gate. |
| Both `sgd_nan` | Stale import from `mojolearn.linear_model`; public `mojolearn` import is correct. Dedicated CNN/SGD gates already exercised finite/NaN/overflow refusal. Fix test import and rerun only this gate. |
| OFF `lu_nan` | OFF `solve()` factors then calls public `lu_solve`, which rejects overflowed nonfinite factors; ON resident gesv returns raw NaNs. Both GPU/host have matching OFF refusal. Gate must compare factor/pivot raw bytes, require nonvacuous NaNs, retain public lu_solve refusal, compare OFF matched refusal and ON raw solve outputs. Do NOT normalize test output or weaken production arithmetic. |

No GPU fault or timeout caused these original failures. Most are setup/expectation defects; DART quality still needs actual adjudication.

## 4. Work under management at handoff — inspect before duplicating

Codex subagents were asked to finish current atomic edits and hold new long jobs while this handoff was written. Existing builds/collectors remain running. Confirm final live status and any delta note at the end of this file.

### Missing linalg and revised qualification (callpath_review)

- Controllers launched: Blackwell **PID39226**, AMD **PID194914**.
- Paths on each: `/root/lq/linalg-a4d-on`, `/root/lq/linalg-a4d-off`; native numerical source remains a4d, GPU+host linalg only.
- At final checkpoint **both arms on BOTH boxes are PASS**, including GPU and host linalg. Blackwell controller39226 is no longer present. Recheck receipts before starting anything; do not rebuild these modules.
- Intended new immutable wave revision: **`/root/lq/medium-wave-a4d-qualified-v2`**.
- Copy/hash-verify original50 libraries plus new linalg pair into fresh source trees; freeze new complete inventories. Link original source/build/plan receipts and original successful gates. Do not modify old inventories or pretend new files were in the original prepare.
- Revision helper is now committed at **320beb6c0**: `tools/identical_wave_revision.py`; four Linux refusal tests PASS. It has NOT been run to create v2 yet. CLI currently takes `--old`, `--out`, `--plan`, `--old-plan`, optional `--repo`, `--supplement-prefix`.
- Reconcile old39PASS checks plus exact repaired supplemental checks against the new explicit plan. Then run broad identity ON/OFF once; timing remains gated on required quality and cross-vendor proof.

### IF/SGD/LU failed-cell repairs (compile_inventory)

- IF plan fix committed. SGD import and LU per-arm fixture repairs are now committed at **e98f533613868c239c4c39c674d8531c1ae02231** (`tools/idn_all_checks.py`). These corrected numerical gates have NOT been executed.
- No active owned runtime/controller at initial checkpoint; supplements NOT launched yet.
- Run only affected corrected gates on both vendors/arms with source a4d + separately pinned harness, preserve original failures, store append-only supplemental receipts. Coordinate plan changes with linalg revision owner.
- Earlier local compile controllers68808/63441 and their four active compiler/wrapper pairs were cancelled at user request. All are gone; partial logs retained as `CANCELLED_USER`, rc=null, NOT pass/failure. Broad diagnostic matrix ended318/349PASS, four cancelled attempts. Do not resume it locally or move it to M2.

### DART quality statistics (remote_plan)

- NO calibration or comparison job started at handoff. No active PID; all owned code committed.
- Tools: `tools/identical_quality_noise.py`, `tools/identical_dart_noise_prepare.py`, usage `tools/identical_quality_noise_USAGE.txt`.
- AMD data-only prepared files: **`/root/lq/dart-temporal-policy/prepared/`** containing `targets.npz`, `target-metadata.json`, `protocol-block1024.json`, `protocol-block2048.json`, `protocol-block4096.json`, `manifest.json`.
- Dataset is temporal-tail + deterministic stride, NOT IID or raw-row chronological. Recovering standardized hour/day/weekday gives hour-level ordering. Classification146February hours; regression694hours plus one explicitly inferred March1 spillover. Preserve these limits; minutes unavailable.
- Approved policy: primary moving-block4096rows, sensitivities1024/2048;2000fixed-seed paired resamples; accuracy and R2 must pass ALL three policies. Logloss diagnostic. It estimates conditional heldout uncertainty, not training-seed or long-horizon drift.
- **Freeze all three OFF-only margins and record hashes BEFORE reading candidate predictions/deltas for the adjudication.** Calibration CLI accepts only OFF; compare requires the frozen hash. Original aggregate quality numbers were already observed during gate runs; do not tune policy to their outcome.
- Saved ON/OFF predictions: `/root/lq/medium-wave-a4d-fb2/{on,off}/quality/dart_reference-outputs/*.npz`; metadata `dart_reference.json` beside them. No new DART fits needed.
- Next blocker is a small provenance adapter: wave `build-products.json` is flat relative-path→hash; statistics require strict PASS native receipt provenance accounting for all loaded native binaries. Validate prepare identity, imported native receipts/attestations, and fresh builder records, then remap copied artifact paths by exact relative paths and hashes. Do not fabricate fresh-build receipts.
- Known OFF donor: `/root/lq/cnn-sgd-a4d01a130-off/native-build.json`, SHA256 `9c39dbb3db7657a87eec12b95d7880e2d4796013a6adf70ea1517ac7753ca7b2`; producer attestation SHA256 `7bfd4867b5fe57cdbbed27d90823a56f5910e1527bf412c3e3137ca73597b222`. Verify rather than trusting this note.
- Execute statistics on Linux CPU, not the Mac. No models, opponents or timers. Result can PASS/FAIL/INCONCLUSIVE/INSUFFICIENT; no quiet threshold relaxation.

## 5. What actually passed, and what was changed

Committed evidence directory: `bench/results/identical-integration-20261004/`. It contains precise scope/source caveats. Many earlier proofs use different source snapshots; they are useful evidence, NOT one final uniform-source qualification.

- **RF/DART lifetime fix366a56a0a:** two persistent pinned phase upload buffers prevent histogram H2D DMA reading overwritten partition metadata. No arithmetic changes or extra wait. Delayed-upload sabotage reproduces old failure and passes fix on NVIDIA/AMD. Full classifier+regressor2009outputs/tree-state arrays match NVIDIA/AMD/Linuxhost; AMD two-repeat4018outputs match. AMD initial diagnostic is explicitly patch-equivalent; newer medium wave compiles all at a4d.
- **CNN/SGD:** initializer export defect fixed `be3348860`; Blackwell1024-thread launch resource failure fixed via matching launch-bound metadata `a4d01a130`, without arithmetic/order/width changes.25small/medium cases perarm/column pass NVIDIA/AMD/Linuxhosts; mediumCNN257x256, SGD4097x65 with multiblock4096batch+one-rowtail. Receipt `cnn-sgd-bits.json`, commit77f4b5a7a. ON/OFF SGD bits also match; this is not required for every optimization.
- **LU:** computed NaN payload boundary canonicalization `301156228` makes factor/solve result words match. Finite outputs unchanged. No test normalization. Per-arm OFF refusal gate remains to repair as above.
- **PCA/TSVD:** host mean/fold repair221787de7;33rawoutputs match all tested GPU/host columns; earlier explained_variance/ratio mismatch fixed.
- **Block Jacobi EIGH:** disabled by default0dd0b4de3 after genuine tight convergence refusal on captured32x32pivot. Keep strict convergence; do not loosen tolerance to enable it. Explicit opt-in remains quarantined. Scalar fallback large4096/513 quality passed.
- **Shared callpath:** vendor-neutral persistent typed storage/grouped completion, scalar/primitives/transport fixture digests matched4090,Blackwell,AMD; earlier small AppleGPU/M4CPU fixtures matched too before Apple work was stopped. Not every estimator is wired. This does NOT prove single-call timing gains from grouped waits.
- **Other targeted tests:** GramCD, Prophet, fusedGMM, blockedPowerTransformer bits; IF lifetime; DBSCANshared-context test repair; KDE; Mamba host generator repair. See individual receipts and original handoff.
- All32handed-off lane heads,3aggregates and PTXhead are ancestors of integration. Two vendor-neutral callpath implementation commits are exact cherry-equivalents; unrelated citation/stale-doc commits intentionally excluded. No missing source merge identified. Ancestry is not proof every later merge hunk is correct.

## 6. PTX is part of this same task

Current-source preliminary proof: **`e82c12650cd9f31b8dc0216157fb59b22a9d9313`**, held4090, `/root/lq/ptx-current-e82c12650`.

-211tests +24subtests PASS.
-Fresh native sm89 and sm80 PTX builds/run both rc0; forced PTX run uses `CUDA_FORCE_PTX_JIT=1`, `CUDA_CACHE_DISABLE=1`.
-Twelve primitive comparisons/fiveadapters/dirtyreuse/tails; both digest `12251619760217914789`.
-Tracked proof `current-ptx-probe.json`, commitb9d8a2583.

**NOT full PTX release qualification/admission.** Historical5c156c5f results cannot qualify the newer source. Final batch binds final combined frozen source, after iteration/merge decisions:

1.4090 native sm89 vs forced sm80 PTX.
2.H100 native sm90a vs forced sm80 PTX.
3.A100 forcedsm80 PTX, actual automatic fallback, `verify --qualify-gpu`.
4.Fresh source-built wheels/manifests/configuration witnesses, all nine fixtures, no prototype-only qualification. No CPU fallback.
5.72-cell reference repair/gemm-int15 exclusions and final reference/admission checks remain accounted for; multiGPUpar-graph-umap needs2devices. Apple-related final reference work is deferred, not silently borrowed from old mixed-source records.

H100/A100 are NOT rented. Do not rent until artifacts and a concrete final job are ready. Detailed prerequisites/command templates: `~/mojolearn-evidence/identical-blackwell/ptx-readiness.json`; it was a readiness snapshot and its earlier DEFERRED status predates the completed preliminary probe. Tools include `nvidia_baseline_github.py`, `nvidia_baseline_gpu_batch.py`, `nvidia_baseline_qualification.py`, `admit_nvidia_ptx.py`. Review options at the pinned tooling source; current commands may depend on deferred canonical reference columns.

## 7. Recommended next actions, in order

1. Check running linalg builds, collectors/leases, dirty files and final delta note. Do not duplicate any build or restart failed original watchers.
2. Finish the four setup/expectation repairs. Compose the immutable52-library v2 revision from proven binaries; run only failed-cell supplements. Keep original50-step quality receipts intact.
3. Finish DART receipt adaptation, freeze baseline-only temporal margins, adjudicate saved predictions. No DART refitting needed.
4. Run47-case medium NVIDIA/AMD identity once using the corrected, fixed source/artifact/plan inventory. Require same within-arm cross-vendor bits; differentON/OFF bits allowed. Save cross-vendor proof with `identical_wave_compare.py`.
5. Run one controlled ON/OFF speed batch per GPU only after its relevant gates/proof qualify. No opponent runs. Do not compare medium timing directly to full-row stored opponent measurements. Keep/drop per candidate; preserve toggles/comments/results.
6. Cover explicitly opt-in candidates: `tools/identical_candidate_recipes.json` has75recipes/38groups, not all covered by defaultON/OFF. `docs/identical/candidate-priority-queue.json` starts with GMMstack1, KMeansaccumulator256, OCSVMchunk256 and concrete fixture eligibility. OCSVM must actually exceed64iterations; skip timing if unreachable. RFdevice-loop needs unlimited-leaf fixture; ETu16needswide data; EIGHblock stays quarantined. Use build planning first, not75blind builds.
7. Remaining CPU-in-GPU-path work is still substantial:15groups accounted in handoff audit, including tokenizer encode/gather, GARCHreplay, ARIMAAIC, OCSVMgather, IVFfinite, GCNselfloops, bisectcentering, HDBSCANapprox, silhouettecounts, trustworthinessstaging, sharedcasts/reducestats/init/CVgathers and GBDTmulticlassleafsolve. Do not imply this cleanup is complete.
8. Apply the shared callpath only where measurements show remaining fixed-call cost (VAR,knnimputer,randomprojections,additivechi2,maxabs); generic API exists but production adapters are not universal. Grouped waits help multi-call workloads, persistent buffers can help single calls.
9. Integrate gated winners with main, retaining OFF controls. Freeze final combinedsource. Execute exhaustive final verification/PTX matrix before any future PyPI release; publishing itself remains excluded.

The avoidable bottlenecks were incomplete harness/dependency prep and rebuild repetition. Reuse proven artifacts, overlap CPU compilation with independent validation, repair affected cells instead of rerunning everything, and keep source/provenance checks honest. Do not spend more time creating planning documents instead of executing the concrete blocked checks.

## 8. Safe status/resume commands

Read-only first (substitute exact remote if needed):

```sh
cd /Users/andrewhendel/mojolearn-wt/identical-all
git status --short
git log -8 --oneline
python3 -c 'import json; print(json.load(open("/Users/andrewhendel/mojolearn-evidence/identical-all/supervisor-status.json")))'
ssh -p10652 root@64.119.209.250 'ps -eo pid,ppid,etime,pcpu,comm | grep -E "mojo|python" | head -20'
ssh root@162.243.193.137 'ps -eo pid,ppid,etime,pcpu,comm | grep -E "mojo|python" | head -20'
```

Inspect complete JSONs by selected fields/counts, not dumping large comparison reports. Use e.g. `rg -n -i -m 15 -C 2 --max-columns 240 'error|fail|traceback|PASS' LOG`. Never treat filtered PASS lines as proof every step passed.

All pipeline restart/repair commands must use fresh output paths or explicit immutable supplements. Existing failure evidence is valuable; do not delete it. Keep a results harvester alive before launching a job.

## 9. Commit/push checkpoint and final delta

At initial handoff read, HEAD was ad970dfca and origin integration was cb60fa738; temporal-policy/IF/linalg changes were awaiting push. R2index is periodically appended by harvesters and may become dirty again after a clean checkpoint. No other source checkout should be committed by this handoff.


### Final agent checkpoint (supersedes earlier pending-code wording)

- Code checkpoint: **320beb6c0ec51d6fc5dd5940791e4970a620a55c**, preceded by e98f53361, ad970dfca, bdc511fff, 513426f2f. All subagents have completed their handoff turns and have no owned dirty files or newly dispatched long jobs.
- Compile worker: no live owned jobs. Repaired IF/SGD/LU supplemental runs have not started.
- Statistics worker: no live jobs. Calibration/comparison and required provenance adapter remain unstarted.
- Cloud revision worker: linalg builds are now complete on both boxes; retained PIDs above are historical and must not be treated as active without inspection. **The qualified-v2 directory has NOT been created, and the success-receipt reconciler/new stage watcher is still TODO.** Do not mistake committed composer code for completed qualification.
- Root commits the handoff and latest R2index and pushes the integration branch as the last handoff step. Verify `git status` and remote tip when resuming; collectors can append a fresh index row later.

Exact composer starting point, after all linalg receipts PASS and after staging the committed helper plus repaired plan outside source:

```sh
python tools/identical_wave_revision.py \
  --old /root/lq/medium-wave-a4d-fb2 \
  --out /root/lq/medium-wave-a4d-qualified-v2 \
  --old-plan /root/lq/wave-fb2-harness/identical_wave_plan.json \
  --plan /path/to/pinned-repaired-identical_wave_plan.json
```

Do not paste the placeholder path. Use the actual staged committed plan on each box. The composer validates original51 `.so` files (50bindings + portable-math helper), adds two linalg binaries, freezes53-file inventory, and links original prepare/quality hashes. It does not run supplemental gates or declare quality PASS.

### Original handoff merge hotspots still worth tracking

The original six merge-resolution areas were `x_prep/prep3.mojo` (MaxAbs FAST+Apple guard), `x_decomp/resident.mojo` (imports/GRP_FAST_FUSED guard), `python/mojolearn/_expansion_prep.py` (PowerTransformer anchor + blocked stats), the two bindings with registrations from both sides, `svd_full.mojo` mean call, and `bindings/_mojolearn.mojo` imports. Much has compiled/run since, but don't equate ancestry with a complete semantic hunk audit. The source-only lane reports and original briefs are under `~/mojolearn-evidence/fam/`, `briefs-2026-10-04/`, `idn-all/`, and ignored local `docs/lanes/identical-wave-20261004/`; they are not all tracked in Git.

Final handoff written at 2026-10-04T19:57:26.228007+00:00. No new long jobs were launched after the checkpoint request.
