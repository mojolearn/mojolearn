# CURRENT DIRECTIVES: re-read after every merge

Lanes merge origin/main before every merge, so this section reaches every
worktree. The orchestrator changes lane instructions HERE instead of
messaging lanes. Newest items are at the top.

00. **End your session at every checkpoint (saves tokens; Andrew,
   2026-09-27).** The checkpoints are:
   - each phase merged: pass-2 proof, option parity, GPU speed, CPU speed
   - or roughly every 10 merged items
   - or whenever your conversation has grown long
   At a checkpoint: make `docs/lanes/progress/<lane>.md` say exactly where
   you are and what comes next, merge, then STOP with a short report. Don't
   run `dev_pod.sh down`; your pods stay up. The orchestrator relaunches a
   fresh agent for your lane immediately, and it continues from the
   progress file.
0. **AMD boxes are live (main 6c9572e4e):** `tools/dev_pod.sh up <lane> 240 --vendor amd`
   (state key `<lane>-amd`; RunPod MI300X first, Hot Aisle 2x MI300X
   fallback). Grab one when you enter pass 2 and hold it. The lane check now
   builds every base binding itself, and `.checks` takes `<driver>\t<patch>`
   pairs, enforced with `--pass 2`. Before EVERY merge, run
   `tools/test_lane_select.py` AND `python/mojolearn/tests/test_host_surface.py`
   on your pod after merging origin/main; both must pass.
0a. **AMD boxes are allocated for you (2026-09-27).** The orchestrator keeps
   one Hot Aisle MI300X per algorithm lane and renews every dev box hourly.
   If `tools/dev_pod.sh list` shows `<lane>-amd`, that box is yours: use it
   with `--vendor amd` on sync/run/extend. Don't request a second one.
   A `<lane>-amd` box may be a GPU slot on a shared 8x MI300X host
   (`tools/dev_pod.sh host status`): use only sync/run/extend with `--vendor
   amd`, never touch other `/root/mojolearn-*` dirs or GPUs on it. Until a lane
   has its own AMD box, it submits AMD identity checks to the `do-amd` steward:
   `tools/apple_steward.py submit` ships identity requests there too while
   `tools/do_amd_steward.sh` has it up (push the commit to origin first), and
   a merge then needs m2pro PASS AND do-amd PASS (`apple_steward.py status`).
1. **Order per lane:** (a) every algorithm in the lane table and Additions
   (PASS 1); (b) proof on every column, holding an AMD box (PASS 2 items
   1-2); (c) **option parity** (item 2 below); (d) **GPU speed**, IDENTICAL
   and FAST, on NVIDIA, AMD and Apple; (e) **CPU speed, LAST** (Andrew,
   2026-09-27): threads, vectorization and cache blocking of the CPU host
   path, after all GPU work is done. Every change is re-proven bitwise on
   every column.
1b. **Done means ALL of this, per algorithm (Andrew, 2026-09-27):**
   - **Both modes work:** IDENTICAL (bitwise across every column) and FAST
     (a faster schedule, allowed to differ in bits, never in quality: a
     paired check against the reference at 5+ seeds on 2+ datasets).
   - **Both paths work:** the CPU host path and the GPU path, on NVIDIA,
     AMD and Apple.
   - **Its verifier lane is admitted:** out of PENDING, with per-seam
     sabotage that bites and the end-to-end lane check AGREE on every
     column (Apple via the M2 Pro steward, AMD via your slot or the AMD
     steward).
   - **Every opponent option** is present (item 2).
   - **Speed work is done:** GPU first, CPU last (item 1).
   - **Every bug found in existing code is fixed at the root and merged**
     (item 3).
2. **Option parity (Andrew, 2026-09-27).** Every algorithm in the lane's
   family, EXISTING ones included, gets every option its reference and
   bench-board opponents have (sklearn, cuML, LightGBM/XGBoost/CatBoost,
   PyTorch, statsmodels, as applicable):
   - Work through the family's `NOT_IMPLEMENTED.tsv` rows marked NOT
     IMPLEMENTED, and add rows for missing options no row names yet.
   - Options refused for an identity reason (float64 on the device, atomics
     with no fixed-order form) stay refused by name; the tsv says why.
   - Each option gets the same gate as an algorithm: AGREE, a sabotage for
     a numeric change, existing bits unchanged. Merge each option as it
     passes.
3. **Fix what you find at the root.** A shortcoming in shared code or
   another algorithm is fixed and merged to main by the lane that finds
   it. No workarounds, no "documented as owed".
4. **Machines:** hold your pods for your whole session. All data comes
   from R2. No time estimates anywhere.

---

# Algorithm expansion: 57 -> 103 algorithms, 12 -> 13 families

Status: PREP LANDED ON MAIN (2026-09-27, `lane/algos-prep`, commits `aabd9c53`
and `ff4e81c0`). The sections below are kept in the order they were decided;
the LAST section, "Prep landed and reviewed", says what holds now and what is
still owed before the first lane starts. No lane has started.

## Goal

Add 46 algorithms. Each one gets a light bitwise-identity lane in the verifier,
plus a sabotage that proves the lane can fail. Speed is out of scope: no
timing, no bench board, no opponents, no rentals.

The four hard ones wait: IVF-PQ, CAGRA, t-SNE and CNN.

## What a lane proves, and what it does not

- **Proves locally:** the CPU build and the Apple Metal build produce the
  same bits on a small fixture, run once (`--repeats 1`). One source edit
  that moves a reduction order or tie-break makes the lane read DIVERGENT.
- **Does not prove:** identity on NVIDIA or AMD. Those columns come in the
  next release record through the changed-lanes selector (light default).
  The 46 new lanes make that release's NVIDIA and AMD columns bigger. Until a
  release record admits a lane, the paper's count stays 57.

## Resources

- Every lane gets 1 CPU slot and shares the Metal queue. Everything runs
  through `tools/mac_slot.sh`: `run` for CPU work, `metal` for the GPU. Only
  one Metal job runs on the Mac at a time; the rest queue for it.
- Builds use `-j 1` and single-threaded runtimes.
- `MAC_SLOTS` defaults to 5. The Mac has 10 cores and 16 GB. The limit that
  matters is memory during Mojo compiles, not cores, and the slot scheduler
  already holds work back when memory is short.
- This is a deliberate exception to "subagents never run tests locally". It
  covers light identity fixtures only. Nothing large runs, and nothing gets
  timed.

## Lanes (proposal: 5 now, trees 6th)

| lane | algorithms | count | shared machinery reused |
|---|---|---:|---|
| **L1 linear** | SGDClassifier/Regressor, GLMs (Poisson/Gamma/Tweedie), Huber, BayesianRidge/ARD, LinearSVC/SVR, LARS/LassoLars, quantile regression | 7 | QN/CD solvers (`glm/`, `solver/`), parked `glm-qn-losses` patch, Cholesky rank-1 update |
| **L2 cluster+neighbors+kernel** | MiniBatchKMeans, BisectingKMeans, MeanShift, OPTICS, AffinityPropagation, LocalOutlierFactor, OneClassSVM, KernelPCA | 8 | KMeans, KDE, DBSCAN/HDBSCAN core distance, kNN, SVM solver, kernel matrices + eigh |
| **L3 decomposition+linalg** | IncrementalPCA, Gaussian/Sparse random projection, SpectralEmbedding, NMF, FastICA, FactorAnalysis, LU solve, lstsq/randomized SVD | 8 | PCA/TSVD, GEMM, QR/eigh/SVD, spectral clustering's embedding |
| **L4 preprocessing + NEW family "naive Bayes & discriminant analysis"** | RobustScaler, MaxAbsScaler, OneHot/Ordinal encoder, TargetEncoder, SimpleImputer, KBinsDiscretizer; GaussianNB, MultinomialNB, BernoulliNB, LDA, QDA; NearestCentroid | 12 | scalers, GBDT quantile binning + CTR, covariance/eigh, metrics reductions |
| **L5 sequence (neural + time series)** | LSTM, GRU, RMSprop, Adagrad; AutoARIMA order search, STL, VAR | 7 | MLP/training optimizers, Mamba scan, ARIMA + KPSS (`tsa/`), AIC/BIC rows already in `arima/NOT_IMPLEMENTED.tsv` |
| **L6 trees** (starts when a slot frees, or once `lane/apple-fast-trees` is quiet) | DecisionTree (CART) classifier/regressor, Bagging, AdaBoost, DART | 4 | RF/ET tree builder, GBDT boosting loop |
| | | **46** | |

Why trees goes last: it touches the same tree builder and the same `_trees`,
`_rf` and `_gbdt` bindings as the FAST trees lane. If both lanes rebuild
those bindings, one overwrites the other's `.so` mid-run. The rule to rebuild
every binding that imports a changed shared module makes the collision
likely.

Neural is not split in two. Without CNN it is four items, and RMSprop and
Adagrad are small. It pairs naturally with time series: both are recurrent
scans. A second neural lane makes sense when CNN is taken up.

## GPU benefit, honestly

- **Clear benefit (GEMM, kNN or tree bound at 1M+):** trees and ensembles,
  SGD, GLMs, Huber, LinearSVC, OneClassSVM, KernelPCA, MeanShift,
  AffinityPropagation, LOF, MiniBatch/Bisecting KMeans, NMF, ICA,
  FactorAnalysis, IncrementalPCA, random projection, spectral embedding,
  LU, lstsq, LSTM, GRU.
- **Benefit only batched (many series at once, which is cuML's pitch):**
  AutoARIMA, STL, VAR.
- **Weak on its own; the value is staying on the GPU inside a pipeline:**
  encoders, imputer, scalers, naive Bayes, LDA/QDA, NearestCentroid. Fits
  are single reductions.
- **Sequential at the core; the GPU only helps the inner step:** OPTICS
  (ordering loop), LARS (one feature per step), AdaBoost (one tree at a time,
  though each tree is GPU work).

## Per-algorithm definition of done (every lane brief carries this)

1. **Reference named in the file.** Name the sklearn or cuML file and line
   at the pinned commit. Every option not carried gets a row in that
   module's `NOT_IMPLEMENTED.tsv`, or is refused by name at the API.
2. **One Mojo source for CPU and GPU.** No vendor branches. IDENTICAL is the
   default. FAST is out of scope. No float64 on the device, and no atomics
   in a reduction whose order could change the result.
3. **Public Python class** with the sklearn-shaped API
   (`fit`/`predict`/`transform`), exported from `mojolearn/__init__.py`.
4. **Correctness sanity:** agrees with the reference library within a
   tolerance on tiny data. This is a check, not an identity claim.
5. **Verifier lane:** a small fixture registered in
   `_verification_catalog.py`, and its paths attributed in
   `tools/lane_select.py`, so the selector never reads the lane as
   UNATTRIBUTED. The fixture uses non-uniform data and exercises ties, so it
   is not blind. The CPU hash must equal the Metal hash, run once.
6. **Sabotage:** a SOURCE edit, never a `-D` define (define-only arms reuse
   cached kernels), that changes a reduction order or tie-break. Show the
   lane goes DIVERGENT, then restore it with the reverse patch, never
   `git checkout`. Also perturb one hash value and show the comparator
   fails. Record both as evidence outside the repo tree.
7. **Rebuild every binding** that imports any shared module you touched.
8. **Stop and ask; never widen.** Anything that needs a shared-core change
   (GEMM, reductions, RNG, serialization) or float64, or anything that
   cannot be made identical, gets reported, not patched around.

## Git discipline (in every brief)

- Each lane gets its own worktree from `origin/main` on branch
  `lane/algos-<name>`.
- Never `git stash`. Never `git add -A`. Check `HEAD` before every commit.
  Touch only your own modules.
- Four shared files conflict across lanes: `__init__.py`,
  `_verification_catalog.py`, the `lane_select` attribution and the bindings
  list. Lanes commit on their branch; the orchestrator merges into main one
  lane at a time and pushes in the same command.
- One algorithm per commit, merged when its lane and sabotage pass. Don't
  batch all eight at the end.

## Open questions for Andrew

1. Run 5 concurrent lanes with trees 6th, or 6 lanes at once (above
   `MAC_SLOTS` = 5)?
2. Keep the weak-GPU items (encoders, imputer, OPTICS, LARS, naive Bayes),
   or cut them and land at about 97?
3. Should lanes merge to main themselves as each algorithm passes, or go
   through me?
4. Does a new lane wait for the next release's NVIDIA and AMD columns before
   it counts, or do we run one small `--par quick`-style cross-vendor leg
   when all six are done?

---

# Option B: one rented NVIDIA pod per lane, 12 lanes, hard cases included

## What changes

- **Where the work runs.** Each lane's agent still edits in a Mac worktree.
  It builds, runs, and runs sabotage on its own NVIDIA pod: the x86 CPU
  column plus the CUDA column. The Mac stops being the build machine, and
  local load drops to 12 mostly idle agents plus the Apple queue.
- **What a lane proves on its pod:** x86 CPU = NVIDIA, bitwise. That alone
  is already a cross-architecture check.
- **Reference libraries are on the box.** cuML, cuVS, scikit-learn and
  PyTorch run next to our code, so correctness checks against the reference
  are immediate. This matters most for CAGRA, IVF-PQ, t-SNE, CNN and LSTM.
- **Pods stay up for the lane's whole life.** Arm each pod with
  `tools/runpod_guard.sh arm <pod> <ssh> 120`. The lane's agent runs
  `extend` every hour as a heartbeat. If the agent or the laptop dies, the
  on-pod watchdog still kills the pod when the lease runs out. Keeping the
  pod up also keeps the pixi env and the warm Mojo cache.
- **Apple is still the bottleneck.** Metal bindings compile only on the Mac,
  one Metal job at a time. So the Apple column becomes a serialized
  **Apple steward** queue. When an algorithm is green on its pod, the
  steward builds it on the Mac and runs the light CPU(Arm) = Metal check
  plus the sabotage. Proposal: give the steward 2 to 3 cores for compiles,
  as an exception to the 1-core rule. The 12 lanes never touch Metal
  themselves.
- **AMD:** still at the release, through the changed-lanes selector.
  Optional: one shared single-MI300X box that runs a daily AMD sweep of the
  new lanes. That catches gfx942 surprises earlier.

## 12 lanes

| # | lane | algorithms | # |
|---|---|---|---:|
| 1 | linear | SGD, GLMs, Huber, BayesianRidge/ARD, LinearSVC/SVR, LARS, quantile regression | 7 |
| 2 | clustering | MiniBatch/Bisecting KMeans, MeanShift, OPTICS, AffinityPropagation | 5 |
| 3 | neighbors + kernel | LOF, NearestCentroid, OneClassSVM, KernelPCA | 4 |
| 4 | decomposition | IncrementalPCA, random projection, spectral embedding, NMF, FastICA, FactorAnalysis | 6 |
| 5 | linalg | LU solve, least squares / randomized SVD | 2 |
| 6 | preprocessing | RobustScaler, MaxAbsScaler, OneHot/Ordinal encoder, TargetEncoder, SimpleImputer, KBinsDiscretizer | 6 |
| 7 | naive Bayes & DA (new family) | Gaussian/Multinomial/Bernoulli NB, LDA, QDA | 5 |
| 8 | time series | AutoARIMA order search, STL, VAR | 3 |
| 9 | trees | CART, Bagging, AdaBoost, DART | 4 |
| 10 | neural: sequence | LSTM, GRU, RMSprop, Adagrad | 4 |
| 11 | neural: CNN (HARD) | conv1d/conv2d + pooling via im2col onto our GEMM | 1 |
| 12 | HARD: ANN search + t-SNE | IVF-PQ, CAGRA, t-SNE | 3 |

That's 50, taking us from 57 to 107 across 13 families. Lanes 5, 7 and 8 are
small; if you want fewer pods, pair them with 4, 6 and 10.

On 12: t-SNE and CAGRA are research-shaped. Their references use atomics
and a build order that is not deterministic, so each needs its own design
note before any code gets written. It is fine for lane 12 to end with a
design and a refusal rather than a shipped algorithm.

## Before the fan-out: one prep step

With 12 lanes, the four shared registries become merge-conflict hotspots:
`__init__.py` exports, `_verification_catalog.py`, the `lane_select`
attribution and the bindings list. Make each of them pick up per-module
fragments automatically. Each lane then touches only its own directory,
and merges stop conflicting.

## Speed work (phase 2, on the same pod)

- **Phase 2 opens only after a lane's algorithm is identity-admitted.**
  That way a slower-but-correct first version is locked in before anyone
  optimizes it.
- **IDENTICAL speed on NVIDIA: yes.** It fits the rules: judged at 1M+
  rows, compared against the opponent's fast mode, reported through
  `tools/bench_board.py`. Every speed commit must re-pass the lane on the
  pod AND go back through the Apple steward. It is one shared source, so an
  NVIDIA speedup can move Apple bits or slow Metal down.
- **FAST mode on NVIDIA: needs your call.** The Sep 24-25 rule says FAST is
  Apple-tailored, trees + classical, and never neural. FAST work on NVIDIA
  or on neural would reverse that. Any FAST work also carries the
  quality rule: a paired check against the reference, at least 5 seeds and
  at least 2 datasets.
- Speed is timed on the lane's 4090 while developing. Claims are made only
  from a bench_board run on the record GPU models.

## Cost (to confirm against live prices before renting)

- 12 x RTX 4090 at about $0.35-0.70/hr each comes to about $100-200/day.
- Swapping lanes 10-12 to H100 (for CNN/LSTM speed versus PyTorch) adds
  about $50/day.
- An optional shared MI300X is about $2.4/hr, or about $60/day.
- Mac disk: 12 worktrees at 3-5 GB is about 60 GB; 107 GB is free.

## Open questions (Option B)

1. 12 pods, or 9 (pair up the small lanes 5, 7 and 8)?
2. Give the Apple steward 2 to 3 Mac cores for Metal compiles?
3. FAST on NVIDIA and on neural: reverse the Apple-only rule, or keep
   phase 2 to IDENTICAL only?
4. A shared AMD box for daily sweeps, or AMD only at the release?
5. Budget cap and duration?

---

# Option B, revised with Andrew's answers (2026-09-27)

Decided:
- 9 NVIDIA pods, one per lane.
- FAST AND IDENTICAL, both, on every lane, including neural.
- AMD only at the release.
- No budget cap.
- Pods are torn down when their lane is merged.

## 9 lanes

| # | lane | algorithms | # |
|---|---|---|---:|
| 1 | linear | SGD, GLMs, Huber, BayesianRidge/ARD, LinearSVC/SVR, LARS, quantile regression | 7 |
| 2 | clustering | MiniBatch/Bisecting KMeans, MeanShift, OPTICS, AffinityPropagation | 5 |
| 3 | neighbors + kernel | LOF, NearestCentroid, OneClassSVM, KernelPCA | 4 |
| 4 | decomposition + linalg | IncrementalPCA, random projection, spectral embedding, NMF, FastICA, FactorAnalysis, LU solve, least squares / randomized SVD | 8 |
| 5 | preprocessing + NB/DA (new family) | 6 preprocessors; Gaussian/Multinomial/Bernoulli NB, LDA, QDA | 11 |
| 6 | sequence | LSTM, GRU, RMSprop, Adagrad; AutoARIMA order search, STL, VAR | 7 |
| 7 | trees | CART, Bagging, AdaBoost, DART | 4 |
| 8 | CNN (HARD) | conv1d/conv2d + pooling via im2col onto our GEMM | 1 |
| 9 | ANN + t-SNE (HARD) | IVF-PQ, CAGRA, t-SNE | 3 |

## Where compute happens

- **Pods (9 x NVIDIA):** all building, the CPU = NVIDIA identity check,
  sabotage, correctness against the reference, and FAST + IDENTICAL speed
  work at 1M+ rows.
- **The laptop:** 9 agent processes that edit, push and ssh. They compile
  nothing locally. Plus ONE Apple steward: 1 CPU slot and the one Metal
  queue.
- **Cloud Mac (proposed, bigger than the Air):** a second Apple steward
  with its own Metal queue. It does the Apple identity check for half of
  the lanes, plus FAST speed work on Apple. It is the same box the GPT-3
  small Apple segment already needs.

## Order of work

- Identity first: each algorithm lands with CPU = NVIDIA plus sabotage on
  its pod, then goes to the Apple stewards.
- Speed after identity, per algorithm: FAST + IDENTICAL at 1M+, and every
  FAST change passes the quality rule (paired check against the reference,
  at least 5 seeds, at least 2 datasets).
- Each binding module gets one Metal build per request, not one per
  algorithm.
- For lanes 8 and 9, a design note plus a named refusal is an acceptable
  outcome where identity cannot be had.

## Pod lifecycle

`arm <pod> <ssh> 240` at start. Extend only while the lane is actively
running. Reap as soon as the lane's last branch is merged. No idle pods.

---

# FINAL DECISIONS (Andrew, 2026-09-27): these override the sections above

- **Apple stewards are two AWS EC2 Macs:** an M2 Pro and an M3 Ultra, in
  the mambik account, us-east-1d. Together with the laptop's M4 that is
  three Apple GPU generations.
  - Every Apple request must PASS on BOTH cloud Macs.
  - Both have a 24-hour minimum. `tools/cloudmac.sh stop-all` terminates
    the instances; the hosts can be released after 24 h.
- **The MacBook is off-limits to subagents** (CPU and GPU). The orchestrator
  may use 2 laptop cores for managing and double-checking.
- **Every lane works in its own worktree and branch, pushes it, and merges
  to main itself** when an algorithm is green on its pod AND on both Apple
  stewards. The merge is: merge `origin/main` in, rerun the lane check, then
  `git push origin HEAD:main`, fast-forward only.
- **CPU is part of every proof.** The CPU host binding must be built and
  compared on the x86 pod and on both Arm cloud Macs. Nothing compared, or
  a missing host `.so`, counts as a failure.
- **`dev_pod.sh up` retries** while RunPod is out of stock. A lane's
  `state.env` appearing is its report to the orchestrator that it got an
  NVIDIA pod.
- **FAST AND IDENTICAL** speed work on every lane, including neural.
- **AMD only at the release.**
- **Local cores (Andrew, 2026-09-27 later):** a subagent MAY use 1 laptop
  CPU core when it truly needs one, only through `tools/mac_slot.sh run`
  (5 slots, memory-admitted, so simultaneous needs queue), and should avoid
  it. The laptop GPU/Metal is never used by subagents.
- **Lane 10: bench (no measuring yet).** It adds every new algorithm to
  `tools/bench_board.py` with its opponents, datasets and quality metric, so
  one board run covers them later. Nothing gets timed until Andrew says so:
  speed work across all lanes comes first. Brief:
  [ALGORITHM_EXPANSION_BENCH_BRIEF.md](ALGORITHM_EXPANSION_BENCH_BRIEF.md).

---

# Prep landed and reviewed (2026-09-27, after the merge to main)

`lane/algos-prep` is on main. Confirmed by ancestry, not by reading the
branch: `aabd9c53` and `ff4e81c0` are ancestors of `origin/main`, the branch
tip is too, and the three docs and five tools it added are byte-identical
on main to what was reviewed. `packaging/check_ext_lists.py` passes on main
with every fragment empty, so every packaging list is what it was before
the prep. The fragment loader was exercised by hand and refuses an import,
another lane's binding and a lane name that already exists.

## What the review found, and who fixes it

Owner is the orchestrator unless a lane is named. Nothing here is a lane's
to widen into.

| # | finding | what to do | status |
|---|---|---|---|
| R1 | **A submitted commit never reaches the cloud Macs.** `apple_steward.py submit` ships the request and patch over ssh, then the Mac's `process` runs `git fetch origin` and checks out the sha. But a cloud Mac's `origin` is the bare repo the laptop pushes to (`cloudmac.sh bootstrap`), and neither `submit`, `flush-deferred`, nor the brief's step 7 pushes the sha there. Every request would fail at "checkout of <sha>". | Until the tool does it, the lane runs `tools/cloudmac.sh push m2pro <sha>` before `submit` (brief step 7, revised). The tool fix: `submit` pushes to each non-deferred Mac and `flush-deferred` pushes before it ships; or `process` fetches the sha from GitHub over https (the repo is public). | **DONE 2026-09-27 (lane/algos-tools): `submit` and `flush-deferred` push the sha (`cloudmac.sh push`), `process` fetches `refs/steward/*`.** |
| R2 | **Seam checks are optional in the gate.** `algos_lane_check.py` prints a note and continues when a fragment has no `tools/identity_lanes/<lane>.checks`. A lane could register identity lanes, pass the GPU == CPU diff and never run an oracle: the "light" hole the brief closes in prose only. | Make a missing `.checks` a FAIL once the fragment registers any lane. Until then the orchestrator refuses to merge a lane whose fragment registers lanes and has no `.checks` listing (brief step 2, revised). | **DONE 2026-09-27: `algos_lane_check.sh --pass 2` fails a fragment with no `.checks`; the steward runs `--pass 2` by default.** |
| R3 | **Per-seam sabotage arms are unverified by any tool.** The steward runs the one end-to-end `--sabotage` patch; the per-seam arms the brief requires live only in the lane's evidence directory. | Extend the `.checks` line format to `<driver>\t<sabotage patch>` and have the lane check run each pair (must FAIL under the patch, PASS after `git apply -R`). Until then, the lane's report at each commit names every seam patch and its result, and the orchestrator reads them. | **DONE 2026-09-27: `.checks` lines are `<driver>` or `<driver><TAB><patch>`; the check runs each driver (PASS), each patch (FAIL, then PASS after `git apply -R`); `--pass 2` requires a patch on every line.** |
| R4 | **main was rewritten today.** The fetch that preceded the review showed a forced update on `origin/main` (`3cf7ae22...b35580c3`). Nine lanes pushing `HEAD:main` fast-forward cannot survive another one. | Turn on force-push protection for `main` for the duration of the fan-out. | **OWED: Andrew, in the GitHub settings.** |
| R5 | **The proof dummy ran on one vendor.** The A40 pod proved x86 CPU == NVIDIA through the whole loop. Nothing has run `algos_lane_check.sh` on a Mac with Metal, and that is the path every steward request takes. | Re-create the `x-prep-dummy` lane on a throwaway branch (its two commits are on main; `git revert ff4e81c0` on the branch), push it to the M2 Pro, and run it through `apple_steward.py work --once` before any lane submits. Then delete the branch. | **OWED: orchestrator, before the first submit.** |
| R6 | **Pod bootstrap is manual.** The brief tells each lane to `git init` and fetch on the pod so `git apply` works; `sync` does not check that the pod tree is at the worktree's base commit, so a patch that applies on the laptop can fail on the pod for a stale-tree reason. | `dev_pod.sh up` seeds the git tree at the lane's base sha itself; `sync` refuses when the pod's HEAD is not the worktree's merge base with origin/main. | **DONE 2026-09-27: `up` seeds the tree at origin/main; `sync` moves HEAD and the index to the worktree's merge base and refuses otherwise.** |
| R7 | The prep shipped `sequence` and `cnn` identical only (`host_surface.EXPANSION_IDENTICAL_ONLY`), against FINAL DECISIONS (FAST and IDENTICAL everywhere). | The tuple is now empty: all nine expansion bindings build FAST and IDENTICAL and join `_CLASSICAL_FAST`; the briefs say so. Separate, not a lane's: a FAST tier for the four existing neural bindings (`build_sets.sh` IDENTICAL_ONLY lists, `_backend` tier table, their build scripts). | **DONE 2026-09-27 for the nine lanes. The existing neural bindings' FAST tier is OWED to the orchestrator, off the lanes' path.** |
| R8 | A fragment may bind a key twice; the second `FAMILIES = ...` silently wins (`host_surface._read_expansion_fragment`). | Refuse a key bound twice. | **DONE 2026-09-27: refused.** |
| R9 | A dirty steward worktree (an unreversed sabotage) fails the request that finds it, and every later one, until someone logs in. | After a reversal fails, `process` should `git checkout -- .` the worktree, log that it did, and go on. Judgment call; the failing request still reads FAIL. | **OWED: tool fix, low priority.** |

## What holds, restated

- Every shared registry reads per-lane files and the loaders enforce
  ownership by name, so nine lanes never edit the same line. The refusals
  were exercised, not just read.
- The lane check refuses NOTHING COMPARED, a missing host binding, a
  define-only sabotage, and a patch that touches nothing the lanes run.
- DEVIATION 5000-5899 and IDENTITY_PATHS rows 100-189 are pre-allocated
  per lane, with one ledger section per lane.
- The pod dead-man and lease reuse `runpod_guard.sh`; the steward queue
  is atomic per request; the bench lane measures nothing.

## Running the lanes

- Pod leases: `dev_pod.sh up <lane> 240`, extended hourly while working. A
  pod is torn down when its lane's last algorithm is merged.
- The paper's count moves only when a release record admits a lane on
  three vendors.
- Speed work (FAST and IDENTICAL) starts per lane after that lane's
  identity passes, never before. FAST commits from all lanes go back through
  the Metal queue, so sequence them.

## Order of operations before the first lane starts

1. R4: protect main.
2. R1 and R2 tool fixes, or at least the brief's workarounds (done in the
   briefs).
3. R5: the dummy through the M2 Pro steward on Metal.
4. Bootstrap the nine pods (`dev_pod.sh up`) and the bench pod.
5. Start the nine lanes with the COMMON BRIEF and their sections.

---

# Everything else: the long tail, assigned (2026-09-27)

Andrew: "may as well do everything now". What follows is every standard
estimator not in the nine lane tables, checked against the public API on
main (127 names) and the `NOT_IMPLEMENTED.tsv` files. Each is assigned to the
lane whose machinery it reuses, in the briefs as an "Additions" table a lane
does after its main table. E = a thin layer with few seams,
M = a new kernel on a known pattern, H = new machinery. The per-seam bill
applies to every one; a wrapper with no numeric seam of its own owes only
the verifier lane, the Python class, the sanity check and the CPU route.

| lane | additions | est. |
|---|---|---|
| linear | Perceptron, PassiveAggressiveClassifier/Regressor, RidgeClassifier, SGDOneClassSVM (all SGD variants: E); RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV (over `cross_val_score`: E); IsotonicRegression (parallel PAVA by prefix scan: M) | +9 |
| cluster | BayesianGaussianMixture (`mixture/` machinery; already a row in its tsv: M) | +1 |
| neighbors + kernel | PolynomialCountSketch, AdditiveChi2Sampler, SkewedChi2Sampler (kernel approximation, `kernel_methods/` tsv rows: E); LabelPropagation, LabelSpreading (kNN graph + fixed-order iteration: E); KNNImputer (E) | +6 |
| decomp + linalg | CCA, PLSRegression (SVD of the cross-covariance: E); SparsePCA, MiniBatchSparsePCA, DictionaryLearning (CD + GEMM: M); LatentDirichletAllocation (variational EM, fixed order: M); Isomap, MDS, LocallyLinearEmbedding (kNN + eigh: M); EllipticEnvelope / MinCovDet (M); ALS matrix factorization for recommendation (GEMM + batched least squares, the clear GPU win in this row: M) | +12 |
| prep + NB/DA | QuantileTransformer, PowerTransformer, Normalizer, PolynomialFeatures, SplineTransformer, Binarizer, LabelEncoder, LabelBinarizer, MultiLabelBinarizer (E); IterativeImputer (M); VarianceThreshold, SelectKBest with f_classif / chi2 / f_regression / mutual_info, RFE as a wrapper (E); ComplementNB, CategoricalNB (E) | +17 |
| sequence | MLPClassifier / MLPRegressor, sklearn-shaped over `SmallMLPTrainer` (E); vanilla RNN after LSTM (E); Lion, Adafactor, LAMB, Adamax, NAdam (E); LR schedulers: step, exponential, one-cycle (E); LayerNorm beside RMSNorm (E); Theta and Croston forecasters, damped-trend ETS (E); GARCH (M) | +15 |
| trees | RandomTreesEmbedding (E); VotingClassifier/Regressor, StackingClassifier/Regressor, MultiOutput, OneVsRest, CalibratedClassifierCV (wrappers: E); SHAP explainers over our forests and GBDT, TreeExplainer first, then KernelExplainer and PermutationExplainer (cuML has the last two; big GPU workloads: M, then H) | +10 |
| cnn | after Conv: BatchNorm, Dropout2d, global pooling, a ResNet block (E once conv exists) | +4 |
| ann | after IVF-PQ: IVF-SQ and IVF-RaBitQ (quantization arms on the same index: E), the refine step, the sample filter (both rows in `ivf/NOT_IMPLEMENTED.tsv`: E) | +4 |
| | | **+78** |

With the fifty in the lane tables that is about 128 additions on top of 57,
and the reconsidered groups below take it to about 193.

## The rest, reconsidered (Andrew, 2026-09-27: "knock out things now")

The first draft of this section left six groups out. Four of them are
ordinary GPU work with a nameable reference and are now assigned; one is
a block, not an estimator, and is assigned as a block; two stay out, for
reasons that are about the algorithm and not about effort.

| group | verdict | where | est. |
|---|---|---|---|
| graph: PageRank, connected components, Louvain | **IN.** PageRank is a pinned-fold GEMV iteration. Connected components is DBSCAN's `weak_cc` as a product. Louvain's reference is order-dependent in parallel form, so ours pins the vertex sweep order and breaks community ties by lowest id, the same move as IDENTITY_PATHS row 15. Reference: cuGraph `cpp/src/link_analysis/pagerank_impl.cuh`, `cpp/src/components/weakly_connected_components_impl.cuh`, `cpp/src/community/louvain_impl.cuh`; networkx as the sequential oracle. | neighbors + kernel (it owns the kNN graph) | +3, E E M |
| GNN layers: GCN, GraphSAGE | **IN, as layers.** Each is an SpMM over a CSR adjacency in fixed row order plus a GEMM, both of which the tree already pins. Reference: PyG `torch_geometric/nn/conv/gcn_conv.py`, `sage_conv.py`. | cnn (after Conv; same im2col-to-GEMM shape of work) | +2, M |
| mixture-of-experts block | **IN, as a block.** Top-k routing with an index tie-break plus expert GEMMs; the routing tie is the seam. Reference: HF `modeling_mixtral.py::MixtralSparseMoeBlock`. | sequence (it owns the neural additions) | +1, M |
| Prophet-style forecaster | **IN.** Piecewise-linear trend with changepoints, Fourier seasonality, holiday regressors, MAP fit by L-BFGS. Parity with the `prophet` package is at a tolerance only (their fit is Stan); identity is ours. Reference: `prophet/forecaster.py` and `stan/prophet.stan` for the model. | sequence (after STL and VAR) | +1, M |
| sparse variational GP (SVGP) | **IN.** Inducing points, a variational posterior, GEMM and Cholesky bound: a clear GPU win. Upgrades the named refusal in `gaussian_process/NOT_IMPLEMENTED.tsv` the way row 12 upgraded RF's log criteria. Reference: GPflow `gpflow/models/svgp.py`. | neighbors + kernel (GP kernels live beside `kernel_methods/`) | +1, M/H |
| HNSW | **IN, as CAGRA's CPU-serving form (Andrew, 2026-09-27).** It is a CPU algorithm by construction (a hierarchical graph walked one hop at a time), and that is exactly how the field uses it: build the graph on the GPU, serve queries on CPU boxes. cuVS's own entry is `hnsw::from_cagra`. So it is the second half of CAGRA, not a competitor to it. The "no CPU path" rule the IVF lane cited did not exist in CONTRIBUTING; CONTRIBUTING now says when a CPU-only algorithm may enter ("CPU-only algorithms"), and the refusal in `ivf_refuse_algorithm` and `ivf/NOT_IMPLEMENTED.tsv` is corrected to NOT IMPLEMENTED, assigned. Identity: a search over a fixed graph is deterministic given the graph and an index tie-break, and the same across every CPU host. Reference: cuVS `cpp/src/neighbors/hnsw.cpp`, hnswlib `hnswalg.h` for the layout and search. | ann, after CAGRA | +1, M |
| Birch | **IN (Andrew, 2026-09-27), as a CPU fit that feeds a GPU step.** The CF-tree build inserts points one at a time and depends on insertion order by definition, so it is a CPU pass with no parallel form. Its global clustering step over the subcluster centroids (scikit-learn's default is AgglomerativeClustering) runs on this library's GPU agglomerative clustering, which is the pairing CONTRIBUTING's "CPU-only algorithms" paragraph asks for. Identity: sequential, deterministic given input order, every seam a fixed-order compare or a squared-norm update. Support-matrix row says CPU for the fit. Reference: scikit-learn `cluster/_birch.py`. | cluster | +1, E |

So the long tail is +88, not +78, and the target is about 195 on top of 57
if every lane finishes both of its tables.

---

# PASS 1 — CODE FIRST, TODAY (Andrew, 2026-09-27): overrides the gates above

All ten lanes start now. For every algorithm, pass 1's gate is only:

1. It builds on the lane's pod.
2. The sklearn/reference sanity check agrees within a tolerance.
3. `tools/algos_lane_check.sh <lanes>` reads **AGREE**: CPU == NVIDIA,
   bitwise, NOTHING COMPARED counts as a failure.

Then it merges at once as a PENDING lane (never counted in the paper).

**Pass 2, after the code exists, is a sweep:** per-seam host oracles,
separating fixtures, per-seam sabotage, DEVIATION numbers, card stages,
IDENTITY_PATHS rows, the Apple steward checks (M2 Pro, then the M3 Ultra),
and then speed. Lanes do NOT submit to the Apple steward in pass 1.

Order inside a lane: main table first, then Additions. One commit per
algorithm. Progress file: `docs/lanes/progress/<lane>.md`, updated at each
merge, so a fresh agent continues where the last one stopped.

---

# PASS 2 — EVERYTHING AT ONCE, per lane, as soon as its pass-1 list is merged (Andrew, 2026-09-27)

Each lane holds its machines for its whole session; set-up and teardown are
the cost being avoided. All data comes from R2 (`tools/dataset_store.sh stage`).

1. **Machines.** Keep the NVIDIA pod. Add an AMD box with
   `tools/dev_pod.sh up <lane> 240 --vendor amd` (RunPod MI300X, else Hot
   Aisle; it retries while there is no stock) and hold it. If the tool
   doesn't have `--vendor amd` yet, keep working on NVIDIA and CPU and try
   again after the next algorithm. Apple: identity through the M2 Pro
   steward; Apple timing through `apple_steward.py submit --kind speed`,
   routed to the M3 Ultra once its GPT-3 segment ends.
2. **Proof, per algorithm** (the COMMON BRIEF's per-seam discipline):
   - a host oracle and a separating fixture per seam
   - a sabotage arm per seam, in `.checks`, that bites
   - a DEVIATION number and a card stage from the lane's ranges
   - an IDENTITY_PATHS row
   - the verifier lane with CPU and GPU paths
   - `algos_lane_check.sh` AGREE on NVIDIA **and** on AMD
   - an M2 Pro steward PASS
   The lane then moves out of PENDING, per the verifier's admission rules.
3. **Speed, after an algorithm's proof passes:**
   - IDENTICAL and FAST on NVIDIA, AMD and Apple, plus the CPU path
     (threads, vectorization), at 1M+ rows or the family's realistic
     large shape, on R2 data.
   - Every IDENTICAL speed change re-passes step 2 on every column.
   - Every FAST change passes the quality rule (paired check against the
     reference, at least 5 seeds, at least 2 datasets).
   - Before/after on the same box. No opponent claims; those go through
     the bench board later.
4. Merge each step as it passes, the same way as pass 1. Keep the progress
   file current.

**Rule (Andrew, 2026-09-27): a lane that finds a shortcoming in shared code
or in another algorithm fixes it itself.** It may edit those files for that
fix, with the same gate as its own work: AGREE, a sabotage for any numeric
change, and existing lanes' bits unchanged (verify them against the
reference before and after). Merge origin/main often. Never rebuild a
shared `.so` that another job is using. Tooling gaps go to the tools lane.

## Pass-2 tooling (lane/algos-tools, 2026-09-27)

- **AMD dev boxes:** `tools/dev_pod.sh up <lane> [minutes] --vendor amd`
  (key `<lane>-amd`; `sync`/`run`/`extend`/`down <lane> --vendor amd` or
  `<lane>-amd`). RunPod MI300X, then a Hot Aisle MI300X VM. On AMD compare
  numbers, never `.so` digests (cold-cache gfx942 codegen varies; see the
  script's header).
- **Per-seam proof:** `tools/identity_lanes/<lane>.checks`, one
  `<driver><TAB><sabotage patch>` per seam; `algos_lane_check.sh <lanes> --pass 2`.
- **Apple speed jobs:** `apple_steward.py submit --kind speed --lane <l>
  --commit <sha> --builds bindings/build_x.sh[,...] --cmd '<timing>' [--mode
  fast|identical]`, M3 Ultra only, spooled while it is deferred.
- **Stewards as daemons:** `tools/cloudmac.sh steward <mac> install|restart|status`.

---

# Phase 3: CPU speed (Andrew, 2026-09-27)

The library is already CPU and GPU: every algorithm ships a CPU host
binding with the GPU's bits, CPU-only installs train and predict, and the
bench board has an `ours-cpu` arm. What it is not is FAST on a CPU. The host
kernels are the same Mojo source compiled for the CPU, written to prove
identity; a few host oracles use the parallel primitives and most run one
core. Phase 3 makes the CPU path fast, without moving a bit.

## Why

- **Train on the GPU, serve on the CPU, no drift.** The same-bits contract
  is what makes it safe to fit on one box and run on another. A fast CPU
  inference path for the forests, GBDT, kNN, the neural blocks and HNSW
  (lane 9) is the serving half of that story for every algorithm.
- Development and CI without a GPU; edge boxes.
- Not a reason to train on a CPU when a GPU is there. Training speed on
  the CPU is second.

## The rule

The same as GPU speed work. A multithreaded fold reorders sums unless it
is pinned, so every thread split is a PIN (partial count a function of
the shape, never of the core count: IDENTITY_PATHS row 7) and every
partial fold is `pinned_block_sum`'s host twin. SIMD width is a PIN too
(a 4-wide and an 8-wide fold are two summation orders). No CPU-specific
numerics: `ftz`, `identical_mul_add`, the portable transcendentals and the
composite-key selectors are the same code on every column. Every CPU
speed commit re-runs the identity check (GPU == CPU, bit for bit) and the
sabotage, exactly as step 5 and 6 of the COMMON BRIEF.

## Scope and order

1. **Inference first**, per family: forests and GBDT prediction, kNN and
   the IVF/CAGRA/HNSW search, the neural blocks' forward pass, the
   transformers' `transform`. Judged on the bench board's `ours-cpu` arm
   against scikit-learn, LightGBM, XGBoost, FAISS-CPU, hnswlib and PyTorch
   CPU on all cores, 1M+ rows or the family's realistic shape.
2. **Training second**, where the CPU case is real: GBDT and forests
   (LightGBM and XGBoost on all cores are the opponents), linear models,
   k-means, the preprocessors.
3. What stays one core: anything sequential by construction (Birch's tree
   build, OPTICS's ordering loop, LARS's steps).

## Lanes and machines

One lane per family, after that family's identity and GPU speed work is
merged; never in the same window as the GPU speed wave, which already
sends every commit through one Metal queue. CPU lanes need no GPU: a
many-core CPU pod (RunPod CPU instances or an EC2 c7i/c8g) is enough for
the x86 column, and the Arm column is the cloud Macs' CPUs through the
same steward. Each lane's brief is the COMMON BRIEF with step 9 reading
"CPU" and the opponents above.

## Not in this fan-out

Phase 3 starts when Andrew says so, after the identity wave and the GPU
speed wave. It is written here so the target is on the record and so no
lane in phases 1 and 2 designs a host kernel that cannot be parallelized
later (keep the fold shape explicit; never bake a sequential order into a
seam that a pinned tree would also satisfy).

---

# CONSOLIDATION PASS (Andrew, 2026-09-27)

Measured: the numeric seams are NOT duplicated. Every identity helper lives
once, except `pinned_mul` (the dedupe lane) and the ledger's row-20 twins
(`pinned_block_sum`, `twiddle_in`). The duplication is in scaffolding. The
one piece that matters is the **fixture RNG**: about 80 copies of `_mix`,
`splitmix`, `_splitmix`, `_u01` and `_hashed` generate the data every check
runs on. If one copy differs, a cross-lane or cross-vendor comparison
silently compares fixtures instead of kernels.

**Now (conflict-free: new files only):** lane `consolidate` adds
`checks/fixture_rng.mojo` and `checks/scaffold.mojo` (grid, hash printing,
upload/download, pointer helpers, `run_case`/`card_path` shape) plus a
binding prelude. It also adds a gate proving every existing fixture-RNG copy
agrees bit for bit with the canonical one over a hashed input set. After
that merges, CURRENT DIRECTIVES tells lanes to use these in all new code.

**After the lanes quiet (on its own branch, merging main regularly, landing
in one merge):**
1. Delete the fixture-RNG copies. The gate ran first, so this is bit-inert.
2. Migrate the check and binding scaffolding to the shared files.
3. Split `tools/identity_break.py` (12,878 lines) into per-family fragments,
   using the prep's mechanism.
4. Rename the `x_` directories once their lanes are certified.

**Never:** split the big certified kernel files (fused attention, the
identical GEMM). One file per contract is deliberate.
