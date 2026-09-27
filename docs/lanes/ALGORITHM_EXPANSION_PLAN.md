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
  pod up also keeps the pixi env and the warm Mojo cache: 10 to 20 minutes
  saved per rebuild compared with a fresh box.
- **Apple is still the bottleneck.** Metal bindings compile only on the Mac,
  one Metal job at a time. So the Apple column becomes a serialized
  **Apple steward** queue. When an algorithm is green on its pod, the
  steward builds it on the Mac and runs the light CPU(Arm) = Metal check
  plus the sabotage. Proposal: give the steward 2 to 3 cores for compiles,
  as an exception to the 1-core rule. The 12 lanes never touch Metal
  themselves.
- **AMD:** still at the release, through the changed-lanes selector.
  Optional: one shared single-MI300X box that runs a daily AMD sweep of the
  new lanes. That catches gfx942 surprises weeks earlier.

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

## Before the fan-out: one prep step (me, about half a day)

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
  Over a 1-2 week fan-out, that is about $1-3k.
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
- Pods live a few hours, then get torn down.

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

## Timeline (realistic)

- T+0:30 is setup: the registry prep, 9 pods bootstrapped, cloud Mac
  bootstrapped. The cloud Mac's first build is the slow part.
- T+0:30 to T+3:00 is **identity wave**. Each algorithm lands with CPU =
  NVIDIA plus sabotage on its pod, then goes to the Apple stewards.
- In parallel, as algorithms pass identity, is **speed wave**: FAST +
  IDENTICAL at 1M+, and every FAST change passes the quality rule (paired
  check against the reference, at least 5 seeds, at least 2 datasets).
- The Apple queue is the critical path. Each binding module gets one Metal
  build, not one per algorithm. With two Macs this should keep pace.
- Lanes 8 and 9 will not finish in 3 hours. Their deliverable is a design
  note plus the first algorithm, or a named refusal.

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
| R1 | **A submitted commit never reaches the cloud Macs.** `apple_steward.py submit` ships the request and patch over ssh, then the Mac's `process` runs `git fetch origin` and checks out the sha. But a cloud Mac's `origin` is the bare repo the laptop pushes to (`cloudmac.sh bootstrap`), and neither `submit`, `flush-deferred`, nor the brief's step 7 pushes the sha there. Every request would fail at "checkout of <sha>". | Until the tool does it, the lane runs `tools/cloudmac.sh push m2pro <sha>` before `submit` (brief step 7, revised). The tool fix: `submit` pushes to each non-deferred Mac and `flush-deferred` pushes before it ships; or `process` fetches the sha from GitHub over https (the repo is public). | **OWED: tool fix. The brief carries the workaround.** |
| R2 | **Seam checks are optional in the gate.** `algos_lane_check.py` prints a note and continues when a fragment has no `tools/identity_lanes/<lane>.checks`. A lane could register identity lanes, pass the GPU == CPU diff and never run an oracle: the "light" hole the brief closes in prose only. | Make a missing `.checks` a FAIL once the fragment registers any lane. Until then the orchestrator refuses to merge a lane whose fragment registers lanes and has no `.checks` listing (brief step 2, revised). | **OWED: tool fix. The brief carries the rule.** |
| R3 | **Per-seam sabotage arms are unverified by any tool.** The steward runs the one end-to-end `--sabotage` patch; the per-seam arms the brief requires live only in the lane's evidence directory. | Extend the `.checks` line format to `<driver>\t<sabotage patch>` and have the lane check run each pair (must FAIL under the patch, PASS after `git apply -R`). Until then, the lane's report at each commit names every seam patch and its result, and the orchestrator reads them. | **OWED: tool fix.** |
| R4 | **main was rewritten today.** The fetch that preceded the review showed a forced update on `origin/main` (`3cf7ae22...b35580c3`). Nine lanes pushing `HEAD:main` fast-forward cannot survive another one. | Turn on force-push protection for `main` for the duration of the fan-out. | **OWED: Andrew, in the GitHub settings.** |
| R5 | **The proof dummy ran on one vendor.** The A40 pod proved x86 CPU == NVIDIA through the whole loop. Nothing has run `algos_lane_check.sh` on a Mac with Metal, and that is the path every steward request takes. | Re-create the `x-prep-dummy` lane on a throwaway branch (its two commits are on main; `git revert ff4e81c0` on the branch), push it to the M2 Pro, and run it through `apple_steward.py work --once` before any lane submits. Then delete the branch. | **OWED: orchestrator, before the first submit.** |
| R6 | **Pod bootstrap is manual.** The brief tells each lane to `git init` and fetch on the pod so `git apply` works; `sync` does not check that the pod tree is at the worktree's base commit, so a patch that applies on the laptop can fail on the pod for a stale-tree reason. | `dev_pod.sh up` seeds the git tree at the lane's base sha itself; `sync` refuses when the pod's HEAD is not the worktree's merge base with origin/main. | **OWED: tool fix. Brief step "Your setup" carries the manual form.** |
| R7 | **`sequence` and `cnn` build identical only** (`host_surface.EXPANSION_IDENTICAL_ONLY`), while FINAL DECISIONS say FAST on neural. Both lane briefs promised FAST speed work in step 9 that their bindings cannot build. | The neural FAST policy change is one shared edit (`EXPANSION_IDENTICAL_ONLY`, `build_sets.sh` IDENTICAL_ONLY lists, `_backend`'s tier table) the orchestrator makes once, after the identity wave. Lanes 6 and 8 do IDENTICAL speed work only until then (briefs revised). | **OWED: orchestrator, after the identity wave.** |
| R8 | A fragment may bind a key twice; the second `FAMILIES = ...` silently wins (`host_surface._read_expansion_fragment`). | Refuse a key bound twice. | **OWED: one-line tool fix. The brief says "each once".** |
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

## The estimate, corrected

"1 to 2 hours per lane" was written before the per-seam bill was put in the
brief. With every numeric seam owing an oracle, a separating fixture, a
sabotage arm that bites, a DEVIATION, a card stage and a ledger row, an Easy
algorithm is half a day to a day of agent time, because most of it is
templated on existing contracts, and a lane of seven Easy items is several
days. Parallelism buys throughput; there is no light form. So:

- Pod leases match the lane's real duration (`dev_pod.sh up <lane> 240`,
  extended hourly while working), not a three-hour window. A pod is torn
  down when its lane's last algorithm is merged.
- Expect one or two algorithms per lane in the first window and the rest
  after. The paper's count moves only when a release record admits a lane
  on three vendors.
- Speed work (FAST and IDENTICAL) starts per lane after that lane's
  identity passes, never before; a FAST commit on nine pods each going
  back through one Metal queue is the second bottleneck, so sequence it.

## Order of operations before the first lane starts

1. R4: protect main.
2. R1 and R2 tool fixes, or at least the brief's workarounds (done in the
   briefs).
3. R5: the dummy through the M2 Pro steward on Metal.
4. Bootstrap the nine pods (`dev_pod.sh up`) and the bench pod.
5. Start the nine lanes with the COMMON BRIEF and their sections.
