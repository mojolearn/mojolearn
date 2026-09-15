# Restart plan: the Sep 14-15 CPU path, harness and multi-GPU fan-out

Written 2026-09-15 about 09:30 ET by the orchestrator session. Andrew may run out of budget and restart
under a different Claude account. This file lets a fresh session pick up every lane with no memory of the
old one. It is on `main`; branch names below are on `origin`.

**SUPERSEDED FOR STATUS by docs/lanes/STOP_STATE_2026-09-15.md (Andrew stopped all lanes, Sep 15 ~10:25 ET). Rules below still hold.**



## 00. OVERRIDES EVERYTHING BELOW: GPU records only for PyPI releases (Andrew, Sep 15 ~10:30 ET)
- Between releases, rent NO GPU boxes: no RunPod, DigitalOcean or Hot Aisle legs for any lane.
- Prove on the Apple M4 locally (Metal vs host CPU, one core) and against GPU columns already in the repo.
- Write anything that needs NVIDIA or AMD bits as OWED to the next release record.
- At a PyPI release, take ONE record: 1 AMD, 1 NVIDIA, 1 Apple column. No second AMD model, no extra NVIDIA architecture, no two-device columns, no full re-record unless Andrew asks.
- The box and leg rules in section 3 apply only to that release record.

## 0. First ten minutes for the new orchestrator session

1. Read this whole file. Then read `docs/lanes/FANOUT_RULES_2026-09-15.md`. Every lane agent must also
   read it. It uses `$SP` for "your session's scratchpad directory".
2. Check what moved since this file was written:
   ```sh
   cd /Users/andrewhendel/CascadeProjects/mojolearn && git fetch origin
   git log --oneline -30 origin/main
   for b in lane/identity-record-next lane/cpu-training-gate-budget lane/cpu-training-gbdt-ordered \
            lane/cpu-training-transformer lane/cpu-training-mamba lane/cpu-training-samba \
            lane/cpu-training-embedding-ivf lane/rl-logprob-parity lane/batch-invariance-2 lane/r2-binding-cache; do
     echo "$b ahead=$(git rev-list --count origin/main..origin/$b 2>/dev/null || echo no-remote)"
     git show origin/$b:docs/lanes/LANE_STATUS_$(echo $b | tr / -).md 2>/dev/null | head -40
     gh run list --branch $b --workflow "CPU identity gate" -L 2
   done
   git worktree list; df -h /System/Volumes/Data
   ```
   A lane whose branch is 0 commits ahead of main is merged: skip it. A lane with a `LANE_STATUS_*.md` on
   its branch has left its own next commands; the agent brief below says to read that file first.
3. Check for rented boxes nobody is watching. Leases self-expire in 60 minutes, but check anyway:
   `bash tools/hotaisle_leg.sh status`. For RunPod, list pods with the key in `~/.runpod/config.toml`. For
   DigitalOcean, see `tools/do_extra_leg.sh`. Reap only boxes whose description names a lane from this
   file and whose driver process is gone.
4. Relaunch only the unfinished lanes, one fresh general-purpose subagent each (never `fork`), with the
   brief in section 3. Each prompt starts "You are the <name> agent. FIRST read
   docs/lanes/FANOUT_RULES_2026-09-15.md completely..." and then the lane text.
5. As lanes report, verify their evidence and merge (section 4).

## 1. Standing rules (Andrew's, binding)

- **Local compute: at most ONE CPU core per subagent** for any local build, test or measurement: `nice -n 19`,
  `OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1`,
  `mojo build -j 1`, one process at a time. Anything long or needing an NVIDIA or AMD GPU goes to a rented box.
- **Never build in the shared checkout** `/Users/andrewhendel/CascadeProjects/mojolearn`. Each agent uses its
  own `git worktree`. After its branch merges it moves untracked leg evidence to `~/mojolearn-evidence/<lane>/`
  and removes the worktree. On Sep 15, 159 stale worktrees had filled the disk to 4.2 GiB free.
- **Boxes:**
  - AMD correctness work uses the box order Hot Aisle, then DigitalOcean, then RunPod (`tools/pick_box.sh`).
    Hot Aisle's balance is low (about $19): use its single-GPU spec only, never `2gpu`.
  - NVIDIA work goes to RunPod (`tools/gemm_remote_leg.sh nvidia`). It defaults to an RTX 4090; set
    `MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 80GB HBM3"`.
  - Always set `MOJOLEARN_GEMM_LEG_LOCAL_CARD`.
  - Two devices in one process means RunPod `MOJOLEARN_GEMM_LEG_GPU_COUNT=2`.
  - Datasets stage from Cloudflare R2 automatically (`R2 STAGED` in the leg log). Never refetch from the internet.
  - Bake commits with `git rev-parse`; never type a full sha.
- **CI:** routine pushes use three hosted CPU environments for inference/plumbing checks. Full CPU training verification runs weekly, manually and before publication, with parallel shards and caching. See CPU_GATE_RECOVERY_2026-09-15.md for retained coverage and validation.
  - Never cancel an owed run. Andrew approved cancelling ONLY duplicate or superseded runs (the same commit
    twice, or an older push on a branch that has since been re-pushed).
  - Push branches only when ready for a gate. WIP pushes carry `[skip ci]`.
- **Merging:** only code that compiled and ran, with evidence and a green three-environment CPU gate on the branch.
  Merge origin/main into the branch; if the resolution touched only lists and generated spans, the merge may
  go to main without its own gate. Pushes to main are fast-forwards of a freshly fetched origin/main.
  `--no-verify` is allowed only for the known refusal of the multi_gpu `.tgz` files already on main.
- **Evidence:** a check must be seen to fail on a sabotaged tree first. When a cell diverges, print its hash
  from every column and name the column that stands alone. Any "never / missing" claim needs a repo-wide
  `git grep`.
- **Writing:** American English, no em dashes, never claim we are faster.

## 2. What is already on main (do not redo)

As of main `4eac5719a` (Sep 15 morning), all gated green unless noted:

- **CPU (host) training, bitwise IDENTICAL to the GPU columns, gated on seven runners**, 100+ lanes:
  - The classical families, all four base GBDT lanes, GBDT losses and adapters (2cf5562b1), RF/ET variants
    (185acb79c), ARIMA (030d4e3b2), batch 3 option variants (b449ffa78), UMAP (05be1206d),
    pca-full-whiten (65978ccd1).
  - metrics-classification (b4f012757), MLP (88e90d336), the workstream D estimators (fda00d33b).
  - misc: kmeans variants, resampling, optimizer and training primitives (b7e5a4287).
- **Exposed:** IVFIndex and Embedding (275ceb3de), IVF euclidean fixed (9b7ee8960), Embedding sabotage arms,
  clauses and `plan="sort"` (6073b3aa2).
- **Fixes:**
  - kmeans-sqrt: DEVIATION 2715 (NVIDIA sqrt in inertia) and 2716 (wrong labels on every column), 22f4c914f.
  - AMD 2xMI300X SR-IOV cross-device stale read: all AMD cross-device bytes stage through host,
    `core/multi_gpu.mojo::transfer_bytes` (d36cd8cb4).
  - NearestNeighbors `p` refusal (25c3ade4f). Metrics test modules, run in the CPU gate (df617c699).
- **Multi-GPU ordered drivers** (2xH100 and 2xMI300X equal to one device): forest pool, GaussianMixture,
  resampling, HDBSCAN, Cholesky, KernelRidge, Nystroem, RBFSampler, plus par-* lanes.
- **Harness:** `tools/identity_break.py` with the `batch` invariance part. The 166-lane record
  `bench/results/identity_break/2026-09-14_166-lanes/` is the CPU gate's GPU columns (Apple M4, H100,
  MI325X, two-device columns); batch IDENTICAL=1188 with zero BATCH_MOVED. Its kmeans-sqrt cells are
  pre-fix, so kmeans-sqrt is checked against `2026-09-14_kmeans-sqrt-fix/` via `TRAINING_FIX_LANES`.

## 3. The lanes in flight, with restart briefs

Status is as of 2026-09-15 09:20 ET. Each lane was told to push WIP with `[skip ci]` and to keep
`docs/lanes/LANE_STATUS_<branch with dashes>.md` on its branch. **Read that file first if it exists; it
supersedes the status line here.**

### L1. record2: the next full identity record (STOPPED Sep 15 ~10:30 ET by Andrew's release-only rule: finish in-flight legs, commit partial, do not switch the gate; do NOT restart)
- **Branch:** `lane/identity-record-next`. No commits yet at writing. Legs were running from worktrees
  `wt-legs-R` and `wt-apple-R`; leg dirs `bench/results/e1g/2026-09-15_*-rec2-*` there are untracked. The
  lane was told to copy them to `~/mojolearn-evidence/record2/`.
- **Goal:** replace the 166-lane record as `TRAINING_GPU_COLUMNS` with a record at a current main commit
  covering every lane:
  - Includes: ivf, ivf-euclidean, embedding, embedding-sort, and par-forest-pool, par-gmm, par-resample,
    par-hdbscan, par-cholesky, par-kernel-ridge, par-nystroem, par-rbf-sampler.
  - The kmeans-sqrt cells come from the fixed build.
  - Columns: Apple M4 locally (one core, one background process, `--merge` if split), H100 (RunPod
    2xH100, one process per GPU), MI325X (DigitalOcean, split legs), 2xH100 and 2xMI300X for all par-*
    lanes, and an Apple column for the par-* lanes.
- **Then:**
  - Write the README and diffs; explain every cell that changed vs the 166-lane record.
  - Point `TRAINING_GPU_COLUMNS` and the gate step at it with exact counts, and show the step fails with a
    wrong count first.
  - Remove `TRAINING_FIX_LANES` and `TRAINING_FIX_COLUMNS` and the kmeans-sqrt fix step if redundant.
  - Gate, then merge. Budget about 4 to 5 GPU hours.
- **Restart brief:** "Resume the record2 lane: read LANE_STATUS on origin/lane/identity-record-next and
  ~/mojolearn-evidence/record2/; reuse complete legs taken at the SAME commit (check commit.txt in each
  leg dir); rerun only missing columns; then the steps above."

### L2. gatehyg: test isolation and CPU gate time budget
- **Branch:** `lane/cpu-training-gate-budget`, 2 commits (36f9c1b27). Gate run 34971932337 was queued.
- **Goal:**
  - (a) `test_byte_lm_host.py::test_native_helpers_fall_back_to_the_cpu_binding_only_for_the_three` failed
    when run after `test_cpu_training_e2.py`. It was fixed at the root on the branch; check the commit message.
  - (b) Keep every CPU gate step under its limit with 40% margin without weakening checks.
  - Reconcile with `lane/cpu-training-gbdt-ordered`, which raised the job timeout to 120 minutes. The two
    branches merge cleanly; the lane is to pick 60 or 120 from measured job timings.
- **Extension (Andrew approved, Sep 15 ~10:00 ET):** after the time-budget merge, speed up CI:
  1. actions/cache for the pixi envs and Mojo toolchain, keyed on pixi.lock.
  2. Cached host bindings keyed on the source tree hash, mode and runner image. Never cache sabotage builds under the prod key.
  3. A same-branch concurrency group that cancels superseded runs. Never cancel main's runs.
  4. If cheap, family-scoped gates on non-main branches, keeping the full gate on main.
  Report before and after wall time. Measured before: 1.5 to 4 hours from push to result.
- **Restart brief:** "Resume gatehyg: read the branch commits and LANE_STATUS; wait for or read gate
  34971932337 (or the latest on the branch); tabulate per-step minutes; reconcile the timeout with
  gbdt-ordered; show the gate still fails on a sabotaged tree; merge each task separately."

### L3. gbdtord: GBDT ordered and categorical CPU training
- **Branch:** `lane/cpu-training-gbdt-ordered`, 10 commits (3b3ab79ad, which merged main and raised the
  job timeout to 120 min). Gate run 34972329290 was queued. Earlier branch gates were green.
- **Goal:** CPU training for gbdt-ordered-rmse, gbdt-feature-freq, gbdt-categorical-ctr and
  gbdt-pointwise-l2-bayesian-eval, IDENTICAL to TRAINING_GPU_COLUMNS, declared in host_surface.py.
- **Restart brief:** "Resume gbdtord: read LANE_STATUS and the branch log; if the latest gate is green,
  merge origin/main (lists and spans only), run docs_facts --check and wheel_ci pins/inventory and
  test_host_surface, push to main, remove the worktree. If red, `gh run view <id> --log-failed | tail -200`
  and fix."

### L4. cputransformer: Transformer block CPU forward and backward
- **Branch:** `lane/cpu-training-transformer`, 2 commits (82ffcb388 code, 6839b4b12 evidence). It adds a
  new `transformer` host family exporting transformer_forward, transformer_forward_fresh,
  transformer_decode_step and transformer_backward.
- **Status:** on the M4 (one core) against the 166-lane record, IDENTICAL=18 train, 18 infer and 18
  batch, require-columns 4 OK; sabotage DIVERGENT on all 18. Wheel CI green. Gate run 34969598898 was in
  progress.
- **Restart brief:** "Resume cputransformer: if gate green, merge origin/main, light checks, push to main,
  remove the worktree; else read the failure and fix."

### L5. cpumamba: Mamba 1/2/3 CPU forward and backward, then the Mamba-1 oracle question
- **Branch:** `lane/cpu-training-mamba` (ddd875cc8). Gate run 34973248141 was queued. It adds a `mamba`
  host family (`bindings/_mojolearn_mamba_host.mojo`, `build_mamba_host.sh`). The backward is the DEVICE
  VJP generated for the host by `tools/mamba_host_gen.py` into `mamba/host/gen/`.
- **Status:** mamba1, mamba2, mamba2-dtlimit and mamba3 read IDENTICAL=36 train, 36 infer and 36 batch on
  the M4 against the 166-lane columns; sabotage DIVERGENT=36.
- **Second task after merging:** the repo's existing Mamba-1 host backward ORACLE does not match the
  device.
  - `git grep` which checks, gates or docs rely on it.
  - Decide against the contract whether the oracle or the device is wrong, and fix with evidence (M4 one
    core; a GPU leg only if the device side is implicated).
  - Correct any doc that claims they match.
- **Restart brief:** "Resume cpumamba: merge the four Mamba lanes when the gate is green (merge main,
  light checks, push), then do the Mamba-1 oracle investigation above."

### L6. cpusamba (named cpumamba13 when launched): Samba stack CPU path
- **Branch:** `lane/cpu-training-samba`, local only at writing (5 commits, c295a7355, likely merges of the
  mamba and transformer branches). Told to push WIP.
- **Goal:** CPU forward, backward and training for `samba` and `samba-untied-dropout-accum`, IDENTICAL to
  TRAINING_GPU_COLUMNS.
  - Reuse the mamba host family (L5), the transformer host family's attention (L4), and the training host
    family (embedding, RMSNorm, linear, clip, accumulate, optimizer).
  - The new pieces are likely the dropout RNG and the untied head.
  - Declare the lanes and shrink the no-CPU-path sentence.
- **Restart brief:** "Resume cpusamba: fetch origin/lane/cpu-training-samba (or recreate from
  origin/main once L4 and L5 are merged); run the samba host cells against TRAINING_GPU_COLUMNS on one
  core, IDENTICAL on every cell, sabotage DIVERGE, declare, gate, merge."

### L7. cpuembed: Embedding, IVF, GPU byte-LM lanes and tokenizer on CPU
- **Branch:** `lane/cpu-training-embedding-ivf`, no commits at writing.
- **Goal:** CPU paths declared for embedding, embedding-sort, ivf, ivf-euclidean, byte-lm and
  byte-lm-resident, and tokenizer (check it may only need a declaration). Also shrink "the Embedding
  layer" out of the no-CPU-path sentence.
  - embedding, embedding-sort, ivf and ivf-euclidean have no columns in the 166-lane record. Their columns
    are in `2026-09-14_ivf-embedding/`, `2026-09-15_embedding-sort/` and `2026-09-14_ivf-euclidean/`.
    Gate via `TRAINING_FIX_LANES`-style lists, or after L1 merges.
  - Reuse the host predictions in embedding_check and ivf_check, and the training host family's embedding
    ops. byte-lm and byte-lm-resident may be served by LanguageModelHostTrainer.
- **Restart brief:** "Resume cpuembed from LANE_STATUS if present, else start from origin/main with the
  goal above."

### L8. rlparity: RL sampler vs trainer log-prob parity (harness part `rlpair`)
- **Branch:** `lane/rl-logprob-parity`, no commits at writing.
- **Why:** Thinking Machines, "Defeating Nondeterminism in LLM Inference" (September 2025). For RL, the
  sampler's log-probs must equal the trainer's. The existing `batch` part never asserts that a decode step
  equals the matching position of a full forward: `_block_fit` hashes prefill and one step separately.
- **Goal:** for the byte LM, TransformerBlock, Mamba1/2/3Block and Samba:
  - **Sampler side:** greedy incremental decode (prefill, then T steps through the state or KV-cache API)
    at B1 in {1, 5}, recording per-token log-softmax of the chosen ids.
  - **Trainer side:** a teacher-forced full-sequence forward through the TRAINING path (dropout off) at B2
    in {1, all} and under microbatch splits.
  - **Assertions under IDENTICAL:** sampler log-probs equal trainer log-probs bytewise (log-softmax via the
    library's own arithmetic); invariance to B1, B2 and splits; continuous batching (a sequence's
    log-probs unchanged when another joins or leaves mid-decode; add the capability or record N/A with
    the reason).
  - A sabotage switch `MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE=1` must turn every cell RLPAIR_MOVED.
  - Evidence: M4 (Metal plus host CPU, one core), one H100 leg, one MI325X leg; diffs under
    `bench/results/identity_break/<date>_rlpair/`.
  - Any RLPAIR_MOVED is a finding: per-column hashes and a brief.
- **Restart brief:** the goal above, from origin/main, reading LANE_STATUS first if the branch exists.

### L9. batch2: batch invariance gaps the harness declares
- **Branch:** `lane/batch-invariance-2`, no commits at writing. Coordinates with L8 on
  `tools/identity_break.py`; merge main before each push.
- **Goal, in order, each gated and merged separately:**
  1. A `batchgrad` part: per-sample and microbatch-accumulated gradients (1, 7, rest) equal to the
     whole-batch gradient bytewise under IDENTICAL, for the sequence blocks' backward, byte-LM and Samba
     trainers, SmallMLP, the training primitives, and classical fits where the contract allows. Read
     `training.accumulate` and `accumulation_is_aligned` for the contract's alignment conditions. Add a
     sabotage switch.
  2. Serving-scale B at the public surface: B in {1, 17, 64, 256} and long L, for the sequence models and
     a representative classical set.
  3. Ragged and padded sequence batches, a NEW capability: a lengths or mask argument for the byte LM and
     TransformerBlock, then Mamba and Samba, so padding never enters a reduction that reaches a real token.
     Assert a sequence in a ragged batch equals it alone. Stop with reasoning where the contract forbids it.
- **Evidence:** M4 one core, one H100 leg, one AMD leg, `bench/results/identity_break/<date>_batch2/`.
  Update the harness docstring's "does NOT test" list.
- **Restart brief:** the goal above, from origin/main, reading LANE_STATUS first if the branch exists.

### L10. bincache: prebuilt Mojo binding cache in R2 (Andrew approved, Sep 15 ~10:00 ET)
- **Branch:** `lane/r2-binding-cache`.
- **Why:** every GPU leg compiles about 22 bindings (about 10 min) before working, while dataset staging from R2 already takes seconds on RunPod, DigitalOcean and Hot Aisle.
- **Goal:** a content-addressed cache in the `mojolearn-data` R2 bucket, used by gemm_remote_leg.sh, do_extra_leg.sh and hotaisle_leg.sh.
  - Key: source tree hash, mode, GPU arch or cpu, target column, toolchain from pixi.lock, container image and OS, and defines.
  - Sabotage builds are never served to prod.
  - Every file is sha256-verified. Leg outputs record which bindings came from the cache.
  - `MOJOLEARN_BINCACHE=0` opts out, and the default stays off until evidence exists.
  - The release pipeline is out of scope.
- **Evidence:** a fresh then cached leg pair on an RTX 4090 and on AMD, with identical binding digests and lane hashes, and the compile minutes saved.
- **Restart brief:** "Resume bincache from LANE_STATUS on origin/lane/r2-binding-cache if present, else implement the goal above from origin/main."

## 4. Orchestrator protocol while lanes run

- Agents that end their turn while waiting on a gate often never wake. Check branch gates yourself
  (`gh run list --branch <b> -L 3`) and merge a green branch per section 1 if its agent is gone.
- After a merge to main, the push queues a main gate. Confirm it goes green; a runner that "failed to be
  acquired" is GitHub infrastructure, not code.
- Never SendMessage a finished agent (it resumes with stale context). Start a fresh one with a brief
  instead. Exception: a lane that is only waiting on its own gate.
- When a lane merges, confirm its worktree is removed and its evidence moved.
- Record each merge sha in this file's section 2 or in a successor handoff, only if Andrew asks for one.

## 5. After every lane is merged (not started yet; Andrew said no new lanes)

These are the remaining items as known on Sep 15. Do not start them without Andrew:
- A full record after L4 to L9 land, so the new CPU and rlpair/batchgrad parts carry GPU columns.
- A PLAN_SORT crossover measurement for Embedding, and the shipped shape (V=128256, d=4096, T=4096) on AMD.
- The par-* lanes' capacity and throughput claims (none are made today).
- 0.8.6 packaging freeze, only when Andrew says ship.
