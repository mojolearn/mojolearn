# Fan-out rules, Sep 14 2026 evening (read fully before doing anything)

## 00. OVERRIDES EVERYTHING BELOW: GPU records only for PyPI releases (Andrew, Sep 15 ~10:30 ET)
- Between releases, rent NO GPU boxes: no RunPod, DigitalOcean or Hot Aisle legs for any lane.
- Prove on the Apple M4 locally (Metal vs host CPU, one core) and against GPU columns already in the repo.
- Write anything that needs NVIDIA or AMD bits as OWED to the next release record.
- At a PyPI release, take ONE record: 1 AMD, 1 NVIDIA, 1 Apple column. No second AMD model, no extra NVIDIA architecture, no two-device columns, no full re-record unless Andrew asks.
- The box and leg rules in section 3 apply only to that release record.


## 0. Round 2 updates (Sep 14 ~20:45 ET)
- Main has moved a lot since launch: GBDT and ARIMA CPU training, IVFIndex and Embedding exposed, and multi-GPU drivers for forest pool, GaussianMixture, resampling, HDBSCAN, Cholesky, KernelRidge, Nystroem and RBFSampler. Always fetch first.
- tools/identity_break.py no longer has a single owner. Anyone may add lanes. Merge origin/main right before every push, and keep both sides of lane-list conflicts.
- The 166-lane record with the batch part is on branch lane/identity-record-166, gate pending. It moves TRAINING_GPU_COLUMNS to bench/results/identity_break/2026-09-14_166-lanes/ once merged. Until then the gate uses the 136-lane record.
- pytest lives in /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python. The python test modules need built bindings in YOUR worktree's python/mojolearn/identical/ (or the host build dirs); otherwise collection refuses.
- Datasets: remote legs stage from Cloudflare R2 automatically. Look for `R2 STAGED` in the leg log, and never refetch from the internet.

## 0b. Round 3 updates (Sep 15 ~00:30 ET)
- Main has the 166-lane record (TRAINING_GPU_COLUMNS), 82 covered CPU training lanes, and UMAP on CPU.
- The kmeans-sqrt fix: the 166-lane JSONs carry the OLD kmeans-sqrt bits. Fixed columns are in bench/results/identity_break/2026-09-14_kmeans-sqrt-fix/.
- The AMD cross-device copy fix: all AMD cross-device bytes go through core/multi_gpu.mojo::transfer_bytes.
- Hot Aisle balance is LOW (~$19). Use only its single-GPU 13core spec, never `2gpu` (60-minute minimum, $5.98). If no single-GPU stock, go straight to DigitalOcean, then RunPod.
- CI queue: each main push queues a full gate. Merge in batches where you can.
- DISK (Sep 15): worktrees filled the Mac. When your branch is merged:
  1. Move any untracked bench/results leg dirs to ~/mojolearn-evidence/<your-agent>/.
  2. Run `git -C /Users/andrewhendel/CascadeProjects/mojolearn worktree remove --force <your worktree>`.
  3. Say so in your report.

Repo: /Users/andrewhendel/CascadeProjects/mojolearn (GitHub mojolearn/mojolearn). origin/main was e22374acd at launch.
Orchestrator scratchpad: $SP (call it $SP).
Prior session's leg bodies and logs, for reference: $SP/prior-session/.
Six agents run at once: gbdt, arima, cpudecl, harness, multigpu, legsdocs. Each owns ONE task.

## 1. Local compute: ONE core per agent (Andrew's explicit rule)
- You MAY build, test and measure locally, but with at most ONE core, ONE process at a time, never parallel, never backgrounded.
- Always: `nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 ...`. `mojo build` gets `-j 1`. pixi tasks get the same env.
- Any local timing you report says "one core, M4, shared machine".
- The Mac has no `timeout` command. Wrap long local runs in a script file under your own directory in $SP and run that script; do not paste long pipelines inline.
- Anything longer than about 20 minutes locally, or needing an NVIDIA or AMD GPU, goes to a rented box (section 3).
- Do not spawn subagents.

## 2. Git
- NEVER build, edit or commit in the shared checkout /Users/andrewhendel/CascadeProjects/mojolearn. Make your own worktree:
  `git -C /Users/andrewhendel/CascadeProjects/mojolearn fetch -q origin && git -C /Users/andrewhendel/CascadeProjects/mojolearn worktree add $SP/wt-<agent> -b <branch> <start-point>`.
  A binding rebuilt in place under a running process kills that process with exit 137 and no output.
- Before every commit run `git rev-parse --abbrev-ref HEAD` and confirm it is your branch.
- Stage explicit paths only. Never use `git add -A` or `git add .`. Never rewrite history, force push, or rebase a pushed branch; merge instead.
- Never type a full sha. Substitute `$(git rev-parse <short>)` and grep the file for it before using it.
- No blobs over ~5 MB in the repo. Raw leg archives stay untracked, under bench/results/e1g/ or ~/mojolearn-evidence/. Commit summaries, cards, diffs and READMEs.
- The pre-commit hook refuses 33 multi_gpu .tgz files that are already on origin/main. `--no-verify` is allowed on a merge commit ONLY for that refusal; say so in the message.
- Commit messages follow the repo's style: one long, specific paragraph saying what changed, what ran where, and what is still owed. End with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01KWrZUhsrGCapB9isp5Sq4h
  ```
- MERGING TO MAIN. Only work that compiled and ran, with evidence and a green gate on the branch (or on a local run for docs-only changes), gets merged. Unrun code stays on its branch, with the reason stated.
  The merge sequence:
  1. `git fetch origin`, then merge origin/main into your branch.
  2. Resolve conflicts. Other agents also edit python/mojolearn/host_surface.py, the generated doc spans and the CPU gate workflow. For spans, run `python3 tools/docs_facts.py --write` and then `--check`.
  3. Rerun the light checks: docs_facts --check, `python3 packaging/wheel_ci.py pins .` and `inventory .` (if present), and the host_surface test module.
  4. `git push origin HEAD:<branch>`, then `git push origin HEAD:main`. Push to main only as a fast-forward of the origin/main you just fetched. If it is rejected, fetch and repeat.
  5. If the merge resolution changed code rather than spans or lists, the merge commit needs its own gate run before you push to main.

## 3. Rented boxes (cost money; follow exactly)
- Order: `bash tools/pick_box.sh --need amd|any` right before renting. For AMD correctness work use Hot Aisle first (tools/hotaisle_leg.sh), then DigitalOcean (tools/do_extra_leg.sh, MI325X), then RunPod (tools/gemm_remote_leg.sh amd).
- NVIDIA work goes on RunPod via tools/gemm_remote_leg.sh nvidia. It defaults to an RTX 4090; for an H100 set `MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 80GB HBM3"`. Read `gpu:` in the log header before naming the evidence directory.
- gemm_remote_leg.sh runs a LOCAL Apple card build unless `MOJOLEARN_GEMM_LEG_LOCAL_CARD` points at an existing card. ALWAYS set it. Existing cards are at bench/results/e1g/*/local/apple.card in the shared checkout and in the prior session's worktrees; copy one into your worktree.
- Two devices in one process (par-* lanes) means RunPod with `MOJOLEARN_GEMM_LEG_GPU_COUNT=2`. Hot Aisle 2gpu pins one GPU per container. RunPod legs next to another agent's pod need `--allow-concurrent`.
- Hot Aisle allows 2 VMs for the whole team, shared by all six agents. If `pick_box.sh` says none, take the next provider. NEVER sit waiting.
- Bodies: RunPod passes no environment, so bake the commit into the body (see $SP/prior-session/dfix2_a_wrap.sh, par2_wrap_amd.sh, batch_record_2gpu_body.sh, ivf_embed_km_body.sh). Leg extras go in `MOJOLEARN_GEMM_LEG_EXTRA=<body>` and `MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-<vendor>-<gpu>-<lane>`. Host bindings on a box build with `env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu`.
- Datasets always stage from R2 (tools/dataset_store.sh stage). The runners do this themselves.
- Leases are 60 minutes, and the scripts' watchdogs delete boxes. Keep a body under ~50 minutes or split it into two legs.
- Run each leg driver in the FOREGROUND of a Bash call with a long timeout (up to 600000 ms), or with `run_in_background: true` and poll its log. If your session ends, the teardown trap deletes the box.
- NEVER cancel an owed run, a queued CI run, or another agent's box. `bash tools/hotaisle_leg.sh status` shows VMs; only reap one whose description names YOUR lane and whose driver is dead.

## 4. CI
- The routine CPU identity gate uses three hosted environments (Linux x86-64, Linux ARM64, Apple silicon macOS), with inference/plumbing checks on pushes. Full CPU training certification runs weekly, manually, and before publication, with parallel lane shards. See CPU_GATE_RECOVERY_2026-09-15.md. The seven-runner queue described at launch is historical.
- Push a branch only when it is ready for a gate. Batch your commits. The workflow cancels superseded runs on the same branch only; main runs are never cancelled. Do not manually cancel other owed runs.
- `gh run list --branch <b> -L 5`, `gh run view <id> --json jobs`, and `gh run view <id> --log-failed | tail -200` for failures. Poll every 5 to 10 minutes, not faster.

## 5. Evidence rules (the lessons behind them are expensive)
- A verification must be seen to FAIL first: run the check on a sabotaged or unfixed tree and watch it fail. Print the matches, not a count.
- A grep returning 0 is not proof. Any "never", "missing" or "not measured" statement needs a repo-wide `git grep` on origin/main, and you read the code, not the prose.
- When a cell diverges, print that cell's hash from EVERY column of every record on main and name the column that stands ALONE before naming a vendor.
- A commit message is not a file change. After committing a correction, grep for the new text.
- IDENTICAL vs DIVERGENT are verdicts from the harness (`tools/identity_break.py --diff`), not from reading code.
- "Written from a read only, nothing compiled" is not done. Compile it (one core locally, or on a box) and run it.

## 6. Writing
- American English. No em dashes. Plain sentences.
- Never claim we are faster. No "price of determinism" claims.
- Doc-correction clause: if you find any document FALSE (any file, any subject), fix it in your branch if it is in your area. Otherwise list file:line and the true statement in your report.

## 7. Final report (your last message; under 450 words)
- Branch and final commit short shas.
- Merged to main (merge sha) or not, and why.
- Evidence: leg directories, CI run IDs, and verdict lines quoted.
- What is still owed, with exact commands.
- False docs found.
- Money spent (boxes rented, minutes).

## CPU product boundary (September 15 follow-up)

Broad CPU training is an internal reference surface. Keep training-only
families out of `host_surface.wheel_families()`. New internal runtime tests
that fit CPU estimators should use the scoped private
`mojolearn._cpu_reference.reference_training()` context/decorator, as the
source identity harness does. Ordinary CPU estimator fits refuse; public
saved-model inference and the already-published byte-LM trainer remain.

## 000. CI is a light touch (Andrew, Sep 15 2026 ~12:15 ET)

- No workflow runs heavy checks on push. The CPU identity gate, the byte LM CPU gate, the forest host gate and wheel CI run only by hand (`gh workflow run`), on the CPU identity gate's weekly schedule, or from the release workflow.
- Community health (a ten minute doc and diagnostics check) still runs on pushes that touch its paths. The external contribution checks still run on outside pull requests.
- Lanes never wait on CI. A lane merges on its one-core M4 small-fixture evidence plus `python3 tools/docs_facts.py --check` and `python3 packaging/wheel_ci.py pins .`.
- Never workflow_dispatch a full gate for a single lane.
- Long local CPU work goes to `tools/runpod_cpu_leg.sh` (Andrew, Sep 15 2026: "move things to runpod cpu"). That means host binding builds past a quick check, identity_break CPU columns, sabotage builds and pytest modules. The Mac keeps Metal checks and one-core quick checks. A CPU pod is one per lane, with the on-pod watchdog and the Mac dead-man armed, and a delete verified through the API. Dry run first, then `--rent`. See `docs/RUNPOD_CPU_LEG.md`.

### CI shape since the light-touch rewrite

- `light-checks.yml` is the ONLY workflow that starts by itself: push to main, one Linux runner, Python only, a ten minute cap. It runs doc facts, build pins, package inventory, and a check that fails if any other workflow gains a push or schedule trigger.
- Every other workflow is manual. Each of its jobs is skipped unless the dispatch input `confirm` is `RUN-HEAVY`, so a wrong trigger costs nothing.
  - The CPU identity gate, byte LM CPU gate, forest host gate, wheel CI, GPU validation, the self-hosted Python tests and community health all follow this rule.
  - The release workflow passes `RUN-HEAVY` to the CPU certification it calls.
  - The daily issue bot and the outside-contributor pull request checks are unchanged.
