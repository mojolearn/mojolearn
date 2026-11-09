# Agent working rules

This is the one rule file for every agent in this repo (Codex, Claude, their lanes and subagents). `CLAUDE.md`
imports it and adds only what is specific to Claude sessions.

- One rule per topic, current text only. When a rule changes, replace it here; do not append a second version.
- The later dated owner statement wins unless it is narrower in scope. Older wording, rationale and finished
  campaigns are kept verbatim in [docs/rules/RULES_HISTORY.md](docs/rules/RULES_HISTORY.md); it is history, not rules.
- Lines marked `<!-- OPEN: question N -->` hold two texts Andrew has not yet chosen between
  ([docs/rules/RULES_CONSOLIDATION_2026-11-09.md](docs/rules/RULES_CONSOLIDATION_2026-11-09.md)). Until he does, follow the
  stricter one.

## The name is Mojolearn, with a capital M (Andrew, 2026-11-02)

- In prose (docs, docstrings, README, site, paper, emails, release notes, commit messages and replies) write
  **Mojolearn**: capital M, the rest lower case. Never "mojolearn", "MojoLearn" or "MOJOLEARN" as the name.
- Code stays lower case: the package, module and command names (`pip install mojolearn`, `import mojolearn`,
  `python -m mojolearn verify`), paths, URLs and domains (mojolearn.org). Put them in backticks in prose.

## Identity and bits

- **The contract (Andrew, 2026-10-13).** In `identical` mode the same bits come out on Apple, NVIDIA and AMD GPUs
  within one version. Identity is decided by the output hash each run already stores.
- **Bits may change between versions (Andrew, 2026-11-05, restated 2026-11-08).** Matching across GPU vendors within
  one version is the only bits requirement. Never hold a performance fix, a correctness fix, a smaller fixture, a
  better fold order, a refactor or a cleanup because it changes an older version's bits. "Keeps the old bits" is
  never a reason to keep a duplicate path, a slower kernel or a serial fold. Change every vendor column and the host
  column together, then record the new reference in the release's one verification.
- <!-- OPEN: question 4 --> **Host column.** Text A (2026-10-13): "Host digests are recorded, not required."
  Text B (2026-11-05): the release checks that "the columns match across the GPUs and the CPU".
- Never accept a vendor-specific reference column to hide a disagreement, and never regenerate expected hashes from
  a failing vendor to make it pass (Andrew, 2026-10-29). A mismatch is a bug to fix in code.
- Preserve historical tables, raw failures and provenance. Fresh reference hashes are not independent correctness
  evidence and not a speed claim (2026-10-29).
- **`fast` needs no identical anything (Andrew, 2026-10-04).** Not the same bits across vendors, runs or arms. `fast`
  is judged on speed and on quality that does not go down: no material drop against `fast` main, and at least as
  good as the best opponent. A `fast` candidate is never held for different bits or a noise-level metric change.
- **Focus on `identical`; Apple `fast` investment is stopped (Andrew, 2026-10-22).** `deterministic` is retired. No
  dedicated `fast` features, fixes, tuning, builds, validation or benchmark campaigns unless Andrew reopens it.
  Existing `fast` code may stay opt-in, Apple only; keep historical records and the rejection tests for retired
  mode values. Apple `identical` stays in the three-vendor contract. The
  current `fast` implementation is not a baseline for the cost of identity. The paper covers `identical` only.

## Release order

### Fix everything before a release (Andrew, 2026-11-08)

- We fix everything we can before a release. We do not release while there is more work to incorporate: known
  regressions, planned refactors, owed fixes and owed deletions all go in first. No piecemeal releases.
- Measure before the release: time the candidate build on NVIDIA and AMD against the last release and the paper
  baseline, fix every regression found, and measure again. Only a candidate with no open regression is frozen.
- Order: all code in, one compile, measure, fix, measure again until clean, freeze, the one verification, publish.

### Verify once, before the release, at the very end (Andrew, 2026-11-05)

- Verification happens once, before the release: freeze the source, run one global verify of every lane on every
  device from the built wheels, see the columns match, publish. Nothing is verified after a release.
- No same-bits checks during development: no identity (ID) run per lane, per fix or per merge. A plan that would run
  identity twice for one release is wrong: say so up front and fix the order or the tool.
- A cross-vendor mismatch found by the verify is a bug to fix in code; then the affected lanes run again.
- Identity runs once (`REPEATS=1`); a mismatch is never a reason to rerun unchanged code. One identity check per
  release: the table admit. No release step re-fits or re-diffs identity after it (Andrew, 2026-10-10).
- No re-verification of green work (Andrew, 2026-11-02): the gates install the exact wheel bytes and run the
  records; nothing reinstalls those bytes from PyPI to verify again, a ledger waiver never triggers a rerun, and a
  verdict is reused automatically for the same bytes.
- Verification uses bounded correctness fixtures and focused installed-wheel, loader, dependency and adapter smokes
  (2026-10-25). Do not promote a source check or a compile to runtime success, or extend a partial result to
  untested paths or devices. NVIDIA PTX verification records the actual GPU, driver and the selected PTX route.

### Reuse good old runs, automatically

<!-- OPEN: question 2 -->
- Text B, the release (Andrew, 2026-11-05; `tools/release.py`): every release runs everything once. Reuse of old
  runs is an opt-in bonus (`--reuse-unchanged`), never the default.
- Text A, the standalone record tool (Andrew, 2026-10-31): `tools/record_identity_column.sh` with no lane options
  records only the lanes whose closure changed since the newest good evidence of the column's device class and
  prints `REUSED n / RECORD m` or `NOTHING-CHANGED`; `FULL=1` is the only way to re-record unchanged lanes; a lane
  whose newest result is bad and whose closure is unchanged is `KNOWN-BAD` until the code is fixed; admission
  accepts reused older records beside new ones when each older lane is superseded or unchanged.
- In either mode a full rerun is explicit in code (a named flag), and cross-vendor agreement is checked in full.

## Measurement and fairness

- Our CPU is never raced or timed (Andrew, 2026-10-02). The CPU is for verification digests, CPU-only installs and
  inference. Race and measure tools default to our GPU arm only.
- Opponents are scored once, stored, and run only by an explicit opponent job. No opponent re-score.
- One scored run per arm, one seed (Andrew, 2026-10-03 and 2026-10-14): the harness may take one excluded warmup.
  `AB_MULTI_RUN=1` is the only override. Queue A/B lines with reps 1, rounds 1, pairs 1.
- No Apple machine is ever added, rented, extended or released without Andrew's express permission.

### Measurement process (owner, 2026-10-05)

1. Freeze one commit per A/B round. Compile it once on cheap boxes. Only a green frozen build goes to the timing
   GPUs. New code waits for the next freeze; never change binaries or settings inside an active frozen run.
2. `identical` switches are decided by NVIDIA and AMD together: the verdict is the AVERAGE of the two vendors'
   ratios against the average noise floor (Andrew, 2026-10-08). A vendor pulling the other way is flagged on the
   cell (vendor split) but does not block the average. Apple never votes and `identical` is never tuned for Apple;
   Apple `identical` hashes must still match (2026-10-13).
3. Measure `identical` on NVIDIA, AMD and Apple and update every board (main board included) as results land,
   through the board tools only. A full-board run is `identical` on the three; `fast` is not rerun.
4. Standing order: when a problem is found, fix it. Do not just comment on it or defer it.
5. Read logs with grep and short tails (see "Logs and context").
6. OPPONENT CLOCKS (Andrew, 2026-10-07/08; `tools/board_clock_audit.py`): every stored opponent cell carries a
   comparability span (input_home, inside_clock, upload_ms_untimed). Our fit clock includes the host-to-device copy
   of X. The board shows both clocks per cell where derivable (whole = kernel + copy): torch-GPU ratios
   kernel/kernel, every other opponent whole/whole, never an invented copy time. Our arm records its copy
   separately (`upload_ms_separate`); only OUR cells on the mixed lanes are re-raced.

Candidate evaluation (owner, 2026-10-06):
- Evaluate a performance candidate with end-to-end A/B runs on the full dataset of every affected estimator, with
  the whole operation inside the clock (preparation, fit, required synchronization, consumed outputs). Component
  timings, synthetic substitutes and reduced workloads are screening only: say "component screening complete".
- Map each candidate to its affected estimators and saved recipes before timing. `--rows full` alone does not prove
  the full dataset ran. Missing coverage stays pending; never substitute a smaller case and report completion.
- Do not promote a default from component gains. Record winners AND losers beside the switch and on the boards.
- Decide each candidate as soon as its evidence is complete on both vendors; do not wait for the whole campaign.
  When the evidence supports it, flip the switch promptly and put the reason INSIDE the source file beside it:
  experiment ids, cases, both vendors' ratios, identity and quality outcome, sample count, limits. A promoted change
  starts a new freeze. Do not repeat decided experiments unchanged.
- As each cell lands, keep its source, binary, hardware and harness provenance, timing and status, and update the
  boards through the board tools. Pending, failed and unqualified cells stay visibly separate. Never invent an
  opponent ratio from an own-only A/B.
- Run the vendors in parallel on separate owned boxes and cells serially within each GPU. Take identity evidence
  from the scored execution. Reuse accepted compile and identity evidence; do not repeat it to evaluate speed.

### Race fairness: what the measurements are for (Andrew, 2026-10-13)

The boards measure the cost of bitwise identity: our `identical` mode against the best real library implementation
of the same algorithm, on the same kind of hardware, with the same settings. Every supported result is shown, wins
and losses alike. THE RULE IS CODE: `tools/race_fairness.py` applies it and `tools/main_board_ingest.py` calls it on
every rebuild; the paper and mojolearn.org only read its marks.

- A race is GOOD when both arms run the same algorithm with the same settings on the same kind of box. Different
  seeds, the opponent's own fast modes, its library version or host, a tie, and anything we do only because
  identity requires it (FP32, fixed reduction order, exact kernels, our copy inside our clock) do not make it bad.
- A race is BROKEN (rerace "x", omitted from the paper and the site) when a work-changing setting differs between
  the arms without identity requiring it, or the opponent is code our harness wrote and not a library's own.
- FAIRNESS IS ABOUT THE PARAMETERS (Andrew, 2026-10-14): every work-changing parameter is set explicitly and the
  same on both arms, never left to a default. Effective settings are read back to prove it.
- QUALITY IS ALWAYS SHOWN, NEVER A GATE (Andrew, 2026-10-14): both arms' quality is shown beside every time. Quality
  judges changes to our own algorithms: speed never at the expense of quality.
- Different executed iteration counts do not make a race unfair; never pad work to equalize them.
- Fix the race code, do not drop an opponent, unless no real library implementation exists for that vendor. When
  one opponent arm is broken, exclude that arm and show the race against the rest.
- Keep measurements with sufficient evidence; exclude pairings with missing or bad evidence; do not rerun
  everything. Audits never rerun anything. No host is excluded for being slow.
- New measurements keep source bytes before execution and full logs after (`tools/bench_board_evidence.py`). A
  retention failure stops the next race; retry the upload without rerunning completed measurements.
- A new harness-written arm is recorded in the registry `HARNESS_ARMS`. Every race-code fix from an audit carries
  a `FAIRNESS FIX (<date> audit)` comment saying what was wrong before.

### Full-machine CPU race resources (owner, 2026-10-05)

- Every CPU opponent arm gets the full machine allocation: no inherited laptop caps, no
  `MOJOLEARN_BENCH_THREADS=1`. Run measurement arms serially; no competing jobs on the box.
- On rented Linux respect the actual cgroup allocation, not the host's visible CPU count. Do not claim all cores
  were busy: record the worker resource policy, thread environment and effective pools with the evidence.
- Preserve algorithm semantics and documented nested-pool settings; never drop a correctness setting to raise
  utilization.

### No dimension targeting, in any mode

No dispatch, tile, threshold, cap or route rule may key on an exact benchmark dimension, a size chosen to sit just
above or below a board row (rows, features, classes, k, vocabulary), or a board dataset name, in `fast` or
`identical`, on any vendor. A rule must come from size, hardware or cost reasoning that covers neighboring shapes,
stated in a comment. Removing such a rule may change bits: change all columns together. Each removal gets an A/B
with the old rule as the B arm, timed on neighboring shapes and one non-board dataset.

## The GPU path

- The GPU path is GPU only and parallel (Andrew, 2026-10-02). No host steps inside a GPU fit, transform or predict,
  and no serial one-thread, one-block or per-sample default. Never substitute a CPU route for a slow GPU kernel.
- Host loops on the GPU path: only driver loops over model structure, tagged `# driver-loop: over <what>`; never
  over rows, elements or a downloaded buffer (`docs/plans/GPU_PATH_HOST_LOOPS.md`, Andrew, 2026-10-17).
- When a parallel kernel needs a different fold order, change the order on every vendor and in the host column
  together. Never keep a serial chain to preserve old bits.

## No Python in the runtime

- **The rule (Andrew, 2026-10-06, restated 2026-10-19).** The only job of the Python package is to connect users to
  the compiled Mojo through PyPI. A public method is one binding call that hands over buffers and returns the
  result. Validation, fitted-state checks, argument and data preparation, binding selection, computation, training
  loops, batching, backward, optimizers, workers, synchronization, stopping decisions and output processing run in
  Mojo, in every mode, on every vendor. No arithmetic on data, no loops or comprehensions over values, no NumPy
  math in Python, not even for a handful of values. A missing value is added to the Mojo kernel, never patched in
  Python. A native call surrounded by Python work is an incomplete migration.
- **No calls back into Python (Andrew, 2026-10-20).** The rule is transitive: no Python functions, methods,
  callbacks, property or descriptor hooks, input protocols, custom trainers, schedules, batch suppliers or step
  hooks during a product operation. Holding a PythonObject in a Mojo struct does not compile its methods. Refuse
  unsupported callback or protocol objects before invoking them, with a clear native alternative.
- **Exported graph boundary (Andrew, 2026-10-27).** Never execute a user's Python `forward`, exporter or
  serialization callback inside product execution; the API may require an already exported graph.
- **Direct CPython C-API calls are allowed during execution (Andrew, 2026-10-28)**, including inside loops. A C-API
  call can invoke Python hooks: audit the operation and admitted types, keep exact-type checks, lifetimes, error
  propagation and GIL safety. Hoist invariant admission and reuse plans and buffers. Do not move algorithm work
  into provider directories to evade checks.
- **scikit-learn tags (owner, 2026-10-24).** `__sklearn_tags__` and helpers used solely to build its tag metadata
  may run in Python. Native operations must not call back into tag handling. Audit it separately from runtime
  violations. No other metadata protocol is exempt.
- <!-- OPEN: question 3 --> **Python fallback for a Mojo bug.** Text A (Andrew, 2026-10-20): "No compatibility
  branch, callback exception or Python fallback is permitted." Text B (owner exception, 2026-10-25): a narrowly
  scoped Python runtime implementation is allowed where a Mojo/Modular bug blocks a correct supported native
  implementation; document INLINE the operation, the bug or suspected failure, the Python work and the retained
  reproduction; record it in release evidence and never call it Python-free.
- Python test and benchmark controllers, test data preparation and independent oracles are testing tools. Keep
  them distinct from product execution and honest about timing boundaries. Installing a Python package is not a
  runtime repair.
- Preserve supported interfaces through native implementation; do not remove a feature to claim compliance.
  Measure the gain; do not assume removing Python makes every workload faster.
- Audit the actual public operation and its transitive callees. Checker baseline entries, including `cpu-route`
  tags, are debt to remove, never exemptions; an empty baseline is not proof. The no-host-routes push hook enforces
  the rule.

### Wait for Mojo and Modular

If Mojo or Modular does not support something yet, do not build a workaround for it; wait for their support (decided
2026-10-04; the parked AMD portable prototype is on `lane/amd-portable`). Do not hand-rewrite compiler output, patch
toolchain internals or ship unsupported build modes. Record the ask for Modular and move on. One authorized repair
(owner, 2026-10-25): the native CPython dictionary API for instance-field lookup in `get_params` and public
attribute access, recorded in `bindings/native_instance_item.mojo`; do not reopen it.

### Framework adapters are the same API in another format (Andrew, 2026-10-24)

The Torch and MAX adapters are front-ends over the same compiled kernels and compute nothing of their own.

- No identity lanes, verifier rows or sabotage arms for an adapter. If an adapter needs a value the kernels do not
  provide, add it to the Mojo kernel.
- An adapter is qualified by what is specific to it: its files build in every release leg, survive the wheel
  audits, and load and run from the INSTALLED wheel on each vendor. That smoke is part of the release gate.
- Adapter bugs are glue bugs; fix them in the adapter, never by changing kernel bits for one front-end.
- Whole-model only. The per-operation surfaces removed on 2026-10-24 (`docs/TOMBSTONES.md`) must not come back.

## Lanes, compiles and boxes

### Lanes

- **Code, then queue.** A lane writes code, commits and pushes. It never runs tests, timing, identity or Metal jobs
  on the laptop, and never ssh-es to, rents, extends or releases a box.
- <!-- OPEN: question 1 --> **Lanes and compiling.** Text A (Andrew, 2026-10-07): lanes never compile, on the laptop
  or through `lq`; the orchestrator merges finished lanes into one integration branch, compiles it once (every
  switch on and off, NVIDIA and AMD targets), fixes what breaks, then merges to main. Text B (Andrew, 2026-10-27,
  "for this migration"): any peer or subagent may rent a RunPod CPU box and compile, and completed provider repairs
  merge into main after a small successful compile.
- A lane makes its code easy to compile: read the call path and the Mojo syntax, keep each switch's code
  self-contained, follow "Mojo rules that cost a build round", and list every binding the integrator must build.
- **One worktree per lane:** `~/mojolearn-wt/<lane>` on branch `lane/<lane>` (`tools/lean_worktree.sh` when old
  evidence is not needed). Commit after every edit and push often. Never `git stash`, rebase, `reset --hard` or
  `checkout --` someone else's edits. Never mutate another agent's worktree.
- **Never `git push --no-verify`.** The pre-push host-route fence judges a branch only on what it adds over its
  merge-base with main. A push to main is judged on the whole tree; if main is red, fix main first (2026-10-07).
- **Nothing in `/private/tmp`** (wiped on reboot). Briefs, notes and scripts live in the worktree or
  `~/mojolearn-evidence/`.
- **Use bash, not zsh,** for loops and variable expansion: zsh does not word-split `$FLAGS`.
- **Final reply, short:** branch and head; one line per change with the cause as `file:line`; deletions; bit
  changes; the A/B define; what it queued (box and id) and any RUN OWED; open doubts. No narrative, no diffs.
- **Peer communication only for execution (Andrew, 2026-10-29).** Contact peers only to resolve ownership,
  coordinate a dependency, hand off required work or prevent conflicting changes; never for advice, reassurance or
  status. Put routine findings in handoff files.
- Every lane brief and spawn prompt carries this reminder, and lanes pass it on: "Keep logs out of context: save
  complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit
  status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or
  infer full success from filtered output."

### Compiles

- Compilation is never described as verified runtime behavior or bitwise identity. A compile pass on one target is
  not a claim that every target works. Development merges need a successful compile of the affected code and the
  source and push guards, not a verification campaign (2026-10-27).
- Mac compiles go through the slot semaphore: `bash tools/compile_slot.sh <command>`. The slot count and the
  laptop's one-compiler-worker rule live in that script and only there.
- For a development repair, build the smallest affected groups (`tools/development_build.py`,
  `bindings/development_groups.json`) and reuse unchanged compiled groups whose source closure, compiler, flags,
  target and dependencies match (Andrew, 2026-10-29). Separately compiled groups are valid evidence for the paths
  they actually execute. Keep standard optimized settings. Do not interrupt an active valid build. Do not start a new loader
  or build scheme to make one repair's compile smaller. Test changed loading and linking boundaries at assembly.
- On a rented box use every allocated core (from affinity or cgroup limits, not the host CPU count).

### Boxes

- `lq` (`~/mojolearn-ops/lq`, no arguments for usage) is the only way a lane reaches a box. `lq add nv|amd RACE
  <branch> <lane[,lane]> <ds[,ds]>` (one build covers every lane x dataset: batch them; RACE is for the algos
  family); `lq add <box> CMD <branch> <tag> '<command>'` (scripts must be committed in the branch); `lq results` and
  `lq log` give grep-sized output. Full-size RACE lines are for speed and quality only.
- <!-- OPEN: question 5 --> Boxes: nv is the RunPod L40S; amd is the DigitalOcean AMD GPU droplet (rule text:
  MI325X; the 2026-11-08 runs used an MI300X); apple is the M2 Pro, for same-bits work only, never timing. The M3
  Ultra belongs to the Apple FAST peer. Each box runs one job at a time; on apple one lane and one dataset per
  line. `identical` speed work targets NVIDIA and AMD.
- **Box jobs persist from the box (Andrew, 2026-10-26).** Outside `lq`, a job on a rented box starts only through
  `tools/nvidia_central.sh submit`, `tools/do_amd_steward.sh run` or `tools/dev_pod.sh run`; each wraps it in
  `tools/box_persist_run.sh`, which uploads the results dir to R2 from the box. `--no-persist` is refused unless
  `MOJOLEARN_ALLOW_UNPERSISTED=1`. Read results with `tools/box_persist.py fetch --tag <tag>`.
- **Results leave the box (Andrew, 2026-10-18).** Every run keeps its logs and results on the Mac and in R2. Key
  results go to the repo or `~/mojolearn-evidence/<run>/` as they land. `down` commands archive before the delete
  and refuse to delete a reachable box whose archive fails. A box rented any other way gets a state file so
  retention and teardown can find it (`~/mojolearn-evidence/devpods/<key>/state.env`).
- Before terminating a machine, prove where every required output, model artifact, accepted binary and unique
  input is stored, by bytes off the machine, not by directory names, mount labels, metadata or hashes.
- Before taking a box down, check who owns it. Never take down another session's box. Arm a deadman on rentals.
- **No background watchers (Andrew, 2026-10-23).** Watcher, monitoring, polling, mirror and periodic collection
  services are disabled; do not install, restart or replace them. Active build and measurement controllers are not
  watchers.

## Experiments

- Every experiment is a `-D MOJOLEARN_<AREA>_FAST_<NAME>` define, default off, measured by an A/B on the box.
- A winner (faster, quality equal) becomes the default with a `_OFF` define and a code comment citing the A/B
  numbers. A loser (slower, noise, quality or semantics change) never reaches main: delete its code from the lane
  before merging; it stays recoverable at the lane's recorded sha.
- Every experiment, kept or dropped, gets one row in `docs/apple-fast/EXPERIMENTS.md` (define, algorithm/dataset,
  branch@sha, A/B tag, before -> after ms, verdict, reason). Search it before writing a new experiment.
- Experiment code may merge to main after compile evidence on one device, with untested targets and measurements
  marked pending and unproven defaults disabled (owner, 2026-10-05).

### Deleting an experiment

**PERMANENT RECORD (Andrew, 2026-10-28): `docs/TOMBSTONES.md` entries, their patches under experiments/removed/ and the inline TOMBSTONE markers are never deleted, rewritten or squashed; deletions of experiments are only ever ADDED to `docs/TOMBSTONES.md`.**

Andrew, 2026-10-10: a deleted experiment is documented both inline and centrally. The deletion commit (or its lane) leaves all four:

1. **Inline tombstone** at every deletion site: `# TOMBSTONE: MOJOLEARN_<DEFINE> (<verdict>) deleted <date> by <sha>; code recoverable at <sha>^.` plus a `# Restore:` line naming the patch.
2. **`docs/TOMBSTONES.md` entry**: an index row and a section with what it tried, the files, the verdict with NV/AMD ratios per dataset and the quality change, run ids, the recoverable sha and the patch path.
3. **`experiments/removed/<DEFINE>.patch`**: the reverse of the deletion restricted to code files (`git diff <sha> <sha>^ -- <files>`), so `git apply` restores the code. Code that never reached main gets no patch; record the lane branch@sha instead.
4. **Refusal and records**: a `removed` refusal in the define's family module `core/experiment_guards_<family>.mojo` (`core/six_lane_experiment_guards.mojo` only for guards shared across families; see `docs/build/HOT_GROUPS.md`), the entry moved to `removed` in its `grid_controls/*.json`, and the `docs/apple-fast/EXPERIMENTS.md` row.

## Mojo rules that cost a build round (Andrew, 2026-11-09)

Each of these compiled badly or hung during the pre-0.8.67 integration. Follow
them when writing Mojo; lanes do not compile, so the compiler will not catch
them for you.

- Never name a parameter `out`. It is a reserved word (an argument
  convention), and the compiler reports "expected argument name". Use `dst`
  or `y_out`.
- A function launched as a kernel must not also be called from another
  kernel. The build fails with "compile offload result missing kernelId" and
  no source location. Put the body in an `@always_inline` helper and have
  both kernels call it.
- No mutual recursion across admission helpers (or any helpers on the host
  bindings' import path). The compiler deadlocks at 0% CPU instead of
  reporting it.
- Hold a device buffer or view in a `var` before taking a pointer for a
  kernel. A pointer from a temporary or an immutable borrow is not mutable
  and the launch is refused.
- No call as a default argument value (for example
  `binding: PythonObject = PythonObject(None)`). Defaults are evaluated at
  compile time. Pass the value at the call site.

## Logs and context

- Save complete build, test and GPU logs to files. Read the exit status and the structured summary first. Never
  print a whole log, race output, build output or large JSON report into context.
- Use targeted `rg` (or `grep`) with bounded context and short tails, for example
  `rg -n -i -m 20 -C 3 --max-columns 240 --max-columns-preview 'error|fail|refused|divergent|summary' run.log` or
  `tail -n 20 run.log`. Narrow to the relevant files first. Parse large JSON for selected fields.
- For a build read only its exit code and the first error:
  `... > build.log 2>&1; echo rc=$?; grep -m 5 -B 2 -A 8 'error' build.log`.
- For a race directory read the one-line summary (`ALGOS lane=... status=... median_ms=...`), not the race log.
- A filtered excerpt or a few passing lines never proves the whole run passed: keep the original exit status,
  expected coverage, failures and skipped or incomplete work, and expand the relevant block when needed.
- Never read a whole subagent transcript or a `tasks/*.output` file; ask for a concise status or read the artifact.
- Put long results in a file under `~/mojolearn-evidence/` and report the path plus a few lines. Keep a routine
  diagnostic reply to about 2,000 tokens.

## Public docs and docstrings (Andrew, 2026-10-12, additions 2026-10-13)

- Docstrings of public names are user documentation (docs.mojolearn.org): what it does, parameters, returns, a
  short example. No status notes, internal file paths, dates, lane or run names, or all-caps asides. Engineering
  notes belong in `#` comments next to the code.
- Public docs state the contract in one sentence, verbatim: "In `identical` mode, Mojolearn returns the same bits on Apple,
  NVIDIA and AMD GPUs." Every `numeric_mode` doc that lists `'fast'` says Apple only. Docs never offer `fast` on
  NVIDIA, AMD or the CPU.
- Omit, never caveat. Something broken, unverified or unfinished is not documented: leave it out of docs-site and
  list it in `docs-site/api_exclude.txt` (name and reason). No "Known defect", "NOT IMPLEMENTED" or status notes;
  write `identical` and `fast` in lower case. Excluded names never appear in public docs, the README or the site.
- The 162M two-route model is research and marketing material; it never appears in docs-site or the library docs.
- We are alpha: no enterprise-grade checks. The docs, mojolearn.org and the measurements are generated from the
  latest origin/main by one publish command, run only when Andrew says so. Old URLs may break; no redirects.
