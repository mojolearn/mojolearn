# Agent working rules

This is the one rule file for every agent in this repo (Codex, Claude, their lanes and subagents). `CLAUDE.md`
imports it and adds only what is specific to Claude sessions.

- One rule per topic, current text only. When a rule changes, replace it here; do not append a second version.
- The later dated owner statement wins unless it is narrower in scope. Older wording, rationale and finished
  campaigns are kept verbatim in [docs/rules/RULES_HISTORY.md](docs/rules/RULES_HISTORY.md); it is history, not rules.
  How this file was consolidated: [docs/rules/RULES_CONSOLIDATION_2026-11-09.md](docs/rules/RULES_CONSOLIDATION_2026-11-09.md).

## The name is Mojolearn, with a capital M (Andrew, 2026-11-02)

- In prose (docs, docstrings, README, site, paper, emails, release notes, commit messages and replies) write
  **Mojolearn**: capital M, the rest lower case. Never "mojolearn", "MojoLearn" or "MOJOLEARN" as the name.
- Code stays lower case: the package, module and command names (`pip install mojolearn`, `import mojolearn`,
  `python -m mojolearn verify`), paths, URLs and domains (mojolearn.org). Put them in backticks in prose.

## Hard rules (Andrew)

Copied verbatim, with their dates, from Andrew's own rules so that every agent has them. They override anything
weaker elsewhere. On a rented box the core count is the allocation (affinity or cgroup limits), not the host's
visible CPUs.

### HARD RULE: no Apple machines, hours or calls without express permission (2026-09-27)

NEVER add, allocate, rent, launch, extend, keep running past its paid
minimum, or make provider API calls to create any Apple machine: AWS EC2 Mac
dedicated hosts, MacStadium, Scaleway, or anything similar. Only Andrew's
express, specific permission allows it.
- No agent may be given that ability.
- Each EC2 Mac host bills a 24-hour minimum and keeps billing until released.
- If Apple capacity is a bottleneck, report it and ask. Never act.

### HARD RULE: use 100 percent of rented CPU (2026-11-05)

Rented hosts and pods are paid for whole. Use all of their cores and memory: every compile, simulation, synthesis
and test on a rented host runs at the host's full core count, limited only by the memory the jobs actually need.
Never pass a serial or reduced job count (`--compile-jobs 1`, a fixed `CORES`, `NUM_CORES`, `JOBS` or thread cap
below the host's vCPUs) to a rented host unless a measured memory limit forces it, and then say so in the report.
Never carry such a limit forward after the reason for it is fixed. A host sitting mostly idle is a defect to fix in
code or in the brief, never an accepted state. The laptop's one-core rule for simulations is unchanged.

### HARD RULE: write code to run in parallel (2026-11-05)

Setting a job count to the core count does nothing for a program that is serial inside. All code written or
changed from now on (compilers, converters, simulation drivers, test runners, study scripts) splits its work into
independent units and runs them across every core, with the worker count taken from the host's cores and the
measured memory per worker. A new serial loop over independent units is a defect. Where a stage is serial by
nature or is a third-party tool that cannot be changed, never rent cores for it: use the smallest host that holds
its memory, or pack independent variants onto one host up to its memory. Within ten minutes of a job starting,
check that the host's busy cores match the plan; a host at one busy core is reported and fixed, not waited on.
Every report on a rented host states busy cores of total.

### HARD RULE: parallelize tests and experiments by default (2026-11-07)

Every test suite, experiment sweep, conversion, compile stage, package load, dump and replay
splits its independent units (models, batches, bodies, layers, experts, prompts, variants,
test files) across all cores of the host from the first version, with the worker count from
nproc and the measured memory per unit. Serial is never the default and never "for now". A
stage that cannot be split says why in its docstring and gets the smallest host that holds
its memory. Every brief states the parallel plan before launch; every report states busy
cores of total; a host at one busy core for ten minutes is fixed or gets a fix lane the same
session, never waited on. This applies to every project, not only Mambik.

### HARD RULE: remove what a change made unnecessary (2026-11-09)

Whenever a change moves a decision out of a mechanism (into a compiler, a scheduler, a
controller, a config, a build step), the same change removes the mechanism that used to make
that decision at run time: the registers, muxes, enables, counters, queues, flags, code paths
and tests that existed only for it. Keep the old form behind a default-on switch only until the
new form is measured, then delete it. Every report states what was removed, or "nothing
identified, because ...". Periodically walk the live design or codebase for mechanisms whose
decision is already known elsewhere, list them with their cost, and open work to remove the top
ones. Pricing a removal and not doing it is a defect; each priced removal is implemented or
retired with a reason within two weeks. Applies to hardware, firmware and software alike.

## Identity and bits

- **The contract (Andrew, 2026-10-13).** In `identical` mode the same bits come out on Apple, NVIDIA and AMD GPUs
  within one version. Identity is decided by the output hash each run already stores.
- **Bits may change between versions (Andrew, 2026-11-05, restated 2026-11-08).** Matching across GPU vendors within
  one version is the only bits requirement. Never hold a performance fix, a correctness fix, a smaller fixture, a
  better fold order, a refactor or a cleanup because it changes an older version's bits. "Keeps the old bits" is
  never a reason to keep a duplicate path, a slower kernel or a serial fold. Change every vendor column and the host
  column together, then record the new reference in the release's one verification.
- **The CPU column is required (Andrew, 2026-11-09).** It is recorded first, on GitHub runners (free), and the GPU
  columns are checked against it. The public contract sentence stays about the three GPU vendors.
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

### A measured winner is never deleted on reasoning (Andrew, 2026-11-09)

- Code that won a measurement (a hand-tuned kernel, a fused path, a switch whose
  A/B it won) is deleted only after a measurement on NVIDIA and AMD shows its
  replacement is at least as fast with quality equal. "The new path subsumes it"
  is a claim to measure, never a reason to delete.
- A refactor that moves a family onto a shared runtime keeps the tuned kernel as
  an op of that runtime (one runtime path, tuned kernels inside it). It does not
  replace the kernel with the generic op.
- Hand-tuned kernels are not touched by a refactor. What a refactor may change
  is the plumbing around them: per-call weight uploads, per-token doors, host
  reads per iteration, duplicate forwards for train and predict.
- Until the measurement exists, the old code stays behind its switch, default
  unchanged. This narrows "remove what a change made unnecessary": unnecessary
  means measured, not argued.

## Fix everything before a release (Andrew, 2026-11-08)

- We fix everything we can before a release. We do not release while there is more work to incorporate: known
  regressions, planned refactors, owed fixes and owed deletions all go in first. No piecemeal releases.
- Measure before the release: time the measurement build on NVIDIA and AMD against the last release and the paper
  baseline, fix every regression found, and measure again. Only a candidate with no open regression is frozen.
- Order: all code in, one compile, measure, fix, measure again until clean, freeze, the one verification, publish.
- A "measurement build" (`tools/release.py candidate`) is wheels built from a commit to measure them. It is NOT a
  release candidate: it never touches main, tags or PyPI, and no release happens until everything is fixed and
  measured clean (Andrew, 2026-11-09).

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

### Reuse of old runs is opt-in (was: Reuse good old runs, automatically)

- Reuse of old identity records is opt-in everywhere (Andrew, 2026-11-05, decided 2026-11-09): every release
  records every lane on every device once, and a reused older record is accepted only when the operator passes the
  explicit reuse flag (`tools/release.py --reuse-unchanged`). `tools/record_identity_column.sh` must default to a
  full record with reuse behind a flag.
- Cross-vendor agreement is always checked in full. A lane whose newest result is bad is fixed in code, then
  recorded.
- Andrew says when it is time to release (2026-11-09). No agent starts, schedules or proposes a release
  date. 0.8.67 is not rushed: the fixes matter most. The Python-at-the-boundary rollout
  (`docs/plans/PYTHON_AT_THE_BOUNDARY.md`: Python objects only in the unpack and pack files of each
  public call, Mojo values everywhere else) is part of 0.8.67 and gates it.

## Measurement and fairness

- Our CPU is never raced or timed (Andrew, 2026-10-02). The CPU is for verification digests, CPU-only installs and
  inference. Race and measure tools default to our GPU arm only.
- Opponents are scored once, stored, and run only by an explicit opponent job. No opponent re-score.
- One scored run per arm, one seed (Andrew, 2026-10-03 and 2026-10-14): the harness may take one excluded warmup.
  `AB_MULTI_RUN=1` is the only override. Queue A/B lines with reps 1, rounds 1, pairs 1.

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
- **Python objects only at the public boundary (Andrew, 2026-11-10).** Python objects, Python library imports and
  CPython interaction belong only in the actual UNPACK/PACK functions of each public operation. RUN uses typed
  Mojo values for requests, options, dimensions, model and optimizer state, buffers and results, and native calls
  between compiled modules. A Mojo wrapper around a PythonObject is not a migration. Preserve the public API and
  borrowed-buffer lifetimes through explicit native ownership; do not hide Python work in provider directories.
  Direct C-API calls within UNPACK/PACK may still invoke Python hooks: retain exact-type admission, error
  propagation and GIL safety, and hoist invariant admission. The 2026-10-28 permission for native C-API calls does
  not authorize Python objects or Python support inside RUN.
- **scikit-learn tags (owner, 2026-10-24).** `__sklearn_tags__` and helpers used solely to build its tag metadata
  may run in Python. Native operations must not call back into tag handling. Audit it separately from runtime
  violations. No other metadata protocol is exempt.
- **Python fallbacks for a Mojo or Modular bug are a frozen list (Andrew, 2026-11-09).**
  `python/mojolearn/_runtime_fallbacks.py` is the COMPLETE list of authorized fallbacks (empty today).
  Nothing is added without Andrew's explicit word. An agent that believes it has found a Mojo bug stops, writes
  the smallest reproducer and the ask for Modular, and reports; it does not write Python. Each entry keeps its reproduction and upstream ask; a native fix removes the entry. A path that
  uses an entry is recorded in release evidence and is never called Python-free.
- **Move native correctness tests to Mojo (Andrew, 2026-11-10).** Kernel, numerical and native state/lifetime tests
  belong in Mojo. Retain Python tests for the actual Python API boundary and framework interoperability, including
  buffer admission, exceptions, returned objects and refusal of callbacks. Python test/benchmark controllers and
  boundary fixtures or independent external oracles are testing tools, never product execution or a fallback.
  Keep timing boundaries honest. Installing a Python package is not a runtime repair.
- Preserve supported interfaces through native implementation; do not remove a feature to claim compliance.
  Measure the gain; do not assume removing Python makes every workload faster.
- Audit the actual public operation and its transitive callees. Checker baseline entries, including `cpu-route`
  tags, are evidence to review, never exemptions; an empty baseline is not proof. Native execution and device
  behavior establish compliance; source-graph reachability does not distinguish admission from GPU computation.
- **The Python source guards are retired (Andrew, 2026-11-10).** Remove the kernel-path/token guard,
  shell-loop counter, GPU-route source-graph fence, runtime-import counter and public-runtime source-count gate
  from push and CI. Do not restore them under another name
  or replace them with source-count gates. They excluded binding dependence or rejected required UNPACK/PACK
  work, including native host input conversion and reading mutable admitted module metadata.
  Their counts never proved a Python-free operation. Complete the typed Mojo boundary migration and retain
  actual runtime correctness checks.

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
- **Only the orchestrator compiles (Andrew, 2026-10-07, decided 2026-11-09),** plus ONE designated integration
  compile-fix lane per integration branch that the orchestrator names in its brief. No other lane compiles, on
  the laptop or through `lq`. The orchestrator merges finished lanes into one integration branch, compiles it once
  (every switch on and off, NVIDIA and AMD targets), fixes what breaks, then merges to main. Lanes never touch main.
- A lane makes its code easy to compile: read the call path and the Mojo syntax, keep each switch's code
  self-contained, follow "Mojo rules that cost a build round", and list every binding the integrator must build.
- **One worktree per lane:** `~/mojolearn-wt/<lane>` on branch `lane/<lane>` (`tools/lean_worktree.sh` when old
  evidence is not needed). Commit after every edit and push often. Never `git stash`, rebase, `reset --hard` or
  `checkout --` someone else's edits. Never mutate another agent's worktree.
- **Never `git push --no-verify`.** Run the remaining push hooks normally. The retired runtime source guards
  must not be restored; native correctness and GPU behavior remain requirements.
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

### Boxes

- `lq` (`~/mojolearn-ops/lq`, no arguments for usage) is the only way a lane reaches a box. `lq add nv|amd RACE
  <branch> <lane[,lane]> <ds[,ds]>` (one build covers every lane x dataset: batch them; RACE is for the algos
  family); `lq add <box> CMD <branch> <tag> '<command>'` (scripts must be committed in the branch); `lq results` and
  `lq log` give grep-sized output. Full-size RACE lines are for speed and quality only.
- Boxes: nv is the RunPod L40S. amd is the DigitalOcean AMD GPU droplet: the MI300X (the paper's GPU) is the
  default, and any other model is flagged as different hardware in the results. apple is the M2 Pro, for
  same-bits work only, never timing. The M3 Ultra belongs to the Apple FAST peer. Each box runs one job at a
  time; on apple one lane and one dataset per line. `identical` speed work targets NVIDIA and AMD.
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
- Register a Python type for a Mojo struct in ONE binding only. A second binding that calls
  `add_type` for the same struct aborts `import mojolearn` ("Error building multiple Python type
  objects"). The other bindings import the type from its owner (`bindings/native_identity_type.mojo`).
- Mojo frees a local at its last named use. A device buffer, ring or list that a view, a pointer or a
  queued launch still reads must be named again after that use (keep it alive past the final copy),
  or the later read hits freed storage.
- Development integration merges require compilation of the affected code (Andrew, 2026-11-10).
  Do not add packed-wheel import or smoke gates to that merge. Report native correctness tests,
  installed runtime tests, measurements and release identity separately; compilation proves none of them.
- Compile repairs from the native migration (2026-11-10): `Copyable` alone does not allow an implicit
  copy; use `.copy()` for borrowed descriptors. Construct floating Boolean fixtures with `1.0 if
  condition else 0.0`. Copy a string slice into a temporary before assigning it back to the source.
- Mojo 1.0's custom `sort` comparator is a local `@parameter` closure, passed with `cmp_fn` over a
  mutable `Span`; a top-level thin function does not satisfy its capturing comparator signature.
  Import it from `std.builtin.sort`. In contrast, `host_parallelize` takes a runtime closure with
  explicit captures (`def task(unit: Int) {imm pointer, imm count}:`), not an `@parameter` worker.
- `match` is a reserved keyword, including for local variables; use `matched` for a Boolean result.
  Put `comptime assert` inside a function body, never at module scope.
- Generic struct parameters used by static methods or initializers must be qualified as
  `Self.E`, `Self.D`, `Self.RES` (and so on), including arguments to generic callees.
- A Python-registered native owner with non-`Writable` state implements both `write_to` and
  `write_repr_to`. Otherwise the binding derives repr through the state fields and fails compilation.
- Construct a native list with a typed literal (`var values: List[Float32] = [1.0, 2.0]`),
  keyword length/fill, or explicit appends. Positional `List[T](a, b, ...)` is not supported.
  A one-element native shape uses `List[Int](length=1, fill=n)`; `List[Int](n)` is not a
  one-element list. Never pass the compiler-internal `__list_literal__` keyword directly.

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
