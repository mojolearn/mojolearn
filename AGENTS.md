# Agent working rules

This is the one rule file for every agent in this repo (Codex, Claude, their lanes and subagents). `CLAUDE.md`
imports it and adds only what is specific to Claude sessions.

- One rule per topic, current text only. When a rule changes, replace it here; do not append a second version.
- The later dated owner statement wins unless it is narrower in scope. Older wording, rationale and finished
  campaigns are kept verbatim in [docs/rules/RULES_HISTORY.md](docs/rules/RULES_HISTORY.md); it is history, not rules.
  How this file was consolidated: [docs/rules/RULES_CONSOLIDATION_2026-11-09.md](docs/rules/RULES_CONSOLIDATION_2026-11-09.md).

The file is in eight parts: hard rules; the product (what Mojolearn is and never trades away); measurement; releases;
how we work (lanes, compiles, boxes, logs, experiments); Mojo rules; public docs; and, last, TEMPORARY DIRECTIONS.
Parts 1 to 7 are standing rules. Part 8 holds what Andrew has ordered for now only: read it first, because while a
direction there is live it overrides the standing rule it names, and it is deleted (moved to RULES_HISTORY) when
Andrew ends it. Nothing temporary is ever written into parts 1 to 7.

## 1. Hard rules (Andrew)

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

## 2. The product

### The name is Mojolearn, with a capital M (Andrew, 2026-11-02)

- In prose (docs, docstrings, README, site, paper, emails, release notes, commit messages and replies) write
  **Mojolearn**: capital M, the rest lower case. Never "mojolearn", "MojoLearn" or "MOJOLEARN" as the name.
- Code stays lower case: the package, module and command names (`pip install mojolearn`, `import mojolearn`,
  `python -m mojolearn verify`), paths, URLs and domains (mojolearn.org). Put them in backticks in prose.

### Identity and bits

- **The contract (Andrew, 2026-10-13).** In `identical` mode the same bits come out on Apple, NVIDIA and AMD GPUs
  within one version. Identity is decided by the output hash each run already stores.
- **Bits may change between versions (Andrew, 2026-11-05, restated 2026-11-08 and 2026-11-12).** Bits must be
  consistent across VENDORS within one version. That is the whole bits rule: nothing is ever held, compared or
  re-verified for matching an older version's bits. Never hold a performance fix, a correctness fix, a smaller fixture, a
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
- **`fast` is reopened, on Apple, NVIDIA and AMD (Andrew, 2026-11-19).** `fast` is `identical` with named relaxations
  switched on, never a separate implementation: each kernel is written once against the identical contract, and each
  relaxation (free reduction order, fused multiply-add allowed, reduced-precision matrix units where quality holds,
  elided waits, tile and block shape tuned per hardware) is one experiment define, measured for speed and checked
  for quality, which must not drop. `fast` and `identical` are both measured on NVIDIA first (the L40S); AMD and Apple
  ride along and are tuned later if need be. The work order is the mode policy table
  (`docs/plans/SPEED_AND_CLEANUP_2026-11-18.md`; lane `ideas-mode-policy`). An `identical` build is never changed by a
  relaxation: a row switched off in an `identical` build is a compile error. `deterministic` stays retired; keep the
  rejection tests for retired mode values. The paper covers `identical` only, and uses the table to state what
  identity costs kernel by kernel. Public docs keep saying `fast` is Apple only until `fast` has been built, checked
  for quality and measured on a vendor; the same change that qualifies it there updates the docs (part 7).

### The GPU path

- The GPU path is GPU only and parallel (Andrew, 2026-10-02). No host steps inside a GPU fit, transform or predict,
  and no serial one-thread, one-block or per-sample default. Never substitute a CPU route for a slow GPU kernel.
- Host loops on the GPU path: only driver loops over model structure, tagged `# driver-loop: over <what>`; never
  over rows, elements or a downloaded buffer (`docs/plans/GPU_PATH_HOST_LOOPS.md`, Andrew, 2026-10-17).
- When a parallel kernel needs a different fold order, change the order on every vendor and in the host column
  together. Never keep a serial chain to preserve old bits.

### No Python in the runtime

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

#### Wait for Mojo and Modular

If Mojo or Modular does not support something yet, do not build a workaround for it; wait for their support (decided
2026-10-04; the parked AMD portable prototype is on `lane/amd-portable`). Do not hand-rewrite compiler output, patch
toolchain internals or ship unsupported build modes. Record the ask for Modular and move on. The authorized narrow
exceptions are:

- (Owner, 2026-10-25) The native CPython dictionary API for instance-field lookup in `get_params` and public
  attribute access, recorded in `bindings/native_instance_item.mojo`; do not reopen it.
- (Owner, 2026-11-10) Replace the IPCA tests' nested native buffer list with six named Mojo buffer fields to avoid
  the confirmed compiler crash. This exception covers test storage only: preserve the numerical implementation,
  input/reference bytes, assertions, tolerances and kernel calls. Retain the original standalone reproducer and
  upstream report; it does not authorize other compiler workarounds or Python fallbacks.

#### Framework adapters are the same API in another format (Andrew, 2026-10-24)

The Torch and MAX adapters are front-ends over the same compiled kernels and compute nothing of their own.

- No identity lanes, verifier rows or sabotage arms for an adapter. If an adapter needs a value the kernels do not
  provide, add it to the Mojo kernel.
- An adapter is qualified by what is specific to it: its files build in every release leg, survive the wheel
  audits, and load and run from the INSTALLED wheel on each vendor. That smoke is part of the release gate.
- Adapter bugs are glue bugs; fix them in the adapter, never by changing kernel bits for one front-end.
- Whole-model only. The per-operation surfaces removed on 2026-10-24 (`docs/TOMBSTONES.md`) must not come back.

### The verify command is protected (Andrew, 2026-11-11)

`python -m mojolearn verify` and everything it needs (the verification modules shipped in the wheel, their fixtures,
records, ledger and the commands they run) is kept working at all times. Preserving verify trumps every other rule
in this file: no Python removal, boundary migration, dead-code deletion, wheel slimming, test migration, refactor or
cleanup may remove, move out of the wheel, weaken or break any part of it. Verification tooling is a testing tool,
not product execution ("No Python in the runtime"), so its Python is never a violation of that rule. A change that
touches anything verify imports says so in its report and states how verify was exercised afterwards.

**Verify follows the product; it never steers it (Andrew, 2026-11-15).** The product is decided by the best
performance at equal quality, with the same bits across vendors. What is protected is that verify WORKS and can still
fail: it runs from the built wheel, covers every lane, and catches a real difference. Its current internals,
fixtures, references, tools and the product paths it happens to call are not protected from change. When a product
change makes any of them stale (a faster default, a removed path or switch, changed bits, a renamed entry), the same
change updates verify to test the new product, and says so. Never keep a slower path, a duplicate, an environment
switch, an old route or old bits because verify or a verify tool uses them, and never shape a fix around what verify
currently does. "Protected" forbids leaving verify broken, weakened or unable to fail; it does not make verify a
reason to hold the product back. The verify smoke (`tools/ops/wheel_verify_smoke.sh`) is how a change shows verify
still works.

### No refactor adds cost on a hot path (Andrew, 2026-11-11)

A change that removes a Python crossing, a copy, a lookup or a layer is judged on the whole path it touches, never
on the count it lowers. It never adds, anywhere a loop or a repeated call can reach: a slot or name lookup that the
old code resolved once; a lock held where the old code released it (or released where the old code read Python
objects); a heap allocation, string build or struct copy per call; a copy of a buffer the old path passed by
pointer; a probe, refusal check or publication done twice; a default-route change; or a second path beside the old
one. Resolve once at import or construction, call through a pointer, pass buffers by view. Every brief for such a
change carries this rule and the lane's report lists, per converted site, the loop it sits in and what the old code
did there. An orchestrator audits every wave against these patterns before the integration compile and fixes what it
finds in the same round; a wave without that audit does not merge.

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

## 3. Measurement

- Our CPU is never raced or timed (Andrew, 2026-10-02). The CPU is for verification digests, CPU-only installs and
  inference. Race and measure tools default to our GPU arm only.
- Opponents are scored once, stored, and run only by an explicit opponent job. No opponent re-score.
- One scored run per arm, one seed (Andrew, 2026-10-03 and 2026-10-14): the harness may take one excluded warmup.
  `AB_MULTI_RUN=1` is the only override. Queue A/B lines with reps 1, rounds 1, pairs 1.
- **Small rows are the median of consecutive calls inside the one scored run (Andrew, 2026-11-15; reworded 2026-11-16).** A row
  whose first call AFTER the warmup takes under 10 ms is scored as the median of N consecutive calls (N = 20) made inside that one
  scored run (that call plus 19 more); the cell records N, the spread and which rule scored it. The warmup itself never decides: it
  carries start-up and compilation (a 6 ms row had a 1,051 ms warmup and was scored from one call). A compare never divides a
  median by a single call without flagging the row (SCORING-MIXED). One scored run per arm is unchanged; nothing is rerun; big rows keep one call. Reason: a single call of a
  millisecond-scale row is dominated by first-touch page faults and the allocator (program-run 3.4 ms once vs 0.76 ms steady).

### Times are compared only on the same box (Andrew, 2026-11-14)

- A regression verdict compares two timings taken on the SAME machine in the SAME job: same host (host id, CPU model,
  NUMA layout, PCIe link) and the same GPU. Times from different hosts are never compared, even when the GPU model is
  the same: on 2026-11-14 two L40S hosts differed 1.2-2.2x on unchanged code and produced 56 false regression flags.
- Every paper or full-board job therefore re-times OUR prior wheels (the paper baseline 0.8.55 and the last release,
  installed from PyPI, our arms only) on the same box in the same job, and the compare's columns are those on-box
  timings. A stored column from another run is reference only and carries the host it was taken on; it never makes a
  REGRESSION or FASTER flag. Opponents are still never rerun: their stored cells keep their host and are shown as such.
- The job records a host probe in `job/box.txt` (host id, CPU model, NUMA nodes, cgroup cores, host memcpy GB/s,
  host-to-device GB/s pageable and pinned, PCIe generation and width under load) and the compare prints both
  arms' probes beside the verdict. A verdict without the probe is not a verdict.
- The measurement box is the NVIDIA L40S (RunPod) until Andrew says otherwise; AMD is the MI300X only.

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
- **What a measurement keeps (Andrew, 2026-11-20).** A race that scores an opponent keeps source bytes before
  execution and full logs after (`tools/bench_board_evidence.py`); a retention failure stops the next race; retry the
  upload without rerunning completed measurements. A job with only our own arms keeps the measurement and nothing
  bulky: each row's times, quality, settings and flags, the box probe, the busy-cores record, the commit and wheel
  identity, and a short log per row. It writes no source archive (the commit is the record of the source), no fitted
  model, input or output arrays after quality and hash are computed, and no binaries; a failed row keeps its full
  log. The same holds on the Mac and in R2: results are kept, archives of what git or a rebuild already holds are not.
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
- **No knowledge of a board dataset in the product (Andrew, 2026-11-15).** No algorithm, kernel, default, threshold or
  route is written, chosen or tuned from what we know about a benchmark dataset: its name, its rows, columns, classes
  or features, its value distribution, sparsity, label balance, duplicates or ordering, or the parameters a board row
  happens to use. A change is justified from the algorithm, the hardware and the sizes of the input in front of it,
  so that it holds for data we have never seen. A fix found on a board row states the general cause and names a
  neighboring shape and a non-board dataset on which it is timed. "It cannot happen on the board" is never a reason.

## 4. Releases

### Fix everything before a release (Andrew, 2026-11-08)

- We fix everything we can before a release. We do not release while there is more work to incorporate: known
  regressions, planned refactors, owed fixes and owed deletions all go in first. No piecemeal releases.
- When to measure (Andrew, 2026-11-12): measure BEFORE the release only when the release's purpose is improving
  speed; then time the measurement build on NVIDIA and AMD against the last release and the paper baseline, fix
  every regression found, and measure again before freezing. Otherwise measure AFTER the release, on the
  published PyPI wheel, and those numbers are what the boards, the paper and the site broadcast. A claim about
  speed is only ever made from a published wheel's measurement.
- Order: all code in, one compile, (measure, fix, measure again when the release is about speed), freeze, the one
  cross-vendor verification, publish, then measure the published wheel.

### Fix everything first, then one compile, then one measurement (Andrew, 2026-11-13: "fix fix fix")

- A measurement is taken only when nothing known is left to fix. Before any box is rented or any GitHub build is
  started for measurement, the orchestrator lists every open fix (FOLLOWUPS, lane reports, audit findings, owed
  deletions) and either lands it in the same integration branch or writes, beside the list, why it cannot be
  known without the measurement. A fix that lands within the hour after a measurement starts is evidence the
  list was not read.
- Work moves in waves: fix wave (every lane the findings call for, at once) -> one integration branch -> one
  GitHub build -> one measurement on NVIDIA and AMD together -> read every flag -> the next fix wave. Never a
  box run for one lane, one row or one family; never a second measurement of the same sha for rows that a later
  sha changes. Between waves the boxes come down.
- The only measurement that may run with known fixes outstanding is a discovery screen whose purpose is to find
  unknown regressions (the full-board run against best-ever), and only when the outstanding fixes do not touch
  the rows it is for; its result opens the next fix wave, not a release decision.
- Owed A/B arms are built and timed with the wave's sha, never a sha behind it, unless the switch they decide is
  untouched by the later sha (say so in the owed row).
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

## 5. How we work

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
  changes (and anything kept serial or duplicated to avoid one: there should be none); the A/B define; what it queued (box and id) and any RUN OWED; open doubts. No narrative, no diffs.
- **Peer communication only for execution (Andrew, 2026-10-29).** Contact peers only to resolve ownership,
  coordinate a dependency, hand off required work or prevent conflicting changes; never for advice, reassurance or
  status. Put routine findings in handoff files.
- Every lane brief and spawn prompt carries this reminder, and lanes pass it on: "Keep logs out of context: save
  complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit
  status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or
  infer full success from filtered output."
- Every lane brief and spawn prompt also carries the bits reminder (Andrew, 2026-11-15): "Bits can change. They must
  match across vendors and the host column within one version, never against older code. Take the faster or parallel
  form, change every column together, and report 'bits change'." A lane that reports "bits unchanged" names what it
  kept serial, slower or duplicated to get that; the orchestrator opens a lane for each such item the same session.

### Compiles

- Compilation is never described as verified runtime behavior or bitwise identity. A compile pass on one target is
  not a claim that every target works. Development merges need a successful compile of the affected code and the
  source and push guards, not a verification campaign (2026-10-27).
- **The integration compile is the GitHub measurement build (Andrew, 2026-11-13).** An integration branch is
  compiled once by `tools/release.py candidate` on the free GitHub runners (Linux CPU, sm_80, sm_89, gfx942: the
  targets that are measured), which is also the measurement build of that sha. No Mac compile round per
  integration: the laptop builds Apple identical and CPU only, builds no GPU target, and costs an hour of the
  laptop per round. The Mac compiles once, at the release, for the Apple wheel and the Apple column of the one
  verification. Per integration the checks are: the GitHub build, the box's start-up smoke, the measurement.
  No identity verify before the release's one verification.
- Mac compiles (the release's Apple build) go through the slot semaphore: `bash tools/compile_slot.sh <command>`.
  The slot count and the laptop's one-compiler-worker rule live in that script and only there.
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

### Logs and context

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
- Pipe command output through `head`, `tail` or `grep` to reduce the result size (Andrew, 2026-11-15). Never `cat` a
  large file into context; read a bounded range instead (the Read tool with offset and limit, or `sed -n a,bp`).
- Put long results in a file under `~/mojolearn-evidence/` and report the path plus a few lines. Keep a routine
  diagnostic reply to about 2,000 tokens.

### Experiments

- Every experiment is a `-D MOJOLEARN_<AREA>_FAST_<NAME>` define, default off, measured by an A/B on the box.
- A winner (faster, quality equal) becomes the default with a `_OFF` define and a code comment citing the A/B
  numbers. A loser (slower, noise, quality or semantics change) never reaches main: delete its code from the lane
  before merging; it stays recoverable at the lane's recorded sha.
- Every experiment, kept or dropped, gets one row in `docs/apple-fast/EXPERIMENTS.md` (define, algorithm/dataset,
  branch@sha, A/B tag, before -> after ms, verdict, reason). Search it before writing a new experiment.
- Experiment code may merge to main after compile evidence on one device, with untested targets and measurements
  marked pending and unproven defaults disabled (owner, 2026-10-05).
- **An unmeasured arm is deleted (Andrew, 2026-11-15).** An opt-in arm, or an old path kept beside a default, that has no A/B
  recorded in `docs/apple-fast/EXPERIMENTS.md` within four weeks of merging is deleted with the four tombstone parts (recoverable
  at its sha). The default path stays byte-identical. An arm recorded as promoted but still opt-in in the code becomes the
  default (bits change on every column together) and its switch goes. The 1,300 undecided defines of
  `docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md` go under this rule.
- **A paused programme's arms are not unmeasured (Andrew, 2026-11-15).** The four-week clock does not run for a
  programme Andrew has paused: while Apple `fast` investment is stopped, its opt-in arms and its never-run experiment
  cards stay in the code, opt-in and default off, to be run when he reopens it. Deleting on a calendar rule is for
  arms that could have been run and were not; nothing is classed as waste because its programme is paused.

#### Deleting an experiment

**PERMANENT RECORD (Andrew, 2026-10-28): `docs/TOMBSTONES.md` entries, their patches under experiments/removed/ and the inline TOMBSTONE markers are never deleted, rewritten or squashed; deletions of experiments are only ever ADDED to `docs/TOMBSTONES.md`.**

**Protected on disk too (Andrew, 2026-11-14): `experiments/removed/`, `docs/TOMBSTONES.md` and `docs/apple-fast/EXPERIMENTS.md` are never pruned, trimmed, moved out of the repo or excluded from a wheel-size, disk-space or dead-code cleanup, whatever their size; a cleanup lane lists them as kept. A decided A/B whose losing arm (or superseded old path) is still in the code is an owed deletion: the deletion carries the four tombstone parts above.**

Andrew, 2026-10-10: a deleted experiment is documented both inline and centrally. The deletion commit (or its lane) leaves all four:

1. **Inline tombstone** at every deletion site: `# TOMBSTONE: MOJOLEARN_<DEFINE> (<verdict>) deleted <date> by <sha>; code recoverable at <sha>^.` plus a `# Restore:` line naming the patch.
2. **`docs/TOMBSTONES.md` entry**: an index row and a section with what it tried, the files, the verdict with NV/AMD ratios per dataset and the quality change, run ids, the recoverable sha and the patch path.
3. **`experiments/removed/<DEFINE>.patch`**: the reverse of the deletion restricted to code files (`git diff <sha> <sha>^ -- <files>`), so `git apply` restores the code. Code that never reached main gets no patch; record the lane branch@sha instead.
4. **Refusal and records**: a `removed` refusal in the define's family module `core/experiment_guards_<family>.mojo` (`core/six_lane_experiment_guards.mojo` only for guards shared across families; see `docs/build/HOT_GROUPS.md`), the entry moved to `removed` in its `grid_controls/*.json`, and the `docs/apple-fast/EXPERIMENTS.md` row.

## 6. Mojo rules that cost a build round (Andrew, 2026-11-09)

Each of these compiled badly or hung during the pre-0.8.67 integration. Follow
them when writing Mojo; lanes do not compile, so the compiler will not catch
them for you.

- Never name a parameter `out`. It is a reserved word (an argument
  convention), and the compiler reports "expected argument name". Use `dst`
  or `y_out`.
- Never name a local, a field or a parameter `fn`. It is a removed keyword: the compiler reports "'fn' has been
  removed; use 'def' instead" and stops parsing the file, which hides every later error in it (2026-11-15: two host
  units of a measurement build). Use `entry`, `cached` or `callee`.
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
  copy; use `.copy()` for borrowed descriptors. Conditional floating expressions infer `Float64`;
  use `Float32(1.0 if condition else 0.0)` in Float32 fixtures. Copy a string slice into a temporary
  before assigning it back to the source. `unsafe_memcpy` requires the `dest=`, `src=` and `count=` keywords.
- A Mojo module containing `main()` cannot be imported as a package. Put reusable native test
  bodies in a suite module with `run_checks()` and keep the standalone `main()` wrapper separate.
- Mojo 1.1/1.2 `sort` takes its comparator as a runtime argument: `sort(span, comparator)` over a
  mutable `Span`. Use a runtime closure with explicit captures when the comparator needs outer values;
  import `sort` from `std.builtin.sort`. `host_parallelize` likewise takes a runtime closure with
  explicit captures (`def task(unit: Int) {imm pointer, imm count}:`). Use `comptime if`/`comptime for`
  for compile-time control flow; those statements no longer accept a parameter decorator.
- `match` and `case` are reserved keywords, including for parameters and local variables (`def f(case: Int)` fails
  with "expected argument name"); use `matched` for a Boolean result and `which` for a case index.
  Put `comptime assert` inside a function body, never at module scope.
- Generic struct parameters used by fields, static methods or initializers must be qualified as
  `Self.E`, `Self.D`, `Self.RES` (and so on), including arguments to generic callees.
  An owned generic field also needs `Deinitable` in its type bound; `Copyable` and `Movable` alone
  do not supply that constraint.
- To transfer an owned field while leaving its aggregate destructible, swap it with an empty
  native value. Moving a nested field with `^` can leave the aggregate partially destroyed;
  copying the buffer to hide that ownership error is not a repair.
- A Python-registered native owner with non-`Writable` state implements both `write_to` and
  `write_repr_to`. Otherwise the binding derives repr through the state fields and fails compilation.
- Construct a native list with a typed literal (`var values: List[Float32] = [1.0, 2.0]`),
  keyword length/fill, or explicit appends. Positional `List[T](a, b, ...)` is not supported.
  A one-element native shape uses `List[Int](length=1, fill=n)`; `List[Int](n)` is not a
  one-element list. Never pass the compiler-internal `__list_literal__` keyword directly.
- Never return a std `PythonObject` subscript or call directly (`return fields[key]`,
  `return obj(args)`) when an argument is a borrowed `PythonObject` (2026-11-12). Mojo 1.2.0 emits
  such a return-position variadic-pack call as an LLVM `tail call`, which lets LLVM drop the
  borrowed key's stack store; the callee then reads a stale pointer and segfaults (RandomForest
  fit and every schema save did). Read through the C API (`py_get`, `py_call`, `py_setitem` in
  `bindings/py_balanced.mojo`) or copy the key into a local first. Evidence and reproducer:
  `~/mojolearn-evidence/verify-20261112/ROOT_CAUSE_tail_call_pack.md`.
- `PyObjectPtr` is an opaque CPython wrapper, not a `Pointer`; it has no
  `unsafe_from_address` initializer. Native owner inspectors use
  `borrowed_native_owner` in `bindings/native_owner_registry.mojo`, which uses
  the supported `upcast_from=Optional(pointer)` constructor and retains the
  borrowed Python reference. Keep the original owner pinned through inspection.

- One buffer, or two views of one buffer, is never passed twice as a mutable argument to one launch (2026-11-19). The
  build fails with "aliasing values passed mutably to 'args' argument ... in 'enqueue_function' call". Pass the buffer
  once and have the kernel read and write through that one pointer, or give the second role its own view built the way
  the neighbouring launch builds one.
- A struct field never holds a pointer with AnyOrigin (`List[Pointer[T, MutAnyOrigin]]`, a `MutPointer` field)
  (2026-11-19). The build fails with "struct fields cannot expose AnyOrigin in their type". Hold the owning buffer or
  view, or the address as an `Int`, and build the pointer in a local `var` at the launch site.
- Code behind an opt-in define is still parsed and type-checked in the default build. An arm's compile error breaks
  every build, so an arm is written to the same standard as default code and merged only with its lane's reading of
  every launch it adds (2026-11-19: five errors in three compile rounds, all in default-off arms).

## 7. Public docs and docstrings (Andrew, 2026-10-12, additions 2026-10-13)

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

## 8. Temporary directions (for now only; each ends when Andrew says)

Each entry says what it overrides and is removed whole when it ends. The standing rules in parts 1 to 7 are not edited
to match a temporary direction, and no agent treats one as permanent or extends it to other work.

### TEMPORARY: measure on NVIDIA only, small board first (Andrew, 2026-11-14; restated 2026-11-15)

Ends when Andrew says. It is not the product direction: AMD and Apple remain in the contract and return to measurement.

- Until Andrew says otherwise, measurement builds are timed on the NVIDIA L40S only: no MI300X leg, no AMD rent, no
  Apple timing, no verify at the end of a measurement. The cross-vendor verification happens once, at the release.
- While this campaign runs it overrides every "NVIDIA and AMD" in the measurement and release rules (restated
  2026-11-15: "forget about AMD for now"): a switch is decided, a measured winner is replaced, a fix wave is measured
  and an owed A/B arm is timed on the NVIDIA L40S alone. No AMD box is rented and no AMD build is waited on. The
  two-vendor average, the AMD and Apple timing columns and the vendor-split flag return when Andrew reopens them.
  Bits across vendors are unaffected: the release's one verification still covers every vendor.
- Order: fix everything known (every regression row, every audit finding, every leftover a lane reported) -> one
  GitHub build -> the PAPER board (the 108 recipes of the 0.8.55 comparison) against 0.8.55 alone ->
  read every flag -> for each regressed row look back through 0.8.55, the earlier releases and every stored
  measurement run (best-ever table, boards in `bench/results/` and R2) at what the BEST run of that row was doing,
  and restore that behaviour or beat it -> fix wave -> one build -> measure again. Repeat until every paper row is
  at least as fast as 0.8.55 with quality equal.
- Only then the FULL board, NVIDIA only, against best-ever; then the same fix loop on its regressions. The full board
  times one prior on the same box, 0.8.55 (Andrew, 2026-11-15).
- One prior only, on every board (Andrew, 2026-11-16): a measurement job times the candidate and 0.8.55 on the same box
  and nothing else. 0.8.66 is not timed on the paper rows or anywhere; this overrides "and the last release" in part 3.
- 0.8.55 is not timed again (Andrew, 2026-11-17): the 0.8.55 times taken on the L40S on 2026-11-17 (full board, host
  26ee0a3b0f86, driver 595.112.07) are the stored baseline for this campaign. A measurement job times OUR new wheel
  only, on the same or a similar L40S, and only the rows a change touched plus the owed A/B switches; nothing already
  known is measured again. This overrides "re-times OUR prior wheels on the same box in the same job" in part 3 while
  it runs: the compare shows the stored 0.8.55 time with its host, driver and date beside each row and says in its
  first line that the baseline is stored. A/B switches are still decided back to back in one job (base and arm).
- ONE integration branch for the whole campaign (Andrew, 2026-11-14: "everything, everything, everything"): every fix
  lane, every audit fix and every leftover is merged into the same integration branch (`preinteg/wave2-20261113` until
  it lands), which is the only branch compiled. main receives that branch whole, never a piece of it; no lane and no
  tooling fix merges to main on its own. Fix as we go: a defect found while a build or a measurement runs gets its
  lane at once on the integration branch, and no measurement is repeated for it alone.
- Carefully, never narrowly: quality is never traded for speed (both arms' quality beside every time; a quality drop
  is a regression), no rule may key on a benchmark shape or dataset ("No dimension targeting"), and no row is
  "fixed" by measuring it on a smaller case. Not piecemeal: one wave fixes everything the last measurement showed.

### TEMPORARY: we time ourselves against ourselves, on the same pod; 0.8.55 is retired as a yardstick (Andrew, 2026-11-19)

Ends when Andrew says. It narrows the NVIDIA-only direction above while it runs and replaces the 2026-11-18 direction
"time our latest against the stored 0.8.55 board" (verbatim in `docs/rules/RULES_HISTORY.md`).

- A speed verdict (REGRESSION, FASTER, an A/B win or loss) comes ONLY from two of our own wheels timed on the SAME pod
  in the SAME job, with every worker bound to the GPU's NUMA node: the candidate beside the last green build
  (`tools/ops/measure_candidate.py stage ... --prior-wheel NAME=DIR@SHA`), or an arm beside its base. A pair whose two
  cells ran on different nodes or hosts is not a verdict.
- The reference is rolling: each green measurement build is timed against the previous green build that was timed.
  Nothing has to be at least as fast as 0.8.55; what must not happen is our latest being slower than our previous
  build, at equal quality, beyond the pair's noise.
- 0.8.55 is never installed, timed, profiled or compared again. Its stored board (the 2026-11-17 L40S full board) stays
  in the evidence as history; it makes no flag, opens no lane and ranks no work. Reason (2026-11-19): against it, the
  2026-11-19 full board showed 178 entries at 1.2x or slower; the same-pod pair job showed all but one of the rows
  examined were the host or the unbound workers, and the one "regression" (cagra taxi) was NUMA placement.
- Our latest build is still TIMED in full: every round ends with the whole board timed on the L40S for the candidate
  and its prior, not only the rows a fix changed.
- Work is opened for a same-pod gap beyond the pair's noise, worst absolute time first.
- Opponents are set aside for now: no opponent job, no opponent re-score, and no work is opened, held or ranked because
  of an opponent cell. (The paper's 0.8.55 comparison and the opponent boards are a release-time question for Andrew.)
- Order, without haste: fixes and A/B verdicts timed same-pod; the boundary cleanup by waves (no speed case: compile
  and native checks only, no measurement of its own); `fast` routes and per-estimator `fast`; the other owed cleanups;
  and only then the release, when Andrew says. Nothing is skipped or shortened to reach it sooner.
