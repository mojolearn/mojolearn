# Agent working rules

## Within-release identity and fresh reference hashes (Andrew, 2026-10-29)

- Bitwise identity is required across supported GPU vendors (Apple, NVIDIA and
  AMD) within the same release. It is NOT required across versions. Do not block
  a correct implementation or sampling arithmetic change merely because it
  changes an older release's hashes or seeded output. Document numerical policy
  changes; preserve supported API semantics, correctness and sampling quality.
- After the current sampler change and remaining release fixes are settled,
  freeze the selected source/configuration and regenerate reference hashes for
  EVERY algorithm. Andrew explicitly requests this fresh reference campaign;
  it supersedes compile-only/no-campaign instructions for that subsequent work.
  Do not start it against a moving sampler implementation.
- Apple or NVIDIA may generate the first candidate hashes. Those hashes are
  provisional until corresponding supported vendor executions agree under the
  same release policy, inputs and configuration. Never accept vendor-specific
  reference columns to hide a disagreement, or regenerate expected hashes from
  a failing vendor merely to make that failure pass.
- Preserve historical tables, raw failures and provenance. Replace the active
  release reference through the supported verifier/ledger tools with complete
  algorithm coverage and explicit failed, skipped and missing cells. Fresh
  reference hashes are not independent correctness evidence or a speed claim.
- Reuse unchanged compiled artifacts when their source closure, compiler,
  flags, target and dependencies match; refreshing all hashes does not itself
  require rebuilding all binaries.
- Keep complete logs on disk. Inspect exit/status summaries, then targeted
  rg/grep (normally at most 20 matches with 2-3 surrounding lines) or short
  tails. Aggregate large JSON/status/worktree outputs before returning them;
  keep routine diagnostic responses around 2,000 tokens or less.

## Use incremental development compilation (Andrew, 2026-10-29)

- Use `tools/development_build.py` and the smallest supported affected groups
  in `bindings/development_groups.json` for development repairs. Inspect the
  source-closure plan first; select groups with repeated `--group` arguments.
  For example, the shared attribute repair selects `--sets host
  --group host:tokenizer --group host:runtime_classical`, not all host/GPU sets.
- Reuse unchanged compiled groups when their source closure, compiler, flags,
  target and dependencies match. A changed file requires rebuilding its actual
  dependent groups, not the entire library. A commit, merge, documentation
  update or reference-hash refresh alone is not a reason to rebuild them.
- The current supported unit is a registered development group, which may
  contain a large extension. Do not claim function-level incremental linking
  exists. Prefer supported finer groups where available; do not interrupt an
  active valid build or introduce a new loader/build scheme merely to make a
  small repair's compilation smaller.
- Keep standard optimized compiler settings and the existing Mac compile-slot
  and resource admission rules. Do not use reduced optimization to shorten a
  build. Record selected-group compile success separately from an incomplete
  full-set result (the development builder may return exit 3 for the latter).
- Assemble a complete current install from changed and proven unchanged
  groups, then run the affected installed runtime checks and verifier. Separate
  development groups are valid evidence for the paths actually executed; do
  not require a monolithic rebuild merely to count that evidence. Test changed
  loading/linking boundaries when assembling the release artifact.

## Direct CPython is allowed during execution (Andrew, 2026-10-28)

- Direct CPython C-API calls are allowed throughout native product execution,
  including execution loops, not only at entry and exit. This supersedes earlier
  boundary-only or blanket CPython/import prohibitions. A Mojo import providing
  native CPython access is not, by itself, Python code execution.
- Python computation and callbacks into Python/user-object hooks remain
  prohibited, subject to the explicitly documented exceptions below. A C-API
  call can invoke such hooks; audit the operation and admitted types rather than
  assuming a native entry point cannot call Python. Preserve exact-type checks,
  reference lifetimes, error propagation and required GIL/thread safety.
- Optimize repeated work: hoist invariant admission and metadata, reuse plans
  and buffers, and remove redundant lookups, conversions and allocations.
  Removing a useful CPython call or import solely to satisfy a purity count is
  not a goal. Do not add complexity or weaken correctness to avoid CPython.
- Providers/adapters serve ecosystem integration (including Python and MAX).
  Do not relocate algorithm work into provider directories to evade checks or
  claim Python removal. Assess the actual transitive execution path; a file
  move alone is not a performance improvement.

## Compile and merge; no verification campaign (Andrew, 2026-10-27)

- Continue the migration with small affected compilations, then merge and push
  successful coherent batches into main. Local or authorized CPU RunPod builds
  qualify the paths actually compiled; keep their scope explicit.
- Do not run runtime, identity or cross-vendor verification campaigns for this
  work. This supersedes the earlier instruction to run a final verification
  pass in this session. Required source/push guards still apply; compilation
  must never be described as verified runtime behavior or bitwise identity.

## Provider migration CPU builds and peer autonomy (Andrew, 2026-10-27)

- Any peer or subagent may rent a RunPod CPU box when useful for this migration.
  Use all allocated CPU cores for compiler workers or bounded build concurrency;
  determine the allocation from affinity/cgroup limits, not the host CPU count.
  Do not inherit one- or two-core development caps. Record the actual allocation
  and settings; configured parallelism does not prove every phase saturates CPUs.
- Arm a 60-minute deadman. A box may remain available for reuse within its
  protected lease. Record ownership and state so other sessions do not delete
  it. Preserve complete run outputs and required binaries on the Mac and in R2
  before teardown. Existing disabled watcher services remain disabled.
- This authorizes focused CPU compilation by peers/subagents, superseding the
  older lane-never-compiles restriction for this migration. Avoid duplicate
  same-source builds; a small successful compile remains required before merge.
- Mac compiles hold one of 4 machine-wide slots: `bash tools/compile_slot.sh <command>`
  (`~/mojolearn-evidence/compile_slot.sh` remains as a shim that execs the main checkout's copy).
- Peers may fan out and resolve routine implementation and merge decisions
  autonomously. Report completed coherent batches, genuine blockers/ownership
  conflicts, and rented-box ownership. Put routine findings in handoff files
  rather than sending repeated coordination messages. Main integration remains
  serialized; do not mutate another agent's worktree.

## Provider migration delivery (Andrew, 2026-10-27)

- Move the whole public operation into native code: validation, fitted-state
  checks, argument preparation, execution and finished-result conversion. A
  native numerical call surrounded by Python work is an incomplete migration.
- Merge completed provider repairs incrementally into main after a successful
  small local compile of the affected code. Do not require broad runtime or
  cross-vendor verification before these development merges; perform the
  focused final verification after integration. Keep pending coverage explicit.
- The initial provider integration was merged locally with a tokenizer compile
  failure; that exception is not permission to merge future compile failures.

## Exported graph boundary (Andrew, 2026-10-27)

- The provider migration may change the API to require an already exported
  graph. Do not execute a user's Python `forward`, exporter or serialization
  callback inside product execution. Mojo owns admission and execution, with
  stable input metadata captured once per public operation.
- Cross-version output changes are allowed. Preserve bitwise identity across
  supported GPU vendors within the new version. If arithmetic changes, limit
  additional vendor verification to the affected algorithms and invalidated
  evidence; adapters do not gain duplicate arithmetic identity lanes.

## Focus on identical; stop Apple fast investment (Andrew, 2026-10-22)

- `deterministic` is retired, with no separate mode, build, evidence campaign
  or public documentation offering. `identical` is the library's focus;
  existing Apple-only `fast` may remain opt-in. The research paper and its
  generated evidence cover only our `identical` mode. Preserve historical raw
  records and rejection tests for retired mode values.
- Direct development and performance work to `identical` mode. Stop investing
  time specifically in Apple `fast`: no dedicated features, fixes, optimization,
  tuning, builds, validation or benchmark campaigns unless Andrew explicitly
  reopens that work.
- This supersedes earlier directions to pursue Apple `fast` experiments or
  improvements. Existing fast code and retained evidence stay in place; this
  direction does not request their removal.
- Apple `identical` remains part of the shared Apple/NVIDIA/AMD bitwise contract
  and its required verification. Work needed for `identical` may touch shared
  code; that does not authorize a separate `fast` workstream.
- The current underoptimized `fast` implementation is not a baseline for claiming
  the isolated performance cost of bitwise identity.

## Logs and context

- Save complete build, test, and GPU logs to files. Inspect the exit status and
  structured summary first; do not stream or paste entire logs into context.
- Prefer targeted `rg` searches (or `grep` when unavailable), bounded context,
  and short tails. For example:
  `rg -n -i -m 20 -C 3 --max-columns 240 --max-columns-preview 'error|fail|refused|divergent|summary' run.log`
  or `tail -n 20 run.log`. Narrow to relevant files before searching directories.
- Parse large JSON reports for selected fields, counts, and failures rather
  than printing whole documents. Keep full evidence on disk and report its path.
- Expand the relevant excerpt when needed to establish the cause. A filtered
  excerpt or a few passing lines never proves the whole run passed: preserve
  exit codes, expected coverage, failures, and incomplete or skipped work.
- Never read whole subagent transcripts or `tasks/*.output` files. Request a
  concise status or inspect the specific retained artifact instead.

## No background watchers (Andrew, 2026-10-23)

- Background watcher, monitoring, polling, mirror and periodic collection services
  are disabled at Andrew's request. Do not install, restart or replace them unless
  Andrew explicitly asks. This includes the periodic box archiver.
- Keep explicit per-run result retention and archive-before-teardown checks.
  Active build and measurement controllers are not background watcher services.

## Lightweight release verification and incremental builds (Andrew, 2026-10-25)

- Bitwise identity is required across supported devices within one release,
  never against a previous release's outputs. Comparing source fingerprints
  with an older build decides which evidence can be reused; it is not a
  requirement to keep the implementation or its results unchanged.
- Development builds and separately compiled groups are valid for runtime and
  identity verification of the paths they actually execute. A monolithic or
  production-layout rebuild is not required merely to count that verification.
  Retain source closures, compiler, flags, dependencies, target, binary hashes,
  actual coverage and results. Do not promote source checks or compilation to
  runtime success, or extend a partial result to untested paths or devices.
- Reuse those results during release. Require an explicit relevant input change
  or observed failure for each rerun; a version, commit name, packaging step or
  unchanged group assembled beside a changed one is not sufficient by itself.
  If assembly changes linking, loading or dispatch, test that affected boundary.
- Release verification uses bounded correctness fixtures and focused installed
  artifact, loader, dependency and adapter smokes. Full-dataset performance
  measurements, opponent races and repeated broad qualification are separate
  campaigns, not automatic publication prerequisites. Preserve required
  within-release identity evidence and negative controls; run missing or
  invalidated checks only. Full-workload evidence remains required to promote
  performance defaults or make performance claims.
- NVIDIA PTX verification may run on a suitable RunPod NVIDIA GPU with a
  compatible driver. Record actual GPU/driver and the selected PTX route;
  executing a native architecture image does not qualify the PTX route.

## Results leave the box (Andrew, 2026-10-18)

Every run on a rented box keeps its logs and results on the Mac and in R2, not only on the box.

- Write run output under `/root/lq`, `/root/live-*`, `/root/measure-*`, `/root/race-*` or `/root/results*` on the box.
  `tools/box_archive.sh` copies their logs and results, with the box's hardware and software snapshot, home to
  `~/mojolearn-evidence/box-archive/<box>-<stamp>/` and to R2 at `measurements/box-archive/<box>/`. It skips caches,
  venvs, `.so`/`.whl` files and anything over 50 MB: those are rebuilt or already in R2.
- The periodic launchd archiver is disabled (2026-10-23). Retain results explicitly during each run;
  `tools/dev_pod.sh down` and `tools/do_amd_steward.sh down` still archive before the delete. They refuse to delete a reachable
  box whose archive fails. `MOJOLEARN_DOWN_NO_ARCHIVE=1` skips the archive, only when the results are already home.
- A box rented any other way must get a state file in `~/mojolearn-evidence/devpods/<key>/state.env`
  or `do-amd-steward/state.env` so explicit retention and teardown can find it.
- Before taking a box down, check who owns it: `ListAgents`, then the box's keeper or brief. Never take down another
  session's box.
- A run's key results also go to the repo or to `~/mojolearn-evidence/<run>/` as soon as they land. The archive is the
  safety net, not the record.

## Lane and subagent prompts

Include this reminder in every lane brief, spawn prompt, and resumed assignment,
and ask delegated agents to propagate it to their own subagents:

> Keep logs out of context: save complete output to files, use targeted rg/grep
> with bounded surrounding lines and short tails, and summarize exit status,
> coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
> never hide failures or infer full success from filtered output.

Final handoffs should give the branch/commit, changes, validation status,
remaining work, and evidence paths without pasting full logs or diffs.

## Wait for Mojo and Modular

If Mojo or Modular does not support something yet, do not build a workaround for it; wait for their support.
Examples: an AMD portable kernel path needs Mojo to keep kernel IR (or bitcode) and to accept generic gfx targets;
until Modular ships that, we do not pursue it (decided 2026-10-04; the parked prototype is on `lane/amd-portable`,
with the evidence in `docs/AMD_PORTABLE_PATH.md` on that branch). Do not hand-rewrite compiler output, patch
toolchain internals, or ship unsupported build modes. Record the ask for Modular instead and move on.

Owner exception (2026-10-25): use the documented native CPython dictionary API
for the reproduced instance-field lookup failure in native `get_params` and
public attribute access. The standard-library-only reproduction points to a
Mojo/Modular interoperability bug; its exact internal cause is unconfirmed.
`bindings/native_instance_item.mojo` records the failure, API ownership contract,
retained evidence and Codex Daybreak screening interruptions inline. This
specific repair is authorized; do not reopen its permission question. Preserve
exact dictionary/string admission, object identity, descriptor precedence and
lifetimes. This specific native API repair needs no Python fallback or callback.
Other explicitly authorized Python exceptions are recorded under "No Python in
the runtime" below. Validate the affected public runtime before broad qualification.

## No dimension targeting, in any mode

No dispatch, tile, threshold, cap or route rule may key on an exact benchmark dimension, a size chosen to sit just above or
below a board row (rows, features, classes, k, vocabulary), or a board dataset name. This applies to FAST and to IDENTICAL on
every vendor. A rule must come from size, hardware or cost reasoning that covers neighboring shapes, stated in a comment.
In IDENTICAL, removing such a rule may change bits: that is allowed, because bits must match across Apple, NVIDIA and
AMD within one version, never across versions (Andrew, 2026-10-13, supersedes the 2026-10-07 vendor exception). Change all columns together. Each removal gets an
A/B with the old rule as the B arm, timed on neighboring shapes and one non-board dataset.

## Measurement process (owner, 2026-10-05)

### Full-dataset candidate evaluation (owner clarification, 2026-10-06)

- ALWAYS evaluate performance candidates with end-to-end A/B measurements on the full dataset for each affected
  estimator. This applies to AMD, NVIDIA default, NVIDIA architecture-specific, and applicable Apple FAST candidates.
  Component timings, synthetic substitutes, tiny public-caller fixtures, and opponent-only runs do not complete this work.
- Before timing, map each candidate to its affected estimators and saved full-workload recipes: dataset/version/hash,
  actual rows/features/classes or sequence dimensions, estimator settings, numeric mode, A/B toggles, and timed boundary.
  See `experiments/performance_ideas/README.md` for the recipe locations. Audit intrinsic lane caps: `--rows full` alone
  does not prove the full intended dataset was measured. Missing or ambiguous coverage stays pending; do not silently
  substitute a smaller case and report completion.
- Include preparation, fit/training, required synchronization, and consumed outputs in the declared whole-operation
  boundary; report fit, inference, cold and repeated use separately where applicable. Measure relevant toggle combinations
  and the proposed complete default configuration: independent component wins can interact and regress a full workload.
- Do not promote a default from component or reduced-workload gains. Require the full-dataset evidence above and the
  existing vendor/quality acceptance rules. Record winners AND losers beside switches and in the boards, with scope,
  sample counts and pending coverage. Say "component screening complete" when only screening is complete, never
  "measurement campaign complete" or "full board complete".
- Reuse accepted compilation and identity evidence. Do not repeat compilation/identity validation merely to evaluate
  performance; repair only actual measurement failures, preserving completed cells and original failure evidence.


1. Freeze one commit per A/B round. Compile it once on cheap boxes (Apple on the M2; NVIDIA/AMD on cheap fast-CPU boxes).
   Only a green frozen build goes to the timing GPUs. New code waits for the next freeze.
2. IDENTICAL switches are decided by NVIDIA and AMD together: the verdict is the AVERAGE of the two vendors' ratios against the
   average noise floor (Andrew, 2026-10-08: "the average of the two must be better"); a vendor pulling the other way is flagged
   on the cell (vendor split) but does not block the average. Apple never votes on IDENTICAL switches and IDENTICAL is never
   tuned for Apple. This performance voting rule does not relax correctness: Apple IDENTICAL hashes MUST match
   NVIDIA and AMD within the same release (Andrew, 2026-10-13). Apple FAST has no cross-vendor identity promise.
   Never accept a separate Apple reference to turn an IDENTICAL mismatch into a pass.
3. Measure IDENTICAL on NVIDIA, AMD and Apple and update every board (main board included) as results land, through the
   board tools only. A full-board run is IDENTICAL on the three; FAST is not rerun.
4. Standing order: when a problem is found, fix it. Do not just comment on it or defer it.
5. Read logs with grep and short tails; never paste whole logs. Tell every subagent the same.
6. OPPONENT CLOCKS (Andrew, 2026-10-07/08; audit tools/board_clock_audit.py): every stored opponent cell carries a comparability span (input_home, inside_clock, upload_ms_untimed). Our board fit clock includes the host-to-device copy of X. The neural torch twins move inputs inside their clock too (whole/whole, comparable as is). The classical torch-gpu columns and the algos torch/cupy/gpytorch GPU arms time with inputs already on the device (kernel-only); cuML/cuVS/cuGraph record their copy separately. The board shows both clocks per cell where derivable (whole = kernel + copy): torch-GPU ratios kernel/kernel, every other opponent whole/whole, never an invented copy time. NO opponent re-score. Our arm records its copy separately (upload_ms_separate; classical since freeze-20261007; algos driver owes the probe), then only OUR cells on the MIXED lanes are re-raced.

## Race fairness: what the measurements are for (Andrew, 2026-10-13)

The boards measure the cost of bitwise identity: mojolearn's IDENTICAL mode against the best real library
implementation of the same algorithm, on the same kind of hardware, with the same settings. Every supported result is
shown, wins and losses alike. Owner clarification later on 2026-10-13: keep measurements with sufficient
evidence; exclude pairings with missing, suspect or bad evidence. Do not rerun everything.

A race is GOOD when both arms run the same algorithm with the same settings on the same kind of box. These do
not make a race bad:
- random seeds or random starts (initial weights, k-means++ draws, sampling): ideally both arms use the same
  seed, but different or random seeds are acceptable;
- anything mojolearn does because bitwise identity requires it and the opponent does not: FP32 instead of TF32
  or bf16, fixed reduction order, exact instead of approximate kernels, extra deterministic work, its
  host-to-device copy inside its clock. That is the identity penalty the boards exist to measure;
- the opponent's own fast settings and modes (bf16 or TF32 arms, torch.compile, its GPU defaults);
- the library version or host the opponent was measured on, when the settings match (no host is ever excluded
  for being slow; the slow-host rule was removed 2026-10-14);
- a tie (ours/theirs between 0.95 and 1.05): it is shown like any other result.

A race is BROKEN, marked rerace "x" and omitted from the paper and mojolearn.org, when:
- a setting that changes the work differs between the arms (tolerance, solver, algorithm, model, precision, a
  cache or precompute one arm uses and the other is told not to), unless the difference is required by bitwise
  identity on our side;
- the opponent is code our harness wrote rather than a library's own implementation.

FAIRNESS IS ABOUT THE PARAMETERS (Andrew, 2026-10-14): every work-changing parameter is set explicitly and the
same on both arms, never left to each library's default. Effective settings are read back to prove it.

QUALITY IS ALWAYS SHOWN, NEVER A GATE (Andrew, 2026-10-14; supersedes "compare quality first"): quality never
decides whether a race is shown. Both arms' quality is recorded and shown beside every time, whoever is better;
it is fine for us to be better. Quality is for judging changes to our own algorithms: speed is good, but never at
the expense of quality. `tools/board_quality.py` margins only annotate the comparison.

Different executed iteration counts do NOT make a race unfair. Retain counts, limits, tolerances and seeds as
evidence; never pad work to equalize counts. Existing supported evidence is reused; rerun only missing or bad
evidence. This campaign covers all NVIDIA/AMD GPU-opponent recipes; no Apple runs are authorized.

When only one opponent arm of a race is broken (for example harness-written), that arm is excluded
(race["opponents_excluded"]) and the race is shown against its remaining opponents; the whole race is hidden
only when no valid opponent is left.

THE RULE IS CODE: tools/race_fairness.py (tests: tools/test_race_fairness.py) applies it to any board.json, and
tools/main_board_ingest.py calls it on every rebuild; `python3 tools/race_fairness.py --check <board.json>...` shows
what it would change and `--review` lists warning signs on rows still shown. Its registry HARNESS_ARMS
is where a new harness-written arm is recorded. The paper (mlsys) and mojolearn.org only read its marks.

Every (mode, phase) row of a race in board.json carries an entry in race["audit"] (verdict, reason, evidence,
opponent, ratio, date, pairing fingerprint); an entry with rerace "x" is broken, and race["rerace"] is "x" when
any of its entries is. tools/main_board_ingest.py (apply_fairness) keeps an entry only while the measurements it
judged are unchanged; missing or unresolved evidence excludes the pairing. Each row lists its `admitted_opponents`;
the paper and website publish only those arms for that mode and phase.

New measurements retain actual source bytes before execution and full logs afterwards using
`tools/bench_board_evidence.py`. Each cell's `measurement` includes versions, open-ended raw and canonical
parameters, data identity and actual shape, worker resources, timing boundaries and content-hashed R2 references.
The board checks effective parameters before timing, preserves records when reusing opponents or importing
cells, and binds audit decisions to the full evidence rather than just median times. Retention failures stop
the next race; retry uploads without rerunning completed measurements. Timing ratios alone never establish
a slow-host exclusion; record a supported audit finding first.

Fix the race code rather than dropping an opponent; drop an opponent only when no real library implementation
exists for that vendor. Every race-code fix from an audit carries a `FAIRNESS FIX (2026-10-13 audit)` comment
saying what was wrong before; results raced before such a change get extra scrutiny. Audits never rerun anything.

## Incremental experiment delivery (owner, 2026-10-05)

- Keep experiment code, the selectable runner, decisions and retained result summaries in `main`.
  Experimental code may merge after existing compile evidence on one device, with untested targets and measurements
  explicitly marked pending and unproven defaults disabled. Do not rebuild merely to merge already compiled code.
  Lane agents never compile (owner, 2026-10-07): they write, commit and push code only. The orchestrator merges finished
  lanes into one integration branch, compiles it once (all switches on and off, NVIDIA and AMD targets, on cheap build
  boxes), fixes what breaks, and only then merges to main.
  A compile pass on one target is not a claim that every target works.
- Run selected A/B candidates through the actual full-size measurement harness in IDENTICAL mode on NVIDIA and AMD,
  using one excluded warmup and one scored sample. Reuse verified frozen binaries when numerical source, flags,
  compiler and target match. Run the vendors in parallel on separate owned boxes and cells serially within each GPU.
  Capture correctness/identity evidence from the scored execution. Apple identity matters; Apple timing does not vote.
- As each cell lands, retain its source, binary, hardware and harness provenance, timing, correctness and failure status.
  Update the experiment board and applicable main/vendor boards through board tools immediately. Pending, unsupported,
  failed or unqualified cells must remain visibly separate from admitted measurements; preserve the original evidence.
  Do not invent opponent ratios from own-only A/B runs or overwrite incomparable historical opponent measurements.
- Evaluate each candidate as soon as its required full-dataset, end-to-end baseline/candidate measurements and accepted
  quality evidence are available on BOTH NVIDIA and AMD for all affected estimator workloads and relevant combinations.
  Do not wait for unrelated candidates or the entire
  campaign to finish. Apply the existing rule: combined improvement, neither vendor materially slower, required identity
  and quality preserved. A single partial cell, compile pass or synthetic-only gain is not enough to flip a default.
- When that evidence supports a change, flip the applicable switch promptly and commit/push it to `main`. Put the reason
  INSIDE the source file beside the switch: experiment/source IDs, cases, timing ratios for both vendors, identity/quality
  outcome, sample count and limitations. Record failed, neutral and rejected candidates there too, including why they stay
  off. Also commit the corresponding machine-readable results and board-tool updates. Do this as decisions become ready.
- A promoted source change starts a new freeze; never mutate binaries or settings inside an active frozen run. Repair and
  rerun only affected cells, preserving valid completed evidence. Do not repeat already decided experiments unchanged.

## Full-machine CPU race resources (owner, 2026-10-05)

- Every CPU race arm gets the full machine allocation: opponents and our CPU arms wherever measured.
  No inherited laptop/test caps or `MOJOLEARN_BENCH_THREADS=1` overrides in a race. Use all-core
  estimator settings where supported and run measurement arms serially; do not introduce competing jobs.
- On dedicated Apple machines, leave CPU libraries unrestricted. On rented Linux machines respect the
  actual cgroup allocation, not the host's larger visible CPU count. Do not claim all cores are busy:
  serial algorithms, library defaults and nested parallelism may legitimately use fewer threads.
- Preserve algorithm semantics (including seeded UMAP) and documented nested-pool settings (implicit's
  BLAS=1 inside its parallel solver). Do not remove correctness settings just to raise utilization.
- Record worker resource policy, thread environment and effective pools with measurement evidence.
  A controller's environment alone does not prove an old worker's thread count. Keep unproven historical
  resource claims explicit; do not label every previous CPU result as one-core without worker evidence.
- These resource rules do not expand the selected race arms or alter another session's active frozen run.

## Measurement artifact retention before teardown

Before terminating a measurement machine, prove where every required output,
model-state artifact, accepted binary and unique input is stored. An attached EBS
volume with DeleteOnTermination=false does not prove that a workspace is on it.
Map the actual filesystem through its physical device to the provider volume ID;
directory names and mount labels are not evidence. For instance-local storage,
verify the retained bytes off the machine before termination. Metadata and hashes
alone are not retained array/model/binary bytes. Keep any incomplete retention
explicit and stop teardown until the required evidence is durably preserved.

## Framework adapters are the same API in another format (Andrew, 2026-10-24)

The Torch and MAX adapters (`adapters/`, `python/mojolearn/adapters/`, shipped as `_adapters/` in the
wheels) are front-ends over the same compiled kernels the Python API calls. They compute nothing of
their own. So they are NOT a new surface for the identity table, the verifier or the fault-injection
arms: the Python API lanes already record and sabotage that arithmetic, and a second column for the
same bits is the double work the release cost rules forbid. Treat them differently from an estimator:

- What qualifies an adapter is what is specific to it: the shipped files build in every release leg,
  survive the wheel audits, and load and run from the INSTALLED wheel on each vendor
  (`tools/release_wheel_smoke.sh --adapters --adapter-install torch,max` on NVIDIA and AMD,
  `tools/adapter_wheel_smoke.py --device mps` on Apple). That smoke is part of the release gate.
- Do not add identity lanes, verifier rows or sabotage arms for an adapter. If an adapter needs a value
  the kernels do not provide, add it to the Mojo kernel, where the existing lanes cover it.
- Adapter bugs are glue bugs (loading, dispatch registration, dtype and layout at the boundary); fix
  them in the adapter, never by changing kernel bits for one front-end.
- Whole-model only: the per-operation surfaces (`torch.ops.mojolearn.operation`, the MAX `mojolearn_matmul`,
  `mojolearn_relu` and `mojolearn_operation0..3` ops, the native `mojolearn_adapter_operation*` entries) were removed on
  2026-10-24 (docs/TOMBSTONES.md) and must not come back; the public adapter surface is
  `torch.compile(model, backend="mojolearn")`, `mojolearn.import_model`, `torch.ops.mojolearn.train` and the MAX
  `mojolearn_train` / `mojolearn_train_resume` ops.

## No Python in the runtime

Owner exception (2026-10-25, reaffirmed): Mojo/Modular bugs can require Python
in product execution. A narrowly scoped Python runtime implementation is allowed
where such a bug blocks a correct supported native implementation, including
calling Python from Mojo/native code. Preserve supported behavior. Document
INLINE the affected operation, the specific bug or suspected failure, the Python
work required, and the retained reproduction/upstream issue. Distinguish a
suspected bug from a confirmed diagnosis. Test the actual public operation and
record the exception explicitly in release evidence; never call it Python-free.
Use a working supported native repair when available. No additional permission
is required for a fallback within this authorized scope.

Previously approved, documented fallbacks after Codex cybersecurity/Daybreak
interruptions also remain authorized. Record those interruptions accurately;
screening alone is not evidence of a Mojo bug. This exception supersedes older
absolute no-Python/no-callback/no-fallback and wait-for-Modular wording only for
these documented paths. It does not authorize toolchain patches or silently
exempt unrelated runtime work.

Owner clarification (2026-10-24): scikit-learn tag metadata is a narrow
compatibility-layer exception to the rules below.

- `__sklearn_tags__` and helpers used solely to construct and return its tag
  metadata may run in Python, including loading the optional scikit-learn tag
  classes. They do not need to be migrated to Mojo solely to satisfy the
  one-native-call rule. Preserve the supported scikit-learn tag contract.
- Mojo/native product operations must not call back into Python tag handling.
  This exception does not permit Python in input validation or preparation,
  binding selection, computation, training loops, batching, backward, optimizers,
  worker orchestration or consumed-output processing. It does not exempt other
  metadata protocols or public methods.
- Audits must distinguish this allowed Python compatibility metadata from
  computational runtime violations. Do not count a tag method as a runtime
  violation solely because its metadata implementation is Python; still check
  its protocol correctness and reject callbacks to it from native operations.

This clarification supersedes the absolute public-method wording below only
for this tag-metadata boundary; the no-callback rule remains in force.

Owner clarification (2026-10-06, supersedes earlier Python API/glue exceptions):
Python is ONLY for PyPI packaging/distribution and testing. NEVER USE PYTHON IN
PRODUCT EXECUTION. This includes public API wrappers, argument/data preparation,
binding selection/orchestration, fit, transform, predict, score, training and
consumed-output processing. Product execution belongs in compiled Mojo/native
bindings; no Python loops, callbacks, worker threads or NumPy/SciPy/scikit-learn
computation, conversions or fallbacks may execute on a product path, in any mode
or on any vendor. Preserve supported interfaces and semantics through native
implementation; do not remove features to claim compliance.
Product orchestration is runtime work: batch/epoch loops, kernel launch sequences,
buffer/workspace management, synchronization and stopping decisions must also run
in Mojo. Moving arithmetic to Mojo while Python still drives these operations is
only a partial repair, not compliance. Measure performance gains; do not assume
that removing Python makes every full workload faster.

Andrew, 2026-10-19 (restated): the ONLY job of the Python package is to connect users to the compiled Mojo through PyPI.
A public method is one binding call that hands over buffers and returns the result; nothing else runs in Python.
This applies to the autograd surface too: `Sequential.fit` is one call and the whole training loop (epochs, batches,
launches, the tape, the backward walk, the optimizer) runs in Mojo; `ops.Session` records in Mojo and `backward` is one
call that walks the tape in Mojo. A user composing operations one call at a time is the API, not a runtime loop; a
Python loop inside the package that walks a tape, a batch or a model is a violation.

Andrew, 2026-10-20 (explicit clarification): NO CALLS BACK INTO PYTHON FROM
MOJO OR ANY OTHER NATIVE PRODUCT CODE. "No Python in runtime" is transitive:
it forbids invoking Python functions, methods, callbacks, property/descriptor
hooks and input protocols during a product operation. This includes custom
trainers, learning-rate functions, batch suppliers, commit/step hooks and
Python orchestration reached through a native extension. Putting a PythonObject
inside a Mojo struct, or moving the call site into Mojo, does not compile its
Python methods. No compatibility branch, callback exception or Python fallback
is permitted. Use native implementations, explicit native configuration/data
objects and native event/checkpoint mechanisms instead.

Holding buffer references or invoking a verified native extension entry point
can be binding work; audit its transitive callees rather than assuming that a
C/native entry point cannot call Python. Ordinary supported buffers and native
interfaces must retain their semantics. Refuse unsupported Python callback or
protocol objects before invoking their hooks, with a clear native alternative.
Python application code may call the API and consume a completed result; do not
move required product execution into caller callbacks to evade this rule.
Audit the actual public operation, including constructors, input preparation,
workers, callbacks and output handling. A passing private native helper does
not qualify an unchanged Python public wrapper. Retained checker debt is never
an exception to these requirements.

Python test/benchmark controllers, test-data preparation and independent test
oracles remain allowed as testing tools. Keep their work explicitly distinguished
from product execution and accurately included in or excluded from declared timing
boundaries. Do not move product work outside a timer or relabel it as testing to
hide a violation. Installing a missing Python package is not a runtime repair.

Audit transitive callees and the actual whole-operation boundary; a helper's old
"outside the clock" comment is not evidence. Existing checker baseline entries
are debt to remove, never exemptions, and an empty baseline is not proof that
all product paths are native. Repair in separate source freezes, preserving
original measurement/failure evidence and valid completed cells. If supported
Mojo/Modular functionality is missing, record the blocker and wait for support;
do not invent a Python fallback or unsupported toolchain workaround.

## Public docs and docstrings (Andrew, 2026-10-12, additions 2026-10-13)

- Docstrings of public names are user documentation (docs.mojolearn.org): what it does, parameters, returns, a short
  example. Engineering notes belong in `#` comments.
- Public docs state the contract in one sentence, verbatim: "In `identical` mode, mojolearn returns the same bits on Apple,
  NVIDIA and AMD GPUs." Every `numeric_mode` doc that lists `'fast'` says Apple only.
- Omit, never caveat: no "Known defect" or other status notes, no all-caps IDENTICAL/FAST asides (write `identical`/`fast`),
  no "NOT IMPLEMENTED" in docstrings. Excluded or unverified names (`docs-site/api_exclude.txt`) and unverified features
  never appear in public docs, the README or the site.
- We are alpha: no enterprise-grade checks. The docs, mojolearn.org and the measurements are generated from the latest
  origin/main by one publish command, run only when Andrew says so. Old URLs may break; no redirects.
- `fast` is Apple only. The 162M two-route model never appears in docs-site or the library docs.
