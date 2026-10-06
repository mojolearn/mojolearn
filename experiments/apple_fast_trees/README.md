# Apple FAST tree experiments — source only

The [idea list](IDEAS.md) was written before parallel implementation. It has
48 individual candidates and 12 interaction experiments. The implementation
records refine the initial proposals to the actual source mechanisms.

The [experiment file index](EXPERIMENT_INDEX.md) lists every new candidate and
interaction, exact A/B defines and implementation files, the existing 60-card
performance catalog, older RF/ExtraTrees comparisons, and historical tree
experiments with their recorded verdicts and source locations.

| Lane | Ideas | Exact implementation records |
| --- | --- | --- |
| Forests, decision trees, isolation forest | F01–F12 | [F.json](F.json) |
| Symmetric/categorical/ordered GBDT | G01–G12 | [G.json](G.json) |
| Non-symmetric GBDT and ranking | N01–N12 | [N.json](N.json) |
| Inference, DART, TreeSHAP | P01–P12 | [P.json](P.json) |

**Uncompiled, unverified, unmeasured. Quality is pending. Sample count: zero.**
These are source implementations, not validated optimizations. Every new
candidate is disabled unless explicitly selected for Apple FAST. Previous
defaults and previous evidence remain distinct. No default is promoted here.
Bits may change across versions; that permission does not relax task quality.

The new worktree branches from local `main` at `07f2b3c00`. All work is retained
on `ideas/apple-fast-trees-20261006`; there is no merge into `main`.

The [integration follow-up](INTEGRATION.md) connects these records and actual
callers to the shared A/B catalog, paired binding builds and full-workload queue.
It also records source-level connection/correctness repairs. The integration
code itself remains unexecuted and unqualified.

## Selecting future experiments

`tools/apple_fast_tree_ideas.py` is a metadata selector. Its `list`, `show`,
`select`, `pipeline-plan`, `workload-template`, and `interaction` commands emit
cards, complete A/B define sets and future orchestration contracts.
It has no compiler, executor, benchmark, queue or verification command. **The
selector itself was not run or checked in this session.** Examples for later:

```sh
python3 tools/apple_fast_tree_ideas.py list --lane F
python3 tools/apple_fast_tree_ideas.py show G01
python3 tools/apple_fast_tree_ideas.py select P05
python3 tools/apple_fast_tree_ideas.py select F01,F02
python3 tools/apple_fast_tree_ideas.py interaction X09
```

P05's existing packed-prediction prerequisite belongs in both A and B.
P01/P03 likewise require the existing ordered-resident OFF switch in both arms
to reach grove kernels. P02 retains ordered traversal and conflicts with that
shared prerequisite; such combinations are reported as blocked.
Other prerequisites and source-reading revisions are specified per card.
An experiment that is not reached by a caller cannot provide evidence for
that caller; fit and inference paths, modes and public engine selection matter.
Related-card lists are coverage obligations, not proof of runtime dispatch.

For X11 and X12, the selector requires explicit members: no complete default
configuration is proposed from unmeasured source. It does not certify those
members as qualified. Future selection must cite their actual quality evidence.
For large interaction groups it emits singles, pairs and the full combination,
not every higher-order subset. Add further combinations if future failures or
dependencies require them. No generated plan authorizes a run in this session.

## Later qualification remains separate

Follow the full-dataset/end-to-end contract and quality gates in [IDEAS.md](IDEAS.md)
and repository `AGENTS.md`. Keep all affected estimator settings and rows,
features, trees, bins, queries and outputs. Bind saved recipes to actual dataset
hashes and dimensions; no measurements, fixtures or provenance were fabricated
for this source-only task. Record fit, cold/repeated prediction and explanation
preparation/use separately. Include complete settings and resource policy, all
failures and losers, actual coverage, neighboring shapes, and a full non-board
dataset. Source paths and a successful future compile do not establish quality
or speed. Only later admitted results belong on boards, through board tools.

No unsupported Mojo/Metal workaround, CPU runtime route, lower precision,
algorithm approximation, benchmark-dimension rule or training-budget reduction
is introduced as a substitute for performance evidence.

## Delivery policy

The owner explicitly prohibited compilation, verification and measurement.
No code checks, tests, manifest checks, selector runs, hooks or GPU jobs were
run. Commit/push uses command-local `core.hooksPath=/dev/null` to prevent Git
hooks from performing prohibited verification; the shared Git configuration
and hook files are not modified. Full Git commit/push output is retained
outside the worktree in the delivery evidence directory reported at handoff.
