# Host-only verifier admission, 2026-09-18

Worktree: ~/mojolearn-wt/next-wheel-coverage, branch lane/next-wheel-coverage.

The tokenized-corpus work added two harness routes but omitted them from
PUBLIC_HOST_ONLY_LANES. On a CPU installation even explicitly requesting
bpe-vocabulary or tokenized-corpus was refused by lane selection. Both are
host operations using the shipped tokenizer binding. They now participate in
default full CPU verification and explicit selection without --include-pending.

The table was generated with _verify_reference.build_table from the four
committed JSON files under bench/results/identity_break/2026-09-18_tokenized-corpus,
restricted to these two lanes. The admission policy rejected both sabotage
files, admitted the two clean two-repeat records, and yielded 18 fixture cells
(72 parts: train and the explicit infer/model/batch applicability results).
merge_reference_lanes preserved every existing cell and reference index and
added the strict admission policy per new lane. Only CPU columns were admitted;
there is no independent GPU witness or new release qualification claim.

Reproduce from the repository root with an importable package and tools on
PYTHONPATH (no native computation occurs during table generation):

```python
from pathlib import Path
from mojolearn import _verify_reference as r
from mojolearn._verify_all import load_harness
root = Path.cwd()
lanes = ['bpe-vocabulary', 'tokenized-corpus']
paths = sorted(str(p) for p in
    (root / 'bench/results/identity_break/2026-09-18_tokenized-corpus').glob('*.json'))
candidate = r.build_table(paths, load_harness(), str(root), lanes=lanes, log=print)
# Apply to the pre-admission table, not repeatedly to an already admitted table.
updated = r.merge_reference_lanes(r.load_table(), candidate, lanes)
r.write_table(updated, r.table_path())
```

Validation: 192 tests passed across test_host_only_verifier.py,
test_host_surface.py and test_verify_reference_admit.py; another 32 passed
across test_verification_coverage.py and targeted table/selection/host-routing
tests. New tests verify default/explicit reachability, shipped binding mapping,
all-nine-fixture clean/replay equality, both recorded fault controls diverging,
and CPU-only reference provenance. These tests inspect retained native results;
they do not claim a fresh native or installed-wheel execution.

Inventory: 236 routes = 163 default public CPU + 23 ordinary pending + 50
parallel (18 logical CPU, 32 GPU-required). No uncovered nonparallel route
remains outside the public/pending inventory. This is not a count of unique
algorithms, exhaustive numerical proof, or completed physical multi-GPU gates.

Release 0.8.7 remains frozen separately at release/087-final. No new tokenizer
API, harness route, or table admission is backported by this change. Run
35350125464 is still collecting the old-source CPU matrix; UMAP recording and
source-matched qualification debt remains. Continue its handoff separately.
