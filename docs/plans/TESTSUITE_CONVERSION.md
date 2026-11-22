# Native checks on the standard library's TestSuite: the recipe

Lane ts-pilot, 2026-11-22 (plan: `docs/plans/CLEANUP_LATER_2026-11-25.md` section 5). The pilot converted two
families end to end: **msel** (small: 3 checks, 6 files) and **archive** (large: 12 checks, 21 files). This file is
the recipe for the remaining families, split into batches that touch disjoint files so lanes can run in parallel.

## What was measured first (cpc1, Mojo 1.2.0 (2016d0cd), 2026-11-22)

- **Importing a TestSuite file.** A source import (`-I .`) of a file that has `main()` works, both as a plain module
  and as `pkg.module` inside a package with `__init__.mojo`; the importer's own `main()` and discovery are unaffected.
  Discovery (`__functions_in_module()`) lists only the importing file's own functions, also `test_*` imported by name.
  Only `mojo package` refuses `main()` ("'main()' is not supported within packages"): that is where the AGENTS.md
  part 6 limit comes from. Rule anyway: shared helpers go in a module with no `main()` and no `test_*` (for example
  `checks/native_tests/testsuite_cases.mojo`), so no one depends on that.
- **Compile cost of discovery.** `xcnn_entry_table_suite` (667 lines, 67 functions): 19.6 s and 1.08 GB as a plain
  `main()` calling `run_checks()`, 21.2 s and 1.09 GB with TestSuite discovery over the same module (+8 % time, +2 %
  memory). `archive_minmax`: 3.9 s / 548 MB before, 5.0 s / 594 MB as TestSuite. The 15 converted files compile in
  2-9 s each at 0.2-0.6 GB peak; all 15 at once took 9 s wall. For scale: today's one compile of `run_cpu.mojo`
  (every registered check) takes 24.6 min at 311 % CPU (about 3 of 32 cores) and 19.5 GB.
- **What TestSuite gives** (behaviour of the built binaries; the stdlib ships compiled `.mojoc` only, no source in the
  pixi env): CLI `--only <test> ...`, `--skip <test> ...`, `--skip-all` (lists every test as SKIP without running it);
  any other argument is refused ("expected '--only' or '--skip'"). In code `suite.skip[test_fn]()` before
  `suite^.run()`. No per-test setup or teardown, no fixtures, no parametrization, no notion of a device or a column:
  a column is a build define (`-D MOJOLEARN_COLUMN_CPU=1`), so a column-specific test is `comptime if` inside it or
  `suite.skip[...]` in `main()` (`testsuite_cases.column_name()`).
- **A crash.** A segfault in one test kills the process with rc 139 and NOTHING of the TestSuite output survives
  when stdout is a file or a pipe (it is buffered): no PASS line of the tests before it, no summary. The runner
  therefore localises it: it lists the tests with `--skip-all` and runs each alone with `--only` (in parallel), and
  reports `CRASHED <test> rc=<rc>` as the check's first error.

## The pieces (all in this branch)

- `checks/native_tests/testsuite_cases.mojo`: `check_cases(label, n, body)` (a case table inside one `test_*`, rows
  across the host cores, every failing row named in the one raised error), `check_rows(label, rows, body)` (the
  same over a `List` of parameter structs, a failing row named by its fields), `assert_same_words` (bit-for-bit, any
  dtype), `assert_close` (rtol/atol, NaN only against NaN), `column_name()`.
- `tools/ops/native_testsuite.sh build|run|is`: builds every TestSuite file of a column to `bin/ts/<name>` (workers =
  min(cores, MemAvailable x 0.8 / `NC_TS_PEAK_MB`, default 1200)), a failed compile is that check's red row; runs one
  file as one process with the crash localisation above.
- `tools/ops/native_checks_steps.sh build` runs it beside the one `run_<column>` compile (the cores that compile leaves
  idle); `shard` runs a TestSuite check's own binary. `tools/ops/native_checks_gpu_job.sh` does the same on the L40S.
- `tools/ops/native_checks_report.py`: a TestSuite check is a file whose SECOND line is `# native-check: cpu`,
  `# native-check: gpu` or `# native-check: cpu gpu`; `list`/`plan` include it, `testsuite` lists them, `result`
  reads `Summary [ s ] N tests run: P passed , F failed , S skipped` into cases / failed_cases / skipped. A name that is
  both a TestSuite file and a registry row is refused.

The box job (`tools/ops/cpu_box_checks_job.sh`) and the GitHub workflow read the steps script from their OWN tools
tree (the box's lane tree, the workflow's commit), so a TestSuite check runs there once that tree holds this change
(`tools/cpu_central.sh tools <sha>`, orchestrator; main for GitHub). With an older tools tree a converted check reads
"no registered check named <name>": red, never silently skipped.

## The recipe (one check)

Before (`archive_codec.mojo`, wrapper and dispatch, 25 lines of 134):

```mojo
from core.host_parallel import host_parallelize
from checks.native_tests.runner import Check, Column, Report
...
def run_checks() raises:
    comptime FIXTURES = 35
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 28:
                _codec(unit)
            else:
                _span(unit - 28)
        except problem:
            print("FAIL archive_codec fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    ...
    print("PASS archive_codec:", FIXTURES, "native fixtures, parallel")

def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)

def register() -> Check:
    return Check("archive_codec", cpu=True, gpu=False, serial="", run=run_cases)
```

After (the same file; `_codec` and `_span` untouched):

```mojo
# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""...docstring unchanged..."""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
...
def test_codec() raises:
    check_cases("codec", 28, _codec)


def test_span() raises:
    check_cases("span", 7, _span)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Steps, per check:

1. If the check is a wrapper plus `<name>_suite.mojo`, the suite's content becomes `<name>.mojo` (the wrapper is
   overwritten, the suite file deleted): the check name and so the report's row name stay the same.
2. Line 2 becomes `# native-check: <columns>`, the columns of the old `register()`. Remove `run_cases`,
   `register`, the runner import, and `host_parallelize` if nothing else uses it.
3. Each fixture range of the old dispatch (`elif unit < K: _fn(unit - J)`) becomes `def test_fn() raises:
   check_cases("fn", K - J, _fn)`; a single call becomes a `test_*` calling it; named cases (`case_x()`) are renamed
   `test_x`. A function that takes more than the unit (`_valid(unit, True)`) gets a one-line adapter
   (`_valid_generic(unit)`). A row table of several parameters is a `List` of a `@fieldwise_init` struct with
   `Copyable, Writable` and `check_rows`.
4. Process-global order (a table that must be unpublished first, a registry) stays in ONE `test_*` in its old order
   (`msel_helpers_entry.test_unpublished_then_published_operations`); a test that needs a publication that is
   idempotent does it itself first, so `--only` works (`msel_split_plan`).
5. Delete the registry's two lines (import and append) for the name.
6. Delete, each with one line in the commit saying what still covers it: a case that duplicates another case, a case
   that only asserts source text or file layout, a case of a path that no longer exists. Same inputs, references,
   tolerances, messages and kernel calls otherwise: this moves test structure, not what is asserted.
7. A check that registered a `serial` reason (a global registry or the one device): its case tables loop in order
   instead of `check_cases` (write `check_cases_serial` once, in `testsuite_cases.mojo`, when the first one comes).

Mechanical help: `~/mojolearn-evidence/ts-pilot/tools/convert_fixture_check.py <repo> <name>` converts a check whose
dispatch is plain fixture ranges (steps 1-3 and the marker) and refuses anything else; steps 4-7 are by hand.

Parallelism: files run one process each across all cores (the runner), case tables across the cores inside a test
(`check_cases`). TestSuite runs a file's `test_*` one after another (the library's loop); keep tests that are long
and independent as rows of one `check_cases` table, not as separate `test_*`.

## Results of the pilot

- 15 checks converted (archive 12, msel 3); all 15 compile (2-9 s each) and pass on cpc1 (69 `test_*`, 574 table rows, the old fixture count exactly).
- Files: 27 -> 18 under `checks/native_tests` for the two families, 3,899 -> 3,592 lines (9 `_suite` files and 15
  wrapper bodies gone); `registry.mojo` 1,223 -> 1,195 lines (30 rows out, 2 docstring lines in); new
  `testsuite_cases.mojo` 147 lines and `tools/ops/native_testsuite.sh`.
- Cases deleted: none. Every fixture of the 15 checks calls a live `core/` or `bindings/` entry (all compile), none
  reads source text or layout, and no two fixtures assert the same input and output (the direct and the generic
  MinMax loaders refuse with different texts and orders, so their tables are separate behaviour).
- Check names: unchanged (347 on the CPU column, 391 on the GPU column, the same lists as the branch point).

## The remaining families, in batches of disjoint files

Groups are import-linked (a file and every native_tests file it imports or is imported by stay together; a family
prefix stays together unless that would join two groups over 40 files). Lines are the files' current lines. The full
file list per batch: `~/mojolearn-evidence/ts-pilot/batches.txt`.

| batch | files | lines | families (files) |
|---|---|---|---|
| 1 | 40 | 12,555 | core (12), ops (4), resident (1), sequence (9), training (12), transformer (2) |
| 2 | 67 | 10,612 | boundary (2), class (2), core (11), decomp (4), ensemble (16), hdbscan (5), iforest (1), nmf (4), nonzero (1), ocsvm (2), optimizer (4), portable (2), prophet (1), query (1), spectral (10), tsqr (1) |
| 3 | 84 | 10,607 | adafactor (1), bf16 (2), causal (4), cross (4), embedding (2), entry (5), et (2), gemm (7), holtwinters (3), int8 (1), kernel (4), kit (2), llama (1), lowbit (1), lr (1), nan (1), neural (2), ops (1), parallel (4), rf (2), ridge (2), sequence (1), svc (2), transformer (1), x (28) |
| 4 | 90 | 10,562 | cfb (2), cnn (6), eigh (1), exec (2), extratrees (29), forest (2), host (2), kpca (4), labels (7), mamba12 (1), pandas (2), rankn (4), transformer (1), transport (2), trees (10), xcnn (15) |
| 5 | 66 | 10,559 | affinity (2), agglo (6), arena (2), arima (2), cluster (2), dbscan (12), decomp (1), generator (1), glm (2), ipca (5), kde (2), kpca (1), lof (1), member (2), native (3), nearest (5), nmf (1), pca (7), pcs (3), small (2), tokenizer (2), transformer (1), tsvd (1) |
| 6 | 92 | 10,559 | ann (4), autograd (23), gp (15), gpc (1), gpr (2), ivf (3), kde (5), logistic (13), mamba (6), mamba3 (3), oob (1), optics (1), resample (2), samba (2), seq (2), svr (1), theta (1), tsvd (5), xtrees (2) |
| 7 | 75 | 10,544 | base (2), centroid (1), decomposition (1), finite (1), fitted (2), fresh (1), hierarchy (3), kmeans (9), linalg (2), linear (8), neighbors (25), ops (2), prep (4), qn (2), registry (2), xseq (10) |
| 8 | 61 | 10,543 | byte (6), gbdt (12), label (2), lm (1), lof (2), mani (5), metrics (5), mixture (2), models (1), sequence (5), solver (2), training (7), transformer (2), tsa (8), umap (1) |
| 9 | 135 | - | the registered modules at the top of `checks/` (`checks.<name>`): they stay where other checks import them; each gets the marker in place and the runner learns `checks/*.mojo` |

## What the runner still needs once the last registered check is converted

- Delete `checks/native_tests/registry.mojo`, `runner.mojo`, `run_cpu.mojo`, `run_gpu.mojo`, the `run_<column>`
  compile in `native_checks_steps.sh build` and `native_checks_gpu_job.sh`, the fallback branch of `shard` and of
  `--one`, the registry half of `native_checks_report.load_checks` (`_REGISTERED`, `_CHECK`, `_CHECK_LINE`), and the
  pixi tasks `check-native-cpu` / `check-native-gpu` (replace with one task that builds and runs named files).
- The box job's `build-cpu` unit becomes the TestSuite build alone: its memory estimate should be the sum of the
  concurrent compiles (today `native_testsuite.sh build` sizes its workers from MemAvailable at its start, outside the
  box's shared memory ledger: fine beside the 19 GB compile, to move into the ledger when it is the only build).
- The compile-define checks with their own pixi tasks (`check-kit-planted-cpu`, `check-border-types-sabotage`, the
  per-route-define builds named in `registry.mojo`'s docstring) build their TestSuite file with their define.
- `tools/test_native_checks_ci.py`: drop the registry assertions.
