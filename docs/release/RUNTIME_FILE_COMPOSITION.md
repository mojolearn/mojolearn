# Installed runtime qualification in bounded file batches

The runtime registry contains 73 whole test files. A timeout or partial pytest
progress is not a qualification receipt. Use the installed qualifier on bounded
whole-file selections, then compose their original receipts through the runtime
producer. A file must finish completely; exact-node receipts cannot substitute
for its whole-file run.

Product and tests have separate revisions. For a committed fixture correction:

```sh
python tools/qualify_native_runtime_installed.py \
  --source-root PRODUCT_SOURCE --source-commit PRODUCT_FULL_SHA \
  --test-source-commit TEST_FULL_SHA --vendor hip --numeric-mode identical \
  --wheel CORE.whl --wheel AMD.whl --python python3.11 \
  --test-file test_not_fitted_native_lifetime.py --out NEW_FILE_OUTPUT
```

The wheel witness and package Python source remain bound to `PRODUCT_FULL_SHA`.
The complete test tree is read from Git objects at `TEST_FULL_SHA`, including
export-ignored test files; dirty working files are never executed. Each worker
records its actual interpreter implementation, version, ABI, platform and
executable digest. Qualifiers predating this field remain useful focused results,
but do not supply the missing interpreter proof needed for composition.

After all bounded runs finish:

```sh
python tools/release_runtime_installed.py compose \
  --source-root PRODUCT_SOURCE --source-commit PRODUCT_FULL_SHA \
  --test-source-commit TEST_FULL_SHA --vendor amd \
  --wheel CORE.whl --wheel AMD.whl \
  --receipt FIRST_FILE_OUTPUT --receipt NEXT_FILE_OUTPUT --out NEW_COMPOSITION
```

Repeat `--receipt` for every disjoint whole-file run. The composition copies the
original receipts, JUnit, controller bytes, worker requests, inventory, logs and
pip freeze records, and keeps one shared copy of the exact wheels and committed
test tree. Retain original outputs as well, including any failed or interrupted
attempt. No aggregate JUnit or successful build receipt is generated.

Re-derivation checks wheel RECORD coverage, source witness, observed installed
module/native hashes, whole-file scope, JUnit outcomes, interpreter identity,
native-source inventories, device policy and dependency versions. Every component
must have the same complete frozen test-tree hashes. A different test commit is
acceptable only if those bytes are identical; changes to unrecorded helper
imports are not inferred harmless. A test-only correction never relabels the
product source or changes wheel identity.

Only precisely declared CPU not-applicable cases are allowed. A wholly
inapplicable file retains its original `selected-tests-not-applicable` status;
its skipped cases do not become passed tests. Every aggregate coverage ID still
needs positive applicable tests. Arbitrary skips, failures, timeouts, duplicate
files, missing interpreter identity and input drift are refused. A valid partial
union reports `incomplete` and names every missing file.

`release_runtime_installed.py summarize` re-derives both ordinary and composed
reports. The existing `installed-runtime` ledger adapter uses that same path and
revalidates original retained evidence. Only complete 73-file coverage across all
five registered IDs can enter the runtime ledger gate. This is installed runtime
coverage, not cross-vendor identity or physical multi-GPU qualification.
