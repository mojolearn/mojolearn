# Installed public native-runtime audit

`tools/qualify_native_runtime_installed.py` installs actual wheel bytes into a
new venv, outside the checkout, then runs frozen public-operation tests. It
never compiles or changes a binary, wheel source witness, loader or oracle.

Use parent-qualified frozen native outputs. On Linux, the existing local
qualification assembler can combine recorded installed assets with explicit
native overlays. Every retained asset remains identified; source equivalence
and successful native compilation are **not** inferred from an overlay or a
passing installed test. Supply the original native receipts with `--native-proof`.
They are copied and hashed for review, not promoted to release proofs.

```sh
python3 tools/qualify_native_runtime_installed.py \
  --source-root /root/frozen-source --source-commit FULL_40_CHAR_COMMIT \
  --installed-site /root/retained-venv/lib/python3.11/site-packages \
  --native-overlay /root/qualified-native-overlays.json \
  --native-proof /root/qualified-build-receipt.json \
  --vendor hip --out /root/results-installed-native-audit
```

`qualified-native-overlays.json` uses the existing
`packaging/linux/qualification_wheels.py` schema: each row names `member`,
`path`, `source_commit`, and retained `build_evidence` files. The assembler
verifies old installed RECORDs and commits current package source; it does
not falsify retained PTX source manifests. A stale PTX set must still refuse.
Its local `linux_x86_64` wheels are not publishable releases.

For already-built Mac or Linux wheels, repeat `--wheel` for the complete core
and selected vendor distribution set. This skips assembly, copies actual wheel
bytes into the evidence directory, and uses the same install/test gates:

```sh
python3 tools/qualify_native_runtime_installed.py \
  --source-root /absolute/frozen-source --source-commit FULL_40_CHAR_COMMIT \
  --wheel /absolute/mojolearn_candidate.whl \
  --native-proof /absolute/qualified-build-receipt.json \
  --vendor metal --out /absolute/new-installed-audit
```

Both routes require a new output directory. Every wheel RECORD entry is checked;
shipped package Python source must match the requested committed freeze. Source
and test hashes, native asset hashes, original build-proof bytes, full pip logs,
pytest logs, JUnit failures/skips, and observed installed module paths/hashes are
retained. Cached extension modules absent from `sys.modules` are also checked.
Tests are copied separately and exposed only as `mojolearn.tests`; no checkout
path is added to the import path. Every observed product module must come from
venv site-packages and match the installed wheel bytes. Inherited mojolearn
binary/path/backend overrides are removed; the requested vendor and mode are
set explicitly and read back before testing.

The default selected scope is public Samba training/inference/checkpoints and
admission, Sequential fit/predict/score/checkpoints and admission, Generator
draws, and CausalLM generation/stopping/stream/resume. It reuses the existing
independent numerical/state oracles and whole-operation native audits. Use
repeatable `--test-file` choices for a smaller declared subset. NumPy and pytest
are installed as test dependencies; additional test dependencies can be named
with `--dependency`. Product preparation or computation is never moved into
the testing controller to hide a runtime violation.

`receipt.json` reports `selected-tests-passed` only with a nonempty pytest run,
zero failures, zero skips, and the installed-origin receipt. Original failures
remain in the new run directory. `release_qualified`, `coverage_complete`, and
`native_source_reuse_qualified` remain false: these selected public checks do
not establish whole-library coverage, multi-GPU evidence, all vendor identity,
or the provenance of a mixed retained binary set. Review the supplied original
native receipts and observed loaded hashes separately.

Controller validation: six file-only negative-control tests cover tampered or
unrecorded wheel bytes, duplicate split ownership, source imports, unshipped
cached extensions, altered installed bytes, escaping test modules, and distinct
preservation of test failures/skips. Actual installed public-runtime execution
is pending the parent's green guard freeze; no lane compilation was performed.
