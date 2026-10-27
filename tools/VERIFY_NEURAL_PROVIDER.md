# Local neural provider verification

Run one command to check the migration. It does not compile, rent machines,
change measurement smoke, or run a full measurement campaign.

Source only (use a Python environment with pytest and NumPy):

```sh
python tools/verify_neural_provider.py --mode source-only --output /tmp/neural-source-check
```

The output directory must be new. Success says
`source_passed_runtime_pending`, never that the public runtime works. This runs
both native dependency closure checks, the physical provider-layout check and
the existing admission, metadata, dispatch and boundary source tests. Missing
checks are failures, not silent omissions.

After compiling the frozen candidate, run the bounded public probes on the Mac:

```sh
python tools/verify_neural_provider.py --mode runtime \
  --python /absolute/candidate/venv/bin/python \
  --package /absolute/candidate/python/mojolearn \
  --build-record /absolute/candidate/DEVELOPMENT_PROVENANCE.json \
  --build-record /absolute/development-build/result.json \
  --output /tmp/neural-candidate-check
```

`--package` names the actual `mojolearn` directory, not its parent. If the loader
uses binaries outside that tree which are not linked from it, name each location
with `--binary-root`. The candidate interpreter must contain pytest/NumPy and the
candidate's dependencies. The wrapper uses that interpreter with isolated Python
path handling and explicitly admits the supplied candidate package. Test fixtures
come only from the checkout's `mojolearn.tests` namespace; this does not add the
checkout's product package to the candidate's import search path.

Runtime mode runs the two authored metadata test files and three existing public
dispatch/provider probes in separate processes. These include actual small
training/prediction operations, generation/continuation, buffer lifetime/refresh,
checkpoint cases and dispatch negative controls. The selected probes require the
local Mac GPU and the candidate native APIs. Missing APIs, exceptions, crashes,
timeouts, skipped/xfail tests, empty collection and missing result files fail the
run. Each independent check still runs after another fails. `--timeout` sets the
per-check wall limit (default 600 seconds). This is a foreground controller, not
a background watcher.

The process enables the native neural dispatch candidate, keeps identical mode,
and the existing probes compare their specified baseline/candidate routes.
Generation metadata probes exercise the local nonresident route. Check-specific
`pending`/`not_qualified` lists remain in the reports; bounded success is called
`selected_runtime_passed`. It does not qualify all 29 measurement recipes, every
resident/fixed15/distributed/training option, CPU fallback, other vendors,
cross-vendor identity or full-workload performance. The measurement smoke's
startup/settings scope is unchanged.

## Evidence

`result.json` records every check, command, exit status, log path, timeout,
per-case counts where available, failed checks and pending scope. Each runtime
check has its own JSON containing its original probe results or pytest cases,
interpreter/platform/settings and observed loaded product files. Full stdout and
stderr stay in the adjacent logs; no filtered success lines substitute for exits.

Before runtime, the verifier hashes the explicitly supplied package/binary files.
After each worker, observed loaded product paths and hashes must match that pinned
set, and at least one native product binary must have loaded. It also checks the
candidate interpreter path/hash. An unexpected installed wheel or changed binary
fails, even when its tests passed. Imported module paths are observed through
Python's import audit events and module registry; this is not a full operating
system audit of every transitive shared-library dependency.

`source-provenance.json` retains the checkout commit, dirty status and source
hashes. Repeated `--build-record` arguments copy existing
`mojolearn.development-build.v1` results, receipts and/or
`DEVELOPMENT_PROVENANCE.json`; referenced receipt files are copied when present.
Successful and failed/incomplete build groups stay visible. A whole build need
not say `activated`: valid selected groups and unchanged wheel groups can be used
together. Loaded binary hashes are linked to matching group/replacement records.
Unchanged baseline wheel provenance stays in the overlay record. The wrapper does
not pretend checkout HEAD proves binary origin or upgrade declared build/source
records into a new release qualification.
