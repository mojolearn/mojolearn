# Consolidated GPU/CPU checks

Run only after the intended branches have been consolidated and committed:

```sh
LANES=ols,ridge bash tools/consolidated_check/mac_job.sh 0/1 /path/to/new-evidence
```

An omitted `LANES` selects every structurally comparable CPU/GPU lane.
`host_surface.lane_exposure` supplies structural exclusions (physical multi-device
claims and surfaces with no CPU route). Every omitted lane and reason remains
in `plan.json`, alongside the complete inventory. Unexpected binding failures
fail the plan; they are not treated as structural exclusions. This is a release/development
comparison, not the lightweight installed-wheel verifier. Each lane runs its GPU
and CPU arms in fresh processes; one GPU job should own a device at a time.
`BUILD_JOBS` controls parallel compilation (default 2). `CPU_THREADS` selects
comma-separated CPU columns (default `default`).

The wrapper stops on install, build or comparison failures. Resume accepts only
matching committed sources, native library hashes, backend/architecture, lane
selection, thread columns and execution settings. A changed run needs a new
output directory. Empty, missing, duplicate or incomplete evidence is refused.
The checker does not allocate machines or submit jobs.

The default fixture is **base only**, with a **120-second timeout per arm**.
`FIXTURES=all` deliberately enables the exhaustive fixture sweep; alternatively
name fixtures such as `base,denormal,odd`. `ARM_TIMEOUT` can explicitly change
the per-arm bound. Both settings are part of resume identity. Base uses the
existing reference-compatible fixture (some algorithms slice its 20,000 rows);
this driver does not relabel stress-sized inputs as newly reduced fixtures.

Physical multi-device lanes need a separate multi-device proof; select applicable
lanes explicitly with `LANES` and preserve the omitted lane names/reasons in the
release record. A CPU refusal is never counted as AGREE by this driver.

## Saved GPU cross-vendor comparison

Local GPU/CPU agreement does not establish agreement between vendors. Compare
completed raw GPU records against the entire selected plan without running
any fits or importing native bindings:

```sh
python tools/consolidated_check/compare.py --plan RUN/plan.json \
  --column metal=APPLE_RECORDS --column hip=AMD_RECORDS \
  --fixtures base --json-out crossvendor.json
```

Directories contain `<lane>.gpu.json`; adjacent CPU records are ignored.
Add `--column cuda=NVIDIA_RECORDS` when that column is required. All named
columns must cover every planned lane and requested fixture. To monitor a
partial snapshot add `--progress`; it still reports INCOMPLETE and exits 2
until complete. AGREE exits 0; numerical differences or unstable repeats exit
1. Missing/refused records, undeclared checks, mismatched inputs, protocols,
commits or backend provenance cannot pass. Entirely N/A cells cannot pass.

Repeat a backend with a second directory to overlay targeted follow-up
records explicitly, for example `--column metal=APPLE_FIX_RECORDS` and
`--column hip=AMD_FIX_RECORDS` after the base directories. A later directory
replaces a whole lane record. Its commit must match the other columns for
that lane, while unrelated lanes can retain earlier commits. An invalid
replacement never silently falls back to older evidence. Duplicate lane
records within one directory are refused as ambiguous.

The JSON records each original file SHA-256 and source commit, the plan
SHA-256, compared numeric/NA/missing counts, and precise per-part differences.
It compares train/infer/model/reload and all recorded property hashes, never
timings. The harness omits sampler/trainer (`rlpair`) fields for lanes without
that probe; this is counted explicitly as UNRECORDED, not numeric coverage.
No reference table or manifest is changed by this tool.
