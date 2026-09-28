# Consolidated GPU/CPU checks

Run only after the intended branches have been consolidated and committed:

```sh
LANES=ols,ridge bash tools/consolidated_check/mac_job.sh 0/1 /path/to/new-evidence
```

An omitted `LANES` selects every exposed lane. This is a release/development
comparison, not the lightweight installed-wheel verifier. Each lane runs its GPU
and CPU arms in fresh processes; one GPU job should own a device at a time.
`BUILD_JOBS` controls parallel compilation (default 2). `CPU_THREADS` selects
comma-separated CPU columns (default `default`).

The wrapper stops on install, build or comparison failures. Resume accepts only
matching committed sources, native library hashes, backend/architecture, lane
selection, thread columns and execution settings. A changed run needs a new
output directory. Empty, missing, duplicate or incomplete evidence is refused.
The checker does not allocate machines or submit jobs.
