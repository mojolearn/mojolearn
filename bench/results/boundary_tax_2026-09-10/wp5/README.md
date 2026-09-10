# WP5 local correctness evidence

Apple M4, Mojo 1.0.0 (ed45d567), source based on d9285a63 with the WP5
working-tree change. No throughput measurement or cross-vendor qualification.
All commands exited zero; compiler warning logs are omitted.

```sh
# Run each build and executable serially, separately under the build lock.
tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . checks/gbdt_cindex_staging_check.mojo -o /tmp/wp5-cindex-fast
tools/with_build_lock.sh nice -n 19 /tmp/wp5-cindex-fast
tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_DETERMINISTIC=1 checks/gbdt_cindex_staging_check.mojo -o /tmp/wp5-cindex-deterministic
tools/with_build_lock.sh nice -n 19 /tmp/wp5-cindex-deterministic
tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/gbdt_cindex_staging_check.mojo -o /tmp/wp5-cindex-identical
tools/with_build_lock.sh nice -n 19 /tmp/wp5-cindex-identical
tools/with_build_lock.sh nice -n 19 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/nan_mode_check.mojo -o /tmp/wp5-nan-identical
tools/with_build_lock.sh nice -n 19 /tmp/wp5-nan-identical
```

Each mode passes rows/features 1/19, 257/35, and 8193/67. Every compressed
word matches an independent border-count oracle and the existing columns
builder. Includes ring wraps, skipped constant columns, mixed packing policies,
NaN substitutions, signed zero, a changed-source negative control, and unseen
NaN refusal after earlier work was queued. This is staging correctness, not a
full trained-model pre-change fingerprint comparison.

Existing IDENTICAL NaN integration passes learning/border-budget/treatment
checks, 4000 predictions through save/load, unseen-NaN refusal, and a Min/Max
negative control moving 3979 rows. See the retained run logs.

OWED: NVIDIA/AMD correctness and representative large-data NVIDIA staging
A/B timings (narrow and wide, e.g. 1M rows and 2000 features where memory
permits), using five interleaved pairs after warmup and reporting baseline
drift per the brief. No small-fixture speed or default-speed claim.
