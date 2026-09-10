# WP8: ET full-capacity stage comparison

FAST Metal, real HIGGS prefix 1M rows × 28 features, 100 trees, depth16.
Scalar reference and vector candidate run interleaved inside one process.
Both use borrowed input and caller-buffer model export; WP8 is the functional
selector under comparison. Each native module reports its compiled policy,
mode and vendor. Hashes, every sample and the model hash are in summary.json.

Five measured pairs plus a warm-up per arm: minima32.650s →11.439s,
2.854× faster (65.0% lower whole-fit time). Scalar endpoint spread1.84%;
all twelve model hashes match, 1,823,474 nodes. No NVIDIA/AMD or IDENTICAL
performance claim follows from this Mac FAST column.

Build the normal FAST trees binding and preserve its DSO, then build the
same source with:

```sh
MOJOLEARN_NUMERIC_MODE=fast \
MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_ET_SCALAR_STAGE_COMPARE=1' \
MOJOLEARN_SKIP_BUILD_GATE=1 nice -n 19 tools/with_build_lock.sh \
  sh bindings/build_trees.sh
```

Preserve that DSO separately and restore the normal package artifact. Run:

```sh
PYTHONPATH=python nice -n 19 tools/with_build_lock.sh python3 \
  tools/bench_et_stage_copy.py --data /path/to/HIGGS.f32.npy \
  --scalar-so /path/to/scalar/_mojolearn_trees.so \
  --vector-so /path/to/vector/_mojolearn_trees.so
```

The run here used /tmp/mojolearn-boundary-{scalar,candidate}/fast/. The normal
FAST package artifact was restored before timing. `scalar-build.log` keeps
compiler diagnostics with trailing whitespace normalized. Vector build
and all-mode export gates are retained in ../wp2/.

Correctness: `extratrees/checks/stage_upload_bytes_check.mojo` changes every
position across vector boundaries and tails, verifies the full device output,
and corrupts only device storage to prove unchanged input actually skips a
copy. Its run log is ../wp1-wp4/wp8-fast.run.log. Final all-mode RF/ET
fingerprints in fingerprints/ report108 stable cells and216 fits, zero
moves/refusals. These small fixtures are correctness evidence, not timing.
