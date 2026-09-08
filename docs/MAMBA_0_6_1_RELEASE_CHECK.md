> The 0.6.1 number was never published; this check applies to the 0.7.0 release, which ships the same combined-Linux profile.

# Mamba 0.6.1: source and retained evidence check

Source-only audit; no new test, build, timing, or GPU execution occurred.

**The earlier Mamba3 FAST key-state miss is already repaired in current source.**
`mamba/impl/mamba_ssm/modules/mamba3.mojo:424` contains `m3_dt_softplus`:
FAST uses `portable_log1pf(exp(x))` for `x <= 20`; S6 calls this helper at
line 486. This preserves small dt contributions before rotary-angle accumulation.
Reading the same helper from retained `run3/source.tar.gz` shows the same branch
and call design. No new FAST kernel patch is indicated by this audit.

Evidence root: `bench/results/resume/2026-09-06-root-feature-nvidia/`.
Its README distinguishes the original failed campaign from successful targeted
followup `run3`, snapshot `4a271ae6d719c79e2e871f4776b74c9a12f380ee`.
`run3/remote/feature-finish/results.tsv` records zero exits for
`build-mamba-fast`, `mamba-api-fast`, `build-mamba-identical`, and
`mamba-api-identical`; the FAST log ends with the passed surface statement and
successful guard terminal. The original approximately `3.98e-7` tolerance excess
must not be presented as an unresolved FAST failure at that followup snapshot.
The archive and `run3/source-snapshot.json` retain source provenance; this audit
does not assert whole-current-tree or current-wheel equality with that snapshot.

**DETERMINISTIC is a separate open validation cell.** The current helper changes
FAST only. DETERMINISTIC falls through `checks/numerics.mojo:633`
`identical_softplus`, whose portable branch selects only IDENTICAL; its other
branch evaluates `log(exp(x) + 1)`. Thus the cancellation mechanism remains
possible in DETERMINISTIC source. The retained runner
`tools/nvidia_feature_finish.sh` iterated `identical fast`, not deterministic.
This is a reason to run the deterministic fixture, not proof it fails. If it
fails at the same dt/angle seam, the bounded proposed repair is extending the
Mamba3-local stable-log1p branch to DETERMINISTIC while preserving IDENTICAL
arithmetic. Obtain root review and fresh per-mode evidence before that change;
no numerical file was edited here.

## Root-only targeted execution, serial and guarded

Use an already prepared isolated remote candidate checkout/environment. Freeze
source and corpus bytes, retain compiler/version, commands, binary hashes,
mode/vendor readbacks, full guard logs and actual exit codes. No automatic
installation, rental, broad campaign, or local Apple execution is authorized by
these example commands. On NVIDIA use the following guard; on AMD substitute
`tools/amd_serial_guard.py`. Keep compiler and BLAS threads at 2 and choose the
one actual GPU architecture explicitly. Do not execute concurrent modes.

```sh
# Root sets PY to the prepared absolute interpreter, GPU architecture and OUT
# to a fresh evidence directory; cwd is the isolated candidate checkout.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MOJOLEARN_CPU_THREADS=2 MOJOLEARN_COMPILE_JOBS=2
export PYTHONPATH="$PWD/python"
# Only if corpus generation is required, retain its generated fixture bytes:
"$PY" tools/nvidia_serial_guard.py --seconds 240 --rss-gib 12 -- \
  "$PY" mamba/corpus/gen_corpus.py

# Run separately for fast, deterministic, identical, with fresh per-mode logs.
export MOJOLEARN_NUMERIC_MODE=deterministic
"$PY" tools/nvidia_serial_guard.py --seconds 600 --rss-gib 12 -- \
  bash bindings/build_mamba.sh
"$PY" tools/nvidia_serial_guard.py --seconds 240 --rss-gib 12 -- \
  "$PY" python/mojolearn/tests/test_mamba_surface.py
```

Build requires an already resolved compatible Pixi environment; do not let a
missing environment silently bootstrap. Retain each command's real guard exit
status, including teardown failures, before continuing. This targeted surface
gate covers all three family fixtures and Mamba3 output/h_last/k_last reports.
Its `Reporter.close` gate remains **atol 1e-6, rtol 1e-5**; do not change it.
The key case is `mamba/corpus/mamba3/m3_base_b2_l4_d32`, B2/L4/model-width32,
`ref64/ssd.k_last.f64`, shape `(2,1,128)`, formerly failing flat index151
(batch1/head0/component23). Include continuation and chunk-boundary arms already
in the surface gate. IDENTICAL asserts raw-bit continuation checks; FAST and
DETERMINISTIC accuracy results cannot be renamed cross-vendor bitwise evidence.

For a diagnostic on the actual **isolated installed candidate wheel**, separately
run each mode under the corresponding guard, without checkout `PYTHONPATH`:

```sh
"$PY" tools/nvidia_serial_guard.py --seconds 120 --rss-gib 12 -- \
  env -u PYTHONPATH MOJOLEARN_NUMERIC_MODE=deterministic \
  "$WHEEL_PY" tools/mamba3_mode_state_probe.py --source-root "$PWD" \
  --expected-vendor cuda --output "$OUT/mamba3-deterministic-state.json"
```

The probe retains loaded binary identity and theta/key bits, particularly
`theta_last_[1,0,11]` and `k_last_[1,0,22:24]`. It is diagnostic, not an accuracy
gate. Pair it with the corpus surface result and independently retained wheel
build provenance. AMD changes expected vendor to `hip`. A new wheel, another
vendor, backward direction, or larger shape requires its own release evidence;
the repaired NVIDIA FAST forward fixture does not certify those scopes.
