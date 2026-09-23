# Python 3.10 and 3.11 against the 0.8.15 Linux wheel (2026-09-23)

Six RunPod CPU pods (`tools/runpod_cpu_leg.sh`, 8 vCPU, $0.24/hr, image
`runpod/base:1.3.1-ubuntu2204`), each torn down and verified gone by the API.
Interpreters from uv: CPython 3.10.12, 3.11.15 and, as a control on the same
image, 3.14.7. Each venv holds `mojolearn==0.8.15` from PyPI
(`mojolearn-0.8.15-py3-none-manylinux_2_35_x86_64.whl`), numpy, pytest and
pytest-timeout. The wheel ships CPU host bindings only; tests that need a GPU
set skip or refuse.

Two arms per version, each in a copy of the origin/main checkout so the
tests' repo-relative paths resolve:

- `pytest_wheel.log`: `python/mojolearn` is the wheel's package as published,
  with main's `tests/` copied in. Tests 0.8.15 itself.
- `pytest_main.log`: `python/mojolearn` is main's source with the wheel's
  binaries under it. Tests what 0.8.16 (a Python-only patch) ships.
- `pytest_tools.log`: `tools/` and `bench/model/tests` on the same interpreter.
- `modules.tsv`: the 26 gate-style files that run as modules
  (`python -m mojolearn.tests.<name>`), exit code and seconds.

## Counts

| arm | py3.10 run1 | py3.11 run1 | py3.14 control | py3.10 final (run5) | py3.11 final (run6) |
|---|---|---|---|---|---|
| wheel (0.8.15 as published, main tests) | 71 failed, 2740 passed, 182 skipped | 72 failed, 2739 passed, 182 skipped | not run | | |
| main source over the wheel binaries | 68 failed, 2743 passed, 182 skipped | 69 failed, 2742 passed, 182 skipped | 34 failed, 2777 passed, 182 skipped | 35 failed, 2779 passed, 182 skipped | 35 failed, 2779 passed, 182 skipped |
| tools + bench/model/tests | collection aborted (INTERNALERROR, tomllib) | collection aborted (4 errors) | 22 failed, 923 passed, 7 skipped | 20 failed, 919 passed, 8 skipped (run4) | 19 failed, 926 passed, 7 skipped (run4) |
| modules (26) | 21 exit 0, 5 red | 21 exit 0, 5 red | not run | | |

Run1 is origin/main 0a89e39cb before any fix; the final columns are
lane/python-310-311 with every fix below (run5 at 020093747's parent for 3.10,
run6 at 020093747 for 3.11; run6's 3.10 arm reran the fixed files only, all
green but the one CPU-only-install cell named below). The 35th failure in both
final columns is `test_verify_all::test_shipped_verifier_hashes_like_the_harness`
at the 300 s cap; it passes alone (below). The other 34 are the 3.14 control's
34, exactly. The wheel arm's three extra failures over the main arm
are the 0.8.15 float16 safetensors defect (`test_safetensors_f16_py310.py`
x2, `test_models_loader.py::test_safetensors_reader_dtypes_round_trip_and_widen_exactly`),
already fixed on main in 4bc3fd091 and green in the main arm on both versions:
the positive control for this run.

## Version-specific breakage found and fixed (branch lane/python-310-311)

| what broke on 3.10 and 3.11 | cells | fix |
|---|---|---|
| `bytes(Array)` in test_native_nonzero: a pure-Python `__buffer__` is honored from 3.12 only | 27 | `tobytes()` (451fbae21, 246a19949) |
| test_device_array_refusal's fake CPU tensor declared `__buffer__`, exported nothing below 3.12, read as a scalar | 1 | an `array.array` subclass with the DLPack surface (d05982ab6) |
| `math.fma` in tools/attention_v2_oracle.py is 3.13+ | 7 tools tests | exact Fraction fma, bit-for-bit against math.fma on 402,197 cases (1c831c52d) |
| packaging/linux/pack_wheel.py raises SystemExit below 3.11 (tomllib) at import; tools/test_release_tracked_inventory imported it at collection and aborted the whole tools session on 3.10 | every tools test | module skip with the packer's sentence (451fbae21) |
| the bundled runtime sets PYTHONEXECUTABLE (first python3 on PATH), PYTHONPATH=":" and MOJO_PYTHON_LIBRARY when a binding loads; a `sys.executable` child on 3.11 took its prefix from PYTHONEXECUTABLE and lost the venv (no numpy). 3.10 and 3.14 ignore it on Linux | 1 (test_crossvendor_coverage CLI) | `_backend._exec_binding` restores the three variables after every binding load, at all five load sites; new tests/test_runtime_env_restored.py (dbf53877f, then 020093747: the runtime writes the C environ with setenv(), which os.environ never sees, so the restore goes through os.unsetenv and os.putenv). Mechanism in `py311/run4/probe_child.json`; the child keeps its venv in `py311/run6/probe_child.json` |

## Not version-specific (same on 3.14, left as they are; a decision is needed)

The 3.14 control on the same box fails 34 python tests and 22 tools tests
with the identical set as 3.10 and 3.11 after the fixes above. They are
CPU-only-install shaped, not Python shaped:

- 12 `test_binding_mode_guard`: asks for `fast` and `deterministic` tiers
  that a CPU-only install does not have.
- `test_host_surface` x4, `test_verify_all` x2, `test_portable_kernel_models`,
  `test_verify_reference_admit`, `test_cpu_training_misc`: they read
  `bench/results/...` records that the shipped source tarball does not carry
  (`bench/results` stays home in a CPU leg).
- `test_cpu_training_*` (6), `test_gbdt_host_modes`, `test_host_path_resolution`,
  `test_mamba23_backward_surface` x2, `test_native_convert::test_missing_symbol...`,
  `test_cpu_training_samba`: expectations written for a checkout with the
  GPU set present or the mamba corpus present (the `no CPU implementation of
  ... gbdt_fit for bootstrap_type='Bayesian'` refusal, the stale-build
  sentence, `corpus_root()` None).
- tools: `test_lane_select` x6 and `test_lane_select_harness` x2 (no git
  history in the tarball), `test_mamba3_regime_summary` x4 and
  `test_verification_evidence` (records not shipped), `test_macos_verify_wheel`
  x3 (`env` absent under `sh` on this image), `test_release`,
  `test_cpu_identity_gate`, `test_check_light_release` (git).
  `tools/test_identical_default` x3 failed on every version (relative import
  in `_backend.py` since f74fdfc22) and is fixed on the branch (71b4d9d5d).
  `tools/test_mac_slot::test_shared_deadline_includes_queue_and_execution`
  failed once on 3.10 in run4 under two concurrent suites and passed in run2
  and on 3.11: load, not version.

Decision needed: whether the CPU-only-install expectations above should skip
by name on an install without a GPU set (and without `bench/results`), or the
wheel test route should ship those records. Nothing here is a 3.10 or 3.11
defect.

The five red modules (`test_arima_surface`, `test_training_surface`,
`test_transformer_surface`, `test_mamba_surface` no corpus, `test_linalg_identity`
cannot find `tools/identity_trace_diff.py` from site-packages) are the same on
both versions and read the wheel's host bindings: `the loaded binary is the one
under identical/`, `accelerator API: cpu`, refusal sentences from the host
transformer binding. Logs under `<ver>/run1/modules_red/`.

`test_verify_all::test_shipped_verifier_hashes_like_the_harness` timed out at
the 300 s cap in run1 on both versions; alone with a 900 s cap it passed in
675 s (3.10) and 688 s (3.11) (`run4/pytest_slow.log`).

## Cost

| leg | pod | billed | spend |
|---|---|---|---|
| 1 (run1, both versions) | 6sta22lyi4e63w | 656 s | $0.0437 |
| 2 (tools arm, first fixes) | 2lhkrrs92klhy4 | ~320 s | $0.0215 |
| 3 (3.14 control, probe) | 481c3l89a77ui5 | ~625 s | $0.0417 |
| 4 (all fixes, slow test, probe) | l22cxx9i6zpb9m | ~1340 s | $0.0893 |
| 5 (env fix through os.environ, did not hold; full suite both versions) | vzzgf45v5abi16 | 560 s | $0.0373 |
| 6 (env fix through the C environ; full suite 3.11) | 9z7t1kpt1bo06a | 535 s | $0.0357 |
| total | | | $0.2692 |

Every pod deleted through the API and confirmed gone (HTTP 404 and absent
from the listing) by the leg; `leg*/teardown.txt`.
