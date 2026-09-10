# Root host-only runtime-shape validation, 2026-09-10

No model operation, GPU numerical check, benchmark, rental or opponent timing
was run. Tree work and foreign fixed_point changes were excluded from commits.
The source changes begin at c803148a; source-sha256.json records the final
owned files and is a direct-source inventory, not a complete dependency closure.
The compiled byte-LM artifact is identified in runtime.json; its local installed
copy is ignored by Git. It was absent before this pass, so no old binary was
replaced. Rebuild on the deployment host for device execution.

## Commands and results

All builds/checks were run by root. Builds used `nice -n 19
 tools/with_build_lock.sh`, two compiler jobs, and Apple IDENTICAL where relevant.
Python validation used OMP_NUM_THREADS=2 and OPENBLAS_NUM_THREADS=2.

- `host-initial.log`: attempted Pixi-interpreter pytest; setup failed because
  that environment has no pytest. No tests ran in that attempt.
- `host-tests.log`: 62 tests plus 4 subtests passed using the existing
  `/tmp/mojolearn-root-validation-20260906/bin/python` validation environment.
- `host-tests-final.log`: 64 tests plus 4 subtests passed after adding early
  checkpoint-size rejection and load-state shape-replacement coverage.
  Command: `PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical
  /tmp/mojolearn-root-validation-20260906/bin/python -m pytest
  python/mojolearn/tests/test_byte_lm_surface.py
  python/mojolearn/tests/test_byte_lm_runtime_shape.py
  python/mojolearn/tests/test_samba_attention_wiring.py -q`.
- `mojo-config.log`: `pixi run mojo run -j 2 -I .
  training/checks/byte_lm_config_check.mojo`, host-only pass.
- `pixi-config-task.log`: final equivalent `pixi run check-byte-lm-config`, pass.
- `native-build.log`: first configured binding build, pass.
- `native-build-final.log`: final strict-integer configured binding, pass.
  Command: `MOJOLEARN_BYTE_LM_OUTDIR=/tmp/mojolearn-byte-runtime-final-build-20260910
  MOJOLEARN_NUMERIC_MODE=identical sh bindings/build_byte_lm.sh`.
  Pre-existing compiler/linker warnings retained.
- `native-host-check.log`: first compiled profile/null-span checks, pass.
- `native-host-check-final.log`: final 4 valid profiles, 7 invalid shapes,
  both ABIs reject null spans. `PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical
  .pixi/envs/default/bin/python tools/byte_lm_config_binding_check.py
  /tmp/mojolearn-byte-runtime-final-build-20260910/_mojolearn_byte_lm.so`.
  No valid borrowed pointer span is submitted; no DeviceContext is created.
- `gemm-probe-build.log`: `pixi run mojo build -j 2 --target-cpu apple-m1
  -D MOJOLEARN_COLUMN_APPLE -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I .
  gemm/checks/gemm_tuned_probe.mojo -o /tmp/mojolearn-gemm-labelled-probe-20260910`.
  Build passes; executable was not run.
- `parser-public-metadata.log`: old/relabelled historical GEMM parser equivalence
  and public runtime-shape metadata negotiation pass; no model execution.

See docs/lanes/HANDOFF_byte_runtime_2026-09-10.md for limitations and next work.
