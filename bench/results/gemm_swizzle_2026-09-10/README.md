# GEMM forced swizzle and bounded M4 evidence

Apple M4, macOS 26.5.2; Mojo 1.0.0 (ed45d567), IDENTICAL. No dispatch default
changed. New forced-only plan19 reuses plan10's kernel with transposed tiles.
`summary.json` gives medians, means, samples and per-shape last/first ratios.
The probe's printed ms are arithmetic means, explicitly labeled in its banner.

Root build commands used `nice -n 19 tools/with_build_lock.sh pixi run mojo build
-j 2 --target-cpu apple-m1 -D MOJOLEARN_COLUMN_APPLE
-D MOJOLEARN_NUMERIC_IDENTICAL=1 -I .`, with either
`gemm/checks/gemm_tuned_probe.mojo` or `gemm/checks/gemm_device_check.mojo`.
Both compile and execute successfully. The device check passes all7 gates,
including raw-word oracle/launch checks across all20 plans. The tuned probe
requires full-output FNV+poison equality;3 shapes pass.

Measurement: `MOJOLEARN_GEMM_BASELINE_PLAN=10 MOJOLEARN_GEMM_PLAN=19
MOJOLEARN_SPEED_ROUNDS=4 MOJOLEARN_SPEED_SHAPES=llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512
/tmp/mojolearn-gemm-swizzle-20260910`, under the same nice/lock with a120-second
process timeout. Actual elapsed9.49s. Process inventories found no other Mojo
or Pixi run before/after; the normal desktop remained active. These inventories
are not continuous GPU-utilization telemetry. Per-shape last/first drift is
roughly -0.4% to+1.3%; there is no whole-window sentinel or confidence interval.
Useful achieved baseline0.156–0.162TFLOP/s is far below the H100 result and
cannot support a model training budget based on that H100 number.

Assembly inspection: `--emit asm` on the Apple build emitted host assembly
only. Cross-compiling with `--target-triple x86_64-unknown-linux-gnu
--target-cpu x86-64-v3 --target-accelerator sm_90 -D MOJOLEARN_COLUMN_NVIDIA`
emitted PTX targetsm_90a. Current128×128KS16 (retained gzip) declares4096 local
bytes/thread and40960 shared bytes/block, with local accesses and required
RN-FMA followed by RN-FTZ multiplication. `ptx-resources.json` counts static
instructions/declarations across specializations. Virtual registers, local
allocation and static instruction counts are not physical register allocation,
additional spill counts, executed instruction counts or achieved occupancy.
No ptxas/SASS or NVIDIA execution occurred. No external opponent was run.
