# Recovered GPU resample gather, 2026-10-04

Branch `lane/apple-fast-resample-gpu-recovery`, base `34795a43c23f64790859977f5520048c51378771`.
Recovered narrowly from `origin/lane/apple-fast-resample@50b96e795`.
Historical `resample-rs-gather-taxi` exited 1 during source parsing and produced
no timing; manager found no istella result. This does not reuse an old speed claim.

`MOJOLEARN_RESAMPLE_FAST_GATHER` is off by default and FAST + Apple only.
`resample/gather_fast.mojo` maps one output cell per GPU thread using the
existing device-generated row indices. Current `RESAMPLE_FAST_IDX_DIRECT`
remains unchanged. Python admits only replace=True, float32 contiguous NumPy
1D/2D arrays with positive row width; `(n,0)`, empty inputs, zero samples,
other dtypes/strides/ranks/list inputs keep the current fallback. Native output
is copied into caller-owned memory and synchronized before return. No host row
index computation or changed draw algorithm is introduced.

The recovered baseline transport still allocates input/output device and pinned
staging buffers per array, with two synchronization points. This is a candidate,
not an accepted optimization. No timing or quality result has been collected.

## M2 compilation and manifest

The manager uses a full source checkout at final SOURCE. This sparse authoring
worktree is not a complete compile environment. For each arm run through the
existing slot semaphore (compile only, no kernel-launch smoke):

```sh
bash ~/mojolearn-evidence/compile_slot.sh env MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_MOJO_BUILD_FLAGS='' bash bindings/build_resample.sh
bash ~/mojolearn-evidence/compile_slot.sh env MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_MOJO_BUILD_FLAGS='-D MOJOLEARN_RESAMPLE_FAST_GATHER' bash bindings/build_resample.sh
```

Copy each output before the next build. Verify into
`~/mq/verified-arms/SOURCE/resample/{A.so,B.so,manifest.json}` with exact
source_sha, binding=resample, numeric_mode=fast, defines_A="",
defines_B="-D MOJOLEARN_RESAMPLE_FAST_GATHER", hashes={A:SHA256,B:SHA256}.

Public import also requires the declared IDENTICAL base helper artifact at
`~/mq/verified-arms/IBASE_SOURCE/ibase/_mojolearn.so`. Its manifest must state
source_sha, binding=ibase, numeric_mode=identical, defines="",
artifact="_mojolearn.so", sha256. IBASE_SOURCE may equal SOURCE or be an
ancestor differing only in explicitly allowlisted recovered gather files,
tools, and documentation. An unrelated old base is refused.

## Quality before timing

The new allowlisted helper takes exactly `SOURCE TAG IBASE_SOURCE`:

```sh
python3 tools/apple_fast_pinned_job.py SOURCE tools/resample_gpu_gather_pair.py SOURCE TAG IBASE_SOURCE
```

Manager must prepare/deploy the allowlist in the serial runner branch and run
`apple_fast_job_preflight.py` before inserting the job. Declare the resample
pair and ibase single artifact, and the three tool/native source files in the
case requirements. Preflight is metadata readiness only.

The helper checks exact source, clean tracked/untracked source, M3 Ultra,
mode/define manifests and both binary hashes. It installs only into its fresh
pinned tree, starts one process per arm, restores by removing owned artifacts,
and saves PASS.json only after exact comparison. No native build fallback,
queue edit, scored timing, or opponent run exists in this helper.

The quality oracle calls public `resample`, compares every output byte against
independent NumPy indexing using public `resample_indices`, checks actual native
gather calls, board widths11/220, odd row/block boundaries, multiple arrays,
NaN payloads/infinities/signed zero/subnormals, seeds0/7, fallbacks, refusals and
held output lifetime after subsequent gathers. Full board-size timing/admission
remains owed even if these representative quality fixtures pass.

After PASS, prepare separate one-call-per-arm taxi/istella timing with all
caller-owned outputs read in the timing boundary and receipt hash pinned. Do
not replay historical successful arms; this source has no successful scores.

Local validation: Python syntax compilation and `git diff --check` only.
Native compile, import, quality and timing are all owed to manager scheduling.
