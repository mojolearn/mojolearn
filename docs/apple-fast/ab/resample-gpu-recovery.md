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

## Verified r2 artifacts and tools-only harness

Compiled A/B and same-source IDENTICAL base pin:
`7eacaa2b2c1a6fa84fa3aa6e4b7d6c14b2d0d817`.
The r2 native change uses fixed-width Int32 kernel dimensions, widened locally
for indexing. The harness branch `lane/apple-fast-resample-gpu-harness-r1`
permits only tools/docs descendant drift from the compiled source, records the
separate harness SHA, and retains exact mode/defines/hash checks. No rebuild is
needed for these helper-only changes. Quality harness checkpoint
`b5c81d0a808254fe73e440b742997415a0841025` is independently ready.

`resample_gpu_gather_spec.py quality TAG` prints the exact quality preflight
spec for the current harness commit. Artifacts are r2 resample A/B plus r2 ibase.

After quality PASS, `resample_gpu_gather_spec.py timing TAG taxi QUALITY_JSON`
(or `istella`) prints a timing preflight spec, including the exact receipt hash
and the existing board NPZ/metadata hashes. This metadata helper reads the
files, so run it in a serial readiness window, outside scored work. A partial
or failed quality job never admits timing. Root remains sole queue writer.

The timing helper directly invokes unchanged `bench_board_algos.py worker`
with lane=resample, arm=ours-fast, and the original rows-full reg-taxi/reg-istella
files. It matches the board's `race(rounds=1)` protocol: `round 0` is one
unscored warmup, `round 1` is the single scored call, followed by `save`, `quit`.
The discarded cold-call draft never produced scored results. No opponent,
build fallback, replay or extra scored round is permitted.

This is the actual board operation/data, not a representative synthetic timer.
Its original `_build_fn` includes conversion and means over all returned X/y
cells within `runner.fit`, so the first complete output read stays inside the
clock. It saves means and the board digest outside that timed call and requires
exact A/B agreement. Input header gates require X shapes1,000,000x11 (taxi) /
1,000,000x220 (istella), float32 C-order, with the corresponding y and Xq.

The measurement reports scored_calls_per_arm=1 and warmup_calls=1. It streams
all worker protocol messages to disk so even timeout/failure preserves any
completed round. Each successful arm is retained; a later failure never
licenses replay of its completed scored round. The output tag is exclusive.


## Quality harness refusal capture repair

`resample-gpu-recovered-q-r1-20261004` on harness
`b5c81d0a808254fe73e440b742997415a0841025` failed in arm A's deliberate invalid
input loop: main returned plain Python `Exception: mojolearn: null int32 buffer
address` for zero output length, while the capture only caught ValueError and
RuntimeError. The original A.log/tag is retained; no numeric mismatch or timing
was produced. B did not run.

Capture now recognizes native plain Exception only for the resample/mojolearn
error prefixes and preserves the exact exception type/message. Other exception
classes still escape. A/B refusal equality, output byte equality, lifetime and
reach requirements are unchanged; no native source, threshold or fixture changed.
Retry quality with a new tag and this new harness pin against the same r2
compiled A/B and ibase artifacts. No scored arm is replayed.


## Timing admission after quality PASS

`resample-gpu-recovered-q-r2-20261004-quality/PASS.json` passed exact output,
refusal, lifetime and reach gates on harness
`7d66c0a052cd55d97b984caf521af32f5d12e229`, compiled r2 pair/base unchanged.
Receipt SHA256: `84812e78275dff3dc61625b4d07dc8ae4ae7327950e5de03e063f7ab285165fe`.
Taxi and istella must have distinct fresh timing tags/specs and run serially.
The shared policy is `verified-scoped-caller`. The timing helper permits the
reviewed timing/spec/policy/docs-only drift since the quality harness and
checks the unchanged quality-script hash. No new native build is owed.
