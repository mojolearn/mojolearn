# Native cross-vendor runtime boundary — 2026-10-20

Internal implementation and qualification record. This is not public API
qualification or a performance result.

The owner explicitly prioritized native-only execution over arbitrary Python
callback compatibility. Worker and Coordinator therefore retain callback keyword
names only to issue a clear refusal. They do not call, wrap or compile supplied
Python functions. Admission also refuses custom scalar, iterator, mapping,
buffer, PathLike and subclass attribute protocols before entering them.

The supported native path uses an exact ParallelByteLanguageModelTrainer,
BatchFeed and optional LearningRateTable. State metadata consists of ordinary
native JSON values; data buffers are exact package Arrays, NumPy ndarrays,
standard array.array, bytes/bytearray or native memoryviews. Standard PosixPath
and PurePosixPath are read without calling Python path methods. The cooperative
byte-LM trainer retains its existing CPU refusal; no fake CPU trainer replaces it.

The native runtime replaces callback uses with explicit configuration:

- `log_path` appends committed metadata through native file I/O.
- `expected_commits` compares configured state/gradient hashes, float32 loss and
  learning-rate words, and hash scheme. A mismatch records a failure event and
  closes the connection before acknowledging the update or taking another step.
- `checkpoint_steps` plus `checkpoint_dir` writes complete state at specified
  absolute steps using the existing v1 streamed archive. File names remain
  `ckpt_00000001.blm`. Writes hash raw words, fsync, and atomically replace a
  sibling temporary file; arrays are not decimal-encoded.
- `drain_events()` supplies metadata after a run. `retain_gradients=True` also
  retains applied gradient bytes, explicitly paying their memory cost.
- Both existing recipe hash schemes, `sha256.v1` and `sliced-sha256-8.v2`, are
  computed natively. Eight slices are the v2 format, not a performance dispatch.

The package CLI is one native call. The lm_segment/live_xvendor experiment
controllers prepare native feeds/schedules, launch whole native runs, and process
returned records afterward. Checkpoint creation and stopping stay inside the
run; artifact manifest/R2 uploads follow it. lm_segment retains preparation,
native run and whole-boundary durations, and the batch-table byte count. Native
batch tables own a complete selected segment; this has a host-memory cost.
Controller threads used to launch complete sessions are test infrastructure,
not worker callbacks or a package training loop. Local workers own independent
sessions; the old Python locking trainer adapter is removed.

## Evidence and pending work

- Rewritten independent socket-peer protocol tests: 13/13 pass on the unchanged
  stage42 CPU binary. The five warnings are existing native-type import
  deprecations. The test oracles never pass a Python trainer to Worker.
- An independent archive encoder matches the legacy codec byte-for-byte on an
  11,776-byte fixture containing signed zero, subnormals, infinities and NaN
  payloads. Its file SHA-256 is
  `530f38d1fc3761f9cbfc305964bb6facba5a3e13a0b60e55dc34460913f126b8`.
- New source adds strict whole-operation admission audits, independent archive
  and corruption tests, and real native GPU-worker checkpoint/expected-failure
  controls. These await the parent's single frozen tokenizer build and runtime
  checks; the lane does not compile.
- Native NVIDIA/AMD, physical multi-GPU, long-running archive, installed-package
  and updated experiment-tool execution are not established by these CPU
  controller results. No speedup or global completion claim is made.
- The independently confirmed Mojo std.python reference-retention defect remains
  an upstream lifetime blocker. This source does not compensate reference counts
  or patch the toolchain.

Full evidence is retained in
`~/mojolearn-evidence/finish-graph-20261020/cross-vendor-*` and
`checkpoint-format-oracle.{py,json,log}`. Original failures and subsequent frozen
qualification results must remain distinguishable.
