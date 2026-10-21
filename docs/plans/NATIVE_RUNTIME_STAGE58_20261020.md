# Selected native runtime qualification, 2026-10-20

Frozen source `72f11f70d` passes the selected public Samba, Sequential,
cross-vendor admission/checkpoint, and parallel-neural regression tests:
122 on Apple GPU; 112 on CPU with ten explicitly GPU-only skips. The unchanged
sequence closure reuses the compiled host and Metal artifacts from `cc0591cc4`;
the tokenizer binding was rebuilt. Other binding families remain retained
source-overlay assets, so this is not installed-wheel or whole-library evidence.

Real Torch mode tests pass 13/13 on both targets. Ordinary Tensor and Parameter
inputs retain their supported native paths; active Python function/dispatch modes
are refused before a hook executes. Native TensorBase descriptors bypass Python
instance method shadows. These tests qualify the selected binding boundary;
they do not make a CPython binding interpreter-independent.

Private OLS/Ridge owner, parameter/metadata and scoring tests pass 67/67 on each
target. Public classical wrappers have not switched to these owners. Archive
compatibility, public float64 prediction output, exception/subclass behavior,
named columns and metadata remain activation work. The CPU wide minimum-norm
case retains the existing explicit native refusal and verifies unfitted state;
the GPU case checks numerical results. No supported shape is silently skipped.

Original compiler errors and test-controller collection errors are retained
alongside their repairs. The separate full-graph session-lifetime failure and
its independent Modular ownership reproduction remain unresolved. Passing this
selected suite does not remove that failure or certify every public path.

Exact counts, source/binary provenance, closure reuse and artifact retention
are recorded in [native_runtime_stage58_20261020.json](native_runtime_stage58_20261020.json).
Actual source, compiled binaries and logs for stages 51–58 are verified on the
Mac and by full R2 readback at
`measurements/core-completion/2026-10-20/native-runtime/stages51-58-artifacts.tgz`.
Later archive-reader and Transformer graph changes are separate source freezes
and require their own qualification. No performance default is promoted here.
