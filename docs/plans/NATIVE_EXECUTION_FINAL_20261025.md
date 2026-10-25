# Native execution repair, 2026-10-25

This source repair does not establish zero Python runtime or release readiness.
It is based on integration `f2ca7c3472f677d517370b978af9cc0be09d66bf`.

## Implemented

- GPC shard fit/predict and forest shard fit now hand their original arguments
  to one native entry. Their validation remains in Mojo; GPC's public NumPy
  result conversion uses the existing native public boundary. The worker
  binding getter callback is gone: import-time connection resolves lazy proxies
  and retains only compiled entries in ordinary namespaces.
- Native shard and cooperative device admission reject custom iterators before
  running their hooks. Exact tuples, lists and ranges retain their behavior.
- `ops_torch_program_run` in both sequence bindings owns Torch input selection,
  exact Tensor/Parameter and mode admission, route selection, contiguous copies,
  allocation, synchronization, program execution and output packing. Host
  staging reads admitted Torch host pointers directly; no NumPy conversion is
  needed. MPS uses the supported host route. CUDA/ROCm GPU zero keeps the
  device-buffer route. Other GPU inputs retain host staging.
- CUDA synchronization uses compiled Torch C entries and restores the caller's
  current device on success or failure. It is device-wide, which is more
  conservative than the previous stream-only synchronization. No performance
  gain is asserted. MPS output transfers explicitly synchronize natively.
- The transfer probe now times the actual native execution path and reads its
  phase timings. Its CUDA/ROCm host comparison invokes that same native entry,
  with an excluded warmup. This probe is component evidence, never a replacement
  for the required full-dataset end-to-end measurements.

## Remaining blockers and debt

`import_model(torch.nn.Module, shapes=... / example_inputs=..., numeric_mode=...)`
still calls `module.eval()` and `torch.onnx.export(..., dynamo=False)`, including
the model's Python forward function. `backend` still constructs the graph's
Python wrapper, captures parameters and imports it. `_Compiled.__call__` still
reads parameter versions and calls the Python exporter again after an in-place
weight change. These supported interfaces were preserved, not refused or
relabelled as native. The backend must remain classified as orchestration debt.

Concrete export evidence is `python/mojolearn/importers/_program.py`'s
`_export_torch`. The installed Torch source at
`.pixi/envs/test/lib/python3.14/site-packages/torch/onnx/_internal/torchscript_exporter/utils.py`
has `_trace_and_get_graph_from_model` calling
`torch.jit._get_trace_graph(model, args, ...)` (line 903 in the inspected install),
and another path explicitly calls `model(*input_args, **input_kwargs)` at 1025.
Moving these calls into Mojo would retain Python execution. No supported native
export interface eliminating execution of arbitrary Python model code was
established. ONNX-file import also still has Python validation/session setup/
result construction; that separate feasible migration is not completed here.

The neighbor/IVF worker `parallel_shard_native` operation `"py"` still invokes
registered `_parallel_shard_ops` Python functions. The remaining parallel
classical/graph/ensemble wrappers also require their own complete native doors.
This patch does not qualify those paths transitively. No blanket count of
native methods proves compliance.

## Validation and integrated checks owed

Source checks: `python3 tools/test_native_execution_boundary.py` (6 tests) and
`python3 tools/test_native_source_rule.py` (2 tests). Python syntax and
`git diff --check` are checked separately. The fake Torch tests retain a mock
execution door for capture/cache wiring; they are not evidence for Mojo runtime
behavior. No lane build, GPU run, identity run or performance run was launched.

An attempted metadata-only import of Torch in the existing test Pixi environment
aborted with exit 134, `OMP: Error #15`, duplicate `libomp.dylib`, before our code
ran. No unsafe environment override was applied. Use the clean qualification
environment for runtime checks. Full logs are retained at
`~/mojolearn-evidence/native-execution-final-20261025/`.

Compile owners: `_mojolearn_x_sequence` and `_mojolearn_x_sequence_host` for the
new Torch door, and every owner importing `parallel_shard_native`,
`parallel_shard_common`, `parallel_plan_door`, or `native_torch_admission`.
The orchestrator's dependency graph must determine that affected set.

After the integrated build:

1. Run `python/mojolearn/tests/test_parallel_gaussian_process.py`, the forest
   cases in `test_parallel_public_drivers.py`, and native callback probes for
   device iterable hooks, wrong estimator classes, NumPy-loaded/unloaded GPC
   outputs, worker binding lookup and transport failure cleanup. Test actual
   worker requests after replacing `_backend.binding` with a raising sentinel.
2. Run `python/mojolearn/tests/test_torch_compile_backend.py` and
   `test_torch_native_modes.py` against the rebuilt bindings. Exercise CPU and
   MPS staging, CUDA/ROCm GPU zero, other-device host staging, mixed-device and
   noncontiguous inputs, supported input dtypes, integer outputs, tuple/scalar
   results, changed parameters and current-device restoration after failures.
   Check exact Tensor/Parameter, subclass rejection, native method shadowing,
   TorchFunctionMode/TorchDispatchMode/PreDispatch rejection before hooks, and
   no Python frames inside the execution entry. Changed-weight export is still
   expected debt, not a negative control that can pass compliance.
3. Run the installed-wheel adapter smoke on each vendor. Do not add identity,
   verifier or fault arms for this adapter entry: its existing kernels already
   have those arms. Reuse unaffected evidence; qualify changed driver identity
   cells and two-physical-GPU negative controls through the existing campaign.
4. Retain transfer probe records and actual full-workload measurements separately.
   Include device-wide CUDA synchronization and MPS output synchronization in
   their clocks. No comparison or default promotion is justified by source work.
