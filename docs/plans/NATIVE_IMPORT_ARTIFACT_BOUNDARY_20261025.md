# Native artifact import boundary — 2026-10-25

This is a partial runtime repair. It does not close the prohibited Python
`import_model` dispatcher, generic Torch export, or changed-weight re-export.
Those interfaces remain supported and visibly classified as debt.

## Implemented

`importers._program._import` now hands raw source/label/shapes/mode to the
compiled `ops_import_model` owner. The owner selects the existing connected
Session mode, admits standard path storage and shape metadata, opens the
inference Session, calls the existing ONNX file/graph lowering, preserves
refusal classes and metadata, constructs the ImportedProgram without Python
constructors, and closes the Session on failure. A native-reader GraphIR keeps
its reader alive and uses the native reader's buffer addresses. No graph
arithmetic, supported-op table, or importer constant-folding behavior changed.

The finite import-time connector retains compiled entries and class/storage
identities. Shape conversion retains native containers and scalar conversions,
including NumPy integer dimensions and standard tuple subclasses with native
iteration. Custom path, mapping, iteration and scalar hooks are refused before
they execute. Missing-file exceptions retain errno/filename; UnsupportedModel
retains source and ordered problems and its original message format.

`ImportedProgram.__init__` still has Python metadata work for direct calls;
this owner bypasses it internally. `importers.onnx.read` and GraphIR resolution
also retain Python metadata work on their separate public paths. The source
inventory's remaining two counted operations are not an exhaustive proof.

## Supported path assessment

| Input or operation | Available native work | What remains |
| --- | --- | --- |
| ONNX path as str/bytes/standard pathlib storage | Existing onnx_import_file, native shape resolution/lowering/capture; this change owns setup and results | Public import_model still classifies source in Python before calling this owner |
| GraphIR from the native ONNX reader | Existing onnx_import and onnx_buffers; this change owns setup/results | Creating/resolving the GraphIR through onnx.read/resolve still uses Python metadata loops |
| onnx.ModelProto | Existing reader can consume its serialized artifact | Public path calls SerializeToString and creates/removes a temporary file in Python; compiled protobuf serialization could be connected only after verifying its native implementation; protobuf's pure-Python implementation remains different |
| Exact torch.nn.Linear, ReLU, Sequential | Existing native Session linear/matmul/activation/capture and tensor admission can implement these semantics | Requires an explicit native model reader and supported configuration mapping; this is engineering work, not an established Modular limitation. No such route is delivered here |
| Captured FX graph over supported operations | Existing native opcode owners and captures provide execution | Requires native graph/attribute/constant lowering and parameter provenance; calling GraphModule forward/export does not provide it |
| Arbitrary torch.nn.Module export | Existing Torch exporter can produce ONNX by running Python model code | No supported callback-free export route was established. Moving exporter calls into Mojo is prohibited; no Python fallback or interface refusal was added |
| Changed bound weights in _Compiled.__call__ | Native Tensor admission/execution already exists | Current versions/cache checks and re-export remain Python. Replacing captured tensors alone is not equivalent to replaying export and import-time constant subgraphs |

Evidence in the installed Torch source:
`torch/onnx/_internal/torchscript_exporter/utils.py` calls
`torch.jit._get_trace_graph(model, args, ...)` around lines 893–903, calls
`_jit_pass_onnx_constant_fold` around line 1180, and reaches native
`graph._export_onnx` only after `_model_to_graph` around lines 1515–1555.
The native serializer at the end does not eliminate the preceding Python trace.
`torch.nn.Linear.forward` itself is Python dispatch to F.linear;
`Sequential.forward` is a Python module loop. Existing native owners can replace
these finite semantics without executing those methods, but an implementation
must preserve parameter layouts, hooks refusal, evaluation behavior and outputs.

The import lowerer additionally executes parameter-only subgraphs at import and
frees uploads no longer read by the captured program
(`autograd/import_lower.mojo`, onnx_import). Export can fold multiple original
weights into one initializer. Safe changed-weight updates therefore need retained
initializer/prologue provenance and a native refresh/reimport plan. General
Python forward can also change its trace based on values. No direct in-place
parameter patch or disabled constant folding is claimed equivalent here.

## Validation and qualification owed

Four source-only tests in `tools/test_native_import_boundary.py` passed. They
check raw forwarding, native registration on both legs, connection metadata and
admission-before-protocol conversion. They do not compile or run Mojo.

Direct compile owners: `_mojolearn_x_sequence` and
`_mojolearn_x_sequence_host`, both importing the new
`autograd/import_frontend_pyapi.mojo`. No compiler or runtime campaign ran in
this lane. After the integrated freeze, run:

```
python -m pytest -q python/mojolearn/tests/test_native_import_artifact_boundary.py
python -m pytest -q python/mojolearn/tests/test_import_lower_native.py python/mojolearn/tests/test_imported_run_native.py
python -m pytest -q python/mojolearn/tests/test_import_torch_module.py python/mojolearn/tests/test_torch_compile_backend.py
```

The seven new runtime cases cover native artifact setup traces, path/GraphIR
parity, NumPy shape dimensions, original error class/attributes, and protocol
refusal. Existing lowering parity compares saved program bytes and all fixture
outputs against its test oracle. Torch export/cache tests verify preserved
behavior; their passing would not establish callback-free import. Preserve the
existing clean environment requirement: the old test environment's Torch import
aborted on duplicate libomp; do not use an unsafe OpenMP override.
