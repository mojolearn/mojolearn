# Neural metadata boundary — 2026-10-27

The 29 GPU neural recipes are mapped individually in `inventory.json`. All are
covered by this source inventory, not by fresh runtime or timing evidence. This
lane changes demonstrated metadata reuse at admission and orchestration; it does
not add artificial provider calls to already-native training engines.

## Actual ownership and repair

| Recipes | Existing native state | This lane |
| --- | --- | --- |
| Sequential fit/predict and tape-user-model (3) | `CastIn`, model program, `FitConfig`, native tape/optimizer loops | retain one admitted owner, native shape and `CastIn` per fit/validation/predict; reuse it for sample checks and ABI serialization |
| MLP/LSTM/GRU/RNN fit (4) | native packed parameters and training loops | MLP prediction snapshots row/column counts and reuses the weight-pack destination address; recurrent training is already native |
| Session step and Program run (2) | `OpsSession`, `ShapeN`, `ProgramInput`, `OpsProgram`, `Tape` | Program admission captures provider shape/strides/dtype once before populating native descriptors |
| SmallMLP train/predict (2) | `MlpState` handle store and native input lists | existing retained native metadata; no new wrapper or cache |
| GEMM fp32/bf16/int8 (3) | explicit native dimensions/addresses at the numerical entry | one public operation per call; no model/batch loop to migrate |
| LM train/forward (2) | resident LM state and native parameter/shape records | boundary/dispatch remains a separate owner; do not call resident training Python-driven solely because its entry uses CPython |
| Transformer forward/decode (2), Mamba forward/decode (6) | native block/session state, with Python-visible model/configuration owners | new input/state at each public call needs admission; shared generation dispatch owned by the other lane |
| Samba train/forward/decode (3) | native kernels/state, mixed provider orchestration | train gradient snapshots tied-weight flag/layer count; one native dropout/RNG record per operation reused in forward/backward |
| Causal generation checkpoints (2) | resident sessions plus growing generation state | generation provider/ABI/dispatch lane owns this migration |

Sequential's `SequentialInputSnapshot` is an **edge owner** in
`bindings/sequential_arrays.mojo`. Its Python reference retains the admitted
storage. Internal dimensions, strides, kind and address use the existing
`core.cast_in.CastIn`; sample axes use a native list. No Python reference was
added to `CastIn`, `ProgramInput`, the model engine or any arithmetic kernel.
`xd`/`vxd` lists remain the existing compiled ABI wire format only. Native checks
use the retained descriptor directly, without unpacking those lists again.

The Samba dropout record contains native probability/scale, seed words and
stream ID for one gradient operation. Forward and backward reuse that record.
Changing input values, batch/sequence shape or stream on the next call captures
fresh metadata. Changing token/batch counters during execution still happens in
native state. Kernel order, floating-point expressions, seeds and algorithmic
settings are preserved; nothing selects a benchmark dimension.

Public fitted MLP arrays remain replaceable. Packing reads the current weight
owners on each prediction and retains temporary converted arrays through their
copy. It does not introduce a persistent cache keyed only by model identity.
Sequential snapshots also end at return: new input, refit, checkpoint load,
validation input, or changed dtype/layout gets fresh admission. Existing dtype,
strided-buffer, sample-shape and unsupported-protocol validation remains in place.

## Directory and dependency boundary

The target dependency direction is provider edge -> native descriptors/model
state -> native execution. No inward layer may import or transitively reach
CPython/provider dispatch. Python tests, benchmark controllers and packaging are
outside this product-runtime restriction.

| Responsibility | Existing location | Staged destination/rule |
| --- | --- | --- |
| Python user API and packaging | `python/mojolearn/` | thin public one-binding-call API; packaging/import setup separate from execution |
| Python input admission, opaque owner retention, output wrapping | `bindings/sequential_*`, `bindings/*_native`, `autograd/*_pyapi.mojo` | consolidate verified provider functions into the existing `providers/python/` edge as a later verified integration step; do not move mixed execution there and call it compliant |
| Torch/MAX whole-model admission/output | existing `adapters/` and shipped `_adapters/` | reuse existing framework adapters and `adapters/native/abi.h`; retain caller owners through synchronous return; no per-operation adapter API |
| Shared native tensor metadata | `core/cast_in.mojo`, `autograd/rankn.mojo`, `autograd/program_io.mojo` | retain these native-only implementations; any future dedicated metadata directory must preserve the same one-way dependencies, not invent a duplicate ABI |
| Persistent model state | `core/small_mlp_state_impl.mojo`, native model/program/session records | constructors, import/load, refit and setters rebuild native state; caller-owned mutable weights require explicit refresh/versioning |
| Native training/execution | `autograd/model.mojo`, `autograd/program_runtime.mojo`, `autograd/ops_api.mojo`, `sequence/`, `training/`, `gemm/`, `mamba/impl/` | keep current kernels; selected native entry closures are checked transitively with zero provider dependencies |

No directories were moved in this lane. `bindings/` is still mixed provider and
orchestration code; `autograd/` still contains both pure execution and `_pyapi`
edge modules. A folder name does not prove isolation. The new checker protects
six concrete descriptor/execution roots and their 79-module dependency closure.
It rejects direct or transitive `std.python`, `bindings`, provider, adapter and `_pyapi`
imports, plus unresolved repository dependencies. It has no allowlist of debt,
no numeric baseline and no bypass. Negative controls inject five distinct
provider dependencies through an intermediate module. This is a source import
guard, not a complete transitive callback or runtime proof for mixed paths.

Future migrations should move one verified provider edge at a time, add its
native execution root to the enforced closure, and remove mixed imports from
that closure before calling it isolated. The existing generation dispatch lane
is the owner of Transformer-specific ABI/providers. Sequential, recurrent and
Program paths must not be routed through a Transformer provider merely to share
a folder. A universal input format would also need to reconcile adapter v1's
limited contiguous/device/dtype contract with Python's strided inputs; this lane
does not silently narrow either interface.

## Public entry versus inner execution

A Python application invoking `fit`, `predict`, `Session.tensor` or `Program.run`
is a public entry. Admission captures current metadata and retains input owners.
Sequential's epochs/batches/tape and Program replay then use native records.
A benchmark's caller loop invoking `step(new_token, state)` once per token is not
an internal Python training loop: each call has new input/current state. It still
can have avoidable provider/dispatch overhead, owned by the generation lane.

Ordinary exact ndarray/Array slots, tuple/list operations and compiled-extension
calls can use CPython without executing Python bytecode. They are distinguished
from actual callback bodies such as `Array.__len__` and custom descriptor/input
hooks. No new Python fallback is introduced, and existing admission checks must
continue refusing unsupported hooks. Runtime tracing plus independent hook
controls are still required; source import checks alone do not qualify runtime.

## Validation and bounded public probe

Only source checks run here. The new public suite is authored and syntax-checked,
not executed against an old wheel. After rebuilding the affected frozen source,
run with that environment's Python:

```sh
MOJOLEARN_NUMERIC_MODE=identical python -m pytest -q \
  python/mojolearn/tests/test_neural_metadata_snapshot.py \
  --junitxml=neural-metadata-runtime.xml
```

The five bounded cases cover:

1. Sequential input owner/dtype/layout/row/value refresh, fresh fit, independent
   numerical ABI comparison, checkpoint roundtrip and output lifetime.
2. Native rank-three sample metadata and mismatched sample/validation shapes.
3. MLP prediction after replacement of its fitted bias arrays.
4. Program replay with new float32/float64/Fortran buffers, changed row count,
   native eager equivalence and output lifetime after Session close.
5. Samba gradients on two input shapes/streams against the existing independent
   explicit operation chain, including unchanged RNG state.

Existing `test_sequential_native_admission.py` and
`test_sequential_native_buffers.py` cover protocol refusals and dtype/stride
admission; `test_samba_gradients_native.py` covers mixed Mamba/attention and tied
or untied gradients. Those are candidate integration checks, not run here.
The implementation remains shared across Apple, AMD and NVIDIA. Owner steering
limits validation of this refactor campaign to the local Mac. The root compiles
and runs the candidate public probes there, recording actual coverage. AMD and
NVIDIA remain untested for this candidate; no new vendor build or measurement
campaign is a prerequisite or authorized by this lane. Their existing campaigns
continue unchanged. The 29-recipe smoke is coverage inventory, not proof that all
29 paths ran here. No Apple fast campaign, release or performance default change
is part of this work.

Two previously failing prep-plan source assertions were repaired without changing
runtime dispatch. Public `trees_public_meta_*` lifecycle doors intentionally call
internal `trees_meta_*` operations; the test now checks that exact bridge while
retaining argument-recipe checks. Keyword admission deliberately uses
`classical_admit_value` after connecting NumPy in its own context; the test now
requires that admission and the preceding exact-dict/key/allowed-name checks.
These were stale test expectations, not a demonstrated implementation defect.

Logs and retained source live under
`~/mojolearn-evidence/neural-metadata-once-20261027/`.
