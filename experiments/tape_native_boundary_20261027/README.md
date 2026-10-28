# Native whole-operation neural boundary

This source repair removes Sequential's internal CPython calls to the compiled
`model_n_params`, `model_fit`, `model_fit_labels`, `model_predict` and
`model_predict_labels` entries. Both public packed compatibility entries and the
new direct descriptor table use `autograd/model_execution.mojo`, which calls the
existing native model engine. Parameters, optimizer arithmetic, label sorting,
validation, resume, seeds and traversal order are retained.

There is one provider implementation: `providers/python/neural_descriptor_pyapi`.
The model table supplies a named engine schema to that shared provider; it does
not implement a second provider. Device and host tables differ only in executor
and existing label-sort implementation. The public Python method remains one
binding call. Input/model metadata is admitted to native lists and `CastIn`
records on each public operation. New inputs, replaced weights, refits and
validation inputs are not cached under object identity. The compiled table may
be retained while its factory/module/ABI identity matches. Buffer owners remain
live through synchronous execution; this adds no concurrent mutation guarantee.

## Actual neural recipe ownership

- `sequential-fit`, `sequential-predict`, `tape-user-model`: public Sequential
  admission -> shared descriptor provider -> model execution -> existing native
  tape/backward/optimizer/batch/epoch owners. This patch changes dispatch, not
  arithmetic. The old calls were compiled extension calls, not Python arithmetic.
- `ops-session-step`: test controller composes public calls. Native Session
  `backward` directly walks its native tape; `step` directly updates native
  optimizer state. No extra provider dispatch is added to already-native loops.
- `program-run`: Program admission constructs `ProgramInput` then directly calls
  `program_execute_run`. The captured program, replay, buffers and execution
  plan are native. Program fit directly calls `program_fit_native` and native
  `train_program_control`. Original external arrays/targets are explicitly kept
  alive until replay/fit returns.
- `mlp-fit`, `rnn-fit`, `lstm-fit`, `gru-fit`: xseq whole-model dispatch is still
  pending in this first commit. It is not qualified by the Sequential repair.
- Remaining 20 recipes: separate generation/Samba and trainer/GEMM lanes own their
  execution repair. The complete recipe map remains in
  `../neural_metadata_once_20261027/inventory.json`.

## Remaining boundary gaps

Program input admission still accepts protocols using `hasattr`, ndarray field
access and nested scalar conversion. Exact ndarray/builtin scalar routes read
metadata at admission, outside replay, but subclasses/custom protocol objects
can invoke hooks. This source patch does not claim that all supported admission
routes are callback-free. Xseq string/object label sorting also has Python
comparison/conversion work; numeric measured labels are a different route.
Sequential mode/vendor validation still calls compiled metadata entries at the
boundary. Python object metadata/output construction at an admitted edge must
not be confused with a Python callback inside native training.

## Verification

`tools/test_sequential_native_provider.py` checks the actual native dependency
closure and the absence of the removed internal dispatch calls. Live native
folder checks have no exemptions. The historical lossless relocation test reads
the original relocation commit, allowing later real product repairs without
rewriting that historical proof.

`python/mojolearn/tests/test_sequential_native_provider.py` is an authored public
candidate probe: it requires the new compiled factory, poisons the old internal
entry dictionary, and checks regression/multiclass/binary fit, resume, changed
prediction inputs and probabilities against the existing packed native engine.
It also asserts the runtime callback audit. Existing Sequential admission,
buffer, validation/early-stop, checkpoint and metadata tests remain required.
The packed oracle shares the native engine: it does not alone prove equivalence
to an older source freeze. Root owns focused Mac compilation, public execution
and the cross-freeze bit check. No runtime or performance claim follows from
source checks. Other vendors remain untested until their actual target builds.

## Xseq extension

The follow-up extends the same model table to the existing native RNN/MLP
engines, numeric label sorting, and consumed-output helpers (argmax, minimum
loss, binary threshold and seeded uniform initialization). There is still one
provider implementation, shared by Sequential and xseq. The helper route keeps
the original default-mode base-helper policy, including size limits, NaN-first
minimum semantics, sabotage/identity switches and host fallbacks. No numerical
kernel or seeded initialization order changes. Numeric labels are published by
a native dtype-preserving byte gather; string/object compatibility remains
explicitly outside the callback-free claim.

The extension admits shapes/options/addresses before each engine call and uses
only typed native lists inside the engine. MLP fitted weights remain mutable:
every prediction repacks the current layer arrays. Original target/input/output
owners stay alive through the synchronous native call.

Five authored candidate probes cover RNN/LSTM/GRU regression and binary/multiclass
MLP. They poison the old engine/helper entries, compare repeated seeded state,
check consumed predictions, and use the strict runtime callback audit. These
probes do not substitute for the frozen-source Mac bit comparison. They were
syntax/source checked only in the lane.

Remaining outside this extension: built-in optional recurrent LR schedules
still use the optimizer schedule compiled entry through CPython; xseq scoring
has separate metrics dispatch; custom object/string label handling still uses
Python-compatible comparison/conversion. Program/Session scalar/protocol
admission and Session construction remain under audit. No all-neural or
all-supported-input qualification is claimed by this commit.

## Program/Session input admission

The next repair connects the existing common admission registry during `ops`
module setup, then reuses it before reading input fields or coercing scalar
values. It does not introduce another provider. Native ndarray/scalar/container
inputs retain their conversion and shape semantics; `Array` uses the existing
zero-copy native exporter view. Python subclasses and custom dtype/float/index
hooks are refused before invocation. The native `ProgramInput` and tape remain
free of Python owners; public arguments and native scalar storage retain their
buffers for the synchronous operation. Metadata refreshes on every new call.
Embedding/gather IDs reuse the captured CastIn kind and width instead of reading
external dtype metadata again.

Authored candidate probes cover rejected attribute/float/ndarray/list hooks,
nested scalar hooks, recovery after refusal, native Array and list input,
changed dtype/layout, and consumed output ownership. Lane validation is source
only. Existing raw private ABI setup without `ops` import must explicitly connect
its admission registry before accepting inputs; the public import connects each
available compiled mode once.

Still pending: Session construction dispatches its selected mode's native
`open` entry through CPython; this cannot be removed by blindly constructing the
initializer module's executor because another mode owns a different session
registry/lifetime. Session creation is outside the current ops-session-step
measurement clock, but remains product API boundary debt. ImportedProgram and
distributed/framework admission paths have separate providers and are not
qualified by the ordinary Program input repair. Optional xseq schedule/scoring
and object-label routes remain as noted above.

## Built-in schedule continuation

Optional recurrent learning-rate schedules now call the already-linked native
`lr_spec_fill` directly with their admitted `LrSpec`. Their numerical engine,
first failing step/error text, float32 decisions, array/list output and step
conversion semantics remain unchanged. The discarded CPython round trip no
longer boxes and re-reads the same spec. The common public optimizer schedule
owner also benefits from this repair; no provider table is added to a function
that is already directly linked. An authored public probe poisons the packed
schedule entries, then checks values, scalar lookup and recurrent fitting with
StepLR. Schedule source checks passed; compiled validation remains pending.
