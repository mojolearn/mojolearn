# Native Session construction and registry lifetime

Session construction now calls the same common descriptor provider used by
Sequential and xseq. Its whole-model table adds two ownership operations:
`SESSION_OPEN` and `SESSION_CLOSE`. Each compiled target opens its own typed
`OpsSession` registry; the requested numeric mode is checked before creating an
executor. It returns a handle plus native registry identity and executor kind.
No handle is guessed or cast into another image's registry.

`NativeDescriptorResource` is a common provider lifetime record, not another
Session-specific provider. It retains the provider/module, handle and opaque
native close context. Close/finalization goes directly to that exact native
table. A failed Python boxing operation has a native RAII owner available to
close the handle. Explicit close marks it closed before dispatch, matching the
old lifetime contract. Registry mutation keeps the GIL as the original native
open/close did; numerical kernel execution is unchanged. Session calls validate
the handle against its resource's registry identity and executor kind before
accessing native state.

The compiled factory/capsule connection happens once per mode at `ops` import.
Constructor calls retain the admitted table and verify entry/module identities;
they do not call `entry.open` or `entry.numeric_mode`. CPU binding proxies keep
their public `_b` object while the provider retains each factory's actual native
module. Mode/default selection still happens for each new Session. Public
methods remain one binding call. Custom Session subclasses/attribute hooks and
foreign scalar protocols are refused before their hooks execute. This does not
promise safe concurrent mutation of Python class definitions or inputs.

The low-level packed `ops_session_create` entry and its existing typed
SessionLifetime remain for private ABI compatibility. They are not the public
Session constructor's execution path. Tape recording, backward, optimizer,
replay and numerical kernels are unchanged.

## Validation scope

Source tests check constructor dispatch, native registry dependency closure,
common resource ownership, pre-execution handle validation and authored probes.
The public candidate probe requires actual new compiled APIs, poisons old
constructor entries, exercises close twice, verifies garbage collection closes
the actual native registry, rejects replaced handles/owners, and tests an
attribute-hook negative control. Root must execute it on the compiled candidate;
source checks alone do not qualify lifetime behavior or bits. Apple validation
is focused; no lane compilation, GPU run or measurement is authorized.

`framework-audit.json` records the separate ImportedProgram/Torch/MAX audit.
Those paths are not repaired or qualified by this constructor change. The actual
29-recipe `program-run` uses ordinary Program.run; torch.compile/imported-model
runtime paths are separate from that measured recipe.


The native ImportedProgram constructor already calls this Session initializer
with the same import-time connection dictionary and exact owner classes. It
therefore receives the same native resource; its existing graph-reader binding
identity check still sees the original public `_b` proxy. Source compatibility
is covered, but actual imported-artifact construction and failure cleanup still
need a compiled runtime check. Internal ONNX entry dispatch and ImportedProgram
execution gaps are listed separately in the audit.

Rebuild scope includes tokenizer-host (registration of the common resource type)
and x_sequence/host images (new session table operations and frontend admission).
The selected development artifacts must travel together; old binaries cannot
qualify these source changes.
