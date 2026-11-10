# Independent class-weight migration source audit

Reviewed shared migration `26205a7311` and `2bb9064243`, followed by the
SVC/tree consolidation `68adcb6199` and admission repair `94d1704364`.
This is source review only. No compiler, product test, timing, installed
loading or identity run was executed in this lane.

## Findings and repairs

1. **Provider-table lookup before admission.** The old
   `label_weights_run` indexed the supplied table before checking its type
   or keys. A custom mapping or colliding key could therefore execute a
   hook before the later compiled-factory check. `68adcb6199` admits the
   exact dictionary and exact string keys first. `94d1704364` reuses the
   established watched/versioned dictionary-key admission cache, with a full
   key scan whenever a reliable unchanged stamp is unavailable. SVC's
   module form uses the existing exact-module descriptor admission.
2. **Tree key admission was broader than native label admission.** Tree
   constructor admission accepts exact pathlib objects for other options.
   Their Python hash/equality methods are not safe class-weight key
   operations. `94d1704364` adds key-only primitive admission before the
   retained `set(dict).difference(classes)` operation. Value-range checking
   remains after unknown-key detection, preserving supported error order.
3. **The first tree repair narrowed native mapping support.** Its exact
   builtin-dictionary requirement also rejected exact `OrderedDict` and
   `defaultdict`, previously admitted by the tree context and constructor
   validation. The producing lane owns preservation of these native mapping
   types while refusing subclasses and callback-bearing keys. This follow-up
   remains pending in this recorded source.

## Admission and mixed numeric semantics

The original shared path checks the outer dictionary type before
`PyDict_Next`, admits every key and value, admits the complete class list,
and only then builds the key set and looks up weights. Removing the old
items list does not expose an unadmitted hash/equality operation. Builtin
and NumPy numeric key comparisons still use the existing set/dictionary
semantics; keys are not coerced to Int64 or restricted to the score-key
migration's integer/text domain. Missing weights retain the default 1.0.

Classical admission preserves its global primitive-value prewalk. SVC
preserves its key-specific refusal and later generic invalid-weight error.
Tree scalar membership remains the original exact real-type membership,
including its narrower alias support. Boolean weights remain invalid.

The per-class value is admitted before conversion. Exact builtin integers
and floats use `PyFloat_AsDouble`; arbitrary-size integer overflow remains
an error. Admitted NumPy scalars are decoded from their native buffer,
without a NumPy conversion/equality slot for weight values. The binary64
conversion and exact source-unit comparison remain separate: an extended
precision value near one must not accidentally take the all-unit shortcut.
The follow-up builtin-unit optimization is valid because no integer other
than one rounds to binary64 one, and a builtin float is already binary64.

Values validate in original class order before native publication. Negative
zero, nonfinite/negative values, positive Float32 underflow, Float32 overflow
and all-zero weights retain their existing handling. The parallel plan
performs independent casts and Boolean summaries, not a new numerical
reduction. It uses the existing host worker policy and floating environment.

## Ownership and numerical work

Native `counts`, widened counts, Float64 inputs and Float32 plan storage
remain alive past every raw-address use. Counts and wide owners are consumed
only after the ratio call; inputs survive GIL-released planning; plan words
survive the synchronous gather. Class/code/output owners remain pinned at
these boundaries. The provider is retained until its GIL-released call
returns or raises; its module, capsule and factory retain the image and
entry table. Descriptor calls finish their device synchronization before
returning, so these scratch lifetimes do not rely on deferred work.

Uninitialized native scratch is valid here: CPU counts explicitly zero each
class; device counts zero their device destination and copy every class
back. Cast, ratio and successful publication write their complete outputs.
The public output is still the same zero-filled Float32 Array allocation.

Count -> widen -> binary64 ratio -> Float32 plan -> row gather remains the
pipeline. The same typed descriptor operations retain their build-time
CPU/GPU selection, upload/download and error handling. Nine numerical source
files are byte-identical to the parent of `26205a7311`; no tuned kernel,
algorithm, mode default or operation setting changed. Existing GPU/host
preparation behavior is retained; this does not claim the entire path is
GPU-only, faster, or free of performance regressions.

## Evidence and execution owed

Raw sources, patches and the one-off source-review report are preserved in
`~/mojolearn-evidence/python-mojo-migration-20261110/class-weight-independent-audit/`.
`source-audit.json` records the five-source native plan closure with zero
Python imports and nine unchanged numerical source files. This report is
not a source guard, compile result or runtime proof.

The producing lane owns the native plan fixtures and Python boundary cases,
including mapping/key hook refusal and dictionary mutation after cached
admission. Their execution remains owed to the orchestrator. Installed
provider allocation, exception cleanup, public family behavior and actual
performance require execution evidence separate from this source review.

Removed machinery reviewed: private Python scalar/parameter lists, boxed
scratch Arrays, duplicated SVC/tree preparation and legacy compiled-Python
class-weight connections. Public outputs and boundary key-set semantics
remain. No A/B define, experiment deletion, identity campaign or release
step was introduced.
