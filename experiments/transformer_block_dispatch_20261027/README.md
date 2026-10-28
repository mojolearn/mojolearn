# Transformer fresh/backward native dispatch (2026-10-27)

This additive private descriptor table covers **the same existing** session fresh,
session backward, ephemeral fresh and ephemeral backward kernel owners. It shares
`NativeDescriptorCall`, `NeuralDescriptorProvider`, native error/result storage and
capsule admission with the primitive/Mamba/trainer tables. It adds no public
per-operation adapter, provider type, arithmetic, route default or Python fallback.

The six named operations are reserve0/release1/session-fresh2/session-backward3/
ephemeral-fresh4/ephemeral-backward5. Session operations use the existing compiled
checked handle admission entry; no extension layout is inspected. The common
provider retains the session/module/table and performs reserve/release under the
GIL, with execution released. Busy/closed guards, failure clearing, fresh zero
cache slots, options tails and ephemeral positive-shape checks are retained.
These entries cannot be used with an unadmitted arbitrary integer owner.

`MOJOLEARN_NEURAL_NATIVE_DISPATCH=1` selects this table only for the checked
identical GPU module. The ordinary default remains unchanged. Existing setup and
fresh switches still choose exactly the same session/ephemeral algorithm. Binding
replacement re-admits exact factory/module identity. CPU, packed/int15, tape and
other block families are unchanged. Fresh forward now constructs native argument
vectors directly; backward still converts its admitted boundary lists once.

This repairs direct compiled-entry dispatch for public Transformer fresh/backward;
it does **not** make Samba's outer mixed loops native or complete all29 recipes.
Public boundary metadata/state operations remain, as do session-create/handle
admission calls at setup. The main generation engine uses its existing typed
session contract; both share native output/error storage. A common interface does
not erase distinct session semantics or allow unsafe generic callable unwrapping.

Lane validation is source-only. Root compiles Transformer and tokenizer changed
closures, then runs the authored public probe on local Mac: repeated forward and
all gradient bits with session and ephemeral switches, operation counters and
refusal/recovery. Concurrent busy/close, extended options, taped backward and other
targets remain explicit pending checks. No speedup/default promotion claim follows.
