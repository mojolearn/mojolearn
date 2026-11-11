# GP native argument capture

`GPArgumentPins` owns fixed public input slots in a Mojo List. Each slot remains
a required Python input/provider pin at UNPACK; numerical owners continue to
create their existing typed requests. The GP lifecycle updates admitted X/y and
resolved primary/base bindings in that native record, without manufacturing
private Python list/tuple argument envelopes. Inherited explicit host methods
retain the same mode checks, module identity/readback and binding resolution
order. The existing `estimator_binding` component owns selection; its shared
implementation is unchanged.

The GPR/GPC operation overload for existing internal callers captures their
boundary tuple once. The public lifecycle already owns native slots and uses
the typed overload directly. Kernel construction likewise captures boundary
arguments once. Sum/product composition supplies a native pin list directly;
the operation/symbol objects become its required public fields, and the same
constructor validates and publishes them. There is one constructor path.

This removes two private container roles: the GP admitted/rebound argument
envelope (all ordinary/host/base-binding variants counted once), and the kernel
composition constructor tuple. It does not count input scalar/image pins or
published kernel fields as removed objects. Generic shared argument helpers
remain for their non-GP callers. No solver, numerical kernel, route, copy or
public archive format changes; no SVM helper/ABI files changed.

The root's independent GP score `raises` repair was adopted locally as
`0a919d6d6a` from `183c32da48`. It is not a producer for collection. The remaining
label flatten/mixed-numeric ordering and archive metadata work is separate.
Existing tests remain; compilation/runtime/performance/identity are unrun in
this lane. Before snapshots and ownership scope are retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/gp-completion-20261111/`.
