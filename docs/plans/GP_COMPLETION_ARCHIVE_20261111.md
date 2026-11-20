# GP archive metadata

GPC load now borrows the admitted saved kind/parameter/length/scale Arrays as
`GPKernelView` and `GPArchiveKernelPlan`. The latter preserves the original
length-sum validation and step-one slice normalization, including negative
lengths/offsets and truncated scale tables. It neither adds validation nor
changes the persisted schema. The original `_archive_exact` dtype/order
admission and its existing C-order copy route remain unchanged.

Only final public saved-kernel nodes and scale lists are packed. A native pin
list preserves the identity of each boxed saved scale across overlapping
slices, as the previous Python scale table did. Those are public output scalar
pins, not numerical state. All four backing Arrays remain named after the last
borrowed-pointer read. Partially built public lists own their initialized slots
on errors.

`host_archive_mode_value` factors the existing string decode/refusal; the
existing `host_archive_mode` wrapper still assigns that exact public scalar for
all non-GP callers. GPC uses the returned scalar directly and drops its private
mode dictionary. No other family archive path changes.

The six distinct removed private roles are archive kind, parameter and length
unpack lists; the scale-table container; each node's temporary slice container;
and the mode-validation dictionary. The final public node tuples/lists,
persisted Array owners and selected scalar identity pins are excluded. There
are no numerical-kernel, solver, transfer-count or archive-format changes.
Compilation, runtime, measurement and identity remain unrun in this lane;
existing tests are retained. Source evidence stays under
`~/mojolearn-evidence/python-mojo-migration-20261110/gp-completion-20261111/`.
