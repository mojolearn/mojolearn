# Decomposition fit-depth admission repair

Independent NMF review at root `2963338519` found that
`decomp_public_begin` converted `_mojolearn_fitting` to an Int before
`lifecycle_begin` reached its existing exact-int admission. A rebound object
could therefore run a Python `__int__` or `__index__` hook before refusal.
The original source is retained in the NMF review evidence.

The existing guard is factored into `lifecycle_depth_value` and reused before
that conversion. `lifecycle_begin` and `lifecycle_admit_state` use the same
helper, removing their duplicate exact-type checks. The helper returns the
admitted Python integer owner so the existing completion-callback/refusal and
integer-overflow order remains unchanged in those callers. Exact integers and
connected model subclasses remain supported; unsupported conversion hooks are
refused before invocation. No model algorithm, parameter default, numerical
kernel or ownership policy changed.

One parametrized NMF boundary definition adds eight independent cases: fit,
fit_transform, transform and inverse_transform on NMF and a plain supported
subclass. Integer, index and truth hooks are traps, and the exact native refusal
message is asserted. No fit is needed to reach these admission cases.

These cases are authored and source-reviewed only, uncompiled and unrun. Root
owns the bundled build and runtime execution. No timing, identity or box work
was performed. Full source/diff/commit/push evidence is retained under
`~/mojolearn-evidence/python-mojo-migration-20261110/nmf-independent-review/`.
The broader NMF semantic and callback review continues separately; this record
does not claim the public transitive closure has passed runtime validation.
