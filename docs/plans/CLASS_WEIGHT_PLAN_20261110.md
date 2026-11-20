# Native shared class-weight state

Lane batch, 2026-11-10. Built on integration612d18e291 plus score-key compile repairs
73f38b4ac4/f61e960437 (local dependency merge e1b74c5a71 and repair head469f937bd7).
Only the new class-weight commit is needed by the integrator. No compile, test,
timing, installed runtime or identity run was performed by this lane.

## Changed

`core/class_weight_plan.mojo` owns native Float64 inputs, exact source-unit flags,
Float32 publication and validation. Scalar UNPACK checks each value in its original
order so a bad early value still precedes a later conversion error. Balanced
values validate in parallel; Float32 narrowing and positivity summaries also use
the existing host worker/work policy. No new serial loop processes data in RUN.
Negative zero is retained in output words; a positive value that narrows to zero,
overflow/nonfinite weights and all-zero weights retain their original refusals.
Extended-precision1+epsilon can round to Float64 one without becoming unweighted.

`bindings/classical_labels.mojo` replaces private counts, widened counts, Float64
ratios and Float32 class-weight Array objects with Mojo Lists. The boxed weight
value list is removed. Existing count -> cast -> ratio -> gather kernel calls,
request order and mathematical operations remain unchanged. Native planning uses
one GIL release after UNPACK. All scratch/source owners survive their final use.
Every connected caller now requires `label_weights_native_api_v1`; the
`NativeWeights` specialization and its compiled-Python dispatch alternative are
removed. Logistic uses the same shared entry directly.

`bindings/class_weight_boundary.mojo` admits an exact input dictionary using
PyDict_Next, removing the old items-list/tuple snapshot. Per-class lookup uses the
same CPython dictionary semantics via its stable C API. Exact builtin scalar
conversion remains in CPython; NumPy scalar bytes use the existing native scalar
decoder, preserving binary64 rounding and exact unit equality without NumPy numeric
slots or warning callbacks. Native scalar validation still occurs before advancing
to the next class. Arbitrary-size integers and supported float/numeric keys are not
narrowed to the score path's Int64/string subset.

## Removed and retained

Removed unused `labels_weighted_rows` and `labels_class_rows_only`: the repository
has no callers/imports of either. Removed Python `_class_weight_rows` entirely after
transitive audit found no product caller or re-export. Its numeric compatibility
branches were dead private machinery, not a live supported frontend. Removed its
`numbers`, portable-math, Array, buffer and label helper imports. `_class_weights.py`
now only contains `_trees_x_bind`, still re-exported by `_expansion_trees`.

Exact dictionary key-set difference remains solely inside UNPACK. This retains
CPython/NumPy mixed-numeric hash/equality and exception semantics; its temporary
Python set/list objects never carry numerical weights into RUN. Public output
Arrays and original class/code owners remain boundary objects. This is not a claim
that every Python object in all label admission has been removed.

Two independent live Mojo copies remain outside this shared-entry conversion:
`svc_linear_native._class_weight_rows` and `trees_public_inputs.trees_class_weight_rows`.
They use separate family/provider tables and caller admission/refusal contracts.
They are the next consolidation targets; they were not blindly redirected across
those contracts. The current shared-entry consumer audit finds Logistic and the
public compiled labels operation only; both now take the typed descriptor route.

No tuned kernel/default was changed. Nine relevant source files are byte-identical
to the adopted integration dependency (`unchanged-kernels.json`). A/B define:none;
end-to-end speed is unmeasured. No release or source-compliance guard was added.

## Queued coverage

24 new native fixtures cover scalar kinds, nonfinite/negative/range/zero refusals,
exact source-one versus rounded-one decisions, signed-zero output, representable
Float32 endpoints, and parallel publication through262271class values. Existing
CPU/GPU descriptor fixtures still exercise the unchanged real count/gather/multiply
kernels. Five Python-boundary cases cover NumPy/default weights, huge integer and
float keys, refusal ordering, extended-precision source equality (explicit platform
skip when unsupported), and sabotage of legacy compiled-call exports while the
typed descriptor still runs. These fixtures have not been executed.

Static source analysis:89 roots,9 physical groups per platform,0 walker errors;
the native core's five-file source closure has no Python imports. Affected groups
on both Darwin and Linux (other integration work can broaden them):

- `host:_mojolearn_native_byte_trainer_host`
- `host:_mojolearn_native_classical_decomp_host`
- `host:_mojolearn_native_classical_family_host`
- `host:_mojolearn_native_classical_host`
- `host:_mojolearn_native_classical_prep_host`
- `host:_mojolearn_native_classical_trees_host`
- `host:_mojolearn_native_orchestration_host`
- `host:_mojolearn_native_parallel_host`
- `host:_mojolearn_native_xseq_host`

Evidence:`~/mojolearn-evidence/python-mojo-migration-20261110/class-weight-plan/`
(`closure.json`, `closure-scan.log`, `unchanged-kernels.json`). Compilation,
native correctness, installed boundary execution, performance and release identity
remain separate evidence. All execution is owed to the orchestrator.
