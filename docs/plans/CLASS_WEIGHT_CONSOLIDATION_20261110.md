# Shared class-weight preparation across SVC and trees

Product head: `9dc30204ffaa4d881d2efcac0ae201055680b00d`.
Dependency: the previously integrated `26205a7311` / `2bb9064243` shared
class-weight plan. Integrate the ordered range `2bb9064243..9dc30204ff`:
`68adcb6199`, `217ec2323a`, `94d1704364`, `a66b9a0d3c`, `9dc30204ff`.
The documentation commit following that range changes no product source.

No compile, test, timing, installed-wheel execution or identity run was started
by this lane. The existing native plan fixtures remain unchanged. New boundary
coverage is authored and unrun. This batch is ready for the orchestrator's next
combined compile; no box job was queued here.

## Shared native state and preserved contracts

`bindings/class_weight_rows_boundary.mojo` now owns the one count -> cast ->
binary64 ratio -> native Float32 plan -> row gather pipeline. Counts, widened
counts, ratios and narrowed class weights are typed Mojo Lists. Public output
Arrays, input dictionary/scalar admission and extension-image pins remain at the
boundary. The three compile-time profiles select only the existing family
admission/provider format; numerical execution is shared.

`svc_linear_native._class_weight_rows` and
`trees_public_inputs.trees_class_weight_rows` now delegate to it. Their public
interfaces remain live. Both weighted-row callers use the same native descriptor
for the existing multiply kernel. Classical labels and Logistic continue using
the shared pipeline through their original entry.

Preserved: balanced ratios and failure on empty class counts; original scalar
validation order; default weight one for absent keys; Float64 conversion and
Float32 rounding; negative zero; positive underflow/overflow and all-zero refusals;
extended precision values near one that must not become the all-unit shortcut;
all-unit provider/output skips; sample-weight validation, copy and nonmutation;
the copied SVC classes list after publishing `classes_`.

Classical admission still requires an exact dict and admits every key/value
before class handling. SVC still admits keys before unknown-key detection and
reports its generic numerical-weight refusal afterward. Trees retain their
exact `scalar_types` membership, direct `set(mapping).difference(classes)` key
semantics and exact native dict, OrderedDict and defaultdict support. Native
dictionary lookup preserves `.get` semantics: missing classes receive one and
never invoke a defaultdict factory. SVC/classical do not gain those mapping
types. Supported mixed numeric keys, arbitrary-size integers and string keys
are not narrowed to the score-key representation.

## Callback repairs and connection removal

SVC connects `_mojolearn_x_trees` at module import through the existing family
connection registry. Execution selects its exact compiled module with
`estimator_binding_mode`, then connects the native descriptor directly. This
removes its runtime Python backend binding call and CPU facade attribute lookup
from class-weight preparation. Other SVC/QN plumbing is outside this batch.

The factory-table descriptor path now admits the exact dictionary and every
exact string key before lookup. It uses the existing mutation-aware dictionary
key admission cache; an unchanged table does not repeat a full namespace scan.
Changed dictionaries are re-admitted. An arbitrary mapping, dict subclass or
colliding custom key can no longer run a Python hook before factory refusal.

The independent audit found that general tree constructor admission also accepts
exact pathlib values, whose hashing/equality are Python methods. Tree class-weight
keys therefore receive a native primitive-only admission before set membership.
This fixes inherited callback execution; it does not remove a supported native
label type. The additional pass does not validate values early or change the
unknown-key versus numerical-value refusal order.

Removed: both duplicate boxed weight lists, private Array scratch collections,
serial SVC widening and both serial Float32 narrowing/positivity loops; their
compiled-Python count/ratio/gather/multiply dispatches; SVC's private scalar-kind
cache, itertools loading, redundant scalar comparisons and unused imports. No
kernel, algorithm, default, measured switch or fused path was deleted or changed.
Ten numerical/provider/plan source files are byte-identical to the dependency.
A/B define: none. Performance and cross-vendor identity are not measured here.

## Consumers and compile closure

Direct consumers are LinearSVC fitting; the compiled labels class-weight entry;
Logistic fitting; RandomForest, ExtraTrees and DecisionTree classifier fitting;
and weighted tree members in AdaBoost, including its existing device-round
preparation. Bagging reaches the normal tree member paths. No live interface
was removed. Tuned device-round execution remains unchanged.

The static import-closure audit inspected all 89 release roots. Nine physical
groups are affected on each of Darwin and Linux (18 platform/group records):

- `_mojolearn_native_classical_host`
- `_mojolearn_native_classical_decomp_host`
- `_mojolearn_native_classical_family_host`
- `_mojolearn_native_classical_prep_host`
- `_mojolearn_native_classical_trees_host`
- `_mojolearn_native_orchestration_host`
- `_mojolearn_native_parallel_host`
- `_mojolearn_native_xseq_host`
- `_mojolearn_native_byte_trainer_host`

The x_trees provider is an existing compiled dependency; its count, gather and
multiply kernels are unchanged. Use the orchestrator's frozen compiled provider
and normal merged group assembly. The scope audit is source evidence only.

## Coverage owed and evidence

`test_class_weight_native_plan_boundary.py` adds six cases: five provider
mapping/key/subclass/mutation refusals plus all-unit provider skipping and signed
zero publication. Its five existing numerical boundary cases remain.
`test_class_weight_shared_families.py` adds 30 cases: nine explicit sample-weight
equivalence/nonmutation cases across SVC/RF/ET; three unknown-key-before-range
cases; twelve scalar-type refusals; four OrderedDict/defaultdict cases; and two
forced pathlib key-collision refusals. Cases are independent pytest work units.
Python is retained here to exercise actual Python boundary objects and public
estimators; native plan correctness remains in the existing 24 Mojo fixtures.

RUN OWED: compile the affected groups and run the 36 new boundary cases plus the
existing five boundary cases and existing RF/QN public class-weight coverage on
the installed build. Broader performance/identity work stays with the release
process; this lane did not launch it.

Full evidence is under
`~/mojolearn-evidence/python-mojo-migration-20261110/class-weight-consolidation/`:
`closure.json`, `closure-audit.log`, the committed-source audit script,
commit logs and `push.log`. The closure record lists changed sources, exact
group members, fixture-definition counts and all ten unchanged-kernel checks.
`git diff --check` passed. Normal push passed the connection checks (10 owners,
15 literal consumers, zero errors). The previously authorized wide-rebuild
override covers inherited `native_torch_admission` history; no push hook was
bypassed and no source guard was added.

Remaining boundaries are explicit: native dict/set key admission retains
CPython mixed-key equality semantics; scalar UNPACK and output Array packing
still hold Python owners. This is not a claim that the complete SVC/tree public
operation is already Python-free.
