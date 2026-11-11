# Mojolearn GP flattened scalar pins

The private flattened Python list in `gp_labels_native::gp_encode_labels_into` becomes native `List[PythonObject]` storage. Its elements are admitted public scalar identity pins; they are not converted to approximate numeric keys or removed from the boundary. This closes one private-container role, not the complete general-input conversion mechanism.

`classical_labels::_flatten_into_source` is the shared existing depth-first walk with a compile-time output sink. The public Python-list wrapper remains for its live callers. Exact list/tuple recognition, nonnested admission-before-copy, depth limit and refusal order remain. The native output is admitted with one existing classical module/type plan and the same list-leaf depth and builtin shortcuts, through an additive native-vector overload. It does not rebuild that plan for every element.

Native pin adapters share the existing text reader bodies: UTF-8 hashing/distinct-class sorting, surrogate code-point reader/merge sort, primitive scalar-kind scan and exact-string scan. The former algorithms are not copied or replaced. The integer label plan consumes its existing native pin overload; its GIL guard now keeps a named interpreter owner, matching the established integration repair. Mixed scalar indices likewise read native pins. The numeric fast paths, code allocation, base encoder/gather routes and public class identity packing remain unchanged.

Source audit: five text-table/sort bodies are byte-for-byte unchanged, and no core file changed. Two new nested-text public cases exercise NumPy string identity and the surrogate path; they are authored and unrun. Evidence is `~/mojolearn-evidence/python-mojo-migration-20261110/gp-completion-20261111/flatten/audit.json` with complete before source and patch. Compilation, runtime, timing and cross-device identity remain owed to the orchestrator.

Still pending: `_label_list` can construct input-conversion lists for non-list buffers/pandas and the GP Array snapshot helper can construct nested lists. The shared GP score truth flatten path and prepared-owner field admission also need review. This producer does not claim the whole family is Python-free.

Build roots: all binding closures of shared `classical_labels`, `classical_pandas`, `p3_labels_metrics_native`, `label_input_sources`, `str_labels_native` and `perf3_label_doors_text`, plus GP family/parallel/trees roots; the standard dependency planner determines the full affected set. No box job, build or test was started by this lane.
