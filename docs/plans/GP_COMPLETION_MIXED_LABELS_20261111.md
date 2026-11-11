# Mojolearn GP mixed scalar label index

Producer baseline: `5e6d28b14a`. Source only; compilation, runtime cases, performance and identity remain untested by this lane.

`bindings/gp_labels_native.mojo::_gp_sorted_classes` previously created a private `dict.fromkeys` index, then `dict(zip(classes, range(...)))` for codes. `bindings/gp_mixed_label_boundary.mojo` now owns both indices as typed hash buckets, insertion links and integer codes. The final public class list and its exact scalar identity pins remain boundary objects. These are two removed private roles; range/zip helpers are subordinate mechanisms, not extra roles.

The admitted builtin/NumPy scalar hash and equality operations remain their existing C slots. The index probes equal-hash keys in first-insertion order and uses identity before equality, with the same existing-key/incoming-key operand order. It keeps the first key object and replaces the mapped code on an equal later key. The final public list is sorted by CPython's existing list sort; there is no replacement ordering algorithm or Float64 conversion. The sorted index is built again and each original input is queried again, preserving mixed-promotion cases whose equality is not transitive. Missing lookups still write -1. NaN/mixed-kind/type admission, class count refusal and code allocation stay in their original order.

The public CPython API table gains `PyObject_Hash` and `PyList_Sort`, with layout/name V11. All old entries stay intact. Exact admitted immutable native scalar types are required before these slots are used. Partial list ownership and errors follow existing C-API conventions; input/class owners remain named through their final reads. No numerical kernel, provider request, transfer or numerical route changes.

The source reasoning was checked against CPython's public scalar/dictionary behavior and its insertion-only same-hash probe behavior; raw primary-source excerpts are retained in `~/mojolearn-evidence/python-mojo-migration-20261110/gp-completion-20261111/`. This is not runtime evidence. Four public cases exercise scalar object identity with mixed NumPy numbers, signed zero, integer hash collisions and arbitrary-size integers; they are unrun.

Build scope: the GP family/parallel/trees binding closures consume the new helper. The central CPython table layout is shared, so the integrator must rebuild all affected table consumers under the standard development-group closure calculation, not reuse outputs from layout V10. No lane compile or box job was started. General label flattening remains pending.
