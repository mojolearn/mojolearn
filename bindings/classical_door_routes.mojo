# SPDX-License-Identifier: Apache-2.0
# Which compiled door owns a `classical_native_call` operation (2026-11-07
# host split; docs/build/HOT_GROUPS.md section 6). A leaf: the classical owner
# (bindings/classical_native.mojo) reads it without importing the doors it
# forwards to (bindings/classical_trees_door.mojo,
# bindings/classical_family_door.mojo), which keeps their code out of its
# compile.

#: Connection-context keys of the doors' retained compiled entries
#: (python/mojolearn/_native_classical_bridge.py `_classical_doors`).
comptime CLASSICAL_TREES_DOOR = "classical_trees_door"
comptime CLASSICAL_FAMILY_DOOR = "classical_family_door"


def classical_trees_operation(operation: String) -> Bool:
    """Whether a string operation belongs to the tree door: the names the
    classical door's own branches matched before the split."""
    return (operation == "host_kde_fit" or operation == "host_iforest_fit"
            or operation == "host_density_save" or operation == "host_density_hash"
            or operation == "host_density_initialize"
            or operation.startswith("host_infer_")
            or operation == "archive_infer_predict" or operation == "archive_infer_predict_proba"
            or operation == "archive_knn_classifier" or operation == "archive_knn_regressor"
            or operation.startswith("trees_") or operation.startswith("host_tree_")
            or operation == "host_archive_predict" or operation == "host_archive_predict_proba"
            or operation == "host_archive_model" or operation == "host_archive_load"
            or operation == "host_archive_classical_model")


def classical_family_operation(operation: String) -> Bool:
    """Whether a string operation belongs to the family door."""
    return operation.startswith("family_")
