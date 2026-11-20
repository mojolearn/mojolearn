# SPDX-License-Identifier: Apache-2.0
# Which compiled door owns a `classical_native_call` operation (2026-11-07
# host split; docs/build/HOT_GROUPS.md section 6). A leaf: the classical owner
# (bindings/classical_native.mojo) reads it without importing the doors it
# forwards to (bindings/classical_trees_door.mojo,
# bindings/classical_family_door.mojo), which keeps their code out of its
# compile.

# bw2-classical (2026-11-20): the doors' connection-context keys
# ("classical_trees_door", "classical_family_door") are held keys of
# bindings/classical_keys_unpack.mojo (a str was built per forwarded call),
# and every exact name the tree door serves is a leaf of
# bindings/operation_names_classical.mojo, resolved by one table lookup. What
# is left here is the route of a name the table does not hold: the prefixes.


def classical_trees_operation(operation: String) -> Bool:
    """Whether a string operation the operation table does not hold belongs
    to the tree door: the prefixes it forwards by name (its exact names are
    all in the table)."""
    return (operation.startswith("host_infer_") or operation.startswith("trees_")
            or operation.startswith("host_tree_"))


def classical_family_operation(operation: String) -> Bool:
    """Whether a string operation belongs to the family door."""
    return operation.startswith("family_")
