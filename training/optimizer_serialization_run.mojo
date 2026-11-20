# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/optimizer_serialization_boundary.mojo` (wave 8 run split):
which serialized optimizer fields are retained, typed Mojo only. Imports no
Python (tools/python_free_dirs.sh); body moved unchanged (lane
rs-sequence-tokenizer, 2026-11-12)."""


def optimizer_serialized_field(name: String) -> Bool:
    # Old state dictionaries may contain these retired control mirrors. They
    # are accepted only as external serialization input, never retained state.
    return name != "_native_optimizer_owner" and name != "_native_optimizer_pristine" and name != "_res" and name != "_res_binding" and name != "_res_host" and name != "_host_fresh" and name != "_pristine" and name != "_p_dev" and name != "_g_dev" and name != "_where" and name != "_KIND"
