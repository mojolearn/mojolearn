# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the shared native label, weight and score steps
(bindings/classical_labels.mojo) that the classical owners call inside their
one native operation: the base binding's label and input helpers, the
metrics binding of both numeric modes and fixed type metadata. Nothing here
runs when a public method is called."""

#: The base binding's input and label helpers (`_buffer._native`), always the
#: identical binary (byte moves and comparisons with no tier).
_BASE_HELPERS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32", "transpose_f32", "all_finite_f32",
                 "all_finite_f64", "prepare_targets", "bincount_i64", "class_ratio_f64",
                 "gather_rows_bytes", "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
                 "unique_inverse", "reduce_stat", "cast_elements", "gather_i32", "fsum_f64")

_LABEL_CONTEXT = {}


def _refusal(error):
    # glue: a refusal recorded at import as "<Class>: <message>", raised again
    # by the native step that needs the entry
    for base in (ImportError, AttributeError, TypeError, RuntimeError, ValueError):  # glue: the fixed refusal classes
        if isinstance(error, base):
            return f"{base.__name__}: {error}"
    return f"RuntimeError: {error}"


def _helpers():
    # glue: the base binding's helpers, resolved once at import
    from ._buffer import _native
    table = {}
    for name in _BASE_HELPERS:  # glue: the fixed helper names
        try:
            table[name] = _native(name)
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            table[name] = _refusal(error)
    return table


def _metrics():
    # glue: the metrics binding's compiled entries of both numeric modes
    from types import BuiltinFunctionType
    from . import _backend
    selected = _backend.default_mode()
    connected = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes
        try:
            module = _backend.binding("_mojolearn_metrics", mode)
            table = {}
            for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode", "umap_numeric_mode"):  # glue: fixed native exports
                entry = getattr(module, name, None)
                if isinstance(entry, BuiltinFunctionType):
                    table[name] = entry
            connected[mode] = {"_mojolearn_metrics": table}
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            connected[mode] = {"_mojolearn_metrics": _refusal(error)}
    return connected


def label_context():
    """The context entries bindings/classical_labels.mojo reads, connected
    once; each owner's bridge merges them into its own context."""
    if not _LABEL_CONTEXT:
        import numbers
        import types
        _LABEL_CONTEXT.update(
            labels_helpers=_helpers(), labels_bindings=_metrics(),
            labels_real_type=numbers.Real, labels_integral_type=numbers.Integral,
            labels_namespace=types.SimpleNamespace,
        )
    return dict(_LABEL_CONTEXT)
