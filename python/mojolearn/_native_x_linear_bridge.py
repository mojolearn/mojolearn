# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the linear expansion
estimators and the coordinate-descent pair (bindings/x_linear_native_door.mojo)
and its fixed ABI. Nothing here runs when a public method is called."""

#: The base binding's fold, index and score helpers the cv fits call
#: (`_buffer._native`, always the identical binary). Lane mf-split5: `fold_ids`
#: and `select_fold_i64` went (the default KFold folds are the splitters' own
#: plan, the integer fold ids one typed body, x_linear/cv_fold_ids_run.mojo).
_BASE_HELPERS = ("assign_fold_i64", "check_indices_i64", "count_fold_hits_i64",
                 "mark_split_roles_f32", "row_means_f64", "gather_rows_bytes", "reduce_stat")


def _refusal(error):
    # glue: a refusal recorded at import, raised again by the operation that needs it
    return type(error).__name__ + ": " + str(error)


def _linear_entries(module):
    # glue: the canonical native connection of one x_linear binding, admitted
    # once at import against the table it publishes. The owner calls that
    # binding's x_linear operations table (core/linear_entry_table_ext.mojo)
    # through this connection's key; no compiled Python entry is retained.
    from ._native_classical_bridge import _native_table_cells
    return _native_table_cells(module, "x_linear")


def connect(tables, solver=None):
    """`(call, context)` for the linear expansion estimators.

    `tables` holds the fit's plain name tables (the SGD loss, penalty and
    learning-rate codes, the summed-step losses, ElasticNet's selection
    codes); `solver` is `_solver_impl`'s `_mojolearn_solver` module (None
    for the x_linear estimators, which never call it)."""
    # glue: compiled entries and plain metadata are retained during import
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context.update(label_context())
    selected = _backend.default_mode()
    linear, metrics = {}, {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            linear[mode] = _linear_entries(_backend.binding("_mojolearn_x_linear", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            linear[mode] = _refusal(error)
        try:
            metrics[mode] = _x_metrics_table(_backend.binding("_mojolearn_x_metrics", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metrics[mode] = _refusal(error)
    cd, host_predict = "ImportError: the coordinate-descent solver is not connected here", None
    cd_modes = {"identical": cd, "fast": cd}
    if solver is not None:
        cd = _connect_cd(solver)
        # An explicit ElasticNet / Lasso numeric_mode selects that tier's image; None keeps `cd`.
        cd_modes[selected] = cd
        other = "fast" if selected == "identical" else "identical"
        try:
            cd_modes[other] = _connect_cd(_backend.binding("_mojolearn_solver", other))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            cd_modes[other] = _refusal(error)
        # an explicit host Lasso / ElasticNet predicts through the classical
        # door's host entry (`_HostCD.predict`)
        from .linear_model import _native_classical, _CLASSICAL
        host_predict = (_native_classical, _CLASSICAL)
    call = _binding()._module.x_linear_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("the linear estimators require a compiled x_linear_native_call")
    import os
    import decimal
    from ._buffer import _native
    from ._portable_math import _POW_CONTEXT
    from ._expansion_metrics import UndefinedMetricWarning
    helpers = {}
    for name in _BASE_HELPERS:  # glue: the fixed helper names, resolved once
        try:
            helpers[name] = _native(name)
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            helpers[name] = _refusal(error)
    # glue: once at import, the base helper table key the native door reads
    # (lane calltable-base-helpers); a binary without one records the refusal
    from ._buffer import base_native_key
    try:
        helpers["base_native_key"] = base_native_key()
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        helpers["base_native_key"] = _refusal(error)
    from ._labels import ClassLabels
    # The connected public class-label type native classifier fits publish.
    context.update(classifier_labels_class=ClassLabels, classifier_labels_namespace=dict(vars(ClassLabels)),
                   classifier_labels_mro=type.__dict__["__mro__"].__get__(ClassLabels))
    context.update(xl_helpers=helpers, xl_undefined_warning=UndefinedMetricWarning,
                   xl_decimal=decimal.Decimal, xl_pow_context=_POW_CONTEXT)
    context.update(xl_bindings=linear, classical_metrics=metrics, cd_solver=cd, cd_solver_modes=cd_modes,
                   cd_host_predict=host_predict, environment=os.environ._data,
                   xl_sgd_clf_loss=dict(tables.get("sgd_clf_loss", {})), xl_sgd_reg_loss=dict(tables.get("sgd_reg_loss", {})),
                   xl_sgd_penalty=dict(tables.get("sgd_penalty", {})), xl_sgd_lr=dict(tables.get("sgd_lr", {})),
                   xl_sgd_sum_losses=frozenset(tables.get("sgd_sum_losses", ())),
                   cd_selection=dict(tables.get("cd_selection", {})))
    _connect_cpu_device(context, solver is not None)
    return call, context


def _x_metrics_table(module):
    # glue: the metric-program entries of one x_metrics image, checked once at import
    from types import BuiltinFunctionType
    table = {}
    for name in ("x_metrics_run", "x_metrics_vendor", "x_metrics_numeric_mode",  # glue: fixed native exports
                 "x_metrics_run_out", "x_metrics_run_ranges", "metrics_program_native_api_v1"):
        entry = getattr(module, name, None)
        if entry is None and name in ("x_metrics_run_out", "x_metrics_run_ranges"):
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"the x_linear owner's score requires compiled {name}")
        table[name] = entry
    return table


#: The CPU identical images `device="cpu"` runs on a GPU installation, by the
#: device binding each one is the CPU column of: the x_linear operations, the
#: coordinate-descent solver (Lasso / ElasticNet), and the two metrics images
#: a score runs (the metric programs of accuracy and R^2, the R^2 kernel).
_CPU_DEVICE_BINDINGS = ("_mojolearn_x_linear", "_mojolearn_solver", "_mojolearn_x_metrics", "_mojolearn_metrics")


def _connect_cpu_device(context, with_solver):
    """Register the CPU images of `device="cpu"` beside the device images:
    they load at the first device="cpu" admission beside a GPU set
    (fx-host-import, `_native_family_bridge.cpu_load_later`), never at
    import; an estimator restored without construction (and an explicit
    host inference class) connects them at its first CPU route
    (bindings/x_linear_route_unpack.mojo `_xl_cpu_table`). Nothing is
    registered in a CPU-only process."""
    # glue: import-time registration, once
    from . import _backend
    if context["vendor"] == "cpu" or _backend._CPU_ONLY is not None:
        # A CPU-only process: its own images are the CPU ones. The text (not
        # a missing key) tells the explicit host class's route
        # (`xl_route_host`, once per operation) to run in this context
        # without a late-connect probe.
        context["xl_cpu_scope"] = "ImportError: this installation is CPU only; its own images are the CPU ones"
        return
    from ._native_family_bridge import cpu_load_later
    cpu_load_later(context, lambda: _load_cpu_device(context, with_solver))


def _load_cpu_device(context, with_solver):
    """Connect the CPU images of `device="cpu"` once, beside the device
    images (`context["xl_bindings"]`, `context["cd_solver"]`).

    Each host binary publishes its native operations table when it
    initializes; the compiled owner resolves an estimator's route against
    the device table or these, once per estimator
    (bindings/x_linear_route_unpack.mojo). `xl_cpu_scope` is the context a
    `device="cpu"` estimator's score programs run in: this context with the
    metrics tables of the CPU images. In a CPU-only process the selected
    images already are the CPU images and nothing is loaded. A host binary
    that is not built is kept as its refusal text, said when a
    `device="cpu"` estimator first needs it."""
    # glue: load the CPU images once and retain their tables; nothing here runs per call
    from types import BuiltinFunctionType
    from . import _backend, host_surface
    if "xl_cpu_scope" in context:
        return
    errors = (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError, KeyError)
    # The manifest's route table, read once here (as _native_trees_bridge does):
    # `_backend._HOST_MODULES` is read only by the CPU-only selector
    # (tools/test_lane_select_narrowing.py).
    routed = host_surface.routed_modules()

    def host(name):
        # glue: the CPU identical image of one device binding (read back as CPU and identical when loaded)
        return _backend.load_host_module(routed[name])

    linear_name, solver_name, x_metrics_name, metrics_name = _CPU_DEVICE_BINDINGS
    try:
        linear = _linear_entries(host(linear_name))
    except errors as error:
        linear = _refusal(error)
    cd = "ImportError: the coordinate-descent solver is not connected here"
    if with_solver:
        try:
            host(solver_name)
            # load_host_module read the image back as CPU and identical: its
            # solver operations table key is (tier 1, vendor 0).
            cd = (1, _NATIVE_VENDORS["cpu"])
        except errors as error:
            cd = _refusal(error)
    try:
        x_metrics = _x_metrics_table(host(x_metrics_name))
    except errors as error:
        x_metrics = _refusal(error)
    try:
        module = host(metrics_name)
        r2 = {}
        for name in ("r2_score", "r2_score_weighted", "metrics_numeric_mode", "umap_numeric_mode"):  # glue: fixed native exports
            entry = getattr(module, name, None)
            if isinstance(entry, BuiltinFunctionType):
                r2[name] = entry
        if "r2_score" not in r2:
            raise TypeError("the CPU metrics image has no compiled r2_score")
    except errors as error:
        r2 = _refusal(error)
    from ._native_labels_bridge import cpu_metrics_entries as _labels_cpu_metrics_entries
    scope = dict(context)
    scope.pop("classical_metric_vendor", None)
    # device="cpu" is identical only: both mode words name the CPU identical tables.
    scope.update(vendor="cpu", classical_metrics={"identical": x_metrics, "fast": x_metrics},
                 labels_base_binding=False,  # the label steps of this scope run the host kernels (label door: the context decides)
                 labels_bindings={"identical": {"_mojolearn_metrics": r2}, "fast": {"_mojolearn_metrics": r2}},
                 # the accuracy COUNT tail of a classifier's score runs on the CPU
                 # x_metrics image (bindings/classical_labels_door.mojo `_accuracy_codes`)
                 labels_metrics_entries=_labels_cpu_metrics_entries())
    context.update(xl_cpu_bindings={"identical": linear}, cd_solver_cpu=cd, xl_cpu_scope=scope)


#: The native tables' vendor words (core/solver_entry_table.mojo
#: solver_vendor_word applies the same mapping at publication).
_NATIVE_VENDORS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}


def _connect_cd(solver):
    # glue: once at import, the connected solver image's (tier, vendor) key
    # of the solver operations table (core/solver_entry_table.mojo); the
    # owner calls the image's published cd_fit / cd_predict entry with it,
    # never the binding's Python exports (lane calltable-solver-estimators).
    try:
        mode_probe = getattr(solver, "solver_numeric_mode", None) or getattr(solver, "solver_host_numeric_mode")
        mode = int(mode_probe())  # import-time: the image's compiled tier
        vendor = _NATIVE_VENDORS.get(str(solver.solver_vendor()), -1)  # import-time: the image's vendor
        if vendor < 0 or mode not in (0, 1):
            raise ImportError("the coordinate-descent solver image publishes no native table")
        return (mode, vendor)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        return _refusal(error)
