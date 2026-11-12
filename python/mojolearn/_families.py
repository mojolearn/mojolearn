# SPDX-License-Identifier: Apache-2.0
"""Per-family import on first use (lane lazy-family-import, 2026-11-12).

`import mojolearn` loads the core only: the tokenizer host unit, which carries
the compiled descriptors, the `Array` helpers of the base binding, and the
process metadata (`numeric_mode`, `vendor`, numeric profiles). Every other
family module is imported when a public name of it is first touched (the
package `__getattr__`, `_LAZY_NAMES` in `__init__.py`) or when it is imported
by name (`from mojolearn.linear_model import Ridge`). The module's own
import-time connections run unchanged at that import and write their
`_native_loaded_modules.CONNECTIONS` records; its native units load then.

After the outermost import of a package module completes, `connect_loaded()`
runs the package's class connections (constructors, attribute slots,
parameter schemas, cross-validation schemas, storage layouts) over the classes
that are loaded and not yet connected, in the order the package used to run
them once at the end of `import mojolearn`. A class is connected once; a later
family load never re-snapshots an already connected class.

`load_all()` imports every family the package used to import eagerly and loads
every unit the selector used to load, then connects them in one pass: the
loaded set before this lane. `python -m mojolearn` (verify and the other
commands) and every worker process call it first.
"""
import importlib
import importlib.machinery
import sys
import threading
from types import ModuleType

_PKG = __name__.rpartition(".")[0]

#: Every module `__init__.py` imported eagerly before this lane, in that order.
#: `load_all()` imports exactly these (the names that were lazy before, such as
#: the CPU inference modules and the NumPy sequence models, stay lazy).
EAGER_BEFORE = (
    "model_selection", "cluster", "decomposition", "density", "ensemble", "extratrees",
    "preprocessing", "linear_model", "neighbors", "randomforest", "linalg", "lowbit", "models",
    "_metrics_impl", "metrics", "_hierarchy_impl", "_iforest_impl", "_solver_impl", "_spectral_impl",
    "manifold", "umap", "neural_network", "_svm_impl", "svm", "_arima_impl", "_tsa_impl",
    "lm_corpus", "_cholesky_impl", "kernel_methods", "mixture", "hdbscan", "resample",
    "_ivf_impl", "embedding", "_gp_impl", "_gpc_impl", "_training_impl", "training", "mamba",
    "_mamba_impl", "transformer", "_transformer_impl", "_samba_impl", "parallel_training",
    "model_pool_training", "offload_training", "parallel_classical", "parallel_ensemble",
    "parallel_forecasting", "parallel_gaussian_process", "parallel_graph", "parallel_ivf",
    "parallel_model_selection", "parallel_neighbors", "parallel_neighbors_reference",
    "parallel_preprocessing", "_expansion_linear", "_expansion_cluster", "_expansion_neighbors",
    "_expansion_decomp", "_expansion_prep", "_expansion_sequence", "_expansion_trees",
    "_expansion_cnn", "_expansion_ann", "_expansion_metrics",
    # imported by the package's class connections before this lane
    "tree", "_gbdt_adapters", "_forest_protocol", "_classical_host", "_umap_impl",
)

#: The expansion doors (host_surface.EXPANSION_LANES order). Each one's
#: `__all__` must be exactly the names `_LAZY_NAMES` routes to it; checked when
#: the door loads, so a drift or a name claimed by two doors refuses by name.
EXPANSION_DOORS = (
    "_expansion_linear", "_expansion_cluster", "_expansion_neighbors", "_expansion_decomp",
    "_expansion_prep", "_expansion_sequence", "_expansion_trees", "_expansion_cnn",
    "_expansion_ann", "_expansion_metrics",
)

#: Connections that join two families: when the key is loaded, the values are
#: imported too, before the class connections run. The tree door's default
#: member estimators (Ridge, LogisticRegression, LinearSVC) and its
#: linear-predict identity (PermutationExplainer) come from these modules.
DEPENDS = {
    "_expansion_trees": ("_expansion_linear", "linear_model", "svm"),
}

_STATE = threading.local()        # per-thread import depth and pass flag
_LOCK = threading.RLock()         # one connection pass at a time
_SUSPENDED = [False]              # load_all imports first, connects once
_CORE = set()                     # modules loaded by `import mojolearn` itself
_CHECKED_DOORS = set()


def loaded(name):
    """The package module `name` if it is imported, else None (never imports)."""
    module = sys.modules.get(_PKG + "." + name)
    return module if type(module) is ModuleType else None


def _package_modules():
    prefix = _PKG + "."
    return frozenset(name for name, module in tuple(sys.modules.items())  # glue: loaded module names
                     if name.startswith(prefix) and type(module) is ModuleType)


def _depth():
    return getattr(_STATE, "depth", 0)


class _CountingLoader:
    """Wraps a package source module's loader: when the outermost package
    import finishes, the class connections run over what it loaded."""

    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        depth = _depth()
        if depth == 0:
            _arm_import_layouts()
        _STATE.depth = depth + 1
        try:
            self._loader.exec_module(module)
        finally:
            _STATE.depth = depth
        if depth == 0 and not getattr(_STATE, "in_pass", False) and not _SUSPENDED[0]:
            connect_loaded()


class _FamilyFinder:
    """First-touch finder for the package's own Python modules. It finds
    nothing itself: it asks the finders behind it and wraps the loader."""

    _WRAPPED = (importlib.machinery.SourceFileLoader, importlib.machinery.SourcelessFileLoader)

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_PKG + "."):
            return None
        rest = fullname[len(_PKG) + 1:]
        if rest.startswith(("tests", "_sets.", "_host.")) or rest.rpartition(".")[2].startswith("_mojolearn"):
            return None
        for finder in tuple(sys.meta_path):  # glue: the interpreter's import finders, in order
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is None:
                continue
            if isinstance(spec.loader, self._WRAPPED):
                spec.loader = _CountingLoader(spec.loader)
            return spec
        return None


_FINDER = _FamilyFinder()


def install(package):
    """Called once at the end of `__init__.py`: record the core, install the
    first-touch finder and connect the core's classes."""
    _CORE.update(_package_modules())
    _CORE.add(__name__)
    if _FINDER not in sys.meta_path:  # glue: the import finders
        sys.meta_path.insert(0, _FINDER)
    connect_loaded()


def _arm_import_layouts():
    # The classical layouts taken while a family imports are rebound once
    # after its installers (_native_classical_bridge.refresh_installed_members).
    bridge = loaded("_native_classical_bridge")
    if bridge is not None:
        bridge.arm_import_layouts()


def _check_doors(package):
    # glue: once per expansion door, when it loads: its `__all__` is exactly
    # the names the package routes to it (no drift, no name in two doors).
    table = vars(package)["_LAZY_NAMES"]
    for door in EXPANSION_DOORS:  # glue: the fixed door inventory
        if door in _CHECKED_DOORS:
            continue
        module = loaded(door)
        if module is None:
            continue
        exported = tuple(vars(module).get("__all__", ()))
        routed = tuple(name for name, where in table.items() if where == "." + door)  # glue: the door's routed names
        if sorted(exported) != sorted(routed):
            raise ImportError(f"mojolearn: {module.__name__} exports {sorted(set(exported) ^ set(routed))!r}, "
                              "which the package routes elsewhere or not at all; a public name belongs "
                              "to one door (_LAZY_NAMES in mojolearn/__init__.py)")
        _CHECKED_DOORS.add(door)


def _hub_needed():
    # glue: estimator classes (anything with get_params) outside the core need
    # the model-selection hub: its native context carries the estimator ABI
    # registry every parameter setter and constructor connection reads.
    for name in _package_modules() - _CORE:  # glue: loaded non-core modules
        module = sys.modules[name]
        if name.startswith(_PKG + ".tests"):
            continue
        for value in tuple(vars(module).values()):  # glue: exact class identities
            if type(value) is type and any("get_params" in vars(base) for base in type.__dict__["__mro__"].__get__(value)):
                return True
    return False


def _import_dependencies():
    for name, needs in DEPENDS.items():  # glue: the fixed cross-family joins
        if loaded(name) is not None:
            for need in needs:  # glue: fixed dependency modules
                if loaded(need) is None:
                    importlib.import_module(_PKG + "." + need)


def _connect_pass(package):
    """The package's class connections over every loaded, unconnected class,
    in the order `__init__.py` ran them once before this lane."""
    from . import _mode
    _check_doors(package)
    _import_dependencies()
    hub = _hub_needed()
    if hub and loaded("model_selection") is None:
        importlib.import_module(_PKG + ".model_selection")
    selection = loaded("model_selection") if hub else None

    # CPU categorical GradientBoosting prediction is a native tree operation
    # too; connect its fixed host metadata once the tree owner is loaded.
    trees_bridge = loaded("_native_trees_bridge")
    if trees_bridge is not None:
        trees_bridge.connect_cpu_prediction()

    # Field-only constructors must precede every method/storage identity snapshot.
    from ._native_linear_constructors import (collect as _collect_linear_constructors,
                                              install as _install_linear_constructors,
                                              connect as _connect_linear_constructors)
    _collect_linear_constructors()
    _install_linear_constructors()
    # Full-reset setters precede every class-layout snapshot; the linear subset keeps its separate contract.
    from ._native_full_reset_params import (collect as _collect_full_reset_params,
                                            install as _install_full_reset_params,
                                            connect as _connect_full_reset_params)
    _collect_full_reset_params()
    _install_full_reset_params()
    # Field-only declared prep/decomp constructors precede every class-layout snapshot too.
    from ._native_declared_constructors import (collect as _collect_declared_constructors,
                                                install as _install_declared_constructors,
                                                connect as _connect_declared_constructors)
    _collect_declared_constructors()
    _install_declared_constructors()
    # Layouts connected at module import (neighbor queries, host inference, ...)
    # predate the installers above: rebind their replaced __init__/set_params.
    from ._native_classical_bridge import refresh_installed_members as _refresh_installed_members
    _refresh_installed_members()

    # CV's fixed native estimator ABIs after every public class exists.
    from ._native_mode_metadata import connect_classes as _connect_mode_metadata_classes
    _connect_mode_metadata_classes(_mode._MODE_METADATA_CONTEXT, package)
    # Native attribute slots precede the estimator and parameter layout snapshots.
    from ._native_public_attributes import connect_classes as _connect_public_attributes
    _connect_public_attributes(package)

    if selection is not None:
        from ._native_msel_dispatch import connect as _connect_native_cv
        _connect_native_cv(selection._NATIVE_CONTEXT, package)
        # The prep owners (RFE, IterativeImputer, TargetEncoder's cv splitter)
        # reach a wrapped estimator or splitter through the same compiled
        # model-selection entry and plan table (bindings/prep_native_estimator.mojo).
        prep = loaded("_expansion_prep")
        if prep is not None:
            prep._ENCODE_CONTEXT.update(native_estimator_entry=selection._NATIVE,
                                        native_estimator_context=selection._NATIVE_CONTEXT)
        trees = loaded("_expansion_trees")
        if trees is not None:
            trees._TREES_CONTEXT.update(native_estimator_entry=selection._NATIVE,
                                        native_estimator_context=selection._NATIVE_CONTEXT)
            # PermutationExplainer's resident linear route recognizes this
            # library's linear predict by identity (bindings/trees_agnostic_native.mojo).
            linear = loaded("_expansion_linear")
            trees._TREES_CONTEXT["trees_linear_predict"] = linear._LinearRegressorMixin.__dict__["predict"]
        # Parameter-read schemas capture the final built-in installers' method
        # and descriptor identities; the connection changes only this private plan table.
        from ._native_params_setter import connect as _connect_params_setter
        _connect_params_setter(selection._NATIVE_CONTEXT, _mode._PARAMS_METADATA_CONTEXT)

    from ._native_params_metadata import connect_classes as _connect_params_metadata_classes
    _connect_params_metadata_classes(_mode._PARAMS_METADATA_CONTEXT, package)

    if selection is not None:
        from ._native_declared_params import connect as _connect_declared_params, CONTEXT as _declared
        _connect_declared_params(package, selection._NATIVE_CONTEXT, _mode._PARAMS_METADATA_CONTEXT)
        # Complete constructor ownership only after the native setter layouts exist.
        from ._native_params_setter import _CONTEXT as _setters
        _connect_linear_constructors(_setters)
        # Connect full-reset ownership and nested routes after all getter/setter plans.
        _connect_full_reset_params(_mode._PARAMS_METADATA_CONTEXT, _setters, _declared, selection._NATIVE_CONTEXT)
        # Declared-family constructors release overwritten values under the declared setter's admission.
        _declared_constructor_setters = _declared
        _connect_declared_constructors(_declared_constructor_setters)

    # Snapshot family storage after every native constructor/attribute installer.
    from ._native_family_layouts import connect as _connect_family_layouts
    _connect_family_layouts(package)

    if selection is not None:
        # Default member estimators are constructed only after every constructor above is connected.
        from ._native_msel_prep import connect_estimator_defaults as _connect_estimator_defaults
        _connect_estimator_defaults()

    # The public parallel drivers' class/storage snapshots are taken last:
    # every class connection above precedes them, so a correct call is not
    # refused as a changed hook.
    family_bridge = loaded("_native_family_bridge")
    context = vars(family_bridge).get("_CONTEXT") if family_bridge is not None else None
    if type(context) is dict:
        from ._native_parallel_public import refresh as refresh_parallel_public
        refresh_parallel_public(context)


def connect_loaded():
    """Run the class connections until no connection imports another module."""
    if getattr(_STATE, "in_pass", False) or not _CORE:
        return
    package = sys.modules[_PKG]
    with _LOCK:
        _STATE.in_pass = True
        try:
            while True:  # glue: until a pass imports nothing new (a fixed point)
                before = _package_modules()
                _connect_pass(package)
                if _package_modules() == before:
                    break
        finally:
            _STATE.in_pass = False


def load_all():
    """Import every family `import mojolearn` used to import, load every unit
    the selector used to load, and connect them: the pre-lane loaded set."""
    from . import _backend
    _SUSPENDED[0] = True
    try:
        for name in EAGER_BEFORE:  # glue: the fixed pre-lane import order
            importlib.import_module(_PKG + "." + name)
    finally:
        _SUSPENDED[0] = False
    connect_loaded()
    _backend.load_everything()
    public = loaded("_native_backend_public")
    if public is not None:
        public.refresh()


def loaded_units():
    """{member module name: realpath of the native file it was loaded from}
    for every loaded native binding (canonical, tier set and host). A unit is
    one native file; several members can share one (`_merged.py`)."""
    import os
    out = {}
    prefix = _PKG + "."
    for name, module in tuple(sys.modules.items()):  # glue: loaded module inventory
        if not name.startswith(prefix) or not name.rpartition(".")[2].startswith("_mojolearn"):
            continue
        path = vars(module).get("__file__") if type(module) is ModuleType else None
        if path and path.endswith((".so", ".dylib", ".pyd")):
            out[name] = os.path.realpath(path)
    return out
