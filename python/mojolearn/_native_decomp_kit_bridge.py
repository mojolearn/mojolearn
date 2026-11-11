# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native decomposition owners that run the
x_decomp cells through bindings/decomp_kit_native.mojo (the dictionary
learning family, PLS, MinCovDet, ALS, the dense linear algebra functions):
their compiled door and the numeric-mode x_decomp entries, retained once."""

# The x_decomp entries the native kit (bindings/decomp_kit_native.mojo) still
# reads from this dict: the image's tier and vendor (its table key), the
# compiled route words, the optional resident entries it probes for, and the
# pool entries its resident records name. Every operation itself runs through
# the image's published table (core/decomp_entry_table.mojo); the solver
# entries the kit used to call here (dict_learning, mcd, pls_fit, als_fit,
# randomized_svd, ...) are table operations since lane pyout-decomp-linalg.
_PROBED = ("x_decomp_numeric_mode", "x_decomp_vendor", "x_decomp_s_flags", "x_decomp_qfix_flags",
           "x_decomp_idn_flags", "x_decomp_dev_ew", "x_decomp_dev_alloc", "x_decomp_dev_free",
           "x_decomp_dev_upload", "x_decomp_dev_svd", "x_decomp_dev_code_rows", "x_decomp_dev_lu",
           "x_decomp_dev_eigh_to")


def _entries(binding):
    # glue: retain the checked compiled x_decomp symbols the kit probes
    from types import BuiltinFunctionType
    entries = {}
    for name in _PROBED:  # glue: fixed native export names
        try:
            entry = getattr(binding, name)
        except (AttributeError, ImportError):
            continue
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"the native decomposition owner requires compiled {name}")
        entries[name] = entry
    # glue: the module itself (the resident-operand records and lu_factor's
    # resident pair free through it)
    entries["x_decomp_module"] = binding
    _native_key(entries)
    return entries


#: The x_decomp table's vendor words (core/linear_entry_table.mojo key), the
#: mapping the x_decomp images apply when they publish (0 host, 1 metal,
#: 2 cuda, 3 hip).
_NATIVE_VENDORS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}


def _native_key(entries):
    # glue: once at import, the connected x_decomp image's (tier, vendor) key,
    # which the native decomposition kit (bindings/decomp_kit_native.mojo)
    # reads to call the image's published operations entry
    # (core/decomp_entry_table.mojo) instead of its Python entries. An image
    # without one leaves the key absent and the kit refuses.
    try:
        mode = int(entries["x_decomp_numeric_mode"]())  # import-time: the image's compiled tier
        vendor = _NATIVE_VENDORS.get(str(entries["x_decomp_vendor"]()), -1)  # import-time: the image's vendor
    except Exception:  # noqa: BLE001 - absent probes leave the key absent
        return
    if vendor >= 0 and mode in (0, 1):
        entries["x_decomp_native_key"] = (mode, vendor)


def connect(door, **classes):
    # glue: compiled entries and plain metadata are retained during import;
    # `door` names the tokenizer-host export, `classes` the classes the
    # owner tells apart natively.
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from .tokenizer import _binding

    context = _state_context()
    connected = {}
    modules = {}
    selected = _backend.default_mode()
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_x_decomp", mode)
            connected[mode] = _entries(module)
            # the module itself: the owner of a device buffer that outlives
            # the call (lu_factor's resident pair) frees through it
            modules[mode] = module
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    # Every output and working buffer is a mojolearn Array, as `_M.out()` made.
    context.update(output_context=_numpy_free_context(), decomp_bindings=connected, decomp_modules=modules,
                   **classes)
    # The shared native label and accuracy steps (EllipticEnvelope.score).
    from ._native_labels_bridge import label_context
    context.update(label_context())
    entry = getattr(_binding()._module, door)
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError(f"the native decomposition owner requires a compiled {door}")
    return entry, context


def connect_public(door, **classes):
    """Connect a compiled decomposition owner once, at import; returns its
    entry and context. An install without the compiled owner keeps the
    package importable, and the estimators explain what is missing."""
    try:
        return connect(door, **classes)
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: the native decomposition owner {door} is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"decomp_unavailable": message}
