# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled saved-model owner for the public
estimator classes.

`save` and `load` of a connected class are one call each into the compiled
classical owner (`bindings/classical_native.mojo`): `save` runs the class's
fixed save schema (`bindings/archive_schema_native.mojo`) and writes the
deterministic NPZ the Python writer wrote; `load` reads the file and builds
the model natively (`bindings/host_archive_native.mojo`). Python here only
connects fixed metadata once, at import: the shared context, each class's
format and save schema, and the lookup tables its builder reads.
"""

_ENTRY = None
_CONTEXT = None

# Optional NumPy ABI metadata a load must never publish: loads return
# mojolearn Arrays on every install, as `_serialize.read_npz` did.
_NUMPY_KEYS = ("ndarray", "scalar_type", "integer", "numeric_dtypes",
               "class_dtypes", "float_types", "float32", "float64", "int64")


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; save and load then explain what is missing.
    message = f"mojolearn: saving and loading models needs the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def connect():
    """`(entry, context)`: the compiled saved-model owner and the context
    every connected class shares. Connected once; later calls return the
    same pair."""
    global _ENTRY, _CONTEXT
    if _CONTEXT is not None:
        return _ENTRY, _CONTEXT
    from types import BuiltinFunctionType, GetSetDescriptorType
    from . import _backend
    from ._native_classical_bridge import _state_context
    try:
        from .tokenizer import _binding
        entry = _binding()._module.classical_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("model archives require the compiled classical_native_call entry")
        save = _state_context()
        archive = _state_context()
    except ImportError as error:
        _ENTRY, _CONTEXT = _unavailable(error), {"archive_unavailable": str(error)}
        return _ENTRY, _CONTEXT
    _backend.default_mode()  # glue: settle the process default the saved mode falls back to
    for key in _NUMPY_KEYS:  # glue: remove optional NumPy ABI metadata once at connection
        archive.pop(key, None)
    archive.update(host_archive_dict_descriptor_type=GetSetDescriptorType,
                   host_archive_formats={}, host_archive_fitted_estimators=[],
                   host_archive_classical_formats=(), host_archive_forest_formats=(),
                   host_archive_lifetime_layouts={}, archive_class_formats={})
    save.update(archive_save_schemas={}, host_archive_context=archive)
    _ENTRY, _CONTEXT = entry, save
    return _ENTRY, _CONTEXT


def register(cls, formats, save, **tables):
    """Connect one public class at its definition: the saved format (a str,
    or a tuple of the formats its loader reads), its save schema (see
    bindings/archive_schema_native.mojo), and the fixed lookup tables its
    native builder reads (host_archive_* names)."""
    from ._mode import SklearnTagsMixin
    _, context = connect()
    if "archive_unavailable" in context:
        return
    archive = context["host_archive_context"]
    accepted = formats if isinstance(formats, tuple) else (formats,)
    for fmt in accepted:  # glue: the class's saved format tags, fixed at class definition
        archive["host_archive_formats"].setdefault(fmt, {})[cls.__name__] = cls
    archive["archive_class_formats"][cls] = (accepted, formats)
    if issubclass(cls, SklearnTagsMixin):
        archive["host_archive_fitted_estimators"].append(cls.__name__)
    archive.update(tables)
    if save is not None:
        context["archive_save_schemas"][cls] = save


def fitted(field, message="this estimator is not fitted yet", kind="RuntimeError"):
    """The save step refusing an unfitted model: `kind(message)` unless the
    instance holds `field`."""
    return ("need", field, kind, message)


def head(fmt, prefix="mojolearn: "):
    """The three members every schema starts with: the format tag, the
    class name and the numeric mode a predict would use now."""
    return (("format", "format", fmt), ("estimator", "estimator"), ("mode", "numeric_mode", prefix))


def lifetime(cls, release, holder=None):
    """Give a device class the native finalizer in place of a Python
    `__del__`, at its definition: dropping an instance releases its resident
    handle through `release`, the compiled entry of the binding that minted
    it (the resident tuple is `(key, handle, binding)`, under `_resident`, or
    under `holder`'s "index"). Native archive loading then builds the class.
    """
    from types import BuiltinFunctionType, MethodType
    _, context = connect()
    if "archive_unavailable" in context:
        return
    archive = context["host_archive_context"]
    descriptor = archive.get("host_archive_finalizer_descriptor")
    if descriptor is None:
        from .tokenizer import _binding
        module = _binding()._module
        finalize = getattr(module, "host_archive_finalize_call", None)
        if not isinstance(finalize, BuiltinFunctionType):
            raise TypeError("device handle release requires the compiled host_archive_finalize_call entry")
        descriptor = module.classical_attribute_descriptor(MethodType(finalize, archive))
        archive["host_archive_finalizer_descriptor"] = descriptor
    fields_descriptor = next(base.__dict__["__dict__"] for base in cls.__mro__  # glue: fixed native instance layout
                             if "__dict__" in base.__dict__)
    archive["host_archive_lifetime_layouts"][cls] = (None, fields_descriptor, "entry", holder, release)
    cls.__del__ = descriptor
