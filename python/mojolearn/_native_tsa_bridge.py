# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the stationarity functions
(`kpss_test`, `select_d`; bindings/tsa_public_native_door.mojo). Python here
only connects the compiled `_mojolearn_tsa` entries of both tiers (the lane
is identical only: a tier without a binary keeps its refusal, raised by the
call that asks for it) and the shared context, once, at import."""

_ENTRIES = ("kpss_test", "select_d")

#: `(metadata prefix, table family)` of the binaries the `_mojolearn_tsa` name
#: resolves to: the device binding, the CPU binding with the fit, and the
#: forecast inference binding (core/tsa_entry_table.mojo TSA_TABLE_*).
_HW_IMAGES = (("tsa", 0), ("tsa_host", 0), ("forecast_host", 1))


#: The same for the binaries the `_mojolearn_arima` name resolves to: the
#: forecast inference binding, the CPU reference binding with the fit, and the
#: device binding (core/arima_entry_table.mojo ARIMA_TABLE_*). The host
#: prefixes are probed first: both host images also answer `arima_vendor`.
_ARIMA_IMAGES = (("forecast_host", 1), ("arima_host", 0), ("arima", 0))


def _hw_cells(module, images=_HW_IMAGES, what="Holt-Winters"):
    """One connected binding: the module and the operations-table key it
    reports (`family`, `mode` 1 identical / 0 fast, `vendor` word)."""
    # glue: read the compiled image's own metadata once, at import
    from types import BuiltinFunctionType
    from . import _backend
    for prefix, family in images:  # glue: fixed compiled metadata names
        vendor = getattr(module, prefix + "_vendor", None)
        mode = getattr(module, prefix + "_numeric_mode", None)
        if isinstance(vendor, BuiltinFunctionType) and isinstance(mode, BuiltinFunctionType):
            # the image itself (a CPU-only install hands a facade for the name)
            image = getattr(vendor, "__self__", None)
            if type(image) is not type(_backend):
                image = module
            return {"module": image, "family": family, "mode": int(mode()), "vendor": str(vendor()),
                    "module_name": getattr(module, "__name__", prefix)}
    raise TypeError("the " + what + " binding reports no vendor and numeric mode")


def connect():
    """`(entry, context)` for `kpss_test` and `select_d`."""
    from types import BuiltinFunctionType
    from . import _backend
    from .tokenizer import _binding
    from ._native_classical_bridge import _state_context
    entry = _binding()._module.tsa_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("the stationarity functions require the compiled tsa_native_call entry")
    context = _state_context()
    from ._native_owner_lifetime import connect_owner_context
    ownership = dict(context)
    connect_owner_context(ownership)
    context["tsa_owner_context"] = ownership
    # glue: reuse the native array materializer's exact storage descriptors
    context.update(owner_array_slots=ownership["owner_array_slots"],
                   arena_native_types=ownership["owner_native_types"], arena_memoryview=memoryview,
                   owner_mmap_class=ownership["owner_mmap_class"])
    import struct
    context["arena_half_copy_canonical"] = struct.pack("<e", struct.unpack("<e", b"\x01\x7c")[0]) == b"\x00\x7e"

    selected = _backend.default_mode()
    bindings = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_tsa", mode)
            symbols = {}
            for name in _ENTRIES:  # glue: fixed compiled entry names
                fn = getattr(module, name)
                if not isinstance(fn, BuiltinFunctionType):
                    raise TypeError(f"the stationarity functions require a compiled {name}")
                symbols[name] = fn
            bindings[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context["tsa_bindings"] = bindings
    # ExponentialSmoothing's device route (bindings/tsa_hw_native_door.mojo):
    # each connected binding is its module (the image pin) and the key of the
    # Holt-Winters operations table it published at initialization
    # (core/tsa_entry_table.mojo). The compiled owner resolves an estimator's
    # entry from that table once; no entry is looked up by name here.
    hw = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            hw[mode] = _hw_cells(_backend.binding("_mojolearn_tsa", mode))
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            hw[mode] = type(error).__name__ + ": " + str(error)
    context["hw_bindings"] = hw
    chosen = hw[selected]
    # "none": the selected binding did not load; its own refusal is said by
    # the first operation, not by the constructor's device check.
    context["hw_install_vendor"] = chosen["vendor"] if type(chosen) is dict else "none"
    # device="cpu" on a GPU installation: the CPU binding, loaded once here
    # beside the device binding. In a CPU-only process the selected binding
    # already is the CPU one and nothing more is loaded. A CPU binary that is
    # not built is kept as its refusal text, said when a device="cpu"
    # estimator first runs.
    if context["hw_install_vendor"] not in ("cpu", "none"):
        try:
            cpu = _hw_cells(_backend.load_host_module("_mojolearn_tsa_host"))
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            cpu = type(error).__name__ + ": " + str(error)
        context["hw_cpu_bindings"] = {"identical": cpu}
    # the host inference class reads its loaded forecast binding from the
    # module table at call time (its own load loads it)
    context["hw_host_module"] = "mojolearn._host._mojolearn_forecast_host"
    # ARIMA's and AutoARIMA's device route (bindings/tsa_arima_route_unpack.mojo):
    # as for Holt-Winters above, each connected binding is its module and
    # the key of the ARIMA operations table it published at initialization
    # (core/arima_entry_table.mojo). No entry is looked up by name.
    arima = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            arima[mode] = _hw_cells(_backend.binding("_mojolearn_arima", mode), _ARIMA_IMAGES, "ARIMA")
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            arima[mode] = type(error).__name__ + ": " + str(error)
    context["arima_bindings"] = arima
    chosen = arima[selected]
    context["arima_install_vendor"] = chosen["vendor"] if type(chosen) is dict else "none"
    # device="cpu" on a GPU installation: the CPU reference binding (with the
    # fit) where it is built, else the forecast inference binding (prediction
    # and forecast; its table holds no fit and the fit is refused in words).
    # A CPU binary that is not built is kept as its refusal text, said when a
    # device="cpu" estimator first runs.
    if context["arima_install_vendor"] not in ("cpu", "none"):
        cpu = None
        for name in ("_mojolearn_arima_host", "_mojolearn_forecast_host"):  # glue: the two CPU images, the one with the fit first
            try:
                cpu = _hw_cells(_backend.load_host_module(name), _ARIMA_IMAGES, "ARIMA")
                break
            except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
                if cpu is None:
                    cpu = type(error).__name__ + ": " + str(error)
        context["arima_cpu_bindings"] = {"identical": cpu}
    from ._buffer import _native
    for name in ("gather_rows_bytes", "scatter_rows_bytes", "cast_elements", "bincount_i64",  # glue: fixed compiled registration names
                 "select_fold_i64", "ic_running_min_f32", "ic_running_min_f64", "strided_copy_bytes"):  # glue: fixed compiled row moves, integer casts and IC/group helpers
        fn = _native(name)
        if not isinstance(fn, BuiltinFunctionType):
            raise TypeError(f"AutoARIMA requires the compiled {name} entry")
        context["base_" + name] = fn
    return entry, context
