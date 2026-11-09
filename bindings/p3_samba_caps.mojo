# SPDX-License-Identifier: Apache-2.0
"""Capability words of one connected Samba training binding, read once.

The connected `entries` namespace (python/mojolearn/_native_samba_bridge.py)
is a `types.SimpleNamespace`, which cannot be weakly referenced, so a prepared
record keyed on it never stores. Every value probed here is fixed by the
compiled binding module (`entries.canonical`): which optional entries exist,
the compile-time switches `samba_resident_enabled()` /
`training_chunked_lm_head_enabled()`, and the binding's numeric mode and
vendor words. The record is kept in the native prepared registry, weakly
keyed by that module, so a step or a token reads one tuple instead of a
dozen `hasattr` probes and binding crossings.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.causal_lm_public import _native_function
from bindings.py_balanced import py_call


comptime _SAMBA_CAPS_SLOT = "p3_samba_caps_v1"

# Record layout (a Python tuple of exact ints/bools/strs).
comptime SAMBA_CAP_MODE = 0  # Int: training_numeric_mode() of the binding (1 identical, 0 fast), -1 absent
comptime SAMBA_CAP_VENDOR = 1  # training_vendor() of the binding, None when absent
comptime SAMBA_CAP_RESIDENT_ABI = 2  # Bool: the three resident train-step entries exist
comptime SAMBA_CAP_RESIDENT_ON = 3  # Bool: samba_resident_enabled() (compile-time switch)
comptime SAMBA_CAP_ACCUMULATED = 4  # Bool: samba_resident_accumulated_step exists
comptime SAMBA_CAP_CHUNKED = 5  # Bool: training_chunked_lm_head_enabled() (False when absent)
comptime SAMBA_CAP_AFN = 6  # Bool: the two fused training-tail entries exist
comptime SAMBA_CAP_HEAD_LOSS = 7  # Bool: samba_head_loss exists


def _has(bi: PythonObject, entries: PythonObject, name: String) raises -> Bool:
    return Bool(py=py_call(bi.hasattr, entries, name))


def _probe(entries: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var mode = PythonObject(-1)
    if _has(bi, entries, "training_numeric_mode"):
        mode = PythonObject(Int(py=_native_function(entries, "training_numeric_mode")()))
    var vendor = PythonObject(None)
    if _has(bi, entries, "training_vendor"):
        vendor = _native_function(entries, "training_vendor")()
    var abi = True
    for name in [String("samba_resident_enabled"), String("samba_resident_open"), String("samba_resident_step")]:  # driver-loop: over the fixed resident Samba ABI names, once per binding
        abi = abi and _has(bi, entries, name)
    var resident_on = abi and Bool(py=_native_function(entries, "samba_resident_enabled")())
    var accumulated = _has(bi, entries, "samba_resident_accumulated_step")
    var chunked = _has(bi, entries, "training_chunked_lm_head_enabled") and Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")())
    var afn = True
    for name in [String("samba_afn_tail_train"), String("samba_afn_embedding_backward_tied")]:  # driver-loop: over the fixed fused-tail names, once per binding
        afn = afn and _has(bi, entries, name)
    var head_loss = _has(bi, entries, "samba_head_loss")
    return Python.tuple(mode, vendor, PythonObject(abi), PythonObject(resident_on), PythonObject(accumulated),
                        PythonObject(chunked), PythonObject(afn), PythonObject(head_loss))


def samba_caps(entries: PythonObject) raises -> PythonObject:
    """The capability record of `entries`; probed once per compiled binding module."""
    var canonical = py_call(native_loaded_module("builtins").getattr, entries, "canonical", PythonObject(None))
    if canonical is None:
        # No binding connected (or a namespace without one): nothing to key
        # on; probe directly, as before.
        return _probe(entries)
    var mods = NativeModules()
    var record = prepared_get(mods, _SAMBA_CAPS_SLOT, canonical)
    if record is not None:
        return record
    record = _probe(entries)
    prepared_put(mods, _SAMBA_CAPS_SLOT, canonical, record)
    return record


def samba_cap(entries: PythonObject, index: Int) raises -> Bool:
    return Bool(py=samba_caps(entries)[index])
