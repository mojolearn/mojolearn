# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native CPU language-model trainer."""
from . import _backend
from types import BuiltinFunctionType

HOST_BASENAME = "_mojolearn_byte_lm_host"
_BINDING = None


def _door_host():
    # glue: the same binary through the door loader `_buffer._host_native` has always used
    return _backend.load_door_host(HOST_BASENAME)


# THE BASE HELPERS THIS BINARY ALSO EXPORTS (DEVIATION 2614; lane-loads-1,
# 2026-11-28): offered by the CPU trainer's connection as well as by the
# inference door, so a box with this binary and no shared host binding still
# converts the trainer's inputs natively. Nothing is loaded until a helper is
# missed.
from . import _buffer  # noqa: E402 - beside the loader it registers
_buffer.register_host_native(_door_host, _buffer._HOST_NATIVE_KEYS, first=True)


def binding():
    # glue: backend import-time connection only; public operations use the
    # native module's common descriptor table, never this loader or a proxy.
    global _BINDING
    host = _backend.load_host_module(HOST_BASENAME)
    if _BINDING is not host:
        # The resident session, its handle factory and the trainer provider
        # are native table entries now (training/byte_lm_entry_table_sessions.mojo,
        # lane byte-lm-sessions-native); they are not admitted by name here.
        for name in ("byte_lm_trainer_step", "byte_lm_run_configured"):
            entry = vars(host).get(name)
            if type(entry) is not BuiltinFunctionType or entry.__self__ is not host:
                raise ImportError(f"mojolearn: {host.__file__} lacks native {name}; "
                                  "rebuild bindings/build_byte_lm_host.sh")
        _BINDING = host
    return host
