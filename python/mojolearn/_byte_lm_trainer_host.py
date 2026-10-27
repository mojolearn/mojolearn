# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native CPU language-model trainer."""
from . import _backend
from types import BuiltinFunctionType

HOST_BASENAME = "_mojolearn_byte_lm_host"
_BINDING = None


def is_cpu_trainer_binding(value):
    """Whether this is the connected CPU language-model binding."""
    return value is _BINDING


def binding():
    # glue: backend import-time connection only; public operations use the
    # native module's common descriptor table, never this loader or a proxy.
    global _BINDING
    host = _backend.load_host_module(HOST_BASENAME)
    if _BINDING is not host:
        for name in ("byte_lm_trainer_api_v1", "byte_lm_native_session_handle_v1",
                     "byte_lm_session_create", "byte_lm_trainer_step", "byte_lm_run_configured"):
            entry = vars(host).get(name)
            if type(entry) is not BuiltinFunctionType or entry.__self__ is not host:
                raise ImportError(f"mojolearn: {host.__file__} lacks native {name}; "
                                  "rebuild bindings/build_byte_lm_host.sh")
        _BINDING = host
    return host
