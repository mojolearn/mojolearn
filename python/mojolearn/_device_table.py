# SPDX-License-Identifier: Apache-2.0
"""The one table behind `device="cpu"`: which estimator classes run on the CPU
beside a GPU set, and for every other class the sentence why not.

`device=None | "gpu" | "cpu"` is a constructor parameter of every estimator
(`_mode._install_numeric_mode` gives it to each class whose constructor does
not name it). None and "gpu" need no table. An explicit "cpu" on a GPU
installation runs only for a class whose public operations take their
compiled modules from the estimator's device route; the compiled constructor
check (bindings/mode_metadata_native_door.mojo, operation `device_admit`)
reads this table and refuses every other class in words, so nothing ever runs
on a device its caller did not name.

The table is GENERATED, one row per class, when the class is defined:
  * a class that declares `_mojolearn_device = True` runs on both devices
    (its family's door resolves through the route: the estimator's own entry
    route, or the shared estimator door once every device question of the
    family goes through it);
  * a class whose GPU binding (`_BINDING`) has no CPU training binding in the
    host surface manifest (`host_surface.cpu_training_routes()`) has no CPU
    implementation;
  * any other class has a CPU implementation that its family's door does not
    reach through the route yet.
No class is listed by hand.
"""
from . import host_surface

#: class -> True, or the sentence the constructor's refusal carries. The
#: compiled check reads this exact dictionary (the mode metadata context's
#: `device_cpu`).
CPU_CLASSES = {}

NO_CPU = "its family has no CPU implementation"
NOT_ROUTED = "its CPU implementation is not connected to device= yet"

_ROUTES = None


def answer(cls):
    """True when `cls` runs `device="cpu"` beside a GPU set, else why not."""
    global _ROUTES
    if getattr(cls, "_mojolearn_device", False) is True:
        return True
    if getattr(cls, "_HOST_INFERENCE_ONLY", False):
        return True  # an explicit host class runs on the CPU by class
    if _ROUTES is None:
        _ROUTES = host_surface.cpu_training_routes()
    binding = getattr(cls, "_BINDING", None)
    if type(binding) is not str or binding not in _ROUTES:
        return NO_CPU
    return NOT_ROUTED


def record(cls):
    # glue: one row per estimator class, at its class statement
    CPU_CLASSES[cls] = answer(cls)


def connect_context(context):
    """Give a compiled constructor or setter door's context what its device
    check reads: this table and the backend's fields (CPU-only or not)."""
    # glue: import-time context of a shared constructor door
    from . import _backend
    context.update(device_cpu=CPU_CLASSES, backend_fields=vars(_backend))
