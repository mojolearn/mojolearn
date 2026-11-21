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
    host surface manifest (`host_surface.cpu_training_routes()`, copied below
    as `CPU_TRAINING_ROUTES` by `tools/device_class_table.py --write-routes`)
    has no CPU implementation;
  * any other class has a CPU implementation that its family's door does not
    reach through the route yet.
No class is listed by hand.
"""
# The manifest is verification tooling: it is not imported here. Importing it
# at every `import mojolearn` (this module is imported by _mode) put the
# verifier's whole manifest, and the reference table it names, in every
# lane's closure (tools/test_lane_select.py, 2026-11-21) and parsed it at
# every import. Its one answer this table needs is generated below;
# tools/test_device_table_routes.py fails when the copy is stale.

#: class -> True, or the sentence the constructor's refusal carries. The
#: compiled check reads this exact dictionary (the mode metadata context's
#: `device_cpu`).
CPU_CLASSES = {}

NO_CPU = "its family has no CPU implementation"
NOT_ROUTED = "its CPU implementation is not connected to device= yet"

#: The GPU bindings whose host binding trains on the CPU (the keys of
#: `host_surface.cpu_training_routes()`).
# BEGIN generated: tools/device_class_table.py --write-routes
CPU_TRAINING_ROUTES = frozenset((
    '_mojolearn',
    '_mojolearn_arima',
    '_mojolearn_embedding',
    '_mojolearn_estimators',
    '_mojolearn_gbdt',
    '_mojolearn_gp',
    '_mojolearn_hdbscan',
    '_mojolearn_ivf',
    '_mojolearn_kernel_methods',
    '_mojolearn_linalg',
    '_mojolearn_mamba',
    '_mojolearn_metrics',
    '_mojolearn_mixture',
    '_mojolearn_preprocessing',
    '_mojolearn_resample',
    '_mojolearn_rf',
    '_mojolearn_solver',
    '_mojolearn_svm',
    '_mojolearn_training',
    '_mojolearn_transformer',
    '_mojolearn_trees',
    '_mojolearn_tsa',
    '_mojolearn_x_ann',
    '_mojolearn_x_cluster',
    '_mojolearn_x_cnn',
    '_mojolearn_x_decomp',
    '_mojolearn_x_linear',
    '_mojolearn_x_metrics',
    '_mojolearn_x_neighbors',
    '_mojolearn_x_prep',
    '_mojolearn_x_sequence',
    '_mojolearn_x_trees',
))
# END generated


def answer(cls):
    """True when `cls` runs `device="cpu"` beside a GPU set, else why not."""
    if getattr(cls, "_mojolearn_device", False) is True:
        return True
    if getattr(cls, "_HOST_INFERENCE_ONLY", False):
        return True  # an explicit host class runs on the CPU by class
    binding = getattr(cls, "_BINDING", None)
    if type(binding) is not str or binding not in CPU_TRAINING_ROUTES:
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
