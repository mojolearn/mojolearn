# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_mixture` family: GaussianMixture (CPU
training for the workstream D estimators, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch. The fit and the four scoring
entries are `mixture/host/gmm_host_oracle.mojo`, the device path of
`mixture/estimator.mojo` restated on the host over the gemm profile's
normative answer, the Cholesky host profile and the k-means host oracle. So
every model array, `n_iter_`, `converged_` and `lower_bound_` are meant to
be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_mixture.mojo`,
mirrored in `python/mojolearn/mixture.py`), so `GaussianMixture` runs
unchanged on a CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_mixture": "_mojolearn_mixture_host"`): the typed transport
`mixture_native_api_v1` (fit and the four scores), `gmm_sample` (7
addresses, 6 params, 3 sample params), `mixture_vendor` answering "cpu" and
`mixture_numeric_mode`. ABSENT, and so refused BY NAME through
`_HostBinding`: `gmm_parallel_available`, the multi-GPU driver's probe.

The sabotage arm (`mixture_host_sabotage`) is
`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every GEMM leaf walks descending.
"""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from mixture.host.gmm_native_host import GMM_NATIVE_HOST_API

from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE
from bindings.mixture_host_scoring import gmm_sample_binding


def mixture_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "mixture host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_mixture_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def mixture_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mixture_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_native_api_v1_binding() raises -> PythonObject:
    """The typed GaussianMixture transport (core/mixture_native_api.mojo)
    the mixture door (bindings/mixture_native.mojo) calls for the fit and
    every score. `gmm_sample` stays a Python entry: the sample door
    (bindings/svm_mixture_native_door.mojo `_gmm_sample`) calls it and the typed
    transport has no sample operation."""
    var api = GMM_NATIVE_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_mixture_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_mixture_host() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_mixture_host")
        module.def_function[mixture_host_numeric_mode_binding]("mixture_host_numeric_mode")
        module.def_function[mixture_host_vendor_binding]("mixture_host_vendor")
        module.def_function[mixture_host_column_binding]("mixture_host_column")
        module.def_function[mixture_host_sabotage_binding]("mixture_host_sabotage")
        module.def_function[mixture_vendor_binding]("mixture_vendor")
        module.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        # TOMBSTONE: the GaussianMixture Python entries gmm_fit, gmm_score_samples, gmm_predict_proba, gmm_predict and gmm_score_bic_aic (dead: the mixture door calls mixture_native_api_v1) deleted 2026-11-11 by 04229c83d5; code recoverable at 04229c83d5^.
        # Restore: git apply experiments/removed/dead_entries_gmm_python.patch
        module.def_function[gmm_sample_binding]("gmm_sample")
        module.def_function[mixture_native_api_v1_binding]("mixture_native_api_v1")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture_host: ", e))
