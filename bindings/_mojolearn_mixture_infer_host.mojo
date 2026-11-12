# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""INFERENCE-ONLY CPU binding for the `_mojolearn_mixture` family: a saved
GaussianMixture's score_samples, predict_proba, predict, score, bic, aic and
sample
on a CPU-only install (the neighbors and density inference lane,
2026-09-15).

This is the mixture binary a wheel ships. It registers the typed transport
`mixture_native_api_v1` the mixture door scores through
(mixture/host/gmm_native_infer_host.mojo, the reference binding's scoring
body) and `gmm_sample` of `bindings/mixture_host_scoring.mojo`, and no
fit: `gmmh_fit`, the EM loop, the k-means and random starts are not
imported, so they are not compiled into this file. `mojolearn.host_model`
loads a saved GaussianMixture into a host class that binds this file
(`python/mojolearn/_classical_host.py`); there is no `gmm_fit` to reach,
beside the public CPU fit guard.

The sabotage arm (`mixture_infer_host_sabotage`) is the reference binding's:
`gemm/host/gemm_oracle.mojo::GEMM_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`), every GEMM leaf walked descending, which
the scoring E step reaches through the whitened distance.
"""
from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from bindings.mixture_host_scoring_pack import gmm_sample_binding
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from mixture.host.gmm_native_infer_host import GMM_NATIVE_INFER_HOST_API
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE


def mixture_infer_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_infer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_infer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "mixture inference host: this binding compiles the CPU column only;"
        " pass -D MOJOLEARN_COLUMN_CPU (bindings/build_mixture_infer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def mixture_infer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every GEMM leaf descending on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(GEMM_ORACLE_HOST_SABOTAGE)


def mixture_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def mixture_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_native_api_v1_binding() raises -> PythonObject:
    """The typed GaussianMixture transport (core/mixture_native_api.mojo)
    the mixture door (bindings/mixture_native_door.mojo) scores a saved model
    through; its fit refuses (this binding compiles none)."""
    var api = GMM_NATIVE_INFER_HOST_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_mixture_infer_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_mixture_infer_host() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_mixture_infer_host")
        module.def_function[mixture_infer_host_numeric_mode_binding]("mixture_infer_host_numeric_mode")
        module.def_function[mixture_infer_host_vendor_binding]("mixture_infer_host_vendor")
        module.def_function[mixture_infer_host_column_binding]("mixture_infer_host_column")
        module.def_function[mixture_infer_host_sabotage_binding]("mixture_infer_host_sabotage")
        module.def_function[mixture_vendor_binding]("mixture_vendor")
        module.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        # TOMBSTONE: the GaussianMixture Python scoring entries gmm_score_samples, gmm_predict_proba, gmm_predict and gmm_score_bic_aic (dead: the mixture door scores through mixture_native_api_v1, registered below) deleted 2026-11-11 by 04229c83d5; code recoverable at 04229c83d5^.
        # Restore: git apply experiments/removed/dead_entries_gmm_python.patch
        module.def_function[gmm_sample_binding]("gmm_sample")
        module.def_function[mixture_native_api_v1_binding]("mixture_native_api_v1")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture_infer_host: ", e))
