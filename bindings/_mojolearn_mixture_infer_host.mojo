# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""INFERENCE-ONLY CPU binding for the `_mojolearn_mixture` family: a saved
GaussianMixture's score_samples, predict_proba, predict, score, bic, aic and
sample
on a CPU-only install (the neighbors and density inference lane,
2026-09-15).

This is the mixture binary a wheel ships. It registers the four scoring
entries of `bindings/mixture_host_scoring.mojo`, the same functions the
reference binding `bindings/_mojolearn_mixture_host.mojo` registers, and no
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
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from bindings.mixture_host_scoring import (
    gmm_sample_binding,
    gmm_predict_binding,
    gmm_predict_proba_binding,
    gmm_score_bic_aic_binding,
    gmm_score_samples_binding,
)
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


@export
def PyInit__mojolearn_mixture_infer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_mixture_infer_host")
        module.def_function[mixture_infer_host_numeric_mode_binding]("mixture_infer_host_numeric_mode")
        module.def_function[mixture_infer_host_vendor_binding]("mixture_infer_host_vendor")
        module.def_function[mixture_infer_host_column_binding]("mixture_infer_host_column")
        module.def_function[mixture_infer_host_sabotage_binding]("mixture_infer_host_sabotage")
        module.def_function[mixture_vendor_binding]("mixture_vendor")
        module.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        module.def_function[gmm_score_samples_binding]("gmm_score_samples")
        module.def_function[gmm_predict_proba_binding]("gmm_predict_proba")
        module.def_function[gmm_predict_binding]("gmm_predict")
        module.def_function[gmm_score_bic_aic_binding]("gmm_score_bic_aic")
        module.def_function[gmm_sample_binding]("gmm_sample")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture_infer_host: ", e))
