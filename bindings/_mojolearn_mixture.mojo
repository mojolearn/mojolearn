# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Gaussian mixture lane (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`mixture/estimator.mojo` is reached and nothing is re-decided: the
covariance_type and init_params STRINGS cross as strings and are decoded
by `covariance_type_from_name` / `init_params_from_name` on the Mojo side,
so every by-name refusal there ('tied', 'diag', 'spherical', 'k-means++',
'random_from_data') fires from Python with its own sentence. Every
parameter refusal (`gmm_validate_params`, DEVIATION 1738), every data
refusal (`gmm_validate_data`) and the collapsed-component raise
(DEVIATION 1723) live one layer down.

THE ABI IS THE GP'S: two length-checked lists per entry point, orders
written out here and mirrored in `python/mojolearn/mixture.py`.

THE MODEL CROSSES AS ITS ARRAYS. `GaussianMixtureModel` is a
`fieldwise_init` struct of host lists and scalars; fit writes every field
into caller-sized buffers and the sample entry rebuilds the struct from
the same buffers (fit and scoring cross through the typed transport,
`mixture_native_api_v1`). `n_iter`, `converged` and `lower_bound` cross too, because
the estimator's header says a model that does not carry them cannot be
compared with another model.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from core.experiment_guards_cluster import CLUSTER_GUARDS_OK
from bindings.device_plan_binding import device_plan_enter_binding, device_plan_exit_binding, device_plan_log_binding
from std.os import abort
from core.hostptr import f32_ptr, i32_ptr
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from providers.python.neural_descriptor_pyapi import neural_descriptor_capsule
from mixture.gmm_native_device import GMM_NATIVE_DEVICE_API

from checks.numerics import GLOBAL_NUMERIC_MODE
# The typed bodies (lane thin-trees-metrics, 2026-10-21).
from mixture.gmm_entries import _gmm_sample_run
from checks.vendor import COMPILED_VENDOR
from mixture.estimator import GaussianMixtureModel
from mixture.mixture_binding_run import rebuild_model_run


def mixture_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL (the GP's shape, for its reason)."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def mixture_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


# TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_parallel_probes.patch


def _rebuild_model(
    addrs: PythonObject, params: PythonObject, what: String
) raises -> GaussianMixtureModel:
    """The model from the scoring entries' shared prefix.

    `addrs[0..4]`: weights (k), means (k*d), covariances (k*d*d),
    precisions_chol (k*d*d), log_det_chol (k), all read.
    `params[0..5]`: k, d, n_iter, converged (0/1), lower_bound, n (rows of
    the x that follows).
    """
    if len(addrs) != 7:
        raise Error(
            what
            + ": addrs must contain 7 addresses (weights, means,"
            " covariances, precisions_chol, log_det_chol, x, out), got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            what
            + ": params must contain 6 values (k, d, n_iter, converged,"
            " lower_bound, n), got "
            + String(len(params))
        )
    var k = Int(py=params[0])
    var d = Int(py=params[1])
    var n_iter = Int(py=params[2])
    var converged = Int(py=params[3]) != 0
    var lower_bound = Float32(Float64(py=params[4]))
    var weights_addr = Int(py=addrs[0])
    var means_addr = Int(py=addrs[1])
    var precisions_addr = Int(py=addrs[3])
    var log_det_addr = Int(py=addrs[4])
    return rebuild_model_run(
        k, d, n_iter, converged, lower_bound,
        weights_addr, means_addr, precisions_addr, log_det_addr,
    )


def gmm_sample_binding(
    addrs: PythonObject, params: PythonObject, sample_params: PythonObject
) raises -> PythonObject:
    """`sample(n_samples)`: the model prefix is `_rebuild_model`'s with
    `params[5]` (n) = 0; `addrs[5]` is X out (n_samples * d float32,
    WRITTEN), `addrs[6]` y out (n_samples int32, WRITTEN).
    `sample_params` is, in this order: 0 n_samples, 1 random_state's low 32
    bits, 2 its high 32 bits. Returns n_samples."""
    var model = _rebuild_model(addrs, params, String("gmm_sample"))
    if len(sample_params) != 3:
        raise Error(
            "gmm_sample: sample_params must contain 3 values (n_samples,"
            " random_state low, random_state high), got "
            + String(len(sample_params))
        )
    var n = Int(py=sample_params[0])
    var seed = (UInt64(Int(py=sample_params[2])) << 32) | UInt64(
        Int(py=sample_params[1])
    )
    var xp = f32_ptr(Int(py=addrs[5]))
    var yp = i32_ptr(Int(py=addrs[6]))
    with GILReleased(Python()):
        _gmm_sample_run(model, n, seed, xp, yp)
    return PythonObject(n)


def mixture_native_api_v1_binding() raises -> PythonObject:
    """The typed GaussianMixture transport (core/mixture_native_api.mojo)
    the mixture door (bindings/mixture_native_door.mojo) calls for fit,
    fit_predict and every score. `gmm_sample` stays a Python entry: the
    sample door (bindings/svm_mixture_native_door.mojo `_gmm_sample`) calls it
    and the typed transport has no sample operation."""
    var api = GMM_NATIVE_DEVICE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_mixture.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_mixture() -> PythonObject:
    comptime assert CLUSTER_GUARDS_OK, "invalid integrated experiment configuration (cluster guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_mixture")
        # TOMBSTONE: the *_parallel_available multi-GPU capability probes (dead native entries: constant 1, no driver or package code looks them up) deleted 2026-10-26 by b3cad1ef6; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_parallel_probes.patch
        m.def_function[device_plan_enter_binding]("device_plan_enter")
        m.def_function[device_plan_exit_binding]("device_plan_exit")
        m.def_function[device_plan_log_binding]("device_plan_log")
        m.def_function[mixture_vendor_binding]("mixture_vendor")
        m.def_function[mixture_numeric_mode_binding]("mixture_numeric_mode")
        # TOMBSTONE: the GaussianMixture Python entries gmm_fit, gmm_fit_predict, gmm_score_samples, gmm_predict_proba, gmm_predict and gmm_score_bic_aic (dead: the mixture door calls mixture_native_api_v1 and no Python, tool or verify code calls them) deleted 2026-11-11 by 04229c83d5; code recoverable at 04229c83d5^.
        # Restore: git apply experiments/removed/dead_entries_gmm_python.patch
        m.def_function[gmm_sample_binding]("gmm_sample")
        m.def_function[mixture_native_api_v1_binding]("mixture_native_api_v1")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_mixture: ", e))
