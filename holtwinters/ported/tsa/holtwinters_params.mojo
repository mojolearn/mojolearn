# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `include/cuml/tsa/holtwinters_params.h` (v26.08.00), transliterated.

`SeasonalType`, `OptimCriterion`, `OptimParams` and `Norm`, as the ints and
the struct they are. The defaults `runner.cuh::HoltWintersOptim` fills in
live in `holtwinters/ported/holtwinters/runner.mojo::default_optim_params`.
"""


# enum SeasonalType { ADDITIVE, MULTIPLICATIVE };
comptime SEASONAL_ADDITIVE = 0
comptime SEASONAL_MULTIPLICATIVE = 1

# enum OptimCriterion
comptime OPTIM_BFGS_ITER_LIMIT = 0
comptime OPTIM_MIN_PARAM_DIFF = 1
comptime OPTIM_MIN_ERROR_DIFF = 2
comptime OPTIM_MIN_GRAD_NORM = 3

# enum Norm { L0, L1, L2, LINF };  (unused by the seasonal fit path)
comptime NORM_L0 = 0
comptime NORM_L1 = 1
comptime NORM_L2 = 2
comptime NORM_LINF = 3


@fieldwise_init
struct OptimParams(Copyable, Movable, ImplicitlyCopyable):
    """`OptimParams<Dtype>` with Dtype = float (the only instantiation here;
    see UNPORTED.tsv for double)."""

    var eps: Float32
    var min_param_diff: Float32
    var min_error_diff: Float32
    var min_grad_norm: Float32
    var bfgs_iter_limit: Int
    var linesearch_iter_limit: Int
    var linesearch_tau: Float32
    var linesearch_c: Float32
    var linesearch_step_size: Float32


def seasonal_from_name(name: String) raises -> Int:
    """`holtwinters.pyx:193-198`: 'additive' | 'add' -> ADDITIVE,
    'multiplicative' | 'mul' -> MULTIPLICATIVE, anything else raises by
    name."""
    if name == "additive" or name == "add":
        return SEASONAL_ADDITIVE
    if name == "multiplicative" or name == "mul":
        return SEASONAL_MULTIPLICATIVE
    raise Error(
        "seasonal='" + name + "': Seasonal must be either 'additive'/'add' or"
        " 'multiplicative'/'mul' (holtwinters.pyx:197)"
    )


def seasonal_name(seasonal: Int) -> String:
    if seasonal == SEASONAL_ADDITIVE:
        return String("additive")
    return String("multiplicative")


def criterion_name(c: Int) -> String:
    if c == OPTIM_BFGS_ITER_LIMIT:
        return String("OPTIM_BFGS_ITER_LIMIT")
    if c == OPTIM_MIN_PARAM_DIFF:
        return String("OPTIM_MIN_PARAM_DIFF")
    if c == OPTIM_MIN_ERROR_DIFF:
        return String("OPTIM_MIN_ERROR_DIFF")
    if c == OPTIM_MIN_GRAD_NORM:
        return String("OPTIM_MIN_GRAD_NORM")
    return String("?" + String(c))
