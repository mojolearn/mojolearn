# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN of `bindings/_mojolearn_arima_host.mojo` (wave 8 run split): the
estimation-method refusal, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh); constants and body moved unchanged (lane
rs-sequence-tokenizer, 2026-11-12)."""


comptime ARIMA_METHOD_MLE = 0
comptime ARIMA_METHOD_CSS = 1
comptime ARIMA_METHOD_CSS_ML = 2


def _refuse_method(method: Int) raises:
    """`arima/estimator.mojo::_refuse_method` (`:114-142`), its words."""
    if method == ARIMA_METHOD_MLE:
        return
    if method == ARIMA_METHOD_CSS:
        raise Error(
            "ARIMA: method='css' is not implemented; the conditional sum of"
            " squares log-likelihood (batched_arima.cu:271-391,"
            " conditional_sum_of_squares and sum_of_squares_kernel) and its"
            " `truncate` parameter have no implementation. Only MLE is offered;"
            " refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    if method == ARIMA_METHOD_CSS_ML:
        raise Error(
            "ARIMA: method='css-ml' is not implemented; it starts the maximum"
            " likelihood fit from the CSS optimum, and the CSS"
            " log-likelihood (batched_arima.cu:271-391) has no implementation. Only"
            " MLE is offered; refused by name (arima/NOT_IMPLEMENTED.tsv)"
        )
    raise Error(
        "ARIMA: method code " + String(method) + " is not one of MLE ("
        + String(ARIMA_METHOD_MLE) + "), CSS (" + String(ARIMA_METHOD_CSS)
        + ") or CSS-ML (" + String(ARIMA_METHOD_CSS_ML) + ")"
    )
