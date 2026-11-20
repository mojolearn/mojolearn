# SPDX-License-Identifier: Apache-2.0
"""Shared scalar admission for typed label RUN and lazy legacy UNPACK."""


def labels_unique_shape(count: Int, kind: Int) raises:
    if count < 1:
        raise Error("unique_inverse: n must be positive, got " + String(count))
    if kind != 0 and kind != 1:
        raise Error("unique_inverse: kind must be 0 (float64) or 1 (int64)")


def labels_prepare_count(count: Int) raises:
    if count < 0:
        raise Error("prepare_targets: n must be non-negative, got " + String(count))


def labels_prepare_label_count(count: Int) raises:
    if count < 1:
        raise Error("prepare_targets: labels need n >= 1")


def labels_argmax_shape(name: String, rows: Int, cols: Int) raises:
    if rows < 0 or cols < 1:
        raise Error(name + ": n_rows must be non-negative and n_cols positive")


def labels_gather_shape(name: String, count: Int, classes: Int) raises:
    if count < 0 or classes < 1:
        raise Error(name + ": n must be non-negative and the table non-empty")


def labels_ratio_shape(classes: Int, count: Int, mode: Int) raises:
    if classes <= 0:
        return
    if mode != 0 and mode != 1:
        raise Error("class_ratio_f64: mode is 0 (balanced) or 1 (prior)")
    if mode == 1 and count == 0:
        raise Error("class_ratio_f64: the prior of zero rows")


def labels_require_address(address: Int, message: String) raises:
    if address == 0:
        raise Error(message)
