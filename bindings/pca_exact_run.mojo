# SPDX-License-Identifier: Apache-2.0
"""RUN of the PCA helpers (bindings/pca_exact.mojo): the compiled-entry and
too-small refusal texts. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word."""


def pca_refusal(var error: Error) -> Error:
    """A compiled PCA entry's refusal. A builtin class prefix is kept; a
    plain refusal of the data or the fitted state (non-finite values, a
    numerically zero matrix) becomes a ValueError naming PCA, as the public
    shell's `_native` raised it."""
    var message = String(error)
    if (message.startswith("ValueError: ") or message.startswith("TypeError: ") or message.startswith("IndexError: ")
            or message.startswith("AttributeError: ") or message.startswith("RuntimeError: ")
            or message.startswith("NotImplementedError: ") or message.startswith("ZeroDivisionError: ")
            or message.startswith("OverflowError: ") or message.startswith("ArithmeticError: ")
            or message.startswith("ImportError: ") or message.startswith("FileNotFoundError: ")):
        return error^
    return Error("ValueError: mojolearn PCA: " + message)


def pca_too_small(rows: Int, cols: Int) -> String:
    # scikit-learn's check_array wording for a block below 2 x 2.
    if rows < 2:
        return ("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", " + String(cols)
                + ")) while a minimum of 2 is required by PCA.")
    return ("Found array with " + String(cols) + " feature(s) (shape=(" + String(rows) + ", " + String(cols)
            + ")) while a minimum of 2 is required by PCA.")
