# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/classical_validate.mojo: typed Mojo only (no Python
# imports).


def classical_check_width(cols: Int, fitted: Int, est: String) raises:
    if cols != fitted:
        raise Error("X has " + String(cols) + " features, but " + est + " is expecting " + String(fitted) + " features as input.")
