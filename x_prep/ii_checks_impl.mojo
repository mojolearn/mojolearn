# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host x_prep binding's column-index check, typed and Python free
(moved word for word from `bindings/_mojolearn_x_prep_host.mojo`)."""
from x_prep.user_host import I32P


def check_gather_columns(cp: I32P, nc: Int, dk: Int) raises:
    """Every one of the `nc` column indices at `cp` lies in `[0, dk)`."""
    for c in range(nc):
        if Int(cp[c]) < 0 or Int(cp[c]) >= dk:
            raise Error("x_prep ii_gather: a column is out of range")
