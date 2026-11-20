# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Connection to the compiled tree expansion binding."""


def _trees_x_bind(est=None):
    from . import _backend
    return _backend.binding("_mojolearn_x_trees", getattr(est, "numeric_mode", None))
