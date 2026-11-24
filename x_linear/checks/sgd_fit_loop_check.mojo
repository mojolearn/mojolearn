# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Standalone wrapper of x_linear/checks/sgd_fit_loop_suite.mojo (the SGD
family's fit_loop_ record on the host column):

    pixi run mojo run -I . x_linear/checks/sgd_fit_loop_check.mojo
"""
from x_linear.checks.sgd_fit_loop_suite import run_checks


def main() raises:
    run_checks()
