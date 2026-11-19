# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Standalone wrapper of `gemm/checks/gemm_epilogue_suite.mojo` (the store
epilogue arm, `-D MOJOLEARN_GEMM_FAST_EPI=1`): fused store against the
two-launch form and the host column, word for word."""
from gemm.checks.gemm_epilogue_suite import run_checks


def run_checks() raises:
    run_checks()
