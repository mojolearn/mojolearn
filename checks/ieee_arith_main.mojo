# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Standalone entry of checks/ieee_arith_check.mojo, for the box start-up gate (tools/ops/box_job_lib.sh
arith_gate): one small build (the check imports only checks/numerics), a ~2 s run on the box's GPU, a nonzero
exit when an identical-path operation (arm 8: portable_sqrtf, portable_divf; raw division; the signed-zero
table) diverges on this backend. The registered check (checks/native_tests/registry.mojo) is the same body."""
from checks.ieee_arith_check import run_checks


def main() raises:
    run_checks()
