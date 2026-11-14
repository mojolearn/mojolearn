# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.core_chunked_lm_head_denormals_gpu_suite import run_checks


def main() raises:
    run_checks()
