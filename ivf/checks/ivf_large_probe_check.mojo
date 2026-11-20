# SPDX-License-Identifier: Apache-2.0
# Standalone wrapper of `ivf/checks/ivf_large_probe_check_suite.mojo`: the checks, fixtures and helpers live there
# (a module with main() cannot be imported; split by lane fx-checks-orphans, 2026-11-18).
# Import from the suite, never from this file.
from ivf.checks.ivf_large_probe_check_suite import (
    check_large_probes,
)


def main() raises:
    check_large_probes()
