# SPDX-License-Identifier: Apache-2.0
"""Standalone wrapper; imported package fixtures must not define main."""
from bench.apple_callpath_quality import run_quality


def main() raises:
    run_quality()
