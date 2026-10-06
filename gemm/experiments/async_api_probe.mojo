# SPDX-License-Identifier: Apache-2.0
"""Compile-time probe of the installed supported NVIDIA async-copy API."""
from max.gpu.memory import async_copy,async_copy_commit_group,async_copy_wait_all,AddressSpace

def main():
    print("N03_API_IMPORT_PASS completion_and_commit_primitives_available")
