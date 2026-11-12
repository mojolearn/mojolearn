# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN piece of `bindings/_mojolearn_x_metrics_host.mojo`: the native publication of the metrics
program entry table at module init. Typed Mojo only, no Python import
(tools/python_free_dirs.sh); moved byte for byte from the binding (lane
rs-metrics-estimators, 2026-11-12)."""
from core.linear_entry_table import LinearEntryKey, linear_table_publish, LINEAR_TABLE_METRICS
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_metrics.descriptor_host import metrics_program_host_v1, METRICS_PROGRAM_HOST_API

def _publish_linear_entry_table() raises:
    # Native publication only; the existing capsule ABI remains for clients
    # not yet converted. The original descriptor owner has image lifetime.
    var api = METRICS_PROGRAM_HOST_API.get_or_create_ptr()
    var key = LinearEntryKey(GLOBAL_NUMERIC_MODE, 0)
    linear_table_publish[LINEAR_TABLE_METRICS](key,
        Int(api[].entries.unsafe_ptr()), [String(METRICS_PROGRAM_ABI)], [MP_OPERATION_COUNT], [metrics_program_host_v1])
