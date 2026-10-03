# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RandomForest builders' NaN refusal, one string for the GPU bindings
(`ensemble/device_layout.mojo`'s device scan) and the CPU-only install
(`ensemble/host/rf_oracle.mojo`'s host scan). No code: importing it pulls
neither host threads nor device code into the importer."""

comptime RF_NAN_REFUSAL = (
    "X contains NaN; the forest has no missing-value arm (a NaN bins left"
    " but partitions right, and the fit would not terminate)"
)
"""The RandomForest builders' NaN refusal, host and GPU bindings alike."""
