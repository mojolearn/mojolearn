# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""metrics: cuML `cpp/src/metrics/` and the RAFT `raft/stats/` headers they call.

`metrics/ported/metrics/` mirrors cuML's thin wrappers file for file;
`metrics/ported/stats/` mirrors the RAFT implementations; `metrics/mojo_only/`
is what they never needed (the pinned reductions, the oracles, the checks).
See `metrics/README.md`.
"""
