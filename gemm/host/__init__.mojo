# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""gemm/host: the host oracle that SHIPS in a CPU host binding
(bindings/_mojolearn_*_host.mojo; python/mojolearn/host_surface.py names
which), moved out of gemm/checks/ by the host surface manifest lane,
2026-09-14. gemm/checks/gemm_oracle.mojo keeps a re-export under the old
name, so every check and bench that imported it still does."""
