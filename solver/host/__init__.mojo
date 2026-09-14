# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""solver/host: the host oracle that SHIPS in a CPU host binding
(bindings/_mojolearn_solver_host.mojo; python/mojolearn/host_surface.py
names it), moved out of solver/checks/ by the host surface manifest lane,
2026-09-14. solver/checks/cd_oracle.mojo keeps a re-export under the old
name. solver/checks/profile_dot.mojo stays where it is: it carries the
DEVICE dot (a DeviceContext entry) beside the host one and is imported by
the GPU implementation, so it is not a host module."""
