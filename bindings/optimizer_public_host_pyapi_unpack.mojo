# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the CPU binding
(`_mojolearn_training_host`; lane neural-optimizers, 2026-10-21; lane
optimizers-thin-glue): the GPU door's entries (bindings/optimizer_public_pyapi.mojo)
over training/host/optimizer_public_host.mojo, through the shared door
bindings/optimizer_public_door.mojo (`HostOptKernels`)."""
from std.python import PythonObject
from std.python._cpython import PyObjectPtr

from bindings.optimizer_public_door import (
    entry_accumulate, entry_clip, entry_io, entry_load_state, entry_state, entry_state_set,
    entry_step,
)
from training.host.optimizer_public_host import (
    host_close, host_download, host_open, host_upload,
)


