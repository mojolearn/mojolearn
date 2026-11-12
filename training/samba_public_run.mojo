# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba public boundary (lane rs-neural, wave 8 run split,
from bindings/samba_public_native.mojo): which public operations write the
stack's flat registry in place. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Same body."""


def _samba_writes_registry(operation: String) -> Bool:
    """The operations that write the stack's flat registry in place (integ-w4,
    lane x45c's weight key in SambaStackProgram): each one is native code
    writing by address and moves the weight epoch, so a resident program
    reloads at its next call. Lane x56d (the contract of every neural
    family, decided 2026-11-09): the registry and the public parameter views
    are read-only to callers (bindings/samba_state_native.mojo `samba_view`),
    so no write outside these operations can leave a resident copy stale:
    the bump-only key no longer has a hole."""
    return (operation == "model_train_step" or operation == "apply" or operation == "publish"
            or operation == "checkpoint_load" or operation == "load_weights")
