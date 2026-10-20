# SPDX-License-Identifier: Apache-2.0
# The execution object is native. This file only forwards one binding call
# per method. Source/test availability is not installed-release qualification.
class DistributedProgram:
    """Train a captured graph with fixed logical microbatches.

    Create one with ``Program.distribute`` or ``Session.load_distributed``.
    Parameters are copied into independent native workers; optimizer buffers
    are partitioned among the physical devices. The source program remains
    an independent snapshot.
    """

    def __init__(self, binding, handle):
        self._b = binding
        self._id = handle

    def fit(self, *arrays, targets=None, loss="mse", epochs=10, batch_size=32,
            shuffle=True, huber_delta=1.0):
        """Train in one native call and return this call's epoch losses."""
        return self._b.ops_distributed_fit(self._id, arrays, targets, loss,
                                          epochs, batch_size, shuffle, huber_delta)

    def run(self, *arrays):
        """Return inference output from the current trained parameters."""
        return self._b.ops_distributed_run(self._id, arrays)

    def state(self):
        """Return canonical parameter/moment arrays and training counters."""
        return self._b.ops_distributed_state(self._id)

    def residency(self):
        """Return physical devices and owned optimizer element ranges."""
        return self._b.ops_distributed_residency(self._id)

    def save(self, path):
        """Save graph, weights, optimizer, schedule and random counters."""
        return self._b.ops_distributed_save(self._id, path)

    def close(self):
        """Release the native workers and their device storage."""
        return self._b.ops_distributed_close(self._id)

    def __enter__(self):
        return self._b.ops_distributed_enter(self._id, self)

    def __exit__(self, exc_type, exc_value, traceback):
        return self._b.ops_distributed_close(self._id)
