# SPDX-License-Identifier: Apache-2.0
# The execution object is native. This file only forwards one binding call
# per method. Source/test availability is not installed-release qualification.
class PartitionedProgram:
    """Run a captured graph with model, tensor and pipeline partitions.

    Create one with ``Program.partition`` or ``Session.load_partitioned``.
    Logical stages and tensor slices remain fixed when physical devices
    change. Parameters and optimizer state have one device owner per slice.
    """

    def __init__(self, binding, handle):
        self._b = binding
        self._id = handle

    def fit(self, *arrays, targets=None, loss="mse", epochs=10, batch_size=32,
            shuffle=True, huber_delta=1.0):
        """Train in one native call and return this call's epoch losses."""
        return self._b.ops_partitioned_fit(self._id, arrays, targets, loss,
                                          epochs, batch_size, shuffle, huber_delta)

    def run(self, *arrays):
        """Return inference output from the current trained parameters."""
        return self._b.ops_partitioned_run(self._id, arrays)

    def state(self):
        """Return canonical parameter/moment arrays and training counters."""
        return self._b.ops_partitioned_state(self._id)

    def residency(self):
        """Return physical devices and unique parameter owners."""
        return self._b.ops_partitioned_residency(self._id)

    def profile(self):
        """Return native operation intervals from the latest run or fit."""
        return self._b.ops_partitioned_profile(self._id)

    def save(self, path):
        """Save graph, weights, optimizer, schedule and random counters."""
        return self._b.ops_partitioned_save(self._id, path)

    def close(self):
        """Release the native workers and their device storage."""
        return self._b.ops_partitioned_close(self._id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
