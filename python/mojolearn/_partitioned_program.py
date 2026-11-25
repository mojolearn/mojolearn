# SPDX-License-Identifier: Apache-2.0
# The execution object is native. This file only forwards one binding call
# per method. Source/test availability is not installed-release qualification.
_OWNED_PROGRAM_CALL = None

class PartitionedProgram:
    """Run a captured graph with model, tensor and pipeline partitions.

    Create one with ``Program.partition`` or ``Session.load_partitioned``.
    Logical stages and tensor slices remain fixed when physical devices
    change. Parameters and optimizer state have one device owner per slice.
    """

    def __init__(self, binding, handle):
        return _OWNED_PROGRAM_CALL(self, "initialize", (binding, handle))

    def fit(self, *arrays, targets=None, loss="mse", epochs=10, batch_size=32,
            shuffle=True, huber_delta=1.0):
        """Train in one native call and return this call's epoch losses.

        Each optimizer step splits its minibatch into its microbatches and combines
        their losses as the sum of the numerators over the sum of the
        denominators, so a part with few valid elements counts for few. With
        the built-in losses (``"mse"``, ``"mae"``, ``"huber"``,
        ``"bce_with_logits"``, ``"cross_entropy"``) each row counts once.
        For a masked or weighted loss, capture a program with two scalar
        outputs, the weighted loss sum and its denominator (the sum of the
        weights over the valid elements), and pass ``loss="output"``: the
        step's loss is then the sum of the sums over the sum of the
        denominators, never an average of the parts' means.

        Example
        -------
        >>> p = s.capture([x, w], [loss_sum, denominator])  # doctest: +SKIP
        >>> pp.fit(X, W, loss="output", epochs=5)  # doctest: +SKIP
        """
        return _OWNED_PROGRAM_CALL(self, "fit", (arrays, targets, loss, epochs, batch_size, shuffle, huber_delta))

    def run(self, *arrays):
        """Return inference output from the current trained parameters."""
        return _OWNED_PROGRAM_CALL(self, "run", (arrays,))

    def state(self):
        """Return canonical parameter/moment arrays and training counters."""
        return _OWNED_PROGRAM_CALL(self, "state", ())

    def residency(self):
        """Return physical devices and unique parameter owners."""
        return _OWNED_PROGRAM_CALL(self, "residency", ())

    def profile(self):
        """Return native operation intervals from the latest run or fit."""
        return _OWNED_PROGRAM_CALL(self, "profile", ())

    def save(self, path):
        """Save graph, weights, optimizer, schedule and random counters."""
        return _OWNED_PROGRAM_CALL(self, "save", (path,))

    def close(self):
        """Release the native workers and their device storage."""
        return _OWNED_PROGRAM_CALL(self, "close", ())

    def __enter__(self):
        return _OWNED_PROGRAM_CALL(self, "enter", ())

    def __exit__(self, exc_type, exc_value, traceback):
        return _OWNED_PROGRAM_CALL(self, "exit", ())
