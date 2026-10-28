# SPDX-License-Identifier: Apache-2.0
# The execution object is native. This file only forwards one binding call
# per method. Source/test availability is not installed-release qualification.
_OWNED_PROGRAM_CALL = None

class DistributedProgram:
    """Train a captured graph with fixed logical microbatches.

    Create one with ``Program.distribute`` or ``Session.load_distributed``.
    Parameters are copied into independent native workers; optimizer buffers
    are partitioned among the physical devices. The source program remains
    an independent snapshot.
    """

    def __init__(self, binding, handle):
        return _OWNED_PROGRAM_CALL(self, "initialize", (binding, handle))

    def fit(self, *arrays, targets=None, loss="mse", epochs=10, batch_size=32,
            shuffle=True, huber_delta=1.0):
        """Train in one native call and return this call's epoch losses."""
        return _OWNED_PROGRAM_CALL(self, "fit", (arrays, targets, loss, epochs, batch_size, shuffle, huber_delta))

    def run(self, *arrays):
        """Return inference output from the current trained parameters."""
        return _OWNED_PROGRAM_CALL(self, "run", (arrays,))

    def state(self):
        """Return canonical parameter/moment arrays and training counters."""
        return _OWNED_PROGRAM_CALL(self, "state", ())

    def residency(self):
        """Return physical devices and owned optimizer element ranges."""
        return _OWNED_PROGRAM_CALL(self, "residency", ())

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


def _connect_owned_programs(connections):
    """Import-time only: connect every registry before any public call."""
    global _OWNED_PROGRAM_CALL
    from . import _partitioned_program
    from types import SimpleNamespace
    records = tuple(vars(value) for value in connections.values() if type(value) is SimpleNamespace)  # glue: installed native mode connections
    for destination in records:  # glue: connect every native image during import
        connect = destination["native_module"].ops_owned_program_connect
        for source in records:  # glue: retain every available native registry provider
            connect(DistributedProgram, _partitioned_program.PartitionedProgram,
                    source["distributed"], source["native_module"])
    if records:
        _OWNED_PROGRAM_CALL = records[0]["native_module"].ops_owned_program_call
        _partitioned_program._OWNED_PROGRAM_CALL = _OWNED_PROGRAM_CALL
