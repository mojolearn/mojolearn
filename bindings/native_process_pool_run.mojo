# SPDX-License-Identifier: Apache-2.0
"""The typed per-worker reap outcome of the native process pool (RUN)."""


struct ReapOutcome(Movable):
    """Per worker, in pool order: the reap's error ("" when reaped), the
    seconds from the start of the reap to it, and the signal it needed."""
    var errors: List[String]
    var seconds: List[Float64]
    var signal: List[Int]

    def __init__(out self, n: Int):
        self.errors = List[String](length=n, fill=String())
        self.seconds = List[Float64](length=n, fill=0.0)
        self.signal = List[Int](length=n, fill=0)
