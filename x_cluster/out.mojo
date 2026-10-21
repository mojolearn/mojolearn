# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What every cluster-lane entry returns (lane/algos-cluster). Host code; no
device, no Python: the Python boundary both bindings share is
`x_cluster/out_pyapi.mojo`."""


struct ClusterOut(Movable):
    """A fit's answer: float arrays, int arrays and float64 scalars, in the
    order the entry documents. `cluster_out_to_py` (x_cluster/out_pyapi.mojo)
    returns `[[f...], [i...], [s...]]`: each float array an `array.array('f')`, each int array an `array.array('i')`
    (one memcpy each), the scalars a list of Python floats."""

    var f: List[List[Float32]]
    var i: List[List[Int32]]
    var s: List[Float64]

    def __init__(out self):
        self.f = List[List[Float32]]()
        self.i = List[List[Int32]]()
        self.s = List[Float64]()
