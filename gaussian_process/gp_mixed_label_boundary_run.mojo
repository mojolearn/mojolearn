# SPDX-License-Identifier: Apache-2.0
"""The typed bucket (first and last row of one hash chain) of the mixed-label scalar index (RUN).
The index itself keeps the admitted Python label identities and stays in the door
(`bindings/gp_mixed_label_boundary_door.mojo`)."""


@fieldwise_init
struct GPScalarBucket(ImplicitlyCopyable, Movable):
    var first: Int
    var last: Int
