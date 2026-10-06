# SPDX-License-Identifier: Apache-2.0
"""C43 raw dense affinity plus retained degree, shared by host and device.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
The consumer reproduces the materialized quotient(s), then the fixed ordered
product. This descriptor is per fit, with no cache across graph mutations.
"""
from x_neighbors.items import FP, _add
from checks.numerics import ftz, identical_div, identical_sqrt, identical_mul_add


def classical_graph_degree(t: Int,a: FP,degree: FP,n: Int,variant: Int):
    var total=Float32(0)
    for j in range(n):
        if variant == 0:
            total=_add(total,a.unsafe_load(t*n+j))
        elif j != t:
            total=_add(total,a.unsafe_load(j*n+t))
    if total == Float32(0):
        total=Float32(1)
    elif variant != 0:
        total=ftz(identical_sqrt(total))
    degree.unsafe_store(t,total)


def classical_graph_product(t: Int,a: FP,degree: FP,x: FP,dst: FP,n: Int,c: Int,variant: Int):
    var row=t//c
    var col=t%c
    var acc=Float32(0)
    for neighbor in range(n):
        var value=ftz(a.unsafe_load(row*n+neighbor))
        if variant == 0:
            value=ftz(identical_div(value,degree.unsafe_load(row)))
        elif row == neighbor:
            value=Float32(0)
        else:
            value=ftz(identical_div(value,degree.unsafe_load(neighbor)))
            value=ftz(identical_div(value,degree.unsafe_load(row)))
        acc=ftz(identical_mul_add(value,ftz(x.unsafe_load(neighbor*c+col)),acc))
    dst.unsafe_store(t,acc)
