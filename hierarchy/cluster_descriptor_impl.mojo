# SPDX-License-Identifier: Apache-2.0
# Transport only: reuse agglo_entry and its existing host/device operations.
from std.memory import memcpy
from core.hostptr import read_f32, f32_ptr, i32_ptr, f64_ptr
from x_cluster.entries import agglo_entry
from x_cluster.ops import ClusterOps
from hierarchy.cluster_descriptor_schema import hierarchy_cluster_validate_buffers


def hierarchy_cluster_fit[O: ClusterOps](mut ops: O, a: List[Int], p: List[Int],
                                         floats: Int) raises:
    hierarchy_cluster_validate_buffers(a, p)
    if floats == 0:
        raise Error("hierarchy cluster provider: null scalar span")
    var fp = f64_ptr(floats)
    var scalars: List[Float64] = [fp[0], fp[1]]
    var x = read_f32(a[0], p[0]) if p[0] > 0 else List[Float32]()
    var edges = read_f32(a[1], p[1]) if p[1] > 0 else List[Float32]()
    var ip = List[Int]()
    for i in range(2, 11):  # small-loop(nine typed agglo_entry settings)
        ip.append(p[i])
    var out = agglo_entry(ops, x, edges, ip, scalars)
    # Validate every slot before publishing any result bytes. Empty partial
    # trees (n_merges=0, cut=0) still return n labels and two scalar words.
    if len(out.i) != 2 or len(out.f) != 1 or len(out.s) != 2:
        raise Error("hierarchy cluster provider: invalid numerical output slots")
    if len(out.i[0]) != 2 * p[7] or len(out.i[1]) != p[2] or len(out.f[0]) != p[7]:
        raise Error("hierarchy cluster provider: invalid numerical output dimensions")
    if p[7] > 0:
        memcpy(dest=i32_ptr(a[2]), src=out.i[0].unsafe_ptr(), count=2 * p[7])
        memcpy(dest=f32_ptr(a[4]), src=out.f[0].unsafe_ptr(), count=p[7])
    memcpy(dest=i32_ptr(a[3]), src=out.i[1].unsafe_ptr(), count=p[2])
    memcpy(dest=f64_ptr(a[5]), src=out.s.unsafe_ptr(), count=2)
    _ = out^
    _ = x^
    _ = edges^
