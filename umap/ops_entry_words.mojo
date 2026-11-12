# SPDX-License-Identifier: Apache-2.0
"""The UMAP operations table's word decoding (core/neighbors_entry_table_umap.mojo),
shared by the device column (umap/ops_entry_device.mojo) and the CPU column
(umap/ops_entry_host.mojo): the refusals the Python exports
`umap_fit_transform`, `umap_fit_transform_ex` and `umap_transform` made, in
the same order and with the same texts, then `UMAPParams`. Every float word
is rounded with `Float32(Float64)`, the exports' `Float32(Float64(py=...))`.
No Python.
"""
from umap.params import UMAPParams

comptime UmapWordPtr = MutPointer[Int, MutUntrackedOrigin]
comptime UmapFloatPtr = MutPointer[Float64, MutUntrackedOrigin]


def umap_fit_config(p: UmapWordPtr, np: Int, f: UmapFloatPtr) raises -> UMAPParams:
    """UMAP_OP_FIT: p [n, d, n_neighbors, n_components, n_epochs, seed
    (, negative_sample_rate)], f [min_dist, spread, set_op_mix_ratio,
    local_connectivity (, learning_rate, repulsion_strength)]."""
    var n = p[0]
    var d = p[1]
    var seed = p[5]
    if d < 1 or seed < 0:
        raise Error("UMAP requires positive features and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=p[2], n_components=p[3],
        n_epochs=p[4], min_dist=Float32(f[0]),
        spread=Float32(f[1]),
        set_op_mix_ratio=Float32(f[2]),
        local_connectivity=Float32(f[3]),
        random_seed=UInt64(seed),
    )
    if np == 7:
        config.learning_rate = Float32(f[4])
        config.repulsion_strength = Float32(f[5])
        config.negative_sample_rate = p[6]
    config.validate(n)
    if (config.n_components != 2 and config.n_components != 3) or (
        n < 2 * config.n_components + 4
    ):
        raise Error("UMAP requires 2D/3D output and enough samples for spectral init")
    return config^


def umap_fit_ex_config(a: UmapWordPtr, p: UmapWordPtr, f: UmapFloatPtr) raises -> UMAPParams:
    """UMAP_OP_FIT_EX: a [x, embedding, initial, target], p [n, d,
    n_neighbors, n_components, n_epochs, seed, negative_sample_rate, metric,
    target_kind, target_dims, target_n_neighbors], f [min_dist, spread,
    set_op_mix_ratio, local_connectivity, learning_rate, repulsion_strength,
    metric_arg, a, b, target_weight]."""
    var n = p[0]
    var d = p[1]
    var seed = p[5]
    if d < 1 or seed < 0:
        raise Error("UMAP requires positive features and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=p[2], n_components=p[3],
        n_epochs=p[4], min_dist=Float32(f[0]),
        spread=Float32(f[1]),
        set_op_mix_ratio=Float32(f[2]),
        local_connectivity=Float32(f[3]),
        random_seed=UInt64(seed),
        learning_rate=Float32(f[4]),
        repulsion_strength=Float32(f[5]),
        negative_sample_rate=p[6],
        metric=p[7],
        metric_arg=Float32(f[6]),
        curve_a=Float32(f[7]),
        curve_b=Float32(f[8]),
    )
    config.validate(n)
    var tkind = p[8]
    var tdims = p[9]
    var tk = p[10]
    var tw = Float32(f[9])
    var init_addr = a[2]
    var target_addr = a[3]
    if config.n_components < 1 or config.n_components > 32:
        raise Error("UMAP supports 1 to 32 output dimensions")
    if init_addr == 0 and n < 2 * config.n_components + 4:
        raise Error("UMAP has too few samples for spectral init")
    if tkind < 0 or tkind > 2 or (tkind != 0 and (target_addr == 0 or tdims < 1)):
        raise Error("UMAP supervised target is malformed")
    if tk < 0 or tk > n or not (tw >= Float32(0.0) and tw <= Float32(1.0)):
        raise Error("UMAP target_n_neighbors or target_weight is out of range")
    return config^


def umap_transform_config(p: UmapWordPtr, np: Int, f: UmapFloatPtr) raises -> UMAPParams:
    """UMAP_OP_TRANSFORM: p [n_train, n_queries, n_features, n_neighbors,
    n_components, n_epochs, seed (, negative_sample_rate (, metric))],
    f [min_dist, spread, set_op_mix_ratio, local_connectivity
    (, learning_rate, repulsion_strength (, metric_arg, a, b))]."""
    var n = p[0]
    var rows = p[1]
    var d = p[2]
    var seed = p[6]
    if n < 2 or rows < 1 or d < 1 or seed < 0:
        raise Error("UMAP transform requires positive dimensions and a nonnegative seed")
    var config = UMAPParams(
        n_neighbors=p[3], n_components=p[4],
        n_epochs=p[5], min_dist=Float32(f[0]),
        spread=Float32(f[1]),
        set_op_mix_ratio=Float32(f[2]),
        local_connectivity=Float32(f[3]), random_seed=UInt64(seed),
    )
    if np >= 8:
        config.learning_rate = Float32(f[4])
        config.repulsion_strength = Float32(f[5])
        config.negative_sample_rate = p[7]
    if np == 9:
        # lane/algos-decomp option parity: metric, metric_arg, a, b
        config.metric = p[8]
        config.metric_arg = Float32(f[6])
        config.curve_a = Float32(f[7])
        config.curve_b = Float32(f[8])
    config.validate(n)
    if config.n_components < 1 or config.n_components > 32:
        raise Error("UMAP transform supports 1 to 32 output dimensions")
    return config^
