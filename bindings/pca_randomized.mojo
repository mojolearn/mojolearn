# SPDX-License-Identifier: Apache-2.0
# Native transcription of _randomized_decompose/_rsvd_core. The established
# C entries, power iterations, rank guard, sign convention and scalar folds
# are preserved. ClassicalKit owns descriptors and every device allocation;
# the PCA-specific thin SVD does not use OLS's additional U reconstruction.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.classical_tsqr import ClassicalKit
from bindings.classical_arrays import classical_host_f32
from bindings.sequential_arrays import sequential_address
from bindings.sequential_description import sequential_fields
from bindings.pca_exact import pca_copied, pca_prefix, pca_whiten_entries
from bindings.py_balanced import py_call, py_get, py_setitem, py_type


def pca_random_draw(mut kit: ClassicalKit, rows: Int, cols: Int, seed: PythonObject) raises -> Int:
    var count = rows * cols
    var params = Python.list(count, seed, 30, 1)
    # Preserve the existing small-draw ABI choice: short draws are consumed
    # immediately; larger resident draws avoid a host round trip. Both GPU
    # entries execute the same Philox normal kernel, not a host RNG fallback.
    if count >= 1024 and kit.resident:
        var out = kit.device_output(rows, cols)
        _ = py_call(kit.entries["x_decomp_dev_rand"], kit.matrices[out].device, params)
        return out
    var out = kit.zeros(rows, cols)
    if count:
        var address = kit.address(out)
        _ = py_call(kit.entries["x_decomp_rand"], address, params)
    return out


def pca_orth(mut kit: ClassicalKit, a: Int) raises -> Int:
    var rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    if rows * cols and kit.resident:
        var out = kit.device_output(rows, cols)
        var flags = 0
        if py_get(kit.entries, "x_decomp_s_flags") is not None:
            flags = Int(py=kit.entries["x_decomp_s_flags"]())
        var entry = kit.entries["x_decomp_dev_orth_ws"] if flags & 2 else kit.entries["x_decomp_dev_orth"]
        var source = kit.device(a)
        _ = py_call(entry, source, kit.matrices[out].device, Python.list(rows, cols))
        return out
    var out = kit.copy_host(a)
    var address = kit.address(out)
    _ = py_call(kit.entries["x_decomp_orth"], address, Python.list(rows, cols))
    return out


def pca_absmax_signs(mut kit: ClassicalKit, a: Int, columns: Bool) raises -> Int:
    var rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    var count = cols if columns else rows
    var out = kit.device_output(1, count) if kit.resident else kit.zeros(1, count)
    if count and rows * cols:
        if kit.resident:
            var source = kit.device(a)
            _ = py_call(kit.entries["x_decomp_dev_absmax"], source, kit.matrices[out].device, Python.list(rows, cols, Int(columns)))
        else:
            var source = kit.address(a)
            var dest = kit.address(out)
            _ = py_call(kit.entries["x_decomp_absmax_sign"], source, dest, Python.list(rows, cols, Int(columns)))
    if not columns:
        out = kit.transpose(out)
    return kit.signs(out)


def pca_tall_svd(mut kit: ClassicalKit, a: Int) raises -> Tuple[Int, Int]:
    var rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    var s = kit.zeros(1, cols)
    var v = kit.zeros(cols, cols)
    var sa = kit.address(s)
    var va = kit.address(v)
    if rows >= cols and cols >= 1 and kit.resident and py_get(kit.entries, "x_decomp_dev_svd") is not None:
        var source = kit.device(a)
        _ = py_call(kit.entries["x_decomp_dev_svd"], source, sa, va, Python.list(rows, cols))
    else:
        var source = kit.address(a)
        _ = py_call(kit.entries["x_decomp_svd"], source, sa, va, Python.list(rows, cols))
    var negated = kit.ew(14, s, scalar=-1.0)
    var order = kit.order(negated)
    var st = kit.transpose(s)
    var sorted_s = kit.take_rows(st, order)
    s = kit.transpose(sorted_s)
    var vt = kit.transpose(v)
    vt = kit.take_rows(vt, order)
    return (s, vt)


def pca_thin_svd(mut kit: ClassicalKit, a: Int) raises -> Tuple[Int, Int, Int]:
    var rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    var u: Int
    var s: Int
    var vt: Int
    if cols <= rows:
        var raw = pca_tall_svd(kit, a)
        s = raw[0]
        vt = raw[1]
        var av = kit.mm(a, vt, tb=True)
        u = kit.ew(3, av, s)
    else:
        var at = kit.transpose(a)
        var raw = pca_tall_svd(kit, at)
        s = raw[0]
        u = kit.transpose(raw[1])
        var ua = kit.mm(u, a, ta=True)
        var st = kit.transpose(s)
        vt = kit.ew(3, ua, st)
    var signs = pca_absmax_signs(kit, u, True)
    u = kit.ew(2, u, signs)
    var signst = kit.transpose(signs)
    vt = kit.ew(2, vt, signst)
    return (u, s, vt)


def pca_order_small(mut kit: ClassicalKit, a: Int) raises -> Int:
    var count = kit.matrices[a].rows * kit.matrices[a].cols
    var out = kit.device_output(count, 1) if kit.resident else kit.zeros(count, 1)
    if count:
        if kit.resident:
            var source = kit.device(a)
            _ = py_call(kit.entries["x_decomp_dev_order_small"], source, kit.matrices[out].device, Python.list(count))
        else:
            var source = kit.address(a)
            var dest = kit.address(out)
            _ = py_call(kit.entries["x_decomp_order_small"], source, dest, Python.list(count))
    return out


def pca_pad_rows(mut kit: ClassicalKit, a: Int, rows: Int) raises -> Int:
    var old_rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    if rows < old_rows:
        raise Error("PCA native pad cannot remove rows")
    # _rsvd_core appends zero words to downloaded .s buffers here. Preserve
    # that exact copy boundary and do not change rank-deficient arithmetic.
    var out = kit.zeros(rows, cols)
    var source = kit.host(a)
    var dest = kit.host(out)
    var sp = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(source)))
    var dp = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=sequential_address(dest)))
    memcpy(dest=dp, src=sp, count=old_rows * cols * 4)
    _ = source^
    _ = dest^
    return out


def pca_randomized_core(mut kit: ClassicalKit, original: Int, components: Int, oversamples: Int,
                       iterations: Int, seed: PythonObject) raises -> Tuple[Int, Int, Int]:
    var transposed = kit.matrices[original].rows < kit.matrices[original].cols
    var a = kit.transpose(original) if transposed else original
    var rows = kit.matrices[a].rows
    var cols = kit.matrices[a].cols
    var width = min(components + oversamples, min(rows, cols))
    if width < 0:
        raise Error("randomized PCA range width must be nonnegative")
    var q = pca_random_draw(kit, cols, width, seed)
    for i in range(max(iterations, 0)):  # driver-loop: over the requested randomized-SVD power iterations
        var aq = kit.mm(a, q)
        q = pca_orth(kit, aq)
        var atq = kit.mm(a, q, ta=True)
        q = pca_orth(kit, atq)
    var aq = kit.mm(a, q)
    q = pca_orth(kit, aq)
    var absolute = kit.ew(13, q)
    var sums = kit.sum(absolute, True)
    var absolute_sums = kit.ew(13, sums)
    var live = kit.ew(23, absolute_sums)
    var nlive = Int(kit.total(live))
    if nlive == 0:
        raise Error("randomized_svd: M is numerically zero")
    var full = kit.matrices[q].cols
    if nlive < full:
        var negated = kit.ew(14, live, scalar=-1.0)
        var dead = kit.ew(22, negated, scalar=1.0)
        var order = pca_order_small(kit, dead)
        q = kit.move(q, order, rows, nlive, Python.list(4, rows * nlive, nlive, full, 0, 0, 0,
                                                      rows * full, rows * nlive, nlive), kit.resident)
    var small = kit.mm(q, a, ta=True)
    var decomposed = pca_thin_svd(kit, small)
    var uh = decomposed[0]
    var s = decomposed[1]
    var vt = decomposed[2]
    if kit.matrices[s].cols < full:
        var st = kit.transpose(s)
        st = pca_pad_rows(kit, st, full)
        s = kit.transpose(st)
        vt = pca_pad_rows(kit, vt, full)
        var uht = kit.transpose(uh)
        uht = pca_pad_rows(kit, uht, full)
        uh = kit.transpose(uht)
    var u = kit.mm(q, uh)
    var component_rows = kit.indices(0, components)
    var out_s = kit.columns(s, 0, components)
    if transposed:
        var out_u = kit.take_rows(vt, component_rows)
        out_u = kit.transpose(out_u)
        var out_v = kit.columns(u, 0, components)
        out_v = kit.transpose(out_v)
        return (out_u, out_s, out_v)
    var out_u = kit.columns(u, 0, components)
    var out_v = kit.take_rows(vt, component_rows)
    return (out_u, out_s, out_v)


def pca_randomized_fit(model: PythonObject, args: PythonObject, entries: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var b = Python.import_module("builtins")
    if Float64(py=fields["tol"]) != 0:
        raise Error("NotImplementedError: PCA tol is ARPACK's tolerance; leave tol=0.0")
    if Bool(py=fields["whiten"]):
        pca_whiten_entries(entries)
    var x = classical_host_f32(args[0], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if min(rows, cols) < 2:
        raise Error("randomized PCA requires at least two rows and features")
    var requested = fields["n_components"]
    var nc = min(rows, cols) if requested is None else Int(py=py_call(b.int, requested))
    if nc < 1 or nc >= min(rows, cols):
        raise Error("svd_solver='randomized' needs 1 <= n_components < min(n_samples, n_features)")
    var normalizer = String(fields["power_iteration_normalizer"])
    if normalizer != "auto" and normalizer != "QR" and normalizer != "LU" and normalizer != "none":
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var power = fields["iterated_power"]
    # Incumbent Halko iteration policy, not a benchmark-shape selection rule.
    var iterations = (7 if Float64(nc) < 0.1 * Float64(min(rows, cols)) else 4) if py_type(power) is b.str and String(power) == "auto" else Int(py=py_call(b.int, power))
    var oversamples = Int(py=py_call(b.int, fields["n_oversamples"]))
    if oversamples < 0:
        raise Error("n_oversamples must be nonnegative")
    var seed = fields["random_state"]
    if seed is None:
        seed = PythonObject(0)
    elif py_type(seed) is not b.int:
        raise Error("TypeError: random_state must be None or an int")
    seed = seed.__and__(0xFFFFFFFF)
    var decomp = py_get(entries, "randomized_bindings")
    if py_type(decomp) is not b.dict:
        raise Error("ImportError: randomized PCA native entries are unavailable")
    var kit = ClassicalKit(decomp, context)
    try:
        var original = kit.add(x, rows, cols)
        var sums = kit.sum(original, True)
        var mean = kit.ew(14, sums, scalar=1.0 / Float64(rows))
        var centered = kit.ew(1, original, mean)
        var decomposed = pca_randomized_core(kit, centered, nc, oversamples, iterations, seed)
        var u = decomposed[0]
        var s = decomposed[1]
        var vt = decomposed[2]
        var signs = pca_absmax_signs(kit, vt, False)
        vt = kit.ew(2, vt, signs)
        var signst = kit.transpose(signs)
        u = kit.ew(2, u, signst)
        var squared = kit.ew(8, s)
        var variance = kit.ew(14, squared, scalar=1.0 / Float64(rows - 1))
        var data_squared = kit.ew(8, centered)
        var total_value = kit.total(data_squared)
        var total = kit.constant(Float32(total_value))
        total = kit.ew(14, total, scalar=1.0 / Float64(rows - 1))
        var ratio = kit.ew(3, variance, total)
        var explained_value = kit.total(variance)
        var explained = kit.constant(Float32(explained_value))
        var noise = kit.ew(1, total, explained)
        noise = kit.ew(14, noise, scalar=1.0 / Float64(min(rows, cols) - nc))
        var components_array = pca_prefix(kit.host(vt), nc, cols, context)
        var mean_array = pca_prefix(kit.host(mean), cols, 0, context)
        var singular_array = pca_prefix(kit.host(s), nc, 0, context)
        var variance_array = pca_prefix(kit.host(variance), nc, 0, context)
        var ratio_array = pca_prefix(kit.host(ratio), nc, 0, context)
        var noise_value = kit.word(noise)
        py_setitem(fields, "components_", components_array)
        py_setitem(fields, "mean_", mean_array)
        py_setitem(fields, "singular_values_", singular_array)
        py_setitem(fields, "explained_variance_", variance_array)
        py_setitem(fields, "explained_variance_ratio_", ratio_array)
        py_setitem(fields, "noise_variance_", PythonObject(noise_value))
        py_setitem(fields, "n_components_", nc)
        py_setitem(fields, "n_features_in_", cols)
        py_setitem(fields, "n_samples_", rows)
        py_setitem(fields, "input_copied_", pca_copied(args[0], context))
        py_setitem(fields, "_mojolearn_fitted", True)
        _ = x^
        return model
    finally:
        kit.close()
