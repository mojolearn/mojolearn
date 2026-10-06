# SPDX-License-Identifier: Apache-2.0
"""Compiled host CNN multi-epoch training; no Python runtime dependency."""
from std.memory import memcpy
from bindings.hostptr import f64_ptr
from x_cnn.training import TrainingSpec, training_steps, validate_spec, optimizer_base, epoch_loss_mean
from x_cnn.ops import (
    FP, IP, AH_ROW, epoch_key, epoch_rows_prm, epoch_rows_at, adam_hyper_at,
    CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW,
    PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW,
)
from x_cnn.host.ops_host import (
    conv_block_forward_into, conv_block_backward_into,
    linear_forward_into, linear_backward_into, softmax_xent_into,
    sgd_into, adam_into,
)


def _at(address: Int) -> FP:
    return FP(unsafe_from_address=address)


def out_f32(p: FP, n: Int):
    # Native trainer output seam; production is a no-op like the binding seam.
    _ = p
    _ = n


def _pool_counts(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def fit_epochs(
    spec: TrainingSpec, losses_addr: Int, curve_addr: Int,
    n: Int, batch: Int, adam: Bool, done: Int, first_epoch: Int,
    epochs: Int, shuffle: Bool, seed: UInt64, fparams: List[Float64],
) raises -> Int:
    """Same step, seed, optimizer and ordered loss contract as the GPU twin."""
    var steps = training_steps(n, batch, epochs, done, first_epoch)
    validate_spec(spec)
    var base = optimizer_base(fparams, adam)
    var completed_steps = done
    if epochs == 0:
        return 0
    var losses = f64_ptr(losses_addr)
    var curve = f64_ptr(curve_addr)
    var nb = len(spec.blocks)
    var flat = spec.dims[0]
    var k = spec.dims[1]
    var row = spec.data[2]
    var rows = List[Int32](length=n, fill=Int32(0))
    var order = rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var hyper = List[Float32](length=steps * AH_ROW + 1 if adam else 1, fill=Float32(0))
    var noidx = List[Int32](length=1, fill=Int32(0))
    var pnoidx = noidx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var xd = _at(spec.arrays[0])
    var yd = _at(spec.arrays[1])
    var xs = _at(spec.data[0])
    var ys = _at(spec.data[1])
    var yl = IP(unsafe_from_address=spec.arrays[1])
    for ep in range(epochs):
        var key = epoch_key(seed, first_epoch + ep)
        var prm = epoch_rows_prm(n, 0, shuffle, key)
        var pp = prm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var none = FP(unsafe_from_address=Int(order))
        for i in range(n):
            epoch_rows_at(i, none, none, none, none, order, pp)
        _ = prm^
        if adam:
            var ap: List[Int32] = [Int32(completed_steps + 1)]
            var pb = base.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var po = hyper.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var pa = ap.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            for t in range(steps):
                adam_hyper_at(t, pb, po, po, po, pa, pa)
            _ = ap^
        for st in range(steps):
            var s0 = st * batch
            var m = min(batch, n - s0)
            var q = 0 if m == batch else 1
            # the two row gathers (x_cnn_res_gather: the batch, then its labels)
            for r in range(m):
                memcpy(dest=xd + r * row, src=xs + Int(rows[s0 + r]) * row, count=row)
            for r in range(m):
                memcpy(dest=yd + r, src=ys + Int(rows[s0 + r]), count=1)
            var src = spec.arrays[0]
            for j in range(nb):
                var cprm = spec.conv[q][j].copy()
                var pprm = spec.pool[q][j].copy()
                var pool = spec.pooled[q][j]
                var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
                var no = _pool_counts(pprm)[1] if pool else ny
                var pi = IP(unsafe_from_address=spec.blocks[j][5]) if pool else pnoidx
                conv_block_forward_into(
                    _at(src), _at(spec.blocks[j][0]), _at(spec.blocks[j][1]), _at(spec.blocks[j][4]), pi, cprm, pprm, pool, True,
                    _at(spec.blocks[j][7]), _at(spec.blocks[j][8]),
                )
                out_f32(_at(spec.blocks[j][4]), no)
                src = spec.blocks[j][4]
            linear_forward_into(_at(src), _at(spec.head[0]), _at(spec.head[1]), _at(spec.arrays[2]), m, flat, k)
            out_f32(_at(spec.arrays[2]), m * k)
            for i in range(m):
                if Int(yl[i]) >= k:
                    raise Error("x_cnn softmax: a label is not a class index")
            var loss = softmax_xent_into(_at(spec.arrays[2]), yl, _at(spec.arrays[3]), _at(spec.arrays[4]), m, k)
            out_f32(_at(spec.arrays[3]), m * k)
            out_f32(_at(spec.arrays[4]), m * k)
            losses[ep * steps + st] = Float64(loss)
            var glast = spec.blocks[nb - 1][6] if nb > 0 else spec.arrays[5]
            linear_backward_into(_at(src), _at(spec.head[0]), _at(spec.arrays[3]), _at(glast), _at(spec.head[2]), _at(spec.head[3]), m, flat, k)
            out_f32(_at(glast), m * flat)
            out_f32(_at(spec.head[2]), k * flat)
            out_f32(_at(spec.head[3]), k)
            for jj in range(nb):
                var j = nb - 1 - jj
                var cprm = spec.conv[q][j].copy()
                var pprm = spec.pool[q][j].copy()
                var pool = spec.pooled[q][j]
                var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
                var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
                var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
                var bsrc = spec.blocks[j - 1][4] if j > 0 else spec.arrays[0]
                var dx = spec.blocks[j - 1][6] if j > 0 else 0
                var need_dx = dx != 0
                var pi = IP(unsafe_from_address=spec.blocks[j][5]) if pool else pnoidx
                var pdb = _at(spec.blocks[j][3])
                var pdx = _at(dx) if need_dx else pdb
                conv_block_backward_into(
                    _at(bsrc), _at(spec.blocks[j][0]), _at(spec.blocks[j][1]), _at(spec.blocks[j][6]), pi, pdx, _at(spec.blocks[j][2]), pdb, cprm,
                    pprm, pool, need_dx, True, _at(spec.blocks[j][7]), _at(spec.blocks[j][8]),
                )
                if need_dx:
                    out_f32(pdx, nx)
                out_f32(_at(spec.blocks[j][2]), OC * ckk)
                out_f32(pdb, OC)
            var h = List[Float32]()
            if adam:
                for e in range(AH_ROW):
                    h.append(hyper[st * AH_ROW + e])
            else:
                var offset = 0 if completed_steps + st == 0 else 6
                for e in range(6):
                    h.append(base[offset + e])
            # one optimizer entry per parameter, in the parameters' order
            for j in range(len(spec.sizes)):
                var nj = spec.sizes[j]
                if adam:
                    adam_into(_at(spec.weights[j]), _at(spec.gradients[j]), _at(spec.buffers[j]), h, nj)
                    out_f32(_at(spec.weights[j]), nj)
                    out_f32(_at(spec.buffers[j]), 2 * nj)
                else:
                    sgd_into(_at(spec.weights[j]), _at(spec.gradients[j]), _at(spec.buffers[j]), h, nj)
                    out_f32(_at(spec.weights[j]), nj)
                    out_f32(_at(spec.buffers[j]), nj)
        curve[ep] = epoch_loss_mean(losses_addr + ep * steps * 8, steps)
        completed_steps += steps
    _ = rows^
    _ = hyper^
    _ = noidx^
    _ = base^
    return epochs * steps
