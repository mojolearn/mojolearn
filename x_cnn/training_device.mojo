# SPDX-License-Identifier: Apache-2.0
"""Compiled GPU CNN multi-epoch training; no Python runtime dependency."""
from bindings.hostptr import f64_ptr
from x_cnn.training import TrainingSpec, training_steps, validate_spec, optimizer_base, epoch_loss_mean
from x_cnn.ops import FP, IP, AH_ROW, epoch_key
from x_cnn.device import (
    res_alloc, res_free, res_upload, res_download, res_gather_pair_perm,
    adam_hyper_resident, conv_block_forward_into, linear_forward_into,
    softmax_xent_res_loss, linear_backward_into, conv_block_backward_into,
    opt_many_resident_h,
)


def _at(address: Int) -> FP:
    return FP(unsafe_from_address=address)


def _at_i(address: Int) -> IP:
    return IP(unsafe_from_address=address)


def fit_epochs(
    spec: TrainingSpec, losses_addr: Int, curve_addr: Int,
    n: Int, batch: Int, adam: Bool, done: Int, first_epoch: Int,
    epochs: Int, shuffle: Bool, seed: UInt64, fparams: List[Float64],
) raises -> Int:
    """One native call owns every epoch, minibatch and optimizer launch.

    Caller supplies float64 losses[epochs*steps], curve[epochs], and live
    resident handles. Preserve epoch_key(seed,epoch), the first-SGD-step flag,
    Adam's absolute step numbers, kernel ordering and the final short batch.
    max_iter remains an exact epoch count; no new early stopping criterion.
    """
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
    var l32 = List[Float32](length=steps, fill=Float32(0))
    var hbuf = res_alloc(steps * AH_ROW if adam else 12)
    var lbuf = 0
    try:
        lbuf = res_alloc(steps)
        if not adam:
            res_upload(hbuf, base.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 12)
        for ep in range(epochs):
            var key = epoch_key(seed, first_epoch + ep)
            if adam:
                adam_hyper_resident(base, completed_steps + 1, steps, hbuf)
            for st in range(steps):
                var s0 = st * batch
                var m = min(batch, n - s0)
                var q = 0 if m == batch else 1
                res_gather_pair_perm(spec.arrays[0], spec.data[0], spec.data[2], spec.arrays[1], spec.data[1], 1, n, s0, m, shuffle, key)
                var src = spec.arrays[0]
                for j in range(nb):
                    conv_block_forward_into[True](
                        _at(src), _at(spec.blocks[j][0]), _at(spec.blocks[j][1]), spec.conv[q][j], spec.pool[q][j], spec.pooled[q][j], _at(spec.blocks[j][4]),
                        _at_i(spec.blocks[j][5]), spec.blocks[j][7], spec.blocks[j][8],
                    )
                    src = spec.blocks[j][4]
                linear_forward_into[True](_at(src), _at(spec.head[0]), _at(spec.head[1]), m, flat, k, _at(spec.arrays[2]))
                softmax_xent_res_loss(_at(spec.arrays[2]), _at_i(spec.arrays[1]), m, k, _at(spec.arrays[3]), _at(spec.arrays[4]), lbuf, st)
                var glast = spec.blocks[nb - 1][6] if nb > 0 else spec.arrays[5]
                linear_backward_into[True](
                    _at(src), _at(spec.head[0]), _at(spec.arrays[3]), m, flat, k, _at(glast), _at(spec.head[2]), _at(spec.head[3])
                )
                for jj in range(nb):
                    var j = nb - 1 - jj
                    var bsrc = spec.blocks[j - 1][4] if j > 0 else spec.arrays[0]
                    var dx = spec.blocks[j - 1][6] if j > 0 else 0
                    var want = dx != 0
                    conv_block_backward_into[True](
                        _at(bsrc), _at(spec.blocks[j][0]), _at(spec.blocks[j][1]), _at(spec.blocks[j][6]), _at_i(spec.blocks[j][5]), spec.conv[q][j], spec.pool[q][j],
                        spec.pooled[q][j], want, _at(dx) if want else _at(spec.blocks[j][3]), _at(spec.blocks[j][2]), _at(spec.blocks[j][3]), spec.blocks[j][7],
                        spec.blocks[j][8],
                    )
                if adam:
                    opt_many_resident_h[True](spec.weights, spec.gradients, spec.buffers, spec.sizes, hbuf + 4 * AH_ROW * st)
                else:
                    opt_many_resident_h[False](spec.weights, spec.gradients, spec.buffers, spec.sizes, hbuf + (0 if completed_steps + st == 0 else 24))
            res_download(lbuf, l32.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), steps)
            for st in range(steps):
                losses[ep * steps + st] = Float64(l32[st])
            curve[ep] = epoch_loss_mean(losses_addr + ep * steps * 8, steps)
            completed_steps += steps
    except e:
        if lbuf != 0:
            res_free(lbuf)
        res_free(hbuf)
        raise e
    res_free(lbuf)
    res_free(hbuf)
    _ = base^
    _ = l32^
    return epochs * steps
