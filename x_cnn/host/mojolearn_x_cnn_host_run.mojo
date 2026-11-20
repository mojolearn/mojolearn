# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN bodies of `bindings/_mojolearn_x_cnn_host.mojo` (wave 8 run split).

Typed Mojo only: the GIL-released blocks of the conv block forward and
backward, the softmax cross entropy and the SGD and Adam steps, moved unchanged. The binding reads
the addresses and parameter words (UNPACK), releases the GIL around each call
here exactly where it did before, and returns the count (PACK).
"""

from x_cnn.ops import FP, IP
from x_cnn.host.ops_host import conv_block_forward_into, conv_block_backward_into
from x_cnn.host.ops_host import softmax_xent_into, sgd_into, adam_into
from x_cnn.host.binding_entries_impl import out_f32


def conv_block_forward_run(
    px: FP, pw: FP, pb: FP, po: FP, pi: IP, cprm: List[Int32], pprm: List[Int32], pool: Bool,
    keep: Bool, pk: FP, py_: FP, no: Int,
) raises:
    """The GIL-released block of `_block_forward`."""
    conv_block_forward_into(px, pw, pb, po, pi, cprm, pprm, pool, keep, pk, py_)
    out_f32(po, no)


def conv_block_backward_run(
    px: FP, pw: FP, pb: FP, pg: FP, pi: IP, pdx: FP, pdw: FP, pdb: FP,
    cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    keep: Bool, pk: FP, py_: FP, nx: Int, OC: Int, ckk: Int,
) raises:
    """The GIL-released block of `conv_block_backward_binding`."""
    conv_block_backward_into(px, pw, pb, pg, pi, pdx, pdw, pdb, cprm, pprm, pool, need_dx, keep, pk, py_)
    if need_dx:
        out_f32(pdx, nx)
    out_f32(pdw, OC * ckk)
    out_f32(pdb, OC)


def sgd_list_run(qws: List[FP], qgs: List[FP], qvs: List[FP], h: List[Float32], cnt: List[Int]):
    """The GIL-released block of `sgd_binding`'s list form."""
    for j in range(len(cnt)):  # driver-loop: over the model's parameters
        sgd_into(qws[j], qgs[j], qvs[j], h, cnt[j])
        out_f32(qws[j], cnt[j])
        out_f32(qvs[j], cnt[j])


def sgd_run(pw: FP, pg: FP, pv: FP, h: List[Float32], n: Int):
    """The GIL-released block of `sgd_binding`'s single-buffer form."""
    sgd_into(pw, pg, pv, h, n)
    out_f32(pw, n)
    out_f32(pv, n)


def adam_list_run(qws: List[FP], qgs: List[FP], qms: List[FP], h: List[Float32], cnt: List[Int]):
    """The GIL-released block of `adam_binding`'s list form."""
    for j in range(len(cnt)):  # driver-loop: over the model's parameters
        adam_into(qws[j], qgs[j], qms[j], h, cnt[j])
        out_f32(qws[j], cnt[j])
        out_f32(qms[j], 2 * cnt[j])


def adam_run(pw: FP, pg: FP, pm: FP, h: List[Float32], n: Int):
    """The GIL-released block of `adam_binding`'s single-buffer form."""
    adam_into(pw, pg, pm, h, n)
    out_f32(pw, n)
    out_f32(pm, 2 * n)


def softmax_xent_run(pl: FP, py_: IP, pg: FP, pp: FP, n: Int, k: Int) -> Float32:
    """The GIL-released block of `softmax_xent_binding`; returns the mean loss."""
    var loss = softmax_xent_into(pl, py_, pg, pp, n, k)
    out_f32(pg, n * k)
    out_f32(pp, n * k)
    return loss
