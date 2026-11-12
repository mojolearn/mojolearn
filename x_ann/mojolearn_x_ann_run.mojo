# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN bodies of `bindings/_mojolearn_x_ann.mojo` (wave 8 run split).

Typed Mojo only: the GIL-released blocks of the IVF-PQ builds, moved
unchanged. The binding reads the addresses and parameters (UNPACK), releases
the GIL around each call here exactly where it did before, and registers the
resident handle and builds the returned value (PACK).
"""

from x_ann.abi import out_f32_to, out_i32_to
from x_ann.ivf_pq_core import IvfPqIndex
from x_ann.stage_timer import AnnStages
from x_ann.ivf_pq_device import IvfPqDevice, ivf_pq_build_device, ivf_pq_build_device_resident


def ivf_pq_build_run(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int, codes_addr: Int, x_addr: Int,
    mut bst: AnnStages, centers_out: Int, offsets_out: Int, indices_out: Int,
    codebooks_out: Int, codes_out: Int,
) raises:
    """The GIL-released block of `ivf_pq_build_binding`."""
    var index = ivf_pq_build_device(
        x, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, codes_addr, x_addr
    )
    bst.host("build")
    out_f32_to(index.centers, centers_out)
    out_i32_to(index.offsets, offsets_out)
    out_i32_to(index.list_indices, indices_out)
    out_f32_to(index.codebooks, codebooks_out)
    out_i32_to(index.codes, codes_out)
    bst.host("copy_out")


def ivf_pq_build_resident_run(
    x: List[Float32], x_addr: Int, n: Int, dim: Int, n_lists: Int, iters: Int,
    seed: Int, pq_dim: Int, pq_bits: Int, pq_iters: Int, mut dev: IvfPqDevice,
    mut index: IvfPqIndex, mut bst: AnnStages, centers_out: Int,
    offsets_out: Int, indices_out: Int, codebooks_out: Int,
) raises:
    """The GIL-released block of `ivf_pq_build_resident_binding`."""
    index = ivf_pq_build_device_resident(
        x, x_addr, n, dim, n_lists, iters, seed, pq_dim, pq_bits, pq_iters, dev
    )
    bst.host("build")
    out_f32_to(index.centers, centers_out)
    out_i32_to(index.offsets, offsets_out)
    out_i32_to(index.list_indices, indices_out)
    out_f32_to(index.codebooks, codebooks_out)
    bst.host("copy_out")
