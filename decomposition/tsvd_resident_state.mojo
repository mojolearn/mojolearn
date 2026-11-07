# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident fitted components of TruncatedSVD (lane
ne1-a-decomp-resident, workstream A of
docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md; row N351).

`TruncatedSVD.transform` and `inverse_transform` uploaded the fitted
`components_` (n_components x n_features) with X on every call
(`tsvd_transform_host`, `inverse_transform_host`). Here the components are
uploaded once per fitted state into the estimators binding's store
(core/fitted_resident_store.mojo, lane ne1-a-gp-svm: lease-keyed, byte cap
of an eighth of device memory, least recently used eviction) and every later
call uploads only X:

    slot 0  components_     transform and inverse_transform

The store's key is a lease the public door (bindings/tsvd_native.mojo
through bindings/fitted_resident_door.mojo) takes once per fitted state and
keeps in the model's prepared-owner record with the model's generation word
and the components object; `TsvdResidentLease` is the native object that
holds it, and its destruction (a refit, a replaced `components_`, the
model's death) frees the copy.

WHAT DOES NOT CHANGE. The resident calls run `tsvd_transform_dev` and
`inverse_transform_dev` (decomposition/estimator.mojo), the bodies the
per-call routes run after their own upload, over a copy of the same bytes:
the same launches in the same order, so the bits are the same, in both
numeric tiers.
"""
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.fitted_resident_store import (
    FittedResidentStore,
    fitted_lease_open,
    fitted_lease_close,
    fitted_resident_view,
    fitted_resident_stats,
)
from decomposition.estimator import tsvd_transform_dev, inverse_transform_dev

# One store per numeric tier (a FAST and an IDENTICAL binding never share
# device memory), as gaussian_process/gp_resident_store.mojo.
comptime TSVD_RESIDENT_STORE = _Global[
    StorageType=FittedResidentStore,
    name=(
        "MojoTsvdResidentFittedIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        else "MojoTsvdResidentFittedFast"
    ),
    init_fn=FittedResidentStore.__init__,
]


def tsvd_resident_store() raises -> UnsafePointer[FittedResidentStore, MutAnyOrigin]:
    return TSVD_RESIDENT_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct TsvdResidentLease(Movable, Writable):
    """The native owner of one fitted state's lease: dropping it frees the
    device copy (a refit, a replaced `components_`, the model's death)."""

    var lease: Int

    def __init__(out self, lease: Int):
        self.lease = lease

    def write_to(self, mut writer: Some[Writer]):
        writer.write("TsvdResidentLease(open=", self.lease != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.lease != 0:
            try:
                fitted_lease_close(tsvd_resident_store(), self.lease)
            except:
                pass


def tsvd_resident_lease_open() raises -> Int:
    return fitted_lease_open(tsvd_resident_store())


def tsvd_resident_stats_words() raises -> List[Int]:
    return fitted_resident_stats(tsvd_resident_store())


def _check_dims(n_rows: Int, n_features: Int, n_components: Int, who: String) raises:
    if n_rows <= 0 or n_features <= 0 or n_components <= 0:
        raise Error(who + ": n_rows, n_features and n_components must be positive")


def tsvd_transform_resident_device(
    ctx: DeviceContext,
    lease: Int,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_addr: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    """`tsvd_transform_host` with `components_` read from the lease's device
    copy (uploaded from `components_addr` at the first call of the lease,
    and again only when the address or the size changed). Lease 0 uploads
    it into a buffer that dies with this call, the per-call route."""
    _check_dims(n_rows, n_features, n_components, "tsvd_transform_resident")
    var view = fitted_resident_view(tsvd_resident_store(), ctx, lease, 0, [components_addr],
                                    [n_components * n_features])
    var components = view.bufs[0].copy()
    tsvd_transform_dev(ctx, x_ptr, components, out_ptr, n_rows, n_features, n_components)
    _ = components^
    _ = view^


def tsvd_inverse_resident_device(
    ctx: DeviceContext,
    lease: Int,
    scores_ptr: MutPointer[Float32, MutUntrackedOrigin],
    components_addr: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_components: Int,
) raises:
    """`inverse_transform_host` (no mean: TruncatedSVD) with `components_`
    read from the lease's device copy, the same slot `transform` reads."""
    _check_dims(n_rows, n_features, n_components, "tsvd_inverse_transform_resident")
    var view = fitted_resident_view(tsvd_resident_store(), ctx, lease, 0, [components_addr],
                                    [n_components * n_features])
    var components = view.bufs[0].copy()
    # The mean is not read without add_mean; the scores' address is a
    # valid float32 address, as TruncatedSVD passed its components before.
    inverse_transform_dev(ctx, scores_ptr, components, scores_ptr, out_ptr, n_rows, n_features, n_components, False)
    _ = components^
    _ = view^
