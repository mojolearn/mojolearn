# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Door-side pins of the clustering families' resident fitted state (UNPACK
and PACK only): the x_cluster `cluster_centers_` (rows N024, N021) and the
SpectralClustering prediction state (N326).

The typed table (generation, binding word, shape words, bytes, LRU tick,
eviction and the dead-key drain) is RUN in
bindings/cluster_resident_table_run.mojo (split from
bindings/cluster_resident_table.mojo, lane resident-pin-tables, 2026-11-12).
This store holds, per slot word (`model key + family`), the minting binding
and the native lease (which pins the fitted objects and owns the device
copy). After each typed call the door takes the released slots
(`cres_run_take_gone`) at once, under the GIL, pops their pins out of the map
and only then retires their leases (a retirement may release the GIL).

Rules (plan section 2A) are listed in the RUN module. Callers hold only
exact mojolearn `Array`s made read-only (`fitted_freeze`). Nothing here
computes on data."""
from std.python import PythonObject
from bindings.native_model_records import model_identity_key, model_generation
from core.shared_slot import SharedSlot
from bindings.kmeans_variants_resident_boundary_unpack import VariantsResidentLease, variants_resident_retire
from bindings.spectral_native_boundary_unpack import SpectralStatePins
from bindings.spectral_resident_boundary_unpack import SpectralResidentLease, spectral_resident_retire
from bindings.cluster_resident_table_run import (
    CRES_X_CLUSTER, CRES_SPECTRAL, CRES_HELD_MAX, CRES_HELD_BYTES, cres_run_forget_slot, cres_run_drain,
    cres_run_has, cres_run_matches, cres_run_touch, cres_run_reserve, cres_run_publish, cres_run_take_gone,
)


struct CresPin(Copyable, Movable):
    # Copyable (all fields are): Dict values must be.
    var binding: PythonObject
    var native_variant: Optional[VariantsResidentLease]
    var native_spectral: Optional[SpectralResidentLease]

    def __init__(out self, binding: PythonObject):
        self.binding = binding
        self.native_variant = None
        self.native_spectral = None


struct CresPins(Defaultable, Movable):
    var pins: Dict[Int, CresPin]

    def __init__(out self):
        self.pins = Dict[Int, CresPin]()


comptime _CRES_PINS = SharedSlot[CresPins, "MojolearnClusterResidentPinsV1", 1]


def _pins() raises -> UnsafePointer[CresPins, MutAnyOrigin]:
    return _CRES_PINS.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _retire(var held: CresPin):
    """The lease's native retirement; quiet."""
    if held.native_spectral:
        spectral_resident_retire(held.native_spectral.take())
        _ = held^
        return
    if held.native_variant:
        # Dropping the table reference cannot retire an active query lease.
        variants_resident_retire(held.native_variant.take())
        _ = held^
        return
    _ = held^


def _settle() raises:
    """Pop the pins of every slot the typed table released (no GIL release
    in between), then retire them."""
    var gone = cres_run_take_gone()
    if len(gone) == 0:
        return
    var out_pins = List[CresPin]()
    var p = _pins()
    for i in range(len(gone)):  # driver-loop: over slots released by this call (at most CRES_HELD_MAX)
        if gone[i] in p[].pins:
            out_pins.append(p[].pins.pop(gone[i]))
    while len(out_pins) > 0:  # driver-loop: over slots released by this call (at most CRES_HELD_MAX)
        _retire(out_pins.pop())


def cres_forget(model: PythonObject, family: Int):
    """Drop `family`'s entry of a live model (its fit, a replaced state)."""
    try:
        cres_run_forget_slot(model_identity_key(model) + family)
        _settle()
    except:
        pass


def _forget_now(slot: Int) raises:
    cres_run_forget_slot(slot)
    _settle()


def _publish(slot: Int, model: PythonObject, binding: PythonObject, var dims: List[Int], var pin: CresPin,
             nbytes: Int) raises:
    var generation = model_generation(model)
    _ = cres_run_publish(slot, generation, dims^, Int(binding._obj_ptr), nbytes)
    # A late entry of the same slot (published by a concurrent store while a
    # retirement released the GIL) is detached here and retired after the
    # new pin is in the map.
    var replaced = Optional[CresPin]()
    var p = _pins()
    if slot in p[].pins:
        replaced = p[].pins.pop(slot)
    p = _pins()
    p[].pins[slot] = pin^
    if replaced:
        _retire(replaced.take())


# Both families retain native Arcs and required boundary pins. They share
# one unchanged bounded LRU and byte accounting mechanism.
def cres_variant_lookup(model: PythonObject, binding: PythonObject, centers: PythonObject,
                        provider: PythonObject, dims: List[Int]) raises -> Optional[VariantsResidentLease]:
    var slot = model_identity_key(model) + CRES_X_CLUSTER
    cres_run_drain()
    _settle()
    if not cres_run_has(slot):
        return None
    var generation = model_generation(model)
    var matched = cres_run_matches(slot, generation, Int(binding._obj_ptr), dims)
    var result = Optional[VariantsResidentLease]()
    var p = _pins()
    if matched and slot in p[].pins:
        # GIL-held metadata only. Copy the Arc and both Python pins before
        # any released-GIL work; never keep a borrowed Dict value across it.
        ref held = p[].pins[slot]
        if held.native_variant:
            ref lease = held.native_variant.value()
            if lease.centers is centers and lease.provider is provider:
                result = held.native_variant.value().copy()
    if result:
        cres_run_touch(slot)
    else:
        _forget_now(slot)
    return result^


def cres_variant_store(model: PythonObject, binding: PythonObject, var dims: List[Int],
                       var lease: VariantsResidentLease, nbytes: Int) raises:
    var slot = model_identity_key(model) + CRES_X_CLUSTER
    cres_run_drain()
    cres_run_reserve(slot, nbytes)
    _settle()
    var pin = CresPin(binding)
    pin.native_variant = lease^
    _publish(slot, model, binding, dims^, pin^, nbytes)


# Spectral shares only the bounded LRU/accounting mechanism. Its native Arc
# retains the minting driver, and its boundary lease pins all five fitted
# objects. No private Python object tuple or callable/handle envelope is made.
def cres_spectral_lookup(model: PythonObject, binding: PythonObject, provider: PythonObject,
                         state: SpectralStatePins, training: PythonObject,
                         dims: List[Int]) raises -> Optional[SpectralResidentLease]:
    var slot = model_identity_key(model) + CRES_SPECTRAL
    cres_run_drain()
    _settle()
    if not cres_run_has(slot):
        return None
    var generation = model_generation(model)
    var matched = cres_run_matches(slot, generation, Int(binding._obj_ptr), dims)
    var result = Optional[SpectralResidentLease]()
    var p = _pins()
    if matched and slot in p[].pins:
        ref held = p[].pins[slot]
        if held.native_spectral:
            ref lease = held.native_spectral.value()
            if lease.provider is provider and lease.training is training and lease.state.same(state):
                result = held.native_spectral.value().copy()
    if result:
        cres_run_touch(slot)
    else:
        _forget_now(slot)
    return result^


def cres_spectral_store(model: PythonObject, binding: PythonObject, var dims: List[Int],
                        var lease: SpectralResidentLease, nbytes: Int) raises:
    var slot = model_identity_key(model) + CRES_SPECTRAL
    cres_run_drain()
    cres_run_reserve(slot, nbytes)
    _settle()
    var pin = CresPin(binding)
    pin.native_spectral = lease^
    _publish(slot, model, binding, dims^, pin^, nbytes)
