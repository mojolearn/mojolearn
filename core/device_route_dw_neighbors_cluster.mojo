# SPDX-License-Identifier: Apache-2.0
"""The device route of one estimator whose family selects compiled BINDINGS
(extension images) by name: mixture, cluster, neighbors, density.

The pilot's route (core/linear_route.mojo) holds thin C entries of one family
table. These families reach their kernels through a typed provider of a named
binding image (`_mojolearn_mixture`, `_mojolearn_x_cluster`, ...), so their
route holds, per binding slot, WHICH image answers: the device image of the
selected tier, or the CPU identical image of the same name
(`_mojolearn_<name>_host`). The device rule and its words are the pilot's and
are imported, not copied: `device=None` is this installation's device, an
explicit `'gpu'` without a GPU set is refused, `'cpu'` always runs.

This file is the rule as native words (no Python): what a request resolves
to, and the one refusal this family form adds. The owner that holds the
resolved images per estimator is
providers/python/device_route_dw_neighbors_cluster_door.mojo.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.

Lane dw-neighbors-cluster (2026-11-20). SHARED CHANGE WANTED: one
`core/device_route.mojo` for every family of this form.
"""
from core.linear_route import LINEAR_DEVICE_CPU, LINEAR_DEVICE_INSTALL, linear_device_installed

# Binding slots of a route owner: one per binding name a family of this lane
# selects. A slot is an index, so a call compares no name.
comptime FAMILY_ROUTE_MIXTURE = 0
comptime FAMILY_ROUTE_X_CLUSTER = 1
comptime FAMILY_ROUTE_X_NEIGHBORS = 2
comptime FAMILY_ROUTE_BASE = 3
comptime FAMILY_ROUTE_ESTIMATORS = 4
comptime FAMILY_ROUTE_HDBSCAN = 5
comptime FAMILY_ROUTE_SLOT_COUNT = 6


def family_route_binding_name(slot: Int) -> String:
    """The binding name of a slot (resolution and messages only)."""
    if slot == FAMILY_ROUTE_MIXTURE:
        return String("_mojolearn_mixture")
    if slot == FAMILY_ROUTE_X_CLUSTER:
        return String("_mojolearn_x_cluster")
    if slot == FAMILY_ROUTE_X_NEIGHBORS:
        return String("_mojolearn_x_neighbors")
    if slot == FAMILY_ROUTE_BASE:
        return String("_mojolearn")
    if slot == FAMILY_ROUTE_ESTIMATORS:
        return String("_mojolearn_estimators")
    if slot == FAMILY_ROUTE_HDBSCAN:
        return String("_mojolearn_hdbscan")
    return String("")


def family_route_cpu_fast_refusal() raises:
    """`device='cpu'` beside a GPU set runs the CPU identical images only
    (the pilot's words, providers/python/linear_route_boundary_door.mojo)."""
    raise Error("ValueError: device='cpu' runs in identical mode only; numeric_mode='fast' is for Apple GPUs")


def family_route_cpu_unconnected(name: String) raises:
    """A GPU installation whose CPU image of `name` was not connected at
    import (the pilot's words, with the binding named)."""
    raise Error("ImportError: device='cpu' is not connected in this installation (" + name + ")")


@fieldwise_init
struct FamilyDeviceRoute(Copyable, Movable):
    """What one estimator's request resolved to.

    `requested` is the constructor value's code (LINEAR_DEVICE_GPU, _CPU or
    _INSTALL). `device` is the device it runs on (never _INSTALL).
    `host_table` is True when the estimator runs the CPU images BESIDE a GPU
    set (an explicit `'cpu'` on a GPU installation): its slots resolve from
    the CPU connection table. In a CPU-only process the process's own table
    already is the CPU table and `host_table` is False. `mode` is 1 for
    identical, 0 for fast."""
    var requested: Int
    var device: Int
    var host_table: Bool
    var mode: Int


def family_route_unresolved() -> FamilyDeviceRoute:
    return FamilyDeviceRoute(LINEAR_DEVICE_INSTALL, -1, False, -1)


def family_route_plan(requested: Int, process_cpu_only: Bool, mode: Int) raises -> FamilyDeviceRoute:
    """Resolve a request in a process that is CPU only or holds a GPU set.

    None is the installation's device; an explicit `'gpu'` in a CPU-only
    process is refused in the pilot's words (`linear_device_installed`);
    `'cpu'` beside a GPU set in fast mode is refused."""
    var device = linear_device_installed(requested, 0 if process_cpu_only else 1)
    var host_table = device == LINEAR_DEVICE_CPU and not process_cpu_only
    if host_table and mode != 1:
        family_route_cpu_fast_refusal()
    return FamilyDeviceRoute(requested, device, host_table, mode)


def family_route_vendor_is_cpu(route: FamilyDeviceRoute) -> Bool:
    """True when the route's images are CPU images (`vendor_used` answers
    `cpu` without asking an image)."""
    return route.device == LINEAR_DEVICE_CPU
