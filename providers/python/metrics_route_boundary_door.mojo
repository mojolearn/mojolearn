# SPDX-License-Identifier: Apache-2.0
"""The CPU metrics program entry of a `device="cpu"` estimator's route: the ONE
metrics route of every family (`device_route_metrics_cells`,
`device_route_metrics_cpu`; lane dw-core round 2).

`score` of Ridge and LinearRegression runs its R^2 program on a metrics
table. A `device="cpu"` estimator in a GPU installation resolves that table
here, ONCE, beside its fit and predict entries
(providers/python/linear_route_boundary_door.mojo): the CPU identical image
`_mojolearn_x_metrics_host`, connected at import under the CPU entry family
(python/mojolearn/_native_classical_bridge.py connect_cpu_device, key
`metrics`). Every later `score` calls through the route's `fifth` entry (core/linear_route.mojo names the slots);
no call reads the process's metrics table, which is the GPU one.

A CPU metrics image that is not built is not refused when the estimator
first fits or predicts: the route holds an entry that says so, in words, to
the `score` that needs it.

Resolution is a metadata operation with no independent numerical work to
split; it runs at an estimator's first operation, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type
from core.entry_table import EntryRef, entry_ref_word
from core.linear_entry_table import LINEAR_TABLE_METRICS, linear_table_entry
from core.device_route import DeviceRoute
from providers.python.device_route_boundary_door import device_cpu_connect_late
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from providers.python.linear_entry_boundary_door import linear_entry_connection
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT


def _metrics_cpu_absent(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("ImportError: score of a device='cpu' estimator needs the CPU metrics binary "
        + "(_mojolearn_x_metrics_host), which is not connected in this installation"), context, sink)
    return 1


def device_route_metrics_cells(mut route: DeviceRoute, cells: PythonObject, b: PythonObject) raises:
    """Give a CPU route resolved in a GPU installation its CPU metrics entry
    (`fifth`), for ANY family: `cells` is the CPU metrics connection
    (`_native_table_cells(_mojolearn_x_metrics_host, "x_metrics")`, connected
    at import), or the text of why it is not connected. The route's owner
    pins `cells` (and with it the metrics connection owner and its image);
    `b` is the builtins module the calling door already admitted. The score
    of the family then calls `route.fifth` when `route.has_fifth`, and reads
    the process's metrics table (the GPU one) only when it is False."""
    route.has_fifth = True
    if py_type(cells) is not b.dict:
        route.fifth = EntryRef(_metrics_cpu_absent, 0)
        return
    var connection = linear_entry_connection(cells, LINEAR_TABLE_METRICS)
    if connection.key.vendor != 0 or connection.key.mode != 1:
        raise Error("RuntimeError: device='cpu' score requires the CPU identical metrics binding")
    var entry = linear_table_entry[LINEAR_TABLE_METRICS](connection.key, 0, String(METRICS_PROGRAM_ABI),
        MP_OPERATION_COUNT)
    route.fifth = EntryRef(entry, entry_ref_word(connection.key.mode, connection.key.vendor, MP_OPERATION_COUNT))
    _ = connection


def device_route_metrics_cpu(mut route: DeviceRoute, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """`device_route_metrics_cells` from the shared family context: the CPU
    metrics connection every family shares (`context["metrics_cpu_cells"]`,
    python/mojolearn/_native_family_bridge.py `shared_context`). Returns the
    cells: the caller stores them among its owner's `images` pins."""
    var cells = py_get(context, "metrics_cpu_cells")
    if cells is None and device_cpu_connect_late(context):
        # A restored estimator (no construction ran the device admission):
        # its route miss admits the CPU device once (lane fx-leftovers item 3).
        cells = py_get(context, "metrics_cpu_cells")
    device_route_metrics_cells(route, cells, b)
    return cells


def linear_route_metrics_cpu(mut route: DeviceRoute, entries: PythonObject, b: PythonObject) raises:
    """Ridge and LinearRegression: the cells of the CPU entry family the
    route was resolved from (the route owner pins `entries`)."""
    device_route_metrics_cells(route, py_get(entries, "metrics"), b)
