# SPDX-License-Identifier: Apache-2.0
"""The x_trees DART ABI shared by its two publishers and its native callers
(lane calltable-gbdt-forest-dart, 2026-11-11).

One `abi("C")` entry per x_trees image runs every DART operation below
(`xtrees/dart_runs_device.mojo`, `xtrees/dart_runs_host.mojo`; table in
`core/trees_entry_table_dart.mojo`). A call has the descriptor shape of
`core/native_descriptor_abi.mojo` and the parameter convention of the glue
(`xtrees/glue_abi.mojo::XtCall`): addresses, then the positional parameters
as an int view and a float view of equal length. The positions are those of
the former Python entries (`x_trees_dart_*`):

  XD_CAPS            no arguments; returns bit 0 when the image serves the
                     round (open/step/add/close), bit 1 when it keeps a
                     resident forest (forest_prepare/predict/release).
  XD_OPEN            addrs [x, y, inits]; params [n, d, k, kind, n_iterations,
                     node_cap]; returns the session handle.
  XD_STEP            addrs [coef, thr, flags, bad, target_0, ...]; params
                     [handle, t, drop_seed, iteration, skip_thr]; returns t.
  XD_ADD             addrs [colid, quesval, left, values, rows]; params
                     [handle, tree, class, lo, n_nodes, shrink (f), factor (f),
                     reg_lambda (f), reg_alpha (f), max_delta_step (f), m];
                     returns n_nodes.
  XD_CLOSE           addrs [bad]; params [handle].
  XD_PREDICT         addrs [x, dst, colids, quesvals, lefts, leaf_values, sizes,
                     coefs, inits]; params [n, d, k, n_trees]. colids ...
                     sizes are n_trees Int words each, coefs n_trees Float64,
                     inits k Float64. Returns n.
  XD_FOREST_PREPARE  addrs [colids, quesvals, lefts, leaf_values, sizes, coefs,
                     inits, out2]; params [d, k, n_trees]; writes [handle,
                     device bytes] to the two Int words at out2; returns the
                     handle.
  XD_FOREST_PREDICT  addrs [x, dst]; params [handle, n, d, k]; returns n.
  XD_FOREST_RELEASE  params [handle].

The caller owns every buffer until the synchronous call returns. Nothing is
retained besides the image's own session and forest registries; no Python
object crosses.
"""
from xtrees.glue_abi import XtCall

comptime XD_DART_ABI = "mojolearn.trees.x_trees.dart.v1"

comptime XD_CAPS = 0
comptime XD_OPEN = 1
comptime XD_STEP = 2
comptime XD_ADD = 3
comptime XD_CLOSE = 4
comptime XD_PREDICT = 5
comptime XD_FOREST_PREPARE = 6
comptime XD_FOREST_PREDICT = 7
comptime XD_FOREST_RELEASE = 8
comptime XD_OP_COUNT = 9

comptime XD_CAP_ROUND = 1
comptime XD_CAP_RESIDENT = 2


def xd_need(c: XtCall, n_addrs: Int, n_params: Int, who: String) raises:
    """The former Python entries' arity refusals, in their words."""
    if c.n_addrs < n_addrs:
        raise Error(who + ": needs " + String(n_addrs) + " buffer addresses")
    if c.n_params != n_params:
        raise Error(who + ": params must hold " + String(n_params) + " values")


def xd_addr(c: XtCall, k: Int) -> Int:
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=c.addrs)[k]


def xd_int(c: XtCall, k: Int) raises -> Int:
    if c.bad and (MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=c.bad)[k] & 1):
        raise Error("x_trees dart: parameter " + String(k) + " is not an integer")
    return MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=c.ints)[k]


def xd_float(c: XtCall, k: Int) raises -> Float64:
    if c.bad and (MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=c.bad)[k] & 2):
        raise Error("x_trees dart: parameter " + String(k) + " is not a number")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=c.floats)[k]


def xd_words(address: Int, count: Int, who: String) raises -> List[Int]:
    """count Int words at address (an address list of the forest)."""
    if count < 0 or (count > 0 and address == 0):
        raise Error(who + ": null or invalid address list")
    var src = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Int](capacity=count)
    for j in range(count):  # driver-loop: over the T trees' address words (model structure)
        out.append(src[j])
    return out^


def xd_f64(address: Int, count: Int, who: String) raises -> List[Float64]:
    """count Float64 words at address (the per-tree coefficients or the class starts)."""
    if count < 0 or (count > 0 and address == 0):
        raise Error(who + ": null or invalid value list")
    var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)
    var out = List[Float64](capacity=count)
    for j in range(count):  # driver-loop: over the T tree coefficients or k class starts (model structure)
        out.append(src[j])
    return out^
