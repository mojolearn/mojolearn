#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A registry the lane map stops at still LOADS what its region loads
(`lane_select.sink_region_loads`, 2026-11-05).

`_native_runtime_manifold_bridge.py` names four bindings, so the enumerator
count made it a sink, and the binding its `connect` loads by a literal
(`_mojolearn_runtime_manifold_host`, whose `manifold_native_call` runs every
UMAP fit) reached no lane: the umap/ tree mapped to no lane at all. Fast:
synthetic texts are seeded into the selector's caches and two real files are
parsed; nothing here derives the lane map (tools/test_lane_select.py holds the
real-tree umap attribution).

    python3 -m pytest tools/test_lane_select_sink_loads.py -q
"""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_select as ls                                        # noqa: E402

HOST = "_mojolearn_runtime_manifold_host"
SINK = os.path.join(ls.PKG, "_synthetic_sink_bridge.py")


class _Seeded:
    """Seed one synthetic package file into the selector's caches."""

    def __init__(self, text):
        self.text = textwrap.dedent(text)

    def __enter__(self):
        ls._read.cache[SINK] = self.text
        ls._parse.cache.pop(SINK, None)
        for cache in (ls._REGIONS, ls._SINK_LOADS):
            for key in [k for k in cache if k[0] == SINK]:
                del cache[key]
        return self

    def __exit__(self, *exc):
        ls._read.cache.pop(SINK, None)
        ls._parse.cache.pop(SINK, None)
        for cache in (ls._REGIONS, ls._SINK_LOADS):
            for key in [k for k in cache if k[0] == SINK]:
                del cache[key]
        return False


BRIDGE = f"""
    def connect():
        from . import _backend
        module = _backend.load_host_module("{HOST}")
        return module.manifold_native_call

    def other():
        return 1
"""


def test_a_literal_load_in_the_region_is_the_lanes_binding():
    with _Seeded(BRIDGE):
        loads, names = ls.sink_region_loads(SINK, {"connect"})
    assert HOST in loads, loads
    assert "manifold_native_call" in names, names


def test_the_whole_file_loads_it_too():
    with _Seeded(BRIDGE):
        loads, _names = ls.sink_region_loads(SINK, None)
    assert HOST in loads


def test_a_function_the_lane_does_not_run_loads_nothing_for_it():
    # The region rule the map already keeps (`_region_nodes`): `connect` is
    # not named, so its load does not run for this lane.
    with _Seeded(BRIDGE):
        loads, names = ls.sink_region_loads(SINK, {"other"})
    assert not loads and not names, (loads, names)


def test_an_import_time_call_of_the_loader_restores_the_load():
    # THE ARM THAT RESTORES THE REACH: the same lane (asking for `other`)
    # takes the binding once the module's own top level calls `connect`.
    # dedent first: an unindented line appended to the indented text leaves
    # nothing common to strip, and the seeded file would not parse
    with _Seeded(textwrap.dedent(BRIDGE) + "\n_ENTRY = connect()\n"):
        loads, names = ls.sink_region_loads(SINK, {"other"})
    assert HOST in loads and "manifold_native_call" in names, (loads, names)


def test_a_table_or_variable_argument_is_still_not_evidence():
    # The registry part stays out: what a variable or a table names is a list,
    # not a load this region performs.
    text = f"""
        _TABLE = {{"m": "{HOST}"}}
        def connect(name):
            from . import _backend
            a = _backend.load_host_module(_TABLE["m"])
            b = _backend.binding(name, "identical")
            return a, b
    """
    with _Seeded(text):
        loads, names = ls.sink_region_loads(SINK, {"connect"})
    assert not loads and not names, (loads, names)


def test_only_the_loaded_bindings_exports_are_names():
    text = f"""
        def connect():
            from . import _backend
            module = _backend.load_host_module("{HOST}")
            return (module.manifold_native_call, module.not_an_export_of_it, "kmeans_fit",
                    getattr(module, "runtime_manifold_host_vendor"))
    """
    with _Seeded(text):
        _loads, names = ls.sink_region_loads(SINK, {"connect"})
    # An attribute read of the loaded module is a name; a string (a registry's
    # getattr table) is not, even when it spells an export of that binding.
    assert names == {"manifold_native_call"}, names


def test_the_real_manifold_bridge_loads_the_manifold_owner():
    rel = os.path.join(ls.PKG, "_native_runtime_manifold_bridge.py")
    assert rel in ls.enumerator_files(), \
        "the bridge is no longer a registry; this case no longer exercises the sink rule"
    loads, names = ls.sink_region_loads(rel, {"connect"})
    assert HOST in loads, loads
    assert "manifold_native_call" in names, names


def test_a_sink_that_is_a_lanes_seed_is_followed():
    """`neighbors.py` defines the k-NN classes the knn lanes call and is a
    sink by the enumerator cap; as a seed its imports are walked."""
    door = os.path.join(ls.PKG, "neighbors.py")
    init = os.path.join(ls.PKG, "__init__.py")
    sinks = ls.enumerator_files() | ls._reexport_registries()
    assert door in sinks, "neighbors.py is no longer a sink; this case no longer exercises the rule"
    follow = ls._followed_seeds({door, init})
    assert door in follow and init not in follow, follow
    imported = os.path.join(ls.PKG, "_native_neighbor_queries.py")
    assert imported in ls._region_closure([door], sinks, follow=follow)
    # THE ARM THAT MUST FAIL: entered only, as before, the import is lost.
    assert imported not in ls._region_closure([door], sinks)


def test_the_package_init_is_still_entered_only():
    init = os.path.join(ls.PKG, "__init__.py")
    sinks = ls.enumerator_files() | ls._reexport_registries()
    assert ls._region_closure([init], sinks, follow=ls._followed_seeds({init})) \
        == ls._region_closure([init], sinks)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
