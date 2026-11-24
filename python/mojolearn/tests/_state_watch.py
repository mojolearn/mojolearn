# SPDX-License-Identifier: Apache-2.0
"""Opt-in watcher for process-wide state that one test leaves behind for the next.

Testing tool only (never product execution). A whole-suite run had 271 failures that pass when their file runs
alone (lane w3-fix-f, 2026-11-23): state shared inside one xdist worker. This names the test that changed it.

    MOJOLEARN_TEST_STATE_WATCH=<dir> python -m pytest ...

Each test process writes `<dir>/state-watch-<worker>-<pid>.jsonl`: one `baseline` line, one `collection` line
(what importing the collected test modules changed after the suite's `load_all`), then one line per test that left
a change behind (after its teardown, so a monkeypatch that restored its attribute is silent):

  modules_added / modules_removed / modules_replaced   package modules (an added one ran a class-connection pass)
  classes_changed   {"class", "added", "removed", "replaced"}: own `__dict__` names of a package class
  module_exports_changed   {"module", "added", "removed", "replaced"}: callables/types/modules a package module holds
  threads           alive non-main threads after the test (every later audit_native_call then fails)
  env_changed       environment keys added, removed or changed
  meta_path_changed, profile, trace   import finders, sys.setprofile / sys.settrace left installed

Unset, nothing is imported or recorded. The worker count is unchanged; each process watches only itself.
"""
import json
import os
import sys
import threading
from types import ModuleType

_PKG = "mojolearn"
_CLASS_DICT = type.__dict__["__dict__"].__get__
_MODULE_DICT = ModuleType.__dict__["__dict__"].__get__


def _product(name):
    return (name == _PKG or name.startswith(_PKG + ".")) and not name.startswith(_PKG + ".tests")


def _snapshot():
    modules = {}
    classes = {}
    exports = {}
    for name, module in tuple(sys.modules.items()):
        if type(module) is not ModuleType or not _product(name):
            continue
        modules[name] = id(module)
        fields = _MODULE_DICT(module)
        held = {}
        for key, value in tuple(fields.items()):
            if isinstance(value, type):
                owner = _CLASS_DICT(value).get("__module__")
                if type(owner) is str and _product(owner):
                    qualname = _CLASS_DICT(value).get("__qualname__", getattr(value, "__name__", "?"))
                    label = f"{owner}.{qualname}" if type(qualname) is str else owner
                    classes[id(value)] = (label, value, {k: id(v) for k, v in tuple(_CLASS_DICT(value).items())})
            if callable(value) or type(value) is ModuleType:
                held[key] = id(value)
        exports[name] = held
    main = threading.main_thread()
    alive = sorted(t.name for t in threading.enumerate() if t is not main and t.is_alive())
    return dict(modules=modules, classes=classes, exports=exports, threads=alive, env=dict(os.environ),
                meta_path=[f"{type(f).__module__}.{type(f).__qualname__}@{id(f)}" for f in sys.meta_path],
                profile=repr(sys.getprofile()) if sys.getprofile() is not None else None,
                trace=repr(sys.gettrace()) if sys.gettrace() is not None else None)


def _keyed_diff(before, after):
    added = sorted(str(k) for k in after.keys() - before.keys())
    removed = sorted(str(k) for k in before.keys() - after.keys())
    replaced = sorted(str(k) for k in after.keys() & before.keys() if after[k] != before[k])
    return added, removed, replaced


def _diff(old, new):
    out = {}
    added, removed, replaced = _keyed_diff(old["modules"], new["modules"])
    if added:
        out["modules_added"] = added
    if removed:
        out["modules_removed"] = removed
    if replaced:
        out["modules_replaced"] = replaced
    changed = []
    for key, (label, _cls, fields) in new["classes"].items():
        prior = old["classes"].get(key)
        if prior is None or prior[1] is not _cls:
            continue  # a class first seen now (its module was just imported)
        a, r, p = _keyed_diff(prior[2], fields)
        if a or r or p:
            changed.append(dict({"class": label}, added=a, removed=r, replaced=p))
    if changed:
        out["classes_changed"] = changed
    moved = []
    for name, held in new["exports"].items():
        prior = old["exports"].get(name)
        if prior is None:
            continue
        a, r, p = _keyed_diff(prior, held)
        if a or r or p:
            moved.append(dict(module=name, added=a, removed=r, replaced=p))
    if moved:
        out["module_exports_changed"] = moved
    if new["threads"] != old["threads"]:
        out["threads"] = new["threads"]
    a, r, p = _keyed_diff(old["env"], new["env"])
    if a or r or p:
        out["env_changed"] = dict(added=a, removed=r, changed=p)
    if new["meta_path"] != old["meta_path"]:
        out["meta_path_changed"] = new["meta_path"]
    for key in ("profile", "trace"):
        if new[key] != old[key]:
            out[key] = new[key]
    return out


class StateWatch:
    """The pytest plugin: registered by conftest when MOJOLEARN_TEST_STATE_WATCH names a directory."""

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        self._path = os.path.join(directory, f"state-watch-{worker}-{os.getpid()}.jsonl")
        self._last = _snapshot()
        self._write(dict(event="baseline", modules=len(self._last["modules"]),
                         classes=len(self._last["classes"]), threads=self._last["threads"]))

    def _write(self, record):
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _step(self, event, nodeid):
        current = _snapshot()
        changes = _diff(self._last, current)
        self._last = current
        if changes:
            self._write(dict(changes, event=event, after=nodeid))

    def pytest_collection_finish(self, session):
        self._step("collection", None)

    def pytest_runtest_logfinish(self, nodeid, location):
        self._step("test", nodeid)


def register(config):
    """Register the watcher once per process when MOJOLEARN_TEST_STATE_WATCH is set."""
    directory = os.environ.get("MOJOLEARN_TEST_STATE_WATCH", "").strip()
    if not directory or config.pluginmanager.has_plugin("mojolearn-state-watch"):
        return
    config.pluginmanager.register(StateWatch(directory), "mojolearn-state-watch")
