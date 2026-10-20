"""Load a binding from the merged native file of its directory.

A release ships one native file per installed directory set (the package
directory, ``identical/``, ``host/``, and each Linux vendor set) instead of one
file per binding: ``bindings/build_merged.sh`` compiles every member into
``<stem>.so``, which exports each member's ``PyInit_<name>``, and writes
``_mojolearn_merged.json`` (``{"files": {"<stem>.so": [names...]}}``) beside it.
CPython finds ``PyInit_<last name part>`` in whatever file the loader is given,
so each member still loads under its own module name.

A member's own ``<name>.so`` always wins when it exists (the per-binding
layout, kept for development and bisecting), so every caller that used to
build ``os.path.join(directory, name + ".so")`` asks ``binding_path`` instead.
This module is import plumbing only: it reads the small index file and joins
paths.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys

INDEX_NAME = "_mojolearn_merged.json"

_INDEX_CACHE = {}


def _index(directory):
    """{member name: merged file path} for `directory`, or {} without an index."""
    path = os.path.join(directory, INDEX_NAME)
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        _INDEX_CACHE.pop(path, None)
        return {}
    hit = _INDEX_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    owners = {}
    for fname, names in record.get("files", {}).items():  # glue: read the merged-file index (file names)
        for name in names:  # glue: member module names of one merged file
            owners[name] = os.path.join(directory, fname)
    _INDEX_CACHE[path] = (stamp, owners)
    return owners


def merged_file(directory, name):
    """The merged file in `directory` that serves `name`, if it exists."""
    path = _index(directory).get(name)
    if path is not None and os.path.isfile(path):
        return path
    return None


def binding_path(directory, name):
    """The file `name` loads from in `directory`: its own ``<name>.so`` when
    present, else the merged file that serves it, else the own path (absent;
    callers name it in their refusal)."""
    own = os.path.join(directory, name + ".so")
    if os.path.exists(own):
        return own
    return merged_file(directory, name) or own


def binding_exists(directory, name):
    """True when `name` can be loaded from `directory`, own file or merged."""
    return os.path.exists(os.path.join(directory, name + ".so")) or merged_file(directory, name) is not None


def merged_members(directory):
    """Every member name a merged file in `directory` serves, sorted."""
    return sorted(n for n, p in _index(directory).items() if os.path.isfile(p))  # glue: list member module names


def merged_files(directory):
    """The merged files `directory`'s index names that exist, sorted."""
    return sorted({p for p in _index(directory).values() if os.path.isfile(p)})  # glue: list merged file paths


def is_merged_file(path):
    """True when `path` is a merged file its directory's index names."""
    real = os.path.realpath(path)
    return any(os.path.realpath(p) == real for p in _index(os.path.dirname(path) or ".").values())  # glue: compare file paths


def load(fullname, path):
    """Create and execute the extension module `fullname` from `path` (its
    own file or a merged one); the caller installs it in sys.modules."""
    loader = importlib.machinery.ExtensionFileLoader(fullname, path)
    spec = importlib.util.spec_from_loader(fullname, loader, origin=path)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class MergedFinder(importlib.abc.MetaPathFinder):
    """Last on sys.meta_path: when the normal finders find no own file for a
    ``mojolearn`` binding (``from . import _mojolearn_trees``), serve it from
    the merged file of the package directory it was looked up in."""

    def find_spec(self, fullname, path, target=None):
        if path is None or not fullname.startswith("mojolearn."):
            return None
        leaf = fullname.rpartition(".")[2]
        if not leaf.startswith("_mojolearn"):
            return None
        for directory in path:  # glue: the package search directories
            merged = merged_file(directory, leaf)
            if merged is not None:
                loader = importlib.machinery.ExtensionFileLoader(fullname, merged)
                return importlib.util.spec_from_file_location(fullname, merged, loader=loader)
        return None


_FINDER = MergedFinder()


def install():
    """Append the finder once (idempotent)."""
    if not any(isinstance(f, MergedFinder) for f in sys.meta_path):  # glue: the import finders
        sys.meta_path.append(_FINDER)
