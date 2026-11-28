"""Write Sphinx pages for public names from the installed package."""

import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType


def write_pages(directory, package, exports, exclusions):
    directory.mkdir(exist_ok=True)
    for old in directory.glob("*.rst"):
        old.unlink()
    objects = {f"{package.__name__}.{name}": getattr(package, name) for name in exports}
    canonical = {id(obj): name for name, obj in objects.items()}
    pending = list(objects.items())

    module_links = {}
    for fullname, obj in pending:
        if not isinstance(obj, ModuleType):
            continue
        links = module_links[fullname] = []
        for name in sorted(getattr(obj, "__all__", ())):
            qualified = f"{obj.__name__.rsplit('.', 1)[-1]}.{name}"
            if name.startswith("_") or name in exclusions or qualified in exclusions:
                continue
            child = getattr(obj, name)
            defining_module = getattr(child, "__module__", "").rsplit(".", 1)[-1]
            if f"{defining_module}.{name}" in exclusions:
                continue
            can_alias = callable(child) or isinstance(child, ModuleType)
            child_name = canonical.get(id(child)) if can_alias else None
            if child_name is None:
                child_name = f"{fullname}.{name}"
                if can_alias:
                    canonical[id(child)] = child_name
                pending.append((child_name, child))
            links.append((name, child_name))

    filenames = filename_map(name for name, _ in pending)

    def write_page(item):
        fullname, obj = item
        page = [fullname, "=" * len(fullname), ""]
        if isinstance(obj, ModuleType):
            page += [f".. automodule:: {fullname}", ""]
            links = module_links[fullname]
            if links:
                page += [".. toctree::", "   :hidden:", ""]
                targets = dict.fromkeys(target for _, target in links)
                page += ["   " + filenames.get(target, target) for target in targets]
                page += [""]
                page += [f"* :doc:`{name} <{filenames.get(target, target)}>`" for name, target in links]
        elif inspect.isclass(obj):
            page.append(f".. autoclass:: {fullname}")
            # The native __getattribute__ binds a module dictionary. Discover only public
            # names before autodoc runs, so Sphinx never formats that dictionary.
            if fullname.rsplit(".", 1)[-1] == obj.__name__:
                owners = {name: base for base in reversed(obj.__mro__) if base.__module__ != "builtins"
                          for name in vars(base) if not name.startswith("_")}
                members = sorted(name for name, owner in owners.items() if name not in exclusions
                                 and f"{owner.__name__}.{name}" not in exclusions
                                 and f"{obj.__name__}.{name}" not in exclusions)
                if members:
                    page += ["   :members: " + ", ".join(members), "   :undoc-members:"]
        elif callable(obj):
            page.append(f".. autofunction:: {fullname}")
        else:
            page.append(f".. py:data:: {fullname}")
            if isinstance(obj, dict):
                page += ["", "   Keys: " + ", ".join(f"``{key}``" for key in obj) + "."]
            elif isinstance(obj, (str, int, float, bool, tuple)):
                page.append(f"   :value: {obj!r}")
        (directory / (filenames.get(fullname, fullname) + ".rst")).write_text("\n".join(page) + "\n")

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=cores) as pool:
        list(pool.map(write_page, pending))


def filename_map(names):
    # Case-insensitive filesystems need different paths for HDBSCAN and hdbscan.
    folded = {}
    for name in names:
        folded.setdefault(name.lower(), []).append(name)
    return {name: name + ("-module" if name.count(".") == 1 else "-lower")
            for group in folded.values() if len(group) > 1 for name in group if name.rsplit(".", 1)[-1].islower()}
