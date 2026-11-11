# SPDX-License-Identifier: Apache-2.0
"""The native entry-table key of every loaded Mojolearn image.

`KEYS` maps a loaded compiled module to its `(tier, vendor)` key: the tier
code its `<prefix>_numeric_mode()` reports (0 fast, 1 identical) and the
vendor word of its `<prefix>_vendor()` (0 cpu, 1 metal, 2 cuda, 3 hip; the
map of `core/parallel_entry_table.mojo::parallel_entry_vendor`). The loader
records each image once, when it reads the image back
(`_backend._check_vendor`, `load_host_module`). Native owners look the key
of an image up in this dictionary at their boundary and call that image's
published entry table natively (core/parallel_entry_table.mojo,
core/archive_entry_table.mojo); they never call the image's Python doors.
"""

KEYS = {}

_VENDOR_WORDS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}


def record(module, vendor):
    # glue: once per loaded image at load time, its own read-back probes
    name = module.__name__.rsplit(".", 1)[-1]
    prefix = "mojolearn" if name == "_mojolearn" else name[len("_mojolearn_"):]
    word = _VENDOR_WORDS.get(str(vendor), -1)
    try:
        mode = int(getattr(module, prefix + "_numeric_mode")())  # import-time: the image's compiled tier
    except Exception:  # noqa: BLE001 - an image without the probe gets no key; its native callers refuse
        return
    if word >= 0 and mode in (0, 1):
        KEYS[module] = (mode, word)


def key_of(module):
    """The recorded key of a loaded image, or None (import-time glue)."""
    return KEYS.get(module)


def connected_key(name, required=False):
    """The key of the IDENTICAL image `name` the binding loader connects
    (a CPU install's host twin included). When it is not built: None (the
    native caller then refuses by the table's word), or, with `required`,
    the loader's ImportError (import-time glue)."""
    from . import _backend
    try:
        with _backend.host_helper_scope():
            module = _backend.binding(name, "identical")
        if type(module) is _backend._HostBinding:
            module = _backend.load_host_module(module._HostBinding__basename)  # import-time: resolve the CPU-only proxy
    except (ImportError, AttributeError):
        if required:
            raise
        return None
    key = KEYS.get(module)
    if key is None and required:
        raise ImportError(f"mojolearn: the compiled {name} binding reports no native table key "
                          "(numeric mode and vendor read-back); rebuild it with bash bindings/build*.sh")
    return key
