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
