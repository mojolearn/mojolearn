# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_class_admission.mojo: typed Mojo only (no Python
# imports). The process-wide class-epoch and dictionary-admission maps; the
# watcher callbacks and record reads that use them stay in the door file.
from core.shared_slot import SharedSlot


struct ClassWatch(Defaultable, Movable):
    # watcher: -2 not set up yet, -1 unavailable (Python < 3.12 or no free
    # watcher slot), otherwise the interpreter's type watcher ID.
    var watcher: Int
    var epochs: Dict[Int, Int]

    def __init__(out self):
        self.watcher = -2
        self.epochs = Dict[Int, Int]()


comptime CLASS_WATCH = SharedSlot[ClassWatch, "MojolearnClassWatchV2"]


struct DictWatch(Defaultable, Movable):
    # address -> watch epoch at the last full key admission. The watcher and
    # the epochs are the process-wide shared dictionary watcher in
    # native_loaded_module (one CPython watcher slot for both uses).
    var admitted: Dict[Int, Int]

    def __init__(out self):
        self.admitted = Dict[Int, Int]()


comptime DICT_WATCH = SharedSlot[DictWatch, "MojolearnDictWatchV3"]
