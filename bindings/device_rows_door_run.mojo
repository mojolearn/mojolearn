# SPDX-License-Identifier: Apache-2.0
"""RUN of the DeviceRows door (bindings/device_rows_door.mojo): the typed store
ids a device kernel entry gathers a fold from. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word; five ids per fold, nothing to split across cores."""


struct DoorRowsStore(ImplicitlyCopyable, Movable):
    """The resident store ids of a DeviceRows fold in model selection's
    store (core/msel_device_x.mojo `msel_store`): the float32 C-order base
    (`base_rows` x `row_words`) and the int64 row indices (`rows` words).
    A device kernel entry gathers the fold on the device out of these
    (core/msel_device_x.mojo msel_device_x_f32); no host word is read."""
    var base_id: Int
    var base_rows: Int
    var row_words: Int
    var index_id: Int
    var rows: Int

    def __init__(out self, base_id: Int, base_rows: Int, row_words: Int, index_id: Int, rows: Int):
        self.base_id = base_id
        self.base_rows = base_rows
        self.row_words = row_words
        self.index_id = index_id
        self.rows = rows

    @staticmethod
    def none() -> Self:
        return DoorRowsStore(-1, 0, 0, -1, 0)

    def active(self) -> Bool:
        return self.base_id >= 0 and self.index_id >= 0 and self.rows > 0
