# SPDX-License-Identifier: Apache-2.0
# The typed view of one pandas block array the resample door captured once
# (providers/python/resample_pandas_pyapi.mojo `_RPArray`): the dtype words
# and the storage layout the cell routines read. No Python object: the door's
# `_RPArray` keeps the ndarray view, dtype and scalar API owners that keep
# `address` valid, and RUN routines (`_rp_number`, `_rp_widen`,
# `_rp_cell_code`, `_rp_plan_column`, `_rp_copy_block`, ...) take this view.
# Composed once per array (lane entangled-array-owner-2, 2026-11-12); a RUN
# routine borrows it, so no cell copies it.


struct RPLayout(Copyable, Movable):
    var code: String
    var kind: String
    var item: Int
    var rank: Int
    var rows: Int
    var cols: Int
    var row_stride: Int
    var col_stride: Int
    var address: Int
    var native: Bool
    var number: Int

    def __init__(out self):
        self.code = String("")
        self.kind = String("")
        self.item = 0
        self.rank = 0
        self.rows = 0
        self.cols = 0
        self.row_stride = 0
        self.col_stride = 0
        self.address = 0
        self.native = True
        self.number = 0

    def at(self, row: Int, col: Int = 0) -> Int:
        return self.address + row * self.row_stride + col * self.col_stride
