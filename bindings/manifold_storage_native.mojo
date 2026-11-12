# SPDX-License-Identifier: Apache-2.0
# The typed native owner of a fitted manifold matrix (lane entangled-mkit,
# 2026-11-12). A fit stores each fitted matrix on the model as one
# `_ManifoldMatrix` (registered by bindings/_mojolearn_runtime_manifold_host.mojo):
# its host words and/or its x_decomp pool id, the image key and dispatch of
# that id, and the side the fit left it on. It holds no Python object: a read
# by a later transform never moves it (the typed kit, bindings/mani_kit_run.mojo,
# reads it through an MFitted), the copy made for the other side is kept here
# once (rows N291, N301), and its pool id goes back to the pool through the
# image's own dispatch when the owner dies.
#
# TOMBSTONE: ManifoldDevice (the Python-holding pooled device buffer: binding,
# release, identifier, alloc/upload/download/move entries) and the words-holder
# role of ManifoldMatrix (store/device/kept_store/kept_device PythonObject
# fields, words_new/words_count/words_address, matrix_* / device_* accessors)
# deleted 2026-11-12 by lane entangled-mkit; code recoverable at the lane's
# base origin/preinteg/wave9-20261112 (e561a66d6b).
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from x_decomp.descriptor_schema import XD_DEV_FREE


struct ManifoldMatrix(Movable, Writable):
    var rows: Int
    var cols: Int
    # host words (the fit's side, or the kept copy of a device matrix)
    var words: List[Float32]
    var has_words: Bool
    # the pool id (the fit's side, or the kept copy of a host matrix), -1 when none
    var device: Int
    var device_key: Int
    var device_entry: EntryRef
    # the side the fit left it on: every route decision of a later transform
    # is the one the first transform made before
    var on_device: Bool

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.words = List[Float32]()
        self.has_words = False
        self.device = -1
        self.device_key = 0
        self.device_entry = entry_ref_unresolved()
        self.on_device = False

    def words_address(self) -> Int:
        """The host words' first word, 0 when the owner has none."""
        if not self.has_words:
            return 0
        return Int(self.words.unsafe_ptr())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ManifoldMatrix(shape=(", self.rows, ", ", self.cols, "))")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        # The pool id goes back through its own image's dispatch. No Python
        # call, hook or exception state is involved.
        if self.device >= 0:
            try:
                _ = entry_ref_call(self.device_entry, XD_DEV_FREE, List[Int](), [self.device], List[Float64]())
            except:
                pass
