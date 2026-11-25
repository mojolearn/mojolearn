# SPDX-License-Identifier: Apache-2.0
"""UNPACK at import: the GPU tokenizer image's entry (`tokenizer_gpu_entry_v1`,
core/tokenizer_gpu_entry_table.mojo) resolved ONCE from the GPU tokenizer
binding's connection tuple (`mojolearn.tokenizer._gpu_binding`: the module
first, or one refusal string). Shared by the corpus door
(bindings/lm_corpus_metadata_door.mojo `corpus_gpu_connect`) and the
`models.Tokenizer` door (bindings/tokenizer_public_native_door.mojo
connect_vocab): one mechanism, read once per connection, never per call.
Import-time metadata only: no numerical work to split."""
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type
from core.entry_table import EntryRef
from core.tokenizer_gpu_entry_table import tokenizer_gpu_resolve, tokenizer_gpu_vendor_word


def tokenizer_gpu_entry_connect(gpu: PythonObject, mut entry: EntryRef) raises -> String:
    """Resolve the image's entry into `entry`. Returns "" when resolved, else the words device='gpu'
    is refused with (the tuple's own string, or the connection failure); `entry` is then left as
    the caller holds it (unresolved)."""
    var bi = native_loaded_module("builtins")
    if py_type(gpu) is not bi.tuple or len(gpu) < 1:
        raise Error("TypeError: GPU tokenizer connection requires its connection tuple")
    if py_type(gpu[0]) is bi.str:
        return String(py=gpu[0])
    try:
        var module = gpu[0]
        var mode = Int(py=py_call(module.x_tokenizer_numeric_mode))
        var vendor = tokenizer_gpu_vendor_word(String(py=py_call(module.x_tokenizer_vendor)))
        entry = tokenizer_gpu_resolve(mode, vendor)
        return String("")
    except error:
        return String("ImportError: device='gpu' needs the GPU tokenizer binding, which is not connected: ") + String(error)
