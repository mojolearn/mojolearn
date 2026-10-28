# SPDX-License-Identifier: Apache-2.0
# Compile-only retained native corpus filesystem closure; never execute.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.lm_corpus_files import (
    corpus_files_initialize, corpus_path, corpus_path_object, corpus_read,
    corpus_write, corpus_hash_bytes, corpus_hash_file, corpus_file_size,
    corpus_exists, corpus_read_json, corpus_write_json, corpus_atomic_begin,
    corpus_atomic_commit, corpus_atomic_abort, corpus_seconds,
)


@export
def peer3_corpus_files_compile(pointer: PyObjectPtr) abi("C") -> Int:
    try:
        corpus_files_initialize()
        var value = PythonObject(from_owned=Python().cpython().Py_NewRef(pointer))
        var path = corpus_path(value, True, True)
        _ = corpus_path_object(path)
        var data = corpus_read(path)
        corpus_write(path, data)
        _ = corpus_hash_bytes(data)
        _ = corpus_hash_file(path)
        _ = corpus_file_size(path)
        _ = corpus_exists(path)
        corpus_write_json(path, corpus_read_json(path))
        var tmp = corpus_atomic_begin(path)
        corpus_atomic_commit(tmp, path)
        corpus_atomic_abort(tmp)
        _ = corpus_seconds()
        return 0
    except:
        return -1
