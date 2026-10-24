# SPDX-License-Identifier: Apache-2.0
"""Python door to the Mojo SHA-256 (core/sha256.mojo) for the binding shells.

A bytes-like Python value (bytes, bytearray, a C-contiguous memoryview) is
copied by C into an `array('B')`, whose address and length go to the typed
kernel; the shell never hashes. These replace hashlib.sha256 in the shells
with the same digests over the same bytes in the same order.
"""
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from core.sha256 import Sha256, sha256_hex_addr, sha256_sliced8_hex


def _sha256_store(value: PythonObject) raises -> PythonObject:
    var store = py_call(native_loaded_module("array").array, "B")
    py_call(store.frombytes, value)
    return store


def sha256_update(mut hasher: Sha256, value: PythonObject) raises:
    """hasher.update(value) for a bytes-like Python value."""
    var store = _sha256_store(value)
    hasher.update(Int(py=store.buffer_info()[0]), len(store))
    _ = store^


def sha256_update_text(mut hasher: Sha256, value: PythonObject) raises:
    """hasher.update(str(value).encode("utf-8"))."""
    hasher.update_text(String(py_call(native_loaded_module("builtins").str, value)))


def sha256_hexdigest(value: PythonObject) raises -> PythonObject:
    """hashlib.sha256(value).hexdigest() for a bytes-like Python value."""
    var store = _sha256_store(value)
    var hex = sha256_hex_addr(Int(py=store.buffer_info()[0]), len(store))
    _ = store^
    return PythonObject(hex)


def sha256_digest(hasher: Sha256) raises -> PythonObject:
    """hasher.digest(): the 32 raw digest bytes as Python bytes."""
    return py_call(native_loaded_module("builtins").bytes.fromhex, PythonObject(hasher.hexdigest()))


def sha256_digest_of(value: PythonObject) raises -> PythonObject:
    """hashlib.sha256(value).digest() for a bytes-like Python value."""
    var hasher = Sha256()
    sha256_update(hasher, value)
    return sha256_digest(hasher)


def sha256_sliced8(value: PythonObject) raises -> PythonObject:
    """The cross-vendor v2 slice digest of a bytes-like Python value."""
    var store = _sha256_store(value)
    var hex = sha256_sliced8_hex(Int(py=store.buffer_info()[0]), len(store))
    _ = store^
    return PythonObject(hex)
