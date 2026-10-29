# Compile only: the exact bytes/bytearray helper is also used by the host door.
from std.python import PythonObject
from bindings.tokenizer_public_native import tokenizer_decode_utf8


def main() raises:
    var value = PythonObject(None)
    _ = tokenizer_decode_utf8(value, value)
