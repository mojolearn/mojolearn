# SPDX-License-Identifier: Apache-2.0
"""Archive strict Unicode errors cannot dispatch registered Python handlers."""
import _codecs
import codecs
import io
import struct
import unicodedata
import zipfile

import numpy as np
import pytest

from mojolearn._classical_host import HostLinearRegression
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_classical_archive_reader_native import _payload, _state
from mojolearn.tests.test_classical_archive_writer_native import _scaler_model
from mojolearn.tests.test_host_inference_native import _refused


def _npy(descr=b"<f4", data=b"\x00\x00\x80\x3f"):
    header = b"{'descr': '" + descr + b"', 'fortran_order': False, 'shape': (1,), }\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def _archive(member, name="extra.npy"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, member)
        with zipfile.ZipFile(io.BytesIO(_payload())) as original:
            for item in original.infolist():
                archive.writestr(item.filename, original.read(item.filename))
    return output.getvalue()


def _utf8_name(raw, *, extra=False):
    payload = _archive(_npy(), "x" * len(raw))
    data = bytearray(payload)
    central = data.index(b"PK\x01\x02")
    if not extra:
        struct.pack_into("<H", data, 6, 2048)
        struct.pack_into("<H", data, central + 8, 2048)
        data[30:30 + len(raw)] = raw
        data[central + 46:central + 46 + len(raw)] = raw
        return bytes(data)
    # Unicode-path extra field: preserve a valid ordinary name, CRC, framing
    # and offsets so malformed Unicode reaches the dedicated extra decoder.
    original = bytes(data[central + 46:central + 46 + len(raw)])
    record = b"\x01" + struct.pack("<I", zipfile.crc32(original)) + raw
    field = struct.pack("<HH", 0x7075, len(record)) + record
    at = central + 46 + len(raw)
    struct.pack_into("<H", data, central + 30, len(field))
    data[at:at] = field
    end = data.rindex(b"PK\x05\x06")
    size = struct.unpack_from("<I", data, end + 12)[0]
    struct.pack_into("<I", data, end + 12, size + len(field))
    return bytes(data)


def _expected_error(operation):
    with pytest.raises(UnicodeError) as caught:
        operation()
    return str(caught.value)


def _strict_call(operation, target, expected, poison):
    calls = []
    original = codecs.lookup_error("strict")
    def foreign(error):
        calls.append(type(error).__name__)
        raise error
    if poison:
        codecs.register_error("strict", foreign)
    try:
        audit = audit_native_call(operation, target)
    finally:
        codecs.register_error("strict", original)
    _refused(audit, ValueError)
    assert type(audit.exception) is ValueError
    assert str(audit.exception) == expected
    assert calls == []


_UTF8 = [b"\xff", b"\x80", b"\xc0\x80", b"\xc2", b"\xc2x", b"\xe0\x80\x80",
         b"\xed\xa0\x80", b"\xe2\x82", b"\xf0\x8f\x80\x80", b"\xf4\x90\x80\x80",
         b"\xf0\x90\x80", b"\xf5\x80\x80\x80"]
_UTF8_NAMES = [b"x" + fragment + b".npy" for fragment in _UTF8] + [b"x\xc2", b"x\xe2\x82", b"x\xf0\x90\x80"]


@pytest.mark.parametrize("raw", _UTF8_NAMES, ids=lambda value: value.hex())
@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("extra", [False, True])
def test_zip_unicode_names_preserve_strict_errors_without_callbacks(raw, poison, extra):
    expected = _expected_error(lambda: raw.decode("utf8"))
    _strict_call(HostLinearRegression.load, io.BytesIO(_utf8_name(raw, extra=extra)), expected, poison)


@pytest.mark.parametrize("word", [0xD800, 0xDFFF, 0x110000, 0xFFFFFFFF])
@pytest.mark.parametrize("poison", [False, True])
def test_npy_unicode_words_preserve_strict_errors_without_callbacks(word, poison):
    raw = struct.pack("<II", 65, word)
    expected = _expected_error(lambda: _codecs.utf_32_le_decode(raw))
    _strict_call(HostLinearRegression.load, io.BytesIO(_archive(_npy(b"<U2", raw))), expected, poison)


_ESCAPES = [b"\\x", b"\\xQ2", b"\\u00", b"\\uQQQQ", b"\\U00110000", b"\\UFFFFFFFF",
            b"\\N", b"\\N{", b"\\N{}", b"\\N{NOT A UNICODE NAME}", b"\\N{KEYCAP DIGIT ONE}"]


@pytest.mark.parametrize("escape", _ESCAPES, ids=lambda value: value.decode("ascii"))
@pytest.mark.parametrize("poison", [False, True])
def test_npy_literal_errors_preserve_strict_errors_without_callbacks(escape, poison):
    expected = _expected_error(lambda: _codecs.unicode_escape_decode(escape))
    _strict_call(HostLinearRegression.load, io.BytesIO(_archive(_npy(escape))), expected, poison)


@pytest.mark.parametrize("text", ["\ud800", "before\udfffafter", "x\ud800\udfffy"])
@pytest.mark.parametrize("poison", [False, True])
def test_public_save_unicode_metadata_preserves_error_and_partial_bytes(text, poison):
    model = _scaler_model("standard")
    model.numeric_mode_ = text
    expected = _expected_error(lambda: _codecs.utf_32_le_encode(text))
    plain = io.BytesIO()
    with pytest.raises(ValueError) as baseline:
        model.save(plain)
    assert str(baseline.value) == expected
    target = io.BytesIO()
    _strict_call(model.save, target, expected, poison)
    assert target.getvalue() == plain.getvalue()
    assert model.numeric_mode_ == text


@pytest.mark.parametrize("descr", [b"<f4", b"<\\x664", b"<\\u00664", b"<\\U000000664",
                                    b"<\\1464", b"<\\N{LATIN SMALL LETTER F}4", b"<\\\nf4"])
def test_valid_npy_literal_bytes_preserve_loaded_state(descr):
    model = audit_native_call(HostLinearRegression.load, io.BytesIO(_archive(_npy(descr)))).assert_native()
    _state(model)


@pytest.mark.parametrize("name", ["caf\u00e9.npy", "\u03bb.npy", "\U0001f600.npy"])
def test_valid_zip_utf8_and_utf32_metadata_preserve_loaded_state(name):
    raw = "A\u03bb\U0001f600".encode("utf-32-le")
    model = audit_native_call(HostLinearRegression.load, io.BytesIO(_archive(_npy(b"<U3", raw), name))).assert_native()
    _state(model)


@pytest.mark.parametrize("entry", ["lookup", "lookup_error", "utf_32_le_decode", "utf_32_le_encode", "unicode_escape_decode", "unicode_name"])
def test_archive_unicode_uses_connected_native_entries(monkeypatch, entry):
    calls = []
    module, name = (unicodedata, "lookup") if entry == "unicode_name" else (_codecs, entry)
    original = getattr(module, name)
    def foreign(*args, **kwargs):
        calls.append(entry)
        return original(*args, **kwargs)
    payload = _archive(_npy(b"<\\N{LATIN SMALL LETTER F}4"))
    model = _scaler_model("standard")
    with monkeypatch.context() as patch:
        patch.setattr(module, name, foreign)
        loaded = audit_native_call(HostLinearRegression.load, io.BytesIO(payload)).assert_native()
        output = io.BytesIO()
        audit_native_call(model.save, output).assert_native()
    _state(loaded)
    assert calls == []
    assert output.getvalue()
