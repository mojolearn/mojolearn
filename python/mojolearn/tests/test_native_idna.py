"""Native hostname conversion versus the independent stdlib IDNA oracle."""
import socket
import stringprep

import pytest

from mojolearn import _backend
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture
def native():
    entry = _backend.load_host_module('_mojolearn_tokenizer_host').native_transport_call
    entry('initialize', ())
    return entry


@pytest.mark.parametrize('host', [
    '', 'localhost', 'LOCAL_host.example.', '127.0.0.1', '::1',
    'bücher.example', 'faß.de', 'mañana.example', '例え。テスト',
    'ＦＯＯ．example', 'δοκιμή.example', 'مثال.example', 'עברית.example',
    'a\u00adb.example', '\u212a.example', '🦄.example',
])
def test_native_encoding_matches_stdlib_without_codec_callbacks(native, host):
    expected = host.encode('idna')
    audit = audit_native_call(native, 'idna', (host,))
    assert audit.assert_native() == expected


@pytest.mark.parametrize('host', [
    '.example', 'a..example', 'a' * 64 + '.example', '\u00ad.example',
    '\ue000.example', 'aא.example', 'אa.example', '\u202e.example',
    'xn--é.example',
])
def test_prohibition_and_label_bounds_preserved(native, host):
    with pytest.raises(UnicodeError):
        host.encode('idna')
    observed = audit_native_call(native, 'idna', (host,))
    assert observed.exception is not None
    report = observed.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work',
                'unobserved_existing_threads'):
        assert not report[key], report


def test_all_mapping_exceptions_against_independent_unicode_oracle(native):
    for code in sorted(stringprep.b3_exceptions):
        host = chr(code) + '.example'
        try:
            expected = host.encode('idna')
        except UnicodeError:
            with pytest.raises(RuntimeError):
                native('idna', (host,))
        else:
            assert native('idna', (host,)) == expected, hex(code)


def test_named_listen_connect_and_numeric_endpoint_paths_are_native(native):
    listener = audit_native_call(native, 'listen', ('localhost', 0, 3.)).assert_native()
    client = accepted = None
    try:
        # Native transport sockets are owned native capsules, not Python
        # socket objects (bindings/native_transport.mojo _transport_own), so
        # the endpoint and the bytes are read through the native entries.
        address = audit_native_call(native, 'address', (listener, False)).assert_native()
        port = audit_native_call(native, 'port', (listener,)).assert_native()
        assert address[1] == port and len(address) in (2, 4)  # IPv4 or IPv6
        client = audit_native_call(native, 'connect', ('localhost', port, 3.)).assert_native()
        accepted = audit_native_call(native, 'accept', (listener, 3.)).assert_native()
        # No test-created thread runs inside any audited native operation.
        audit_native_call(native, 'send', (client, {'cmd': 'hostname'}, b'hostname-native')).assert_native()
        head, payload = audit_native_call(native, 'receive', (accepted, 32)).assert_native()
        assert head == {'cmd': 'hostname', 'nbytes': 15} and payload == b'hostname-native'
    finally:
        for item in (accepted, client, listener):
            if item is not None:
                audit_native_call(native, 'close', (item,)).assert_native()
