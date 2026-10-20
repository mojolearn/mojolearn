"""Native transport vs independent Python wire/process oracles (test only)."""
import json
import os
import socket
import struct

import pytest

from mojolearn import _backend
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture
def native():
    binding = _backend.load_host_module('_mojolearn_tokenizer_host')
    operation = binding.native_transport_call
    operation('initialize', ())
    return operation


@pytest.fixture
def sockets(native):
    server = native('listen', ('127.0.0.1', 0, 5.0))
    client = socket.create_connection(server.getsockname(), timeout=5)
    accepted = native('accept', (server, 5.0))
    try:
        yield client, accepted
    finally:
        client.close()
        native('close', (accepted,))
        native('close', (server,))


def receive_exact(peer, count):
    result = b''
    while len(result) < count:
        part = peer.recv(count - len(result))
        assert part, 'native sender closed before its declared payload'
        result += part
    return result


def assert_native_failure(audited):
    assert audited.exception is not None
    report = audited.report()
    assert not report['python_frames'], report
    assert not report['forbidden_native_calls'], report
    assert not report['wrapper_work'], report
    assert not report['unobserved_existing_threads'], report


def test_native_send_matches_independent_wire(sockets, native):
    client, accepted = sockets
    header = dict(cmd='step', step=17, name='worker α', losses=[-0.0, 1.25, 2],
                  nested=dict(ready=True, unused=None))
    payload = b'\x00\xffnative\x00bytes'
    audited = audit_native_call(native, 'send', (accepted, header, payload))
    audited.assert_native()
    length, = struct.unpack('>I', receive_exact(client, 4))
    raw = receive_exact(client, length)
    assert raw == json.dumps(dict(header, nbytes=len(payload)), sort_keys=True,
                             separators=(',', ':'), ensure_ascii=True).encode()
    assert receive_exact(client, len(payload)) == payload
    assert 'nbytes' not in header


def test_native_receive_fragmented_independent_wire(sockets, native):
    client, accepted = sockets
    payload = bytes(range(256)) * 19
    header = dict(cmd='apply', step=9, nbytes=len(payload), name='κ', flags=[False, None])
    raw = json.dumps(header).encode()
    wire = struct.pack('>I', len(raw)) + raw + payload

    # The small fixture fits the socket buffer. Queue fragmented test writes
    # before the audited operation, without another unobserved Python thread.
    for start in range(0, len(wire), 7):
        client.sendall(wire[start:start + 7])
    audited = audit_native_call(native, 'receive', (accepted, len(payload)))
    audited.assert_native()
    assert audited.result == (header, payload)


@pytest.mark.parametrize('wire,limit,message', [
    (struct.pack('>I', (1 << 20) + 1), 0, 'oversized header'),
    (b'\x00\x00\x00\x0e{"nbytes":100}', 4, 'payload exceeds'),
    (b'\x00\x00\x00\x02[]', 0, 'must be an object'),
    (b'\x00\x00\x00\x0f{"nbytes":true}', 9, 'must be an integer'),
])
def test_native_receive_refuses_invalid_frames(sockets, native, wire, limit, message):
    # The fixture deliberately declares lengths independently of the native
    # serializer; rebuild just the prefix to avoid accidental truncation in
    # the payload-bound cases masking the intended admission failure.
    if len(wire) > 4:
        wire = struct.pack('>I', len(wire) - 4) + wire[4:]
    client, accepted = sockets
    client.sendall(wire)
    audited = audit_native_call(native, 'receive', (accepted, limit))
    assert_native_failure(audited)
    assert message in str(audited.exception)


def test_native_receive_peer_close_is_not_empty_success(sockets, native):
    client, accepted = sockets
    client.sendall(b'\x00\x00')
    client.shutdown(socket.SHUT_WR)
    audited = audit_native_call(native, 'receive', (accepted, 0))
    assert_native_failure(audited)
    assert 'peer closed' in str(audited.exception)


def test_native_process_arguments_environment_exit_and_missing_executable(native, tmp_path):
    # /bin/sh is the independent child oracle. Product transport does not
    # invoke a shell unless explicitly supplied by its caller here.
    output = tmp_path / 'child.txt'
    argv = ['/bin/sh', '-c', 'printf "%s:%s" "$QUALIFICATION_VALUE" "$1" > "$2"; exit 7',
            'worker', 'space and $literal', str(output)]
    env = ['PATH=/usr/bin:/bin', 'QUALIFICATION_VALUE=native']
    spawned = audit_native_call(native, 'spawn', ('/bin/sh', argv, env))
    spawned.assert_native()
    assert spawned.exception is None
    waited = audit_native_call(native, 'wait', (spawned.result, False))
    waited.assert_native()
    assert waited.exception is None
    assert os.waitstatus_to_exitcode(waited.result) == 7
    assert output.read_text() == 'native:space and $literal'
    missing = audit_native_call(native, 'spawn', ('/does-not-exist-mojolearn-test',
                                                 ['/does-not-exist-mojolearn-test'], env))
    assert_native_failure(missing)
