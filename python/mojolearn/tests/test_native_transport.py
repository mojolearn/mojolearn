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
    client = socket.create_connection(native('address', (server, False)), timeout=5)
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


def test_native_timeout_and_explicit_close_are_not_empty_success(native):
    server = native('listen', ('127.0.0.1', 0, 0.05))
    try:
        audit = audit_native_call(native, 'accept', (server, 0.05))
        assert_native_failure(audit)
        assert 'timed out' in str(audit.exception)
    finally:
        native('close', (server,))
    native('close', (server,))
    audit = audit_native_call(native, 'port', (server,))
    assert_native_failure(audit)
    assert 'closed' in str(audit.exception)


def test_native_handle_destruction_closes_connection(native):
    # A capsule has no cycle and owns exactly one descriptor. Dropping the
    # last binding handle must close it even without an explicit close call.
    server = native('listen', ('127.0.0.1', 0, 5.0))
    client = socket.create_connection(native('address', (server, False)), timeout=5)
    accepted = native('accept', (server, 5.0))
    try:
        del accepted
        assert client.recv(1) == b''
    finally:
        client.close()
        native('close', (server,))


def test_native_receive_releases_gil_for_application_producer(sockets, native):
    import threading
    import time
    client, accepted = sockets
    wire = b'\x00\x00\x00\x0c{"nbytes":0}'
    # The producer is test/application code, outside the native-operation
    # checker. The old socket C implementation released the GIL here too.
    def produce():
        time.sleep(0.03)
        client.sendall(wire)
    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert native('receive', (accepted, 0)) == ({'nbytes': 0}, b'')
    finally:
        producer.join(timeout=2)
    assert not producer.is_alive()


@pytest.mark.parametrize('kind', ['array', 'numpy', 'memoryview'])
def test_native_send_retains_supported_buffer_bytes(sockets, native, kind):
    import array
    if kind == 'numpy':
        import numpy as np
        value = np.arange(8, dtype='i4')
    else:
        value = array.array('i', range(8))
        if kind == 'memoryview':
            value = memoryview(value)
    expected = memoryview(value).cast('B').tobytes()
    client, accepted = sockets
    audited = audit_native_call(native, 'send', (accepted, {}, value))
    audited.assert_native()
    size, = struct.unpack('>I', receive_exact(client, 4))
    assert json.loads(receive_exact(client, size)) == {'nbytes': len(expected)}
    assert receive_exact(client, len(expected)) == expected


def test_transport_refuses_python_buffer_and_dictionary_key_hooks(sockets, native):
    calls = []
    class Foreign:
        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(b'bad')
        def __eq__(self, other):
            calls.append('eq')
            return other == 'nbytes'
        def __hash__(self):
            return hash('nbytes')
    foreign = Foreign()
    header = {foreign: 1}
    calls.clear()
    _, accepted = sockets
    for metadata, payload in [({}, foreign), (header, b'')]:
        audited = audit_native_call(native, 'send', (accepted, metadata, payload))
        assert_native_failure(audited)
    assert calls == []


def test_posix_transport_does_not_invoke_python_audit_hooks():
    import subprocess
    import sys
    import textwrap
    program = textwrap.dedent('''\
        import json, os, signal, socket, sys
        from mojolearn import _backend
        native = _backend.load_host_module('_mojolearn_tokenizer_host').native_transport_call
        native('initialize', ())
        active, events = [False], []
        def hook(event, args):
            if active[0] and (event.startswith('socket.') or event in
                             ('os.posix_spawn', 'os.kill', 'time.sleep')):
                events.append(event)
                raise RuntimeError('native operation invoked an application audit hook')
        sys.addaudithook(hook)
        active[0] = True
        try:
            socket.socket()
        except RuntimeError:
            pass
        finally:
            active[0] = False
        assert events == ['socket.__new__'], events
        events.clear()
        server = client = accepted = None
        pid = None
        active[0] = True
        try:
            server = native('listen', ('127.0.0.1', 0, 5.0))
            port = native('port', (server,))
            client = native('connect', ('localhost', port, 5.0))
            accepted = native('accept', (server, 5.0))
            native('send', (client, {'message': 'native'}, b'payload'))
            assert native('receive', (accepted, 7)) == (
                {'message': 'native', 'nbytes': 7}, b'payload')
            assert native('address', (client, True))[1] == port
            pid = native('spawn', ('/bin/sleep', ['/bin/sleep', '60'], ['PATH=/usr/bin:/bin']))
            assert native('wait', (pid, True)) == -1
            native('terminate', (pid, signal.SIGTERM))
            status = native('wait', (pid, False))
            pid = None
            assert os.waitstatus_to_exitcode(status) == -signal.SIGTERM
        finally:
            active[0] = False
            if pid is not None:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            for handle in (accepted, client, server):
                if handle is not None:
                    native('close', (handle,))
        assert events == [], events
        print(json.dumps({'events': events, 'transport': 'complete'}))
    ''')
    result = subprocess.run([sys.executable, '-c', program], text=True,
                            capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        'events': [], 'transport': 'complete'}


def test_native_ipv6_address_matches_independent_socket(native):
    server = native('listen', ('::1', 0, 5.0))
    address = native('address', (server, False))
    client = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    client.settimeout(5)
    accepted = None
    try:
        assert len(address) == 4 and address[0] == '::1'
        client.connect(address)
        accepted = native('accept', (server, 5.0))
        assert native('address', (accepted, True)) == client.getsockname()
        native('send', (accepted, {}, b'v6'))
        count, = struct.unpack('>I', receive_exact(client, 4))
        assert json.loads(receive_exact(client, count)) == {'nbytes': 2}
        assert receive_exact(client, 2) == b'v6'
    finally:
        client.close()
        if accepted is not None:
            native('close', (accepted,))
        native('close', (server,))


def _live_descriptor_identities(fd_directory):
    identities = {}
    for name in os.listdir(fd_directory):
        try:
            stat = os.fstat(int(name))
        except OSError:
            continue  # listdir's temporary descriptor is already closed
        identities[name] = [stat.st_dev, stat.st_ino, stat.st_mode]
    return identities


def test_descriptor_snapshot_does_not_hide_reused_enumeration_fd(monkeypatch):
    from types import SimpleNamespace
    phase = {'listener_open': False}

    def fstat(fd):
        if fd == 4 and not phase['listener_open']:
            raise OSError('closed enumeration descriptor')
        return SimpleNamespace(st_dev=1, st_ino=100 + fd, st_mode=0o140777)

    monkeypatch.setitem(_live_descriptor_identities.__globals__, 'os',
                        SimpleNamespace(listdir=lambda _: ['3', '4'], fstat=fstat))
    before = _live_descriptor_identities('/unused-test-directory')
    phase['listener_open'] = True
    after = _live_descriptor_identities('/unused-test-directory')
    assert {name: identity for name, identity in after.items()
            if before.get(name) != identity} == {'4': [1, 104, 0o140777]}


def test_owned_transport_descriptors_do_not_survive_exec(native, tmp_path):
    import sys
    # Compare actual descriptor identity before/after native listen, then
    # inspect that same OS identity in a directly spawned independent child.
    fd_directory = '/proc/self/fd' if os.path.isdir('/proc/self/fd') else '/dev/fd'
    # Raw listdir names include its own temporary directory fd. The listener
    # can reuse that number; compare only live fstat identities in BOTH samples.
    before = _live_descriptor_identities(fd_directory)
    server = native('listen', ('127.0.0.1', 0, 5.0))
    output = tmp_path / 'descriptor-check.json'
    try:
        after = _live_descriptor_identities(fd_directory)
        identities = {name: identity for name, identity in after.items()
                      if before.get(name) != identity}
        assert len(identities) == 1, identities
        script = '''import json, os, sys
expected = json.loads(sys.argv[1])
leaked = []
for name, identity in expected.items():
    try:
        stat = os.fstat(int(name))
    except OSError:
        continue
    if [stat.st_dev, stat.st_ino, stat.st_mode] == identity:
        leaked.append(name)
with open(sys.argv[2], 'w') as output:
    json.dump(leaked, output)
'''
        pid = native('spawn', (sys.executable,
            [sys.executable, '-c', script, json.dumps(identities), str(output)],
            [key + '=' + value for key, value in os.environ.items()]))
        status = native('wait', (pid, False))
        assert os.waitstatus_to_exitcode(status) == 0
        assert json.loads(output.read_text()) == []
    finally:
        native('close', (server,))


@pytest.mark.parametrize('operation', ['accept', 'receive', 'wait', 'sleep'])
@pytest.mark.parametrize('handler_raises', [False, True])
def test_interrupted_syscalls_return_without_swallowing_signal_handlers(operation, handler_raises):
    import subprocess
    import sys
    import textwrap
    program = textwrap.dedent('''\
        import json, os, signal, sys, time
        from mojolearn import _backend
        native = _backend.load_host_module('_mojolearn_tokenizer_host').native_transport_call
        native('initialize', ())
        operation, handler_raises = sys.argv[1], sys.argv[2] == 'True'
        seen, handles, pid = [], [], None
        class SignalRaised(Exception):
            pass
        def handler(signum, frame):
            seen.append(signum)
            if handler_raises:
                raise SignalRaised('application signal handler preserved')
        signal.signal(signal.SIGALRM, handler)
        if operation in ('accept', 'receive'):
            server = native('listen', ('127.0.0.1', 0, 60.0))
            handles.append(server)
            args = (server, 60.0)
            if operation == 'receive':
                client = native('connect', ('127.0.0.1', native('port', (server,)), 60.0))
                handles.append(client)
                accepted = native('accept', (server, 60.0))
                handles.append(accepted)
                native('timeout', (accepted, -1.0))
                args = (accepted, 0)
        elif operation == 'wait':
            pid = native('spawn', ('/bin/sleep', ['/bin/sleep', '60'], ['PATH=/usr/bin:/bin']))
            args = (pid, False)
        else:
            args = (60.0,)
        started = time.monotonic()
        outcome = None
        handles_closed, child_reaped = 0, False
        signal.setitimer(signal.ITIMER_REAL, 0.1)
        try:
            try:
                native(operation, args)
            except InterruptedError:
                outcome = 'InterruptedError'
            # CPython may deliver its pending application handler during the
            # first except body after the native binding has returned. The
            # outer catch covers that delivery as well as immediate delivery.
            # This checkpoint is testing/application code, outside native work.
            time.sleep(0)
        except SignalRaised:
            outcome = 'SignalRaised'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if pid is not None:
                os.kill(pid, signal.SIGKILL)
                waited, status = os.waitpid(pid, 0)
                assert waited == pid, (waited, pid)
                child_reaped = True
            for handle in reversed(handles):
                native('close', (handle,))
                handles_closed += 1
            elapsed = time.monotonic() - started
        assert elapsed < 3, elapsed
        assert seen == [signal.SIGALRM], seen
        assert signal.getsignal(signal.SIGALRM) is handler
        assert handles_closed == len(handles), (handles_closed, len(handles))
        assert child_reaped == (pid is not None), (child_reaped, pid)
        assert outcome == ('SignalRaised' if handler_raises else 'InterruptedError'), outcome
        print(json.dumps({'operation': operation, 'outcome': outcome, 'elapsed': elapsed,
                          'signals': seen, 'handles_closed': handles_closed,
                          'child_reaped': child_reaped}))
    ''')
    result = subprocess.run([sys.executable, '-c', program, operation, str(handler_raises)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])['operation'] == operation
