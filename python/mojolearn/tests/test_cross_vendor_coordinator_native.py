# SPDX-License-Identifier: Apache-2.0
"""Native coordinator checked against independent socket peers and fold words."""
import gc
import hashlib
import json
import socket
import struct
import subprocess
import sys

import pytest

from mojolearn import cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call


def wire(peer, header, payload=b''):
    raw = json.dumps(dict(header, nbytes=len(payload)), separators=(',', ':')).encode()
    peer.sendall(struct.pack('>I', len(raw)) + raw + payload)


def read_wire(peer):
    def exact(n):
        result = b''
        while len(result) < n:
            part = peer.recv(n - len(result))
            assert part
            result += part
        return result
    head = json.loads(exact(struct.unpack('>I', exact(4))[0]))
    return head, exact(head['nbytes'])


def _transport():
    from mojolearn import _backend
    native = _backend.load_host_module('_mojolearn_tokenizer_host').native_transport_call
    native('initialize', ())
    return native


def native_pair(native):
    """(independent Python peer, owned native coordinator socket). Since the
    cross-vendor protocol moved into native code (e28c8bcf3, cc4c14732) the
    coordinator reads only owned native sockets (bindings/native_transport.mojo
    refuses a Python socket object); the peer stays an independent Python
    socket oracle."""
    server = native('listen', ('127.0.0.1', 0, 5.0))
    try:
        peer = socket.create_connection(native('address', (server, False)), timeout=5)
        accepted = native('accept', (server, 5.0))
    finally:
        native('close', (server,))
    return peer, accepted


def constructor(**kwargs):
    owner = object.__new__(cv.Coordinator)
    audit_native_call(owner.__init__, **kwargs).assert_native()
    return owner


@pytest.mark.parametrize('chained', [False, True])
def test_native_coordinator_fixed_logical_fold_against_independent_wire(chained):
    splits = [[0, 1], [2]] if chained else [[2], [0], [1]]
    owner = constructor(port=0, workers=len(splits), logical_shards=3, steps=1, chained=chained)
    native = _transport()
    sockets = []
    for _ in splits:
        sockets.append(native_pair(native))
    gradients = [struct.pack('<I', word) for word in [0x3f800000, 0x33800000, 0x33800000]]
    expected_total = struct.pack('<I', 0x3f800000)  # fixed left fold is 1.0
    try:
        for index, (ids, (peer, coordinator)) in enumerate(zip(splits, sockets)):
            owner.peers.append(dict(sock=coordinator, name=f'w{index}', shards=ids,
                                    protocol=cv.PROTOCOL, state='start', completed=0,
                                    n_total=1, vendor=f'oracle{index}', chained=chained))
            wire(peer, dict(step=0, shards=ids, losses=[float(3-k) for k in ids]),
                 b'' if chained else b''.join(gradients[k] for k in ids))
            if chained:
                wire(peer, dict(step=0), expected_total)
            wire(peer, dict(completed=1, state='complete'))
        rows = audit_native_call(owner._drive).assert_native()
        assert len(rows) == 1 and rows[0]['losses'] == [3., 2., 1.]
        assert rows[0]['state'] == 'complete' and rows[0]['step'] == 1
        assert rows[0]['total_sha256'] == hashlib.sha256(expected_total).hexdigest()
        for index, (peer, _) in enumerate(sockets):
            head, data = read_wire(peer)
            assert head['cmd'] == 'step' and not data
            if chained:
                head, data = read_wire(peer)
                assert head['cmd'] == 'fold'
                assert data == (b'' if index == 0 else expected_total)
            head, data = read_wire(peer)
            assert head['cmd'] == 'apply'
            assert data == (b'' if chained and index == len(splits)-1 else expected_total)
            assert read_wire(peer)[0]['cmd'] == 'done'
    finally:
        for peer, coordinator in sockets:
            peer.close()
            native('close', (coordinator,))


def test_native_ready_signal_and_repeated_custom_exception_preserve_ownership():
    owner = constructor(port=0, workers=1, logical_shards=1, steps=0)
    assert not audit_native_call(owner.ready.is_set).assert_native()
    assert not audit_native_call(owner.ready.wait, .001).assert_native()
    audit_native_call(owner.ready.set).assert_native()
    assert audit_native_call(owner.ready.wait, 0).assert_native()
    audit_native_call(owner.ready.clear).assert_native()
    gc.collect()
    references = sys.getrefcount(cv.CrossVendorMismatch)
    for _ in range(128):
        audited = audit_native_call(owner._refuse, 'independent ownership check')
        assert type(audited.exception) is cv.CrossVendorMismatch
        report = audited.report()
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            assert not report[key], report
        del audited, report
    gc.collect()
    assert sys.getrefcount(cv.CrossVendorMismatch) == references


def test_whole_native_accept_run_with_independent_process_peer():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    owner = constructor(host='127.0.0.1', port=port, workers=1, logical_shards=1,
                        steps=1, timeout=10., accept_timeout=10.)
    # This independent process is test infrastructure. The product run itself
    # accepts, validates, reduces, commits and closes with no Python threads.
    script = r'''
import hashlib,json,socket,struct,sys,time
port=int(sys.argv[1]); deadline=time.monotonic()+10
while True:
    try:
        sock=socket.create_connection(('127.0.0.1',port), timeout=5); break
    except OSError:
        if time.monotonic()>deadline: raise
        time.sleep(.01)
def exact(n):
    out=b''
    while len(out)<n:
        part=sock.recv(n-len(out)); assert part; out+=part
    return out
def send(head,data=b''):
    raw=json.dumps(dict(head,nbytes=len(data))).encode()
    sock.sendall(struct.pack('>I',len(raw))+raw+data)
def receive():
    head=json.loads(exact(struct.unpack('>I',exact(4))[0])); return head,exact(head['nbytes'])
send(dict(protocol='mojolearn.cross-vendor.v1',name='oracle',shards=[0],state='initial',completed=0,n_total=1,vendor='test'))
head,data=receive(); assert head['cmd']=='step' and head['step']==0 and not data
grad=struct.pack('<f',.75)
send(dict(step=0,shards=[0],losses=[2.]),grad)
head,data=receive(); assert head['cmd']=='apply' and data==grad
assert head['total_sha256']==hashlib.sha256(grad).hexdigest()
send(dict(completed=1,state='agreed'))
assert receive()[0]['cmd']=='done'
sock.close(); print('independent peer complete')
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(port)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        rows = audit_native_call(owner.run).assert_native()
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
        assert rows[0]['state'] == 'agreed' and rows[0]['losses'] == [2.]
        assert owner.bound_port == port
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
