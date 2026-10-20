# SPDX-License-Identifier: Apache-2.0
"""Native feeds, schedules and complete worker vs independent protocol oracle."""
import hashlib
import json
import math
import socket
import subprocess
import sys

import numpy as np
import pytest

from mojolearn import Array, _backend, cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import state_fixture


def construct(cls, *args, **kwargs):
    owner = object.__new__(cls)
    audit_native_call(owner.__init__, *args, **kwargs).assert_native()
    return owner


@pytest.mark.parametrize('layout', ['C', 'F', 'arrayF'])
def test_native_batch_feed_owns_and_indexes_absolute_steps(layout):
    source = np.arange(3 * 4 * 2 * 5, dtype=np.int32).reshape(3, 4, 2, 5)
    expected = source.copy()
    if layout != 'C':
        source = np.asfortranarray(source)
    value = Array.from_buffer(source) if layout == 'arrayF' else source
    feed = construct(cv.BatchFeed, value, start_step=11)
    source[...] = -1
    for step, shard in [(11, 0), (12, 3), (13, 2)]:
        batch = audit_native_call(feed.__call__, step, shard).assert_native()
        np.testing.assert_array_equal(np.asarray(batch), expected[step - 11, shard])
    failed = audit_native_call(feed.__call__, 14, 0)
    assert isinstance(failed.exception, ValueError)
    assert not failed.report()['python_frames']


def test_native_rate_table_preserves_float32_schedule_and_bounds():
    rates = [.1, .03, .002]
    table = construct(cv.LearningRateTable, rates, start_step=5)
    rates[0] = 17
    assert audit_native_call(table.__call__, 5).assert_native() == float(np.float32(.1))
    assert audit_native_call(table.__call__, 7).assert_native() == float(np.float32(.002))
    failed = audit_native_call(table.__call__, 4)
    assert isinstance(failed.exception, ValueError)
    assert not failed.report()['python_frames']


@pytest.mark.parametrize('chained', [False, True])
def test_complete_native_gpu_worker_independent_uniform_loss_decay_and_hash(tmp_path, chained):
    if _backend.vendor() == 'cpu':
        pytest.skip('existing byte-LM cooperative worker requires a GPU')
    state, shape = state_fixture()
    fields = ('parameters', 'm', 'v', 'flags')
    initial = hashlib.sha256(b''.join(state[k].tobytes() for k in fields)).hexdigest()
    after = dict(state)
    after['parameters'] = state['parameters'] * np.float32(1 - .0625 * .25)
    expected = hashlib.sha256(b''.join(after[k].tobytes() for k in fields)).hexdigest()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    fixture = dict(port=port, initial=initial, expected=expected, count=shape.n_total,
                   loss=math.log(shape.vocab_size), chained=chained)
    fixture_path = tmp_path / 'worker-oracle.json'
    fixture_path.write_text(json.dumps(fixture))
    script = r'''
import hashlib,json,socket,struct,sys
cfg=json.load(open(sys.argv[1])); server=socket.socket(); server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.bind(('127.0.0.1',cfg['port'])); server.listen(); server.settimeout(15)
sock,_=server.accept(); sock.settimeout(15)
def exact(n):
    result=b''
    while len(result)<n:
        part=sock.recv(n-len(result)); assert part; result+=part
    return result
def receive():
    head=json.loads(exact(struct.unpack('>I',exact(4))[0])); return head,exact(head['nbytes'])
def send(head,data=b''):
    raw=json.dumps(dict(head,nbytes=len(data))).encode(); sock.sendall(struct.pack('>I',len(raw))+raw+data)
hello,data=receive(); assert hello['state']==cfg['initial'] and hello['completed']==0 and not data
assert hello['shards']==[0,1] and hello['n_total']==cfg['count']
send(dict(cmd='step',step=0,chained=cfg['chained'],first=cfg['chained']))
head,data=receive(); assert head['shards']==[0,1] and head['step']==0
assert all(abs(x-cfg['loss'])<2e-6 for x in head['losses'])
if cfg['chained']:
    assert not data
    send(dict(cmd='fold',step=0)); head,data=receive()
    assert head['step']==0 and len(data)==cfg['count']*4
else:
    assert len(data)==cfg['count']*8
assert data==bytes(len(data)), 'uniform zero-output model has nonzero gradient bits'
total=bytes(cfg['count']*4)
send(dict(cmd='apply',step=0,total_sha256=hashlib.sha256(total).hexdigest(),losses=[cfg['loss']]*2),b'' if cfg['chained'] else total)
head,data=receive(); assert head['completed']==1 and head['state']==cfg['expected'] and not data
send(dict(cmd='done')); sock.close(); server.close(); print('worker oracle complete')
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(fixture_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    trainer = None
    try:
        feed = construct(cv.BatchFeed, np.tile(np.arange(5, dtype=np.int32), (1, 2, 1, 1)))
        schedule = construct(cv.LearningRateTable, [.0625])
        worker = construct(cv.Worker, state, shards=[0, 1], batches=feed,
                           lr_for_step=schedule, address=('127.0.0.1', port),
                           connect_timeout=10., timeout=10., chained=chained, name='native')
        trainer = worker.trainer
        assert audit_native_call(worker.run).assert_native() == 1
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
        assert audit_native_call(cv.trainer_state_hash, trainer).assert_native() == expected
        events = audit_native_call(worker.drain_events).assert_native()
        assert len(events) == 1 and events[0]['step'] == 1 and events[0]['state'] == expected
        assert 'total_bytes' not in events[0]
        assert audit_native_call(worker.drain_events).assert_native() == []
    finally:
        if trainer is not None:
            audit_native_call(trainer.close).assert_native()
        if child.poll() is None:
            child.kill()
            child.communicate()
