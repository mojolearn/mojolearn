# SPDX-License-Identifier: Apache-2.0
"""Native feeds, schedules and complete worker vs independent protocol oracle."""
import hashlib
import array
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


@pytest.mark.parametrize('owner_kind', ['array', 'numpy'])
def test_borrowed_array_raw_memoryview_keeps_native_exporter_admitted(owner_kind):
    exporter = array.array('f', [1., -0., 3.5]) if owner_kind == 'array' else np.array([1., -0., 3.5], np.float32)
    value = Array.from_buffer(exporter)
    assert value._mv.obj is None
    assert value._pin.obj is exporter
    original = np.asarray(value).tobytes()
    result = audit_native_call(cv.ordered_fold, [value]).assert_native()
    assert result == original
    exporter[0] = 8.
    assert result == original  # native fold owns its result, despite borrowed input
    assert audit_native_call(cv.ordered_fold, [value]).assert_native() == np.asarray(value).tobytes()


@pytest.mark.parametrize('chained', [False, True])
@pytest.mark.parametrize('policy', ['default', 'checkpoint', 'checkpoint_v2', 'mismatch', 'mismatch_loss'])
def test_complete_native_gpu_worker_independent_uniform_loss_decay_and_hash(tmp_path, chained, policy):
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
                   loss=math.log(shape.vocab_size), chained=chained, mismatch=policy.startswith('mismatch'))
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
hello,data=receive(); assert hello['state']==cfg['initial'] and hello['completed']==0 and not data, ('initial state',hello,cfg['initial'],len(data))
assert hello['shards']==[0,1] and hello['n_total']==cfg['count']
send(dict(cmd='step',step=0,chained=cfg['chained'],first=cfg['chained']))
head,data=receive(); assert head['shards']==[0,1] and head['step']==0
assert all(abs(x-cfg['loss'])<2e-6 for x in head['losses']), ('loss',head['losses'],cfg['loss'])
if cfg['chained']:
    assert not data
    send(dict(cmd='fold',step=0)); head,data=receive()
    assert head['step']==0 and len(data)==cfg['count']*4
else:
    assert len(data)==cfg['count']*8
assert data==bytes(len(data)), ('uniform zero-output model has nonzero gradient bits', [(i,hex(v[0])) for i,v in enumerate(struct.iter_unpack('<I',data)) if v[0]][:8])
total=bytes(cfg['count']*4)
send(dict(cmd='apply',step=0,total_sha256=hashlib.sha256(total).hexdigest(),losses=[cfg['loss']]*2),b'' if cfg['chained'] else total)
if cfg['mismatch']:
    assert sock.recv(1)==b'', 'mismatched expectation must close before acknowledgment'
    sock.close(); server.close(); print('native expected-hash stop complete'); sys.exit(0)
head,data=receive(); assert head['completed']==1 and head['state']==cfg['expected'] and not data, ('committed state',head,cfg['expected'],len(data))
send(dict(cmd='done')); sock.close(); server.close(); print('worker oracle complete')
'''
    stdout_path = tmp_path / 'worker-oracle.stdout.log'
    stderr_path = tmp_path / 'worker-oracle.stderr.log'
    stdout_file = stdout_path.open('w')
    stderr_file = stderr_path.open('w')
    child = subprocess.Popen([sys.executable, '-c', script, str(fixture_path)],
                             stdout=stdout_file, stderr=stderr_file, text=True)
    trainer = None
    run_audit = None
    primary = None
    try:
        feed = construct(cv.BatchFeed, np.tile(np.arange(5, dtype=np.int32), (1, 2, 1, 1)))
        schedule = construct(cv.LearningRateTable, [.0625])
        config = {}
        if policy != 'default':
            config = dict(expected_commits=[dict(step=1, state='0' * 64 if policy == 'mismatch' else expected)],
                          checkpoint_dir=tmp_path, checkpoint_steps=(1,), retain_gradients=True,
                          log_path=tmp_path / 'worker-events.jsonl')
        if policy == 'mismatch_loss':
            config['expected_commits'][0]['losses_f32_hex'] = ['00000000'] * 2
        if policy == 'checkpoint_v2':
            def sliced(raw):
                return hashlib.sha256(''.join(hashlib.sha256(raw[len(raw)*i//8:len(raw)*(i+1)//8]).hexdigest()
                                              for i in range(8)).encode('ascii')).hexdigest()
            wanted_state = hashlib.sha256(''.join(sliced(after[key].tobytes()) for key in fields).encode('ascii')).hexdigest()
            wanted_gradient = sliced(bytes(shape.n_total * 4))
            config['hash_scheme'] = 'sliced-sha256-8.v2'
            config['expected_commits'][0].update(state_sha256=wanted_state, gradient_sha256=wanted_gradient,
                                                  hash_scheme='sliced-sha256-8.v2', lr_f32_hex='3d800000',
                                                  losses_f32_hex=[format(np.float32(math.log(shape.vocab_size)).view(np.uint32).item(), '08x')] * 2)
        worker = construct(cv.Worker, state, shards=[0, 1], batches=feed,
                           lr_for_step=schedule, address=('127.0.0.1', port),
                           connect_timeout=10., timeout=10., chained=chained, name='native', **config)
        trainer = worker.trainer
        run_audit = audit_native_call(worker.run)
        if policy.startswith('mismatch'):
            assert isinstance(run_audit.exception, cv.CrossVendorMismatch)
            assert 'configured expectation' in str(run_audit.exception)
            for field in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
                assert not run_audit.report()[field], run_audit.report()
        else:
            assert run_audit.assert_native() == 1
        child.wait(timeout=15)
        assert child.returncode == 0, stderr_path.read_text()[-4000:]
        assert audit_native_call(cv.trainer_state_hash, trainer).assert_native() == expected
        events = audit_native_call(worker.drain_events).assert_native()
        assert len(events) == 1 and events[0]['step'] == 1 and events[0]['state'] == expected
        if policy == 'default':
            assert 'total_bytes' not in events[0]
        else:
            assert events[0]['total_bytes'] == bytes(shape.n_total * 4)
            logged = json.loads((tmp_path / 'worker-events.jsonl').read_text())
            assert logged['state'] == expected and 'total_bytes' not in logged
            if policy.startswith('checkpoint'):
                from mojolearn import _byte_lm_checkpoint as checkpoint
                restored = audit_native_call(checkpoint.load, events[0]['checkpoint_path']).assert_native()
                assert restored['completed_steps'] == restored['next_batch_index'] == 1
                assert audit_native_call(cv.state_hash, restored).assert_native() == expected
                assert hashlib.sha256((tmp_path / 'ckpt_00000001.blm').read_bytes()).hexdigest() == events[0]['checkpoint_sha256']
            else:
                assert events[0]['expected_mismatch'] == ('state' if policy == 'mismatch' else 'losses_f32_hex')
                assert not (tmp_path / 'ckpt_00000001.blm').exists()
        assert audit_native_call(worker.drain_events).assert_native() == []
    except BaseException as error:
        primary = error
        raise
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=15)
        stdout_file.close()
        stderr_file.close()
        evidence = {'child_returncode': child.returncode,
                    'stdout': str(stdout_path), 'stderr': str(stderr_path),
                    'worker_audit': None if run_audit is None else run_audit.report()}
        (tmp_path / 'worker-oracle-result.json').write_text(json.dumps(evidence, indent=2))
        if primary is not None:
            primary.add_note('Independent child return code %s; complete stdout/stderr: %s, %s\n%s'
                             % (child.returncode, stdout_path, stderr_path, stderr_path.read_text()[-4000:]))
        if trainer is not None:
            cleanup = audit_native_call(trainer.close)
            if primary is None:
                cleanup.assert_native()
            elif cleanup.exception is not None:
                primary.add_note('Trainer cleanup also failed: %r' % cleanup.exception)
