"""Inert v2 payload/response fixtures; no provider requests, SSH or credentials."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from runpod_gpu_request import OVERRIDE, build_v2_request, main


def request(**kwargs):
    values = dict(name='owned', image='image', gpus='NVIDIA L4,NVIDIA L40S',
                  disk=80, count=1, attempt=0, env={})
    values.update(kwargs)
    return build_v2_request(**values)


def test_v2_uses_documented_open_ended_floor_and_direct_ssh_fields():
    assert request() == {
        'name': 'owned', 'image': 'image', 'cloud': 'SECURE', 'disk': 80,
        'startSsh': True, 'ports': ['22/tcp'],
        'gpu': {'id': 'NVIDIA L4', 'count': 1, 'minCudaVersion': '13.0'},
    }


def test_v2_rotates_only_requested_gpu_types_without_changing_allocation():
    rows = [request(count=2, attempt=i) for i in range(4)]
    assert [row['gpu']['id'] for row in rows] == ['NVIDIA L4', 'NVIDIA L40S'] * 2
    assert all(row['gpu']['count'] == 2 and row['gpu']['minCudaVersion'] == '13.0' for row in rows)


def test_v2_explicit_exact_override_is_validated_and_excludes_floor_field():
    row = request(env={OVERRIDE: '13.2,13.0', 'MOJOLEARN_DEVPOD_CLOUD_TYPE': 'COMMUNITY',
                       'MOJOLEARN_DEVPOD_MIN_VCPU': '8'})
    assert row['gpu'] == {'id': 'NVIDIA L4', 'count': 1, 'allowedCudaVersions': ['13.0', '13.2'],
                          'minVcpuCountPerGpu': 8}
    assert row['cloud'] == 'COMMUNITY'


@pytest.mark.parametrize('kwargs', [
    {'env': {OVERRIDE: '12.9'}}, {'env': {OVERRIDE: ''}}, {'env': {OVERRIDE: '13.x'}},
    {'env': {'MOJOLEARN_DEVPOD_MIN_VCPU': '0'}}, {'env': {'MOJOLEARN_DEVPOD_CLOUD_TYPE': 'ALL'}},
    {'gpus': 'NVIDIA L4,'}, {'gpus': 'AMD Instinct MI300X OAM'}, {'count': 0},
    {'count': 9}, {'disk': 0}, {'attempt': -1},
])
def test_v2_invalid_inputs_refuse_before_create(kwargs):
    with pytest.raises(ValueError):
        request(**kwargs)


def test_v2_cli_needs_no_catalog_or_network(tmp_path, monkeypatch):
    for name in (OVERRIDE, 'MOJOLEARN_DEVPOD_CLOUD_TYPE', 'MOJOLEARN_DEVPOD_MIN_VCPU'):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / 'request.json'
    assert main(['v2', str(out), 'owned', 'image', 'NVIDIA L4', '80', '1', '0']) == 0
    assert json.loads(out.read_text())['gpu']['minCudaVersion'] == '13.0'


def parsed(tmp_path, value, field, *args, version=2):
    (tmp_path/'rp.body').write_text(json.dumps(value))
    env = {**os.environ, 'TMPD': str(tmp_path),
           'LIB': str(Path(__file__).with_name('runpod_pod_lib.sh'))}
    result = subprocess.run(['bash', '-c', 'source "$LIB"; rp_select_api "$1"; shift; rp_py "$@"',
                             'fixture', str(version), field, *args],
                            env=env, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def test_v2_response_cost_status_and_direct_ssh_without_trusting_command(tmp_path):
    row = {'id': 'owned123', 'status': 'RUNNING', 'cost': .69,
           'ssh': {'direct': {'host': '192.0.2.5', 'port': 23456, 'username': 'root',
                              'command': 'do-not-execute-this'}, 'proxy': {'host': 'ignore'}}}
    assert parsed(tmp_path, row, 'id') == 'owned123'
    assert parsed(tmp_path, row, 'cost') == '0.69'
    assert parsed(tmp_path, row, 'status') == 'RUNNING'
    assert parsed(tmp_path, row, 'ssh') == '-p 23456 root@192.0.2.5'
    assert parsed(tmp_path, {'cost': 0}, 'cost') == '0'


def test_v2_unready_or_malformed_direct_ssh_is_not_a_ready_box(tmp_path):
    assert parsed(tmp_path, {'ssh': {'direct': None, 'proxy': {'host': 'ssh.runpod.io'}}}, 'ssh') == ''
    assert parsed(tmp_path, {'ssh': {'direct': {'host': '-o ProxyCommand=bad', 'port': 22, 'username': 'root'}}}, 'ssh') == ''
    assert parsed(tmp_path, {'ssh': {'direct': {'host': '192.0.2.5', 'port': True, 'username': 'root'}}}, 'ssh') == ''


@pytest.mark.parametrize('pagination', [None, {}, {'hasNextPage': True, 'nextCursor': 'more'},
                                       {'hasNextPage': False, 'nextCursor': 'more'}])
def test_v2_partial_listing_cannot_prove_absence_or_disarm(tmp_path, pagination):
    row = {'pods': [], 'pagination': pagination}
    assert parsed(tmp_path, row, 'listingvalid') == 'no'
    assert parsed(tmp_path, row, 'hasid', 'owned123') == ''
    assert parsed(tmp_path, row, 'byname', 'owned') == ''


def test_v2_complete_listing_and_v1_legacy_shape(tmp_path):
    row = {'pods': [{'id': 'owned123', 'name': 'owned'}],
           'pagination': {'hasNextPage': False, 'nextCursor': None}}
    assert parsed(tmp_path, row, 'listingvalid') == 'yes'
    assert parsed(tmp_path, row, 'byname', 'owned') == 'owned123'
    assert parsed(tmp_path, row, 'hasid', 'owned123') == 'yes'
    assert parsed(tmp_path, row, 'hasid', 'absent') == 'no'
    assert parsed(tmp_path, [], 'listingvalid', version=1) == 'yes'
    old = {'desiredStatus': 'RUNNING', 'costPerHr': 2.39,
           'publicIp': '192.0.2.9', 'portMappings': {'22': 12345}}
    assert parsed(tmp_path, old, 'ssh', version=1) == '-p 12345 root@192.0.2.9'
    assert parsed(tmp_path, old, 'cost', version=1) == '2.39'
    assert parsed(tmp_path, old, 'status', version=1) == 'RUNNING'


@pytest.mark.parametrize('case, retained_guard, expected_code', [
    ('stock', False, 1), ('schema', False, 42),
    ('uncertain', True, 42), ('partial-after', True, 42),
])
def test_create_recovery_retains_full_errors_and_never_retries_uncertainty(
        tmp_path, case, retained_guard, expected_code):
    # Execute only the isolated controller function against an in-memory API
    # fixture. No real curl, lease process, SSH, keys or provider calls exist.
    root = Path(__file__).resolve().parents[1]
    source = (root/'tools/dev_pod.sh').read_text()
    function = source[source.index('runpod_create() {'):source.index('\nwrite_state() {')]
    fake = r'''
die() { echo "$*" >&2; exit 42; }
say() { :; }
now() { echo 0; }
arm_mac_deadman() { mkdir -p "$D/deadman"; echo 999999999 > "$D/deadman.pid"; }
pkill() { :; }
kill() { :; }
rp_call() {
    if [ "$1" = POST ]; then
        printf 'POST\n' >> "$TMPD/calls"
        RP_CODE=400
        [ "$CASE" != uncertain ] || RP_CODE=503
        python3 - "$TMPD/rp.body" "$CASE" <<'PY'
import json, sys
message = 'out of stock' if sys.argv[2] in ('stock', 'partial-after') else 'schema rejected'
with open(sys.argv[1], 'w') as out:
    json.dump({'error': message + ':' + 'retained-detail-' * 80}, out)
PY
    else
        RP_CODE=200
        python3 - "$TMPD/rp.body" "$CASE" "$TMPD/calls" <<'PY'
import json, pathlib, sys
partial = sys.argv[2] == 'partial-after' and pathlib.Path(sys.argv[3]).exists()
with open(sys.argv[1], 'w') as out:
    json.dump({'pods': [], 'pagination': {'hasNextPage': partial, 'nextCursor': 'more' if partial else None}}, out)
PY
    fi
}
mkdir -p "$D"
code=0
runpod_create 0 'NVIDIA L4,NVIDIA L40S' image 0 || code=$?
exit "$code"
'''
    script = 'source "$ROOT/tools/runpod_pod_lib.sh"\n' + function + fake
    temp = tmp_path/'tmp'
    temp.mkdir()
    state = tmp_path/'state'
    state.mkdir()
    lane = state/'fixture'
    env = {**os.environ, 'ROOT': str(root), 'TMPD': str(temp), 'STATE_ROOT': str(state),
           'D': str(lane), 'KEY': 'fixture', 'CASE': case, 'DISK_GB': '80',
           'NV_GPU_COUNT': '1', 'READY_TIMEOUT': '1', 'minutes': '1'}
    for name in (OVERRIDE, 'MOJOLEARN_DEVPOD_CLOUD_TYPE', 'MOJOLEARN_DEVPOD_MIN_VCPU'):
        env.pop(name, None)
    result = subprocess.run(['bash', '-c', script], env=env, text=True, capture_output=True)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert (temp/'calls').read_text() == 'POST\n'
    assert (lane/'deadman.pid').exists() is retained_guard
    evidence, = state.glob('fixture.requests-*')
    response = json.loads((evidence/'response-1.json').read_text())
    assert response['error'].endswith('retained-detail-' * 80)
    assert json.loads((evidence/'request-1.json').read_text())['gpu']['minCudaVersion'] == '13.0'


def test_slow_readiness_preserves_status_and_ssh_errors_without_restarting(tmp_path):
    source = Path(__file__).with_name('dev_pod.sh').read_text()
    start = source.index('    _deadline=$(( $(now) + READY_TIMEOUT )); SSH_TARGET=""; _ready_attempt=0')
    end = source.index('    if [ "$4" = 1 ]; then', start)
    body = source[start:end]
    (tmp_path / 'evidence').mkdir()
    env = dict(os.environ, TMPD=str(tmp_path), RP='https://invalid.test/v2', POD_ID='owned',
               READY_TIMEOUT='900', SSH_CONNECT_TIMEOUT='120', SSH_OPTS='',
               _request_evidence=str(tmp_path / 'evidence'))
    fixture = r"""
now() { date +%s; }
rp_call() { RP_CODE=200; printf '{"id":"owned","status":"RUNNING"}' > "$TMPD/rp.body"; }
rp_py() { echo '-p 23456 root@192.0.2.5'; }
with_timeout() {
    if [ ! -f "$TMPD/first" ]; then touch "$TMPD/first"; echo 'Connection timed out during banner exchange' >&2; return 255; fi
    echo SSH-OK
}
sleep() { :; }
delete_pod() { echo SHOULD_NOT_DELETE >&2; exit 91; }
die() { echo SHOULD_NOT_FAIL >&2; exit 92; }
"""
    result = subprocess.run(['bash', '-c', fixture + body], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    evidence = tmp_path / 'evidence'
    assert len(list(evidence.glob('readiness-*.json'))) == 2
    assert 'banner exchange' in (evidence / 'ssh-1.log').read_text()
    assert (evidence / 'ssh-2.log').read_text().strip() == 'SSH-OK'
    assert len((evidence / 'readiness.tsv').read_text().splitlines()) == 2
