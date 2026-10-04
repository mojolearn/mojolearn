"""Native smoke cleanup must keep the guard after an ambiguous create."""
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'tools/release_wheel_smoke.sh').read_text()


def run_cleanup(tmp_path, attempted=1, pod='', verified=False, create_code=None, list_code='200', listing='[]', response='unparsed'):
    deadman = tmp_path / 'deadman'
    deadman.mkdir()
    (deadman / 'curlrc').write_text('mock secret')
    scratch = tmp_path / 'tmp'
    scratch.mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    calls = tmp_path / 'calls'
    env = dict(os.environ, TMPD=str(scratch), DEADMAN_DIR=str(deadman), OUT=str(out), CALLS=str(calls),
               RP_CREATE_ATTEMPTED=str(attempted), POD_ID=pod, POD_TERMINATED='0', DEADMAN_PID='fake-pid',
               POD_NAME='unique-fixture', COST_HR='', T_POST='', DO_CREATE_ATTEMPTED='0',
               HA_CREATE_ATTEMPTED='0', HA_DEADMAN_PID='', HA_SLOT='', DO_DEADMAN_PID='', DO_LOCK_HELD='0',
               RP='https://invalid.test', PROVIDER='auto', GPU='fixture', CREATE='fixture')
    body = 'set -u\n. tools/runpod_pod_lib.sh\n'
    body += '''now() { echo 1; }
say() { :; }
die() { echo "$*" >&2; exit 1; }
kill() { echo "kill:$*" >> "$CALLS"; }
pkill() { echo "pkill:$*" >> "$CALLS"; }
delete_pod() { echo "delete:$1" >> "$CALLS"; }
'''
    body += f'verify_gone() {{ return {0 if verified else 1}; }}\n'
    body += re.search(r'^teardown\(\) \{\n.*?^\}', SOURCE, re.M | re.S).group() + '\ntrap teardown EXIT\n'
    if create_code is not None:
        import shlex
        body += f'''rp_call() {{
if [ "$1" = POST ]; then RP_CODE={shlex.quote(create_code)}; printf '%s' {shlex.quote(response)} > "$TMPD/rp.body";
else RP_CODE={shlex.quote(list_code)}; printf '%s' {shlex.quote(listing)} > "$TMPD/rp.body"; fi
}}
'''
        start = SOURCE.index('    T_POST=$(now)', SOURCE.index('rent_runpod()'))
        end = SOURCE.index('    printf', SOURCE.index('        die "create FAILED', start))
        # This chunk is inside rent_runpod so its no-stock return remains meaningful.
        body += 'create_path() {\n' + SOURCE[start:end] + '\n}\ncreate_path\nexit $?\n'
    else:
        body += 'exit 0\n'
    result = subprocess.run(['bash', '-c', body], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    return result, deadman, calls


@pytest.mark.parametrize('code,list_code,listing,response', [
    ('000','200','[]','unparsed'),
    ('200','503','[]','{"error":"no instances currently available"}'),
    ('200','200','{"error":"not authorized"}','{"error":"no instances currently available"}'),
    ('200','200','[]','no instances currently available'),
])
def test_ambiguous_or_invalid_listing_retains_guard(tmp_path, code, list_code, listing, response):
    result, deadman, calls = run_cleanup(tmp_path, create_code=code, list_code=list_code, listing=listing, response=response)
    assert result.returncode != 0
    assert (deadman / 'curlrc').exists()
    assert not calls.exists()


def test_explicit_stock_refusal_valid_listing_can_disarm(tmp_path):
    result, deadman, calls = run_cleanup(tmp_path, create_code='200', response='{"error":"no instances currently available"}')
    assert not deadman.exists()
    assert 'kill:fake-pid' in calls.read_text()


@pytest.mark.parametrize('attempted,pod,verified,retained', [(0,'',False,False),(1,'known',True,False),(1,'known',False,True)])
def test_cleanup_outcomes(tmp_path, attempted, pod, verified, retained):
    result, deadman, _ = run_cleanup(tmp_path, attempted=attempted, pod=pod, verified=verified)
    assert deadman.exists() == retained
    assert (result.returncode != 0) == retained
