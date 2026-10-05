"""Exercise CPU runner cleanup with mocked provider calls, never rent a pod."""
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/runpod_cpu_leg.sh'


def cleanup_body():
    source = RUNNER.read_text()
    match = re.search(r'^teardown\(\) \{\n.*?^\}', source, re.M | re.S)
    assert match
    return match.group()


def harness(tmp_path, *, attempted, pod='', verified=False, status=0, create=False):
    temp = tmp_path / 'temporary'
    temp.mkdir()
    deadman = tmp_path / 'deadman'
    deadman.mkdir()
    (deadman / 'curlrc').write_text('fake credential retained only for cleanup')
    out = tmp_path / 'out'
    out.mkdir()
    calls = tmp_path / 'calls'
    env = dict(os.environ, TMPD=str(temp), DEADMAN_DIR=str(deadman), OUT=str(out), CALLS=str(calls),
               CREATE_ATTEMPTED=str(attempted), POD_ID=pod, POD_TERMINATED='0', DEADMAN_PID='fake-pid',
               POD_NAME='mojolearn-cpu-fixture-unique', RP='https://invalid.test/v1')
    body = '''set -u
say() { :; }
now() { echo 1; }
bssh() { :; }
write_timings() { :; }
delete_pod() { echo "delete:$1" >> "$CALLS"; }
kill() { echo "kill:$*" >> "$CALLS"; }
pkill() { echo "pkill:$*" >> "$CALLS"; }
die() { echo "$*" >&2; exit 1; }
'''
    body += f'verify_gone() {{ return {0 if verified else 1}; }}\n'
    body += cleanup_body() + '\ntrap teardown EXIT\n'
    if create:
        # Execute the runner's real POST/result-recovery path against a timed-out
        # POST and an empty subsequent listing, an eventually-consistent API case.
        body += '''rp_call() {
  if [ "$1" = POST ]; then RP_CODE=000; echo 'unparsed response' > "$TMPD/rp.body";
  else RP_CODE=200; echo '[]' > "$TMPD/rp.body"; fi
}
rp_py() { :; }
'''
        source = RUNNER.read_text()
        start = source.index('T_POST=$(now)')
        end = source.index('say "pod $POD_ID created.', start)
        body += source[start:end]
    else:
        body += f'exit {status}\n'
    result = subprocess.run(['bash', '-c', body], env=env, capture_output=True, text=True, timeout=10)
    return result, deadman, calls, out


def test_ambiguous_post_retains_name_cleanup_and_credentials(tmp_path):
    result, deadman, calls, out = harness(tmp_path, attempted=0, create=True)
    assert result.returncode == 1
    assert 'POST HTTP 000' in result.stderr  # not the recovery listing's 200
    assert (deadman / 'curlrc').is_file()
    assert not calls.exists()  # no kill/pkill can cancel the only remaining guard
    receipt = (out / 'teardown.txt').read_text()
    assert 'create_attempted=1' in receipt and 'terminated_verified=0' in receipt
    assert 'pod=unknown' in receipt


def test_no_post_can_cancel_prearmed_guard(tmp_path):
    result, deadman, calls, _ = harness(tmp_path, attempted=0, status=1)
    assert result.returncode == 1
    assert not deadman.exists()
    assert 'kill:fake-pid' in calls.read_text()


def test_verified_known_pod_can_cancel_guard(tmp_path):
    result, deadman, calls, out = harness(tmp_path, attempted=1, pod='known-pod', verified=True)
    assert result.returncode == 0
    assert not deadman.exists()
    assert 'delete:known-pod' in calls.read_text()
    assert 'kill:fake-pid' in calls.read_text()
    assert 'terminated_verified=1' in (out / 'teardown.txt').read_text()


def test_failed_delete_retains_guard_and_fails_successful_work(tmp_path):
    result, deadman, calls, out = harness(tmp_path, attempted=1, pod='known-pod', verified=False)
    assert result.returncode == 1
    assert (deadman / 'curlrc').is_file()
    assert calls.read_text().splitlines() == ['delete:known-pod']
    assert 'terminated_verified=0' in (out / 'teardown.txt').read_text()


def test_cpu_runner_uses_shared_provider_primitives():
    source = RUNNER.read_text()
    assert '. "$ROOT/tools/runpod_pod_lib.sh"' in source
    for name in ('load_key', 'rp_call', 'rp_py', 'delete_pod', 'verify_gone', 'write_deadman'):
        assert not re.search(r'^' + name + r'\(\)', source, re.M)
    subprocess.run(['bash', '-n', str(RUNNER)], check=True)
