"""Exercise CPU runner cleanup with mocked provider calls, never rent a pod."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/runpod_cpu_leg.sh'


def cleanup_body():
    source = RUNNER.read_text()
    match = re.search(r'^teardown\(\) \{\n.*?^\}', source, re.M | re.S)
    assert match
    return match.group()


def harness(tmp_path, *, attempted, pod='', verified=False, status=0, create=False, retained=True):
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
    body += f'retain_results() {{ return {0 if retained else 1}; }}\n'
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


def test_retention_failure_never_deletes_even_after_success(tmp_path):
    result, deadman, calls, out = harness(tmp_path, attempted=1, pod='known-pod', verified=True, retained=False)
    assert result.returncode == 1
    assert not calls.exists()
    assert (deadman / 'curlrc').is_file()
    assert 'refusing deletion' in result.stdout
    assert 'terminated_verified=0' in (out / 'teardown.txt').read_text()


def test_retention_preserves_all_bytes_and_is_archive_discoverable():
    source = RUNNER.read_text()
    assert 'python3 - /root/leg_out /root/results-cpu-leg-retained.tgz' in source
    assert '"$expected" = "$actual"' in source
    assert 'put-file "$OUT/results-complete.tgz"' in source
    assert 'RESULTS_RETAINED=1' in source
    assert '"$ARCHIVE_STATE/state.env"' in source
    assert '/root/results-cpu-leg' in source


def archive_results(tmp_path, result_root):
    # Run the exact remote archive program against local fixtures. There is no
    # SSH, provider call, upload or teardown in these tests.
    source = RUNNER.read_text()
    marker = "<<'RETENTION_PY' || return 1\n"
    program = source.split(marker, 1)[1].split('\nRETENTION_PY\n', 1)[0]
    archive_path = tmp_path / 'results-complete.tgz'
    result = subprocess.run([sys.executable, '-', str(result_root), str(archive_path)],
                            input=program, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[0] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return archive_path


def test_retention_resolves_top_link_and_keeps_large_native_bytes(tmp_path):
    result_root = tmp_path / 'results-cpu-leg'
    result_root.mkdir()
    leg_out = tmp_path / 'leg_out'
    leg_out.symlink_to(result_root, target_is_directory=True)
    native = result_root / 'accepted.so'
    # The generic log archiver skips native files and anything over 50 MB;
    # this complete-retention archive must keep both categories byte-for-byte.
    with native.open('wb') as stream:
        stream.write(b'\x7fELF\x00accepted-native-binary\xff')
        stream.seek(51 * 1024 * 1024)
        stream.write(b'last-byte\x00')
    (result_root / 'cmd.exit').write_text('0\n')
    archive_path = archive_results(tmp_path, leg_out)
    with tarfile.open(archive_path) as archive:
        assert archive.getmember('leg_out').isdir()
        member = archive.getmember('leg_out/accepted.so')
        assert member.isfile() and member.size == native.stat().st_size
        assert hashlib.sha256(archive.extractfile(member).read()).digest() == hashlib.sha256(native.read_bytes()).digest()
        assert archive.extractfile('leg_out/cmd.exit').read() == b'0\n'


def test_retention_preserves_nested_compiler_links_without_archiving_targets(tmp_path):
    result_root = tmp_path / 'results-cpu-leg'
    source = result_root / 'source'
    source.mkdir(parents=True)
    compiler = tmp_path / 'shared compiler cache'
    compiler.mkdir()
    (compiler / 'mojo').write_bytes(b'compiler cache must stay outside result archive')
    (source / '.pixi').symlink_to(compiler, target_is_directory=True)
    (source / 'cycle').symlink_to(result_root, target_is_directory=True)
    (source / 'broken').symlink_to('../absent')
    (source / 'binding.mojo').write_text('# exact fault-arm source\n')
    leg_out = tmp_path / 'leg_out'
    leg_out.symlink_to(result_root, target_is_directory=True)
    archive_path = archive_results(tmp_path, leg_out)
    with tarfile.open(archive_path) as archive:
        assert set(archive.getnames()) == {
            'leg_out', 'leg_out/source', 'leg_out/source/.pixi',
            'leg_out/source/cycle', 'leg_out/source/broken', 'leg_out/source/binding.mojo',
        }
        for name, target in (('.pixi', str(compiler)), ('cycle', str(result_root)),
                             ('broken', '../absent')):
            member = archive.getmember('leg_out/source/' + name)
            assert member.issym() and member.linkname == target
        assert archive.extractfile('leg_out/source/binding.mojo').read() == b'# exact fault-arm source\n'


def test_cpu_readiness_switches_to_v2_before_status_and_guard_readback(tmp_path):
    """A live v2 SSH endpoint remains reachable after deprecated v1 status403."""
    source = RUNNER.read_text()
    begin = source.index('rp_select_api 2\n_deadline=')
    end = source.index('T_SSH=$(now)', begin)
    body = source[begin:end]
    output = tmp_path / 'calls'
    env = dict(os.environ, CALLS=str(output), POD_ID='owned123', COST_HR='.56', READY_TIMEOUT='900')
    fixture = r"""
set -eu
now() { echo 100; }
rp_select_api() { RP_API_VERSION=$1; RP="https://invalid.test/v$1"; }
rp_call() { printf '%s %s\n' "$1" "$2" >> "$CALLS"; test "$2" = https://invalid.test/v2/pods/owned123; }
rp_py() { test "$RP_API_VERSION" = 2; case "$1" in ssh) echo '-p 23456 root@192.0.2.5';; cost) echo .56;; esac; }
bssh() { test "$SSH_TARGET" = '-p 23456 root@192.0.2.5'; echo SSH-OK; }
die() { exit 90; }
"""
    result = subprocess.run(['bash', '-c', fixture + body], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == ['GET https://invalid.test/v2/pods/owned123']
    assert 'RUNPOD_API_VERSION=2' in source[source.index('ARCHIVE_STATE=', end):]


def test_runpod_ssh_timeouts_refuse_invalid_values_before_provider_access(tmp_path):
    for script in ('runpod_cpu_leg.sh', 'dev_pod.sh', 'runpod_guard.sh'):
        for value in ('0', '301', 'bad', '1; exit 0'):
            env = dict(os.environ, MOJOLEARN_RUNPOD_SSH_CONNECT_TIMEOUT=value,
                       MOJOLEARN_LEASE_DIR=str(tmp_path / 'leases'))
            result = subprocess.run(['bash', str(ROOT / 'tools' / script), 'list'],
                                    env=env, text=True, capture_output=True)
            assert result.returncode == 2, (script, value, result.stderr)
            assert 'SSH connect timeout must be an integer from 1 to 300' in result.stderr
