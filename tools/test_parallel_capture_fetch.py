# SPDX-License-Identifier: Apache-2.0
import io
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import parallel_capture_fetch as capture

COMMIT = 'a' * 40


def archive_bytes(commit=COMMIT, malicious=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        payload = ('commit=' + commit + '\n').encode()
        member = tarfile.TarInfo('leg.txt')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
        if malicious:
            member = tarfile.TarInfo(malicious)
            archive.addfile(member, io.BytesIO())
    return stream.getvalue()


def test_retains_valid_source_snapshot_and_receipt(tmp_path, monkeypatch):
    def run(command, stdout, **kwargs):
        assert command[-2] == 'root@example.org'
        assert command[-1] == capture.REMOTE
        stdout.write(archive_bytes())
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(subprocess, 'run', run)
    record = capture.snapshot('example.org', 22, tmp_path, COMMIT, 1)
    assert record['status'] == 'RETAINED'
    assert len(record['sha256']) == 64
    capture.validate(tmp_path / record['path'], COMMIT)


@pytest.mark.parametrize('fault', ['wrong-source', 'unsafe', 'connection', 'timeout'])
def test_failed_fetch_never_replaces_prior_snapshot(tmp_path, monkeypatch, fault):
    prior = tmp_path / 'snapshot-0001.tar.gz'
    original = archive_bytes()
    prior.write_bytes(original)
    def run(command, stdout, **kwargs):
        if fault == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        stdout.write(archive_bytes('b' * 40 if fault == 'wrong-source' else COMMIT,
                                   '../credential' if fault == 'unsafe' else None))
        return SimpleNamespace(returncode=255 if fault == 'connection' else 0, stderr=b'failed')
    monkeypatch.setattr(subprocess, 'run', run)
    record = capture.snapshot('example.org', 22, tmp_path, COMMIT, 2)
    assert record['status'] == 'FAILED'
    assert prior.read_bytes() == original
    assert not (tmp_path / 'snapshot-0002.tar.gz').exists()
    assert not list(tmp_path.glob('*.partial'))


def test_host_rejected_before_network(tmp_path):
    with pytest.raises(SystemExit):
        capture.main(['--host', '-oProxyCommand=bad', '--port', '22', '--commit', COMMIT,
                      '--out', str(tmp_path / 'capture')])
