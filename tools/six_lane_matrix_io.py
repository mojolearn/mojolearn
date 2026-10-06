"""Lossless storage for planning matrices; no estimator or numerical work.

Committed matrices use deterministic gzip to stay below Git hosting limits.
External run matrices and old frozen commits can still contain plain JSON.
Logical digests always describe the exact uncompressed bytes, not a JSON
reserialization. Generic artifact/file digests elsewhere remain unchanged.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

MATRIX_PATH = 'experiments/six_lane_integration/matrix.json'


def matrix_bytes(path):
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def read_matrix(path):
    return json.loads(matrix_bytes(path))


def matrix_digest(path):
    return hashlib.sha256(matrix_bytes(path)).hexdigest()


def gzip_bytes(data):
    # Empty filename and zero mtime keep the same bytes across checkout paths.
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0,
                       compresslevel=9) as compressed:
        compressed.write(data)
    return stream.getvalue()


def write_matrix(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = gzip_bytes(data) if path.suffix == '.gz' else data
    path.write_bytes(stored)
    return dict(logical_bytes=len(data), logical_sha256=hashlib.sha256(data).hexdigest(),
                stored_bytes=len(stored), stored_sha256=hashlib.sha256(stored).hexdigest())


def git_matrix_bytes(repo, source, path=MATRIX_PATH):
    """Read exact bytes from a historical raw or future compressed frozen blob."""
    source = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', source + '^{commit}'],
        stderr=subprocess.PIPE).decode().strip()
    names = [path] if path.endswith('.gz') else [path, path + '.gz']
    for name in names:
        result = subprocess.run(['git', '-C', str(repo), 'show', source + ':' + name],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            # A corrupt existing raw/gzip blob is an error, never a fallback.
            return gzip.decompress(result.stdout) if name.endswith('.gz') else result.stdout
    raise subprocess.CalledProcessError(result.returncode, result.args,
                                        output=result.stdout, stderr=result.stderr)


def read_git_matrix(repo, source, path=MATRIX_PATH):
    return json.loads(git_matrix_bytes(repo, source, path))
