"""Lossless, compressed publication history; metadata only, no quality evaluation."""
import hashlib
import json
from pathlib import Path

from six_lane_matrix_io import gzip_bytes, matrix_bytes


NAME = 'continuation-quality-reviews.json'
POLICY = 'Saved review versions bound to exact receipt SHA256; no quality recomputation or default admission.'


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def _rows(raw):
    rows = json.loads(raw)['rows']
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Review history rows must be a list of objects')
    return rows


def publish_review_history(out, extra_rows):
    """Append exact review versions and migrate legacy JSON without dropping bytes.

    A legacy snapshot retains the exact decoded original, including formatting.
    The current gzip contains the same version-deduplicated rows as the old
    publisher. Its small index separates review versions from race/sample counts.
    If migration was interrupted with both forms present, merge both; corruption
    is an error, never a reason to discard an existing history.
    """
    out = Path(out)
    plain = out / NAME
    compressed = out / (NAME + '.gz')
    index_path = out / 'continuation-quality-reviews-index.json'
    previous_index = json.loads(index_path.read_bytes()) if index_path.exists() else {}
    migrations = {item['decoded_sha256']: item
                  for item in previous_index.get('legacy_snapshots', [])}
    previous_rows = []
    if compressed.exists():
        previous_rows.extend(_rows(matrix_bytes(compressed)))
    if plain.exists():
        original = plain.read_bytes()
        original_rows = _rows(original)
        previous_rows.extend(original_rows)
        digest = _digest(original)
        snapshot = out / 'publication-inputs' / digest / (NAME + '.gz')
        if snapshot.exists():
            if matrix_bytes(snapshot) != original:
                raise ValueError('Legacy review snapshot differs from original bytes')
        else:
            _atomic(snapshot, gzip_bytes(original))
        migrations[digest] = dict(
            original_path=NAME, path=str(snapshot.relative_to(out)),
            decoded_sha256=digest, decoded_bytes=len(original),
            row_count=len(original_rows), stored_sha256=_digest(snapshot.read_bytes()))
    versions = {_digest(json.dumps(row, sort_keys=True).encode()): row
                for row in previous_rows + list(extra_rows)}
    raw = (json.dumps(dict(rows=list(versions.values()), policy=POLICY), indent=2) + '\n').encode()
    stored = gzip_bytes(raw)
    _atomic(compressed, stored)
    index = dict(schema='mojolearn.compressed-review-history/1',
                 path=compressed.name, decoded_sha256=_digest(raw), decoded_bytes=len(raw),
                 stored_sha256=_digest(stored), stored_bytes=len(stored),
                 row_count=len(versions), legacy_snapshots=list(migrations.values()),
                 count_semantics='Historical review versions, not new races or scored samples.',
                 policy=POLICY)
    _atomic(index_path, (json.dumps(index, indent=2, sort_keys=True) + '\n').encode())
    # Both the exact legacy snapshot and the replacement index are durable before
    # removing the oversized legacy file. Failed writes leave it recoverable.
    if plain.exists():
        plain.unlink()
    return index
