#!/usr/bin/env python3
"""Refuse cached candidate binaries unless every Mojo source byte matches."""
import hashlib
import json
from pathlib import Path
cache=Path('/root/wp67-compile')
source=Path('/root/mojolearn')
rows={}
for path in cache.rglob('*.mojo'):
    relative=path.relative_to(cache)
    if '.pixi' in relative.parts:
        continue
    raw=path.read_bytes()
    assert (source/relative).read_bytes()==raw,relative
    rows[str(relative)]=hashlib.sha256(raw).hexdigest()
assert len(rows)>500
Path('/root/gemm_leg_out/wp67/preflight-source-hashes.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
