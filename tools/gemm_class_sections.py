#!/usr/bin/env python3
"""Digest ELF64 allocated sections, never load segments or whole build files."""
import hashlib
import json
import struct
import sys
from pathlib import Path
b = Path(sys.argv[1]).read_bytes()
assert b[:5] == b'\x7fELF\x02', 'ELF64 required'
e = '<' if b[5] == 1 else '>'
off = struct.unpack_from(e+'Q', b, 40)[0]
size, n, names_index = struct.unpack_from(e+'HHH', b, 58)
rows = [struct.unpack_from(e+'IIQQQQIIQQ', b, off+i*size) for i in range(n)]
name_row = rows[names_index]
names = b[name_row[4]:name_row[4]+name_row[5]]
out = {}
for r in rows:
    name = names[r[0]:].split(b'\0', 1)[0].decode()
    if r[2] & 2:
        out[name] = dict(size=r[5], type=r[1], sha256=hashlib.sha256(
            b[r[4]:r[4]+r[5]] if r[1] != 8 else b'').hexdigest())
assert '.text' in out
print(json.dumps(out, indent=2, sort_keys=True))
