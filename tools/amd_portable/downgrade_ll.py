"""Rewrite Mojo (LLVM 24) textual IR into syntax an older LLVM (ROCm 6.4 = 19)
parses. Each rule is a syntax-only removal of an attribute/flag the older
parser does not know; every rule that fired is printed so the evidence records
exactly how much rewriting a given module needed."""
import re, sys
RULES = [
    ('captures(none)', re.compile(r'\s+captures\((?:none|[a-z_, ()]*?)\)(?=[\s,)])')),
    ('nocreateundeforpoison', re.compile(r'\s+nocreateundeforpoison\b')),
    ('initializes(...)', re.compile(r'\s+initializes\(\([^)]*\)\)')),
    ('samesign', re.compile(r'\bicmp samesign\b')),
    ('memory(target_mem...)', re.compile(r',\s*target_mem\d*:\s*\w+')),
    ('nuw gep', re.compile(r'getelementptr inbounds nuw\b')),
    ('nusw gep', re.compile(r'getelementptr nusw\b')),
    ('noalias.addrspace md', re.compile(r', !noalias\.addrspace ![0-9]+')),
    ('dead_on_return', re.compile(r'\s+dead_on_return\b')),
]
REPL = {'samesign': 'icmp', 'nuw gep': 'getelementptr inbounds', 'nusw gep': 'getelementptr'}
src = open(sys.argv[1]).read()
# LLVM 24 spells a float constant as f0x<8 hex> (its own bits); LLVM 19 wants
# the value widened to a double and printed as 0x<16 hex>. Exact (f32 -> f64).
import struct
def _f32(m):
    v = struct.unpack('<f', struct.pack('<I', int(m.group(1), 16)))[0]
    return '0x%016X' % struct.unpack('<Q', struct.pack('<d', v))[0]
src, k = re.subn(r'\bf0x([0-9A-Fa-f]{8})\b', _f32, src)
if k:
    print(f"rule f0x-float-literal: {k}", file=sys.stderr)
if re.search(r'\b(?:h|bf|f)0x[0-9A-Fa-f]+', src):
    sys.exit("downgrade_ll: unhandled typed hex float literal")
for name, rx in RULES:
    src, k = rx.subn(REPL.get(name, ''), src)
    if k:
        print(f"rule {name}: {k}", file=sys.stderr)
open(sys.argv[2], 'w').write(src)
