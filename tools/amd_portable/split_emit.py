"""Split probe_emit output into <kernel>.<target>.<kind>.{ll,s} files."""
import sys, pathlib
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cur, buf = None, []
def flush():
    if cur:
        k, t, kind = cur
        ext = 's' if kind == 'asm' else 'll'
        (out / f"{k}.{t}.{kind}.{ext}").write_text(''.join(buf))
for line in src.read_text().splitlines(keepends=True):
    if line.startswith('=== '):
        flush(); cur = tuple(line.split()[1:4]); buf = []
    else:
        buf.append(line)
flush()
print(len(list(out.iterdir())), "files")
