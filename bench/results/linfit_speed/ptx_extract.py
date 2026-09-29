"""Extract the PTX modules a Mojo .so embeds (null-terminated text), write
each to OUT/k<i>.ptx, and name the .entry kernels in each."""
import re, sys, os
so, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
b = open(so, "rb").read()
i = 0
k = 0
while True:
    j = b.find(b".version", i)
    if j < 0:
        break
    s = b.rfind(b"\x00", 0, j) + 1
    e = b.find(b"\x00", j)
    txt = b[s:e].decode("utf-8", "replace")
    names = re.findall(r"\.entry\s+(\S+)\(", txt)
    if names:
        p = os.path.join(out, "k%d.ptx" % k)
        open(p, "w").write(txt)
        print(p, len(txt), [n[:120] for n in names])
        k += 1
    i = e + 1
