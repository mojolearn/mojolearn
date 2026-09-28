import sys,re
head=open("/Users/andrewhendel/mojolearn-wt/cluster-apple3/x_cluster/device_ops.mojo").read()
hn=[l for l in head.split("\n") if l.startswith("from checks.numerics import")][0]
p="x_cluster/device_ops.mojo"
s=open(p).read().split("\n")
out=[]
for l in s:
    if l.startswith("from checks.numerics import"):
        l=hn
    if l.startswith("from std.sys.compile import is_defined"):
        continue
    out.append(l)
    if l=="from std.os import getenv":
        out.append("from std.sys.compile import is_defined")
open(p,"w").write("\n".join(out))
