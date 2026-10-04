"""Extract the AMDGPU LLVM IR modules a dump_llvm EXTRACTION build embeds as
text, keyed by kernel symbol, into <out>/<kernel>.ll. Usage: <binary> <out>"""
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n = 0
for m in re.finditer(rb'; ModuleID = [^\0]*?target triple = "amdgcn-amd-amdhsa"[^\0]*', data):
    text = m.group(0).decode()
    ks = re.findall(r'define [^@\n]*amdgpu_kernel void @([A-Za-z0-9_$.]+)', text)
    if len(ks) != 1:
        print(f"REFUSE module at {m.start()}: {len(ks)} kernels"); continue
    (out / f"{ks[0]}.ll").write_text(text); n += 1
    print(f"IR {ks[0]} {len(text)} bytes")
print(f"{n} modules")
