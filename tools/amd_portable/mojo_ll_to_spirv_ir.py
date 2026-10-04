"""Route B feasibility: rewrite (downgraded) Mojo amdgcn IR into the
spirv64-amd-amdhsa form ROCm's amdgcnspirv flow produces. Handles ONLY the
address spaces the probe kernels use (AMDGPU generic 0 -> SPIR generic 4,
global 1 -> 1); anything else (private 5, LDS 3, constant 4, buffer
fat pointers 7/8/9) makes it refuse, because remapping those safely needs a
real IR pass, not text edits."""
import re, sys
s = open(sys.argv[1]).read()
bad = sorted(set(re.findall(r'addrspace\(([02-9])\)', s)))
if bad:
    sys.exit(f"refuse: address spaces {bad} need a real remapping pass")
s = re.sub(r'^target datalayout = .*$',
           'target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n32:64-S32-G1-P4-A0"',
           s, flags=re.M)
s = s.replace('target triple = "amdgcn-amd-amdhsa"', 'target triple = "spirv64-amd-amdhsa"')
s = re.sub(r'\bptr\b(?! addrspace)', 'ptr addrspace(4)', s)
s = s.replace('amdgpu_kernel void', 'spir_kernel void')
s = re.sub(r'"target-cpu"="[^"]*"', '', s)
s = re.sub(r'"amdgpu-[a-z-]+"(="[^"]*")?', '', s)
open(sys.argv[2], 'w').write(s)
