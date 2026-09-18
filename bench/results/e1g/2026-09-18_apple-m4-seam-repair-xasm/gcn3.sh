#!/bin/bash
# Variant V1: branch sources, but the tuned kernel calls _tuned_step (not _tuned_step_admitted).
EV=~/mojolearn-evidence/apple-seam-repair-2026-09-18
. $EV/env.sh
WT=~/mojolearn-wt/apple-seam-xasm; cd $WT
F=gemm/checks/gemm_identical.mojo
cp $EV/xasm/bc/$F /tmp/claude-501/gi_backup.mojo
trap 'cp $EV/xasm/bc/$F $WT/$F' EXIT
for v in ${VARIANTS:-v1}; do
  cp $EV/xasm/bc/$F $F
  case $v in
    v1) sed -i '' 's/= _tuned_step_admitted(/= _tuned_step(/' $F ;;
    v2) python3 - "$F" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("    var emin_a = _EXP_NONE\n    var emin_b = _EXP_NONE\n","    var emin_a = SIMD[DType.uint32, 1 if TUNED_BLOCK_ADMIT else 1](0x7F800000)[0]\n    var emin_b = emin_a\n",1)
open(p,'w').write(s)
PY
    ;;
    v3) git show c7442abed:$F > $F ;;
    v4) python3 - "$F" <<'PY'
import sys,subprocess
p=sys.argv[1]; s=open(p).read()
m=subprocess.run(['git','show','c7442abed:'+p],capture_output=True,text=True).stdout
def ker(t):
    a=t.index("def identical_gemm_tuned_kernel["); b=t.index("\ndef ",a+10); return a,b
a,b=ker(s); ma,mb=ker(m)
s=s[:a]+m[ma:mb]+s[b:]
open(p,'w').write(s)
PY
    ;;
    v5) python3 - "$F" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
blk='''        comptime if TUNED_BLOCK_ADMIT:
            # The window's operand words, exactly once each (see
            # `TUNED_BLOCK_ADMIT`). Unused slots are +0.0 and constrain nothing.
            emin_a = _nz_exp_min(pa, emin_a)
            emin_b = _nz_exp_min(pb, emin_b)
'''
assert blk in s; s=s.replace(blk,'',1); open(p,'w').write(s)
PY
    ;;
    v6) python3 - "$F" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
a=s.index("    comptime if TUNED_BLOCK_ADMIT:\n        comptime assert (NTH"); b=s.index("            return\n\n    comptime if SPLIT:",a)
s=s[:a]+s[b+len("            return\n\n"):]; open(p,'w').write(s)
PY
    ;;
  esac
  rm -rf $EV/xasm/gcn-$v; mkdir -p $EV/xasm/gcn-$v
  nice -n 19 $MOJO build -j 1 --emit asm --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64-v3 --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_XASM_NONCE_7=1 -D MOJOLEARN_XASM_$v=1 -I . gemm/checks/gemm_device_check.mojo -o $EV/xasm/gcn-$v/out.s > $EV/xasm/gcn-$v/build.log 2>&1
  echo "$v rc=$? n=$(ls $EV/xasm/gcn-$v/*.amdgcn 2>/dev/null | wc -l)" >> $EV/xasm/gcn3_rc.txt
done
