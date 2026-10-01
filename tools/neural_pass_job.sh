#!/bin/bash
# One box, one build (released 0.8.33 + this tree's Python and neural/linalg bindings): the GEMM knob section
# (AMD: confirm #55's 1024 default; NVIDIA: sweep MOJOLEARN_GEMM_TILE_MIN_BLOCKS and MOJOLEARN_GEMM_SPLIT_CELLS),
# then the attention arm sweep on a -D MOJOLEARN_ATTN_ARM_TRIAL=1 byte_lm build (variants of the column default),
# and on AMD a -D MOJOLEARN_ATTN_FWD_MFMA=1 trial build. Digest 4a8e781b0739a038 and the losses must hold.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/neural-pass-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; PLUG=mojolearn-nvidia; TORCH="torch==2.13.0"; TIDX=https://download.pytorch.org/whl/cpu; else BK=hip; AR=gfx942; PLUG=mojolearn-amd; TORCH="torch==2.13.0"; TIDX=https://download.pytorch.org/whl/cpu; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
so_of() { ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1; }
bld() { rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so; MOJOLEARN_MOJO_BUILD_FLAGS="${3:-}" bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?; s=$(so_of $1); [ -n "$s" ] && cp $s $O/$2.so && sha256sum $O/$2.so >> $O/bindings.sha256; }
for m in transformer byte_lm training mamba linalg; do bld $m $m; done
bld byte_lm byte_lm-trial "-D MOJOLEARN_ATTN_ARM_TRIAL=1"
[ $V = amd ] && bld byte_lm byte_lm-mfma "-D MOJOLEARN_ATTN_ARM_TRIAL=1 -D MOJOLEARN_ATTN_FWD_MFMA=1"
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.33 $PLUG==0.8.33 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install $TORCH --index-url $TIDX >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$(dirname $($P -c "import mojolearn,os,glob;print(glob.glob(os.path.join(os.path.dirname(mojolearn.__file__),'..','*','$BK','$AR','identical','_mojolearn_byte_lm.so'))[0] if glob.glob(os.path.join(os.path.dirname(mojolearn.__file__),'..','*','$BK','$AR','identical','_mojolearn_byte_lm.so')) else os.path.join(os.path.dirname(mojolearn.__file__),'$BK','$AR','identical','x'))" 2>/dev/null))
echo "binding dir $T" > $O/target.txt
cp $S/_version.py /tmp/vnp.py; cp python/mojolearn/*.py $S/; cp /tmp/vnp.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
inst() { cp $O/$1.so $T/_mojolearn_$2.so; }
for m in transformer byte_lm training mamba linalg; do inst $m $m; done
st() { env $2 timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --calls 8 > $O/stage-$1.log 2>&1; rc stage-$1 $?; }
race() { env $3 timeout 3600 $P tools/bench_board_neural.py race --lane $1 --shape full --arms ours --rounds 3 --out $O/race-$1-$2 --work $O/work --ours-python $P > $O/race-$1-$2.log 2>&1; rc race-$1-$2 $?; }
gck() { env $2 timeout 3600 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo > $O/gemm-check-$1.log 2>&1; rc gemm-check-$1 $?; }
if [ $V = amd ]; then
  st default1 X=1; st default2 X=1; st tile0 MOJOLEARN_GEMM_TILE_MIN_BLOCKS=0
  race gemm default X=1; race gemm tile0 MOJOLEARN_GEMM_TILE_MIN_BLOCKS=0; gck default X=1
else
  st default X=1
  for t in 256 512 1024; do st tile$t MOJOLEARN_GEMM_TILE_MIN_BLOCKS=$t; done
  for c in 524288 1048576; do st split$c MOJOLEARN_GEMM_SPLIT_CELLS=$c; done
  for t in 256 1024; do for c in 524288 1048576; do st tile$t-split$c "MOJOLEARN_GEMM_TILE_MIN_BLOCKS=$t MOJOLEARN_GEMM_SPLIT_CELLS=$c"; done; done
  race gemm default X=1; race gemm tile1024 MOJOLEARN_GEMM_TILE_MIN_BLOCKS=1024
  gck tile1024-split1M "MOJOLEARN_GEMM_TILE_MIN_BLOCKS=1024 MOJOLEARN_GEMM_SPLIT_CELLS=1048576"
fi
# attention arm sweep on the trial byte_lm build
inst byte_lm-trial byte_lm
D=$($P -c "from mojolearn._backend import binding; print(binding('_mojolearn_byte_lm','identical').byte_lm_attention_arm()[1])" 2> $O/arm-default.err | tail -1)
echo "default arm: $D" > $O/arms.txt
$P - "$D" > $O/arm-variants.txt <<'PY'
import sys
d = sys.argv[1]; out = {"default": d}
def sub(a, b):
    return d.replace(a, b) if a in d else None
cands = {
    "fgrid_r64": sub("fgrid_r32", "fgrid_r64"),
    "kvgrid_r64": sub("kvgrid_r32", "kvgrid_r64"),
    "kvsplit": sub("kvgrid_r32", "kvsplit"),
    "kvrecompute": sub("kvgrid_r32", "kvrecompute"),
    "zdefer": d + "_zdefer",
    "zlag": d + "_zlag",
    "bswz_toggle": d.replace("_bswz", "") if "_bswz" in d else d + "_bswz",
    "no_dres": sub("_dres", ""),
}
for k, v in cands.items():
    if v and v != d:
        out[k] = v
for k, v in out.items():
    print(k, v)
PY
cat $O/arm-variants.txt >> $O/arms.txt
while read name word; do st arm-$name MOJOLEARN_ATTN_ARM=$word; done < $O/arm-variants.txt
if [ $V = amd ]; then inst byte_lm-mfma byte_lm; st mfma-default X=1; fi
inst byte_lm byte_lm
echo done > $O/done
