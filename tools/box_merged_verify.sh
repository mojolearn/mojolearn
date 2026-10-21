#!/bin/bash
# tools/box_merged_verify.sh nvidia|amd <sha> -- the post-build GPU check of one
# commit on one rented box (2026-10-21): the merged bindings built from the tree
# (fast, identical, host), `python -m mojolearn verify` in identical mode, then
# the incremental reference-column record of the lanes whose source closure
# changed (ONLY_CHANGED=table tools/record_identity_column.sh), so a new verify
# lane gets its NVIDIA/AMD digests without re-fitting every row. Everything
# logs under /root/out; the Mac pulls that folder, so a teardown loses nothing.
set -u; V=$1; SHA=$2; O=/root/out; mkdir -p $O; exec >> $O/job.log 2>&1
st() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a $O/status.txt; }
st "START $V $SHA"
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
command -v pixi >/dev/null || { curl -fsSL https://pixi.sh/install.sh | bash; export PATH=/root/.pixi/bin:$PATH; }
cd /root/mojolearn 2>/dev/null || { cd /root && git clone -q https://github.com/mojolearn/mojolearn.git mojolearn && cd mojolearn; }
git fetch -q origin && git reset -q --hard && git checkout -q -f --detach $SHA
case "$(git rev-parse HEAD)" in $SHA*) st "tree on $SHA";; *) st "TREE NOT ON $SHA; stop"; st DONE; exit 3;; esac
pixi install > $O/pixi.log 2>&1; st "pixi rc=$?"
PY=$(pixi run -q python -c "import sys; print(sys.executable)")
if [ $V = amd ]; then ARCHENV="MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_TARGET_COLUMN=amd"; else
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
  ARCHENV="MOJOLEARN_GPU_ARCHS=sm_$CC MOJOLEARN_TARGET_COLUMN=nvidia"
fi
st "arch env: $ARCHENV ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)$(rocm-smi --showproductname 2>/dev/null | grep -m1 -o 'MI3[0-9A-Z]*'))"
find python/mojolearn -name "*.so" -delete
command -v /usr/bin/time >/dev/null || apt-get install -y -qq time > /dev/null 2>&1
mb() { tag=$1; shift; /usr/bin/time -v env "$@" pixi run -q sh bindings/merged_build.sh ${tag%%-*} > $O/build-$tag.log 2> $O/build-$tag.time; rc=$?
  st "BUILD $tag rc=$rc seconds=$(grep -oP 'Elapsed \(wall clock\) time.*: \K.*' $O/build-$tag.time) peak_rss_gb=$(awk '/Maximum resident/ {printf "%.1f", $6/1048576}' $O/build-$tag.time) errors=$(grep -c ': error:' $O/build-$tag.log)"; }
mb gpu-identical $ARCHENV MOJOLEARN_NUMERIC_MODE=identical &
mb gpu-fast $ARCHENV MOJOLEARN_NUMERIC_MODE=fast &
mb host MOJOLEARN_TARGET_COLUMN=cpu &
wait
find python/mojolearn -name "_mojolearn_native*.so" -exec ls -l {} \; | awk '{print $5, $9}' > $O/merged-files.txt; st "merged files: $(wc -l < $O/merged-files.txt)"
export PYTHONPATH=$PWD/python
( cd / && MOJOLEARN_NUMERIC_MODE=identical $PY -m mojolearn verify > $O/verify-identical.log 2>&1; st "VERIFY identical rc=$? $(grep -m1 '^RESULT' $O/verify-identical.log | cut -c1-200)" )
( cd / && MOJOLEARN_NUMERIC_MODE=fast $PY -m mojolearn verify > $O/verify-fast.log 2>&1; st "VERIFY fast rc=$? $(grep -m1 '^RESULT' $O/verify-fast.log | cut -c1-200)" )
ONLY_CHANGED=table REPEATS=1 bash tools/record_identity_column.sh $V $O/column > $O/record.log 2>&1; st "RECORD rc=$? $(ls $O/column 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
st DONE
