#!/bin/sh
# tools/nvidia_step_time/geom.sh -- lane/nvidia-step-time: GEMM tile geometry
# and register-budget trials at the T3 shapes on the box. `geom.sh build`
# compiles every variant (CPU only; safe beside a running replay); `geom.sh
# run` prices them (GPU) against ab/<ref>.hashes (AB_REF, default noadm).
set -u
cd /root/mojolearn || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
O=/root/gemm_leg_out/nv-step-time/ab
B=/root/nv_bin
mkdir -p $O
VARS=${VARS:-"c4:MOJOLEARN_GEMM_KPACK_CPT4 c4l:MOJOLEARN_GEMM_KPACK_CPT4,MOJOLEARN_GEMM_LB512 r4l:MOJOLEARN_GEMM_KPACK_RPT4,MOJOLEARN_GEMM_LB512 r4c4l:MOJOLEARN_GEMM_KPACK_RPT4,MOJOLEARN_GEMM_KPACK_CPT4,MOJOLEARN_GEMM_LB512"}
case "${1:-}" in
build)
    for v in $VARS; do
        tag=${v%%:*}; defs=$(echo "${v#*:}" | tr ',' '\n' | sed 's/^/-D /; s/$/=1/' | tr '\n' ' ')
        ( pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA $defs --target-accelerator sm_90a -I . \
            bench/gemm_excp_ab_main.mojo -o $B/ab_$tag > $O/$tag.build.log 2>&1; echo "build $tag exit=$?"
          sh tools/nvidia_step_time/session.sh ptx $tag $(echo "${v#*:}" | tr ',' '\n' | sed 's/$/=1/') > /dev/null 2>&1
          echo "ptx $tag: $(grep -h 'registers\|spill' /root/gemm_leg_out/nv-step-time/ptx/$tag/*.ptxas.txt | grep -o 'Used [0-9]* registers\|[0-9]* bytes spill stores' | tr '\n' ' ')" ) &
    done
    wait ;;
run)
    for v in $VARS; do
        tag=${v%%:*}
        MOJOLEARN_EXCP_AB_KINDS=ordinary,tiny,mixed $B/ab_$tag > $O/$tag.log 2>&1
        grep '^EXCP_AB call' $O/$tag.log | sed 's/ ms=.*//' > $O/$tag.hashes
        if cmp -s $O/${AB_REF:-noadm}.hashes $O/$tag.hashes; then r=IDENTICAL; else r=DIFFER; fi
        echo "ab $tag vs_${AB_REF:-noadm}=$r $(grep '^EXCP_AB call' $O/$tag.log | grep ordinary | sed 's/.*call=\([a-zA-Z_]*\).* ms=\([0-9.]*\).*/\1=\2/' | tr '\n' ' ')"
    done ;;
esac
