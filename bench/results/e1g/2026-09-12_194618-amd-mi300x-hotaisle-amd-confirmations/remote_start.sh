set -u
rm -f /root/gemm_leg.done /root/gemm_leg_console.log
mkdir -p /root/gemm_leg_out /root/mojolearn
RT="docker"
if [ "$RT" = docker ] || [ "$RT" = podman ]; then
    "$RT" rm -f mojolearn-leg > /dev/null 2>&1
    setsid nohup sh -c '"$0" run --rm --name mojolearn-leg --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --ipc=host --network host \
        -e HOME=/root -v /root:/root -w /root/mojolearn rocm/dev-ubuntu-22.04:6.4.1-complete \
        timeout -k 30 3212 sh /root/gemm_leg.sh; rc=$?; "$0" rm -f mojolearn-leg > /dev/null 2>&1; \
        mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' "$RT" \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
else
    setsid nohup sh -c 'timeout -k 30 3212 sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
fi
echo "REMOTE_PID=$!"
