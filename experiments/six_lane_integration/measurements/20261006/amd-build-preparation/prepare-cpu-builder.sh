#!/bin/bash
# Reproducible Linux CPU environment preparation for missing native gfx942 builds.
# No model execution or numerical verification; cpu_build_guard remains unchanged.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential curl ca-certificates git rsync
# Ordinary DO CPU VMs expose a virtio-pci display render node. Isolate device
# access rather than relaxing the unchanged CPU-only guard. The build still
# explicitly targets gfx942. systemd services need the actual root HOME for pixi.
systemd-run --unit=amd-missing-build --setenv=HOME=/root \
  --property=PrivateDevices=yes \
  --property=StandardOutput=append:/root/leg_out/pipeline.log \
  --property=StandardError=append:/root/leg_out/pipeline.log \
  /bin/bash /root/box_script.sh
