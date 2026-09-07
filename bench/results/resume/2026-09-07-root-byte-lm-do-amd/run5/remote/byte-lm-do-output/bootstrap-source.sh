#!/usr/bin/env bash
# Root-executed only, on an already guarded DO AMD droplet. No rentals here.
set -euo pipefail
cd /root/mojolearn
OUT=/root/byte-lm-do-output
mkdir "$OUT"
exec > "$OUT/bootstrap.log" 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$OUT/bootstrap-exit.txt"; printf "bootstrap_exit=%s\n" "$status"' EXIT
printf 'bootstrap_started=%s\n' "$(date -u +%FT%TZ)"
uname -a > "$OUT/host.txt"
cat /etc/os-release > "$OUT/os-release.txt"
cp /root/do-byte-lm-remote.sh "$OUT/bootstrap-source.sh"
seconds=${1:?remaining work seconds}
PY=${2:?absolute Python interpreter}
head64=${3:-0}
[[ "$head64" == 0 || "$head64" == 1 ]] || exit 2
[[ "$seconds" =~ ^[0-9]+$ && "$PY" = /* ]] || exit 2
((seconds >= 180 && seconds <= 3000)) || exit 2
started=$(date +%s)
cores=$(/usr/bin/python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
/usr/bin/python3 - <<'PYTOPOLOGY' > "$OUT/drm-topology.json"
import json, os
from pathlib import Path
records = []
for render in sorted(Path('/sys/class/drm').glob('renderD*')):
    device = (render / 'device').resolve()
    chain = []
    for ancestor in [device, *list(device.parents)[:16]]:
        row = {'path': str(ancestor)}
        subsystem = ancestor / 'subsystem'
        row['subsystem'] = str(subsystem.resolve()) if subsystem.is_symlink() else None
        for key in ('vendor', 'device', 'mem_info_vram_total', 'mem_info_vram_used'):
            field = ancestor / key
            if field.is_file():
                with field.open() as stream: row[key] = stream.read(128).strip()
        chain.append(row)
    node = Path('/dev/dri') / render.name
    info = node.stat() if node.exists() else None
    records.append(dict(render=render.name, sysfs_dev=(render / 'dev').read_text().strip(),
                        node_dev=f'{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}' if info else None,
                        ancestry=chain))
print(json.dumps(records, indent=2))
PYTOPOLOGY
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MAX_JOBS=2 MOJOLEARN_COMPILE_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE=1073741824
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=true
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100
export MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_BYTE_LM_EXPECT_VENDOR=hip
export MOJOLEARN_COMMIT=0ba549761ecbb8db9924656d9779c97da9dd0885
export MOJOLEARN_PYTHON="$PY"
printf '%s\n' vendor=amd "commit=$MOJOLEARN_COMMIT" > "$OUT/leg.txt"
/usr/bin/python3 -B tools/training_validation_admit.py --inventory-root /root/mojolearn > "$OUT/source_inventory.json"
# Optional root-reviewed dependency setup, before importing Torch or models.
# Typical DO Ubuntu24 source Python is /usr/bin/python3; root pins official
# cp312 ROCm6.4.1 Torch/Triton wheel URLs in this retained setup script.
if [[ -f /root/do-byte-lm-setup.sh ]]; then
 echo dependency_setup_started
 cp /root/do-byte-lm-setup.sh "$OUT/dependency-setup-source.sh"
 /usr/bin/python3 tools/amd_serial_guard.py --seconds 600 --rss-gib 12 -- \
  bash /root/do-byte-lm-setup.sh > "$OUT/dependency-setup.log" 2>&1
 echo dependency_setup_passed
fi
[[ -x "$PY" ]] || exit 2
# Mandatory image-specific runtime check. A bare ROCm image is not a Torch image.
/usr/bin/python3 tools/amd_serial_guard.py --seconds 45 --rss-gib 12 -- \
 "$PY" -c 'import sys,torch; assert torch.version.hip and not torch.version.cuda; assert torch.cuda.is_available(); print(sys.executable,torch.__version__,torch.version.hip,torch.cuda.get_device_name(0))' > "$OUT/torch-hip-preflight.log" 2>&1
if [[ ! -x /root/.pixi/bin/pixi ]]; then
  curl --fail --location --max-redirs 4 --proto '=https' --proto-redir '=https' \
  --max-time 30 --max-filesize 262144 -sS https://pixi.sh/install.sh \
  -o "$OUT/pixi-installer.sh"
 bash -n "$OUT/pixi-installer.sh"
 /usr/bin/python3 tools/amd_serial_guard.py --seconds 120 --rss-gib 12 -- \
  bash "$OUT/pixi-installer.sh" > "$OUT/pixi-installer.log" 2>&1
fi
export PATH=/root/.pixi/bin:$PATH
/usr/bin/python3 tools/amd_serial_guard.py --seconds 600 --rss-gib 12 -- pixi install --locked
remaining=$((seconds - $(date +%s) + started - 30))
((remaining >= 120)) || exit 124
validation_budget=$remaining
if [[ "$head64" == 1 && "$validation_budget" -gt 1500 ]]; then validation_budget=1500; fi
export MOJOLEARN_BYTE_LM_VALIDATION_SECONDS=$validation_budget
export MOJOLEARN_BYTE_LM_VALIDATION_OUT="$OUT/byte-lm-validation"
# Existing serial helper retains all twelve jobs, raw arrays and root receipts.
# Outer timeout is provided by the controller; helper owns each per-job guard.
echo twelve_job_validation_started
bash tools/byte_lm_validation_serial.sh
echo twelve_job_validation_passed

/usr/bin/python3 -B tools/byte_lm_validation_admit.py "$OUT/byte-lm-validation" > "$OUT/full128-admission.json"
if [[ "$head64" == 1 ]]; then
 remaining=$((seconds - $(date +%s) + started - 30))
 if ((remaining >= 600)); then
  export MOJOLEARN_PYTHON=/root/mojolearn/.byte-lm-validation-venv/bin/python
  export MOJOLEARN_BYTE_LM_RESUME_SECONDS=600
  bash tools/byte_lm_resume_serial.sh head64 "$OUT/byte-lm-validation" "$OUT/head64-campaign"
 else
  printf '%s\n' SKIPPED_DEADLINE > "$OUT/head64-status.txt"
 fi
fi
