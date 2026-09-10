#!/bin/bash
set -eu
cd /root/attn
python3 /root/jobs/attn-public/grid_torch.py > /root/jobs/attn-public/torch_fp32.log 2>&1
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' > /root/jobs/attn-public/torch_version.log
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/jobs/attn-public/gpu.log
