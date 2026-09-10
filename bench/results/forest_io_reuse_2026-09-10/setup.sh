#!/bin/bash
set -euo pipefail
mkdir -p /root/forest_out
exec > /root/forest_out/setup.log 2>&1
trap 'echo "setup_exit=$?" > /root/forest_out/setup.exit' EXIT
tar -xzf /root/source.tar.gz -C /root/mojolearn
cd /root/mojolearn
printf "%s\n" 591c82a9b541121b4fed845dbf42877b47e1fae6 > SHIPPED_COMMIT.txt
curl -fsSL https://pixi.sh/install.sh | bash
export PATH=/root/.pixi/bin:$PATH
pixi install --frozen
python3 -m pip install --disable-pip-version-check numpy pandas scikit-learn catboost 'cuml-cu12==26.8.0' --extra-index-url https://pypi.nvidia.com
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/forest_out/gpu.txt
pixi run mojo --version > /root/forest_out/toolchain.txt
python3 -c 'import numpy, cupy, cuml; print(numpy.__version__,cupy.__version__,cuml.__version__)' > /root/forest_out/python-versions.txt
touch /root/forest_out/setup.done
