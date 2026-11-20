# SPDX-License-Identifier: Apache-2.0
"""tools/runpod_pod_lib.sh pod_pip_conf_script: the pip.conf a pod bring-up writes ON the pod, so torch and its
CUDA wheels come from the PyTorch CDN. The script text is composed on the Mac and run here under bash with a fake
nvidia-smi and a temporary pip dir: no pod, no network."""
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = ROOT / "tools" / "runpod_pod_lib.sh"

SMI = """#!/bin/sh
cat <<'EOF'
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
+-----------------------------------------------------------------------------------------+
EOF
"""


def script():
    return subprocess.run(["bash", "-c", '. "$1"; pod_pip_conf_script', "_", str(LIB)],
                          capture_output=True, text=True, check=True).stdout


def run_on_fake_pod(tmp_path, smi=SMI):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if smi is not None:
        (bin_dir / "nvidia-smi").write_text(smi)
        (bin_dir / "nvidia-smi").chmod(0o755)
    env = dict(os.environ, PATH="%s:/usr/bin:/bin" % bin_dir, MOJOLEARN_POD_PIP_DIR=str(tmp_path / "pip"))
    done = subprocess.run(["bash", "-c", script()], capture_output=True, text=True, env=env)
    return done, tmp_path / "pip" / "pip.conf"


def test_the_script_writes_the_root_pip_conf_with_an_extra_index_only():
    text = script()
    assert "/root/.config/pip" in text
    assert "extra-index-url = https://download.pytorch.org/whl/cu%s" in text
    assert "index-url =" not in text.replace("extra-index-url =", "")   # PyPI stays the index


def test_the_cuda_version_from_nvidia_smi_names_the_index(tmp_path):
    done, conf = run_on_fake_pod(tmp_path)
    assert done.returncode == 0 and "POD_PIP_CONF cu128" in done.stdout
    assert conf.read_text() == "[global]\nextra-index-url = https://download.pytorch.org/whl/cu128\n"


def test_no_nvidia_smi_writes_nothing_and_exits_0(tmp_path):
    done, conf = run_on_fake_pod(tmp_path, smi=None)
    assert done.returncode == 0 and "POD_PIP_CONF none" in done.stdout and not conf.exists()


def test_both_bring_ups_run_it():
    for tool in ("nvidia_central.sh", "dev_pod.sh"):
        assert "$(pod_pip_conf_script)" in (ROOT / "tools" / tool).read_text(), tool
