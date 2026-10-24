# SPDX-License-Identifier: Apache-2.0
"""The framework adapters are whole-model only (Andrew, 2026-10-24): source and docs checks, no torch,
MAX, GPU or built binding needed. The per-operation surfaces (torch.ops.mojolearn.operation, MAX
mojolearn_matmul / mojolearn_relu / mojolearn_operation0..3, the native mojolearn_adapter_operation*
entries) were removed (docs/TOMBSTONES.md) and must not come back."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_per_operation_surfaces_are_gone_from_the_docs():
    # Andrew, 2026-10-24: whole-model only; the removed per-op names have nothing left to exclude
    rows = {line.split("\t", 1)[0] for line in
            (REPO / "docs-site/api_exclude.txt").read_text().splitlines() if "\t" in line and not line.startswith("#")}
    names = ["torch.ops.mojolearn.operation", "max.mojolearn_matmul", "max.mojolearn_relu"] + \
        [f"max.mojolearn_operation{i}" for i in range(4)]
    for name in names:
        assert name not in rows, name
    guide = (REPO / "docs-site/user_guide/framework_adapters.md").read_text()
    for name in ("mojolearn.operation", "mojolearn_matmul", "mojolearn_relu", "mojolearn_operation"):
        assert name not in guide, name
    for name in ('backend="mojolearn"', "torch.ops.mojolearn.train", "mojolearn_train", "import_model"):
        assert name in guide, name
    assert "In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs." in guide


def test_the_adapters_register_whole_model_entries_only():
    # The per-operation surfaces were removed on 2026-10-24 (docs/TOMBSTONES.md) and must not come back:
    # between per-op calls the framework would compute with its own bits.
    cpp = (REPO / "adapters/torch/registration.cpp").read_text()
    assert re.findall(r'm\.def\("(\w+)\(', cpp) == ["train"]
    assert set(re.findall(r'm\.impl\("(\w+)"', cpp)) == {"train"}
    max_ops = (REPO / "adapters/max_ops/__init__.mojo").read_text()
    assert re.findall(r'@extensibility\.register\("(\w+)"\)', max_ops) == ["mojolearn_train", "mojolearn_train_resume"]
    exports = re.findall(r"@export\s+def (\w+)", (REPO / "adapters/native/c_api.mojo").read_text())
    assert exports == ["mojolearn_adapter_abi", "mojolearn_adapter_train"]
    header = (REPO / "adapters/native/abi.h").read_text()
    assert re.findall(r"^int64_t (\w+)\(", header, re.M) == ["mojolearn_adapter_abi", "mojolearn_adapter_train"]
    for path in ("python/mojolearn/adapters/torch.py", "python/mojolearn/adapters/max.py"):
        text = (REPO / path).read_text()
        assert "ops.mojolearn.operation(" not in text and '"mojolearn_operation' not in text, path


def test_the_checkers_and_smoke_name_no_per_operation_tool():
    for path in ("tools/adapter_wheel_smoke.py", "tools/adapter_gate.py", "tools/release_wheel_smoke.sh",
                 "tools/run_training_adapter_checks.sh", "tools/check_native_adapters.py",
                 "packaging/adapters/build.sh"):
        text = (REPO / path).read_text()
        for gone in ("check_max_native_adapters", "check_native_adapter_abi", "check_native_adapter_transport",
                     "check_max_adapter_fault", "adapter_payload", "ops.mojolearn.operation(",
                     "mojolearn_operation0", "mojolearn_matmul"):
            assert gone not in text, (path, gone)
