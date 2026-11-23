# SPDX-License-Identifier: Apache-2.0
"""tools/block_tables_check.py: the per-layer tables of autograd/model_blocks.mojo agree for every block layer code
(lane mamba2-nan, 2026-11-23). Source reading only; no build.

The live source must be consistent, and the check must fail on the three lines merge 9e745eb1a8 dropped from the
ssd_scan rows (an empty block_tensor_dims branch, the `.D` name, the D init) and on a missing forward rule."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("block_tables_check", ROOT / "tools/block_tables_check.py")
btc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(btc)
SOURCE = (ROOT / "autograd/model_blocks.mojo").read_text(encoding="utf-8")


def _check(tmp_path, text):
    path = tmp_path / "model_blocks.mojo"
    path.write_text(text, encoding="utf-8")
    return btc.check(path)


def test_live_source_is_consistent():
    problems, table = btc.check()
    assert problems == []
    # every code from BLOCK_FIRST to BLOCK_LAST is read, and the scan layers carry their tensors
    values = btc.constants(SOURCE)
    assert len(table) == values["BLOCK_LAST"] - values["BLOCK_FIRST"] + 1
    # (sizes, names, inits, the appends of the dims branch: a shape's axes, 0 without a branch)
    assert table["LAYER_SSD_SCAN"][:3] == (2, 2, 2) and table["LAYER_SSD_SCAN"][3] > 0
    assert table["LAYER_SELECTIVE_SCAN"][:3] == (2, 2, 2) and table["LAYER_SELECTIVE_SCAN"][3] > 0
    assert table["LAYER_BATCH_NORM"][:3] == (4, 4, 4)
    assert table["LAYER_SILU"] == (0, 0, 0, 0)


def _drop(text, function, needle):
    """`text` with the first line containing `needle` inside `function` removed."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {function}"))
    k = next(i for i in range(start, len(lines)) if needle in lines[i])
    return "".join(lines[:k] + lines[k + 1:])


def test_a_dropped_name_is_caught(tmp_path):
    text = _drop(SOURCE, "block_param_names", 'out.append(".D")')
    problems, _ = _check(tmp_path, text)
    assert any("parameter sizes" in p for p in problems), problems


def test_a_dropped_init_is_caught(tmp_path):
    lines = SOURCE.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("def block_init_bounds"))
    head = next(i for i in range(start, len(lines)) if "code == LAYER_SSD_SCAN" in lines[i])
    k = next(i for i in range(head, len(lines)) if "out.append(Float32(-2))" in lines[i])
    problems, _ = _check(tmp_path, "".join(lines[:k] + lines[k + 1:]))
    assert any("LAYER_SSD_SCAN" in p and "inits" in p for p in problems), problems


def test_an_empty_dims_branch_is_caught(tmp_path):
    lines = SOURCE.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("def block_tensor_dims"))
    head = next(i for i in range(start, len(lines)) if "code == LAYER_SSD_SCAN" in lines[i])
    body = lines[head + 1]
    assert "out.append(" in body
    text = "".join(lines[:head + 1] + [body.split("out.append(")[0] + "pass\n"] + lines[head + 2:])
    problems, _ = _check(tmp_path, text)
    assert "block_tensor_dims: the LAYER_SSD_SCAN branch is empty" in problems


def test_a_missing_forward_is_caught(tmp_path):
    text = SOURCE.replace("    if code == LAYER_SSD_SCAN:  # Mamba-2 (lane neural-blocks-3)\n        var wide",
                          "    if code == -43:\n        var wide", 1)
    assert text != SOURCE
    problems, _ = _check(tmp_path, text)
    assert "block_forward: no rule for LAYER_SSD_SCAN" in problems


def test_the_command_exits_zero():
    assert btc.main(["block_tables_check"]) == 0
