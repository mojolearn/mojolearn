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
    # (with the codes the lanes' own modules define and the never-assigned BLOCK_UNUSED range: lane nn-integrate)
    delegated = {v for v in btc.delegated_values() if values["BLOCK_FIRST"] <= v <= values["BLOCK_LAST"]}
    assert len(table) + len(delegated) + len(btc.unused_codes(values)) == (
        values["BLOCK_LAST"] - values["BLOCK_FIRST"] + 1)
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


# ------------------------------------------------------------------ every file (lane layer-tables, 2026-11-23)
def _files(**changed):
    """A reader of the live tree with some files' text replaced: {relative path: text}."""
    live = btc.reader(ROOT)
    return lambda rel: changed[rel] if rel in changed else live(rel)


def _live(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_every_file_agrees():
    problems, table = btc.check_files(btc.reader(ROOT))
    assert problems == []
    assert "LAYER_BATCH_NORM" in table


def test_the_literal_code_list_of_load_state_dict_is_caught():
    # the defect found by lane nn-activations: codes 42 to 46 missing from sequential_parameter_layers
    text = _live(btc.WEIGHTS).replace("        elif is_block(code):\n",
                                      "        elif code == 35 or code == 39 or code == 40 or code == 41:\n", 1)
    assert text != _live(btc.WEIGHTS)
    problems, _ = btc.check_files(_files(**{btc.WEIGHTS: text}))
    assert any("sequential_parameter_layers" in p and "literal 35" in p for p in problems), problems


def test_a_code_without_a_constructor_is_caught():
    text = _live(btc.FIXTURES).replace("    'lm_head':", "    'lm_head_gone':", 1)
    problems, _ = btc.check_files(_files(**{btc.FIXTURES: text}))
    assert any("no constructor for LAYER_LM_HEAD" in p for p in problems), problems


def test_a_code_without_a_layer_name_is_caught():
    text = _live(btc.PYMODEL).replace('"trapezoid_scan", "lm_head", "batch_norm",', '"trapezoid_scan", "lm_head", "",', 1)
    assert text != _live(btc.PYMODEL)
    problems, _ = btc.check_files(_files(**{btc.PYMODEL: text}))
    assert any("layer_names: no name for LAYER_BATCH_NORM" in p for p in problems), problems


def test_a_name_without_an_add_branch_is_caught():
    text = _live(btc.DESCRIPTION).replace('elif kind == "lm_head":', 'elif kind == "lm_head_gone":', 1)
    problems, _ = btc.check_files(_files(**{btc.DESCRIPTION: text}))
    assert any("no branch for 'lm_head'" in p for p in problems), problems


def test_a_new_block_code_must_appear_everywhere():
    text = _live(btc.BLOCKS)
    last = btc.constants(text)["BLOCK_LAST"]
    text = text.replace(f"comptime BLOCK_LAST = {last}", f"comptime LAYER_NEW = {last + 1}\ncomptime BLOCK_LAST = {last + 1}", 1)
    problems, _ = btc.check_files(_files(**{btc.BLOCKS: text}))
    for needle in ("block_shape: no rule for LAYER_NEW", "_LAYER: no name for LAYER_NEW",
                   "layer_names: no name for LAYER_NEW", "no constructor for LAYER_NEW"):
        assert any(needle in p for p in problems), (needle, problems)


def test_the_tree_mode_reads_a_commit():
    problems, table = btc.check_files(btc.reader(ROOT, "HEAD"))
    assert table and isinstance(problems, list)


def test_a_delegated_module_is_checked():
    # lane nn-integrate: codes whose tables live in their own module (autograd/nn_conv_blocks.mojo) are counted,
    # and a name dropped there is caught as in model_blocks' own tables
    _, table = btc.check_files(btc.reader(ROOT))
    assert table["LAYER_CONV_TRANSPOSE2D"][:3] == (2, 2, 2) and table["LAYER_CONV_TRANSPOSE2D"][3] > 0
    assert table["LAYER_UPSAMPLE"] == (0, 0, 0, 0)
    rel = "autograd/nn_conv_blocks.mojo"
    text = _live(rel).replace('        out.append(".bias")\n', "", 1)
    assert text != _live(rel)
    problems, _ = btc.check_files(_files(**{rel: text}))
    assert any("LAYER_CONV_TRANSPOSE2D" in p and "parameter sizes" in p for p in problems), problems
