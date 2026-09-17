# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""bench/model/diff.py over synthetic records: IDENTICAL, DIVERGENT, MOVED,
ONE-COLUMN and REQUIRE FAIL read as they must, a torch record is refused
by --diff, and --ratio prints the mandated wording (ours over the
incumbent's fast default, "identical mode takes X times the incumbent's
time", never the word faster, no dash characters) with the box named.

    pixi run -e test test-model-diff     (RUN OWED; nothing here was run on this Mac)
"""
import importlib.util
import json
import os

import pytest

def row(out, key):
    """The table row of `out` whose first cell is `key` (whitespace agnostic)."""
    for line in out.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] == key:
            return line
    raise AssertionError(f"no row for {key} in:\n{out}")


HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("model_diff", os.path.join(HERE, "..", "diff.py"))
diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diff)

H = {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64, "d": "d" * 64}


def cell(ids="a", logits="b", pre=2.0, dec=4.0, verdict="STABLE"):
    if verdict == "REFUSED":
        return dict(verdict="REFUSED", error="refused by name", n_prompt_tokens=3)
    return dict(n_prompt_tokens=8, n_generated=16, ids_sha256=H[ids], ids_sha256_runs=[H[ids]] * 3,
                first_logits_sha256=H[logits], text="x", prefill_ms_runs=[pre * 8] * 3,
                decode_ms_runs=[dec * 16] * 3, prefill_ms_per_token=pre, prefill_ms_per_token_range=[pre, pre],
                decode_ms_per_token=dec, decode_ms_per_token_range=[dec, dec], verdict=verdict)


def record(path, column, arms, kind="ours", box=None, model="m" * 64, prompts="p" * 64, mode="identical"):
    j = dict(schema="mojolearn.model_leg.v1", kind=kind, column=column, stamp_utc="2026-09-17T00:00:00Z",
             commit="e13177f8a", box=box or dict(hostname="box1", gpu="nvidia: NVIDIA H100 80GB HBM3", cpu_model="x"),
             model=dict(id="HuggingFaceTB/SmolLM2-360M", config_sha256=model, weights_sha256="w" * 64),
             protocol=dict(max_new=64, prompts_sha256=prompts, runs=3),
             library=(dict(numeric_mode=mode, vendor="cuda") if kind == "ours"
                      else dict(torch="2.4.1+cu124", transformers="4.56.2", device="cuda", dtype="bfloat16")),
             arms={a: dict(prompts=p) for a, p in arms.items()}, complete=True)
    with open(path, "w") as fh:
        json.dump(j, fh)
    return path


def test_identical_three_columns(tmp_path, capsys):
    arms = {"float32": {"p01": cell(), "p02": cell("c", "d")}, "int8": {"p01": cell("b", "a")}}
    paths = [record(str(tmp_path / f"{n}.json"), n, arms) for n in ("apple-m4", "nvidia-h100", "amd-mi325x")]
    assert diff.main(["--diff", *paths, "--require-columns", "3"]) == 0
    out = capsys.readouterr().out
    assert out.count("IDENTICAL x3") == 3
    assert "summary: IDENTICAL=3" in out
    assert "summary (float32): IDENTICAL=2" in out
    assert "summary (int8): IDENTICAL=1" in out
    assert "require-columns 3: OK" in out
    assert "DIVERGENT" not in out


def test_divergent_names_which_hash(tmp_path, capsys):
    a = record(str(tmp_path / "a.json"), "apple-m4", {"float32": {"p01": cell(), "p02": cell("c", "d")}})
    b = record(str(tmp_path / "b.json"), "nvidia-h100", {"float32": {"p01": cell(), "p02": cell("c", "a")}})
    assert diff.main(["--diff", a, b]) == 1
    out = capsys.readouterr().out
    assert "IDENTICAL x2" in row(out, "float32/p01")
    assert "DIVERGENT" in row(out, "float32/p02")
    assert "first logits differ, ids agree" in row(out, "float32/p02")
    assert "summary: DIVERGENT=1, IDENTICAL=1" in out


def test_one_column_and_require(tmp_path, capsys):
    a = record(str(tmp_path / "a.json"), "apple-m4", {"bfloat16": {"p01": cell()}})
    assert diff.main(["--diff", a]) == 0
    out = capsys.readouterr().out
    assert "ONE-COLUMN" in out and "summary: ONE-COLUMN=1" in out
    assert diff.main(["--diff", a, "--require-columns", "2"]) == 1
    out = capsys.readouterr().out
    assert "REQUIRE FAIL bfloat16/p01: ONE-COLUMN rests on 1 real hash(es)" in out
    assert "REQUIRE FAIL: --require-columns 2 with 1 JSONs given" in out


def test_moved_and_refused(tmp_path, capsys):
    a = record(str(tmp_path / "a.json"), "apple-m4",
               {"float32": {"p01": cell(verdict="MOVED"), "p02": cell(verdict="REFUSED"), "p03": cell()}})
    b = record(str(tmp_path / "b.json"), "amd-mi325x",
               {"float32": {"p01": cell(), "p02": cell(verdict="REFUSED"), "p03": cell()}})
    assert diff.main(["--diff", a, b]) == 1
    out = capsys.readouterr().out
    assert "MOVED" in row(out, "float32/p01")
    assert "REFUSED" in row(out, "float32/p02")
    assert "IDENTICAL x2" in row(out, "float32/p03")


def test_a_non_identical_column_is_named(tmp_path, capsys):
    a = record(str(tmp_path / "a.json"), "apple-m4", {"float32": {"p01": cell()}}, mode="fast")
    diff.main(["--diff", a])
    assert "READ BACK numeric_mode='fast'; it is not an identical column" in capsys.readouterr().out


def test_diff_refuses_torch_and_other_model(tmp_path):
    a = record(str(tmp_path / "a.json"), "apple-m4", {"float32": {"p01": cell()}})
    t = record(str(tmp_path / "t.json"), "nvidia-h100-torch", {"torch-bf16-shipped": {"p01": cell()}}, kind="torch")
    with pytest.raises(SystemExit, match="never for bits"):
        diff.main(["--diff", a, t])
    o = record(str(tmp_path / "o.json"), "nvidia-h100", {"float32": {"p01": cell()}}, model="z" * 64)
    with pytest.raises(SystemExit, match="different models"):
        diff.main(["--diff", a, o])


def test_ratio_wording(tmp_path, capsys):
    ours = record(str(tmp_path / "ours.json"), "nvidia-h100-sm_90a",
                  {"float32": {"p01": cell(pre=2.0, dec=4.0), "p02": cell(pre=3.0, dec=6.0), "a01": cell(verdict="REFUSED")}})
    torch = record(str(tmp_path / "torch.json"), "nvidia-h100-torch",
                   {"torch-bf16-shipped": {"p01": cell(pre=1.0, dec=2.0), "p02": cell(pre=1.0, dec=1.5), "a01": cell()},
                    "torch-deterministic": {"p01": cell(pre=1.5, dec=3.0), "p02": cell(pre=1.5, dec=3.0), "a01": cell()}},
                   kind="torch")
    assert diff.main(["--ratio", ours, torch]) == 0
    out = capsys.readouterr().out
    assert ("identical mode (float32) takes 2.50 times the incumbent's time (torch-bf16-shipped) for prefill "
            "on nvidia: NVIDIA H100 80GB HBM3: median over 2 prompts, range 2.00 to 3.00") in out
    assert ("identical mode (float32) takes 3.00 times the incumbent's time (torch-bf16-shipped) for decode "
            "on nvidia: NVIDIA H100 80GB HBM3: median over 2 prompts, range 2.00 to 4.00") in out
    assert "excluded prompts (not STABLE on both columns, or no time): a01" in out
    assert ("the incumbent's own determinism setting (torch-deterministic) takes 1.50 times its fast default's "
            "time (torch-bf16-shipped) for prefill") in out
    assert "faster" not in out.lower()
    assert "—" not in out and "–" not in out
    assert "pending" not in out.lower()


def test_ratio_names_a_box_mismatch(tmp_path, capsys):
    ours = record(str(tmp_path / "ours.json"), "apple-m4", {"float32": {"p01": cell()}},
                  box=dict(hostname="mac", gpu="apple: Apple M4", cpu_model="Apple M4"))
    torch = record(str(tmp_path / "torch.json"), "h100-torch", {"torch-bf16-shipped": {"p01": cell()}}, kind="torch")
    diff.main(["--ratio", ours, torch])
    out = capsys.readouterr().out
    assert "the two records name different boxes" in out
    assert "for prefill on apple: Apple M4" in out
