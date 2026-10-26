"""Admission of active development groups; no native compilation or runtime claims."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binding_stamps as stamps
import development_build as dev
from test_development_build import repo, cache_plan, ENV, FACTS, C_FACTS, A  # noqa: F401


@pytest.fixture
def activation(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(dev.merged, "default_platform", lambda: "linux")
    cache = tmp_path / "cache"
    # Build and verification controllers intentionally use different Pythons.
    with monkeypatch.context() as producer:
        producer.setattr(sys, "executable", "/build/controller/python")
        plan = dev.make_plan(repo, cache, ["identical", "host"], ENV)
    cache_plan(cache, plan)
    for row in plan["groups"] + plan["support"]:
        path = cache / "artifacts" / row["key"] / "receipt.json"
        saved = json.loads(path.read_text())
        saved["state"] = "built"
        dev.atomic_json(path, saved)
    assert dev.activate(plan, repo, cache)["state"] == "activated"
    return repo, cache, plan


def check(activation):
    root, cache, _ = activation
    return stamps.cmd_check(True, root / "python/.binding-stamps", root / "python/mojolearn", root, cache)


def index_path(root):
    return root / "python/mojolearn/identical/_mojolearn_merged.json"


def test_active_receipts_admit_with_different_test_python_and_ignore_inactive_history(activation):
    root, cache, plan = activation
    old = root / "python/mojolearn/identical/.development/old/_mojolearn_old.so"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"unreceipted historical bytes")
    aggregate = root / "python/mojolearn/identical/_mojolearn_native.so"
    aggregate.write_bytes(b"inactive release aggregate")
    assert check(activation) == 0
    active, ignored = stamps.development_bindings(root / "python/mojolearn", root, cache)
    assert len(active) == len(plan["groups"])
    assert old in ignored and old not in active
    assert aggregate in ignored and aggregate not in active


@pytest.mark.parametrize("fault", ["source", "recipe", "compiler", "compiler_packages", "c_compiler", "support_source",
    "active_bytes", "cache_bytes", "failed_receipt", "missing_receipt", "wrong_key", "wrong_flags", "missing_group",
    "extra_member", "index_path", "extra_provenance", "shadow", "support_bytes", "missing_support", "active_symlink"])
def test_refuses_invalid_active_evidence(activation, monkeypatch, fault):
    root, cache, plan = activation
    row = plan["groups"][0]
    receipt_path = cache / "artifacts" / row["key"] / "receipt.json"
    installed = root / "python/mojolearn" / row["mode"] / ".development" / row["key"] / (row["stem"] + ".so")
    if fault == "source":
        (root / "tree/leaf.mojo").write_text("comptime leaf = 2\n")
    elif fault == "recipe":
        (root / "bindings/merged_build.sh").write_text("echo changed recipe\n")
    elif fault.startswith("compiler"):
        monkeypatch.setattr(dev, "toolchain_facts", lambda _: dict(FACTS, **{fault: "changed"}))
    elif fault == "c_compiler":
        monkeypatch.setattr(dev, "c_toolchain_facts", lambda *_: dict(C_FACTS, sha256="changed"))
    elif fault == "support_source":
        (root / dev.PORTABLE_MATH_INPUTS[0]).write_text("changed dependency\n")
    elif fault == "active_bytes":
        installed.write_bytes(b"different active binary")
    elif fault == "cache_bytes":
        (receipt_path.parent / (row["stem"] + ".so")).write_bytes(b"different cached binary")
    elif fault == "missing_receipt":
        receipt_path.unlink()
    elif fault in ("failed_receipt", "wrong_key", "wrong_flags"):
        saved = json.loads(receipt_path.read_text())
        if fault == "failed_receipt":
            saved["state"] = "failed"
        elif fault == "wrong_key":
            saved["key"] = "0" * 64
        else:
            saved["fields"]["environment"]["MOJOLEARN_NUMERIC_MODE"] = "fast"
        dev.atomic_json(receipt_path, saved)
    elif fault in ("missing_group", "extra_member", "index_path", "extra_provenance"):
        path = index_path(root)
        index = json.loads(path.read_text())
        name = next(iter(index["files"]))
        if fault == "missing_group":
            del index["files"][name]
        elif fault == "extra_member":
            index["files"][name].append("_mojolearn_unknown")
        elif fault == "index_path":
            index["files"]["../../outside.so"] = index["files"].pop(name)
        else:
            index["provenance"]["../outside"] = "0" * 64
        dev.atomic_json(path, index)
    elif fault == "shadow":
        (root / "python/mojolearn/identical" / (A + ".so")).write_bytes(b"shadow")
    elif fault in ("support_bytes", "missing_support"):
        path = root / "python/mojolearn" / plan["support"][0]["install"]
        if fault == "support_bytes":
            path.write_bytes(b"corrupt dependency")
        else:
            path.unlink()
    elif fault == "active_symlink":
        installed.unlink()
        installed.symlink_to(receipt_path.parent / (row["stem"] + ".so"))
    assert check(activation) == 1


def test_unknown_direct_binding_still_requires_legacy_stamp(activation):
    root, _, _ = activation
    (root / "python/mojolearn/identical/_mojolearn_unknown.so").write_bytes(b"unknown")
    assert check(activation) == 1


def test_custom_cache_environment(activation, monkeypatch):
    root, cache, _ = activation
    monkeypatch.setenv("MOJOLEARN_BINCACHE_DEVELOPMENT_DIR", str(cache))
    assert stamps.cmd_check(True, root / "python/.binding-stamps", root / "python/mojolearn", root) == 0


def test_legacy_stamps_still_check_sources():
    assert stamps.self_test() == 0
