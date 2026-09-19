import json
from mojolearn import _verify


def test_environment_receipt_keeps_mode_evidence(monkeypatch):
    mode = _verify.ModeReport()
    mode.requested = "identical"
    mode.loaded = "fast"
    mode.compiled = "fast"
    mode.binary = "/tmp/binding.so"
    mode.ident_dir = "/tmp/identical"
    mode.under_identical = False
    mode.problem = "compiled mode differs"
    original = {"mode": mode, "device": "test GPU", "commit": "a" * 40}
    monkeypatch.setattr(_verify, "environment", lambda supplied: dict(original))
    report = json.loads(json.dumps(_verify.environment_json(mode)))
    assert report == {**original, "mode": mode.as_dict()}
    assert original["mode"] is mode
    assert report["mode"]["problem"] == "compiled mode differs"
