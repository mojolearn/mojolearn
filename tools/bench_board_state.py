"""Untimed benchmark receipts for scored outputs and public model exports.

This is evidence glue, never part of the library runtime or measured operation.
Output identity is deliberately separate from model identity. A missing public
export remains explicit and must not be treated as a verified model match.
"""
import hashlib
import json
from pathlib import Path
import tempfile


def canonical_hash(value):
    """SHA256 of typed, framed nested state (including array shape and dtype)."""
    import numpy as np
    h = hashlib.sha256(b"mojolearn-benchmark-state-v1\0")

    def frame(kind, raw):
        h.update(kind.encode() + b":" + str(len(raw)).encode() + b":" + raw)

    def visit(obj):
        if isinstance(obj, dict):
            frame("dict", str(len(obj)).encode())
            for key in sorted(obj):
                frame("key", str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            frame("sequence", str(len(obj)).encode())
            for item in obj:
                visit(item)
        elif isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError("object arrays cannot establish state identity")
            arr = np.ascontiguousarray(obj)
            frame("array", json.dumps([arr.dtype.str, list(arr.shape)], separators=(",", ":")).encode())
            frame("bytes", arr.tobytes())
        elif isinstance(obj, np.generic):
            visit(np.asarray(obj))
        elif obj is None or isinstance(obj, (str, bool, int, float)):
            frame(type(obj).__name__, json.dumps(obj, allow_nan=False).encode())
        else:
            raise TypeError("unsupported exported state type: " + type(obj).__name__)
    visit(value)
    return h.hexdigest()


def _model_receipt(model):
    """Hash a public snapshot; never fit, predict, or inspect opaque handles."""
    if model is None:
        return {"status": "unavailable", "reason": "runner does not retain an exportable model"}
    info = {"class": type(model).__name__, "schema": "mojolearn-benchmark-state-v1"}
    if not type(model).__module__.startswith("mojolearn"):
        return dict(info, status="unavailable", reason="opponent model export is not normalized")
    if callable(getattr(model, "state_dict", None)):
        state = model.state_dict()
        return dict(info, status="ok", source="public state_dict", sha256=canonical_hash(state))
    if callable(getattr(model, "save", None)):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="mojolearn-scored-state-") as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as saved:
                # Device routing metadata is not numerical state. Preserve
                # the complete export hash too so this exclusion is auditable.
                state = {key: saved[key] for key in saved.files}
                full = canonical_hash(state)
                omitted = [key for key in ("device",) if key in state]
                for key in omitted:
                    del state[key]
                return dict(info, status="ok", source="public save npz", sha256=canonical_hash(state),
                            complete_export_sha256=full, excluded_metadata=omitted)
    return dict(info, status="unavailable", reason="no public state_dict or save export")


def model_receipt(model):
    try:
        return _model_receipt(model)
    except Exception as exc:
        # Keep the scored measurement and its original diagnostic. Failed
        # exports are never an identity claim and can be repaired separately.
        return {"status": "error", "class": type(model).__name__,
                "error": type(exc).__name__ + ": " + str(exc)}


def runner_model(runner):
    callback = getattr(runner, "model_for_hash", None)
    if callback is not None:
        return callback()
    for name in ("est", "nn", "kd", "trainer", "model"):
        model = getattr(runner, name, None)
        if model is not None:
            return model
    return None


def scored_receipt(outputs, runner=None, model=None):
    present = outputs is not None and bool(outputs) and all(value is not None for value in outputs.values())
    return {"schema": "mojolearn-scored-receipt/1", "output_status": "ok" if present else "unavailable",
            "output_sha256": canonical_hash(outputs) if present else None,
            "model": model_receipt(model if model is not None else runner_model(runner))}
