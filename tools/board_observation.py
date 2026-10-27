"""Lossless storage of duplicate main-board observation cells."""
import copy
import hashlib
import json


def _cell_bytes(cell):
    return json.dumps(cell, sort_keys=True, separators=(",", ":"), default=str).encode()


def compact_observation(ob):
    """Reference an exactly duplicated cell inside the same retained pairing."""
    out = dict(ob)
    measured = out.get("measured_cell")
    if measured:
        encoded = _cell_bytes(measured)
        for index, cell in enumerate((out.get("paired_race") or {}).get("cells", [])):
            if _cell_bytes(cell) == encoded:
                out.pop("measured_cell")
                out["measured_cell_ref"] = {"paired_cells_index": index,
                                            "sha256": hashlib.sha256(encoded).hexdigest()}
                break
    if out.get("measured_cell_ref") is None:
        out.pop("measured_cell_ref", None)
    return out


def restore_observation(ob):
    out = dict(ob)
    ref = out.pop("measured_cell_ref", None)
    if ref is not None:
        cells = (out.get("paired_race") or {}).get("cells", [])
        index = ref.get("paired_cells_index")
        if type(index) is not int or not 0 <= index < len(cells):
            raise ValueError("invalid paired measurement reference")
        cell = cells[index]
        if hashlib.sha256(_cell_bytes(cell)).hexdigest() != ref.get("sha256"):
            raise ValueError("paired measurement reference bytes changed")
        out["measured_cell"] = copy.deepcopy(cell)
    return out


