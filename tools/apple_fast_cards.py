#!/usr/bin/env python3
"""The Apple ``fast`` experiment cards: one table per family and the one loader every tool uses.

Each family keeps its cards in one JSON table, ``<family dir>/cards.json``, with one row per card
(the card id, and every field of every former card file under that file's path). A former card
file path, such as ``experiments/apple_fast_trees/F.json`` or
``experiments/performance_ideas/F01/manifest.json``, stays the card's logical address: tools,
cards and comments still name it, and :func:`document` rebuilds that file's document, value for
value, from the table. Nothing reads a per-card file any more; the files are gone.

Families (the table path, then the former card files it holds):

- ``classical``: ``experiments/apple_fast_classical_20261006/cards.json``; ``ideas.json``,
  ``full_workloads.json``, ``build_bindings.json``, ``measurement_inventory.json``,
  ``measurement_index.json``, ``lanes/*.json`` and the 54 ``performance_ideas/AFCL-*/manifest.json``.
- ``trees``: ``experiments/apple_fast_trees/cards.json``; ``F.json``, ``G.json``, ``N.json``, ``P.json``.
- ``neural``: ``experiments/apple_fast_neural_20261006/cards.json``; ``attention.json``,
  ``training.json``, ``mamba.json``, ``cnn_embedding.json``, ``interactions.json``.
- ``bf16``: ``experiments/apple_fast_bf16_20261015/cards.json``; ``current_decision.json``.
- ``perf``: ``experiments/performance_ideas/apple_fast/cards.json``; ``performance_ideas/F01``-``F20``
  ``manifest.json``, ``compile_audit.json`` and ``F16/compile_matrix.json``.

Scripts (pipelines, capture and task scripts, callers) stay files in their family directory; the
F01-F20 callers live in ``experiments/performance_ideas/apple_fast/callers/`` (table key
``path_moves`` records each move). Reading a table is a file read and a dict walk; it is not on
any product path.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mojolearn.apple-fast-cards/1"
TABLES = {
    "classical": "experiments/apple_fast_classical_20261006/cards.json",
    "trees": "experiments/apple_fast_trees/cards.json",
    "neural": "experiments/apple_fast_neural_20261006/cards.json",
    "bf16": "experiments/apple_fast_bf16_20261015/cards.json",
    "perf": "experiments/performance_ideas/apple_fast/cards.json",
}

_CACHE: dict[tuple[str, str], tuple[int, int, dict]] = {}


def table_path(family: str, root: Path = ROOT) -> Path:
    return Path(root) / TABLES[family]


def load(family: str, root: Path = ROOT) -> dict[str, Any]:
    """The family's table, read once per file version."""
    path = table_path(family, root)
    stat = path.stat()
    key = (str(Path(root).resolve()), family)
    hit = _CACHE.get(key)
    if hit is None or hit[:2] != (stat.st_mtime_ns, stat.st_size):
        table = json.loads(path.read_text())
        if table.get("schema") != SCHEMA or table.get("family") != family:
            raise ValueError(f"{path}: not a {SCHEMA} table for {family}")
        hit = (stat.st_mtime_ns, stat.st_size, table)
        _CACHE[key] = hit
    return hit[2]


def rows(family: str, root: Path = ROOT) -> list[dict[str, Any]]:
    return load(family, root)["rows"]


def row(family: str, ident: str, root: Path = ROOT) -> dict[str, Any]:
    for item in rows(family, root):
        if item["id"] == ident:
            return item
    raise KeyError(f"{family}: no card {ident}")


def _index(root: Path) -> dict[str, tuple[str, dict]]:
    out: dict[str, tuple[str, dict]] = {}
    for family in TABLES:
        if not table_path(family, root).is_file():
            continue
        for doc in load(family, root)["documents"]:
            out[doc["legacy_path"]] = (family, doc)
    return out


def family_of(legacy_path: str, root: Path = ROOT) -> str | None:
    hit = _index(root).get(str(legacy_path))
    return hit[0] if hit else None


def is_card_path(legacy_path: str, root: Path = ROOT) -> bool:
    return str(legacy_path) in _index(root)


def legacy_paths(family: str, root: Path = ROOT) -> list[str]:
    return [doc["legacy_path"] for doc in load(family, root)["documents"]]


def document(legacy_path: str, root: Path = ROOT) -> Any:
    """Rebuild the document a former card file held, from the family table (a fresh copy)."""
    hit = _index(root).get(str(legacy_path))
    if hit is None:
        raise KeyError(f"not an Apple fast card path: {legacy_path}")
    family, doc = hit
    by_id = {item["id"]: item for item in rows(family, root)}
    path = doc["legacy_path"]
    if doc["container"] == "whole":
        return copy.deepcopy(by_id[doc["card_ids"][0]]["parts"][path])
    cards = [by_id[ident]["parts"][path] for ident in doc["card_ids"]]
    out: dict[str, Any] = {}
    for key in doc["key_order"]:
        if key == doc.get("card_key"):
            if doc["container"] == "dict":
                out[key] = {ident: card for ident, card in zip(doc["card_ids"], cards)}
            else:
                out[key] = cards
        else:
            out[key] = doc["fields"][key]
    return copy.deepcopy(out)


def documents(family: str, root: Path = ROOT) -> dict[str, Any]:
    return {path: document(path, root) for path in legacy_paths(family, root)}


def digest(legacy_path: str, root: Path = ROOT) -> str:
    """sha256 of the rebuilt document in canonical JSON (sorted keys, no spaces)."""
    payload = json.dumps(document(legacy_path, root), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_file(legacy_path: str, root: Path = ROOT) -> Path:
    """The file that now holds a former card path: the family table, or the path itself."""
    family = family_of(legacy_path, root)
    return table_path(family, root) if family else Path(root) / legacy_path


def exists(relative: str, root: Path = ROOT) -> bool:
    """True for a file in the tree or a former card path the tables hold."""
    return (Path(root) / relative).is_file() or is_card_path(relative, root)


def read_json(relative: str, root: Path = ROOT) -> Any:
    """A JSON document by repository path: a card from its table, any other file from disk."""
    if is_card_path(relative, root):
        return document(relative, root)
    return json.loads((Path(root) / relative).read_text())
