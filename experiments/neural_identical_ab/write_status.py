"""Render retained lane metadata as a source-delivery ledger, not a checker."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_status():
    catalog = json.loads((HERE / "catalog.json").read_text())
    handoffs = {}
    for path in sorted((HERE / "lanes").glob("*.json")):
        lane = json.loads(path.read_text())
        rows = lane.get("experiments", []) + lane.get("cross_owner_experiments", [])
        seen_ids = set()
        for row in rows:
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            handoffs.setdefault(row["id"], []).append(dict(row, handoff=path.name))
    lines = ["# Neural source delivery status", "",
             "Branch `ideas/neural-identical-ab-20261006-r3`; forked from main `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.", "",
             "This ledger summarizes source authoring, not verification. No compilation, tests, static checkers, candidate execution, quality evaluation or timing were run. All new candidates remain opt-in. Existing enabled switches are identified in their handoffs; no new default was promoted.", "",
             "64 idea cards were completed before four-lane programming (root plus three agents). The completion pass connects selected arms to native/public callers and matching arithmetic profiles. Broad research variants, prior source and remaining work are distinct below. A wired arm is source programming, never a claim of successful compilation or qualification. See MODEL_INTEGRATION.md for the selected delivery scope.", "",
             "| Card | Idea | Recorded source status by owner | Handoff |", "| --- | --- | --- | --- |"]
    for card in catalog["experiments"]:
        entries = handoffs.get(card["id"], [])
        status = "; ".join(f"{e['handoff'].removesuffix('.json')}: {e.get('implementation_status', 'scope_in_handoff')}" for e in entries) or "pending handoff"
        links = "; ".join(f"[{e['handoff']}](lanes/{e['handoff']})" for e in entries) or "pending"
        lines.append(f"| {card['id']} | {card['title']} | {status} | {links} |")
    lines.extend(["", "## Material remaining work", "",
                  "- Per-card handoffs describe the selected native caller, ownership and arithmetic contract. Additional variants in a broad idea are not silently claimed as implemented; see [model integration scope](MODEL_INTEGRATION.md).",
                  "- Mamba repository host source was authored through the new explicit source-write-only generator mode. Its post-generation verification and stale comparison were disabled. This writes source, not compiler IR; no compiler or toolchain was patched.",
                  "- NN41 preserves an inherited recurrent-scan quality failure in its lane notes. Existence of an opt-in route does not resolve that failure or justify a promotion.",
                  "- NN20 preserves earlier attention-v2 nonpromotion evidence. Its newly connected profile cannot inherit qualification or erase prior failures.",
                  "- Full dataset/corpus hashes, intrinsic-cap audits, settings, exact workload commands and transitive affected-model coverage must be resolved before a future campaign. No diagnostic fixture substitutes for those recipes.",
                  "- All compilation, same-version four-column identity, model-quality and joint NVIDIA/AMD full-workload A/B measurements are intentionally unrun. Apple does not vote on IDENTICAL timing.", "",
                  "## Source evidence", "",
                  "The [idea list](../../docs/plans/NEURAL_IDENTICAL_AB_IDEAS_2026-10-06.md), [catalog](catalog.json), lane JSON/Markdown files and actual Mojo changes are the retained evidence. There are no newly produced build/test/GPU logs or performance results to cite. The metadata-only [planner](../../tools/neural_identical_ab.py) is authored but unexecuted.", "",
                  "For cross-owner NN13/NN14, read both GEMM and state/CNN handoffs: the GEMM owner records its boundary; the state/CNN supplement records the actual caller work. This ledger preserves both rather than erasing partial coverage.", ""])
    (HERE / "IMPLEMENTATION_STATUS.md").write_text("\n".join(lines))
    record = dict(schema=1, branch="ideas/neural-identical-ab-20261006-r3",
                  base_commit=catalog["baseline"], source_delivery_only=True,
                  compilation="not_run_by_request", verification="not_run_by_request",
                  measurements="not_run_by_request", defaults_promoted=False,
                  cards=[dict(id=c["id"], title=c["title"], handoffs=handoffs.get(c["id"], [])) for c in catalog["experiments"]])
    (HERE / "implementation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    write_status()
