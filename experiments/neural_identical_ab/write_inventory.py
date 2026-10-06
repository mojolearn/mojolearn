"""Author the repository experiment/file index and normalized neural selectors.

This reads retained metadata and source text to write documentation. It does
not import ML code, execute an experiment, inspect a binary, or check source.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs/plans/NEURAL_AB_EXPERIMENT_INVENTORY_2026-10-06.md"


def unique(values):
    return list(dict.fromkeys(values))


def links(paths):
    return ", ".join(f"[{p}](../../{p})" for p in unique(paths)) or "none recorded"


def normalize(raw, origin):
    row = dict(raw)
    row["inventory_sources"] = [str(origin.relative_to(ROOT))]
    row["implementation_paths"] = unique(raw.get("implementation_paths", []) + raw.get("primary_source_paths", []) + raw.get("shared_source_paths", []))
    row["caller_paths"] = raw.get("caller_paths", [])
    row["legacy_experiments"] = [r["id"] if isinstance(r, dict) else r for r in raw.get("legacy_experiments", [])]
    row["legacy_experiments"] += [r["id"] for r in raw.get("performance_ideas_equivalents", [])]
    row["limitations"] = raw.get("limitations", []) + raw.get("remaining_integration_limitations", [])
    if raw.get("full_workload_selection") and not raw.get("workloads"):
        row["workloads"] = [raw["full_workload_selection"]]
    arms = raw.get("ab", raw)
    for arm in ("baseline", "candidate"):
        row[arm] = dict(arms[arm])
        row[arm]["runtime"] = row[arm].get("runtime", row[arm].get("runtime_settings", {}))
    for name, description in raw.get("required_compile_parameters", {}).items():
        row.setdefault("parameters", {})[name] = dict(minimum=1, description=description)
        for arm in ("baseline", "candidate"):
            row[arm]["compile_defines"].append(name + "={" + name + "}")
    return row


def write():
    cards = json.loads((HERE / "catalog.json").read_text())["experiments"]
    authored = {}
    # State/CNN owns actual NN13/NN14 callers; training owns NN59's runnable
    # complete-layer recipe. Preserve every lane's source notes alongside it.
    for name in ("gemm", "attention", "state_cnn", "training"):
        path = HERE / "lanes" / (name + "_integration_inventory.json")
        content = json.loads(path.read_text())
        for raw in content.get("experiments", content.get("inventory", [])):
            row = normalize(raw, path)
            earlier = authored.get(row["id"])
            if earlier:
                for key in ("implementation_paths", "caller_paths", "inventory_sources", "limitations"):
                    row[key] = unique(earlier.get(key, []) + row.get(key, []))
                row["legacy_experiments"] = unique(earlier.get("legacy_experiments", []) + row.get("legacy_experiments", []))
            authored[row["id"]] = row
    # The LM-head source owner supplies extra contracts after the training
    # owner supplies the selectable model recipe; merge notes, not controls.
    supplement = HERE / "lanes/lm_head_integration_inventory.json"
    for raw in json.loads(supplement.read_text()).get("inventory", []):
        extra = normalize(raw, supplement)
        row = authored[raw["id"]]
        for key in ("implementation_paths", "caller_paths", "inventory_sources", "limitations"):
            row[key] = unique(row.get(key, []) + extra.get(key, []))
    rows = []
    for card in cards:
        row = authored[card["id"]]
        row.update(title=card["title"], lane=card["lane"], mode="identical", default_enabled=False,
                   qualification="uncompiled_unverified_unmeasured", intended_models=card["callers"])
        if not row.get("workloads"):
            row["workloads"] = [dict(harness="tools/bench_board_algos.py" if card["lane"] == "state_cnn" else "tools/bench_board_neural.py",
                                      affected_models=card["callers"], scope="Resolve saved full-workload recipe for each caller; no component substitute")]
        rows.append(row)
    (HERE / "arms.json").write_text(json.dumps(dict(schema=1, qualification="source_only", experiments=rows), indent=2) + "\n")

    existing = []
    for path in sorted((ROOT / "experiments/performance_ideas").glob("*/manifest.json")):
        old = json.loads(path.read_text())
        files = [str(path.relative_to(ROOT))] + old.get("implementation_paths", [])
        files += [str(p.relative_to(ROOT)) for p in sorted(path.parent.iterdir())
                  if p.is_file() and p.name != "manifest.json"]
        existing.append(dict(id=old["id"], title=old["title"], mode=old["mode"], vendors=old["vendors"],
                             retained_status=old["status"], files=unique(files),
                             candidate_defines=old.get("candidate_defines", []), baseline_defines=old.get("baseline_defines", []),
                             timing_contract=old["timing_contract"], blocker=old.get("blocker")))
    profiles = json.loads((ROOT / "experiments/identical_speed/profiles.json").read_text())["profiles"]
    text = (ROOT / "tools/neural_experiments.py").read_text()
    source_block = text.split("EXPERIMENTS = {", 1)[1].split("\nSETS =", 1)[0]
    toggles = re.findall(r'^    "([a-z0-9_]+)":', source_block, re.M)
    suites = [
        dict(name="Existing runtime neural toggles", ids=toggles, files=["tools/neural_experiments.py", "tools/neural_stage_timing.py"],
             scope="Runtime schedule/ownership screening. Existing same-baseline digest policy is specific to those schedule arms, not the acceptance rule for new arithmetic versions."),
        dict(name="Existing IDENTICAL GEMM profiles", ids=list(profiles), files=["experiments/identical_speed/profiles.json", "experiments/identical_speed/selected-batch.json", "tools/identical_speed_ab.py"],
             scope="Synthetic/component screening; retained profile vendor subsets do not qualify the new full-neural callers."),
        dict(name="Resident callpath", ids=["I1 persistent call", "I2 batch/per-item wait", "I3 batch/grouped wait"],
             files=["experiments/identical_callpath/README.md", "experiments/identical_callpath/COVERAGE.md", "core/identical_callpath.mojo"] + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "experiments/identical_callpath").glob("*.mojo"))],
             scope="Existing shared infrastructure and scaler adapters; classical reference only, no new classical runtime work here. I1/I2/I3 are local names, not performance_ideas I01/I02/I03."),
        dict(name="Byte-LM pool and logical/device shards", ids=["pool baseline/candidate", "logical/device shard baseline/candidate"],
             files=["tools/byte_lm_pool_ab_matrix.sh", "tools/byte_lm_pool_ab_compare.py", "tools/byte_lm_optimizer_pool_check.py", "tools/lm_shards_ab_matrix.sh", "tools/lm_shards_ab_compare.py", "tools/lm_shards_probe.py"],
             scope="Existing state/rollback/partition matrices. These scripts execute checks and some overwrite installed bindings; none were run in this task."),
        dict(name="Apple neural schedule families", ids=["attention arms", "GEMM geometry", "neural stage A/B"],
             files=[str(p.relative_to(ROOT)) for p in sorted((ROOT / "tools/apple_speed_neural").glob("*.sh"))],
             scope="Existing Apple-specific exploration; Apple does not vote on new IDENTICAL promotions."),
        dict(name="Legacy GEMM native experiment modules", ids=[], files=[str(p.relative_to(ROOT)) for p in sorted((ROOT / "gemm/experiments").glob("*")) if p.is_file()],
             scope="Includes shared/new neural files and old probes/checks/build orchestration. File presence is not an execution or qualification claim."),
        dict(name="Neural and full-workload orchestration", ids=[],
             files=["tools/bench_board_neural.py", "tools/bench_board_algos.py", "tools/bench_board.py", "tools/performance_ideas.py", "tools/performance_full_ab_queue.py", "tools/performance_ideas_gpu_queue.py", "tools/neural_family_screen.py", "tools/neural_family_screen_body.sh", "tools/neural_family_screen_summary.py", "tools/bench_neural_decode.py"],
             scope="Saved recipe locations and existing orchestration. Neural-only builders are in scope; full algorithms registry also includes classical estimators."),
    ]
    extra_inventory = HERE / "lanes/attention_integration_inventory.json"
    attention = json.loads(extra_inventory.read_text())
    extras = attention.get("discovered_existing_experiments", attention.get("existing_experiments", []))
    state_registry = json.loads((HERE / "lanes/state_cnn_integration_inventory.json").read_text()).get("legacy_experiment_registry", {})
    for name, old in state_registry.items():
        suites.append(dict(name="State/CNN reference: " + name, ids=old.get("controls", []),
                           files=old.get("paths", []), scope=old.get("relationship", "Existing source reference")))
    index = dict(schema=1, neural=rows, existing_manifest_experiments=existing, additional_suites=suites,
                 attention_existing_experiments=extras, qualification="source inventory only; no new execution evidence")
    (HERE / "experiment_inventory.json").write_text(json.dumps(index, indent=2) + "\n")

    lines = ["# Neural A/B experiment and file inventory — 2026-10-06", "",
             "Branch `ideas/neural-identical-ab-20261006-r3`, forked from main at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.", "",
             "This document indexes all 64 new neural idea cards, their selected source arms and callers, the 60 existing I/A/N/F manifests found in this checkout, and additional named A/B suites. Existing classical and FAST entries are references; the new implementation scope is neural IDENTICAL only. Research variants beyond each selected arm are not claimed as implemented.", "",
             "All new source remains uncompiled, unverified and unmeasured by request. No identity checks, quality evaluation, timing, board updates or default promotions were performed. Bits may differ across versions or A/B arms; within one selected version they must agree on NVIDIA, AMD, Apple and host. Source wiring does not prove that obligation.", "",
             "## Selectors and model integration", "",
             "[arms.json](../../experiments/neural_identical_ab/arms.json) contains concrete A/B compile defines, environment and runtime settings. [tools/neural_identical_ab.py](../../tools/neural_identical_ab.py) `configure` writes an arm JSON consumed by `bench_board_neural.py --neural-ab-config` and an optional shell environment consumed by the existing native builders. It never compiles or executes. The native build must apply the same profile flags to every affected GPU and host binding; a config file is not proof that installed binaries match it.", "",
             "The retained [performance_ideas runner](../../tools/performance_ideas.py) still owns its I/A/N/F frozen-manifest evidence protocol. New NN selectors link reused IDs and sources; old component evidence is not copied into new model qualification. `neural_identical_ab.py list --include-existing` exposes both registries.", "",
             "Explicit operations reach transformer tape/VJP and windowed cache writes, Mamba forward/VJP and owned weights, the residual/dropout layer, stateless MLP session batching, the chunked LM head, and configured Samba clipping/accumulation. Neural algorithm recipes can apply the CNN unpooled-block setting in both arms. Normal lanes remain available for ordinary model operations. Full input hashes, cap audits, all affected estimator workloads and relevant combinations are still future campaign prerequisites. A layer-only operation is labeled as such, not as a full model train step.", "",
             "## New neural experiments", "",
             "A is candidate; B is baseline. The exact controls below are compile defines unless prefixed `env:` or `runtime:`. `is_defined` flags are disabled by omission, never by setting them to zero. Hardware parameters such as NN10's fill budget require an explicit recorded value. Each linked lane inventory retains admission, interactions and limitations.", ""]
    for row in rows:
        lines += [f"### {row['id']} — {row['title']}", "",
                  "Files: " + links(row["implementation_paths"]) + ".", "",
                  "Callers: " + links(row["caller_paths"]) + ".", ""]
        workload_paths = []
        for workload in row.get("workloads", []):
            workload_paths += workload.get("recipe_paths", [])
            workload_paths += [workload[key] for key in ("harness", "recipe_path") if workload.get(key)]
        if workload_paths:
            lines += ["Workload/selector files: " + links(workload_paths) + ".", ""]
        for arm, label in (("candidate", "A"), ("baseline", "B")):
            a = row[arm]
            parts = ["`" + x + "`" for x in a.get("compile_defines", [])]
            parts += [f"env: `{k}={v}`" for k, v in a.get("environment", {}).items()]
            if a.get("runtime"):
                parts.append("runtime: `" + json.dumps(a["runtime"], sort_keys=True) + "`")
            lines += [label + ": " + ("; ".join(parts) or "incumbent, new flags absent") + ".", ""]
        if row.get("variants"):
            lines += ["Additional selectable variants: " + ", ".join(f"`{name}`" for name in row["variants"]) + ".", ""]
        lines += ["Arm/caller notes: " + links(row["inventory_sources"]) + ". Existing related IDs: " +
                  (", ".join(str(x) for x in row["legacy_experiments"]) or "none separately claimed") + ".", ""]
        for note in row.get("limitations", []):
            lines += ["- " + str(note)]
        lines.append("")
    lines += ["## Existing I/A/N/F manifest experiments", "",
              "Statuses below are retained metadata from before this continuation, not newly verified results. Read each manifest's timing contract, coverage and retained failures. A build status or component result does not establish full-dataset model qualification. F entries are Apple FAST references. A/N entries retain their original vendor scope; they are not automatically four-column numerical profiles.", ""]
    for row in existing:
        lines += [f"### {row['id']} — {row['title']}", "",
                  f"Mode `{row['mode']}`; vendors {', '.join(row['vendors'])}; retained status `{row['retained_status']}`.", "",
                  "Files: " + links(row["files"]) + ".", "",
                  "A defines: " + (", ".join(f"`{x}`" for x in row["candidate_defines"]) or "none in manifest") + ".", "",
                  "B defines: " + (", ".join(f"`{x}`" for x in row["baseline_defines"]) or "none in manifest") + ".", ""]
        if row["blocker"]:
            lines += ["Retained blocker: " + row["blocker"], ""]
    lines += ["## Other existing suites and A/B files found", ""]
    for suite in suites:
        lines += ["### " + suite["name"], "", suite["scope"], "",
                  "Names: " + (", ".join("`" + s + "`" for s in suite["ids"]) or "see per-file declarations") + ".", "",
                  "Files: " + links(suite["files"]) + ".", ""]
    if extras:
        lines += ["Additional attention-family records are retained in " + links([str(extra_inventory.relative_to(ROOT))]) + ":", ""]
        for extra in extras:
            lines += ["### " + extra["id"] + " — " + extra.get("title", "Existing attention experiment"), "",
                      "Files: " + links(extra.get("implementation_paths", [])) + ".", ""]
            for key, label in (("candidate", "A"), ("baseline", "B")):
                arm = extra.get(key, {})
                parts = ["`" + value + "`" for value in arm.get("compile_defines", [])]
                parts += [f"env: `{k}={v}`" for k, v in arm.get("environment", {}).items()]
                lines += [label + ": " + ("; ".join(parts) or "see source control") + ".", ""]
            if extra.get("additional_runtime_values"):
                lines += ["Other recorded runtime arms: " + ", ".join(f"`{x}`" for x in extra["additional_runtime_values"]) + ".", ""]
            lines += ["- " + note for note in extra.get("limitations", [])]
            lines.append("")
    lines += ["## Retained limits and evidence", "",
              "NN41's inherited recurrent-scan quality failure and NN20's earlier attention-profile nonpromotion remain open evidence; new wiring does not erase either. Explicit dimension-targeted norm scheduling found during this source pass has a general hardware-cost candidate and an explicit historical B; neighboring shapes and a non-board workload remain required before promotion.", "",
              "This index describes experiments found in the inspected registries and named suites; it does not claim every historical branch or every compiler option in the repository was enumerated. The machine-readable companion is [experiment_inventory.json](../../experiments/neural_identical_ab/experiment_inventory.json), with selector source in [write_inventory.py](../../experiments/neural_identical_ab/write_inventory.py). Model scope and outstanding research variants remain in [MODEL_INTEGRATION.md](../../experiments/neural_identical_ab/MODEL_INTEGRATION.md).", ""]
    DOC.write_text("\n".join(lines))


if __name__ == "__main__":
    write()
