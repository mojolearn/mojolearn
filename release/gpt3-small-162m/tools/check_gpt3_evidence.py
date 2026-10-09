#!/usr/bin/env python3
"""Import local T3 evidence, or check the self-contained paper supplement.

No GPU, network access, checkpoint download, or training is needed.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "supplement/gpt3-six-segment-2026-09-28"
FIELDS = ("state_sha256", "gradient_sha256", "losses_f32_hex",
          "lr_f32_hex", "hash_scheme", "batch_index")
BOUNDS = (0, 1000, 2000, 2400, 3900, 4900, 5000)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def import_evidence(source):
    ledger = json.loads((source / "t3-b1/ledger.json").read_text())["landed"]
    manifest = {}

    def save(path, name, project=None, compress=False, text_projection=False):
        raw = path.read_bytes()
        value = raw.decode() if text_projection else None
        data = raw if project is None else (json.dumps(project(value if text_projection else json.loads(raw)), indent=2) + "\n").encode()
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, mtime=0) if compress else data)
        manifest[name] = {"source": str(path.relative_to(source)),
                          "source_sha256": sha(raw), "stored_sha256": sha(target.read_bytes()),
                          "projection": project is not None}
        return name

    receipt_keys = ("schema", "route", "segment", "label", "from_checkpoint",
                    "first_step", "last_step", "devices", "logical_shards",
                    "recipe_sha256", "tokens_sha256", "hash_scheme", "verdict",
                    "disagreements", "steps_completed", "last_completed", "checkpoints",
                    "control", "zero_moments")

    def receipt(j):
        out = {k: j[k] for k in receipt_keys if k in j}
        out["box"] = {k: v for k, v in j.get("box", {}).items() if k != "host"}
        return out

    save(source / "t3_recipe.json", "recipe.json")
    records = {}
    for key, entry in sorted(ledger.items()):
        folder = key.replace("/", "-")
        base = Path(entry["results"])
        paths = sorted(base.glob("**/segment/segment.json"))
        if key == "A/3":
            paths.sort(key=lambda p: "nvidia-" not in str(p))
        assert paths, key
        main = paths[0]
        chains = []
        for n, attempt in enumerate(entry.get("attempts", [])):
            chains.append(save(Path(attempt["dir"]) / "segment/chain.jsonl",
                               f"{folder}/attempt-{n+1}.jsonl.gz", compress=True))
        chains.append(save(main.with_name("chain.jsonl"), f"{folder}/chain.jsonl.gz", compress=True))
        record = {"chains": chains,
                  "receipt": save(main, f"{folder}/receipt.json", receipt),
                  "checkpoints": entry["checkpoints"], "arrival_verdict": entry["arrival"],
                  "steps_completed": entry["steps_completed"], "verdict": entry["verdict"]}
        record["environment_files"] = []
        if entry.get("attempts"):
            launch = Path(entry["attempts"][0]["results"]) / "extra_body.sh"
            record["initial_checkpoint_refs"] = save(launch, f"{folder}/initial-checkpoint-refs.json",
                lambda body: sorted(set(re.findall(r"runs/t3/2026-09-22/[AB]/[1-6]/ckpt_[0-9]+\.blm", body))),
                text_projection=True)
        for name in ("gpu.txt", "binding.txt", "wheel.sha256"):
            path = main.parent.parent / name
            if path.exists():
                record["environment_files"].append(save(path, f"{folder}/{name}"))
        arrival = main.parent.parent / "arrival/segment.json"
        if arrival.exists():
            record["arrival_receipt"] = save(arrival, f"{folder}/arrival.json", receipt)
            record["arrival_chain"] = save(arrival.with_name("chain.jsonl"),
                                           f"{folder}/arrival.jsonl.gz", compress=True)
        if key == "A/3":
            record["worker_chain"] = save(paths[1].with_name("chain.jsonl"),
                                           f"{folder}/worker.jsonl.gz", compress=True)
            record["worker_receipt"] = save(paths[1], f"{folder}/worker.json", receipt)
        if key.startswith("B/"):
            body = source / f"t3-b1/bodies/{folder}.ledger.json"
            record["checkpoint_source"] = save(body, f"{folder}/checkpoint-source.json",
                lambda j: {k: j[k] for k in ("run", "route", "segment", "from_key", "from_sha", "boundary")})
        records[key] = record
    # The Apple model is reported separately from the generic Darwin receipt.
    apple = next(Path(ledger["A/6"]["results"]).glob("**/gpu.txt"))
    save(apple, "A-6/gpu.txt")
    # Keep exact records for positive controls and distinct deliberate faults.
    controls = {}
    for family, names in [("t3-controls", ["zero-moments", "k63", "swap-5-40", "swap-0-1"]),
                          ("t3-controls-0.8.19", ["positive", "split", "ulp"]),
                          ("t3-controls-amd-0.8.19", ["positive", "ulp"])]:
        for name in names:
            matches = list((source / family).glob(f"**/{name}/segment.json"))
            assert len(matches) == 1, (family, name, matches)
            label = f"{family}/{name}"
            controls[label] = {
                "receipt": save(matches[0], f"controls/{label}/receipt.json", receipt),
                "chain": save(matches[0].with_name("chain.jsonl"),
                              f"controls/{label}/chain.jsonl.gz", compress=True)}
    write_json(DEST / "index.json", {"segments": records, "controls": controls})
    write_json(DEST / "manifest.json", manifest)


def check_evidence():
    manifest = json.loads((DEST / "manifest.json").read_text())
    for name, entry in manifest.items():
        assert sha((DEST / name).read_bytes()) == entry["stored_sha256"], name
    index = json.loads((DEST / "index.json").read_text())
    recipe_bytes = (DEST / "recipe.json").read_bytes()
    recipe = json.loads(recipe_bytes)

    def read_json(name):
        return json.loads((DEST / name).read_text())

    def chain(name):
        lines = gzip.decompress((DEST / name).read_bytes()).decode().splitlines()
        rows = {}
        prev = None
        for line in lines:
            row = json.loads(line)
            assert row["prev"] == prev, (name, row["step"], "broken chain")
            assert row["step"] not in rows, (name, "duplicate step")
            rows[row["step"]] = row
            prev = sha(line.encode())
        return rows

    def diffs(a, b):
        return [f for f in FIELDS if a.get(f) != b.get(f)]

    routes, segments, replay_rows = {}, [], []
    for key, entry in sorted(index["segments"].items()):
        receipt = read_json(entry["receipt"])
        assert entry["verdict"] == receipt["verdict"] == "PASS"
        assert receipt["disagreements"] == []
        assert receipt["recipe_sha256"] == sha(recipe_bytes)
        assert receipt["hash_scheme"] == "sliced-sha256-8.v2"
        rows, repeats = {}, 0
        for name in entry["chains"]:
            for step, row in chain(name).items():
                if step in rows:
                    assert not diffs(rows[step], row), (key, step, "recovery overlap")
                    repeats += 1
                rows[step] = row
        i = int(key[-1])
        assert set(rows) == set(range(BOUNDS[i-1]+1, BOUNDS[i]+1)), key
        assert entry["steps_completed"] == len(rows)
        for step, row in rows.items():
            assert len(row["losses_f32_hex"]) == 64
            assert row["batch_index"] == [(step-1)*64, step*64]
            assert row["lr_f32_hex"] == recipe["schedule"]["table_f32_hex"][step-1]
        start = receipt["from_checkpoint"]
        if i == 1:
            assert start["step"] == 0
        elif start["step"] == BOUNDS[i-1]:
            previous = index["segments"][f"{key[0]}/{i-1}"]["checkpoints"]
            assert previous[start["file"]] == start["sha256"]
        else:
            assert entry["checkpoints"][start["file"]] == start["sha256"]
            initial_refs = read_json(entry["initial_checkpoint_refs"])
            assert f"runs/t3/2026-09-22/{key[0]}/{i-1}/ckpt_{BOUNDS[i-1]:08d}.blm" in initial_refs
        if key.startswith("B/"):
            provenance = read_json(entry["checkpoint_source"])
            assert provenance["from_sha"] == start["sha256"]
            expected = "A/1" if i == 1 else (f"B/{i}" if len(entry["chains"]) > 1 else f"B/{i-1}")
            assert f"/{expected}/" in provenance["from_key"]
        if "arrival_chain" in entry:
            arrival = chain(entry["arrival_chain"])
            ar = read_json(entry["arrival_receipt"])
            assert ar["verdict"] == "PASS" and ar["disagreements"] == []
            assert set(arrival) == {BOUNDS[i-1]-1, BOUNDS[i-1]}
            for step, row in arrival.items():
                assert not diffs(row, routes[f"{key[0]}/{i-1}"][step])
            replay_rows.append({"segment": key, "steps": sorted(arrival), "verdict": "PASS"})
        routes[key] = rows
        segments.append({"segment": key, "steps": len(rows), "devices": receipt["devices"],
                         "overlap_steps_checked": repeats, "token_sha256": receipt["tokens_sha256"]})
    assert len({s["token_sha256"] for s in segments}) == 1
    pairs = []
    for i in range(1, 7):
        a, b = routes[f"A/{i}"], routes[f"B/{i}"]
        assert all(not diffs(a[s], b[s]) for s in a), i
        ca, cb = (index["segments"][f"{r}/{i}"]["checkpoints"] for r in "AB")
        shared = sorted(set(ca) & set(cb))
        assert all(ca[c] == cb[c] for c in shared)
        endpoint = f"ckpt_{BOUNDS[i]:08d}.blm"
        assert endpoint in shared
        pairs.append({"segment": i, "first_step": min(a), "last_step": max(a),
                      "equal_steps": len(a), "divergent_steps": 0,
                      "equal_checkpoint_digests": len(shared), "endpoint_sha256": ca[endpoint]})
    live = chain(index["segments"]["A/3"]["worker_chain"])
    assert set(live) == set(routes["A/3"])
    assert all(not diffs(live[s], routes["A/3"][s]) for s in live)
    controls = []
    for name, entry in index["controls"].items():
        rows = chain(entry["chain"])
        receipt = read_json(entry["receipt"])
        differences = [{"step": s, "fields": diffs(row, routes["A/1"][s])}
                       for s, row in rows.items() if diffs(row, routes["A/1"][s])]
        positive = name.split("/")[-1] in ("positive", "split")
        assert (not differences) == positive
        assert receipt["verdict"] == ("PASS" if positive else "FAIL")
        controls.append({"control": name, "verdict": receipt["verdict"], "steps": sorted(rows),
                         "differences": differences})
    report = {"schema": "mlsys.gpt3-six-segment-audit.v1", "compared_fields": list(FIELDS),
              "pairs": pairs, "segments": segments, "arrival_replays_checked": replay_rows,
              "live_worker_equal_steps": len(live), "controls": controls,
              "equal_steps": sum(p["equal_steps"] for p in pairs),
              "tokens_per_route": 5000*64*4*2048,
              "final_checkpoint_sha256": pairs[-1]["endpoint_sha256"]}
    write_json(ROOT / "results/gpt3-six-segment-2026-09-28.json", report)
    print(f"PASS: {report['equal_steps']} paired steps; {len(live)} live-worker steps; "
          f"{len(replay_rows)} raw arrival replays; {len(controls)} controls; final checkpoint equal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-root", type=Path)
    args = parser.parse_args()
    if args.import_root:
        import_evidence(args.import_root)
    check_evidence()
