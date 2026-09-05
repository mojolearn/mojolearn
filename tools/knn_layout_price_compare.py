#!/usr/bin/env python3
"""Validate four-arm native public k-NN evidence without running GPU work.

Usage: python tools/knn_layout_price_compare.py DIRECTORY [--other-vendor DIR]
Build/run status labels and log names are fixed by the campaign contract.
Source hashes identify the local parser/driver files, not proof of remote build
provenance. Preserve the campaign source revision/build manifest separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from knn_smallk_price_compare import QUERIES, ROUNDS, distribution, parse_log, require

ARMS = {"baseline": (0, 0), "selector": (1, 0),
        "transpose": (0, 1), "both": (1, 1)}
SOURCE_FILES = (
    "tools/knn_layout_price_compare.py", "tools/knn_smallk_price_compare.py",
    "bench/knn_layout_dispatch_check.mojo", "bench/knn_layout_dispatch_price.mojo",
    "bench/knn_smallk_dispatch_check.mojo", "bench/knn_smallk_dispatch_price.mojo",
    "neighbors/impl/neighbors/detail/knn_brute_force.mojo",
    "neighbors/checks/select_smallk_identical_candidate.mojo",
    "neighbors/checks/transposed_index_distance_candidate.mojo",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def flags(lines, arm, path):
    selector, transpose = ARMS[arm]
    expected = ["LAYOUT_FLAGS", "mode", "IDENTICAL", "selector", str(selector),
                "transpose", str(transpose)]
    require([line for line in lines if line[0].startswith("LAYOUT_FLAGS")] == [expected],
            f"{path}: missing, duplicate or incorrect LAYOUT_FLAGS")
    require(lines[0] == expected, f"{path}: LAYOUT_FLAGS must be the first record")


def cell_bytes(line, index_bound, path, allow_negative=False):
    bits, neighbor = map(int, line[-2:])
    require(0 <= bits <= 0xffffffff and 0 <= neighbor < index_bound,
            f"{path}: invalid output bit word or neighbor")
    value = struct.unpack("!f", struct.pack("!I", bits))[0]
    require(math.isfinite(value) and (allow_negative or value >= 0),
            f"{path}: nonfinite or invalid negative distance")
    return struct.pack("!II", bits, neighbor)


def expected_check_records(selector):
    yield ["SMALLK_DISPATCH", "experimental", str(selector), "mode", "IDENTICAL",
           "specialized_k", "8,10,16", "fallback_k", "4,15"]
    for profile, n in (("dyadic", 257), ("duplicates", 1025)):
        for q in (1, 257, 1000):
            for k in (4, 8, 10, 15, 16):
                fixture = [profile, str(n), str(q), "8", str(k)]
                yield ["DISPATCH_CALL", *fixture, "0", "tile"]
                for cell in range(q * k):
                    yield ["DISPATCH_CELL", *fixture, str(cell)]
                yield ["DISPATCH_CALL", *fixture, "1", "tile"]
                yield ["DISPATCH_FIXTURE_PASS", *fixture]
    yield ["KNN", "SMALL-K", "PUBLIC", "DISPATCH", "PASS", "fixtures", "30",
           "selected_pairs", "133348"]
    # Current distance_ops IDs, in the explicit driver's invocation order:
    # L2Expanded, L2SqrtExpanded, L1, CosineExpanded.
    for metric in (0, 1, 3, 2):
        for cell in range(257 * 10):
            yield ["LAYOUT_CELL", str(metric), "65", "257", "17", "10", str(cell)]
    yield ["KNN", "LAYOUT", "PUBLIC", "DISPATCH", "PASS", "metric_fixtures", "4",
           "additional_selected_pairs", "10280"]


def parse_check(path, arm):
    data = path.read_bytes()
    lines = [line.split() for line in data.decode("utf-8").splitlines() if line.strip()]
    require(lines, f"{path}: empty check log")
    flags(lines, arm, path)
    # Ignore ordinary runtime diagnostics, never misspelled/extra protocol records.
    reserved = ("SMALLK_", "DISPATCH_", "LAYOUT_", "KNN", "PRICE_")
    records = [line for line in lines[1:] if line[0].startswith(reserved)]
    expected = iter(expected_check_records(ARMS[arm][0]))
    outputs = {"DISPATCH_CELL": bytearray(), "LAYOUT_CELL": bytearray()}
    for line in records:
        want = next(expected, None)
        require(want is not None, f"{path}: extra check record")
        if want[0] in outputs:
            require(len(line) == len(want) + 2 and line[:-2] == want,
                    f"{path}: missing, duplicate, reordered or malformed selected cell")
            outputs[want[0]].extend(cell_bytes(line, int(want[2]), path,
                                               allow_negative=want[0] == "LAYOUT_CELL"))
        elif want[0] == "DISPATCH_CALL":
            require(len(line) == len(want) + 1 and line[:-1] == want,
                    f"{path}: incorrect dispatch call order/fixture")
            tile = int(line[-1])
            maximum = 256 if want[6] == "0" else 128
            require(1 <= tile <= min(int(want[3]), maximum),
                    f"{path}: invalid dispatch query tile")
        else:
            require(line == want, f"{path}: incorrect check header/pass marker/order")
    require(next(expected, None) is None, f"{path}: incomplete check records")
    require(records and lines[-1] == records[-1], f"{path}: check pass must be final record")
    witness = {kind: bytes(value) for kind, value in outputs.items()}
    require(len(witness["DISPATCH_CELL"]) == 133348 * 8
            and len(witness["LAYOUT_CELL"]) == 10280 * 8,
            f"{path}: wrong check cell counts")
    return witness, {"log": path.name, "sha256": digest(data),
                     "dispatch_selected_pairs": 133348, "layout_selected_pairs": 10280,
                     "output_sha256": {kind: digest(value) for kind, value in witness.items()}}


def parse_price(path, queries, arm):
    # The established parser gates the old activation header, fixture, positive
    # timing and complete output. Add four-arm activation plus strict cell order.
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    require(lines, f"{path}: empty price log")
    flags(lines, arm, path)
    selector = ARMS[arm][0]
    cells, sample = parse_log(path, queries, "experimental" if selector else "legacy")
    protocol = [line for line in lines[1:] if line[0].startswith(
        ("SMALLK_", "PRICE_", "KNN", "LAYOUT_", "DISPATCH_"))]
    expected_kinds = ["SMALLK_PRICE", *(["PRICE_CELL"] * (queries * 10)), "PRICE_MS", "KNN"]
    require([line[0] for line in protocol] == expected_kinds,
            f"{path}: missing, extra or reordered pricing protocol records")
    require([int(line[6]) for line in protocol if line[0] == "PRICE_CELL"]
            == list(range(queries * 10)), f"{path}: selected cells are out of order")
    return cells, sample


def validate_directory(directory):
    directory = Path(directory)
    require((directory / "completion.txt").read_text() == "COMPLETE\n",
            f"{directory}: missing exact completion marker")
    prices = {f"q{q}-r{r}-{arm}" for q in QUERIES for r in ROUNDS for arm in ARMS}
    builds = {f"build-{kind}-{arm}" for kind in ("check", "price") for arm in ARMS}
    checks = {f"check-{arm}" for arm in ARMS}
    expected = prices | builds | checks
    statuses = {}
    for line in (directory / "status.tsv").read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == 2 and fields[0] not in statuses,
                f"{directory}: malformed or duplicate status record")
        statuses[fields[0]] = fields[1]
    require(set(statuses) == expected and set(statuses.values()) == {"0"},
            f"{directory}: expected all 120 successful build/check/price statuses")
    campaign_logs = {p.stem for pattern in ("q*-r*-*.log", "check-*.log", "build-*.log")
                     for p in directory.glob(pattern)}
    require(campaign_logs == expected, f"{directory}: missing or extra campaign logs")
    build_hashes = {name + ".log": digest((directory / (name + ".log")).read_bytes())
                    for name in sorted(builds)}
    check_witness = None
    check_reports = {}
    for arm in ARMS:
        current, report = parse_check(directory / f"check-{arm}.log", arm)
        if check_witness is None:
            check_witness = current
        require(current == check_witness, f"{directory}: complete check outputs differ for {arm}")
        check_reports[arm] = report
    witnesses = {}
    shapes = {}
    for q in QUERIES:
        samples = {arm: [] for arm in ARMS}
        for r in ROUNDS:
            for arm in ARMS:
                cells, sample = parse_price(directory / f"q{q}-r{r}-{arm}.log", q, arm)
                if q not in witnesses:
                    witnesses[q] = cells
                require(cells == witnesses[q], f"{directory}: q{q} r{r} {arm} output bytes differ")
                samples[arm].append({"round": r, **sample})
        ratios = {}
        for arm in ARMS:
            raw = [samples["baseline"][r]["milliseconds"] / samples[arm][r]["milliseconds"]
                   for r in ROUNDS]
            ratios[arm] = {"raw": raw, **distribution(raw)}
        shapes[str(q)] = {
            "selected_pairs": q * 10,
            "output_sha256": digest(b"".join(struct.pack("!II", *cell) for cell in witnesses[q])),
            "samples": samples,
            "milliseconds": {arm: distribution([s["milliseconds"] for s in samples[arm]])
                             for arm in ARMS},
            "paired_baseline_over_arm": ratios,
        }
    return {"directory": str(directory.resolve()), "status": "PASS", "status_count": 120,
            "price_log_count": 108, "check_log_count": 4, "build_log_count": 8,
            "activation": {arm: {"selector": flags_[0], "transpose": flags_[1]}
                           for arm, flags_ in ARMS.items()},
            "scope": "native-public-upload-search-download-synchronize",
            "quartile_method": "inclusive linear interpolation",
            "ratio_direction": "baseline milliseconds / arm milliseconds; above 1 favors arm",
            "status_sha256": digest((directory / "status.tsv").read_bytes()),
            "build_log_sha256": build_hashes, "checks": check_reports,
            "shapes": shapes}, {"checks": check_witness, "prices": witnesses}


def compare_directories(directory, other=None):
    report, witnesses = validate_directory(directory)
    root = Path(__file__).resolve().parent.parent
    result = {"status": "PASS", "primary": report,
              "local_source_sha256": {name: digest((root / name).read_bytes()) for name in SOURCE_FILES},
              "source_hash_scope": "local parser and selected driver/dependency files at comparison time; not remote build attestation"}
    if other is not None:
        other_report, other_witnesses = validate_directory(other)
        require(witnesses == other_witnesses, "cross-vendor full check/price output bytes differ")
        result.update(other_vendor=other_report, cross_vendor_bits="PASS",
                      cross_vendor_speed_comparison="not performed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--other-vendor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_directories(args.directory, args.other_vendor)
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered)
        print(rendered, end="")
    except (ValueError, OSError, UnicodeError, OverflowError, IndexError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
