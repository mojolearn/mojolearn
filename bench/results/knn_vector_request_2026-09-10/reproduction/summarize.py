"""Summarize same-process request arms; no opponent process is timed here."""
import json
import statistics
import sys
from pathlib import Path

result = {}
for filename in sys.argv[1:]:
    path = Path(filename)
    lines = path.read_text().splitlines()
    assert any(line.startswith("FULL_OUTPUT_BITS_MATCH") for line in lines), path
    samples = {0: {}, 1: {}, 2: {}}
    for line in lines:
        fields = line.split()
        if fields[:2] == ["KNN_REF_ROUND", "request"]:
            samples[int(fields[4])][int(fields[2])] = float(fields[5])
    assert samples[0].keys() == samples[1].keys() and len(samples[0]) >= 7, path
    if samples[2]:
        assert samples[2].keys() == samples[0].keys(), path
    medians = [statistics.median(samples[a].values()) for a in (0, 1)]
    paired = [samples[1][r] / samples[0][r] for r in samples[0]]
    default_ms = statistics.median(samples[2].values()) if samples[2] else None
    result[path.name] = {
        "default_median_ms": default_ms,
        "pairs": len(paired), "scalar_median_ms": medians[0],
        "vector_median_ms": medians[1], "median_ratio": medians[1] / medians[0],
        "paired_ratio_median": statistics.median(paired),
        "vector_faster_pairs": sum(x < 1 for x in paired),
        "paired_ratio_min": min(paired), "paired_ratio_max": max(paired),
    }
print(json.dumps(result, indent=2))
