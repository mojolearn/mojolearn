import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser(description="One fixture, two retained-buffer executions")
p.add_argument("fixture", choices=("base", "batch", "head", "long"))
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--scheduler", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, MOJOLEARN_VALIDATION_FIXTURE=a.fixture,
           MOJOLEARN_IDENTITY_TRACE=str(a.out / "stages.trace"))
env.pop("MOJOLEARN_TRANSFORMER_TIMING", None)
cmd = ["python3", str(a.scheduler), "--deadline", str(time.monotonic() + 60),
       "--timeout", "60", "--wait-timeout", "60", "--timing-json",
       str(a.out / "scheduler.json"), "metal", "/usr/bin/time", "-l", str(a.binary)]
with (a.out / "run.log").open("w") as log:
    result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
text = (a.out / "run.log").read_text()
report = {"fixture": a.fixture, "exit_code": result.returncode,
          "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest()}
(a.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
if result.returncode:
    raise SystemExit(f"FAIL {a.fixture}: preserve logs; no automatic retry")
samples = re.findall(r"execution (\d+) forward_cells (\d+) backward_cells (\d+) waits (\d+) launches (\d+) device_and_dump_ms ([0-9.eE+-]+)", text)
assert len(samples) == 2, "two completed executions required"
assert "PASS: 67 stages match oracle twice; changed inputs reach reused outputs" in text
report["executions"] = [dict(repeat=int(r), forward_cells=int(f), backward_cells=int(b), waits=int(w), launches=int(l), device_and_dump_ms=float(t)) for r,f,b,w,l,t in samples]
report["staging"] = [dict(direction=d, bytes=int(n), previous_largest_stage_bytes=int(m)) for d,n,m in re.findall(r"staging (\w+) bytes (\d+) previous_largest_stage_bytes (\d+)", text)]
assert len(report["staging"]) == 4
rss = re.search(r"(\d+)\s+maximum resident set size", text)
report["process_peak_rss_bytes"] = int(rss[1]) if rss else None
card = a.out / "stages.trace"
rows = [line for line in card.read_text().splitlines() if line and not line.startswith("#")]
assert len(rows) == 67, f"expected 67 stage records, got {len(rows)}"
assert len({row.split("\t")[1] for row in rows}) == 67
report["card_records"] = len(rows)
report["card_sha256"] = hashlib.sha256(card.read_bytes()).hexdigest()
report["result"] = "PASS"
(a.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
