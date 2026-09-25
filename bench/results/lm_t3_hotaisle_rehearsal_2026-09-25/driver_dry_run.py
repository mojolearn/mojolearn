"""The driver's A/4 start on Hot Aisle after a stop at checkpoint 2600, with
every subprocess mocked: nothing is rendered, uploaded or rented.

Inputs (read only): ~/mojolearn-evidence/gpt3-run/t3_spec.json, the real
ledger's A/1 to A/3 entries (~/mojolearn-evidence/gpt3-run/t3/ledger.json),
and the hung attempt's chain, manifest and log
(~/mojolearn-evidence/gpt3-run/t3/legs/A-4/hang-attempt-1/). The partial
entry is written by the driver's own record_partial from a scratch copy of
those files laid out the way a runner fetches them. The R2 size check of
record_partial is off in the scratch spec only (the driver would GET one
byte of each checkpoint under the run's prefix; this dry run reads nothing
there).

    PYTHONPATH=python .pixi/envs/test/bin/python \
        bench/results/lm_t3_hotaisle_rehearsal_2026-09-25/driver_dry_run.py <scratch dir>
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
EV = Path.home() / "mojolearn-evidence" / "gpt3-run"
spec_mod = importlib.util.spec_from_file_location("lm_run_driver", REPO / "tools" / "lm_run_driver.py")
drv = importlib.util.module_from_spec(spec_mod)
sys.argv = [sys.argv[0]] + sys.argv[1:]
spec_mod.loader.exec_module(drv)

scratch = Path(sys.argv[1]).resolve()
if scratch.exists():
    shutil.rmtree(scratch)
out = scratch / "out"
out.mkdir(parents=True)

# the scratch spec: A/4 on Hot Aisle
spec = json.loads((EV / "t3_spec.json").read_text())
a4 = next(s for s in spec["routes"]["A"] if s["segment"] == "4")
a4["provider"] = "hotaisle"
spec["resume_check_r2"] = False
spec_path = scratch / "t3_spec.hotaisle.json"
spec_path.write_text(json.dumps(spec, indent=1) + "\n")
spec = drv.load_spec(spec_path)

# the scratch ledger: A/1 to A/3 as the real ledger holds them
real = json.loads((EV / "t3" / "ledger.json").read_text())
ledger = drv.Ledger(out)
for seg in ("1", "2", "3"):
    ledger.land(dict(route="A", segment=seg), real["landed"]["A/%s" % seg])

# the hung attempt, laid out as a runner fetches it (no segment.json)
e = next(x for x in drv.segment_plan(spec) if (x["route"], x["segment"]) == ("A", "4"))
hang = EV / "t3" / "legs" / "A-4" / "hang-attempt-1"
res = out / "legs" / "A-4" / "leg-do-1"
box = res / "remote" / "lm-segment-A-4"
(box / "segment").mkdir(parents=True)
for name in ("chain.jsonl", "manifest.tsv", "log.txt"):
    shutil.copyfile(hang / name, box / "segment" / name)
(box / "arrival").mkdir()
(box / "arrival" / "segment.json").write_text(json.dumps(dict(verdict="PASS")) + "\n")
part = drv.record_partial(spec, e, res, out, ledger)
print("== the partial entry (ledger.json 'partial'['A/4'])")
print(json.dumps(part, indent=1))

# A/4's next start, every subprocess mocked
calls = []


def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, check=None, **kw):
    calls.append((list(map(str, cmd)), env))
    return mock.Mock(returncode=0, stdout="https://presigned.example/put\n", stderr="")


with mock.patch.object(drv.subprocess, "run", side_effect=fake_run), mock.patch.object(drv.time, "sleep"):
    got = drv._start(spec, e, out, ledger)

renders = [(c, env) for c, env in calls if len(c) > 2 and c[1].endswith("lm_segment_leg.py")]
puts = [c for c, env in calls if c[0] == "curl" and "-T" in c]
legs = [(c, env) for c, env in calls if c[0] == "bash"]


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


print("\n== the renders (the second is the Hot Aisle body)")
for c, _ in renders:
    print(" ".join(c))
    print("   --from %s  --steps %s  --devices %s  --from-sha %s  --from-key %s  --expect-key %s  --out %s"
          % tuple(flag(c, f) for f in ("--from", "--steps", "--devices", "--from-sha", "--from-key", "--expect-key", "--out")))
print("\n== the partial chain the driver PUTs first (mocked; its R2 key is the --expect-key)")
for c in puts:
    print("   curl -T %s <presigned PUT>" % c[c.index("-T") + 1])
print("\n== the leg")
for c, env in legs:
    keys = sorted(k for k in env if k.startswith("MOJOLEARN_") and env.get(k) != __import__("os").environ.get(k))
    print(" ".join("%s=%s" % (k, env[k]) for k in keys))
    print("   " + " ".join(c))
print("\n== _start returned", got)
print("\n== driver.log")
print((out / "driver.log").read_text())
