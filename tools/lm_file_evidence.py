#!/usr/bin/env python3
"""File a GPT-3 Small run's evidence from the driver's out directory into
bench/results, small files only.

    PYTHONPATH=python python tools/lm_file_evidence.py \
        --spec ~/mojolearn-evidence/gpt3-run/t3_spec.json \
        --out ~/mojolearn-evidence/gpt3-run/t3 \
        --dest bench/results/lm_t3_2026-09-23

What goes in (the layout bench/results/lm_t2_2026-09-23 was filed by hand):
  spec.json, recipe.json, ledger.json, driver.log at the top;
  one directory per box, <route>-<segment> for a one-box segment and
  <route>-<segment>-<vendor> for each box of a live segment, plus
  <route>-<segment>-live.log (the live orchestrator's log), holding
  status.txt, leg.txt, gpu.txt, uname.txt, binding.txt, commit.txt,
  wheel.sha256, checkpoints.sha256, uploads.json, provider.txt (the rental's
  provider and box id, read from the leg's own files), and under segment/ and
  arrival/ the segment.json, manifest.tsv and log.txt.

The chain (chain.jsonl, about 1.2 KB a step, 1.2 MB for a 1,000-step
segment) is not copied. Each one is reduced to chain.summary.tsv, one row a
step with the learning-rate bits, the mean loss, the seconds, the hash
seconds, the state and gradient digests and a digest of the 64 per-shard
losses, beside chain.sha256 (the digest, line count and byte count of the
file the box wrote). The full chain is in R2 under the run's prefix and in
the driver's out directory; the summary keeps every digest a cross-vendor
comparison reads. Any other file above --cap bytes is skipped and listed in
SKIPPED.txt with its size.

A segment the driver resumed after an attempt that did not land (see
tools/lm_run_driver.py) also files each earlier attempt's small files and
chain summary under <route>-<segment>/segment-attempt-<n>/, with a row of
its own.

Only PASSED, landed segments are filed (the ledger decides); --segments
narrows to a list like A/1,A/2. Rerunning adds newly landed segments and
rewrites what it filed before. The tool prints one README table row per
filed segment; the README is written by hand.
"""
import argparse
import hashlib
import json
import shutil
import statistics
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import lm_run_driver as drv  # noqa: E402

BOX_FILES = ("status.txt", "leg.txt", "gpu.txt", "uname.txt", "binding.txt", "commit.txt",
             "wheel.sha256", "checkpoints.sha256", "uploads.json")
SUBDIR_FILES = ("segment.json", "manifest.tsv", "log.txt")
SUBDIRS = ("segment", "arrival")


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _f32(hexs):
    return struct.unpack(">f", bytes.fromhex(hexs))[0]


def summarize_chain(src, dest_dir):
    """chain.jsonl -> chain.summary.tsv + chain.sha256; returns the rows' seconds and hash seconds."""
    rows, secs, hsecs = [], [], []
    n_lines = 0
    for line in Path(src).read_text().splitlines():
        if not line.strip():
            continue
        n_lines += 1
        d = json.loads(line)
        losses = d.get("losses_f32_hex") or []
        mean = statistics.fmean(_f32(x) for x in losses) if losses else float("nan")
        losses_digest = hashlib.sha256(",".join(losses).encode()).hexdigest()[:16]
        rows.append("\t".join([str(d.get("step")), str(d.get("lr_f32_hex", "")), "%.6f" % mean,
                               "%.3f" % float(d.get("seconds", 0.0)), "%.3f" % float(d.get("hash_seconds", 0.0)),
                               str(d.get("state_sha256", "")), str(d.get("gradient_sha256", "")), losses_digest,
                               str(d.get("hash_scheme", "")), str(d.get("label", ""))]))
        secs.append(float(d.get("seconds", 0.0)))
        hsecs.append(float(d.get("hash_seconds", 0.0)))
    header = "\t".join(["step", "lr_f32_hex", "loss_mean", "seconds", "hash_seconds", "state_sha256",
                        "gradient_sha256", "losses_sha256_16", "hash_scheme", "label"])
    (dest_dir / "chain.summary.tsv").write_text("\n".join([header] + rows) + "\n")
    (dest_dir / "chain.sha256").write_text("%s  chain.jsonl\nlines %d\nbytes %d\n" % (
        _sha256_file(src), n_lines, Path(src).stat().st_size))
    return secs, hsecs


def _roots(box_root, results):
    """Where a box's small files may sit: the box's output directory first, then each
    ancestor up to and including the rental's results directory (the runner writes leg.txt
    and commit.txt beside the fetched `remote/` tree; the box writes its own leg.txt in it)."""
    roots = [Path(box_root)]
    p = Path(box_root)
    while p != Path(results) and p.parent != p:
        p = p.parent
        roots.append(p)
    return roots


def _copy(src, dest, cap, skipped, rel):
    size = src.stat().st_size
    if size > cap:
        skipped.append("%s\t%d bytes" % (rel, size))
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return True


def file_box(box_root, results, dest_dir, cap, skipped):
    """One box's small files and its chain summaries; returns (seconds, hash_seconds, arrival verdict, checkpoints)."""
    box_root = Path(box_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    provider, box_id = provider_of(box_root, results)
    (dest_dir / "provider.txt").write_text("provider=%s\nbox_id=%s\n" % (provider, box_id))
    for name in BOX_FILES:
        for root in _roots(box_root, results):
            if (root / name).exists():
                _copy(root / name, dest_dir / name, cap, skipped, "%s/%s" % (dest_dir.name, name))
                break
    secs, hsecs, arrival, ckpts = [], [], None, []
    for sub in SUBDIRS:
        d = box_root / sub
        if not d.is_dir():
            continue
        (dest_dir / sub).mkdir(exist_ok=True)
        for name in SUBDIR_FILES:
            if (d / name).exists():
                _copy(d / name, dest_dir / sub / name, cap, skipped, "%s/%s/%s" % (dest_dir.name, sub, name))
        if (d / "chain.jsonl").exists():
            s, h = summarize_chain(d / "chain.jsonl", dest_dir / sub)
            if sub == "segment":
                secs, hsecs = s, h
        if (d / "segment.json").exists():
            seg = json.loads((d / "segment.json").read_text())
            if sub == "arrival":
                arrival = seg.get("verdict")
            else:
                ckpts = [c["file"] for c in seg.get("checkpoints", []) if isinstance(c, dict)]
        for extra in ("coordinator.jsonl",):
            if (d / extra).exists():
                _copy(d / extra, dest_dir / sub / extra, cap, skipped, "%s/%s/%s" % (dest_dir.name, sub, extra))
    return secs, hsecs, arrival, ckpts


PROVIDER_NAMES = {"digitalocean": "DigitalOcean", "runpod": "RunPod", "hotaisle": "Hot Aisle", "vultr": "Vultr"}


def provider_of(box_root, results):
    """(provider, box id) of the rental that ran a box, from the leg's own files:
    `provider=` in leg.txt where the leg writes it (DigitalOcean, Hot Aisle,
    Vultr), else the id file each runner leaves (droplet_id.txt is a
    DigitalOcean droplet, pod_id.txt a RunPod pod)."""
    provider, box_id = None, None
    for root in _roots(box_root, results):
        leg = root / "leg.txt"
        if leg.exists():
            for line in leg.read_text().splitlines():
                k, _, v = line.partition("=")
                if k == "provider" and v and not provider:
                    provider = v.strip()
                if k in ("droplet", "pod", "vm", "instance") and v and not box_id:
                    box_id = v.strip()
        if (root / "droplet_id.txt").exists():
            provider = provider or "digitalocean"
            box_id = box_id or (root / "droplet_id.txt").read_text().strip()
        if (root / "pod_id.txt").exists():
            provider = provider or "runpod"
            box_id = box_id or (root / "pod_id.txt").read_text().strip()
        if provider and box_id:
            break
    return provider or "unknown", box_id or "?"


def _gpu_name(dest_dir):
    """The box's GPU from gpu.txt: nvidia-smi's first field, or rocm-smi's Card Series."""
    p = dest_dir / "gpu.txt"
    if not p.exists():
        return "?"
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    for ln in lines:
        if "Card Series" in ln:
            return ln.split(":", 2)[-1].strip()
    for ln in lines:
        if not ln.startswith("="):
            return ln.split(",")[0].strip()
    return "?"


def file_segment(e, record, dest, cap, skipped):
    results = Path(record["results"])
    seg_dir, seg = drv._segment_dir(results)
    if seg is None:
        return []
    rows = []
    boxes = [(seg_dir.parent, seg, "")]
    if e["vendor"] == "live":
        since = seg.get("utc_start") or ""
        for p in drv._this_attempt(results, "segment.json"):
            if p.parent.name != "segment":
                continue
            w = json.loads(p.read_text())
            if (w.get("live") or {}).get("role") == "worker" and (w.get("utc_end") or "") >= since:
                boxes.append((p.parent.parent, w, ""))
        live_log = results / "live.log"
        if live_log.exists():
            _copy(live_log, dest / ("%s-%s-live.log" % (e["route"], e["segment"])), cap, skipped,
                  "%s-%s-live.log" % (e["route"], e["segment"]))
    for box_root, sj, _ in boxes:
        label = sj.get("label") or ""
        vendor = label.split("-")[0] if label else "box"
        name = "%s-%s" % (e["route"], e["segment"]) if e["vendor"] != "live" else "%s-%s-%s" % (e["route"], e["segment"], vendor)
        dest_dir = dest / name
        secs, hsecs, arrival, ckpts = file_box(box_root, results, dest_dir, cap, skipped)
        first, last = sj.get("first_step"), sj.get("last_completed") or sj.get("last_step")
        provider, box_id = provider_of(box_root, results)
        rows.append("| %s/%s%s | %s (%s %s) | %s to %s | %s | %s | %s | %s |" % (
            e["route"], e["segment"], "" if e["vendor"] != "live" else " (%s)" % vendor, _gpu_name(dest_dir),
            PROVIDER_NAMES.get(provider, provider), box_id,
            first, last, "%.1f" % statistics.median(secs) if secs else "?",
            "%.1f" % statistics.median(hsecs) if hsecs else "?",
            arrival or ("none (the seed)" if e["from_ckpt"] == "init" else "worker" if (sj.get("live") or {}).get("role") == "worker" else "none"),
            ", ".join(c.replace("ckpt_", "").replace(".blm", "").lstrip("0") or "0" for c in ckpts) or ("the coordinator's" if (sj.get("live") or {}).get("role") == "worker" else "?")))
    # a resumed segment: every earlier attempt's small files (saved by the
    # driver in the box's layout) under segment-attempt-<n>/, its chain
    # reduced to a summary like the landed box's, one row each
    for a in record.get("attempts") or []:
        adir = Path(a["dir"])
        if not adir.is_dir():
            continue
        name = "%s-%s" % (e["route"], e["segment"])
        dest_dir = dest / name / ("segment-attempt-%d" % a["n"])
        secs, hsecs, arrival, _ = file_box(adir, adir, dest_dir, cap, skipped)
        provider, box_id = provider_of(adir, adir)
        rows.append("| %s/%s (attempt %d, did not land) | %s (%s %s) | %s to %s | %s | %s | %s | %s |" % (
            e["route"], e["segment"], a["n"], _gpu_name(dest_dir), PROVIDER_NAMES.get(provider, provider), box_id,
            a["first_step"], a["last_chain_step"], "%.1f" % statistics.median(secs) if secs else "?",
            "%.1f" % statistics.median(hsecs) if hsecs else "?", arrival or "none",
            ", ".join(c.replace("ckpt_", "").replace(".blm", "").lstrip("0") or "0" for c in sorted(a.get("checkpoints") or {})) or "none"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True, help="the driver's --out directory")
    ap.add_argument("--dest", required=True, help="bench/results/lm_t3_<date>")
    ap.add_argument("--segments", default="", help="comma list like A/1,A/2; default every PASSED segment")
    ap.add_argument("--cap", type=int, default=400_000, help="bytes; a larger file is skipped and listed")
    args = ap.parse_args(argv)
    spec = drv.load_spec(args.spec)
    out, dest = Path(args.out), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    skipped = []
    shutil.copyfile(args.spec, dest / "spec.json")
    recipe = Path(spec["recipe"])
    if recipe.exists():
        shutil.copyfile(recipe, dest / "recipe.json")
    for name in ("ledger.json", "driver.log"):
        if (out / name).exists():
            _copy(out / name, dest / name, args.cap, skipped, name)
    ledger = drv.Ledger(out)
    want = set(s.strip() for s in args.segments.split(",") if s.strip())
    rows = ["| segment | box | steps | s per step (median) | hash s | arrival replay | checkpoints |",
            "|---|---|---|---|---|---|---|"]
    filed = []
    for e in drv.segment_plan(spec):
        key = "%s/%s" % (e["route"], e["segment"])
        if want and key not in want:
            continue
        rec = ledger.landed(e)
        if not rec or rec.get("verdict") != "PASS":
            continue
        rows += file_segment(e, rec, dest, args.cap, skipped)
        filed.append(key)
    if skipped:
        (dest / "SKIPPED.txt").write_text("\n".join(skipped) + "\n")
    print("filed %s into %s" % (", ".join(filed) or "nothing", dest))
    if skipped:
        print("skipped %d file(s) over %d bytes; see SKIPPED.txt" % (len(skipped), args.cap))
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
