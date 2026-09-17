# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What bench/model/harness.py (ours) and bench/model/torch_twin.py (the
incumbent) share, so the two records carry the same fields and the same
timing rule and bench/model/diff.py can read either.

THE RECORD (schema `mojolearn.model_leg.v1`), one JSON per column per model:

  schema, kind ("ours" | "torch"), column (the label the leg passed),
  stamp_utc, commit, box {hostname, os, arch, cpu_model, gpu},
  model {id, path, config_sha256, weights_sha256, files}, protocol
  {max_new, greedy, runs, statistic, warm_prompt, prompts_sha256},
  library {...} (ours: mojolearn version, numeric_mode and vendor READ
  BACK from the binaries, per-binding numeric_mode read-backs, binding
  sha256s; torch: torch, transformers, cuda, hip, device, dtype and every
  determinism switch read back), and

  arms: { "<arm>": { "prompts": { "<prompt id>": CELL } } }

where an arm is a weight format for ours ("float32", "bfloat16", "int8") and
an incumbent setting for torch ("torch-bf16-shipped", "torch-deterministic",
"torch-fp32-cpu-shipped", ...). A CELL:

  n_prompt_tokens, n_generated, ids_sha256 (the generated ids, all runs
  agreeing), ids_sha256_runs (per run), first_logits_sha256 (the bytes of
  the logits that chose the first generated token, float32), text (the
  decoded continuation, for reading, never for the verdict),
  prefill_ms_runs, decode_ms_runs (per run, whole prompt and whole
  continuation), prefill_ms_per_token, decode_ms_per_token (medians over
  the runs), prefill_ms_per_token_range, decode_ms_per_token_range,
  verdict ("STABLE" when every run hashed the same ids, "MOVED" otherwise,
  "REFUSED" with `error` when the arm could not run the prompt).

TIMING RULE. The first prompt is generated once, untimed (the warm-up).
Every prompt then runs `runs` times (three by default); each run times the
prefill (one forward over the prompt) and the greedy generation separately,
and the record keeps every run and the median. Nothing here times a
download: the model is on disk before the first clock starts, and the
harness refuses to run against a path that does not exist.
"""
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

SCHEMA = "mojolearn.model_leg.v1"

#: the arm names torch_twin.py writes; diff.py --ratio reads any arm whose
#: name starts with "torch"
TORCH_FAST_ARMS = ("torch-bf16-shipped", "torch-fp32-cpu-shipped", "torch-bf16-mps-shipped",
                   "torch-fp32-mps-shipped")
TORCH_DETERMINISTIC_ARM = "torch-deterministic"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids):
    """The generated ids as little-endian int32, hashed. A list of Python
    ints, so the same ids hash the same whatever produced them."""
    h = hashlib.sha256()
    for t in ids:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()


def bytes_of(x):
    """The raw bytes of a buffer-like object: mojolearn.Array (tobytes),
    numpy, torch (after .cpu().contiguous()), memoryview, bytes."""
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if hasattr(x, "tobytes"):
        return x.tobytes()
    return memoryview(x).tobytes()


def read_prompts(path):
    """bench/model/prompts.txt: one prompt per line as `<id><TAB><text>`.
    Lines starting with `##` are comments; blank lines are skipped. The text
    is taken verbatim after the tab (leading and trailing spaces kept), and
    the two-character escapes `\\n` and `\\t` are expanded so a prompt can
    carry a newline or a tab on one line. Returns [(id, text)] in file
    order and refuses duplicate ids."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("##"):
                continue
            if "\t" not in line:
                raise SystemExit(f"{path}:{n}: expected `<id><TAB><text>`")
            pid, text = line.split("\t", 1)
            pid = pid.strip()
            if not pid or pid in seen:
                raise SystemExit(f"{path}:{n}: empty or duplicate prompt id {pid!r}")
            seen.add(pid)
            out.append((pid, text.replace("\\n", "\n").replace("\\t", "\t")))
    if not out:
        raise SystemExit(f"{path}: no prompts")
    return out


def prompts_sha256(path):
    return sha256_file(path)


def git_commit(default=""):
    """The commit this checkout is at. A leg's box has no .git (the runners
    ship a git archive), so MOJOLEARN_COMMIT, then commit.txt at the tree
    root, then /root/gemm_leg_out/leg.txt's commit= line are read first,
    exactly as tools/identity_three_columns_leg.sh resolves it."""
    env = os.environ.get("MOJOLEARN_COMMIT", "").strip()
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (os.path.join(root, "commit.txt"), os.path.join(root, "COMMIT")):
        try:
            with open(cand) as fh:
                c = fh.readline().strip()
                if c:
                    return c
        except OSError:
            pass
    try:
        with open("/root/gemm_leg_out/leg.txt") as fh:
            for line in fh:
                if line.startswith("commit="):
                    c = line[len("commit="):].strip()
                    if c:
                        return c
    except OSError:
        pass
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return default


def cpu_model():
    """`sysctl -n machdep.cpu.brand_string` on macOS; `model name` from
    /proc/cpuinfo, then lscpu "Model name" on Linux (the recipe of
    tools/identity_break.py cpu_model)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() or None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def gpu_names():
    """What nvidia-smi or rocm-smi say this box carries, as a string; the
    Metal column names the CPU model (the SoC is the GPU); None with
    neither tool. /dev/dri is not AMD evidence, only /dev/kfd or a tool is."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            return "nvidia: " + "; ".join(l.strip() for l in out.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True,
                             timeout=20)
        if out.returncode == 0:
            names = [l.strip() for l in out.stdout.splitlines() if "Card" in l or "series" in l.lower()]
            if names:
                return "amd: " + "; ".join(names)
    except (OSError, subprocess.SubprocessError):
        pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple: " + (cpu_model() or "Apple silicon")
    return None


def box_record():
    return dict(hostname=platform.node(), os=platform.platform(), arch=platform.machine(),
                cpu_model=cpu_model(), gpu=gpu_names(), python=sys.version.split()[0])


def model_record(path, model_id):
    """config_sha256 is over the canonical JSON of config.json (sorted keys,
    no whitespace), so the same configuration hashes the same whatever the
    file's formatting; weights_sha256 is over the sorted list of
    (basename, sha256) of every *.safetensors and *.bin under the path,
    which is the model's bytes and nothing else."""
    cfg_path = os.path.join(path, "config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    files = []
    for name in sorted(os.listdir(path)):
        if name.endswith((".safetensors", ".bin")) and os.path.isfile(os.path.join(path, name)):
            files.append([name, sha256_file(os.path.join(path, name))])
    weights = sha256_bytes(json.dumps(files, separators=(",", ":")).encode("utf-8")) if files else None
    return dict(id=model_id, path=os.path.abspath(path), config_sha256=sha256_bytes(canon),
                weights_sha256=weights, files=files, architectures=cfg.get("architectures"),
                vocab_size=cfg.get("vocab_size"), max_position_embeddings=cfg.get("max_position_embeddings"),
                torch_dtype=cfg.get("torch_dtype"))


def require_model_dir(path):
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "config.json")):
        raise SystemExit(f"REFUSING: {path} is not a model directory with config.json; the model is "
                         "fetched BEFORE the harness starts, so no download is ever inside a timing")


class Clock:
    """perf_counter milliseconds around a callable, with an optional device
    synchronize called before the start and after the end (torch needs it;
    mojolearn's calls return after the bytes are on the host)."""

    def __init__(self, sync=None):
        self.sync = sync

    def ms(self, fn, *a, **k):
        if self.sync:
            self.sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        if self.sync:
            self.sync()
        return (time.perf_counter() - t0) * 1e3, r


def median_and_range(values):
    if not values:
        return None, None
    return statistics.median(values), [min(values), max(values)]


def cell_from_runs(n_prompt, n_generated, ids_runs, first_logits_sha, text, prefill_ms_runs,
                   decode_ms_runs, extra=None):
    """Fold the per-run measurements of one prompt into a CELL."""
    hashes = [sha256_ids(r) for r in ids_runs]
    verdict = "STABLE" if len(set(hashes)) == 1 else "MOVED"
    pre_tok = [p / max(n_prompt, 1) for p in prefill_ms_runs]
    dec_tok = [d / max(n_generated, 1) for d in decode_ms_runs]
    pm, pr = median_and_range(pre_tok)
    dm, dr = median_and_range(dec_tok)
    cell = dict(n_prompt_tokens=n_prompt, n_generated=n_generated, ids_sha256=hashes[0],
                ids_sha256_runs=hashes, first_logits_sha256=first_logits_sha, text=text,
                prefill_ms_runs=prefill_ms_runs, decode_ms_runs=decode_ms_runs,
                prefill_ms_per_token=pm, prefill_ms_per_token_range=pr,
                decode_ms_per_token=dm, decode_ms_per_token_range=dr, verdict=verdict)
    if extra:
        cell.update(extra)
    return cell


def refused_cell(error, n_prompt=None):
    return dict(verdict="REFUSED", error=str(error)[:600], n_prompt_tokens=n_prompt)


def new_record(kind, column, model, protocol, library):
    return dict(schema=SCHEMA, kind=kind, column=column,
                stamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                commit=git_commit(), box=box_record(), model=model, protocol=protocol,
                library=library, arms={}, complete=False)


def write_record(path, record):
    """Written after every arm, so a run cut by a lease keeps what it has
    (`complete` false), as tools/identity_break.py --json does."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def split_generated(prompt_ids, out_ids):
    """`generate` may return the prompt followed by the continuation, or the
    continuation alone. Both are accepted and the record says which
    (`generate_returns`)."""
    out = [int(t) for t in out_ids]
    n = len(prompt_ids)
    if len(out) >= n and out[:n] == [int(t) for t in prompt_ids]:
        return out[n:], "prompt+continuation"
    return out, "continuation"


def flatten_row(x):
    """A (1, L) or (L,) sequence of ids to a flat Python list of ints."""
    try:
        seq = list(x)
    except TypeError:
        return [int(x)]
    # (1, L) -> L; (1, 1) -> 1; a flat row is left alone
    while len(seq) == 1 and hasattr(seq[0], "__len__") and not isinstance(seq[0], (bytes, str)):
        seq = list(seq[0])
    return [int(t) for t in seq]
