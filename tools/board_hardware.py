#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BOX TABLE and the per-row hardware / version fields of every board.

Andrew 2026-10-10: every row names its exact hardware and version; never mix GPU models in a column.

One place for what each lq box is (tools/main_board_ingest.py, tools/bench_board.py and the release boards read it):

    box   GPU (exact model)       provider    lq job ids   main-board column
    nv    NVIDIA L40S             RunPod      n####        nvidia-l40s   (released 2026-10-09)
    nv2   NVIDIA L40S             RunPod      v####        nvidia-l40s
    amd   AMD Instinct MI325X     DO          a####        amd-mi325x
    amd2  AMD Instinct MI300X     Hot Aisle   b####        amd-mi300x
    m2    Apple M2 Pro            AWS EC2     (Mac queue)  -
    m3    Apple M3 Ultra          AWS EC2     (Mac queue)  -
    local Apple M4                local       -            -

Every board row (ours and each opponent) carries:
  hardware         text, e.g. "NVIDIA L40S (nv2, RunPod)", "CPU AMD EPYC 9554 64-Core Processor (host cc560ebdaf91,
                   NVIDIA L40S box)", or "unknown (copied from <source>)": never a guess
  hardware_key     the GPU model the row's machine carries ("NVIDIA L40S"), None when unknown
  hardware_cpu     a CPU arm's CPU model (and our host's, where recorded)
  hardware_source  where the hardware came from (box table, stored record, cell record, source board box,
                   opponent-archive)
  version          ours: "mojolearn <version> ..." with the commit; an opponent: "<library> <version>"
  version_source   where the version came from
A ratio is computed only between rows of the same hardware_key (a CPU arm: the same machine class, and the same CPU
model when both are recorded); any other opponent row is NOT-COMPARABLE (different hardware): `hardware_comparable`
False, no ratio. Standard library only.
"""
from __future__ import annotations

import gzip
import json
import os
import re

RULE = "Andrew 2026-10-10: every row names its exact hardware and version; never mix GPU models in a column"
NOT_COMPARABLE = "NOT-COMPARABLE (different hardware)"

# Andrew 2026-10-10: every row names its exact hardware and version; never mix GPU models in a column.
BOXES = {
    "nv": {"vendor": "nvidia", "gpu": "NVIDIA L40S", "arch": "sm_89", "provider": "RunPod", "job_prefix": "n",
           "column": "nvidia-l40s", "note": "released 2026-10-09"},
    "nv2": {"vendor": "nvidia", "gpu": "NVIDIA L40S", "arch": "sm_89", "provider": "RunPod", "job_prefix": "v",
            "column": "nvidia-l40s"},
    "amd": {"vendor": "amd", "gpu": "AMD Instinct MI325X", "arch": "gfx942", "provider": "DO", "job_prefix": "a",
            "column": "amd-mi325x"},
    "amd2": {"vendor": "amd", "gpu": "AMD Instinct MI300X", "arch": "gfx942", "provider": "Hot Aisle",
             "job_prefix": "b", "column": "amd-mi300x"},
    "m2": {"vendor": "apple", "gpu": "Apple M2 Pro", "arch": "metal", "provider": "AWS EC2", "job_prefix": None,
           "column": None},
    "m3": {"vendor": "apple", "gpu": "Apple M3 Ultra", "arch": "metal", "provider": "AWS EC2", "job_prefix": None,
           "column": None},
    "local": {"vendor": "apple", "gpu": "Apple M4", "arch": "metal", "provider": "local", "job_prefix": None,
              "column": None},
}

# Andrew 2026-10-10: every row names its exact hardware and version; never mix GPU models in a column.
#: The main-board columns, one per GPU MODEL (never per vendor).
COLUMNS = {
    "nvidia-l40s": {"vendor": "nvidia", "api": "cuda", "name": "NVIDIA L40S"},
    "amd-mi325x": {"vendor": "amd", "api": "hip", "name": "AMD Instinct MI325X"},
    "amd-mi300x": {"vendor": "amd", "api": "hip", "name": "AMD Instinct MI300X"},
}

#: Model tokens of a recorded GPU name (lower case, alphanumerics only) -> the canonical model.
MODEL_TOKENS = (
    ("mi325x", "AMD Instinct MI325X"), ("mi300x", "AMD Instinct MI300X"), ("mi300a", "AMD Instinct MI300A"),
    ("mi250x", "AMD Instinct MI250X"), ("mi250", "AMD Instinct MI250"), ("mi210", "AMD Instinct MI210"),
    ("mi355x", "AMD Instinct MI355X"), ("l40s", "NVIDIA L40S"), ("rtx4090", "NVIDIA GeForce RTX 4090"),
    ("rtx5090", "NVIDIA GeForce RTX 5090"), ("h100", "NVIDIA H100"), ("h200", "NVIDIA H200"),
    ("a100", "NVIDIA A100"), ("b200", "NVIDIA B200"), ("rtxpro6000", "NVIDIA RTX PRO 6000"),
    ("m4max", "Apple M4 Max"), ("m4pro", "Apple M4 Pro"), ("applem4", "Apple M4"),
    ("m3ultra", "Apple M3 Ultra"), ("m3max", "Apple M3 Max"), ("m2pro", "Apple M2 Pro"), ("m2ultra", "Apple M2 Ultra"),
)
#: Text that names a GPU (a recorded string without any of this is not a GPU name: "gpu", "selftest",
#: "nvidia-smi not found. This is AMD country." on a ROCm box, ...).
_GPU_NAME_RE = re.compile(r"\b(geforce|tesla|quadro|rtx|instinct|radeon|mi\d{3}[a-z]?|h100|h200|a100|b200|l40s?|"
                          r"apple m\d)\b", re.I)


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def canonical_model(name):
    """A recorded GPU name -> the canonical model ("NVIDIA L40S"), the cleaned name for an unlisted GPU, or None
    when the text names no GPU."""
    if not name:
        return None
    n = _norm(name)
    for tok, model in MODEL_TOKENS:
        if tok in n:
            return model
    s = " ".join(str(name).split())
    return s if _GPU_NAME_RE.search(s) else None


def box_for_job(job, default=None):
    """An lq job id -> its box by the id prefix (n nv, v nv2, a amd, b amd2); `default` otherwise."""
    j = str(job or "")
    m = re.match(r"^([A-Za-z])\d{3,}$", j)
    if m:
        for box, b in BOXES.items():
            if b["job_prefix"] == m.group(1):
                return box
    return default


def column_for_box(box, vendor=None):
    b = BOXES.get(box)
    return b["column"] if b else None


def box_hardware(box):
    """'NVIDIA L40S (nv2, RunPod)' / 'Apple M4 (local)'; None for a box not in the table."""
    b = BOXES.get(box)
    if not b:
        return None
    if box == "local":
        return "%s (local)" % b["gpu"]
    return "%s (%s, %s)" % (b["gpu"], box, b["provider"])


def gpu_mismatch(box, vendor=None, recorded=None):
    """None when a run's recorded vendor / GPU name agrees with the box table, else the 'hardware mismatch'
    reason. A recorded text that names no GPU (or only the box's own arch, e.g. gfx942) is not judged."""
    b = BOXES.get(box)
    if b is None:
        return "hardware mismatch: box %r is not in the box table (tools/board_hardware.py)" % box
    if vendor and b["vendor"] != vendor:
        return "hardware mismatch: box %s is %s (%s), the run recorded vendor %s" % (box, b["gpu"], b["vendor"], vendor)
    model = canonical_model(recorded)
    if model is not None and model != b["gpu"]:
        return "hardware mismatch: box %s is %s, the run recorded GPU %s" % (box, b["gpu"], " ".join(str(recorded).split()))
    return None


# ---------------------------------------------------------------------------
# Row fields
# ---------------------------------------------------------------------------

def parse_device_text(text):
    """'gpu (NVIDIA L40S)' -> ('gpu', 'NVIDIA L40S'); 'cpu' -> ('cpu', None)."""
    m = re.match(r"^\s*(gpu|cpu)\s*(?:\((.*)\))?\s*$", str(text or ""))
    return (m.group(1), (m.group(2) or "").strip() or None) if m else (None, None)


def _set(cell, hardware, key, source, cpu=None):
    cell["hardware"] = hardware
    cell["hardware_key"] = key
    cell["hardware_source"] = source
    if cpu:
        cell["hardware_cpu"] = cpu


def our_fields(cell, *, box=None, version_text, gpu_name=None, hostname=None, cpu_model=None, source):
    """Our row: the box table entry when the box is known, else the run's recorded GPU and host."""
    if box in BOXES:
        _set(cell, box_hardware(box), BOXES[box]["gpu"], "box table (tools/board_hardware.py)", cpu_model)
    else:
        model = canonical_model(gpu_name)
        name = " ".join(str(gpu_name).split()) if gpu_name else None
        if name:
            _set(cell, "%s (host %s)" % (name, hostname or "unrecorded"), model, source, cpu_model)
        else:
            _set(cell, "unknown (%s records no GPU)" % source, None, source, cpu_model)
    cell["version"] = version_text
    cell["version_source"] = source
    return cell


def _box_text(board_box):
    host = (board_box or {}).get("host") or {}
    return host.get("hostname")


def archive_index(path):
    """opponent-cells-all.jsonl.gz of one box -> {(arm, family, lane, dataset, median_ms): cell}."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            k = _arch_key(c)
            if k is not None:
                out.setdefault(k, c)
    return out


def _arch_key(c):
    ms = c.get("median_ms")
    if not isinstance(ms, (int, float)):
        return None
    return (c.get("arm"), c.get("family"), c.get("lane"), c.get("dataset"), round(float(ms), 6))


def _pkg_version(packages, library):
    # the pip names the opponent sets install a library under (tools/opponent_wheels.sh): faiss-cpu, cuml-cu12, ...
    for name in (library, library + "-cu12", library + "-cu13", library + "-cpu", library + "-gpu",
                 library.replace("_", "-")):
        if name in (packages or {}):
            return packages[name], name
    return None, None


def opponent_fields(cell, board_box=None, label=None, archive=None):
    """Fill hardware / version of one opponent row from what its run recorded, in order: the stored record the
    row came from, the row's own fields (device_name, library_version), the board box that measured it (only a
    row measured in that run, not a stored one), the opponent archive (the same arm, lane, dataset and median).
    What none of them records stays 'unknown (copied from <source>)'. Returns {field: source}."""
    used = {}
    where = label or "its board"
    stored = cell.get("stored") or {}
    dev = cell.get("device") or parse_device_text(stored.get("device"))[0]
    box = board_box or {}
    measured_here = not stored
    arch = archive.get(_arch_key(cell)) if archive else None
    # -- hardware
    hw = None
    sdev, sname = parse_device_text(stored.get("device"))
    host = stored.get("box") or (_box_text(box) if measured_here else None)
    box_gpu = (box.get("gpu") or {}).get("name") if measured_here else None
    box_cpu = (box.get("host") or {}).get("cpu_model") if measured_here else None
    if sname:
        hw = (sname, "stored record")
    elif cell.get("device_name") and (dev == "cpu" or canonical_model(cell.get("device_name"))):
        hw = (cell["device_name"], "cell record")
    elif dev == "cpu" and box_cpu:
        hw = (box_cpu, "source board box")
    elif dev != "cpu" and box_gpu:
        hw = (box_gpu, "source board box")
    elif arch and arch.get("device_name"):
        hw = (arch["device_name"], "opponent-archive")
    # the machine class (GPU model) a CPU arm ran on: the stored record's machine, else the measuring board's GPU
    # (a stored record is reused only on an exact store key match, and that key holds the machine: the board
    # box's GPU, tools/bench_board_store.py KEY_FIELDS)
    machine = canonical_model(stored.get("machine") or (box.get("gpu") or {}).get("name")) if dev == "cpu" else None
    if hw is not None and hw[1] == "cell record" and (box_cpu if dev == "cpu" else box_gpu) \
            and _norm(hw[0]) != _norm(box_cpu if dev == "cpu" else box_gpu):
        host = None     # the row's own device is not the board box's: it ran elsewhere, host unrecorded
        machine = None  # and its machine class is not known either (never assumed from the board box)
    if hw is None:
        _set(cell, "unknown (copied from %s)" % where, None, "unknown")
    elif dev == "cpu":
        _set(cell, "CPU %s (host %s, %s box)" % (hw[0], host or "unrecorded", machine or "unknown GPU"),
             machine, hw[1], hw[0])
    else:
        model = canonical_model(hw[0])
        _set(cell, "%s (host %s)" % (" ".join(str(hw[0]).split()), host or "unrecorded"), model, hw[1])
    used["hardware"] = cell["hardware_source"]
    # -- version
    lib = cell.get("library") or "?"
    ver, vsrc = cell.get("library_version"), "cell record"
    if not ver and measured_here:
        ver, _name = _pkg_version(box.get("packages"), lib)
        vsrc = "source board box packages"
    if not ver and arch and arch.get("library_version"):
        ver, vsrc = arch["library_version"], "opponent-archive"
    if ver:
        cell["version"], cell["version_source"] = "%s %s" % (lib, ver), vsrc
    else:
        cell["version"], cell["version_source"] = "%s version unknown (copied from %s)" % (lib, where), "unknown"
    used["version"] = cell["version_source"]
    return used


def annotate_release(result):
    """A bench_board run's board (release boards, opponent boards): every row without hardware / version gets
    them from the run's own box record. Rows that already carry them are kept."""
    box = result.get("box") or {}
    gpu = (box.get("gpu") or {}).get("name")
    host = (box.get("host") or {}).get("hostname")
    cpu = (box.get("host") or {}).get("cpu_model")
    mj = box.get("mojolearn") or {}
    wheel = (mj.get("wheel") or {}).get("file")
    commit = ((box.get("repo") or {}).get("commit") or "")[:9]
    ver = "mojolearn %s%s%s" % (mj.get("version") or "version unrecorded",
                                 " (wheel %s)" % wheel if wheel else "",
                                 ", scripts @%s" % commit if commit else "")
    for group in ("races", "extra_races"):
        for rr in (result.get(group) or {}).values():
            for cells in (rr.get("cells") or [], rr.get("infer_cells") or []):
                for c in cells:
                    if c.get("hardware") and c.get("version"):
                        continue
                    if c.get("library") == "mojolearn":
                        if c.get("device") == "cpu":
                            _set(c, "CPU %s (host %s)" % (cpu or "unrecorded", host or "unrecorded"),
                                 canonical_model(gpu), "board box", cpu)
                            c["version"], c["version_source"] = ver, "board box"
                        else:
                            our_fields(c, version_text=ver, gpu_name=c.get("device_name") or gpu, hostname=host,
                                       cpu_model=cpu, source="board box")
                    else:
                        opponent_fields(c, box, "this board's record")
    return result


def comparable(ours, c):
    """True / False: may a ratio divide our row by row c? None when ours records no hardware (old boards)."""
    if ours is None or "hardware_key" not in ours:
        return None
    a, b = ours.get("hardware_key"), c.get("hardware_key")
    if not a or not b or a != b:
        return False
    if c.get("device") == "cpu" and ours.get("hardware_cpu") and c.get("hardware_cpu") \
            and ours["hardware_cpu"] != c["hardware_cpu"]:
        return False
    return True


def mark(cells):
    """Set `hardware_comparable` on every opponent row against our IDENTICAL (else FAST) GPU row, in place."""
    ours = next((c for c in cells if c.get("library") == "mojolearn" and c.get("mode") == "identical"
                 and c.get("device") != "cpu"), None) or \
        next((c for c in cells if c.get("library") == "mojolearn" and c.get("device") != "cpu"), None)
    for c in cells:
        if c.get("library") == "mojolearn":
            continue
        v = comparable(ours, c)
        if v is None:
            c.pop("hardware_comparable", None)
        else:
            c["hardware_comparable"] = v
    return cells
