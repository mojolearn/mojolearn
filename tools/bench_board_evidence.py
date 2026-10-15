#!/usr/bin/env python3
"""Retain benchmark recipes, source bytes and logs independently of timings.

This is benchmark tooling, never product execution. Source is captured before
the race; logs and resolved worker records are sealed afterwards. Object names
are content hashes, and R2 uploads are verified by reading the bytes back.
"""
import argparse
import atexit
import hashlib
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

SCHEMA = "mojolearn.measurement.v1"
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "measurements/records/v1"
RECEIPTS = "MOJOLEARN_MEASUREMENT_RECEIPTS"
_registered = False
_libraries = set()
_arms = {}
ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
            "MOJOLEARN_BENCH_CPU_POLICY", "MOJOLEARN_NUMERIC_MODE",
            "MOJOLEARN_GPU_ARCH", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONPATH")


def runtime_environment(environ=None):
    """Retain numerical/runtime switches, excluding credentials and receipt paths."""
    environ = os.environ if environ is None else environ
    prefixes = ("MOJOLEARN_", "CUDA_", "HIP_", "ROCR_", "ROCM_", "OMP_", "OPENBLAS_",
                "MKL_", "TORCH_", "PYTORCH_", "CUBLAS_", "CUDNN_", "NCCL_", "MODULAR_", "MAX_")
    ignored = {RECEIPTS, "MOJOLEARN_BOARD_RECEIPTS", "MOJOLEARN_BOARD_SESSION"}
    return {k: v for k, v in sorted(environ.items())
            if (k in ENV_KEYS or k.startswith(prefixes)) and k not in ignored
            and not any(word in k.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "CREDS", "API_KEY", "R2_"))}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, default=str) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encoded(value))
    tmp.replace(path)


def archive(path, members):
    """Deterministic archive; names are controlled, never command strings."""
    # Full factor arrays make maximum gzip compression spend minutes on
    # retention after subsecond GPU calls. Fast compression preserves all
    # bytes and deterministic archives; it is outside every scored clock.
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                              compresslevel=1, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as out:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                out.addfile(info, io.BytesIO(data))
    return dict(sha256=file_hash(path), bytes=Path(path).stat().st_size,
                local_path=str(Path(path).resolve()), status="local")


def source_snapshot(directory, repo=ROOT, drivers=()):
    repo, directory = Path(repo), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
    suffixes = {".py", ".mojo", ".sh", ".c", ".h", ".cpp", ".hpp", ".toml", ".lock"}
    members, inventory = {}, {}
    for name in names:
        p = repo / name
        if (not name or p.suffix not in suffixes or not p.is_file()
                or name.startswith(("bench/results/", "docs/", "docs-site/"))):
            continue
        members[name] = p.read_bytes()
        inventory[str(p.resolve())] = digest(members[name])
    for driver in drivers:
        p = Path(driver).resolve()
        if str(p) not in inventory:
            name = "external/" + digest(str(p).encode())[:16] + "/" + p.name
            members[name] = p.read_bytes()
            inventory[str(p)] = digest(members[name])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(commit=commit, files={k: digest(v) for k, v in members.items()},
                    scope="tracked repository code and explicit driver files, actual working bytes")
    members["source-manifest.json"] = encoded(manifest)
    ref = archive(directory / "source.tar.gz", members)
    ref.update(commit=commit, manifest_sha256=digest(encoded(manifest)))
    return ref, inventory


def credentials():
    # Never record credentials, signed URLs, arbitrary environment variables or
    # pip direct_url metadata in the public board or the source archive.
    env = dict(os.environ)
    p = Path(os.environ.get("MOJOLEARN_R2_CREDS", str(Path.home() / ".mojolearn_r2")))
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("R2_") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("\"'"))
    import bincache
    return bincache.creds_from_env(env)


def upload(ref, request=None):
    """PUT then GET; keep the local bytes on every failure. Retry is idempotent."""
    bucket = ref.get("r2_bucket")
    if request is None:
        import bincache
        creds = credentials()
        bucket = creds["R2_BUCKET"]
        if Path(ref["local_path"]).stat().st_size > 128 * 1024 * 1024:
            return upload_streamed(ref, creds)
        request = lambda method, key, data=None: bincache.r2_request(creds, method, key, data=data, timeout=300)
    body = Path(ref["local_path"]).read_bytes()
    if digest(body) != ref["sha256"]:
        raise ValueError("local evidence bytes changed")
    key = "%s/%s.tar.gz" % (PREFIX, ref["sha256"])
    # FAIRNESS FIX (2026-10-13 audit): a transient transfer failure stopped
    # final retention even when the original upload had completed. Retry the
    # missing transfer only; never remeasure or reseal a completed race.
    uploaded = False
    for _ in range(3):
        try:
            if not uploaded:
                status, _ = request("PUT", key, body)
                if status not in (200, 201):
                    raise ValueError("R2 upload failed (HTTP %s)" % status)
                uploaded = True
            status, back = request("GET", key)
            if status != 200 or digest(back) != ref["sha256"]:
                raise ValueError("R2 evidence read-back mismatch")
            return dict(ref, r2_key=key, r2_bucket=bucket, status="verified", verification="GET sha256")
        except OSError:
            # URLError can contain a credential-bearing presigned URL.
            continue
    raise ValueError("R2 evidence request failed; local bytes retained") from None


def upload_streamed(ref, creds, opener=None):
    """Retain large archives with bounded memory and an independent GET hash.

    A retry first checks whether the previous PUT completed. Original local
    bytes survive every failure. This is transfer work, never a rerun.
    """
    import bincache
    opener = opener or urllib.request.urlopen
    path = Path(ref["local_path"])
    size = path.stat().st_size
    if file_hash(path) != ref["sha256"] or size != ref["bytes"]:
        raise ValueError("local evidence bytes changed")
    key = "%s/%s.tar.gz" % (PREFIX, ref["sha256"])

    def verified():
        h, count = hashlib.sha256(), 0
        url = bincache.presign("GET", key, 900, creds)
        with opener(urllib.request.Request(url, method="GET"), timeout=300) as response:
            if response.status != 200:
                return False
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                count += len(block)
                h.update(block)
        return count == size and h.hexdigest() == ref["sha256"]

    for _ in range(3):
        try:
            try:
                present = verified()
            except OSError:
                present = False
            if not present:
                url = bincache.presign("PUT", key, 900, creds)
                with path.open("rb") as stream:
                    request = urllib.request.Request(url, data=stream, method="PUT",
                                                     headers={"Content-Length": str(size)})
                    with opener(request, timeout=300) as response:
                        if response.status not in (200, 201):
                            raise ValueError("R2 upload failed (HTTP %s)" % response.status)
                present = verified()
            if present:
                return dict(ref, r2_key=key, r2_bucket=creds["R2_BUCKET"],
                            status="verified", verification="GET sha256")
        except OSError:
            continue  # never expose a credential-bearing URL in the error
    raise ValueError("R2 streamed evidence transfer failed; local bytes retained")


def public_ref(ref):
    return {k: v for k, v in ref.items() if k != "local_path"}


def fetch(ref, destination, request=None):
    """Download exactly the board's retained object, verifying before writing."""
    if request is None:
        import bincache
        creds = credentials()
        if ref.get("r2_bucket") and ref["r2_bucket"] != creds["R2_BUCKET"]:
            raise ValueError("configured R2 bucket differs from the measurement record")
        request = lambda key: bincache.r2_request(creds, "GET", key)
    try:
        status, body = request(ref["r2_key"])
    except urllib.error.URLError:
        raise ValueError("R2 evidence download failed") from None
    if status != 200 or digest(body) != ref["sha256"]:
        raise ValueError("downloaded evidence checksum mismatch")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def hardware_context(vendor=None):
    """Worker-observed physical GPU/driver and OS, outside any measured clock. Inside the campaign smoke's one
    worker process (tools/smoke_inproc.py) every arm is observed by that one process, so it is read once."""
    vendor = vendor or os.environ.get("MOJOLEARN_BOARD_VENDOR")
    if os.environ.get("MOJOLEARN_SMOKE_IN_PROCESS") == str(os.getpid()):
        import copy
        import smoke_inproc
        return copy.deepcopy(smoke_inproc.once(("hardware_context", vendor), lambda: _hardware_context(vendor)))
    return _hardware_context(vendor)


def _hardware_context(vendor):
    row = dict(vendor=vendor, host=platform.node(), os=platform.platform(),
               kernel=platform.release(), machine=platform.machine())
    try:
        if vendor in ("nvidia", "amd"):
            import bench_board_provenance
            row["gpu_driver_identity"] = bench_board_provenance.hardware_receipt(
                "cuda" if vendor == "nvidia" else "hip")
            if vendor == "amd":
                row["rocm_version"] = Path("/opt/rocm/.info/version").read_text().strip()
        elif vendor == "apple":
            devices = json.loads(subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=60))["SPDisplaysDataType"]
            if len(devices) != 1:
                raise ValueError("one unambiguous GPU required")
            g = devices[0]
            row["gpu"] = {k: g[k] for k in ("sppci_model", "_name", "sppci_cores", "spdisplays_mtlgpufamilysupport") if k in g}
            if not row["gpu"]:
                raise ValueError("GPU identity missing")
            row["driver_os_build"] = subprocess.check_output(["sw_vers", "-buildVersion"], text=True, timeout=10).strip()
        else:
            raise ValueError("GPU vendor missing")
        row["status"] = "verified"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        row.update(status="unavailable", error=type(exc).__name__)
    return row


def worker_context():
    """The actual worker's environment, collected outside the fit clock."""
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    quota = Path("/sys/fs/cgroup/cpu.max")
    runtime = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        runtime["torch"] = dict(version=getattr(torch, "__version__", None),
            cuda=getattr(torch.version, "cuda", None), hip=getattr(torch.version, "hip", None),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        runtime["cupy"] = dict(version=cupy.__version__, driver=cupy.cuda.runtime.driverGetVersion(),
                               runtime=cupy.cuda.runtime.runtimeGetVersion())
    return dict(python=sys.version, executable=sys.executable, host=platform.node(),
                hardware=hardware_context(), runtime=runtime,
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                          if d.metadata.get("Name")},
                environment=runtime_environment(),
                cpu_quota=quota.read_text().strip() if quota.exists() else None,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                thread_pools=pools)


def register_worker(library, arm=None):
    global _registered
    if not os.environ.get(RECEIPTS):
        return
    _libraries.add(library)
    if arm:
        _arms[arm] = library
    if _registered:
        return
    _registered = True
    def finish():
        row = dict(library=library, libraries=sorted(_libraries), arms=dict(_arms),
                   argv=sys.argv, pid=os.getpid(), final=True)
        try:
            import bench_board_probe
            row.update(worker_context())
            row["versions"] = {a: bench_board_probe.library_version(lib) for a, lib in _arms.items()}
            files = {}
            for name, mod in list(sys.modules.items()):
                filename = getattr(mod, "__file__", None)
                if not filename or not name.startswith("mojolearn"):
                    continue
                p = Path(filename)
                if p.is_file() and p.suffix in (".py", ".so", ".dylib"):
                    files[str(p.resolve())] = file_hash(p)
            row.update(status="recorded", loaded_mojolearn_files=files)
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__)
        write(Path(os.environ[RECEIPTS]) / (str(os.getpid()) + ".json"), row)
    atexit.register(finish)


def begin(ctx, race, settings, data):
    """Snapshot before any parameter probe or scored execution."""
    import tempfile
    if not data.get("identity") or any(v is None for v in data.get("files", {}).values()):
        raise ValueError("measurement input identity is missing; refused before timing")
    if "measurement_source" not in ctx:
        source, inventory = source_snapshot(Path(ctx["out"]) / "evidence" / "source",
            drivers=[ctx[k] for k in ("tree_driver", "classical_driver", "more_driver", "neural_driver", "algos_driver") if ctx.get(k)])
        if ctx.get("evidence_r2", True):
            source = upload(source)  # refuse before spending GPU time if storage is unavailable
        ctx["measurement_source"], ctx["measurement_inventory"] = source, inventory
    parent = Path(ctx["out"]) / "evidence" / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="race-", dir=parent))
    (directory / "workers").mkdir()
    recipe = dict(schema=SCHEMA, source=public_ref(ctx["measurement_source"]), race=race,
                  settings=settings, dataset=data, hardware=ctx.get("box"),
                  artifact_identity=ctx.get("artifact_identity"),
                  policy="one excluded warmup; scored rounds recorded individually; no timer changes")
    write(directory / "recipe.json", recipe)
    return directory, recipe


def classical_output_refs(rec, members):
    """Validate retained arrays against the digest from their scored call."""
    import io
    import numpy as np
    import bench_board_state as state
    refs, missing = {}, {}
    for field, phase, suffix in (("cells", "fit", ""), ("infer_cells", "infer", "-infer")):
        for cell in rec.get(field) or []:
            if cell.get("status") != "ok" or cell.get("stored"):
                continue
            name = "run/work/%s-%s-%s%s.npz" % (rec["lane"], rec["dataset"], cell["arm"], suffix)
            if name not in members:
                missing[(phase, cell["arm"])] = "scored output bytes missing: " + name
                continue
            try:
                with np.load(io.BytesIO(members[name]), allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                h = hashlib.sha256()
                for key in sorted(arrays):
                    h.update(key.encode()); h.update(np.ascontiguousarray(arrays[key]).data)
                if not arrays or h.hexdigest()[:16] != cell.get("hash"):
                    raise ValueError("scored digest differs")
                output_hash = state.canonical_hash(arrays)
                receipts = cell.get("state_receipts") or []
                if receipts and receipts[-1].get("output_sha256") != output_hash:
                    raise ValueError("scored receipt differs")
            except (ValueError, OSError, KeyError, zipfile.BadZipFile, EOFError):
                # FAIRNESS FIX (2026-10-13 audit): malformed saved bytes must
                # become a retention failure, not escape before one is recorded.
                missing[(phase, cell["arm"])] = "scored output bytes do not match recorded output: " + name
                continue
            refs[(phase, cell["arm"])] = dict(member=name, bytes=len(members[name]),
                sha256=digest(members[name]), output_sha256=output_hash, output_digest=cell["hash"])
    return refs, missing


def scored_state_refs(rec, directory, members):
    """Verify exact scored-call output/model bytes, independently of timers.

    Model snapshots and output identity are separate. An unavailable public
    model export is an explicit coverage gap, not an output-identity failure.
    These receipts do not themselves compare vendors or certify a complete
    fitted-state contract.
    """
    import numpy as np
    import bench_board_state as state
    refs, missing = {}, {}
    directory = Path(directory).resolve()

    def retained(ref, expected, *, model=None):
        if not ref:
            raise ValueError("scored bytes not retained")
        relative = Path(ref["path"]).resolve().relative_to(directory)
        name = str(relative)
        if relative.parts[0] != "scored" or name not in members:
            raise ValueError("scored archive member missing")
        payload = members[name]
        if len(payload) != ref.get("bytes") or digest(payload) != ref.get("sha256"):
            raise ValueError("retained file checksum differs")
        if ref.get("encoding") == "typed canonical state npz":
            values = state.load_typed_state(io.BytesIO(payload))
        else:
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                values = {key: saved[key] for key in saved.files}
            if model:
                full = state.canonical_hash(values)
                if full != model.get("complete_export_sha256"):
                    raise ValueError("complete model export digest differs")
                for key in model.get("excluded_metadata", []):
                    del values[key]
        if state.canonical_hash(values) != expected:
            raise ValueError("scored canonical digest differs")
        return dict(member=name, bytes=len(payload), sha256=digest(payload),
                    canonical_sha256=expected, encoding=ref.get("encoding"))

    for cell in rec.get("cells") or []:
        if cell.get("stored") or cell.get("arm") in rec.get("stored_arms", []):
            continue
        key = ("fit", cell["arm"])
        receipts = cell.get("state_receipts") or []
        if not receipts:
            if (rec.get("family") in ("trees", "neural", "classical", "classical2", "algos")
                    and cell.get("mode") in ("identical", "fast")
                    and cell.get("status") in ("ok", "not-scored", "recompiled-in-clock")):
                missing[key] = ["scored output receipt missing"]
            continue
        rows, failures = [], []
        for index, receipt in enumerate(receipts):
            row = dict(receipt_index=index, model=dict(status="unavailable", reason="no model receipt"))
            try:
                if receipt.get("retention_error"):
                    raise ValueError(receipt["retention_error"])
                if receipt.get("output_status") != "ok":
                    raise ValueError("scored output unavailable")
                row["output"] = retained(receipt.get("retained_output"), receipt.get("output_sha256"))
            except (ValueError, TypeError, OSError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                failures.append("scored output %d: %s" % (index, exc))
            model = receipt.get("model") or {}
            row["model"] = ({k: model[k] for k in ("status", "class", "source", "reason", "error") if k in model}
                            or row["model"])
            if model.get("status") == "ok":
                try:
                    row["model"]["retained"] = retained(model.get("retained_export"), model.get("sha256"), model=model)
                    row["model"]["completeness"] = "public snapshot; full fitted-state contract not certified"
                except (ValueError, TypeError, OSError, KeyError, zipfile.BadZipFile, EOFError) as exc:
                    failures.append("scored model %d: %s" % (index, exc))
            elif model.get("status") == "error" and model.get("retention_error"):
                failures.append("scored model %d: %s" % (index, model.get("error", "export failed")))
            rows.append(row)
        refs[key] = rows
        if failures:
            missing[key] = failures
    return refs, missing


def seal(ctx, directory, recipe, rec):
    """Capture measured records and logs, then attach the durable refs per arm."""
    directory = Path(directory)
    changed = [p for p, h in ctx["measurement_inventory"].items() if not Path(p).is_file() or file_hash(p) != h]
    workers = [json.loads(p.read_text()) for p in sorted((directory / "workers").glob("*.json"))]
    members = {"recipe.json": encoded(recipe), "result.json": encoded(rec), "workers.json": encoded(workers)}
    for p in directory.glob("*.json"):
        if p.name != "upload.json":
            members["records/" + p.name] = p.read_bytes()
    for p in (directory / "scored").rglob("*.npz"):
        if p.is_file():
            members[str(p.relative_to(directory))] = p.read_bytes()
    started = (directory / "recipe.json").stat().st_mtime
    # FAIRNESS FIX (2026-10-13 audit): drivers save scored arrays under work/,
    # while receipts previously retained only logs/raw. Keep the actual output
    # and independent quality-reference bytes in the per-race verified archive.
    for sub in ("logs", "raw", "work"):
        for p in (Path(ctx["out"]) / sub).rglob("*"):
            if p.is_file() and p.suffix in (".log", ".json", ".npz", ".npy", ".bin", ".safetensors", ".pt") and p.stat().st_mtime >= started:
                members["run/" + str(p.relative_to(ctx["out"]))] = p.read_bytes()
    missing = []
    output_refs, output_missing = {}, {}
    if rec.get("family") == "classical":
        # FAIRNESS FIX (2026-10-13 audit): the classical wrapper used to
        # delete these arrays before seal(). A verified archive of hashes
        # and logs alone does not retain the scored outputs.
        output_refs, output_missing = classical_output_refs(rec, members)
    state_refs, state_missing = scored_state_refs(rec, directory, members)
    for field in ("log", "race_json"):
        if rec.get(field):
            p = Path(ctx["out"]) / rec[field]
            if p.is_file():
                members["run/" + rec[field]] = p.read_bytes()
            else:
                missing.append(field)
    # Retain actual loaded product code/binary bytes, not only their hashes.
    for worker in workers:
        for filename, sha in worker.get("loaded_mojolearn_files", {}).items():
            p = Path(filename)
            if p.is_file() and file_hash(p) == sha:
                members["loaded/" + sha + "/" + p.name] = p.read_bytes()
            else:
                missing.append("loaded bytes: " + filename)
    if changed:
        missing.append("source changed during run")
    before = (rec.get("preflight") or {}).get("arms") or {}
    after = (rec.get("params") or {}).get("arms") or {}
    if (rec.get("preflight") or {}).get("verdict") != "MATCHED":
        missing.append("preflight did not pass")
    if (rec.get("params") or {}).get("schema") != "mojolearn.parameter-parity.v2":
        missing.append("strict timed-worker parameter audit")
    for arm in rec.get("arms", []):
        if arm in rec.get("stored_arms", []) or arm in rec.get("unsupported_arms", {}):
            continue
        params = after.get(arm)
        if not params or not before.get(arm):
            missing.append("timed parameter read-back missing: " + arm)
        elif any(params.get(k) != before[arm].get(k) for k in ("params", "raw_params", "implementation")):
            missing.append("effective parameters/implementation changed after preflight: " + arm)
    ref = archive(directory / "run.tar.gz", members)
    write(directory / "upload.json", dict(source=ctx["measurement_source"], run=ref))
    error = None
    if ctx.get("evidence_r2", True):
        try:
            ref = upload(ref)
        except (OSError, ValueError, SystemExit) as exc:
            error = "R2 retention failed; retry sync without rerunning the measurement"
    write(directory / "upload.json", dict(source=ctx["measurement_source"], run=ref))
    for phase, field in (("fit", "cells"), ("infer", "infer_cells")):
        for cell in rec.get(field) or []:
            if cell.get("stored") or cell.get("arm") in rec.get("stored_arms", []):
                continue
            params = cell.get("params") or ((rec.get("params") or {}).get("arms") or {}).get(cell.get("arm"))
            absent = list(missing)
            if (phase, cell.get("arm")) in output_missing:
                absent.append(output_missing[(phase, cell["arm"])])
            absent.extend(state_missing.get((phase, cell.get("arm")), []))
            if not params:
                absent.append("effective parameters")
            fitted = cell.get("fitted_parameters") or []
            for actual in fitted:
                if actual is not None and (actual.get("params") != (params or {}).get("raw_params")
                        or actual.get("implementation") != (params or {}).get("implementation")):
                    absent.append("actual fitted estimator settings changed after preflight")
            if cell.get("mode") == "opponent" and not (params or {}).get("implementation"):
                absent.append("library implementation read-back")
            if not (cell.get("library_version") or cell.get("version")):
                absent.append("worker library version")
            # A different arm importing the same library is not this arm's receipt.
            witnessed = [w for w in workers if w.get("status") == "recorded" and
                         cell.get("arm") in (w.get("arms") or {})]
            if not witnessed:
                absent.append("worker receipt")
            probe = ((rec.get("preflight") or {}).get("probe_workers") or {}).get(cell.get("arm")) or {}
            for w in witnessed:
                version = (w.get("versions") or {}).get(cell.get("arm"))
                if not version or version != probe.get("version") or version != (cell.get("library_version") or cell.get("version")):
                    absent.append("worker library version changed or missing: " + cell.get("arm", "?"))
                if cell.get("device") == "gpu" and ((w.get("hardware") or {}).get("status") != "verified"
                        or w.get("hardware") != probe.get("hardware")):
                    absent.append("worker GPU/driver/OS changed or missing: " + cell.get("arm", "?"))
                if w.get("environment") != probe.get("environment") or "environment" not in probe:
                    absent.append("worker runtime settings changed or missing: " + cell.get("arm", "?"))
            if cell.get("mode") != "opponent" and not any(w.get("loaded_mojolearn_files") for w in witnessed):
                absent.append("loaded product code receipt")
            if not (measured_shape(cell, rec) or cell.get("batch_rows")):
                absent.append("actual measured input shape")
            if not (cell.get("comparability") or {}).get("span"):
                absent.append("timing boundary")
            record = dict(schema=SCHEMA, phase=phase, source=recipe["source"], artifacts=public_ref(ref),
                          software=dict(library=cell.get("library"), version=cell.get("library_version") or cell.get("version")),
                          parameters=params, dataset=recipe["dataset"], shape=measured_shape(cell, rec),
                          invocation=dict(implementation=(params or {}).get("implementation"),
                              effective_arguments=(params or {}).get("raw_params"),
                              readback_source=(params or {}).get("source"),
                              note="Observed library object/function and effective settings; exact call site is in the retained source archive."),
                          seed=(params or {}).get("params", {}).get("seed"),
                          iteration_receipts=cell.get("iteration_receipts"),
                          fitted_parameters=fitted,
                          parameter_exceptions=[e for e in (rec.get("params") or {}).get("exceptions", [])
                                                if e.get("arm") == cell.get("arm")],
                          hardware=witnessed[0].get("hardware") if witnessed else None,
                          controller_hardware=recipe["hardware"],
                          numeric_mode=cell.get("mode_witness") or cell.get("mode"),
                          timing=dict(span=(cell.get("comparability") or {}).get("span"), rounds=cell.get("rounds"),
                                      warmup_ms=cell.get("warmup_ms"), times_ms=cell.get("times_ms")),
                          command=rec.get("command"), workers=witnessed,
                          completeness="complete" if not absent else "incomplete", missing=absent)
            if (phase, cell.get("arm")) in output_refs:
                record["retained_outputs"] = dict(output_refs[(phase, cell["arm"])],
                    status="verified" if ref.get("status") == "verified" else "local")
            if (phase, cell.get("arm")) in state_refs:
                rows = state_refs[(phase, cell["arm"])]
                status = "verified" if ref.get("status") == "verified" else "local"
                for row in rows:
                    for value in (row.get("output"), row["model"].get("retained")):
                        if value:
                            value["status"] = status
                record["scored_state"] = dict(receipts=rows, comparison="not compared",
                    scope="captured scored calls; trees retain the final scored fit only",
                    cross_vendor_promise=cell.get("mode") == "identical",
                    eligibility="cross-vendor output comparison requires identical mode, same numerical source, inputs, initial state and settings")
                if rows and rows[-1].get("output"):
                    record["retained_outputs"] = dict(rows[-1]["output"],
                        output_sha256=rows[-1]["output"]["canonical_sha256"])
            record["id"] = digest(encoded(record))
            cell["measurement"] = record
    # A separate receipt avoids a circular hash between the raw archive and
    # its own reference. The full finalized board record is retained too.
    receipt = archive(directory / "receipt.tar.gz", {"result.json": encoded(rec)})
    write(directory / "receipt-upload.json", receipt)
    if ctx.get("evidence_r2", True):
        try:
            receipt = upload(receipt)
        except (OSError, ValueError, SystemExit):
            error = "R2 retention failed; retry sync without rerunning the measurement"
    write(directory / "receipt-upload.json", receipt)
    rec["evidence_receipt"] = public_ref(receipt)
    rec["evidence_directory"] = str(directory.relative_to(Path(ctx["out"])))
    if error or changed:
        rec["evidence_failure"] = error or "source changed during measurement"
    elif state_missing:
        rec["evidence_failure"] = "incomplete scored-byte retention; local results retained, no further races permitted"
    elif any(c.get("status") == "ok" and (c.get("measurement") or {}).get("completeness") != "complete"
             for c in (rec.get("cells") or []) + (rec.get("infer_cells") or [])
             if not c.get("stored") and c.get("arm") not in rec.get("stored_arms", [])):
        rec["evidence_failure"] = "incomplete worker evidence; local results retained, no further races permitted"
    return rec


def measured_shape(cell, rec):
    shape = cell.get("shape")
    if shape:
        return shape
    params = ((cell.get("measurement") or {}).get("parameters")) or cell.get("params") or {}
    arrays = (params.get("raw_params") or params.get("params") or {}).get("input_arrays")
    if isinstance(arrays, list) and arrays and all(
            isinstance(a, dict) and isinstance(a.get("shape"), list) and a["shape"]
            and all(type(n) is int and n > 0 for n in a["shape"])
            and a.get("dtype") and isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64
            for a in arrays):
        return {"input_arrays": arrays}
    return rec.get("inputs")


def recover_shapes(board_path):
    """Recover omitted shape metadata from verified original worker readback.

    FAIRNESS FIX (2026-10-13 audit): function arms already recorded dimensions
    and input hashes in their parameters, but finalization overlooked them.
    Keep the original receipt; never repeat or alter a timing/quality observation.
    """
    path = Path(board_path)
    board = json.loads(path.read_text())
    for rr in board.get("races", {}).values():
        cells = (rr.get("cells") or []) + (rr.get("infer_cells") or [])
        candidates = [c for c in cells if "actual measured input shape" in
                      (c.get("measurement") or {}).get("missing", []) and measured_shape(c, rr)]
        if not candidates:
            continue
        directory = path.parent / rr["evidence_directory"]
        original = rr["evidence_receipt"]
        retained = directory / "original-shape-receipt.tar.gz"
        fetch(original, retained)
        with tarfile.open(retained) as archive_file:
            old = json.load(archive_file.extractfile("result.json"))
        old_records = {c["measurement"]["id"]: c for field in ("cells", "infer_cells")
                       for c in old.get(field, []) if c.get("measurement")}
        for cell in candidates:
            record = cell["measurement"]
            original_cell = old_records.get(record["id"]) or {}
            # Board rendering/ingest adds display-only hardware/version labels
            # after finalization. Verify the observations and worker records.
            if any(original_cell.get(k) != cell.get(k) for k in (
                    "measurement", "params", "quality", "median_ms", "times_ms", "warmup_ms",
                    "status", "fitted_parameters", "iterations", "comparability", "hash")):
                raise ValueError("original R2 receipt does not match shape recovery input")
            previous_id = record.pop("id")
            record["shape"] = measured_shape(cell, rr)
            record["missing"] = [m for m in record["missing"] if m != "actual measured input shape"]
            record["completeness"] = "incomplete" if record["missing"] else "complete"
            record["metadata_recovery"] = dict(original_receipt=original, previous_id=previous_id,
                reason="dimensions and input hashes from original worker parameter readback")
            record["id"] = digest(encoded(record))
        if all((c.get("measurement") or {}).get("completeness") == "complete" for c in cells if c.get("status") == "ok"):
            rr.pop("evidence_failure", None)
        receipt = upload(archive(directory / "recovered-shape-receipt.tar.gz", {"result.json": encoded(rr)}))
        rr["evidence_receipt"] = public_ref(receipt)
    write(path, board)


def sync(board_path):
    """Retry retention only; never launch a measurement or replace timing data."""
    path = Path(board_path)
    board = json.loads(path.read_text())
    for rr in board.get("races", {}).values():
        if not rr.get("evidence_directory"):
            continue
        state_path = path.parent / rr["evidence_directory"] / "upload.json"
        state = json.loads(state_path.read_text())
        state = {k: upload(v) if v.get("status") != "verified" else v for k, v in state.items()}
        write(state_path, state)
        receipt_path = state_path.with_name("receipt-upload.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "verified":
                receipt = upload(receipt)
                write(receipt_path, receipt)
            rr["evidence_receipt"] = public_ref(receipt)
        records_changed = False
        for cell in (rr.get("cells") or []) + (rr.get("infer_cells") or []):
            record = cell.get("measurement")
            if record and record["artifacts"]["sha256"] == state["run"]["sha256"]:
                before = encoded(record)
                record.update(source=public_ref(state["source"]), artifacts=public_ref(state["run"]))
                outputs = record.get("retained_outputs")
                if outputs and outputs.get("status") == "local" and state["run"].get("status") == "verified":
                    # The output member and scored digest were validated at
                    # seal time. This retry verified the same archive bytes.
                    outputs["status"] = "verified"
                if state["run"].get("status") == "verified":
                    for row in (record.get("scored_state") or {}).get("receipts", []):
                        for value in (row.get("output"), (row.get("model") or {}).get("retained")):
                            if value and value.get("status") == "local":
                                value["status"] = "verified"
                record.pop("id", None)
                record["id"] = digest(encoded(record))
                records_changed |= encoded(record) != before
        if str(rr.get("evidence_failure", "")).startswith("R2 retention failed"):
            rr.pop("evidence_failure", None)
            records_changed = True
        if records_changed:
            # FAIRNESS FIX (2026-10-13 audit): the old receipt still described
            # local-only archives. Preserve it and retain the finalized record
            # so later evidence recovery sees the same IDs as the board.
            previous = rr.get("evidence_receipt")
            if previous:
                rr.setdefault("evidence_receipt_history", []).append(previous)
            receipt = upload(archive(state_path.with_name("synced-receipt.tar.gz"),
                                     {"result.json": encoded(rr)}))
            rr["evidence_receipt"] = public_ref(receipt)
            write(receipt_path, receipt)
    write(path, board)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["sync", "fetch", "recover-shapes"])
    ap.add_argument("board")
    ap.add_argument("--race")
    ap.add_argument("--arm", default="ours")
    ap.add_argument("--kind", choices=["source", "artifacts"], default="artifacts")
    ap.add_argument("--destination")
    args = ap.parse_args()
    if args.command == "sync":
        sync(args.board)
    elif args.command == "recover-shapes":
        recover_shapes(args.board)
    else:
        if not args.race or not args.destination:
            ap.error("fetch requires --race and --destination")
        board = json.loads(Path(args.board).read_text())
        race = board["races"][args.race]
        cell = next(c for c in race["cells"] if c.get("arm") == args.arm)
        fetch(cell["measurement"][args.kind], args.destination)
