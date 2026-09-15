#!/usr/bin/env python3
"""tools/runpod_cpu_cache.py -- the Mac half of the R2 caches that
tools/runpod_cpu_leg.sh restores on a RunPod CPU pod (2026-09-15).

Two content-addressed objects live beside the datasets in the R2 bucket
mojolearn-data (the host bindings are tools/bincache.py's, not this file's):

  runpod-cpu/v1/pixi-bin/<pixi version>/linux-64/pixi.tar      the pixi binary
  runpod-cpu/v1/pixi-env/linux-64/<env>/<key>.tar              .pixi/envs/<env>, tar.gz

Each has a `.sha` sidecar holding the sha256 of the object. The box uploads
the object FIRST and the sidecar LAST, and restores only when both exist and
the bytes hash to the sidecar, so a half-written upload is never restored.

THE ENV KEY is sha256 over the canonical JSON of `env_fields`: the env name,
the platform, pixi.lock's sha256, pixi.toml without its task tables (as
tools/bincache.py reads it), the absolute prefix on the box (conda writes it
into the installed files), the pinned pixi version and the image. Any of those
moving is a new key and a cold install; nothing is ever overwritten in place
because a PUT URL is minted only for an object that does not exist.

CREDENTIALS NEVER REACH A BOX: this runs on the Mac with ~/.mojolearn_r2 in
its environment (tools/runpod_cpu_leg.sh passes it as an env prefix) and
prints presigned URLs that the runner pipes to the box over ssh stdin.

  python3 tools/runpod_cpu_cache.py keys   --repo DIR --envs default,test [--prefix /root/mojolearn --image I --pixi-version V]
  python3 tools/runpod_cpu_cache.py status --repo DIR --envs ...     (needs creds; lists, mints nothing)
  python3 tools/runpod_cpu_cache.py plan   --repo DIR --envs ... [--expires S]   (needs creds; prints the URL map)
  python3 tools/runpod_cpu_cache.py sizes                            (needs creds; every runpod-cpu and CPU bincache object)

Stdlib only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache as bc  # noqa: E402

SCHEMA_ENV = "mojolearn-runpod-cpu-env-v1"
PREFIX = "runpod-cpu/v1"
PLATFORM = "linux-64"
DEFAULT_IMAGE = "runpod/base:0.6.3-cpu"
DEFAULT_PIXI = "0.77.0"
DEFAULT_BOX_REPO = "/root/mojolearn"


def env_fields(repo, env, prefix, image, pixi_version):
    repo = Path(repo)
    toml = (repo / "pixi.toml").read_text(errors="replace")
    return dict(schema=SCHEMA_ENV, env=env, platform=PLATFORM,
                pixi_lock_sha256=bc.sha256_file(repo / "pixi.lock"),
                pixi_toml_build_sha256=bc.sha256_bytes(bc.pixi_toml_build_part(toml).encode()),
                prefix=prefix, pixi_version=pixi_version, image=image)


def objects(a):
    """[(kind, name, key, object path without suffix, fields)]"""
    out = [("pixi", "pixi", a.pixi_version, "%s/pixi-bin/%s/%s/pixi" % (PREFIX, a.pixi_version, PLATFORM),
            dict(pixi_version=a.pixi_version, platform=PLATFORM))]
    for env in [e for e in a.envs.split(",") if e]:
        if not bc.SEG_RE.match(env):
            raise SystemExit("bad env name %r" % env)
        f = env_fields(a.repo, env, a.prefix, a.image, a.pixi_version)
        key = bc.key_of(f)
        out.append(("env", env, key, "%s/pixi-env/%s/%s/%s" % (PREFIX, PLATFORM, env, key), f))
    return out


def present(creds, path):
    names = {o[0]: o[2] for o in bc.list_objects(creds, path)}
    tar, sha = path + ".tar", path + ".sha"
    return (tar in names and sha in names), names.get(tar, 0)


def cmd_keys(a):
    for kind, name, key, path, f in objects(a):
        print("%s\t%s\t%s\t%s" % (kind, name, key, path))
        if kind == "env":
            print("#fields\t%s\t%s" % (name, json.dumps(f, sort_keys=True)))
    return 0


def cmd_status(a):
    creds = bc.creds_from_env()
    for kind, name, key, path, _ in objects(a):
        ok, size = present(creds, path)
        print("%s:%s\t%s\t%s\tbytes=%d" % (kind, name, "hit" if ok else "miss", path, size))
    return 0


def cmd_plan(a):
    creds = bc.creds_from_env()
    lines, summary = [], []
    for kind, name, key, path, _ in objects(a):
        ok, size = present(creds, path)
        lines.append("#key\t%s:%s\t%s\t%s" % (kind, name, key, path))
        verb = "get" if ok else "put"
        method = "GET" if ok else "PUT"
        for suffix in ("tar", "sha"):
            lines.append("%s\t%s.%s\t%s\t%s" % (kind, name, suffix, verb,
                                                bc.presign(method, "%s.%s" % (path, suffix), a.expires, creds)))
        summary.append("%s:%s=%s" % (kind, name, "hit" if ok else "miss"))
    sys.stdout.write("\n".join(lines) + "\n")
    print("CACHE PLAN " + " ".join(summary), file=sys.stderr)
    return 0


def cmd_sizes(a):
    creds = bc.creds_from_env()
    total = 0
    for prefix in (PREFIX + "/", bc.OBJECT_PREFIX + "/none/", bc.SABOTAGE_PREFIX + "/none/"):
        for name, modified, size in bc.list_objects(creds, prefix):
            print("%12d\t%s\t%s" % (size, modified, name))
            total += size
    print("%12d\ttotal" % total)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="runpod_cpu_cache.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("keys", "status", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        p.add_argument("--envs", default="default")
        p.add_argument("--prefix", default=DEFAULT_BOX_REPO)
        p.add_argument("--image", default=DEFAULT_IMAGE)
        p.add_argument("--pixi-version", default=DEFAULT_PIXI)
        p.add_argument("--expires", type=int, default=7200)
    sub.add_parser("sizes")
    a = ap.parse_args(argv)
    return dict(keys=cmd_keys, status=cmd_status, plan=cmd_plan, sizes=cmd_sizes)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
