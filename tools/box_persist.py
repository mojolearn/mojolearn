#!/usr/bin/env python3
"""tools/box_persist.py -- THE MAC SIDE of box-persisted jobs: every job on a rented RunPod or DigitalOcean box
writes its results to R2 FROM THE BOX, so they survive the Mac disconnecting (Andrew, 2026-10-26).

The R2 credentials never leave this Mac (~/.mojolearn_r2, or $MOJOLEARN_R2_CREDS). This tool signs presigned
PUT and GET URLs for one job's keys only and hands them to the box as a 0600 json on ssh stdin; the box runs
the job under tools/box_persist_run.sh, which uploads a tarball of the results directory every few minutes and
on any exit. The URLs expire (at most 7 days, the SigV4 limit).

  box_persist.py mint --tag TAG [--slots N] [--every-min M] [--days D] [--max-mb MB] [--readback final|all|none]
                      [--exclude PATTERN]... [--out PATH] [--tool NAME]
        writes the job's persist json (0600) to ~/mojolearn-evidence/box-persist/<tag>/persist.json and prints
        its path. Keys: measurements/box-jobs/<YYYY-MM-DD>/<tag>/{00..N-1}.tgz(.sha256), final.tgz(.sha256),
        manifest.json.
  box_persist.py prepare <mint options> [--box-root /root/box-persist]
        mint, then print a bash installer for the box on stdout: it writes box_persist_run.sh to
        <box-root>/bin/ and the json to <box-root>/jobs/<tag>/persist.json (0600) and prints
        BOX_PERSIST_STAGED <json path>. Pipe it to `ssh <box> bash -s`; nothing lands in any argv.
  box_persist.py fetch --tag TAG [--date YYYY-MM-DD] [--dest DIR] [--all|--list] [--extract]
        reads the job's manifest.json from R2 and every sha256 sidecar, prints what landed (LANDED name sha
        bytes), downloads final.tgz (else the newest slot; --all: every object) and verifies each against its
        sidecar (VERIFIED / VERIFY-FAILED, exit 1 on any failure). --list downloads nothing.
  box_persist.py opt-out --tool NAME --detail TEXT
        the explicit --no-persist path of the box entry points: always logged to
        ~/mojolearn-evidence/box-persist/unpersisted.log; exit 0 only with MOJOLEARN_ALLOW_UNPERSISTED=1.
  box_persist.py new-tag PREFIX
        prints PREFIX-<utc>-<random>, a fresh job tag.

The box entry points that use this by default: tools/nvidia_central.sh submit, tools/do_amd_steward.sh run,
tools/dev_pod.sh run. Environment: MOJOLEARN_PERSIST_HOME (local state, default
~/mojolearn-evidence/box-persist), MOJOLEARN_R2_CREDS, MOJOLEARN_R2_ENDPOINT (tests: a fake S3 at
http://host:port; path-style /<bucket>/<key> either way). Tests: tools/test_box_persist.py.
"""
import argparse
import datetime
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "box_persist_run.sh")
MAX_EXPIRY_SEC = 7 * 24 * 3600          # SigV4 presigned URLs: 7 days at most
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
KEY_ROOT = "measurements/box-jobs"


def home():
    return os.environ.get("MOJOLEARN_PERSIST_HOME") or os.path.expanduser("~/mojolearn-evidence/box-persist")


def die(msg, code=1):
    sys.stderr.write("box_persist: %s\n" % msg)
    sys.exit(code)


# ---------------------------------------------------------------- credentials and signing
def load_creds():
    path = os.environ.get("MOJOLEARN_R2_CREDS") or os.path.expanduser("~/.mojolearn_r2")
    if not os.path.isfile(path):
        die("no R2 credentials file %s: cannot mint persistence URLs" % path)
    vals = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^\s*(?:export\s+)?(R2_[A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m:
                v = m.group(2)
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                vals[m.group(1)] = v
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        if not vals.get(k):
            die("%s is missing %s" % (path, k))
    return vals


def sigv4_query(method, scheme, host, uri, akid, secret, amzdate, expires, region="auto", service="s3"):
    """A SigV4 query-string presigned URL (UNSIGNED-PAYLOAD, only the host header signed)."""
    datestamp = amzdate[:8]
    scope = "%s/%s/%s/aws4_request" % (datestamp, region, service)
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256",
         "X-Amz-Credential": "%s/%s" % (akid, scope),
         "X-Amz-Date": amzdate,
         "X-Amz-Expires": str(int(expires)),
         "X-Amz-SignedHeaders": "host"}
    cqs = "&".join("%s=%s" % (urllib.parse.quote(k, safe="-_.~"), urllib.parse.quote(v, safe="-_.~"))
                   for k, v in sorted(q.items()))
    creq = "\n".join([method, uri, cqs, "host:%s\n" % host, "host", "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(creq.encode()).hexdigest()])
    k = hmac.new(("AWS4" + secret).encode(), datestamp.encode(), hashlib.sha256).digest()
    for part in (region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return "%s://%s%s?%s&X-Amz-Signature=%s" % (scheme, host, uri, cqs, sig)


class Signer:
    def __init__(self, creds=None, now=None):
        self.c = creds or load_creds()
        self.now = now or datetime.datetime.now(datetime.timezone.utc)
        ep = os.environ.get("MOJOLEARN_R2_ENDPOINT", "")
        if ep:
            p = urllib.parse.urlparse(ep)
            self.scheme, self.host, self.base = p.scheme, p.netloc, p.path.rstrip("/")
        else:
            self.scheme, self.host, self.base = "https", "%s.r2.cloudflarestorage.com" % self.c["R2_ACCOUNT_ID"], ""

    def url(self, method, key, secs):
        if not 1 <= secs <= MAX_EXPIRY_SEC:
            die("expiry %ss is outside 1..%s (7 days, the SigV4 limit)" % (secs, MAX_EXPIRY_SEC))
        uri = self.base + "/" + urllib.parse.quote("%s/%s" % (self.c["R2_BUCKET"], key), safe="/~")
        return sigv4_query(method, self.scheme, self.host, uri, self.c["R2_ACCESS_KEY_ID"],
                           self.c["R2_SECRET_ACCESS_KEY"], self.now.strftime("%Y%m%dT%H%M%SZ"), secs)


# ---------------------------------------------------------------- mint
def prefix_for(date, tag):
    return "%s/%s/%s" % (KEY_ROOT, date, tag)


def mint(a):
    if not TAG_RE.match(a.tag):
        die("tag must match %s" % TAG_RE.pattern)
    if not 1 <= a.slots <= 99:
        die("--slots must be 1..99")
    if a.every_min <= 0:
        die("--every-min must be > 0")
    if not 0 < a.days <= 7:
        die("--days must be in (0, 7]: a presigned URL lives 7 days at most")
    s = Signer()
    secs = int(a.days * 86400)
    now = s.now
    date = now.strftime("%Y-%m-%d")
    prefix = prefix_for(date, a.tag)
    exp_epoch = int(now.timestamp()) + secs
    j = {
        "tag": a.tag, "date": date, "prefix": prefix, "tool": a.tool or "",
        "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_epoch": str(exp_epoch),
        "expires_utc": datetime.datetime.fromtimestamp(exp_epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "every_sec": str(int(round(a.every_min * 60))) if a.every_sec is None else str(a.every_sec),
        "slots": str(a.slots), "max_bytes": str(int(a.max_mb * 1024 * 1024)), "readback": a.readback,
        "excludes": " ".join(a.exclude or []),
    }
    for p in a.exclude or []:
        if '"' in p or "\\" in p or "\n" in p:
            die("an --exclude pattern may not hold quotes, backslashes or newlines")
    objs = [("slot.%02d" % i, "%02d.tgz" % i) for i in range(a.slots)] + [("final", "final.tgz")]
    for name, obj in objs:
        key = "%s/%s" % (prefix, obj)
        j[name + ".put"] = s.url("PUT", key, secs)
        j[name + ".get"] = s.url("GET", key, secs)
        j[name + ".sha.put"] = s.url("PUT", key + ".sha256", secs)
        j[name + ".sha.get"] = s.url("GET", key + ".sha256", secs)
    j["manifest.put"] = s.url("PUT", prefix + "/manifest.json", secs)
    j["manifest.get"] = s.url("GET", prefix + "/manifest.json", secs)
    for k, v in j.items():   # the box reads this file with sed: one "key": "value" per line, no quotes in values
        if '"' in v or "\n" in v or "\\" in v:
            die("value of %s cannot go in a flat json" % k)
    out = a.out or os.path.join(home(), a.tag, "persist.json")
    if os.path.exists(out):
        die("%s exists: tags are one per job (box_persist.py new-tag)" % out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(flat_json(j))
    index_row(a.tag, date, prefix, a.slots, j["every_sec"], j["expires_utc"], a.tool or "")
    return out, j


def flat_json(d):
    lines = ["  %s: %s" % (json.dumps(k), json.dumps(v)) for k, v in d.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def index_row(tag, date, prefix, slots, every, expires, tool):
    os.makedirs(home(), exist_ok=True)
    idx = os.path.join(home(), "index.tsv")
    new = not os.path.exists(idx)
    with open(idx, "a") as fh:
        if new:
            fh.write("utc\ttag\tdate\tprefix\tslots\tevery_sec\texpires_utc\ttool\n")
        fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            tag, date, prefix, slots, every, expires, tool))


def installer(json_path, j, box_root):
    with open(RUNNER) as fh:
        runner = fh.read()
    with open(json_path) as fh:
        body = fh.read()
    for marker, text in (("BOX_PERSIST_RUN_EOF", runner), ("BOX_PERSIST_JSON_EOF", body)):
        if ("\n%s\n" % marker) in ("\n" + text):
            die("heredoc marker %s occurs in the payload" % marker)
    r = box_root
    t = j["tag"]
    return """set -eu
umask 077
R=%(r)s
mkdir -p "$R/bin" "$R/jobs/%(t)s" "$R/active" "$R/state"
cat > "$R/bin/box_persist_run.sh.new" <<'BOX_PERSIST_RUN_EOF'
%(runner)sBOX_PERSIST_RUN_EOF
chmod 755 "$R/bin/box_persist_run.sh.new" && mv -f "$R/bin/box_persist_run.sh.new" "$R/bin/box_persist_run.sh"
cat > "$R/jobs/%(t)s/persist.json" <<'BOX_PERSIST_JSON_EOF'
%(body)sBOX_PERSIST_JSON_EOF
chmod 600 "$R/jobs/%(t)s/persist.json"
command -v curl > /dev/null 2>&1 || echo "box_persist: WARNING no curl on this box: the job will run but nothing reaches R2" >&2
echo "BOX_PERSIST_STAGED $R/jobs/%(t)s/persist.json"
""" % {"r": r, "t": t, "runner": runner if runner.endswith("\n") else runner + "\n",
       "body": body if body.endswith("\n") else body + "\n"}


# ---------------------------------------------------------------- fetch
def http_get(url, dest=None, timeout=600):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if dest is None:
                return r.status, r.read()
            h = hashlib.sha256()
            with open(dest, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    fh.write(b)
            return r.status, h.hexdigest()
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)


def find_job(tag):
    """(date, slots) of a tag from the local index, else from its local persist json."""
    idx = os.path.join(home(), "index.tsv")
    date, slots = None, None
    if os.path.exists(idx):
        with open(idx) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 5 and f[1] == tag:
                    date, slots = f[2], f[4]
    if date is None:
        p = os.path.join(home(), tag, "persist.json")
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            date, slots = d.get("date"), d.get("slots")
    try:
        slots = int(slots)
    except (TypeError, ValueError):
        slots = 99
    return date, slots


def fetch(a):
    if not TAG_RE.match(a.tag):
        die("bad tag")
    date, nslots = find_job(a.tag)
    date = a.date or date
    if not date:
        die("no date for tag %s (no index row; pass --date YYYY-MM-DD)" % a.tag)
    s = Signer()
    prefix = prefix_for(date, a.tag)
    dest = a.dest or os.path.join(home(), a.tag, "fetched")
    os.makedirs(dest, exist_ok=True)
    code, body = http_get(s.url("GET", prefix + "/manifest.json", 3600), None, 120)
    manifest = None
    if code == 200:
        manifest = json.loads(body.decode())
        with open(os.path.join(dest, "manifest.json"), "wb") as fh:
            fh.write(body)
        print("MANIFEST state=%s exit=%s updated=%s uploads=%d" % (
            manifest.get("state"), manifest.get("exit"), manifest.get("updated_utc"), len(manifest.get("uploads", []))))
    else:
        print("MANIFEST missing (http %s): probing every sidecar" % code)
    latest = {}
    if manifest:
        for u in manifest.get("uploads", []):
            latest[u["name"]] = u          # later entries win: a ring slot is overwritten
        names = sorted(latest, key=lambda n: latest[n]["seq"])
        if "final.tgz" not in names:
            names.append("final.tgz")   # a final upload whose manifest PUT failed still counts
    else:
        names = ["%02d.tgz" % i for i in range(nslots)] + ["final.tgz"]
    landed = []
    for n in names:
        code, body = http_get(s.url("GET", "%s/%s.sha256" % (prefix, n), 3600), None, 60)
        if code != 200:
            if n in latest:
                print("MISSING %s (sidecar http %s)" % (n, code))
            continue
        sha = body.decode().split()[0]
        u = latest.get(n, {})
        if u and u.get("sha256") != sha:
            # the ring slot was overwritten after this manifest: the sidecar is the truth
            u = {}
        landed.append((n, sha, u))
        print("LANDED %s sha256=%s bytes=%s utc=%s label=%s" % (n, sha, u.get("bytes", "?"), u.get("utc", "?"), u.get("label", "?")))
    if not landed:
        print("NOTHING LANDED under %s/" % prefix)
        return 1
    if a.list:
        return 0
    if a.all:
        want = landed
    else:
        fin = [x for x in landed if x[0] == "final.tgz"]
        want = fin or [max(landed, key=lambda x: x[2].get("seq", -1))]
    bad = 0
    for n, sha, _ in want:
        p = os.path.join(dest, n)
        code, got = http_get(s.url("GET", "%s/%s" % (prefix, n), 3600), p, 3600)
        if code == 200 and got == sha:
            print("VERIFIED %s -> %s" % (n, p))
            if a.extract:
                x = os.path.join(dest, n[:-4])
                os.makedirs(x, exist_ok=True)
                with tarfile.open(p) as tf:
                    tf.extractall(x, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(x)
                print("EXTRACTED %s -> %s" % (n, x))
        else:
            bad += 1
            print("VERIFY-FAILED %s http=%s sha256=%s want=%s" % (n, code, got, sha))
    return 1 if bad else 0


# ---------------------------------------------------------------- opt-out
def opt_out(a):
    allowed = os.environ.get("MOJOLEARN_ALLOW_UNPERSISTED", "") == "1"
    os.makedirs(home(), exist_ok=True)
    with open(os.path.join(home(), "unpersisted.log"), "a") as fh:
        fh.write("%s\t%s\t%s\t%s\n" % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       a.tool, "ALLOWED" if allowed else "REFUSED", a.detail))
    if not allowed:
        sys.stderr.write("box_persist: %s --no-persist REFUSED: every rented-box job persists its results to R2 from "
                         "the box. MOJOLEARN_ALLOW_UNPERSISTED=1 runs it unpersisted (logged to %s).\n"
                         % (a.tool, os.path.join(home(), "unpersisted.log")))
        return 1
    sys.stderr.write("box_persist: %s runs UNPERSISTED (MOJOLEARN_ALLOW_UNPERSISTED=1; logged)\n" % a.tool)
    return 0


def new_tag(prefix):
    p = re.sub(r"[^A-Za-z0-9._-]", "-", prefix)[:60] or "job"
    return "%s-%s-%s" % (p, datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"), secrets.token_hex(3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def mint_args(p):
        p.add_argument("--tag", required=True)
        p.add_argument("--slots", type=int, default=4)
        p.add_argument("--every-min", type=float, default=15.0)
        p.add_argument("--every-sec", type=int, default=None, help=argparse.SUPPRESS)   # tests
        p.add_argument("--days", type=float, default=3.0)
        p.add_argument("--max-mb", type=float, default=3800.0)   # one R2 PUT takes 5 GB at most
        p.add_argument("--readback", choices=("final", "all", "none"), default="final")
        p.add_argument("--exclude", action="append")
        p.add_argument("--out")
        p.add_argument("--tool")
    mint_args(sub.add_parser("mint"))
    pp = sub.add_parser("prepare")
    mint_args(pp)
    pp.add_argument("--box-root", default="/root/box-persist")
    f = sub.add_parser("fetch")
    f.add_argument("--tag", required=True)
    f.add_argument("--date")
    f.add_argument("--dest")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--list", action="store_true")
    f.add_argument("--extract", action="store_true")
    o = sub.add_parser("opt-out")
    o.add_argument("--tool", required=True)
    o.add_argument("--detail", default="")
    t = sub.add_parser("new-tag")
    t.add_argument("prefix")
    a = ap.parse_args(argv)
    if a.cmd == "mint":
        out, _ = mint(a)
        print(out)
        return 0
    if a.cmd == "prepare":
        if not re.match(r"^/[A-Za-z0-9._/-]+$", a.box_root):
            die("--box-root must be an absolute plain path")
        out, j = mint(a)
        sys.stdout.write(installer(out, j, a.box_root))
        sys.stderr.write("box_persist: job %s persists to R2 %s/ (fetch: tools/box_persist.py fetch --tag %s)\n"
                         % (a.tag, j["prefix"], a.tag))
        return 0
    if a.cmd == "fetch":
        return fetch(a)
    if a.cmd == "opt-out":
        return opt_out(a)
    if a.cmd == "new-tag":
        print(new_tag(a.prefix))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
