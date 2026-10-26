#!/usr/bin/env python3
"""tools/test_box_persist.py -- tests tools/box_persist.py + tools/box_persist_run.sh with NO R2 and NO box: a fake
S3 server (PUT/GET, SigV4 query-signature and expiry checked) on 127.0.0.1, the box root in a temp dir, and fake
ssh shims for tools/do_amd_steward.sh run and tools/dev_pod.sh run.

  python3 tools/test_box_persist.py                 the tests (about a minute)
  python3 tools/test_box_persist.py --serve PORT DIR   the fake S3 alone (objects under DIR), for docker tests
"""
import datetime
import hashlib
import http.server
import io
import json
import os
import shutil
import signal
import socketserver
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import box_persist  # noqa: E402

AKID, SECRET, BUCKET = "AKFAKE", "fakesecret/123", "fakebucket"


# ---------------------------------------------------------------- the fake S3
class FakeS3(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port=0, store_dir=None):
        self.objects = {}
        self.log = []
        self.clock_offset = 0          # seconds added to the server clock (expiry tests)
        self.fail_puts = 0             # answer 500 to this many PUTs
        self.store_dir = store_dir
        super().__init__(("0.0.0.0" if store_dir else "127.0.0.1", port), Handler)

    @property
    def endpoint(self):
        return "http://127.0.0.1:%d" % self.server_address[1]

    def key_of(self, path):
        p = urllib.parse.unquote(path)
        pre = "/%s/" % BUCKET
        return p[len(pre):] if p.startswith(pre) else None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth(self, method):
        u = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
        need = ("X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Date", "X-Amz-Expires", "X-Amz-SignedHeaders", "X-Amz-Signature")
        if any(k not in q for k in need):
            return None, 403
        if not q["X-Amz-Credential"].startswith(AKID + "/"):
            return None, 403
        want = box_persist.sigv4_query(method, "http", self.headers["Host"], u.path, AKID, SECRET,
                                      q["X-Amz-Date"], int(q["X-Amz-Expires"]))
        if not want.endswith("X-Amz-Signature=" + q["X-Amz-Signature"]):
            return None, 403
        t0 = datetime.datetime.strptime(q["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
        if time.time() + self.server.clock_offset > t0 + int(q["X-Amz-Expires"]):
            return None, 403
        k = self.server.key_of(u.path)
        return k, (200 if k else 404)

    def _reply(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_PUT(self):
        k, code = self._auth("PUT")
        n = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(n) if n else b""
        if code != 200:
            return self._reply(code)
        if self.server.fail_puts > 0:
            self.server.fail_puts -= 1
            return self._reply(500)
        self.server.objects[k] = data
        self.server.log.append(("PUT", k, len(data)))
        if self.server.store_dir:
            p = os.path.join(self.server.store_dir, k)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
        self._reply(200)

    def do_GET(self):
        k, code = self._auth("GET")
        if code != 200:
            return self._reply(code)
        if k not in self.server.objects:
            return self._reply(404)
        self.server.log.append(("GET", k, 0))
        self._reply(200, self.server.objects[k])


# ---------------------------------------------------------------- helpers
def sh256(b):
    return hashlib.sha256(b).hexdigest()


def tar_names(b):
    with tarfile.open(fileobj=io.BytesIO(b)) as tf:
        return sorted(n[2:] if n.startswith("./") else n for n in tf.getnames())


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = FakeS3()
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.objects.clear()
        self.srv.log.clear()
        self.srv.clock_offset = 0
        self.srv.fail_puts = 0
        self.w = tempfile.mkdtemp(prefix="bpt.")
        self.addCleanup(shutil.rmtree, self.w, True)
        creds = os.path.join(self.w, "r2")
        with open(creds, "w") as fh:
            fh.write("R2_ACCOUNT_ID=acct\nexport R2_ACCESS_KEY_ID=%s\nR2_SECRET_ACCESS_KEY='%s'\nR2_BUCKET=%s\n" % (AKID, SECRET, BUCKET))
        self.box = os.path.join(self.w, "box")
        self.env = dict(os.environ, MOJOLEARN_R2_CREDS=creds, MOJOLEARN_R2_ENDPOINT=self.srv.endpoint,
                        MOJOLEARN_PERSIST_HOME=os.path.join(self.w, "home"), PERSIST_POLL_SEC="0.2",
                        PERSIST_TERM_GRACE_SEC="3")
        self.env.pop("MOJOLEARN_ALLOW_UNPERSISTED", None)
        self.env.pop("BOX_PERSIST_ROOT", None)
        self.n = 0

    def tool(self, *args, check=True, inp=None):
        r = subprocess.run([sys.executable, os.path.join(HERE, "box_persist.py")] + list(args), env=self.env,
                           capture_output=True, text=True, input=inp)
        if check and r.returncode != 0:
            self.fail("box_persist.py %s rc=%s\n%s" % (args, r.returncode, r.stderr[-2000:]))
        return r

    def stage(self, every_sec=3600, slots=2, extra=()):
        """prepare + run the installer into the temp box root; returns (tag, json path on the box)"""
        self.n += 1
        tag = "t%d-%s" % (self.n, os.urandom(3).hex())
        r = self.tool("prepare", "--tag", tag, "--slots", str(slots), "--every-sec", str(every_sec),
                      "--days", "1", "--box-root", self.box, *extra)
        inst = subprocess.run(["bash", "-s"], input=r.stdout, capture_output=True, text=True, env=self.env)
        self.assertEqual(inst.returncode, 0, inst.stderr)
        line = [x for x in inst.stdout.splitlines() if x.startswith("BOX_PERSIST_STAGED ")]
        self.assertTrue(line, inst.stdout)
        return tag, line[0].split(" ", 1)[1]

    def runner(self, js, res, cmd, wait=True, timeout=60):
        p = subprocess.Popen(["bash", os.path.join(self.box, "bin", "box_persist_run.sh"), js, res, "--", "bash", "-c", cmd],
                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not wait:
            return p
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err

    def prefix(self, tag):
        return "measurements/box-jobs/%s/%s" % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), tag)

    def obj(self, tag, name):
        return self.srv.objects.get("%s/%s" % (self.prefix(tag), name))

    def manifest(self, tag):
        b = self.obj(tag, "manifest.json")
        self.assertIsNotNone(b, "no manifest.json landed")
        return json.loads(b.decode())

    def plog(self, res):
        with open(os.path.join(res, "persist.log")) as fh:
            return fh.read()

    def wait_for(self, cond, secs=20):
        t = time.time()
        while time.time() - t < secs:
            if cond():
                return True
            time.sleep(0.1)
        return False


# ---------------------------------------------------------------- the tests
class TestSigning(Base):
    def test_aws_presign_known_answer(self):
        # the AWS SigV4 documentation's presigned GET example (query auth, UNSIGNED-PAYLOAD, host only)
        u = box_persist.sigv4_query("GET", "https", "examplebucket.s3.amazonaws.com", "/test.txt",
                                    "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                                    "20130524T000000Z", 86400, region="us-east-1")
        self.assertTrue(u.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"), u)

    def test_mint_json(self):
        r = self.tool("mint", "--tag", "m1", "--slots", "3", "--every-min", "5", "--days", "2", "--exclude", "big")
        p = r.stdout.strip()
        self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o600)
        with open(p) as fh:
            text = fh.read()
        j = json.loads(text)
        for k in ("slot.00.put", "slot.02.sha.put", "final.put", "final.get", "manifest.put"):
            self.assertIn(k, j)
        self.assertNotIn("slot.03.put", j)
        self.assertEqual(j["every_sec"], "300")
        self.assertEqual(j["excludes"], "big")
        self.assertIn("/%s/measurements/box-jobs/" % BUCKET, j["final.put"])
        self.assertNotIn(SECRET, text)
        self.assertEqual(len(text.strip().splitlines()), len(j) + 2)     # one key per line (the box reads it with sed)
        self.assertIn("\tm1\t", open(os.path.join(self.w, "home", "index.tsv")).read())
        self.assertNotEqual(self.tool("mint", "--tag", "m1", check=False).returncode, 0, "a reused tag must be refused")
        self.assertNotEqual(self.tool("mint", "--tag", "m2", "--days", "8", check=False).returncode, 0, "over 7 days")
        self.assertNotEqual(self.tool("mint", "--tag", "bad/tag", check=False).returncode, 0)


class TestRun(Base):
    def test_success_final_upload(self):
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        rc, out, _ = self.runner(js, res, 'echo hello; echo data > "$MOJOLEARN_RESULTS_DIR/out.txt"; exit 3')
        self.assertEqual(rc, 3, "the command's exit status passes through")
        self.assertIn("hello", out, "output still streams to stdout")
        fin = self.obj(tag, "final.tgz")
        self.assertIsNotNone(fin)
        self.assertEqual(self.obj(tag, "final.tgz.sha256").decode().split()[0], sh256(fin))
        names = tar_names(fin)
        self.assertIn("out.txt", names)
        self.assertIn("command.log", names)
        m = self.manifest(tag)
        self.assertEqual((m["state"], m["exit"]), ("final", "3"))
        self.assertEqual(m["uploads"][-1]["sha256"], sh256(fin))
        log = self.plog(res)
        self.assertIn("UPLOADED name=final.tgz", log)
        self.assertIn("READBACK-OK name=final.tgz", log)
        self.assertFalse(os.listdir(os.path.join(self.box, "active")), "the job is unregistered at its end")

    def test_slot_rotation(self):
        tag, js = self.stage(every_sec=1, slots=2)
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, "sleep 5.5")
        self.assertEqual(rc, 0)
        m = self.manifest(tag)
        periodic = [u["name"] for u in m["uploads"] if u["label"] == "periodic"]
        self.assertGreaterEqual(len(periodic), 3, m)
        self.assertEqual(periodic[:3], ["00.tgz", "01.tgz", "00.tgz"])
        self.assertIsNotNone(self.obj(tag, "00.tgz"))
        self.assertIsNotNone(self.obj(tag, "01.tgz"))
        self.assertIsNone(self.obj(tag, "02.tgz"))
        self.assertEqual(self.obj(tag, "01.tgz.sha256").decode().split()[0], sh256(self.obj(tag, "01.tgz")))
        self.assertEqual(m["uploads"][-1]["name"], "final.tgz")

    def test_exclusions(self):
        tag, js = self.stage(extra=("--exclude", "scratch"))
        res = os.path.join(self.w, "res")
        for d in ("venv/lib", ".pixi/envs", "artifacts", ".cache", "keep", "scratch", "venv-old", "sub/__pycache__"):
            os.makedirs(os.path.join(res, d), exist_ok=True)
        for f in ("venv/lib/x", ".pixi/envs/y", "artifacts/m.whl", "artifacts/notes.txt", ".cache/z", "keep/real.txt",
                  "scratch/s", "venv-old/v", "sub/__pycache__/c.pyc", "sub/a.py", "my.persist.json"):
            with open(os.path.join(res, f), "w") as fh:
                fh.write("x")
        shutil.copy(js, os.path.join(res, "persist.json"))
        rc, _, _ = self.runner(js, res, "true")
        self.assertEqual(rc, 0)
        names = [n for n in tar_names(self.obj(tag, "final.tgz")) if n not in ("", ".")]
        for n in names:
            for bad in ("venv", ".pixi", ".whl", ".cache", "scratch", "__pycache__", "persist.json"):
                self.assertNotIn(bad, n, names)
        for good in ("keep/real.txt", "artifacts/notes.txt", "sub/a.py", "persist.log", "command.log"):
            self.assertIn(good, names)

    def test_size_cap_note(self):
        tag, js = self.stage(extra=("--max-mb", "0.05"))
        res = os.path.join(self.w, "res")
        os.makedirs(res)
        with open(os.path.join(res, "huge.bin"), "wb") as fh:
            fh.write(os.urandom(200000))
        with open(os.path.join(res, "small.txt"), "w") as fh:
            fh.write("keep me")
        rc, _, _ = self.runner(js, res, "true")
        self.assertEqual(rc, 0)
        b = self.obj(tag, "final.tgz")
        names = tar_names(b)
        self.assertNotIn("huge.bin", names)
        self.assertIn("small.txt", names)
        self.assertIn("PERSIST_TRUNCATED.txt", names)
        with tarfile.open(fileobj=io.BytesIO(b)) as tf:
            note = tf.extractfile("./PERSIST_TRUNCATED.txt").read().decode()
        self.assertIn("huge.bin", note)
        self.assertLessEqual(len(b), int(0.05 * 1024 * 1024))

    def _signal(self, sig, code):
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        marker = os.path.join(self.w, "child.pid")
        p = self.runner(js, res, 'echo $$ > %s; echo partial > "$MOJOLEARN_RESULTS_DIR/p.txt"; sleep 60' % marker, wait=False)
        self.assertTrue(self.wait_for(lambda: os.path.exists(os.path.join(res, "p.txt"))))
        p.send_signal(sig)
        p.communicate(timeout=60)
        self.assertEqual(p.returncode, code)
        fin = self.obj(tag, "final.tgz")
        self.assertIsNotNone(fin, "the trap uploads final.tgz")
        self.assertIn("p.txt", tar_names(fin))
        m = self.manifest(tag)
        self.assertEqual(m["exit"], str(code))
        self.assertIn("signal=", m["why"])
        child = int(open(marker).read())
        self.assertTrue(self.wait_for(lambda: not _alive(child), 10), "the command was stopped")

    def test_term_trap(self):
        self._signal(signal.SIGTERM, 143)

    def test_int_trap(self):
        self._signal(signal.SIGINT, 130)

    def test_hup_ignored(self):
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        p = self.runner(js, res, 'sleep 2; echo survived > "$MOJOLEARN_RESULTS_DIR/s.txt"', wait=False)
        time.sleep(0.7)
        p.send_signal(signal.SIGHUP)
        p.communicate(timeout=60)
        self.assertEqual(p.returncode, 0, "an ssh drop (HUP) does not stop the job")
        self.assertIn("s.txt", tar_names(self.obj(tag, "final.tgz")))

    def test_flush_all(self):
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        p = self.runner(js, res, 'echo mid > "$MOJOLEARN_RESULTS_DIR/mid.txt"; sleep 60', wait=False)
        try:
            self.assertTrue(self.wait_for(lambda: os.path.exists(os.path.join(res, "mid.txt"))))
            self.assertTrue(os.path.exists(os.path.join(self.box, "active", tag + ".env")))
            r = subprocess.run(["bash", os.path.join(self.box, "bin", "box_persist_run.sh"), "--flush-all", "30"],
                               env=self.env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("flush-all done", r.stdout)
            self.assertIn("mid.txt", tar_names(self.obj(tag, "final.tgz")))
            self.assertEqual(self.manifest(tag)["state"], "flushed")
            self.assertIsNone(p.poll(), "the job keeps running after a flush")
        finally:
            p.terminate()
            p.communicate(timeout=60)
        self.assertEqual(self.manifest(tag)["state"], "final")

    def test_flush_all_no_jobs(self):
        self.stage()
        r = subprocess.run(["bash", os.path.join(self.box, "bin", "box_persist_run.sh"), "--flush-all", "5"],
                           env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0)
        self.assertIn("no active jobs", r.stdout)

    def test_expired_urls(self):
        tag, js = self.stage()
        with open(js) as fh:
            j = json.load(fh)
        j["expires_epoch"] = str(int(time.time()) - 100)
        os.chmod(js, 0o600)
        with open(js, "w") as fh:
            fh.write(box_persist.flat_json(j))
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, "echo ok")
        self.assertEqual(rc, 0, "the command runs regardless")
        self.assertIn("EXPIRED", self.plog(res))
        self.assertEqual([x for x in self.srv.log if x[0] == "PUT"], [], "nothing is sent past the expiry")

    def test_server_rejects_expired(self):
        tag, js = self.stage()
        self.srv.clock_offset = 2 * 86400          # the URLs (1 day) are past their expiry at the server
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, "exit 0")
        self.assertEqual(rc, 0)
        self.assertIn("REJECTED name=final.tgz http=403", self.plog(res))
        self.assertIsNone(self.obj(tag, "final.tgz"))

    def test_unreachable_endpoint(self):
        self.env["MOJOLEARN_R2_ENDPOINT"] = "http://127.0.0.1:9"
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, "exit 5")
        self.assertEqual(rc, 5)
        self.assertIn("FAILED name=final.tgz http=000", self.plog(res))

    def test_retry_after_5xx(self):
        tag, js = self.stage()
        self.srv.fail_puts = 1                     # curl --retry covers one transient 500
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, "true")
        self.assertEqual(rc, 0)
        self.assertIsNotNone(self.obj(tag, "final.tgz"), self.plog(res))


class TestFetch(Base):
    def test_fetch_verify_and_tamper(self):
        tag, js = self.stage(every_sec=1, slots=2)
        res = os.path.join(self.w, "res")
        rc, _, _ = self.runner(js, res, 'echo result > "$MOJOLEARN_RESULTS_DIR/r.txt"; sleep 2.5')
        self.assertEqual(rc, 0)
        dest = os.path.join(self.w, "fetched")
        r = self.tool("fetch", "--tag", tag, "--dest", dest, "--all", "--extract")
        self.assertIn("MANIFEST state=final exit=0", r.stdout)
        self.assertIn("VERIFIED final.tgz", r.stdout)
        self.assertIn("VERIFIED 00.tgz", r.stdout)
        with open(os.path.join(dest, "final", "r.txt")) as fh:
            self.assertEqual(fh.read().strip(), "result")
        r = self.tool("fetch", "--tag", tag, "--list")
        self.assertIn("LANDED final.tgz", r.stdout)
        self.assertNotIn("VERIFIED", r.stdout)
        self.srv.objects["%s/final.tgz" % self.prefix(tag)] = b"tampered"
        r = self.tool("fetch", "--tag", tag, "--dest", os.path.join(self.w, "f2"), check=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("VERIFY-FAILED final.tgz", r.stdout)

    def test_fetch_without_manifest(self):
        tag, js = self.stage()
        res = os.path.join(self.w, "res")
        self.runner(js, res, "true")
        del self.srv.objects["%s/manifest.json" % self.prefix(tag)]
        r = self.tool("fetch", "--tag", tag, "--dest", os.path.join(self.w, "f"))
        self.assertIn("MANIFEST missing", r.stdout)
        self.assertIn("VERIFIED final.tgz", r.stdout)

    def test_fetch_nothing(self):
        tag, _ = self.stage()
        r = self.tool("fetch", "--tag", tag, "--list", check=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("NOTHING LANDED", r.stdout)


class TestOptOut(Base):
    def test_opt_out(self):
        r = self.tool("opt-out", "--tool", "x", "--detail", "d1", check=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("REFUSED", r.stderr)
        self.env["MOJOLEARN_ALLOW_UNPERSISTED"] = "1"
        r = self.tool("opt-out", "--tool", "x", "--detail", "d2", check=False)
        self.assertEqual(r.returncode, 0)
        log = open(os.path.join(self.w, "home", "unpersisted.log")).read()
        self.assertIn("REFUSED\td1", log)
        self.assertIn("ALLOWED\td2", log)


class TestEntryPoints(Base):
    """do_amd_steward.sh run and dev_pod.sh run, with an ssh shim that runs the box command locally."""

    def shim(self):
        b = os.path.join(self.w, "bin")
        os.makedirs(b, exist_ok=True)
        with open(os.path.join(b, "ssh"), "w") as fh:
            fh.write('#!/bin/bash\n# test shim: drop ssh options and the target, run the command here\n'
                     'while [ $# -gt 0 ]; do case "$1" in -o|-p|-i) shift 2 ;; -t|-tt|-T) shift ;; *) break ;; esac; done\n'
                     'shift\nc="$*"\n[ -z "${SHIM_TREE:-}" ] || c=${c//\/root\/mojolearn/$SHIM_TREE}\nexec bash -c "$c"\n')
        os.chmod(os.path.join(b, "ssh"), 0o755)
        tree = os.path.join(self.w, "tree")
        os.makedirs(tree, exist_ok=True)
        self.env.update(PATH=b + ":" + self.env["PATH"], MOJOLEARN_BOX_PERSIST_ROOT=self.box)
        return tree

    def landed_tags(self):
        return sorted({k.split("/")[3] for k in self.srv.objects if k.endswith("/final.tgz")})

    def test_do_amd_steward_run(self):
        tree = self.shim()
        st = os.path.join(self.w, "steward")
        os.makedirs(st)
        with open(os.path.join(st, "state.env"), "w") as fh:
            fh.write("DROPLET_ID=1\nIP=127.0.0.1\n")
        script = os.path.join(ROOT, "tools", "do_amd_steward.sh")
        env = dict(self.env, MOJOLEARN_STEWARD_DO_STATE=st, SHIM_TREE=tree)   # the shim maps /root/mojolearn to the tree
        r = subprocess.run(["bash", script, "run", "stw", "out", 'echo steward-job > "$MOJOLEARN_RESULTS_DIR/x.txt"; echo hi'],
                           env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        tags = self.landed_tags()
        self.assertEqual(len(tags), 1, tags)
        self.assertTrue(tags[0].startswith("stw-"))
        self.assertIn("x.txt", tar_names(self.obj(tags[0], "final.tgz")))
        self.assertTrue(os.path.exists(os.path.join(tree, "out", "x.txt")), "the results dir is relative to the tree")
        self.assertIn("hi", r.stdout)
        r = subprocess.run(["bash", script, "run", "--no-persist", "stw", "out", "true"],
                           env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0, "--no-persist refuses without MOJOLEARN_ALLOW_UNPERSISTED=1")

    def test_dev_pod_run(self):
        tree = self.shim()
        dps = os.path.join(self.w, "devpods")
        os.makedirs(os.path.join(dps, "lanex"))
        with open(os.path.join(dps, "lanex", "state.env"), "w") as fh:
            fh.write("PROVIDER=runpod\nVENDOR=nvidia\nPOD_ID=fake\nSSH_TARGET='-p 22 root@127.0.0.1'\nBOX_DIR=%s\nBOX_SUDO=0\nBOX_ENV=\n" % tree)
        env = dict(self.env, MOJOLEARN_DEVPOD_STATE=dps, MOJOLEARN_AMD_CENTRAL_CONF=os.path.join(self.w, "none"))
        script = os.path.join(ROOT, "tools", "dev_pod.sh")
        r = subprocess.run(["bash", script, "run", "lanex", "--results", "myres", 'echo devpod > "$MOJOLEARN_RESULTS_DIR/d.txt"; exit 7'],
                           env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 7, r.stderr[-1500:])
        tags = self.landed_tags()
        self.assertEqual(len(tags), 1, (tags, r.stderr[-800:]))
        self.assertTrue(tags[0].startswith("dp-lanex-"))
        self.assertTrue(os.path.exists(os.path.join(tree, "myres", "d.txt")), "a relative --results is in the tree")
        self.assertIn("d.txt", tar_names(self.obj(tags[0], "final.tgz")))
        r = subprocess.run(["bash", script, "run", "lanex", "--no-persist", "true"], env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        r = subprocess.run(["bash", script, "run", "lanex", "--no-persist", "true"],
                           env=dict(env, MOJOLEARN_ALLOW_UNPERSISTED="1"), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


class TestHooks(unittest.TestCase):
    """every self-destruct path flushes before its delete; every entry point stages persistence"""

    def read(self, f):
        with open(os.path.join(HERE, f)) as fh:
            return fh.read()

    def test_selfdestruct_paths_flush_before_delete(self):
        for f, delete in (("runpod_guard.sh", "-X DELETE"), ("gpu_queue_box.sh", "-X DELETE"),
                          ("do_amd_steward.sh", "-X DELETE '$DO_API/droplets/$1'"), ("hotaisle_vm_lib.sh", "-X DELETE '@API@")):
            s = self.read(f)
            i, j = s.find("--flush-all"), s.find(delete)
            self.assertGreater(i, 0, f)
            self.assertLess(i, j, "%s: the flush comes before the delete" % f)

    def test_entry_points_stage(self):
        for f, what in (("nvidia_central.sh", "bp_prepare \"nvc-$lane\""), ("do_amd_steward.sh", "bp_prepare \"$tag\""),
                        ("dev_pod.sh", "bp_prepare \"dp-$KEY\"")):
            self.assertIn(what, self.read(f), f)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--serve":
        s = FakeS3(int(sys.argv[2]), store_dir=sys.argv[3])
        print("fake S3 on port %d, bucket %s, objects under %s" % (s.server_address[1], BUCKET, sys.argv[3]), flush=True)
        s.serve_forever()
    else:
        unittest.main(verbosity=2)
