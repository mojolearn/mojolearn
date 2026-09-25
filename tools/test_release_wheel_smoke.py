"""tools/release_wheel_smoke.sh without a rental: the dry run, its refusals,
the transfer deadline, the whole --ssh path against a local ssh shim whose
`timeout` stands in for the smoke, and the provider decision (RunPod once,
DigitalOcean when RunPod has no stock) against a local HTTP shim that answers
as both clouds. Nothing here reaches RunPod or DigitalOcean: the dry run's
free listings are skipped when no key is configured, and the rented paths
point RP / MOJOLEARN_SMOKE_DO_API at the shim."""
import http.server
import json
import os
import pathlib
import signal
import subprocess
import tempfile
import threading
import time
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'tools/release_wheel_smoke.sh'
COMMIT = 'c' * 40


def make_wheel(directory, commit=COMMIT, name='mojolearn-0.0.0-py3-none-manylinux_2_35_x86_64.whl'):
    path = pathlib.Path(directory) / name
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mojolearn/identity_columns/COMMIT', commit + '\n')
        z.writestr('mojolearn/verify_reference/models/models.json', '{"models": []}')
    return path


SSH_SHIM = '#!/bin/bash\n# every call passes the remote command as its last argument\nexec bash -c "${@: -1}"\n'
SHA_SHIM = '#!/bin/sh\nexec shasum -a 256 "$@"\n'
# The stand-in smoke: `timeout -k 20 N PY qualify.py WHEEL --scope ... --output OUT`
TIMEOUT_SHIM = r'''#!/usr/bin/env python3
import hashlib, json, pathlib, sys
args = sys.argv[4:]
wheel = pathlib.Path(args[2])
out = pathlib.Path(args[args.index('--output') + 1]); out.mkdir(parents=True)
commit = args[args.index('--expected-source-commit') + 1]
status = pathlib.Path(__file__).with_name('STATUS').read_text().strip()
json.dump(dict(wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
               status=status, scope='expanded', source_commit=commit, jobs=[{'name': 'x'}],
               installed=dict(vendor='cuda')), open(out / 'results.json', 'w'))
print('fake smoke', status)
sys.exit(0 if status == 'PASSED' else 1)
'''
# The DigitalOcean path's box: the box command itself is replaced by a column
# that ran to the end (the column's own judging is the --ssh path's business).
DO_SSH_SHIM = r'''#!/bin/bash
cmd="${@: -1}"
case "$cmd" in
  *"nohup bash "*"/box.sh "*)
    d="$FAKE_BOX_DIR"
    if [ -n "${FAKE_COLUMN:-}" ]; then cp "$FAKE_COLUMN" "$d/column.json"; else printf '{"lanes": {}}\n' > "$d/column.json"; fi
    echo 0 > "$d/column.exit"
    echo install_exit=0 > "$d/column.txt"; echo done > "$d/box.done"
    echo STARTED; exit 0 ;;
esac
exec bash -c "$cmd"
'''


class FakeCloud(http.server.BaseHTTPRequestHandler):
    """RunPod under /rp and DigitalOcean under /do, as the runner sees them."""

    def log_message(self, *args):
        pass

    def _send(self, code, body=None):
        data = b'' if body is None else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        s = self.server
        s.log.append(('GET', self.path))
        p = self.path
        if p.startswith('/rp/pods'):
            return self._send(200, [])
        if p.startswith('/do/sizes'):
            return self._send(200, {'sizes': [{'slug': 'gpu-mi325x1-256gb', 'price_hourly': 3.8,
                                               'regions': ['tor1', 'nyc2'], 'available': True}]})
        if p.startswith('/do/droplets/'):
            if s.droplet and p.rsplit('/', 1)[1] == str(s.droplet['id']):
                return self._send(200, {'droplet': s.droplet})
            return self._send(404, {'id': 'not_found', 'message': 'The resource you were accessing could not be found.'})
        if p.startswith('/do/droplets'):
            return self._send(200, {'droplets': [s.droplet] if s.droplet else []})
        self._send(404, {})

    def do_POST(self):
        s = self.server
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}')
        s.log.append(('POST', self.path, body))
        if self.path == '/rp/pods':
            if s.runpod == 'nostock':
                # the real answer of 2026-09-23, HTTP 200 and all
                return self._send(200, {'error': 'create pod: There are no instances currently available', 'status': 500})
            return self._send(401, {'error': 'unauthorized'})
        if self.path == '/do/droplets':
            if body.get('region') in s.refuse_regions:
                return self._send(422, {'id': 'unprocessable_entity', 'message': 'Region is not available'})
            s.droplet = {'id': 4242, 'name': body['name'], 'status': 'active',
                         'size': {'slug': body['size']}, 'region': {'slug': body['region']},
                         'networks': {'v4': [{'type': 'public', 'ip_address': '127.0.0.1'}]}}
            return self._send(202, {'droplet': dict(s.droplet, status='new', networks={'v4': []})})
        self._send(404, {})

    def do_DELETE(self):
        s = self.server
        s.log.append(('DELETE', self.path))
        if self.path.startswith('/do/droplets/') and s.droplet and self.path.endswith('/' + str(s.droplet['id'])):
            s.droplet = None
            return self._send(204)
        self._send(404, {})


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.wheel = make_wheel(self.dir)
        self.cloud = None

    def tearDown(self):
        if self.cloud:
            self.cloud.shutdown()
            self.cloud.server_close()
        # the on-"droplet" self-destruct the ssh shim started on this machine
        pid_file = self.dir / 'box' / 'wheel-smoke-guard' / 'selfkill.pid'
        if pid_file.exists():
            pid = int(pid_file.read_text().strip() or 0)
            if pid:
                subprocess.run(['pkill', '-P', str(pid)], capture_output=True)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.tmp.cleanup()

    def run_smoke(self, *args, env=None, timeout=120):
        e = dict(os.environ, MOJOLEARN_RUNPOD_KEY_FILE=str(self.dir / 'no-key'),
                 MOJOLEARN_DO_TOKEN_FILE=str(self.dir / 'no-token'),
                 MOJOLEARN_DO_GPU_LOCK=str(self.dir / 'gpu.lock'),
                 # never the real Hot Aisle key or API from a test
                 MOJOLEARN_HOTAISLE_KEY_FILE=str(self.dir / 'no-hotaisle-key'),
                 MOJOLEARN_HOTAISLE_API='http://127.0.0.1:9/nothing-here',
                 MOJOLEARN_HOTAISLE_SLOT_PREFIX=str(self.dir / 'ha-slot'),
                 MOJOLEARN_HOTAISLE_CREATE_LOCK=str(self.dir / 'ha-create.lock'))
        e.pop('RUNPOD_API_KEY', None)
        e.update(env or {})
        return subprocess.run(['bash', str(SMOKE), *args], capture_output=True, text=True,
                              timeout=timeout, env=e)

    def selection(self, backend='hip'):
        path = self.dir / f'selection-{backend}.json'
        path.write_text(json.dumps({'backend': backend, 'lanes': ['hf-checkpoint']}))
        return str(path)

    def test_dry_run_plans_one_pod_and_two_files(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--out', str(self.dir / 'o'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('DRY RUN', r.stdout)
        self.assertIn('"gpuTypeIds": ["NVIDIA GeForce RTX 4090"]', r.stdout)
        self.assertIn('"allowedCudaVersions": ["13.0"]', r.stdout)
        self.assertIn('qualify_verifier_wheel.py', r.stdout)
        self.assertIn('provider runpod', r.stdout)
        self.assertNotIn('gpu-mi325x1-256gb', r.stdout)
        self.assertFalse((self.dir / 'o').exists())

    def test_refuses_wrong_commit_and_unrepaired_wheel(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', 'd' * 40)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('records source commit ' + COMMIT, r.stderr)
        raw = make_wheel(self.dir, name='mojolearn-0.0.0-py3-none-linux_x86_64.whl')
        r = self.run_smoke(str(raw), '--expected-source-commit', COMMIT)
        self.assertIn('not a final manylinux', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--rent', '--ssh', 'x')
        self.assertIn('exclusive', r.stderr)

    def test_with_timeout_kills_a_stalled_transfer(self):
        t = time.time()
        r = subprocess.run(['bash', '-c', 'die(){ exit 1; }; . tools/runpod_pod_lib.sh; with_timeout 2 sleep 30'],
                           cwd=ROOT, capture_output=True, timeout=20)
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(time.time() - t, 10)

    # ---------------------------------------------------------------- providers
    def test_hip_dry_run_default_auto_plans_runpod_then_digitalocean(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--column', self.selection(), '--out', str(self.dir / 'o'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('provider auto: RunPod once, Hot Aisle when RunPod has no', r.stdout)
        self.assertIn('DigitalOcean when Hot Aisle refuses before a create', r.stdout)
        self.assertIn('hotaisle MI300X VM spec auto', r.stdout)
        self.assertIn('Hot Aisle dead-men compose', r.stdout)
        self.assertIn('no Hot Aisle key here', r.stdout)
        self.assertIn('"gpuTypeIds": ["AMD Instinct MI300X OAM"]', r.stdout)
        self.assertIn('size gpu-mi325x1-256gb  regions tor1,nyc2', r.stdout)
        self.assertIn('"region":"tor1","size":"gpu-mi325x1-256gb","image":188571990', r.stdout)
        self.assertIn('dead-men compose', r.stdout)
        self.assertIn(str(self.dir / 'gpu.lock') + ' is free', r.stdout)
        self.assertIn('no DigitalOcean token here', r.stdout)
        self.assertIn('have no fallback', r.stdout)
        self.assertFalse((self.dir / 'o').exists())

    def test_hip_dry_run_provider_do_plans_only_the_droplet(self):
        (self.dir / 'gpu.lock').mkdir()
        (self.dir / 'gpu.lock' / 'owner').write_text('lane=someone-else\n')
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--provider', 'do', '--do-regions', 'nyc2', '--column', self.selection(),
                           '--out', str(self.dir / 'o'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('provider do (DigitalOcean only)', r.stdout)
        self.assertNotIn('gpuTypeIds', r.stdout)
        self.assertIn('"region":"nyc2"', r.stdout)
        self.assertIn('is HELD', r.stdout)
        self.assertIn('lane=someone-else', r.stdout)
        self.assertIn('a rent would refuse', r.stdout)

    def test_provider_refusals(self):
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--provider', 'do')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--provider do is for --vendor hip', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--provider', 'lambda')
        self.assertIn('--provider must be runpod, hotaisle, do or auto', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--provider', 'hotaisle')
        self.assertIn('--provider hotaisle is for --vendor hip', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--column', self.selection(), '--hotaisle-spec', '4gpu')
        self.assertIn('--hotaisle-spec must be 1gpu, 2gpu or auto', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--column', self.selection(), '--hotaisle-cap', 'ten')
        self.assertIn('--hotaisle-cap must be a dollar figure', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--column', self.selection(), '--do-regions', 'tor1;rm')
        self.assertIn('--do-regions', r.stderr)

    def start_cloud(self, runpod='nostock', refuse_regions=()):
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FakeCloud)
        srv.log, srv.runpod, srv.droplet, srv.refuse_regions = [], runpod, None, set(refuse_regions)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.cloud = srv
        return 'http://127.0.0.1:%d' % srv.server_address[1]

    def rented_env(self, base, shim=DO_SSH_SHIM):
        shims = self.dir / 'bin'
        shims.mkdir()
        for name, body in (('ssh', shim), ('sha256sum', SHA_SHIM)):
            (shims / name).write_text(body)
            (shims / name).chmod(0o755)
        for name, text in (('runpod.key', 'fake-runpod-key'), ('do.token', 'dop_v1_fake')):
            (self.dir / name).write_text(text + '\n')
            (self.dir / name).chmod(0o600)
        box = self.dir / 'box' / 'wheel-smoke'
        return {'PATH': str(shims) + ':' + os.environ['PATH'],
                'RP': base + '/rp', 'RP_V2': base + '/rpv2', 'MOJOLEARN_SMOKE_DO_API': base + '/do',
                'MOJOLEARN_RUNPOD_KEY_FILE': str(self.dir / 'runpod.key'),
                'MOJOLEARN_DO_TOKEN_FILE': str(self.dir / 'do.token'),
                'MOJOLEARN_SMOKE_REMOTE_DIR': str(box), 'FAKE_BOX_DIR': str(box)}

    def rent(self, provider, runpod='nostock', refuse_regions=(), env=None, extra=()):
        base = self.start_cloud(runpod, refuse_regions)
        e = self.rented_env(base)
        e.update(env or {})
        out = self.dir / 'out'
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--provider', provider, '--column', self.selection(), '--rent',
                           '--lease', '10', '--smoke-seconds', '60', '--out', str(out), *extra, env=e, timeout=240)
        return r, out, self.cloud.log

    # ---------------------------------------------------------------- reference columns
    def columns(self, amd_hash):
        """The Apple reference and the column the fake AMD box brings home."""
        def col(vendor, h):
            return {'mode': 'identical', 'repeats': 1, 'vendor': vendor, 'commit': COMMIT, 'complete': True,
                    'cells': {'rf-clf/base': {'verdict': 'STABLE', 'hashes': [h], 'parts': [{'predict': h}]}}}
        apple, amd = self.dir / 'apple.json', self.dir / 'amd.json'
        apple.write_text(json.dumps(col('apple-m4', 'aaaa')))
        amd.write_text(json.dumps(col('amd-mi300x', amd_hash)))
        return apple, amd

    def test_column_agreeing_with_the_apple_reference_passes(self):
        apple, amd = self.columns('aaaa')
        r, out, _ = self.rent('do', env={'FAKE_COLUMN': str(amd)}, extra=('--ref-column', str(apple)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('no DIVERGENT cell against the reference column(s)', r.stdout)
        self.assertIn('summary: IDENTICAL=1', (out / 'diff-ref-hip.txt').read_text())

    def test_a_divergent_cell_against_the_apple_reference_fails(self):
        apple, amd = self.columns('bbbb')
        r, out, _ = self.rent('do', env={'FAKE_COLUMN': str(amd)}, extra=('--ref-column', str(apple)))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('DIVERGENT against the reference column(s)', r.stdout)
        self.assertIn('rf-clf/base', r.stdout)
        self.assertIn('destroy_confirmed=1', (out / 'teardown.txt').read_text())

    def test_cpu_column_is_still_a_reference_by_its_old_name(self):
        apple, amd = self.columns('bbbb')
        r, out, _ = self.rent('do', env={'FAKE_COLUMN': str(amd)}, extra=('--cpu-column', str(apple)))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('DIVERGENT against the reference column(s)', r.stdout)

    def test_ref_column_refusals(self):
        apple, _ = self.columns('aaaa')
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--ref-column', str(apple))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--ref-column needs --column', r.stderr)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--column', self.selection(), '--ref-column', str(self.dir / 'missing.json'))
        self.assertIn('no reference column', r.stderr)

    def posts(self, log, path):
        return [x for x in log if x[0] == 'POST' and x[1] == path]

    def test_auto_falls_back_to_digitalocean_when_runpod_has_no_stock(self):
        r, out, log = self.rent('auto')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.posts(log, '/rp/pods')), 1, log)            # RunPod tried ONCE
        self.assertIn("RunPod has no 'AMD Instinct MI300X OAM' to give", r.stdout)
        self.assertIn('Hot Aisle REFUSED: no usable Hot Aisle key', r.stdout)
        self.assertIn('FALLING BACK to DigitalOcean', r.stdout)
        do_posts = self.posts(log, '/do/droplets')
        self.assertEqual(len(do_posts), 1, log)
        self.assertEqual(do_posts[0][2]['region'], 'tor1')
        self.assertEqual(do_posts[0][2]['size'], 'gpu-mi325x1-256gb')
        self.assertEqual(do_posts[0][2]['image'], 188571990)
        self.assertEqual(do_posts[0][2]['tags'], ['smoke'])
        # the order: RunPod create, then the droplet, its DELETE, and a GET that answered 404
        seq = [(x[0], x[1]) for x in log if x[1] in ('/rp/pods', '/do/droplets', '/do/droplets/4242') and x[0] != 'GET'
               or (x[0], x[1]) == ('GET', '/do/droplets/4242')]
        self.assertEqual(seq.index(('POST', '/rp/pods')), 0)
        self.assertLess(seq.index(('POST', '/do/droplets')), seq.index(('DELETE', '/do/droplets/4242')))
        self.assertIn(('GET', '/do/droplets/4242'), seq[seq.index(('DELETE', '/do/droplets/4242')):])
        provider = (out / 'provider.txt').read_text()
        self.assertIn('runpod=no_stock http=200', provider)
        self.assertIn('gpu_lock=taken', provider)
        self.assertIn('provider=do droplet=4242 size=gpu-mi325x1-256gb region=tor1 image=188571990 ip=127.0.0.1', provider)
        teardown = (out / 'teardown.txt').read_text()
        self.assertIn('destroy_confirmed=1', teardown)
        self.assertIn('spend=$', teardown)
        self.assertIn('DigitalOcean dead-man cancelled', r.stdout)
        self.assertIn('released the shared GPU lock', r.stdout)
        self.assertFalse((self.dir / 'gpu.lock').exists())
        self.assertTrue((out / 'column-hip.json').is_file())
        self.assertIn('rented=do target=root@127.0.0.1', (out / 'smoke.txt').read_text())
        watchdog = (out / 'watchdog_check.txt').read_text()
        self.assertIn('WATCHDOG_ALIVE', watchdog)
        self.assertIn('ID_BAKED_IN=1', watchdog)
        self.assertIn('TOKEN_GET_200', watchdog)
        self.assertIn("droplets/4242", (out / 'droplet_deadman.sh').read_text())
        self.assertNotIn('dop_v1_fake', r.stdout + r.stderr)

    def test_auto_tries_the_next_region_on_422(self):
        r, out, log = self.rent('auto', refuse_regions=('tor1',))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual([x[2]['region'] for x in self.posts(log, '/do/droplets')], ['tor1', 'nyc2'])
        self.assertIn('tor1 refused gpu-mi325x1-256gb (HTTP 422', r.stdout)
        self.assertIn('region=nyc2', (out / 'provider.txt').read_text())
        self.assertIn('destroy_confirmed=1', (out / 'teardown.txt').read_text())

    def test_provider_runpod_never_falls_back(self):
        r, out, log = self.rent('runpod')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('create FAILED (HTTP 200)', r.stderr)
        self.assertEqual(len(self.posts(log, '/rp/pods')), 1)
        self.assertEqual([x for x in log if x[1].startswith('/do')], [])
        self.assertFalse((self.dir / 'gpu.lock').exists())
        self.assertIn('dead-man cancelled', r.stdout)

    def test_auto_falls_back_only_on_no_stock(self):
        r, out, log = self.rent('auto', runpod='auth')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('create FAILED (HTTP 401)', r.stderr)
        self.assertEqual([x for x in log if x[1].startswith('/do')], [])
        self.assertFalse((out / 'provider.txt').exists())

    def test_digitalocean_refuses_a_held_gpu_lock(self):
        lock = self.dir / 'gpu.lock'
        lock.mkdir()
        (lock / 'owner').write_text('lane=another-leg\nnonce=x\n')
        r, out, log = self.rent('do')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('the shared GPU lock', r.stderr)
        self.assertIn('lane=another-leg', r.stderr)
        self.assertIn('Nothing was created', r.stderr)
        self.assertEqual(self.posts(log, '/do/droplets'), [])
        self.assertEqual(self.posts(log, '/rp/pods'), [])
        self.assertTrue((lock / 'owner').exists())     # not ours, untouched

    def test_digitalocean_refuses_a_bad_token_file(self):
        base = self.start_cloud()
        e = self.rented_env(base)
        (self.dir / 'do.token').chmod(0o644)
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--vendor', 'hip',
                           '--provider', 'do', '--column', self.selection(), '--rent',
                           '--out', str(self.dir / 'out'), env=e)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('mode 644, must be 600', r.stderr)
        self.assertEqual([x for x in self.cloud.log if x[1].startswith('/do')], [])

    # ---------------------------------------------------------------- an existing box
    def ssh_path(self, status):
        shims = self.dir / 'bin'
        shims.mkdir()
        for name, body in (('ssh', SSH_SHIM), ('sha256sum', SHA_SHIM), ('timeout', TIMEOUT_SHIM)):
            (shims / name).write_text(body)
            (shims / name).chmod(0o755)
        (shims / 'STATUS').write_text(status)
        out = self.dir / 'out'
        r = self.run_smoke(str(self.wheel), '--expected-source-commit', COMMIT, '--ssh', 'fake@box',
                           '--out', str(out), env={
                               'PATH': str(shims) + ':' + os.environ['PATH'],
                               'MOJOLEARN_SMOKE_REMOTE_DIR': str(self.dir / 'box' / 'wheel-smoke')})
        return r, out

    def test_ssh_path_brings_home_a_passed_receipt(self):
        r, out = self.ssh_path('PASSED')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        receipt = json.loads((out / 'results.json').read_text())
        self.assertEqual(receipt['status'], 'PASSED')
        self.assertIn('verdict=PASSED', (out / 'smoke.txt').read_text())

    def test_ssh_path_fails_a_failed_smoke(self):
        r, out = self.ssh_path('FAILED')
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('verdict=FAILED', (out / 'smoke.txt').read_text())


if __name__ == '__main__':
    unittest.main()
