import importlib.util, json, os, pathlib, subprocess, sys
root, out = map(pathlib.Path, sys.argv[1:3])
vendor, arch, commit = sys.argv[3:]
sys.path.insert(0, str(root / 'tools'))
from check_linux_release_qualification import native_inventory, inventory_digest
if (root / '.git').exists():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=10).strip()
    if actual != commit:
        raise SystemExit('Checkout commit differs from root pin')
    # Tracked source must be clean; output evidence is retained separately.
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root, timeout=15, check=True)
else:
    if (root / 'commit.txt').read_text().strip() != commit:
        raise SystemExit('Archive commit witness differs from root pin')
spec = importlib.util.spec_from_file_location('release_backend_probe', root / 'python/mojolearn/_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
actual_arch, how = backend._device_arch(vendor)
if actual_arch != arch:
    raise SystemExit('Physical device does not match requested architecture: ' + repr((actual_arch, arch, how)))
inventory = native_inventory(root)
# DEVIATION 2290: this schema string names the script (release061_remote_build.sh,
# kept by name for its controllers), not the version built. The version is
# python/mojolearn/_version.py and the assembly profile is release-linux3.
record = dict(schema='mojolearn.release061.build-preflight.v1', source_commit=commit,
              source_inventory=inventory, source_sha256=inventory_digest(inventory),
              vendor=vendor, device_architecture=actual_arch, architecture_probe=how,
              cpu_affinity=sorted(os.sched_getaffinity(0)), pixi_environment='default',
              scope='Physical device and source witness; no installed qualification')
(out / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
