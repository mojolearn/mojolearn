import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(pathlib.Path.cwd() / 'tools'))
from verify_linux_surface_qualification import MODES, expected_bindings
expected_count = sum(len(expected_bindings(mode, True)) for mode in MODES)
before = json.loads((out / 'preflight.json').read_text())
proof = json.loads((out / 'build/build-provenance.json').read_text())
if (proof.get('complete') is not True or proof.get('build_exit') != 0
        or proof.get('source_commit') != before['source_commit']
        or proof.get('source_inventory') != before['source_inventory']
        or proof.get('source_sha256') != before['source_sha256']):
    raise SystemExit('Build proof differs from preflight or is incomplete')
prefix = 'mojolearn/' + before['vendor'] + '/' + before['device_architecture'] + '/'
if len(proof.get('extensions', {})) != expected_count or not all(p.startswith(prefix) for p in proof['extensions']):
    raise SystemExit('Built architecture differs from physical GPU witness')
byte_members = {p for p in proof['extensions'] if p.endswith('/_mojolearn_byte_lm.so')}
if byte_members != {prefix + 'identical/_mojolearn_byte_lm.so'}:
    raise SystemExit('Byte LM must exist exactly once, in IDENTICAL')
print(json.dumps(dict(status='BUILT_NOT_INSTALLED', extensions=expected_count,
                     vendor=before['vendor'], architecture=before['device_architecture'])))
