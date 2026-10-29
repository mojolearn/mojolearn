"""Bounded real-install setup regression; use an EXCLUSIVE disposable install.

Temporarily removes one actual native owner, restores it in finally, and starts
fresh installed-package processes. No stubs, monkeypatches or checkout imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--disposable-installed-package', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    package = args.disposable_installed_package.resolve()
    assert package.name == 'mojolearn' and package.parent.name == 'site-packages'
    args.evidence.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical')
    env.pop('PYTHONPATH', None)
    cases = []
    prefix = "from pathlib import Path\nimport mojolearn\nassert Path(mojolearn.__file__).resolve().parent == Path(" + repr(str(package)) + ")\n"

    def run(name, source):
        command = [str(args.python), '-I', '-c', prefix + source]
        with (args.evidence / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=args.evidence, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        cases.append({'case': name, 'command': command, 'exit_code': result.returncode})

    run('catalog-present-numpy', """
import json, sys
assert 'mojolearn.models._catalog' not in sys.modules
assert 'catalog' in dir(mojolearn.models)
from mojolearn.models import catalog, verify_catalog_entry
rows = catalog()
assert type(rows) is list
assert catalog is mojolearn.models.catalog
assert callable(verify_catalog_entry)
print(json.dumps({'catalog_rows': len(rows), 'lazy_exports': True}))
""")
    owner = package / 'host/_mojolearn_x_sequence_host.so'
    held = owner.with_suffix('.so.setup-regression-held')
    assert owner.is_file() and not held.exists()
    before = digest(owner)
    owner.rename(held)
    try:
        run('missing-identical-owner', """
import json
try:
    import mojolearn.ops
except ImportError as error:
    message = str(error)
    assert 'no host binding covers _mojolearn_x_sequence' in message, message
    assert 'fast' not in message, message
    print(json.dumps({'expected_missing_owner_error': message}))
else:
    raise AssertionError('Missing required native owner was not refused')
""")
    finally:
        held.rename(owner)
    after = digest(owner)
    assert before == after
    receipt = {'cases': cases, 'native_binary_restored_sha256': after,
               'scope': 'Real exclusive installed clone; actual catalog and missing-owner diagnostic; CPU only'}
    (args.evidence / 'setup-cases.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return int(any(case['exit_code'] for case in cases))


if __name__ == '__main__':
    sys.exit(main())
