#!/usr/bin/env python3
"""Assemble an explicitly alpha Python overlay; never import package/native code.

This separate path does not alter stable release gates. Root must validate the
result on its advertised platforms before publication. Native availability is
reported from files only: symbols, usability and current numerical correctness
are not inferred from inherited binaries.
"""
import argparse
import ast
import base64
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

MAX_WHEEL = 8 * 1024**3
MAX_MEMBER = 2 * 1024**3
MAX_TOTAL = 16 * 1024**3
MAX_FILES = 30000
NOTICE = ('ALPHA PYTHON API OVERLAY: native/runtime bytes are inherited from the '
          'identified base wheel. Current Python/native compatibility and numerical '
          'qualification are NOT inherited. Missing optional native modules remain '
          'unavailable; file presence does not prove symbols or feature support.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    path = PurePosixPath(name)
    require(name and '\\' not in name and '\x00' not in name
            and not path.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/'))
            and ':' not in name and str(path) == name, 'unsafe wheel path: ' + repr(name))


def hash_stream(stream, sink=None):
    digest, count = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        count += len(chunk)
        require(count <= MAX_MEMBER, 'member exceeds byte bound')
        digest.update(chunk)
        if sink is not None:
            sink.write(chunk)
    return digest.digest(), count


def record_hash(raw_digest):
    return 'sha256=' + base64.urlsafe_b64encode(raw_digest).rstrip(b'=').decode('ascii')


def assemble(base, python_root, version, out, allow_alpha_final_version=False):
    alpha_version = re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)a(0|[1-9][0-9]*)', version)
    final_version = re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version)
    require(alpha_version or (allow_alpha_final_version is True and final_version),
            'version must be an explicit alpha, such as 0.6.0a1')
    require(base.is_file() and not base.is_symlink() and base.stat().st_size <= MAX_WHEEL,
            'base wheel must be a bounded regular file')
    require(base.suffix == '.whl', 'base must be a wheel')
    parts = base.stem.split('-')
    require(len(parts) in (5, 6) and parts[0] == 'mojolearn', 'unexpected wheel filename')
    old_version = parts[1]
    parts[1] = version
    filename = '-'.join(parts) + '.whl'
    package = python_root / 'mojolearn'
    require(package.is_dir() and not package.is_symlink(), '--python-root must contain mojolearn/')
    replacements = {}
    source_bytes = 0
    for source in sorted(package.rglob('*.py')):
        if any(part in ('tests', '__pycache__') for part in source.relative_to(package).parts):
            continue
        require(not source.is_symlink() and not any(
            package.joinpath(*source.relative_to(package).parts[:i]).is_symlink()
            for i in range(1, len(source.relative_to(package).parts))),
                'Python source symlink refused')
        require(source.stat().st_size <= 16 * 1024**2, 'oversize Python source')
        name = 'mojolearn/' + source.relative_to(package).as_posix()
        safe_name(name)
        source_bytes += source.stat().st_size
        require(source_bytes <= 64 * 1024**2 and len(replacements) < MAX_FILES,
                'Python source overlay exceeds bounds')
        replacements[name] = source.read_bytes()
    for source, name in ((package / 'ALPHA_API.md', 'mojolearn/ALPHA_API.md'),
                         (python_root / 'mojolearn_diagnostics.py', 'mojolearn_diagnostics.py')):
        if source.exists() or source.is_symlink():
            require(source.is_file() and not source.is_symlink()
                    and source.stat().st_size <= 1024**2, 'invalid/oversize alpha documentation or diagnostics')
            replacements[name] = source.read_bytes()
    require('mojolearn/__init__.py' in replacements and 'mojolearn/_backend.py' in replacements,
            'incomplete Python source tree')
    source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in replacements.items() if name.endswith('.py')}
    doc_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in replacements.items() if name.endswith('.md')}
    require(sum(map(len, replacements.values())) <= 64 * 1024**2 and len(replacements) <= MAX_FILES,
            'Python source overlay exceeds bounds')
    replacements['mojolearn/_version.py'] = ('__version__ = ' + repr(version) + '\n').encode()
    expected = None
    for node in ast.parse(replacements['mojolearn/_backend.py']).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_MODULES' for t in node.targets):
            expected = ast.literal_eval(node.value)
    require(isinstance(expected, tuple) and expected and all(isinstance(n, str) for n in expected),
            'cannot statically read native module registry')
    with base.open('rb') as handle:
        # Wheel hash bound is separate from the per-member streaming bound.
        whole = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            whole.update(chunk)
    with zipfile.ZipFile(base) as archive:
        infos = archive.infolist()
        require(len(infos) <= MAX_FILES and sum(i.file_size for i in infos) <= MAX_TOTAL,
                'wheel expansion exceeds bounds')
        names = set()
        for info in infos:
            safe_name(info.filename.rstrip('/') if info.is_dir() else info.filename)
            require(info.filename not in names and not info.flag_bits & 1
                    and not stat.S_ISLNK(info.external_attr >> 16)
                    and info.file_size <= MAX_MEMBER, 'duplicate/encrypted/symlink/oversize member')
            names.add(info.filename)
        dist = 'mojolearn-' + old_version + '.dist-info'
        require({n.split('/')[0] for n in names if '.dist-info/' in n} == {dist}, 'ambiguous dist-info')
        record_name = dist + '/RECORD'
        require(record_name in names and archive.getinfo(record_name).file_size <= 16 * 1024**2,
                'missing/oversize RECORD')
        require(not any(n.endswith(('/RECORD.jws', '/RECORD.p7s')) for n in names),
                'signed wheel requires a separate re-signing procedure')
        records = {}
        for row in csv.reader(io.StringIO(archive.read(record_name).decode('utf-8'))):
            require(len(row) == 3 and row[0] not in records, 'invalid/duplicate RECORD row')
            safe_name(row[0])
            records[row[0]] = row[1:]
        file_names = {i.filename for i in infos if not i.is_dir()}
        require(set(records) == file_names and records[record_name] == ['', ''], 'RECORD inventory mismatch')
        hashes = {}
        for info in infos:
            if info.is_dir() or info.filename == record_name:
                continue
            with archive.open(info) as stream:
                digest, size = hash_stream(stream)
            require(records[info.filename] == [record_hash(digest), str(size)], 'RECORD hash/size mismatch: ' + info.filename)
            hashes[info.filename] = digest.hex()
        wheel_raw = archive.read(dist + '/WHEEL')
        require(b'Root-Is-Purelib: false' in wheel_raw, 'base must be a platform wheel')
        wheel_headers = BytesParser(policy=policy.compat32).parsebytes(wheel_raw)
        tags = {p + '-' + a + '-' + platform for p in parts[-3].split('.')
                for a in parts[-2].split('.') for platform in parts[-1].split('.')}
        require(set(wheel_headers.get_all('Tag', [])) == tags, 'wheel filename/tag mismatch')
        metadata_raw = archive.read(dist + '/METADATA')
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_raw)
        require(metadata.get_all('Name') == ['mojolearn'] and metadata.get_all('Version') == [old_version],
                'base metadata identity mismatch')
        metadata.replace_header('Version', version)
        classifiers = [v for v in metadata.get_all('Classifier', []) if not v.startswith('Development Status ::')]
        del metadata['Classifier']
        for classifier in classifiers + ['Development Status :: 3 - Alpha']:
            metadata['Classifier'] = classifier
        # Preserve the original UTF-8 description bytes; compat32's text
        # payload round-trip otherwise tries to serialize non-ASCII as ASCII.
        body_parts = re.split(br'\r?\n\r?\n', metadata_raw, maxsplit=1)
        metadata.set_payload(NOTICE.encode('utf-8') + b'\n\n' +
                             (body_parts[1] if len(body_parts) == 2 else b''))
        new_dist = 'mojolearn-' + version + '.dist-info'
        replacements[new_dist + '/METADATA'] = metadata.as_bytes(policy=policy.compat32.clone(max_line_length=0))
        native = {n: h for n, h in hashes.items() if n.endswith(('.so', '.dylib', '.dll', '.pyd')) or '.so.' in n}
        directories = sorted({str(PurePosixPath(n).parent) for n in native if PurePosixPath(n).name.startswith('_mojolearn')})
        missing = {d: [module for module in expected if not any(
            str(PurePosixPath(n).parent) == d and (PurePosixPath(n).name == module + '.so'
            or PurePosixPath(n).name.startswith(module + '.')) for n in native)] for d in directories}
        provenance = dict(schema='mojolearn.alpha-overlay.v1', version=version, release_profile='alpha-api',
            base_wheel=base.name, base_wheel_sha256=whole.hexdigest(), notice=NOTICE,
            native_runtime_sha256=native, python_source_sha256=source_hashes,
            documentation_source_sha256=doc_hashes,
            inherited_non_python_sha256={n: h for n, h in hashes.items()
                if not n.endswith('.py') and not n.startswith(dist + '/') and n not in replacements
                and not (n.startswith('mojolearn/') and any(p in ('tests', '__pycache__') for p in PurePosixPath(n).parts))},
            overlay_python_sha256={n: hashlib.sha256(v).hexdigest() for n, v in replacements.items() if n.endswith('.py')},
            missing_optional_native_modules_by_present_directory=missing,
            required_module_registry=list(expected), availability='file inventory only; absent vendor/tier directories remain absent',
            current_numerical_qualification='NOT INHERITED; requires separate root validation')
        replacements[new_dist + '/ALPHA_PROVENANCE.json'] = (json.dumps(provenance, sort_keys=True, indent=2) + '\n').encode()
        replacements[new_dist + '/ALPHA_NOTICE.md'] = (NOTICE + '\n').encode()
        # Preserve every non-Python payload byte, including runtime libraries.
        out.mkdir(parents=False, exist_ok=False)
        temporary = out / (filename + '.partial')
        output_records = []
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for info in infos:
                name = info.filename
                if info.is_dir() or name == record_name:
                    continue
                target = new_dist + name[len(dist):] if name.startswith(dist + '/') else name
                if target in replacements:
                    continue
                if name.startswith('mojolearn/') and any(p in ('tests', '__pycache__') for p in PurePosixPath(name).parts):
                    continue
                # Omit stale Python modules from the base API tree.
                if name.startswith('mojolearn/') and name.endswith('.py'):
                    continue
                copied = zipfile.ZipInfo(target, info.date_time)
                copied.external_attr = info.external_attr
                copied.compress_type = info.compress_type
                with archive.open(info) as src, output.open(copied, 'w', force_zip64=True) as dst:
                    digest, size = hash_stream(src, dst)
                require(digest.hex() == hashes[name], 'base bytes changed during overlay')
                output_records.append((target, record_hash(digest), str(size)))
            for name, raw in sorted(replacements.items()):
                output.writestr(name, raw)
                output_records.append((name, record_hash(hashlib.sha256(raw).digest()), str(len(raw))))
            new_record = new_dist + '/RECORD'
            buffer = io.StringIO(newline='')
            csv.writer(buffer, lineterminator='\n').writerows(sorted(output_records) + [(new_record, '', '')])
            output.writestr(new_record, buffer.getvalue().encode())
        # Hard-link publication refuses a destination created concurrently.
        (out / filename).hardlink_to(temporary)
        temporary.unlink()
    return out / filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-wheel', required=True, type=Path)
    parser.add_argument('--python-root', required=True, type=Path)
    parser.add_argument('--version', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-alpha-final-version', action='store_true',
                        help='Explicitly allow X.Y.Z numbering while retaining the alpha-api profile and all qualification limits')
    args = parser.parse_args()
    print(assemble(args.base_wheel, args.python_root, args.version, args.out, args.allow_alpha_final_version))
