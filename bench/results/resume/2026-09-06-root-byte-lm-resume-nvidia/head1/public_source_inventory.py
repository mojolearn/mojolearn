import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = {}
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        raise ValueError('Public archive transport refuses source symlinks')
    if path.is_file():
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                h.update(chunk)
        result[path.relative_to(root).as_posix()] = h.hexdigest()
print(json.dumps(result, sort_keys=True, separators=(',', ':')))
