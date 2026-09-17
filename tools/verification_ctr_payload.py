#!/usr/bin/env python3
"""Validate and stage the same saved CTR models for macOS and Linux wheels."""
import argparse
import hashlib
from pathlib import Path
import runpy
import shutil


def model_entries(root):
    root = Path(root)
    manifest = runpy.run_path(str(root / 'python/mojolearn/host_surface.py'))
    expected = runpy.run_path(str(root / 'python/mojolearn/_verification_ctr_models.py'))['MODEL_SHA256']
    models = root / manifest['GBDT_CTR_MODELS_DIR']
    result = {}
    for name, digest in expected.items():
        path = models / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'missing or changed verification CTR model: {path}')
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    entries = model_entries(Path(__file__).resolve().parents[1])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, path in entries.items():
        shutil.copy2(path, args.output / name)
    print(f'staged {len(entries)} digest-checked CTR verification models')


if __name__ == '__main__':
    main()
