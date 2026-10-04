#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run an M2-built native quality executable on M3; no compile fallback.
SOURCE BINARY_SHA256 BINARY_PATH. Manager stages binary under a unique source.
This is correctness only, with no latency measurement or production binding.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    source, expected, name = sys.argv[1:]
    assert re.fullmatch('[0-9a-f]{40}', source)
    assert re.fullmatch('[0-9a-f]{64}', expected)
    root = Path(__file__).resolve().parents[1]
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'experiments/apple_callpath',
                    'bench/apple_callpath_quality.mojo', 'tools/callpath_quality_run.py'], cwd=root, check=True)
    binary = Path(name).expanduser()
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert actual == expected, 'staged binary hash mismatch'
    print('CALLPATH-PROVENANCE ' + json.dumps(dict(source=source, binary_sha256=actual,
          pinned_proposal='9ab2d3d3fb770498ef025db08f595a0149792bb7',
          numeric_mode='fast', vendor='apple', timing=False)), flush=True)
    subprocess.run([str(binary.resolve())], check=True)


if __name__ == '__main__':
    main()
