"""Bind newer quality tooling to an unchanged, ancestor compiled source."""
import hashlib
from pathlib import Path
import re
import subprocess

# Exact allowlist: no native/production input or variant config may drift.
TOOL_ONLY = frozenset((
    'tools/shared_gemm_source_contract.py',
    'tools/shared_gemm_scoped.py',
    'tools/shared_gemm_preflight.py',
    'tools/shared_gemm_build_ibase.sh',
    'tools/shared_gemm_downstream_pair.py',
    'tools/shared_gemm_downstream_quality.py',
    'docs/apple-fast/ab/shared-gemm-downstream.md',
))


def validate_source(root, compiled_source):
    if not re.fullmatch('[0-9a-f]{40}', compiled_source):
        raise RuntimeError('invalid compiled source')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', compiled_source, head], cwd=root, check=True)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()
    if any(Path(name).suffix in ('.mojo', '.py', '.toml', '.lock') for name in untracked):
        raise RuntimeError('untracked possible runtime inputs')
    changed = subprocess.check_output(['git', 'diff', '--name-only', compiled_source, head], cwd=root, text=True).splitlines()
    forbidden = sorted(set(changed)-TOOL_ONLY)
    if forbidden:
        raise RuntimeError('compiled runtime/config inputs changed: '+repr(forbidden))
    hashes = {}
    for name in sorted(TOOL_ONLY):
        path = Path(root)/name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(compiled_source=compiled_source, harness_source=head,
                tools_only_changes=changed, harness_hashes=hashes)
