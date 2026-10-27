"""Source identity for verifier inventories, in source trees and installed wheels."""
import hashlib
from pathlib import Path

SUPPLEMENTAL_CHECKS = ('_verify_graph', '_verify_light', '_verify_partitioned_program', '_verify_distributed_program')


def supplemental_sources(package):
    return {name: hashlib.sha256((Path(package) / (name + '.py')).read_bytes()).hexdigest()
            for name in SUPPLEMENTAL_CHECKS}


def harness_sources(harness):
    paths = {'harness': harness.__file__}
    fragments = getattr(harness, 'lane_fragment_paths', lambda: ())()
    paths.update(('lane/' + name, path) for name, path in fragments)
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in sorted(paths.items())}


def public_sources(package):
    """Fingerprint the installed Python API sources, excluding test tooling."""
    package = Path(package)
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob('*.py'))
            if 'tests' not in path.relative_to(package).parts
            and not path.name.startswith(('_verify', '_verification', '_identity_'))
            and path.name != '__main__.py'}


