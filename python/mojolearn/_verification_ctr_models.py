"""Checked-in GPU-trained CTR fixtures used only for CPU inference verification."""
import hashlib
from pathlib import Path

MODEL_SHA256 = {'gbdt-categorical-ctr-tables.base.npz': 'bf3bea6dea6870d5ad3ab64cf82bf3cb6c2af1ca5a65ff7d0da6ae7b12342a83',
 'gbdt-categorical-ctr-tables.denormal.npz': 'caf6bc21aff5297d6ee83e04c3d9121c7907788df496347bd2cfa50e5e47f25c',
 'gbdt-categorical-ctr-tables.denormal_ftz.npz': 'caf6bc21aff5297d6ee83e04c3d9121c7907788df496347bd2cfa50e5e47f25c',
 'gbdt-categorical-ctr-tables.dupes.npz': '48404a9e0deead0e6024d20988c4f607930efb3b527df98593aa7a56a79504ff',
 'gbdt-categorical-ctr-tables.hashed.npz': '14956a34ee65064c8e5277dab23361d948980e0b6b9ea51efaa84f585e2ba342',
 'gbdt-categorical-ctr-tables.negative.npz': '182c3af211932dae9241267351247f74a1d67837fe79f7cb1295f2c715e32142',
 'gbdt-categorical-ctr-tables.odd.npz': 'd12b8a92dfaf4f57f16fb903f82fd3f2cc172a8ae8c7cd06799bc5b3629b0183',
 'gbdt-categorical-ctr-tables.ties.npz': '1a89f2f0c4b4089c3677b91579bc5a57f06cd3f27cb28b92f5ea517a15ae6dde',
 'gbdt-categorical-ctr-tables.wide.npz': '9b400c71cad50af0726d22ea08c4f14642155172a195bce41e7d546fb3fb9caf',
 'gbdt-tensor-ctr-tables.base.npz': 'd701b87375180e4b1963373d5607ba9cd9a12fa887bfe4d328206e124cb5b776',
 'gbdt-tensor-ctr-tables.denormal.npz': '1a4382769846fc64aaea1d993b3cdba90cf07f27afb41dbafe4dcbe48d69c32b',
 'gbdt-tensor-ctr-tables.denormal_ftz.npz': '1a4382769846fc64aaea1d993b3cdba90cf07f27afb41dbafe4dcbe48d69c32b',
 'gbdt-tensor-ctr-tables.dupes.npz': 'f99b90a46f47a09cc3eb5e36f8874bb9132e3a2bb819e69e9d9065fc5aafd69e',
 'gbdt-tensor-ctr-tables.hashed.npz': '5c1025887f1a411225e7bf2734ec8af3655f64eebb7c64e95743aea04b79d582',
 'gbdt-tensor-ctr-tables.negative.npz': 'd388697c55275e5fbbedebf975df1e2d4591375b96bf189435784c009cc1e3fb',
 'gbdt-tensor-ctr-tables.odd.npz': '4148bc1f3f9b7e8723e45ee3721e76cae772c558ad40d628d021fd856807d88b',
 'gbdt-tensor-ctr-tables.ties.npz': '4274b123a863584af4daaed2602a6f6111e23b759f54a94ceb48f1068e12461c',
 'gbdt-tensor-ctr-tables.wide.npz': 'e768f6d7a856ca3fc5841d062a47b655bc757afd3b0a6f7d7e8074e3df1221f5'}


def resolve_model(lane, fixture, package_dir=None):
    from . import host_surface
    name = f'{lane}.{fixture}.npz'
    expected = MODEL_SHA256.get(name)
    if expected is None:
        raise RuntimeError(f'no bundled CTR model declared for {lane}/{fixture}')
    package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    bundled = package_dir / 'verify_reference' / 'ctr_models' / name
    # Source diagnostics use the same tracked assets the wheel builders copy.
    path = bundled if bundled.is_file() else package_dir.parent.parent / host_surface.GBDT_CTR_MODELS_DIR / name
    if not path.is_file():
        raise RuntimeError(f'missing verification CTR model: {bundled}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'verification CTR model digest mismatch: {path}')
    return str(path)
