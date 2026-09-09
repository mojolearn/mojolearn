import hashlib, json, os, pathlib, runpy, sys
import mojolearn
audit = json.loads(pathlib.Path(os.environ['MOJOLEARN_WHEEL_AUDIT']).read_text())
installed = pathlib.Path(mojolearn.__file__).resolve()
assert installed.is_relative_to(pathlib.Path(sys.prefix).resolve()), ('Checkout shadowing', installed)
assert 'site-packages' in installed.parts
assert mojolearn.vendor() == os.environ['MOJOLEARN_EXPECT_VENDOR']
assert mojolearn.numeric_mode() == os.environ['MOJOLEARN_NUMERIC_MODE']
print(json.dumps({'package': str(installed), 'version': mojolearn.__version__,
                  'vendor': mojolearn.vendor(), 'mode': mojolearn.numeric_mode()}), flush=True)
from mojolearn import _backend
architecture = {}
# DEVIATION 2290: release_audit writes assembly_profile 'release-linux3';
# 'release-0.6.1' is the deprecated alias retained audits may still carry.
release_profile = audit.get('assembly_profile') in ('release-linux3', 'release-0.6.1')
if release_profile:
    assert not os.environ.get('MOJOLEARN_GPU_ARCH'), 'Architecture override forbidden'
    device_arch, probe = _backend._device_arch(os.environ['MOJOLEARN_EXPECT_VENDOR'])
    selected = _backend.gpu_arch()
    # DEVIATION 2293: the selected binding may be the architecture-specific
    # build for this exact device (device sm_90 -> selected sm_90a), which is
    # what _backend.py prefers. The audit's runtime_architecture is the name
    # the wheel carries, so it must equal the SELECTED one; the device must be
    # the chip that selection targets. Anything else is still refused.
    assert selected == audit['runtime_architecture'], ('Wrong actual GPU architecture', device_arch, selected)
    assert selected in (device_arch, device_arch + 'a'), ('Wrong actual GPU architecture', device_arch, selected)
    architecture = dict(device_architecture=device_arch, selected_architecture=selected,
                        architecture_probe=probe, architecture_override_absent=True)
# Older bindings expose vendor but no tier getter; report that gap explicitly.
getters = {'_mojolearn': 'mojolearn_numeric_mode', '_mojolearn_gbdt': 'gbdt_numeric_mode',
           '_mojolearn_svm': 'svm_numeric_mode', '_mojolearn_metrics': 'umap_numeric_mode',
           '_mojolearn_linalg': 'linalg_numeric_mode', '_mojolearn_arima': 'arima_numeric_mode',
           '_mojolearn_training': 'training_numeric_mode', '_mojolearn_gp': 'gp_numeric_mode',
           '_mojolearn_mamba': 'mamba_numeric_mode', '_mojolearn_transformer': 'transformer_numeric_mode'}
if release_profile and mojolearn.numeric_mode() == 'identical':  # DEVIATION 2290
    getters['_mojolearn_byte_lm'] = 'byte_lm_numeric_mode'
all_bindings = set(getters) | {'_mojolearn_estimators', '_mojolearn_rf', '_mojolearn_trees', '_mojolearn_solver', '_mojolearn_tsa'}
readback = {}
for name in sorted(all_bindings):
    binding = _backend.binding(name)
    path = pathlib.Path(binding.__file__).resolve()
    assert path.is_relative_to(installed.parent), ('Noninstalled extension', path)
    assert _backend.read_vendor(binding) == os.environ['MOJOLEARN_EXPECT_VENDOR']
    row = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    member = path.relative_to(installed.parent).as_posix()
    if architecture:
        assert member.startswith(os.environ['MOJOLEARN_EXPECT_VENDOR'] + '/' + architecture['selected_architecture'] + '/')
    assert audit['extension_hashes'].get(member) == row['sha256'], ('Installed binary differs from wheel', member)
    if name in getters:
        row['mode_code'] = int(getattr(binding, getters[name])())
        assert row['mode_code'] == {'fast': 0, 'identical': 1, 'deterministic': 2}[os.environ['MOJOLEARN_NUMERIC_MODE']]
    else:
        row['mode_readback'] = 'unavailable; build provenance and functional gate required'
    readback[name] = row
record = {'package': str(installed), 'version': mojolearn.__version__,
          'vendor': mojolearn.vendor(), 'mode': mojolearn.numeric_mode(),
          'wheel_sha256': audit['sha256'], 'installed_bindings': readback}
record.update(architecture)
pathlib.Path(os.environ['MOJOLEARN_INSTALLED_RECORD']).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
