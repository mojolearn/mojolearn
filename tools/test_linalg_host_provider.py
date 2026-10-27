"""CPU descriptor transport source checks; compilation/runtime stay pending."""
from pathlib import Path
import re
from check_generation_native_boundary import audit
ROOT=Path(__file__).resolve().parents[1]
NAMES=('gemm','gemm_bf16','gemm_int8','quantize_int8','dequantize_int8','gemm_int15','quantize_int15','dequantize_int15','to_bf16','from_bf16','from_f16')


def body(source,name):
    return re.search(r'^def '+name+r'\(.*?(?=^def |\Z)',source,re.M|re.S).group()


def test_host_transport_preserves_numerical_calls_and_refusals():
    old=(ROOT/'bindings/_mojolearn_linalg_host.mojo').read_text()
    new=(ROOT/'gemm/host/linalg_native.mojo').read_text()
    for name in NAMES:
        before=body(old,name+'_binding'); after=body(new,name+'_native')
        assert re.findall(r'\b\w+_host_run\([^\n]+',before)==re.findall(r'\b\w+_host_run\([^\n]+',after),name
        assert re.findall(r'raise Error\(.*?(?=\n    \w|\n    if|\Z)',before,re.S)==re.findall(r'raise Error\(.*?(?=\n    \w|\n    if|\Z)',after,re.S),name
    _,violations=audit(ROOT,('gemm.host.linalg_dispatch',))
    assert not violations,violations


def test_host_and_device_have_identical_transport_contract():
    device=(ROOT/'gemm/linalg_dispatch.mojo').read_text()
    host=(ROOT/'gemm/host/linalg_dispatch.mojo').read_text()
    assert host==device.replace('from gemm.linalg_native import','from gemm.host.linalg_native import').replace('MojoLinalgNativeTableV1','MojoLinalgHostNativeTableV1')
    shell=(ROOT/'bindings/_mojolearn_linalg_host.mojo').read_text()
    assert 'module.def_function[linalg_native_api_v1_binding]("linalg_native_api_v1")' in shell
    assert 'neural_descriptor_capsule(table[].name, table[].entries)' in shell


def test_shared_family_admission_import_is_executable_source():
    source=(ROOT/'bindings/family_native.mojo').read_text()
    assert '\nfrom providers.python.estimator_binding_pyapi import estimator_arguments\n' in source
    assert '\nfrom the ' not in source
