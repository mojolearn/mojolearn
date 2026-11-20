# SPDX-License-Identifier: Apache-2.0
"""The build and wheel read-backs admit a byte-LM binary built with an arm define (packaging/byte_lm_profile.py).

Defect, 2026-11-19 (GitHub run 44525867606, owed B arm `gemm-leaf512`): every shard compiled, then
packaging/linux/build_sets.sh refused the set with "Wrong byte LM profile", because the read-back compared
`byte_lm_profile()` to the bare base string while a MOJOLEARN_GEMM_FAST_LEAF512 binary reports base +
`-gemm-leaf512` (training/byte_lm_config.mojo byte_lm_arithmetic_suffix), which the runtime admits.
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_byte_lm_profile', ROOT / 'packaging' / 'byte_lm_profile.py')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

SCRIPTS = ('packaging/linux/build_sets.sh', 'packaging/macos/build_release_wheel.sh', 'packaging/macos/verify_wheel.sh')


def test_base_matches_the_mojo_default_profile():
    src = (ROOT / 'training' / 'byte_lm_config.mojo').read_text()
    assert 'comptime BYTE_DEFAULT_PROFILE = "%s"' % P.BASE in src


def test_arm_suffixes_are_admitted():
    for tail in ('', '-gemm-leaf512', '-gemm-leaf256', '.nn-gemm-v2-l256-c4', '.nn-clip-v2-leaf128',
                 '.nn-gemm-v2-l512-c2.nn-clip-v2-leaf128-gemm-leaf512'):
        assert P.admitted(P.BASE + tail), tail
        assert P.check(P.BASE + tail) == P.BASE + tail


def test_other_profiles_are_refused():
    bad = ('', 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-hd8-ff64-v256-blocks2.fp32.v2', P.BASE + '0',
           P.BASE[:-1] + '2', P.BASE + ' -gemm-leaf512', P.BASE + '-', P.BASE + '-Gemm', None)
    for profile in bad:
        assert not P.admitted(profile), profile
        try:
            P.check(profile)
        except AssertionError as exc:
            assert 'Wrong byte LM profile' in str(exc)
        else:
            raise AssertionError('admitted %r' % (profile,))


def test_no_read_back_demands_the_bare_base():
    for rel in SCRIPTS:
        text = (ROOT / rel).read_text()
        assert "byte_lm_profile() == '" not in text, rel
        assert 'byte_lm_profile.py' in text, rel


def test_leaf_tag_still_produced_by_the_define():
    # The tag this admission exists for: a leaf other than 128 appends -gemm-leaf<N>.
    src = (ROOT / 'training' / 'byte_lm_config.mojo').read_text()
    assert 'String("-gemm-leaf") + String(CONTRACT_K_LEAF_MIN)' in src
