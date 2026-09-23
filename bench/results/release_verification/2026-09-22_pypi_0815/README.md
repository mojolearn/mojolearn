# mojolearn 0.8.15

Source commit 51f93ced484d8e32e6e5f6f60f1fbe01ec207dd5. AMD gfx942 builds made byte-reproducible
(GEMM and Holt-Winters index arithmetic unsigned, loop counters); native libraries rebuilt.

## Release verification: every lane each GPU can run, against the CPU column

The change touched the shared GEMM kernel, so the selector chose every lane that runs on one
device (209 on Apple, NVIDIA and AMD; 215 on the CPU). Fixtures base, denormal and odd, every
cell fitted once. The NVIDIA and AMD columns ran from the exact final Linux wheel, installed.

| column | where | train cells | infer/model cells | DIVERGENT |
|---|---|---|---|---|
| CPU (reference) | Apple M4 CPU | 645 | | |
| Apple M4 Metal | this Mac, fresh stamped bindings | 627 IDENTICAL | 978 IDENTICAL | 0 |
| AMD MI325X gfx942 | DigitalOcean, installed wheel, Python 3.12 | 627 IDENTICAL | 978 IDENTICAL | 0 |
| NVIDIA H100 sm_90a | RunPod, installed wheel, Python 3.11 | 624 IDENTICAL | 978 IDENTICAL | 0 |

The remaining cells are host-only lanes no GPU runs and the CPU's GPU-saved-file model parts.
On NVIDIA, hf-checkpoint (3 cells) was refused, not compared: the float16 safetensors reader used
memoryview.cast("e"), which Python 3.11 lacks. Fixed on main (4bc3fd091) after this release;
0.8.15 still carries it for Python 3.10 and 3.11.

- cpu/column.json sha256 47f90ade1472b2f4be5db1f34bd00b2f44c26fb03200da420ec4ed1236b6c4cb
- metal/column.json sha256 bf1b95273d9619c67f2501f1a2d0081aee7e374b563770fed15fe4cf6554d828
- amd column.json sha256 0f44003b76025e2d3f53816d1d1e071a09bc545f9574537a8793df11744b9b8c
- nvidia column.json sha256 745de3e97e72e28485348212cb2d0d06d36ad5005140dacb2f954f3429374b50

## Wheels (light route)

| platform | sha256 | smoke | tag |
|---|---|---|---|
| manylinux_2_35_x86_64 | 281677583838358deb50f3a86c2505c695c1ab4ee1f42cfa3476a17ad1d9a8bd | NVIDIA H100, 13 jobs PASSED | alpha-api-0.8.15-linux-20260922 |
| macosx_11_0_arm64 | 846851babfeb045434bd1f4ae317bce94ff8f0049efa07cce59754928e81b380 | Apple M4, 13 jobs PASSED | alpha-api-0.8.15-macos-20260922 |

The Linux wheel packs the sm_89, sm_90a and gfx942 builds of the source commit; the 32 host
bindings are byte-identical across them and auditwheel repair passed.
`pip install mojolearn==0.8.15` on the M4: version 0.8.15, vendor metal, no other package required.
