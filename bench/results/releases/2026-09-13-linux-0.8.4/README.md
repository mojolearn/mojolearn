# 0.8.4 Linux release legs (2026-09-13), frozen commit bfde0442

| set | box | image | result |
|---|---|---|---|
| CUDA sm_90a | RunPod H100 80GB HBM3 | runpod/pytorch 2.4.0 cuda12.4.1 ubuntu22.04 | `bench/results/e1g/2026-09-13_154135-nvidia-mamba/remote/release-build`, build exit 0 |
| CUDA sm_89 | RunPod L40S | same image | `bench/results/e1g/2026-09-13_154305-nvidia-mamba/remote/release-build`, build exit 0 |
| HIP gfx942, first | DigitalOcean MI325X, `hip-gfx942/` | DO Ubuntu 24.04 ROCm image (GCC 13.3, binutils 2.42) | build exit 0, 23 extensions, NOT PACKED (below) |
| HIP gfx942, packed | Hot Aisle MI300X, `hip-gfx942-hotaisle-2204/` | rocm/dev-ubuntu-22.04:6.4.1-complete (GCC 11.4, binutils 2.38) | build exit 0, packed |

The first HIP set built cleanly, but its vendor-neutral CPU training binding
`host/_mojolearn_byte_lm_host.so` (sha256 e6bd2be4...) differed from the two CUDA legs'
copies (507b28c3..., byte-identical to each other), and `packaging/linux/pack_wheel.py`
refuses a wheel whose legs disagree on that one file. Section by section the two files
have equal .rodata, .eh_frame and .strtab and differ in .text, .plt, .rela.*, .dynamic,
.init, .comment and the build id, which is what a different system linker and crt
objects produce from one Mojo object; both need at most GLIBC_2.34. 0.8.4 is the first
release that carries this binding, so the rule had never met two build images before.
The same script on the same commit inside the 22.04 container (the CUDA legs' GCC and
binutils) produced sha256 507b28c3... and the wheel packed.

The build sets themselves (`build/`) are gitignored; the wheel and the proofs are under
`~/mojolearn-evidence/release-0.8.4-2026-09-13/`.
