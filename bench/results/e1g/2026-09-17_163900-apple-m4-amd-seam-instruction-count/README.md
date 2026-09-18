# The AMD identity seam costs 8 issued instructions per product step, and one scalar instruction per launch would replace 7 of them

2026-09-17, lane `lane/gemm-next`, on the M4, one core, `nice 19`, NO GPU and
NO RENTAL: this is a COMPILE result read out of the emitted gfx942 GCN, not a
timing. `docs/lanes/BRIEF_gemm_kernel_2026-09-11.md` section 14.5 point 2 said
AMD "pays a software flush after every step ... about six issued instructions"
and called it unmeasured. The count is seven, the total seam is eight, and the
alternative is now known to be reachable from this Mojo.

## How

`amd_seam_count.mojo` (the source, kept here rather than in `gemm/checks/`
because it is a compile probe and not a check) declares three kernels over the
same four-step accumulation and LAUNCHES all three, so none is dead-stripped:

    seam_shipped   ftz(identical_mul_add(a, b, acc))   what AMD ships today
    seam_bare      identical_mul_add(a, b, acc)        the native FMA alone
    seam_mode      llvm.amdgcn.s.setreg once, then the native FMA alone

One compile, one file, so the three are comparable:

    mojo build --emit asm --target-accelerator gfx942 -I . \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 gemm/checks/_scratch_amd_seam_count.mojo

`--target-accelerator gfx942` is load-bearing. `MOJOLEARN_GPU_ARCHS=gfx942`
alone does NOT retarget `mojo build`: the first attempt here emitted
`target triple = "air64-apple-macosx26.0.0"` and would have "proved" the
intrinsic on Apple's backend. The sidecars kept here all begin
`.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"`; read that line before
reading any count below.

## The shipped seam, from `*_seam_sh*.amdgcn`, loop body `.LBB0_1`

    v_fmac_f32_e32   v4, v5, v6           the FMA
    v_and_b32_e32    v6, 0x7fffff, v4     ftz: mantissa
    v_and_b32_e32    v7, 0x7f800000, v4   ftz: exponent
    v_cmp_ne_u32_e32 vcc, 0, v6           ftz: mantissa != 0
    v_cmp_eq_u32_e64 s[0:1], 0, v7        ftz: exponent == 0
    s_and_b64        vcc, s[0:1], vcc     ftz: both
    v_and_b32_e32    v5, 0x80000000, v4   ftz: sign
    v_cndmask_b32_e32 v4, v4, v5, vcc     ftz: select

**Eight issued instructions per product step** (the two `global_load`s, the
address increment, the loop counter and the branch are excluded from all three
counts). NVIDIA's seam is two. That is the AMD column's structural
disadvantage, and it is in the loop, once per product step.

## The alternative, from `*_seam_mo*.amdgcn`

    s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0    ONCE, at kernel entry

and the loop body is then `v_fmac_f32_e32 v4, v5, v6` and nothing else, byte
for byte the same body as `*_seam_ba*.amdgcn`. **One instruction per product
step plus one scalar instruction per kernel launch, against eight per step.**

The Mojo spelling is `llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(0))`.
`0x0901` is `hwreg(HW_REG_MODE = 1, offset = 4, width = 2)`, the f32 FP_DENORM
field: `1 | (4 << 6) | ((2 - 1) << 11) = 2305`. The disassembler prints the
field back, which is how the encoding was checked rather than asserted. Note
that `llvm.amdgcn.s.setreg.imm32.b32` (the ISA mnemonic) is NOT an LLVM
intrinsic name and is rejected by name; `llvm.amdgcn.s.setreg` is the one.

All three kernels still carry `.amdhsa_float_denorm_mode_32 3` in their
descriptors, i.e. the kernel is ENTERED with f32 denormals allowed and the
`s_setreg` overrides it at run time. So this needs no compiler flag, no
build-system change and no per-column build.

## WHAT THIS DOES NOT SAY

1. **It is an instruction count, not a time.** The NVIDIA decomposition
   (brief 16.3) found that column's accumulate loop was NOT where the time
   went. Whether the MI300X's loop is issue-bound is UNMEASURED, so no
   speedup is claimed here and none should be quoted from it.
2. **It does not say the bits would be the same.** `s_setreg` is worth
   nothing unless AMD's hardware output flush is ROUND-THEN-FLUSH. If AMD
   flushes before rounding, as Apple's FMA does, this arm computes the wrong
   value at the 315 boundary triples and is a defect, not an optimization.
   That is the one question, it is not answered here, and the harness that
   answers it already exists: one more lane in
   `gemm/checks/gemm_seam_probe.mojo` (set the mode, hash the bare FMA,
   compare against the `rtf` reference hash `62a6b5621e27c707`), which is
   about ten seconds of MI300X.

So the sequence is: the mode probe FIRST, and only a `rtf` answer makes this a
speed lever at all.
