# 0.7.0 Linux build proofs

The packer needs three per-architecture build proofs from ONE commit
(sm_89, sm_90, gfx942). This directory holds them and the failures that
came before them.

## `hip-gfx942.attempt1-4918af8a-refused/`

The first gfx942 build, at the original frozen commit `4918af8a`. It is
kept because it found a real defect in the release and is what the
changelog and the 0.7.0 notes cite.

`build_rc=1` after 1013 s. The FAST and DETERMINISTIC estimators bindings
do not compile for gfx942: the QR panel kernel added with the PCA full
solver on 2026-09-07 runs a 32-thread block and reduces through MAX's
block primitive, which refuses any block narrower than the warp, and AMD's
wavefront is 64 wide. IDENTICAL already used its own halving tree and
built. The 0.6.0 Linux wheel predates that kernel, which is why nothing
caught it earlier. Fixed at `de719ac9`: the QR kernel uses the halving
tree in every mode. IDENTICAL bits are unchanged; FAST bits on that one
kernel move to the same value.

The 77 MB of compiled `.so` sets this attempt produced are NOT evidence
(the build failed, and the outputs are superseded by the fix). They were
moved out of the source repo under the oversized-blob rule to
`~/mojolearn-evidence-local/2026-09-08-linux-0.7.0-refused-amd-build/build/`.
Every log, exit code, provenance file and command transcript stays here.

## `hip-gfx942.attempt2-de719ac9-uplink-lost/` (the retry at `de719ac9`)

Prep succeeded through `PIXI_ENV_OK`. The build never started: `rc=9`,
"could not start the build". This was NOT a defect on the droplet and NOT
a defect in the build. This machine's network died about ninety seconds
after the droplet came up, taking the ssh that launches the build with it
(`Can't assign requested address`, `Broken pipe` at 05:54:57), and every
subsequent API call returned `HTTP 000` until 08:58.

The two RunPod legs launched in the same minute died the same way and are
recorded under `bench/results/e1g/2026-09-08_055319-nvidia-mamba` (sm_90,
H100 `cy2rvg0tcyoyde`) and `bench/results/e1g/2026-09-08_055703-nvidia-mamba`
(sm_89, L40S `jldtqvpv34yqck`). Both polled boxes they could not reach for
their full deadline, fetched empty directories, and could not confirm
their own terminates. Both pods and the droplet were confirmed gone
afterwards through the API; the dead-man layers held and nothing outlived
its lease.

See DEVIATION 2292 in `docs/RELEASE_0_6_1_RUNPOD_PROFILE7.md` for the
guard added so a leg refuses to rent on a flapping uplink and so the logs
name which end of the wire went quiet.

The directory is named for its outcome so the eventual successful
`hip-gfx942/` proof lands beside it without being confused for it.

## Shipped

`hip-gfx942/` is the gfx942 proof at fe6067ba (the two CUDA proofs are
`bench/results/e1g/2026-09-09_073914-nvidia-mamba` for sm_89 and
`bench/results/e1g/2026-09-09_073744-nvidia-mamba` for sm_90a). Packed,
audited and published as `mojolearn-0.7.0-py3-none-manylinux_2_35_x86_64.whl`,
sha256 57e9a0ec34a7071adc27e87be0eb90dd6f4c6bd101669675db35e1b6d6cde232, GitHub
release `alpha-api-0.7.0-20260909`, on 2026-09-09. `hip-gfx942.340de7a1-superseded-by-2300/`
and `qualification/` are the earlier round at 340de7a1, whose wheel afb5dd8a was
never published: its install-and-test found the Mamba-3 deterministic defect
fixed at fe6067ba. The fe6067ba wheel was published without installed
qualification.
