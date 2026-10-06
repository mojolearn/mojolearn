# Implementation and qualification ledger

The source branch is `codex/performance-experiment-ideas-20261005`, forked from
the performance experiment ideas document at `32e455e029`. Each of the 60 cards
has its own experiment commit. Later corrections retain the same card ID in
separate commits; integration merges preserve that history.

These are opt-in experiments. No candidate is promoted by this ledger. All
40 shared/AMD/NVIDIA cards use IDENTICAL; all 20 Apple cards use FAST. NVIDIA
and AMD decide IDENTICAL performance acceptance, while Apple and host also
owe the same-version identity witnesses. Apple FAST measures task quality and
completed public calls on the M3 Ultra.

## Requested completion plan

Finish and merge all supported candidates into the dedicated integration
branch first. Freeze ONE commit, then run ONE exhaustive compile campaign for
the distinct source/define/target configurations. Retry only compiler failures
that require source fixes. Preliminary campaigns were stopped at the owner's
request and do not establish final all-card compilation. Compilation runs on the rented
NVIDIA builder and the retained M3 Ultra for Apple Metal. The M2 instance and
its Dedicated Host are being retired after preserving their work.

For IDENTICAL, collect exact-output GPU witnesses where hardware is available
and record missing vendor/host checks explicitly. Every new experiment stays
default off and pending numerical/performance qualification. Preserve existing
promoted defaults. After the compile matrix passes, merge into main, retaining
the per-card commit history. Do not require a timing campaign before this merge.

The machine-readable [progress ledger](progress.json) records this plan and
links the compile coverage. A manifest or successful parser check alone does
not establish that every candidate configuration compiled.

## What the implementation establishes

Each manifest names real source or an executable campaign for existing source,
its independent candidate/control defines, vendor-specific build recipe,
correctness gates, and completion timing contract. Existing scheduling ideas
are implemented as campaigns rather than copied production kernels. New paths
include bounded scratch and state reuse, GQA tile sharing, live optimizer
status, speculative canonical solver trials, compact graph/IVF tasks, retained
forest histograms, ragged float sorting, component batching, factor and TSQR trailing-strip reuse, persistent bounded KDE scratch,
and compensated Apple products and likelihood gradients.

`source_ready` means the source and recipe exist. `build_passed` records a
particular successful compilation; neither status is device qualification.
Individual checks and external build logs identify which instantiations were
actually compiled. No device timing, four-column qualification, opponent
comparison, or performance win is inferred from a successful compiler exit.

## Card inventory

The first source commit is shown for review; follow-up commits contain expanded
caller checks and fixes. Read a card's scope alongside its manifest. The [Apple FAST coverage](apple_fast/coverage.md)
records the real callers, independent oracles, dependencies and remaining sub-arms.

| Card | Mode | First source commit | Recipe status | Scope or caller |
| --- | --- | --- | --- | --- |
| [I01](I01/manifest.json) | IDENTICAL | `c9047b411` | build_passed | manifest |
| [I02](I02/manifest.json) | IDENTICAL | `8af89de6d` | build_passed | manifest |
| [I03](I03/manifest.json) | IDENTICAL | `3e420ddd1` | build_passed | manifest |
| [I04](I04/manifest.json) | IDENTICAL | `98f8ffc82` | build_passed | manifest |
| [I05](I05/manifest.json) | IDENTICAL | `8c26726d6` | build_passed | manifest |
| [I06](I06/manifest.json) | IDENTICAL | `588c7c15c` | source_ready | [scope](I06/coverage.md) |
| [I07](I07/manifest.json) | IDENTICAL | `b84a712be` | source_ready | [scope](I07/coverage.md) |
| [I08](I08/manifest.json) | IDENTICAL | `7554a8921` | source_ready | [scope](I08/coverage.md) |
| [I09](I09/manifest.json) | IDENTICAL | `e878272c6` | source_ready | [scope](I09/coverage.md) |
| [I10](I10/manifest.json) | IDENTICAL | `b206d07c5` | source_ready | [scope](I10/coverage.md) |
| [I11](I11/manifest.json) | IDENTICAL | `988448acb` | source_ready | [scope](I11/coverage.md) |
| [I12](I12/manifest.json) | IDENTICAL | `a1b3883a4` | source_ready | [scope](I12/coverage.md) |
| [I13](I13/manifest.json) | IDENTICAL | `758b4b29c` | source_ready | [scope](I13/coverage.md) |
| [I14](I14/manifest.json) | IDENTICAL | `d43719eb1` | source_ready | [scope](I14/coverage.md) |
| [I15](I15/manifest.json) | IDENTICAL | `861af59c7` | source_ready | [scope](I15/coverage.md) |
| [I16](I16/manifest.json) | IDENTICAL | `58c32e802` | source_ready | [scope](I16/coverage.md) |
| [I17](I17/manifest.json) | IDENTICAL | `efa812db6` | source_ready | [scope](I17/coverage.md) |
| [I18](I18/manifest.json) | IDENTICAL | `673eeb28c` | source_ready | [scope](I18/coverage.md) |
| [I19](I19/manifest.json) | IDENTICAL | `c3fc4818d` | source_ready | [scope](I19/coverage.md) |
| [I20](I20/manifest.json) | IDENTICAL | `164d32cb3` | build_passed | [scope](I20/coverage.md) |
| [I21](I21/manifest.json) | IDENTICAL | `bce4f9a3a` | source_ready | [scope](I21/coverage.md) |
| [I22](I22/manifest.json) | IDENTICAL | `0a20cc4c2` | source_ready | [scope](I22/coverage.md) |
| [I23](I23/manifest.json) | IDENTICAL | `ec616906a` | source_ready | [scope](I23/coverage.md) |
| [I24](I24/manifest.json) | IDENTICAL | `d08a5beec` | source_ready | [scope](I24/coverage.md) |
| [A01](A01/manifest.json) | IDENTICAL | `856db1e30` | source_ready | manifest |
| [A02](A02/manifest.json) | IDENTICAL | `d986c950d` | build_passed | manifest |
| [A03](A03/manifest.json) | IDENTICAL | `c0639fdce` | source_ready | manifest |
| [A04](A04/manifest.json) | IDENTICAL | `cd0f72238` | build_passed | manifest |
| [A05](A05/manifest.json) | IDENTICAL | `69d10e5ec` | source_ready | manifest |
| [A06](A06/manifest.json) | IDENTICAL | `9178f93a8` | build_passed | manifest |
| [A07](A07/manifest.json) | IDENTICAL | `9e0284bdc` | build_passed | manifest |
| [A08](A08/manifest.json) | IDENTICAL | `8114db1fb` | build_passed | manifest |
| [N01](N01/manifest.json) | IDENTICAL | `278571c06` | source_ready | manifest |
| [N02](N02/manifest.json) | IDENTICAL | `b697705b2` | build_passed | manifest |
| [N03](N03/manifest.json) | IDENTICAL | `598815aec` | build_passed | manifest |
| [N04](N04/manifest.json) | IDENTICAL | `956a06c91` | source_ready | manifest |
| [N05](N05/manifest.json) | IDENTICAL | `354893b1e` | build_passed | manifest |
| [N06](N06/manifest.json) | IDENTICAL | `b1f07842d` | build_passed | manifest |
| [N07](N07/manifest.json) | IDENTICAL | `23c1574b6` | build_passed | manifest |
| [N08](N08/manifest.json) | IDENTICAL | `7138c44f8` | build_passed | manifest |
| [F01](F01/manifest.json) | FAST | `042343c16` | source_ready | [caller](F01/caller.py) |
| [F02](F02/manifest.json) | FAST | `b9fc859d5` | source_ready | [caller](F02/caller.py) |
| [F03](F03/manifest.json) | FAST | `76bf7224d` | source_ready | [caller](F03/caller.py) |
| [F04](F04/manifest.json) | FAST | `b8e686812` | source_ready | [caller](F04/caller.py) |
| [F05](F05/manifest.json) | FAST | `c0eabd979` | source_ready | [caller](F05/caller.py) |
| [F06](F06/manifest.json) | FAST | `3c6bb2b64` | source_ready | [caller](F06/caller.py) |
| [F07](F07/manifest.json) | FAST | `6a5b65e10` | source_ready | [caller](F07/caller.py) |
| [F08](F08/manifest.json) | FAST | `a5c1c46af` | source_ready | [caller](F08/caller.py) |
| [F09](F09/manifest.json) | FAST | `cabf8eb6d` | source_ready | [caller](F09/caller.py) |
| [F10](F10/manifest.json) | FAST | `3e7dd19a5` | source_ready | [caller](F10/caller.py) |
| [F11](F11/manifest.json) | FAST | `8a0e1fdaf` | source_ready | [caller](F11/caller.py) |
| [F12](F12/manifest.json) | FAST | `09897f665` | source_ready | [caller](F12/caller.py) |
| [F13](F13/manifest.json) | FAST | `4b2a66c16` | source_ready | [caller](F13/caller.py) |
| [F14](F14/manifest.json) | FAST | `bff01c5ef` | source_ready | [caller](F14/caller.py) |
| [F15](F15/manifest.json) | FAST | `b7e3abbec` | source_ready | [caller](F15/caller.py) |
| [F16](F16/manifest.json) | FAST | `8d08a7812` | source_ready | [caller](F16/caller.py) |
| [F17](F17/manifest.json) | FAST | `3895b3584` | source_ready | [caller](F17/caller.py) |
| [F18](F18/manifest.json) | FAST | `c8fe56166` | source_ready | [caller](F18/caller.py) |
| [F19](F19/manifest.json) | FAST | `a83298be9` | source_ready | [caller](F19/caller.py) |
| [F20](F20/manifest.json) | FAST | `4ff4e0cb6` | source_ready | [caller](F20/caller.py) |

## Scope still requiring separate work

The individual IDENTICAL coverage files explicitly identify original-card
sub-arms that are not implemented by the current candidate. These include
new versioned scan/reduction profiles (I09, I22, I23), new Gram/shared-fold
contracts (I12), broader target support for geometric coarse-candidate certification
(I15), a device tree-builder API for model fragments (I17), multi-tree/weighted
histogram algebra (I18), and broader metric/preprocessing pass fusion (I24).
The supported I12 SGD, I19 dictionary/ordinal/bootstrap/selection/radix layout,
I21 centered-load reuse, F06 compaction/covariance reuse and F12 document-ID
storage arms are now supplied. Conditional original-card proposals must not be represented
as completed merely because their card has an executable experiment.

The next qualification work is to run matched frozen candidate/control pairs
on the existing device queues. I18's `paired_identity.py` checks complete forest
witnesses on one device and deliberately emits a device-witness receipt rather
than a global quality pass. Apple paired builds attest every prerequisite and
both libraries at one source SHA; independent native oracles run before public
caller measurements. New compilation uses the NVIDIA RunPod builder and the
retained M3 Ultra; older M2 artifacts retain their original source attestations.

No production defaults or performance boards are changed by this work.
