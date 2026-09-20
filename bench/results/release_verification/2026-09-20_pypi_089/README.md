# mojolearn 0.8.9 release receipt — 2026-09-20

Both wheels were built from frozen source `819a47ae48166e91951f54f54e64ee173658e32a` and carry the `alpha-api` release profile.

| Wheel | SHA256 | Publication status |
| --- | --- | --- |
| `mojolearn-0.8.9-py3-none-macosx_11_0_arm64.whl` | `6703bd788ee926df92d9f43bebbde3703c2ffbc4eb6277c841acb6bc30ef45e4` | Downloaded from public PyPI and hash verified |
| `mojolearn-0.8.9-py3-none-manylinux_2_35_x86_64.whl` | `b2f7856e5959a518ce0ebc23af0240e9092a9f3a5256faf77cc389565915638f` | Downloaded from public PyPI and hash verified; [workflow passed](https://github.com/mojolearn/mojolearn/actions/runs/35535614776) |

Both exact wheels passed all 13 installed light checks with zero failures. macOS additionally passed its installed-wheel smoke matrix across five Python versions and three numeric modes. The Linux wheel contains NVIDIA sm_89/sm_90a and AMD gfx942 binaries plus CPU bindings; its three native build proofs, measured manylinux audit/repair, portable-math audit, RECORD validation and Twine check passed.

Zenodo archived the frozen source for each GitHub release:

- [macOS release DOI: 10.5281/zenodo.22863999](https://doi.org/10.5281/zenodo.22863999).
- [Linux release DOI: 10.5281/zenodo.22864054](https://doi.org/10.5281/zenodo.22864054).
- [Concept DOI: 10.5281/zenodo.22068632](https://doi.org/10.5281/zenodo.22068632), which resolves to the latest archive.

These source archives do not contain later supplemental vendor evidence. The additional AMD run is preserved in commit `cecc7c25f` (603 cells, 2,187 exact numeric CPU comparisons), and NVIDIA in `ca0a984bb` (648 cells, 2,934 exact comparisons). Both used frozen819 binaries, all nine fixtures and full verifier parts. This receipt does not claim that a subsequent all-three-vendor coverage audit is complete.

Local release receipts and archived logs: `mojolearn-evidence/releases/publish089-macos`, `mojolearn-evidence/releases/publish089-linux`, and `mojolearn-evidence/releases/final819-linux-wheel`. Installed Linux results: `mojolearn-evidence/final819-linux-wheel-expanded/results.json`.
