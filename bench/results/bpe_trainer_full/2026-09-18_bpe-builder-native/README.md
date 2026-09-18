# 2026-09-18 bpe-builder-native: the 50,256-rank vocabulary job, full size

Recipe of `mojolearn-bpe-50257-v1` (pinned in bench/results/dataset_store/manifest.tsv): the
first 10,000,000 bytes of enwik8 (sha256 5985c81c...) and of pile_github (0bee7f53...), each
ONE document, 50,256 ranks, min_frequency 2. Corpora staged from R2 (strict), heads checked
against the M4 recipe before anything ran (`*/heads.sha256`). One core per arm. RunPod CPU pods,
image runpod/base:1.3.1-ubuntu2204. The trained files themselves are NOT here (derived from
third-party text); their hashes are.

| arm | box | wall | peak RSS | ranks.tsv sha256 | tokenizer.json sha256 |
|---|---|---|---|---|---|
| OLD: main 0e4715f7c `train_main`, dense V x V table (M4 run, tokenized-corpus lane) | Apple M4, nice 19 | 4,274.27 s | 20.2 GB table (19 GB in `top`, mostly compressed) | 3d547b17821cf465... | 7ae8b893219619cf... |
| OLD, same binary source, bounded at 600 s (`leg1/old_bounded/`) | EPYC 7713P | stopped at 601.9 s (rc 124, by design) | **33,997,832 KiB = 34.8 GB** (ru_maxrss and VmHWM agree; settles at 20.0 GB RSS) | - | - |
| NEW `train_main` (`legB/`) | EPYC 9575F | **1,116.6 s** | **91,736 KiB = 94 MB** | **3d547b17821cf46502f275a441dd6ded9682a4ddcacde993a1ff836f39c4122d** | **7ae8b893219619cf73e64b262367cc6e6d2f9f1aa9c1a32b8fb7e3cb84d972e9** |
| NEW through the public door `BpeVocabularyTrainer(backend="mojo")` (`leg1/door.*`) | EPYC 7713P | 2,532.9 s | 156,880 KiB = 161 MB (whole Python process) | 3d547b17...4122d | 7ae8b893...72e9 |
| `lm_corpus.prepare()` defaults on all of enwik8 (`leg1/prepare.*`) | EPYC 7713P | 1,291.2 s end to end | 733,420 KiB = 751 MB | vocabulary identity 32ff1125... | ids sha256 c25481a7... |

All three trained runs report 50,256 tokens, 50,000 merges, 47,825 ties, 220,165 pre-token
groups, equal to the M4 dense-table run.

WHY THE OLD PEAK IS 34.8 GB, NOT 20.2. The table was built by appending V * V zeros to a
`List[Int]`; the last capacity doubling holds the old 17.2 GB buffer and the copy at once
(VmHWM 32.4 GB at 40 s, 34.0 GB at 50 s, then RSS falls to 20.0 GB). The M4 survived it by
memory compression.

The two EPYC generations differ (Zen 3 7713P vs Zen 5 9575F), so the door and CLI walls are
not a CLI-vs-door comparison. Leg 1's CLI arms failed to BUILD (bare `mojo` finds no `std`
outside `pixi run`, `leg1/build_old.log`); the new CLI ran on a second pod (leg B) after
RunPod refused six creates for capacity, and the old arm's peak was measured on leg 1's pod
by hand. No same-box full-length wall for the old arm exists.
