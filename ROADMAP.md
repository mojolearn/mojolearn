# What else the GPU is worth pointing at

**Nothing here is in scope, and nothing here changes `README.md`'s one rule.**
This tree is still a transliteration of CatBoost's symmetric learner and
"COPY, DO NOT IMPROVE" still governs every file in `ported/`. This document
exists so a decision made later is made against a written list rather than a
remembered one.

## The thesis this list follows from

The advantage is **access**, not tier. On Apple Silicon, XGBoost, LightGBM and
CatBoost all fall back to CPU, because their GPU backends are CUDA and OpenCL
and neither exists here. There is a processor in the machine that no
competitor can reach at all.

That is categorical rather than incremental, and it does not depend on the
GPU being fast in absolute terms. It also does not depend on trees. **The
wedge is the device, so anything that runs on the device inherits it.**

The comparison that measures it is therefore always **our GPU against their
CPU on the same machine**, never against their GPU on hardware they own.

## The precedent, on the other side

**RAPIDS cuML** GPU-accelerates most of the scikit-learn surface on NVIDIA. It
does not run on Apple Silicon. The general shape of what this could become is
that library's Mac-shaped counterpart, and gradient boosting is one tile of
it rather than the whole thing.

## The candidates, ranked by GPU benefit times how underserved

### 1. Random Forest. The standout, and nearly free from here.

Trees are **independent**, so unlike boosting it parallelizes across trees as
well as inside them. Boosting is strictly sequential across rounds, which is
the single biggest limit on how much a GPU can do for a GBDT; RF does not have
it. So RF is a **better** GPU fit than the thing this repository is porting.

And it reuses what is already being built: the histogram kernels, the
binning, the split search, the partition machinery. The marginal work is
bagging, feature subsampling per split, and averaging instead of summing.

Highest ratio of value to remaining effort on this list by a wide margin.

### 2. k-means and k-NN.

Distance computation is embarrassingly parallel and arithmetic-dense, which is
the shape a GPU actually wants, unlike histogram building which is bandwidth
bound. Widely used, small surface, hard to get subtly wrong.

k-NN is also the retrieval primitive under vector search, which is the one
place this list touches current AI infrastructure directly.

**Which index, and the answer is not the popular one.** The two approximate
nearest-neighbour structures people reach for behave oppositely on a GPU:

- **HNSW is a BAD GPU fit and should not be attempted.** It is a graph
  traversal: pointer chasing, irregular access, and every hop depends on the
  previous one. That is close to the worst case for a wide machine. This is
  not a prediction; FAISS ships HNSW as CPU-only for exactly this reason.
- **IVF is a GOOD GPU fit.** Distances to centroids are a matrix multiply,
  selecting `nprobe` lists is a top-k, and the search inside those lists is
  brute force. Dense and regular throughout. FAISS-GPU implements IVFFlat and
  IVFPQ and not HNSW, which is the same conclusion reached by people who
  measured it.

**And the shortcut worth taking first: at laptop scale, exact brute-force k-NN
on the GPU can beat approximate k-NN on the CPU.** Brute force is a matrix
multiply plus a top-k, which is the operation a GPU is best at. For corpora up
to a few million vectors there may be no index needed at all.

That is a better claim than owning an ANN index, because it is *exact*.
Competing implementations return approximate neighbours and we would return
the right ones, faster, on hardware they cannot use. Build brute force first,
measure where it stops winning, and only then consider IVF. Never HNSW.

### 3. Gaussian Processes.

An O(n^3) Cholesky is close to the ideal GPU workload: dense, regular, high
arithmetic intensity. GPs are underserved on every platform, not only this
one, so the competitive gap is wider than for anything else here.

The largest single win available on this list, and the most work.

### 4. UMAP and t-SNE. HARD. Do not start here.

**What they are for**, since the names say nothing: reducing high-dimensional
data to two dimensions so it can be *plotted*. You have 768-dimensional
embeddings and you want a scatter plot showing whether they cluster sensibly.
Exploratory visualization rather than a modeling step. Dominant in single-cell
genomics and in inspecting embedding spaces.

Heavy pairwise computation, very widely used, and cuML accelerates both, so
Mac users currently get the CPU implementations and wait.

**But this is the hardest entry on the list and it is not close.** UMAP needs
fuzzy simplicial set construction, a spectral initialization, and an SGD
layout phase, each with its own correctness traps. t-SNE needs either
Barnes-Hut or an FFT-based approximation to avoid being O(n^2), and both are
substantial algorithms in their own right. Neither is a few hundred lines and
neither is verifiable against a hand calculation the way a histogram is.

High user value, high difficulty. Correct to want, wrong to attempt before
the easy entries have shipped and been used.

### 5. Bootstrap, permutation tests, Monte Carlo.

Embarrassingly parallel by construction, and statisticians run them on
laptops constantly. Least glamorous entry and possibly the highest ratio of
user-visible speedup to implementation difficulty.

## Difficulty, stated separately from value

The ranking above is by value. This one is by cost, and they do not agree.

| item | difficulty | note |
|---|---|---|
| Bootstrap, permutation, Monte Carlo | **trivial** | a parallel loop |
| k-means | **easy** | a few hundred lines, and once correct it stays correct |
| Random Forest | **easy from here** | a delta on machinery already built |
| k-NN, exact brute force | **easy** | matrix multiply plus top-k |
| k-NN, IVF index | **moderate** | only if brute force stops winning |
| Gaussian Processes | **hard** | Cholesky is a call; numerical robustness, kernel choice and marginal-likelihood gradients are the work |
| UMAP, t-SNE | **hard** | see above |
| k-NN, HNSW | **do not** | wrong shape for the hardware |

**On maintenance, which is the real objection to a broad scope.** Most of the
treadmill is per REPOSITORY, not per algorithm: language churn, interpreter
releases, packaging, CI. That is paid once whether one algorithm ships or six.

Per-algorithm carry varies enormously, and not in the direction intuition
suggests. Gradient boosting is *expensive* to maintain because it has a large
parameter surface and accuracy gates against three competitors. k-means has
almost none, because there is nothing in it to drift. So the cost of a broad
scope is concentrated in FIRST implementation and validation rather than in
ongoing carry, and several entries above are cheap on both.

## The uncomfortable note about the current project

**Gradient boosting is the WEAKEST entry on this list for GPU acceleration.**
Histogram building is memory-bandwidth bound rather than compute bound, which
is why GBDT GPU speedups run 2x to 10x where dense linear algebra runs far
higher, and why the crossover against a good multicore CPU sits somewhere
around a hundred thousand rows rather than near zero.

Several items above would show a larger GPU win for less work. That is not an
argument to abandon this one, which is the hardest and the most competitive
and therefore the most convincing if it lands. It is an argument against
believing that finishing it is the only thing standing between here and a
product.

## Is any of this "AI"

Not in the 2026 sense of the word, which means generative models. All of it is
**machine learning and statistics**, which is what scikit-learn is and always
was.

Two things follow, and they point in opposite directions.

Against: if the audience is investors or press, "AI" is the word that opens
doors and none of this is it.

For: tabular ML is where a large share of deployed, revenue-bearing model
work actually sits. Fraud, credit, churn, demand forecasting, ranking,
survival. It is unglamorous, it is enormous, and its practitioners are exactly
the people sitting in front of Macs running scikit-learn on a CPU while the
GPU idles.

There is one real bridge: **k-NN is retrieval**, and retrieval is the
infrastructure under RAG. Fast approximate nearest neighbour on a laptop GPU
is an AI-adjacent claim that is honestly earned rather than stretched.
